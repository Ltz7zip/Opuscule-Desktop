"""DDPM 调度器：前向加噪 + 反向去噪采样。

阶段 1 用线性 β 调度（与原论文一致）。阶段 2 可换 cosine 调度。
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


class DDPM:
    """DDPM 调度器。

    Parameters
    ----------
    n_steps : int
        扩散步数 T。
    beta_start, beta_end : float
        线性 β 调度上下界。
    device : torch.device | str
        预计算 ᾱ 等张量所在设备。
    """

    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 2e-2, device: torch.device | str = "cpu"):
        """初始化 DDPM 调度器：预计算线性 β 调度及 ᾱ。"""
        self.n = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)        # (n,)
        self.betas = betas
        self.alphas = alphas

    # ---------- 前向 ----------
    def q_sample(self, x0, t, noise=None):
        """前向加噪：x_t = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    # ---------- 训练损失 ----------
    def p_loss(self, model, x0, t):
        """训练目标：预测噪声 ε。"""
        x_t, noise = self.q_sample(x0, t)
        pred = model(x_t, t)
        return F.mse_loss(pred, noise)

    # ---------- 反向采样 ----------
    @torch.no_grad()
    def sample(self, model, shape, device, progress: bool = False):
        """反向采样（DDPM 完整 n 步）。

        Parameters
        ----------
        model : nn.Module
            去噪 U-Net。
        shape : tuple
            输出形状 (B, C, H, W)。
        device : torch.device
        progress : bool
            是否显示进度条。
        """
        try:
            from tqdm import trange
        except ImportError:
            trange = range  # type: ignore

        x = torch.randn(shape, device=device)
        iters = range(self.n - 1, -1, -1)
        if progress:
            try:
                from tqdm import tqdm as _tqdm
                iters = _tqdm(iters, desc="sampling")
            except ImportError:
                pass

        for i in iters:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t)
            a = self.alphas[i]
            ab = self.alpha_bar[i]
            x = (1 / a.sqrt()) * (x - (1 - a) / (1 - ab).sqrt() * eps)
            if i > 0:
                x += self.betas[i].sqrt() * torch.randn_like(x)
        return x

    # ---------- 条件反向采样 ----------
    @torch.no_grad()
    def sample_conditional(self, model, shape, cond, device,
                           progress: bool = False):
        """条件反向采样（DDPM 完整 n 步，带条件输入）。

        Parameters
        ----------
        model : nn.Module
            条件去噪 U-Net，forward(x_t, t, cond) → eps。
        shape : tuple
            输出形状 (B, 1, H, W)。
        cond : torch.Tensor
            条件张量 (B, cond_ch, H, W)，与 x 同空间尺寸。
        device : torch.device
        progress : bool
            是否显示进度条。
        """
        try:
            from tqdm import trange
        except ImportError:
            trange = range  # type: ignore

        x = torch.randn(shape, device=device)
        iters = range(self.n - 1, -1, -1)
        if progress:
            try:
                from tqdm import tqdm as _tqdm
                iters = _tqdm(iters, desc="cond_sampling")
            except ImportError:
                pass

        for i in iters:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t, cond)
            a = self.alphas[i]
            ab = self.alpha_bar[i]
            x = (1 / a.sqrt()) * (x - (1 - a) / (1 - ab).sqrt() * eps)
            if i > 0:
                x += self.betas[i].sqrt() * torch.randn_like(x)
        return x

    # ---------- DDIM 快速采样 ----------
    @torch.no_grad()
    def ddim_sample_conditional(self, model, shape, cond, device,
                                n_infer: int = 50, eta: float = 0.0,
                                progress: bool = False):
        """DDIM 条件采样（n_infer 步，比 DDPM 快 20x）。

        Parameters
        ----------
        model : nn.Module
            条件去噪 U-Net。
        shape : tuple
            输出形状 (B, 1, H, W)。
        cond : torch.Tensor
            条件张量 (B, cond_ch, H, W)。
        device : torch.device
        n_infer : int
            采样步数（默认 50）。
        eta : float
            随机性参数：0=确定性 DDIM，1=等价 DDPM。
        progress : bool
        """
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        # 选 n_infer 个均匀间隔的时间步
        step_indices = torch.linspace(0, self.n - 1, n_infer, dtype=torch.long)
        iters = step_indices.flip(0).tolist()
        if progress and _tqdm:
            iters = _tqdm(iters, desc="ddim")
        x = torch.randn(shape, device=device)
        alphas_cumprod = self.alpha_bar.to(device)
        prev_alphas = {i: alphas_cumprod[step_indices[k-1]] if k > 0
                       else torch.tensor(1.0, device=device)
                       for k, i in enumerate(step_indices.tolist())}

        for k, i in enumerate(iters):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t, cond)
            ab = alphas_cumprod[i]
            ab_prev = prev_alphas[i] if i in prev_alphas else alphas_cumprod[max(0, i-1)]
            # x0 预测
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)
            # DDIM 一步
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab) / ab_prev).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).sqrt() * eps
            x = ab_prev.sqrt() * x0_pred + dir_xt
            if sigma > 0:
                x += sigma * torch.randn_like(x)
        return x

    # ---------- SDEdit 条件精修 ----------
    @torch.no_grad()
    def sdedit_conditional(self, model, x0, cond, device,
                           t_start: int = 100, n_infer: int = 20,
                           eta: float = 0.0, progress: bool = False):
        """SDEdit 条件精修：对 D8 mask 加部分噪声 → 去噪 → 精修。

        保留 D8 拓扑结构，修正局部不自然模式（锯齿、辫状）。

        Parameters
        ----------
        model : nn.Module
            条件去噪 U-Net。
        x0 : (B, 1, H, W)  原始 mask（D8 河流 mask，值 0/1）
        cond : (B, cond_ch, H, W)  条件输入
        device : torch.device
        t_start : int
            噪声起始步（越小越接近原图，推荐 50-200）
        n_infer : int
            去噪步数
        eta : float
            随机性（0=确定性，1=等价 DDPM）
        progress : bool
        """
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        # 1. 对 x0 加噪声到 t_start
        ab_start = self.alpha_bar[t_start].to(device)
        noise = torch.randn_like(x0)
        x = ab_start.sqrt() * x0 + (1 - ab_start).sqrt() * noise

        # 2. 从 t_start 反向采样到 0
        step_indices = torch.linspace(0, t_start, n_infer, dtype=torch.long)
        iters = step_indices.flip(0).tolist()
        if progress and _tqdm:
            iters = _tqdm(iters, desc="sdedit")

        alphas_cumprod = self.alpha_bar.to(device)
        prev_alphas = {i: alphas_cumprod[step_indices[k-1]] if k > 0
                       else torch.tensor(1.0, device=device)
                       for k, i in enumerate(step_indices.tolist())}

        for k, i in enumerate(iters):
            t = torch.full((x0.size(0),), i, device=device, dtype=torch.long)
            eps = model(x, t, cond)
            ab = alphas_cumprod[i]
            ab_prev = prev_alphas[i] if i in prev_alphas else alphas_cumprod[max(0, i-1)]
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab) / ab_prev).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).sqrt() * eps
            x = ab_prev.sqrt() * x0_pred + dir_xt
            if sigma > 0:
                x += sigma * torch.randn_like(x)
        return x
