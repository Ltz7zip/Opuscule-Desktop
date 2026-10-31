"""阶段 5B：多模态条件编码器（优化版）。

在原有 12 通道板块底图基础上，支持额外条件模态：
- **草图条件**（sketch）：用户手绘粗略轮廓，控制陆地/海洋分布
- **高度图条件**（elevation hint）：指定区域目标高程范围
- **生物群落图条件**（biome map）：指定区域生物群落分布
- **海平面条件**（sea level）：控制全球海平面高度
- **气候带条件**（climate zone）：指定区域气候类型
- **板块边界条件**（plate boundary）：控制板块边界位置

优化项（相比初版）
------------------
1. **交叉注意力融合**：模态特征 → Q/K/V 交叉注意力注入板块底图
2. **模态权重控制**：可调节各模态控制强度（0~1）
3. **空间掩码**：仅在指定区域应用模态条件（局部控制）
4. **残差连接**：保留原始板块信息，防止条件覆盖
5. **预设模态**：海平面/气候带/板块边界 3 种预设条件
6. **条件校验**：输入形状/值域自动校验与预处理

用法
----
    >>> from terrain.dl.multicond import MultiModalConditioner, ModalityConfig
    >>> cfg = ModalityConfig(sketch_weight=0.8, biome_weight=0.5)
    >>> cond = MultiModalConditioner(modality_config=cfg)
    >>> out, meta = cond(plate_12ch, sketch=sketch_map, biome=biome_map)
    >>> # 带空间掩码的局部控制
    >>> mask = np.zeros((H, W)); mask[100:200, 100:200] = 1
    >>> out, meta = cond(plate_12ch, sketch=sketch_map, sketch_mask=mask)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 1. 模态配置 ====================

class ModalityConfig:
    """模态控制配置：调节各模态权重与开关。

    用法
    ----
        >>> cfg = ModalityConfig(sketch_weight=0.8, biome_weight=0.5)
        >>> cond = MultiModalConditioner(modality_config=cfg)
    """

    def __init__(self,
                 sketch_weight: float = 1.0,
                 elev_weight: float = 1.0,
                 biome_weight: float = 1.0,
                 sea_level_weight: float = 1.0,
                 climate_weight: float = 1.0,
                 plate_boundary_weight: float = 1.0,
                 use_cross_attention: bool = True,
                 residual_connection: bool = True):
        """初始化模态配置。

        Parameters
        ----------
        sketch_weight : float  草图控制强度（0=忽略, 1=完全控制）
        elev_weight : float  高度提示控制强度
        biome_weight : float  生物群落控制强度
        sea_level_weight : float  海平面控制强度
        climate_weight : float  气候带控制强度
        plate_boundary_weight : float  板块边界控制强度
        use_cross_attention : bool  使用交叉注意力融合
        residual_connection : bool  残差连接保留原始板块信息
        """
        self.weights = {
            "sketch": sketch_weight,
            "elev_hint": elev_weight,
            "biome": biome_weight,
            "sea_level": sea_level_weight,
            "climate": climate_weight,
            "plate_boundary": plate_boundary_weight,
        }
        self.use_cross_attention = use_cross_attention
        self.residual_connection = residual_connection

    def get_weight(self, modality: str) -> float:
        """获取模态权重。"""
        return self.weights.get(modality, 1.0)

    def to_dict(self) -> dict:
        """转字典。"""
        return {
            "weights": dict(self.weights),
            "use_cross_attention": self.use_cross_attention,
            "residual_connection": self.residual_connection,
        }


# ==================== 2. 交叉注意力融合 ====================

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合：模态特征注入板块底图。

    Q = 板块底图（主），K/V = 模态特征（条件）。
    通过注意力机制选择性注入模态信息，
    比简单拼接更精细，可控制注入区域。

    架构：Multi-Head Cross Attention + FFN + LayerNorm + 残差。
    """

    def __init__(self, base_ch: int = 12, cond_ch: int = 24,
                 n_heads: int = 4, dropout: float = 0.0):
        """初始化交叉注意力融合。

        Parameters
        ----------
        base_ch : int  板块底图通道（Q）
        cond_ch : int  条件特征通道（K/V）
        n_heads : int  注意力头数
        dropout : float  dropout 概率
        """
        super().__init__()
        assert n_heads > 0, f"n_heads 必须为正数，得到 {n_heads}"
        assert base_ch % n_heads == 0, \
            f"base_ch={base_ch} 必须能被 n_heads={n_heads} 整除"
        self.n_heads = n_heads
        self.head_dim = base_ch // n_heads

        # Q/K/V 投影
        self.q_proj = nn.Conv2d(base_ch, base_ch, 1)
        self.k_proj = nn.Conv2d(cond_ch, base_ch, 1)
        self.v_proj = nn.Conv2d(cond_ch, base_ch, 1)

        # 输出投影
        self.out_proj = nn.Conv2d(base_ch, base_ch, 1)
        self.norm1 = nn.LayerNorm(base_ch)
        self.norm2 = nn.LayerNorm(base_ch)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(base_ch, base_ch * 2),
            nn.SiLU(),
            nn.Linear(base_ch * 2, base_ch),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, base: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """前向：base + CrossAttention(base, cond)。

        Parameters
        ----------
        base : Tensor (B, base_ch, H, W)  板块底图（Q）
        cond : Tensor (B, cond_ch, H, W)  模态条件（K/V）

        Returns
        -------
        Tensor (B, base_ch, H, W)  注意力增强的板块底图
        """
        B, C, H, W = base.shape

        # Q/K/V 投影
        q = self.q_proj(base)      # (B, C, H, W)
        k = self.k_proj(cond)      # (B, C, H, W)
        v = self.v_proj(cond)      # (B, C, H, W)

        # 展平空间维 → (B, H*W, C)
        q = q.flatten(2).transpose(1, 2)   # (B, HW, C)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        # 多头注意力
        q = q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / \
            (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        if self.dropout:
            attn = self.dropout(attn)

        out = torch.matmul(attn, v)   # (B, n_heads, HW, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)

        # 输出投影 → (B, C, H, W)
        out = out.transpose(1, 2).view(B, C, H, W)
        out = self.out_proj(out)

        # 残差 + LayerNorm
        base_flat = base.flatten(2).transpose(1, 2)   # (B, HW, C)
        out_flat = out.flatten(2).transpose(1, 2)
        normed = self.norm1(base_flat + out_flat)

        # FFN
        ffn_out = self.ffn(normed)
        normed = self.norm2(normed + ffn_out)

        # 回 (B, C, H, W)
        return normed.transpose(1, 2).view(B, C, H, W).contiguous()


# ==================== 3. 单模态编码器（增强） ====================

class SketchEncoder(nn.Module):
    """草图编码器（增强版）：多尺度轮廓特征提取。

    输入：二值/灰度草图 (B, 1, H, W)，值域 [0, 1]
    输出：编码特征 (B, out_ch, H, W)

    架构：3 层卷积 + 残差连接 + 边缘增强（Sobel）。
    """

    def __init__(self, out_ch: int = 4, base: int = 32):
        """初始化草图编码器。"""
        super().__init__()
        # conv1 接受 2 通道（原始 + 边缘）
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, base, 3, padding=1), nn.SiLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.SiLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1), nn.SiLU())
        self.out = nn.Conv2d(base, out_ch, 1)

        # 残差连接
        self.skip = nn.Conv2d(1, out_ch, 1)

        # Sobel 边缘检测（固定权重）
        self.register_buffer("sobel_x",
                             torch.tensor([[-1, 0, 1], [-2, 0, 2],
                                           [-1, 0, 1]],
                                          dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y",
                             torch.tensor([[-1, -2, -1], [0, 0, 0],
                                           [1, 2, 1]],
                                          dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        """前向：草图 → 条件通道（含边缘特征）。"""
        # 边缘特征
        edge_x = F.conv2d(sketch, self.sobel_x, padding=1)
        edge_y = F.conv2d(sketch, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        # 拼接原始 + 边缘 → 2 通道
        x = torch.cat([sketch, edge], dim=1)   # (B, 2, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.out(x)
        # 残差
        return out + self.skip(sketch)


class ElevationHintEncoder(nn.Module):
    """高度提示编码器：保留高程绝对值 + 梯度信息。

    输入：高度提示 (B, 1, H, W)，值域 [-1, 1]（归一化高程）
    输出：编码特征 (B, out_ch, H, W)
    """

    def __init__(self, out_ch: int = 4, base: int = 32):
        """初始化高度提示编码器。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, out_ch, 1),
        )
        self.skip = nn.Conv2d(1, out_ch, 1)

    def forward(self, elev_hint: torch.Tensor) -> torch.Tensor:
        """前向：高度提示 → 条件通道。"""
        return self.net(elev_hint) + self.skip(elev_hint)


class BiomeEncoder(nn.Module):
    """生物群落编码器：one-hot 降维 + 空间关系提取。

    输入：群落图 (B, n_biomes, H, W)，one-hot 或概率分布
    输出：编码特征 (B, out_ch, H, W)
    """

    def __init__(self, n_biomes: int = 14, out_ch: int = 4, base: int = 32):
        """初始化生物群落编码器。"""
        super().__init__()
        self.reduce = nn.Conv2d(n_biomes, base, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, out_ch, 1),
        )

    def forward(self, biome: torch.Tensor) -> torch.Tensor:
        """前向：群落图 → 条件通道。"""
        x = self.reduce(biome)
        return self.refine(x)


# ==================== 4. 预设模态编码器 ====================

class SeaLevelEncoder(nn.Module):
    """海平面条件编码器：控制全球海平面高度。

    输入：海平面偏移 (B, 1, H, W)，值域 [-1, 1]
        0 = 当前海平面，正值 = 升高，负值 = 降低
    输出：编码特征 (B, out_ch, H, W)

    原理：海平面变化影响陆地/海洋分布，编码为全局条件。
    """

    def __init__(self, out_ch: int = 4, base: int = 32):
        """初始化海平面编码器。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, out_ch, 1),
        )

    def forward(self, sea_level: torch.Tensor) -> torch.Tensor:
        """前向：海平面偏移 → 条件通道。"""
        return self.net(sea_level)


class ClimateZoneEncoder(nn.Module):
    """气候带条件编码器：指定区域气候类型。

    输入：气候带图 (B, n_climate_zones, H, W)，one-hot
        气候带：赤道/副热带/温带/副寒带/寒带（5 类）
    输出：编码特征 (B, out_ch, H, W)
    """

    def __init__(self, n_zones: int = 5, out_ch: int = 4, base: int = 32):
        """初始化气候带编码器。"""
        super().__init__()
        self.reduce = nn.Conv2d(n_zones, base, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, out_ch, 1),
        )

    def forward(self, climate: torch.Tensor) -> torch.Tensor:
        """前向：气候带 → 条件通道。"""
        x = self.reduce(climate)
        return self.refine(x)


class PlateBoundaryEncoder(nn.Module):
    """板块边界条件编码器：控制板块边界位置。

    输入：板块边界图 (B, 3, H, W)，3 通道
        ch0: 发散边界（洋中脊），ch1: 汇聚边界（海沟），ch2: 转换边界
    输出：编码特征 (B, out_ch, H, W)
    """

    def __init__(self, out_ch: int = 4, base: int = 32):
        """初始化板块边界编码器。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base * 2, out_ch, 1),
        )

    def forward(self, boundary: torch.Tensor) -> torch.Tensor:
        """前向：板块边界 → 条件通道。"""
        return self.net(boundary)


# ==================== 5. 多模态融合器（优化版） ====================

class MultiModalConditioner(nn.Module):
    """多模态条件编码器（优化版）。

    优化特性
    --------
    - 交叉注意力融合：模态特征 → Q/K/V 注入板块底图
    - 模态权重控制：ModalityConfig 调节各模态控制强度
    - 空间掩码：仅在指定区域应用模态条件（局部控制）
    - 残差连接：保留原始板块信息，防止条件覆盖
    - 预设模态：海平面/气候带/板块边界

    用法
    ----
        >>> cfg = ModalityConfig(sketch_weight=0.8, use_cross_attention=True)
        >>> cond = MultiModalConditioner(modality_config=cfg)
        >>> out, meta = cond(plate_12ch, sketch=sketch_map)
    """

    def __init__(self, base_ch: int = 12,
                 sketch_ch: int = 4,
                 elev_ch: int = 4,
                 biome_ch: int = 4,
                 sea_level_ch: int = 4,
                 climate_ch: int = 4,
                 plate_boundary_ch: int = 4,
                 n_biomes: int = 14,
                 n_climate_zones: int = 5,
                 base: int = 32,
                 modality_config: ModalityConfig | None = None):
        """初始化多模态条件器。

        Parameters
        ----------
        base_ch : int  板块底图通道数
        sketch_ch / elev_ch / biome_ch : int  各模态输出通道
        sea_level_ch / climate_ch / plate_boundary_ch : int  预设模态通道
        n_biomes : int  生物群落类别数
        n_climate_zones : int  气候带类别数
        base : int  各编码器基础通道
        modality_config : ModalityConfig  模态控制配置
        """
        super().__init__()
        self.base_ch = base_ch
        self.sketch_ch = sketch_ch
        self.elev_ch = elev_ch
        self.biome_ch = biome_ch
        self.sea_level_ch = sea_level_ch
        self.climate_ch = climate_ch
        self.plate_boundary_ch = plate_boundary_ch

        self.config = modality_config or ModalityConfig()

        # 各模态编码器
        self.sketch_enc = SketchEncoder(out_ch=sketch_ch, base=base)
        self.elev_enc = ElevationHintEncoder(out_ch=elev_ch, base=base)
        self.biome_enc = BiomeEncoder(n_biomes=n_biomes, out_ch=biome_ch,
                                      base=base)
        self.sea_level_enc = SeaLevelEncoder(out_ch=sea_level_ch, base=base)
        self.climate_enc = ClimateZoneEncoder(n_zones=n_climate_zones,
                                               out_ch=climate_ch, base=base)
        self.plate_boundary_enc = PlateBoundaryEncoder(
            out_ch=plate_boundary_ch, base=base)

        # 拼接总通道
        self.total_cond_ch = (sketch_ch + elev_ch + biome_ch +
                              sea_level_ch + climate_ch +
                              plate_boundary_ch)

        # 交叉注意力融合（可选）
        if self.config.use_cross_attention:
            self.cross_attn = CrossAttentionFusion(
                base_ch=base_ch, cond_ch=self.total_cond_ch,
                n_heads=4, dropout=0.0)
        else:
            self.cross_attn = None

        # 拼接融合层
        total_out = base_ch + self.total_cond_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(total_out, total_out, 1),
            nn.SiLU(),
            nn.Conv2d(total_out, total_out, 1),
        )

    @property
    def output_channels(self) -> int:
        """输出条件图总通道数。"""
        return self.base_ch + self.total_cond_ch

    def forward(self, plate_12ch: torch.Tensor,
                sketch: torch.Tensor | None = None,
                elev_hint: torch.Tensor | None = None,
                biome: torch.Tensor | None = None,
                sea_level: torch.Tensor | None = None,
                climate: torch.Tensor | None = None,
                plate_boundary: torch.Tensor | None = None,
                sketch_mask: torch.Tensor | None = None,
                elev_mask: torch.Tensor | None = None,
                biome_mask: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, dict]:
        """多模态条件融合（优化版）。

        Parameters
        ----------
        plate_12ch : Tensor (B, 12, H, W)  板块底图（必须）
        sketch : Tensor (B, 1, H, W), optional  草图 [0, 1]
        elev_hint : Tensor (B, 1, H, W), optional  高度提示 [-1, 1]
        biome : Tensor (B, n_biomes, H, W), optional  生物群落
        sea_level : Tensor (B, 1, H, W), optional  海平面偏移 [-1, 1]
        climate : Tensor (B, n_zones, H, W), optional  气候带
        plate_boundary : Tensor (B, 3, H, W), optional  板块边界
        sketch_mask : Tensor (B, 1, H, W), optional  草图空间掩码 [0, 1]
        elev_mask : Tensor (B, 1, H, W), optional  高度提示空间掩码
        biome_mask : Tensor (B, 1, H, W), optional  生物群落空间掩码

        Returns
        -------
        cond : Tensor (B, base_ch+total_cond_ch, H, W)  融合条件图
        meta : dict  各模态使用情况与权重
        """
        B, _, H, W = plate_12ch.shape
        device = plate_12ch.device
        cfg = self.config
        meta = {"used": ["plate"], "weights": {}, "masked": []}

        # 板块底图（必须）
        parts = []

        # 草图
        s = self._encode_modality(
            "sketch", sketch, self.sketch_enc, self.sketch_ch,
            B, H, W, device, sketch_mask, meta)
        parts.append(s)

        # 高度提示
        e = self._encode_modality(
            "elev_hint", elev_hint, self.elev_enc, self.elev_ch,
            B, H, W, device, elev_mask, meta)
        parts.append(e)

        # 生物群落
        b = self._encode_modality(
            "biome", biome, self.biome_enc, self.biome_ch,
            B, H, W, device, biome_mask, meta)
        parts.append(b)

        # 海平面（无掩码）
        sl = self._encode_modality(
            "sea_level", sea_level, self.sea_level_enc,
            self.sea_level_ch, B, H, W, device, None, meta)
        parts.append(sl)

        # 气候带
        cl = self._encode_modality(
            "climate", climate, self.climate_enc,
            self.climate_ch, B, H, W, device, None, meta)
        parts.append(cl)

        # 板块边界
        pb = self._encode_modality(
            "plate_boundary", plate_boundary,
            self.plate_boundary_enc, self.plate_boundary_ch,
            B, H, W, device, None, meta)
        parts.append(pb)

        cond_features = torch.cat(parts, dim=1)   # (B, total_cond_ch, H, W)

        # 交叉注意力融合（可选）
        if self.cross_attn is not None:
            enhanced_base = self.cross_attn(plate_12ch, cond_features)
            if cfg.residual_connection:
                enhanced_base = enhanced_base + plate_12ch
        else:
            enhanced_base = plate_12ch

        # 拼接 + 融合
        cond = torch.cat([enhanced_base, cond_features], dim=1)
        cond = self.fuse(cond)

        meta["output_channels"] = cond.shape[1]
        return cond, meta

    def _encode_modality(self, name: str,
                         x: torch.Tensor | None,
                         encoder: nn.Module,
                         out_ch: int,
                         B: int, H: int, W: int,
                         device: torch.device,
                         mask: torch.Tensor | None,
                         meta: dict) -> torch.Tensor:
        """编码单个模态（带权重与掩码）。"""
        weight = self.config.get_weight(name)

        if x is not None:
            encoded = encoder(x)
            # 应用权重
            if weight != 1.0:
                encoded = encoded * weight
            # 应用空间掩码（局部控制）
            if mask is not None:
                encoded = encoded * mask
                meta["masked"].append(name)
            meta["used"].append(name)
            meta["weights"][name] = weight
            return encoded
        else:
            # 缺失模态用零填充
            return torch.zeros(B, out_ch, H, W, device=device)


# ==================== 6. 条件校验与预处理 ====================

def validate_sketch(sketch: np.ndarray) -> np.ndarray:
    """校验并预处理草图。

    - 确保形状为 (H, W) 或 (1, H, W)
    - 值域裁剪到 [0, 1]
    - 二值化选项

    Returns
    -------
    ndarray (1, H, W) float32  校验后的草图
    """
    if sketch.ndim == 2:
        sketch = sketch[None]
    assert sketch.ndim == 3 and sketch.shape[0] == 1, \
        f"草图形状应为 (1, H, W)，得到 {sketch.shape}"
    sketch = np.clip(sketch, 0, 1).astype(np.float32)
    return sketch


def validate_elev_hint(elev_hint: np.ndarray) -> np.ndarray:
    """校验并预处理高度提示。

    - 确保形状为 (1, H, W)
    - 值域裁剪到 [-1, 1]

    Returns
    -------
    ndarray (1, H, W) float32
    """
    if elev_hint.ndim == 2:
        elev_hint = elev_hint[None]
    assert elev_hint.ndim == 3 and elev_hint.shape[0] == 1, \
        f"高度提示形状应为 (1, H, W)，得到 {elev_hint.shape}"
    elev_hint = np.clip(elev_hint, -1, 1).astype(np.float32)
    return elev_hint


def validate_biome(biome: np.ndarray, n_biomes: int = 14
                   ) -> np.ndarray:
    """校验并预处理生物群落图。

    - 支持标签图 (H, W) int → one-hot (n_biomes, H, W)
    - 支持 one-hot/probability (n_biomes, H, W) 直接通过

    Returns
    -------
    ndarray (n_biomes, H, W) float32
    """
    if biome.ndim == 2:
        # 标签图 → one-hot
        H, W = biome.shape
        one_hot = np.zeros((n_biomes, H, W), dtype=np.float32)
        for i in range(n_biomes):
            one_hot[i] = (biome == i).astype(np.float32)
        return one_hot
    elif biome.ndim == 3:
        assert biome.shape[0] == n_biomes, \
            f"群落图通道数应为 {n_biomes}，得到 {biome.shape[0]}"
        return biome.astype(np.float32)
    else:
        raise ValueError(f"群落图维度应为 2 或 3，得到 {biome.ndim}")


def make_sea_level_field(sea_level_offset: float,
                         H: int, W: int) -> np.ndarray:
    """生成海平面条件场。

    Parameters
    ----------
    sea_level_offset : float  海平面偏移 [-1, 1]
        正值 = 海平面升高（更多陆地被淹）
        负值 = 海平面降低（更多海底露出）
    H, W : int  空间尺寸

    Returns
    -------
    ndarray (1, H, W) float32
    """
    offset = np.clip(sea_level_offset, -1, 1)
    return np.full((1, H, W), offset, dtype=np.float32)


def make_climate_zone_field(latitudes: np.ndarray,
                            n_zones: int = 5) -> np.ndarray:
    """根据纬度生成气候带 one-hot 图。

    5 个气候带：
    - 0: 赤道带 (|lat| < 10°)
    - 1: 副热带 (10° ≤ |lat| < 35°)
    - 2: 温带 (35° ≤ |lat| < 55°)
    - 3: 副寒带 (55° ≤ |lat| < 66.5°)
    - 4: 寒带 (|lat| ≥ 66.5°)

    Parameters
    ----------
    latitudes : ndarray (H,)  纬度数组 [-90, 90]
    n_zones : int  气候带数

    Returns
    -------
    ndarray (n_zones, H, W) float32  one-hot 气候带图
    """
    H = len(latitudes)
    W = H   # 假设方形
    abs_lat = np.abs(latitudes)

    # 分带
    zones = np.zeros(H, dtype=np.int32)
    zones[abs_lat < 10] = 0
    zones[(abs_lat >= 10) & (abs_lat < 35)] = 1
    zones[(abs_lat >= 35) & (abs_lat < 55)] = 2
    zones[(abs_lat >= 55) & (abs_lat < 66.5)] = 3
    zones[abs_lat >= 66.5] = 4

    # one-hot
    one_hot = np.zeros((n_zones, H, W), dtype=np.float32)
    for i in range(n_zones):
        one_hot[i, zones == i, :] = 1.0
    return one_hot


# ==================== 7. 便捷函数（增强版） ====================

def encode_multimodal(plate_12ch: np.ndarray,
                      sketch: np.ndarray | None = None,
                      elev_hint: np.ndarray | None = None,
                      biome: np.ndarray | None = None,
                      sea_level: np.ndarray | None = None,
                      climate: np.ndarray | None = None,
                      plate_boundary: np.ndarray | None = None,
                      sketch_mask: np.ndarray | None = None,
                      elev_mask: np.ndarray | None = None,
                      biome_mask: np.ndarray | None = None,
                      modality_config: ModalityConfig | None = None,
                      device: torch.device | None = None,
                      ) -> np.ndarray:
    """NumPy 接口的多模态条件编码（增强版）。

    支持所有模态 + 空间掩码 + 权重控制。

    Parameters
    ----------
    plate_12ch : ndarray (12, H, W)  板块底图
    sketch / elev_hint / biome : optional  核心模态
    sea_level / climate / plate_boundary : optional  预设模态
    sketch_mask / elev_mask / biome_mask : optional  空间掩码
    modality_config : ModalityConfig, optional  模态控制配置
    device : torch.device, optional  推理设备

    Returns
    -------
    ndarray (C, H, W)  融合条件图
    """
    from .config import DEVICE
    dev = device or DEVICE
    cond = MultiModalConditioner(modality_config=modality_config)
    cond = cond.to(dev).eval()

    def to_t(x):
        """把 (H,W) 或 (C,H,W) 数组转成 (1,C,H,W) 张量并移到目标设备。"""
        if x.ndim == 2:
            x = x[None]
        return torch.from_numpy(x).float().unsqueeze(0).to(dev)

    def to_mask_t(m):
        """把掩码数组转成 (1,1,H,W) 张量并移到目标设备。"""
        if m.ndim == 2:
            m = m[None]
        return torch.from_numpy(m).float().unsqueeze(0).to(dev)

    p = to_t(plate_12ch)
    s = to_t(sketch) if sketch is not None else None
    e = to_t(elev_hint) if elev_hint is not None else None
    b = to_t(biome) if biome is not None else None
    sl = to_t(sea_level) if sea_level is not None else None
    cl = to_t(climate) if climate is not None else None
    pb = to_t(plate_boundary) if plate_boundary is not None else None

    sm = to_mask_t(sketch_mask) if sketch_mask is not None else None
    em = to_mask_t(elev_mask) if elev_mask is not None else None
    bm = to_mask_t(biome_mask) if biome_mask is not None else None

    with torch.no_grad():
        out, _ = cond(p, sketch=s, elev_hint=e, biome=b,
                      sea_level=sl, climate=cl, plate_boundary=pb,
                      sketch_mask=sm, elev_mask=em, biome_mask=bm)
    return out.squeeze(0).cpu().numpy()
