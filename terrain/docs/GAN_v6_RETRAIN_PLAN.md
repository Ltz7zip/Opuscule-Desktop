# GAN v6 重训方案（海洋波纹专项）

> 版本：v6.0 方案稿，2026-10-31
> 状态：**待评审，未落地任何代码**
> 涉及文件：`terrain/core/terrain_engine.py`（唯一需要改的文件）
> 不涉及：`terrain/core/generate_world.py`（v82 的 9 处渲染改动全部保留，不做回退）

---

## 1. 问题定义

用户反馈：海洋区域出现明显波纹。v77→v82 五轮配色/晕渲调整后判定为「改善极为有限」，说明问题不在渲染层。

### 1.1 定量判据（已建立）

同尺度 256px 窗（≈45° 经度跨度）、取最平坦的 12 个深海窗、`p - gaussian(p,40)` 去趋势、Hann 窗、径向谱按总能量归一，各波长带相对谱能占比：

| 波长带(px) | 150-120 | 120-100 | 100-80 | 80-60 | 60-40 | 40-28 | 28-18 | 18-12 | 12-8 | 8-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| GAN v82b | 0.196 | 0.226 | 0.186 | 0.113 | 0.079 | **0.108** | **0.060** | 0.027 | 0.006 | 0.001 |
| ETOPO1 实拍 | 0.307 | 0.295 | 0.220 | 0.117 | 0.044 | **0.011** | **0.004** | 0.001 | 0.000 | 0.000 |

结论：

- 40-28px 带：GAN 是真实海底的 **10 倍**
- 28-18px 带：**15 倍**
- 绝对起伏反而偏小（深海窗 std：GAN 149m vs 真实 470m）

即：**不是纹理太多，是纹理挤在一个错误的尺度上，且大尺度起伏不足。**

### 1.2 已排除的机理

| 假设 | 排除依据 |
|---|---|
| 上采样产生棋盘/锯齿 | 解码器 `Upsample` 已是 resize-conv（bilinear + 3×3 conv），见 `terrain_engine.py` L779-791；用户提议的改造是 no-op |
| 输入 4× 最近邻台阶 | `np.repeat(ch, block)` 的台阶能量只落在 8/5.3/4/3.2/2.7/2px，实测该段接近噪声底 |
| 瓦片网格指纹 | 主峰随版本漂移（v80 140px → v82 85px），而瓦片步长三版未变（232 世界px） |
| 训练/推理尺度失配 | ETOPO1 降采样 10× 后径向谱形不变（地形自仿射），尺度失配不解释谱超标 |
| 渲染层 | v77-v82 五轮改动（去颗粒、带通后移、分形补细节、坡强门控等）效果有限 |

### 1.3 未排除的次级嫌疑

- 232px 步长的谐波（116/77/58/46/39px）落在波纹带内
- 重叠仅 24px（9.4%）的汉宁羽化偏薄，相邻瓦片纹理统计在边界突变可能形成准周期调制

这两项**不在本次重训范围内**，属于推理期问题，留待重训后再评估。

### 1.4 为什么必须改损失，而不是只加 epoch

线上 checkpoint：`epoch=29`（= loop 最后一轮）且是 `val_score` 最优 → 训练被 30 epoch 预算**截断**，不是收敛。所以加 epoch 有效。

但当前谱约束设计有缺陷：

- `FFTLoss` 比较的是**原始幅值 L1**，被 DC 与低频主导，对 30px 带几乎没有梯度
- `w_fft = 2.0` 是常数，谱约束在总损失里占比 <7%

**不改这一项，跑到 240 epoch 只是把同一个错误谱学得更熟。**

---

## 2. 改动清单（7 处，全部在 `terrain/core/terrain_engine.py`）

### ① 新增损失类：插在 `FFTLoss` 之后（L955 与 L958 之间）

```python
class OceanBandSpectralLoss(nn.Module):
    """深海带能谱比损失：惩罚 12~40px 波段相对谱能超标（波纹所在带）。

    与 FFTLoss 的三个关键区别
    -------------------------
    1. 先除以本图总能量 → 只约束**谱形**，不约束绝对幅度（幅度归 L1/Dist 管）；
       FFTLoss 比原始幅值，被 DC 与低频主导，对 30px 带几乎没有梯度。
    2. 用 log10 比 → 「该带能量是真实的 10 倍」就是 1.0，语义与验收表一致。
    3. 只统计海洋像元（平滑掩膜加权）→ 不牵连陆地山地的真实高频。

    实测（同尺度 256px 最平坦深海窗，径向谱按总能量归一）：
        40-28px  本工程 0.108 / ETOPO1 0.011  → 10×
        28-18px  本工程 0.060 / ETOPO1 0.004  → 15×
    """
    def __init__(self, size=256, detrend_sigma=40.0,
                 edges=(12, 18, 28, 40, 64, 128),
                 w=(0.3, 0.5, 1.5, 3.0, 0.4, 0.15, 0.0), eps=1e-8):
        """``__init__``: 预计算去趋势高通响应 hp 与波长分带索引 bidx（只读常量）。"""
        super().__init__()
        fr = torch.fft.fftfreq(size) * size          # 整幅内的周期数
        fx = torch.fft.rfftfreq(size) * size
        r = torch.sqrt(fr[:, None] ** 2 + fx[None, :] ** 2)
        lam = size / r.clamp_min(1e-6)               # 波长 (px)
        # 与 /tmp/dl_spec_cmp.py 口径一致: p - gaussian(p, 40)
        # 频域里等价于乘 1 - exp(-(2πσr/size)²/2)，零成本，不必做空间卷积
        self.hp = 1.0 - torch.exp(-0.5 * (2 * np.pi * detrend_sigma * r / size) ** 2)
        self.bidx = torch.bucketize(lam, torch.tensor(edges)).reshape(-1)
        self.w = torch.tensor(w)                     # 权重峰值在 28-40px
        self.nb = len(w)
        self.eps = eps
        self._cpu = False

    def _spec(self, x):
        """rfft2；MPS 复数支持不完整时一次性降级到 CPU（同 FFTLoss 策略）。"""
        if not self._cpu:
            try:
                return torch.fft.rfft2(x)
            except (RuntimeError, TypeError):
                self._cpu = True
        return torch.fft.rfft2(x.cpu())

    def _shares(self, fake, real, mask):
        """返回 (fake 带能量份额, real 带能量份额)；无有效海洋块时返回 (None, None)。"""
        valid = mask.mean((1, 2, 3)) > 0.5           # 陆地为主的块不参与
        if not bool(valid.any()):
            return None, None
        fake, real, mask = fake[valid], real[valid], mask[valid]
        B, dev = int(valid.sum()), fake.device
        msum = mask.sum((1, 2, 3), keepdim=True).clamp_min(1e-6)
        mf = (fake * mask).sum((1, 2, 3), keepdim=True) / msum
        mr = (real * mask).sum((1, 2, 3), keepdim=True) / msum
        x = torch.cat([(fake - mf) * mask, (real - mr) * mask], 0)
        P = self._spec(x).abs() ** 2 * self.hp.to(dev)
        e = P.new_zeros(2 * B, self.nb).index_add_(
            1, self.bidx.to(dev).expand(2 * B, -1), P.reshape(2 * B, -1))
        e = e / e.sum(1, keepdim=True).clamp_min(self.eps)   # 归一化成「带能量份额」
        return e[:B].mean(0), e[B:].mean(0)                  # 批均值降低单块噪声

    def forward(self, fake, real, mask):
        """mask: 平滑海面掩膜 (B,1,H,W)，由 (target < 0.5) 池化得到。"""
        ef, er = self._shares(fake, real, mask)
        if ef is None:
            return fake.new_tensor(0.0)
        ratio = (torch.log10(ef + self.eps) - torch.log10(er + self.eps)).abs()
        return (self.w.to(ratio.device) * ratio.clamp(max=1.5)).sum().to(fake.device)

    @torch.no_grad()
    def band_ratios(self, fake, real, mask):
        """诊断用：返回 7 个带的 log10(fake/real)，训练日志里按 ×倍数 打印。"""
        ef, er = self._shares(fake, real, mask)
        if ef is None:
            return None
        return torch.log10(ef + self.eps) - torch.log10(er + self.eps)
```

**设计要点**

- `ratio.clamp(max=1.5)`：单带最多罚到 31×。防止早期某带份额趋 0 时 log 爆掉主导梯度。
- 分带权重 `w=(0.3, 0.5, 1.5, 3.0, 0.4, 0.15, 0.0)` 对应 `<12 / 12-18 / 18-28 / 28-40 / 40-64 / 64-128 / >128` px。
- 掩膜阈值 `mask.mean() > 0.5`：陆地为主的块完全跳过，避免把山地高频当波纹压制。

### ② 训练配置（L2494-2499）

```python
    size, block = 256, 4
    n_train, n_val, batch, epochs, warmup = 1200, 120, 8, 120, 4   # epochs 30 → 120
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints", "best_terrain_gan_v6.pth")   # v5 → v6
```

`CosineAnnealingLR(T_max=epochs)` 会自动拉伸到 120 轮，不需要额外改。`warmup=4`、`patience=10` 保持不变。

### ③ 实例化损失（L2531 旁）

```python
    crit_fft = FFTLoss().to(device)
    crit_spec = OceanBandSpectralLoss(size=size).to(device)   # 新增
```

### ④ 权重调度（L2551-2556）

```python
        w_edge, w_fft, w_lap = 4.0, 1.0, 1.5      # w_fft 2.0 → 1.0
        w_spec = 4.0 * warm                        # 新增：与 GAN 同步预热
        w_slope = 1.0 + 2.0 * prog
        w_tile = 1.5 * warm
```

`w_fft` 降一半的原因：新的带能谱损失接管了中高频谱形，旧的只保留低频骨架约束；两者都保持 2.0 会在低频上重复计权。

### ⑤ 损失合成（L2569 附近与 L2600-2605）

先在海面掩膜处算一次（放在 `bm_t` 之后，每步只算一次）：

```python
            bm_t = F.avg_pool2d(inp[:, 0:1] * ch_std_t[:, 0:1] + ch_mean_t[:, 0:1], block)
            # v6: 平滑海面掩膜（0.5 = 海平面, 9px 软边）
            sea_m = F.avg_pool2d((target < 0.5).float(), 9, 1, 4)
```

再接入总损失：

```python
            loss = (w_l1 * crit_l1(fake, target) + w_dist * crit_dist(fake, target)
                    + w_edge * crit_edge(fake, target) + w_fft * crit_fft(fake, target)
                    + w_lap * crit_lap(fake, target) + w_gan * loss_gan + w_fm * loss_fm
                    + w_cons * F.l1_loss(bm_fake, bm_t)
                    + w_slope * crit_slope(fake, target)
                    + w_spec * crit_spec(fake, target, sea_m)      # 新增
                    + w_tile * loss_tile)
```

### ⑥ 验证日志（L2622-2641）

```python
        v_l1 = v_edge = v_dist = v_slope = 0.0
        v_spec = 0.0
        last_band = None
        with torch.no_grad():
            for inp, target in val_loader:
                inp, target = inp.to(device), target.to(device)
                fake = gen(inp)
                sea_m = F.avg_pool2d((target < 0.5).float(), 9, 1, 4)
                v_l1 += crit_l1(fake, target).item()
                v_edge += crit_edge(fake, target).item()
                v_dist += crit_dist(fake, target).item()
                v_slope += crit_slope(fake, target).item()
                v_spec += float(crit_spec(fake, target, sea_m))
                last_band = crit_spec.band_ratios(fake, target, sea_m)
```

打印行追加 `Spec`，并每 10 轮把 7 个带的倍数打出来：

```python
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch:03d}/{epochs}] | D: {avg_d:.4f} | G: {avg_g:.3f} | "
                  f"Val L1: {v_l1:.4f} | Edge: {v_edge:.4f} | Dist: {v_dist:.4f} | "
                  f"Slope: {v_slope:.4f} | Spec: {v_spec/m:.3f} | "
                  f"w_GAN:{w_gan:.1f} w_L1:{w_l1:.0f} | {time.time()-t_ep:.0f}s/ep")
            if last_band is not None and epoch % 10 == 0:
                print("   带比(fake/real ×): " + "  ".join(
                    f"{b}:{10**float(v):.2f}" for b, v in
                    zip(("<12", "12-18", "18-28", "28-40", "40-64", "64-128", ">128"),
                        last_band)))
```

带比从 10× 降到 1~2× 的过程是**唯一能在线判断波纹是否被压住的指标**。只靠 `val_score` 看不出来，因为 `val_score = L1 + 2·Edge + 5·Dist` 完全不含谱项。

### ⑦ 采样配额（L706）

```python
                p = 0.85 if f > 0.45 else (0.45 if f > 0.08 else 0.35)   # 0.10 → 0.35
```

按面积加权估算，开阔洋在训练集里的占比从 ~19% 升到 ~45%。问题正好出在这一类样本上；陆地占比靠 `f>0.45` 那档守住（绝对样本数不变）。

---

## 3. 不变更项（明确冻结）

| 项 | 值 | 理由 |
|---|---|---|
| `block` | 4 | 条件分辨率 64²；改成 8 已被证明会「把山摊平」 |
| `base` | 32 | 与现有 checkpoint 结构兼容 |
| `in_ch` | 12 | 12 通道条件（含构造场） |
| `warmup` | 4 | 与 GAN 的预热节奏绑定 |
| `patience` | 10 | 早停是正常行为，见风险 1 |
| `n_train` | 1200 | 每轮样本数 |
| 全部渲染逻辑 | v82 的 9 处改动 | 与训练无关，重训后照旧生效 |
| `generate_world.py` | 不动 | 只通过 `--ckpt` 切换权重 |

---

## 4. 验收判据

跑完后对 2048 全量图取最平坦的 12 个深海 256px 窗，径向谱按总能量归一（复用 `/tmp/dl_spec_cmp.py` 口径）：

| 波长带 | v82 实测 | 真实 ETOPO1 | 目标 | 性质 |
|---|---|---|---|---|
| 40-28px | 0.108 | 0.011 | **≤0.020** | 核心指标 |
| 28-18px | 0.060 | 0.004 | **≤0.010** | 核心指标 |
| 40-64px | 0.079 | 0.044 | ≤0.060 | 顺带改善 |
| 64-128px | 0.186 | 0.220 | **≥0.150** | 防过平滑守门 |
| 深海窗 std | 149m | 470m | **≥250m** | 防过平滑守门 |

最后两行是**防过平滑**的关键：只压 30px 带最省事的解法是把海底抹平，必须同时确认低频能量和绝对起伏没掉。

---

## 5. 风险

1. **`patience=10` 可能在 120 轮中途早停。**
   这是正常现象：`val_score`（L1+Edge+Dist）不含谱项，谱改善不体现在它上面。所以冒烟阶段必须确认「带比在降」，而不是只看 `val_score`。

2. **谱损失跑在 CPU 上。**
   MPS 复数 FFT 支持不完整，沿用 `FFTLoss` 的降级策略。每步多一次 256²×16 的 CPU FFT，约 +8~12% 时间：120 轮从 10-16h 变为 **11-17h**。

3. **过平滑。**
   见第 4 节两条守门指标。

4. **开阔洋采样提到 0.35 会挤压陆地样本。**
   陆地绝对样本数不变（`f>0.45` 档仍为 0.85），但相对占比下降。若陆地纹理出现退化，回落到 0.25 重跑。

5. **从零训，不做 warm-start。**
   谱损失改变了损失面，从 v5 微调会让判别器沿用旧谱判据。代价是 LR 曲线重走一遍，收益是结论干净。若时间紧张，可作为第二优先级的对照实验。

---

## 6. 执行顺序

| 步骤 | 内容 | 判据 | 成本 |
|---|---|---|---|
| 1 | 落地第 2 节 7 处改动 | 语法检查 + `smoke_test` 通过 | 分钟级 |
| 2 | 3 epoch 冒烟 | 无 NaN；`Spec` 有限且递减；带比中 28-40 从 ~10 开始降 | ~35min |
| 3 | 120 epoch 全量重训 | 早停或跑满，`best_terrain_gan_v6.pth` 落盘 | 11-17h |
| 4 | 单种子 2048 全量生成 | 写入 `terrain/output/_seeds/gan_v6_20261031/`，不碰主产出 | ~5min |
| 5 | `/tmp/dl_spec_cmp.py` 复核 | 第 4 节全部达标 | 分钟级 |

任一项不达标 → `--ckpt terrain/checkpoints/best_terrain_gan_v5.pth` 直接回退，v5 全程未被写入，零成本。

### 训练入口

`main()` 发现 v5 权重存在就只做可视化，所以用一次性脚本调 `train()`：

```python
# /tmp/train_v6.py
import sys
sys.path.insert(0, "/Users/ltz2024/Opuscule Desktop")
from terrain.core.terrain_engine import GlobalDEM, train
train(GlobalDEM("/Users/ltz2024/Opuscule Desktop/terrain/data/ETOPO1_Ice_g_gmt4.grd"))
```

```bash
cd "/Users/ltz2024/Opuscule Desktop" && \
PYTHONPATH="." nohup .venv/bin/python /tmp/train_v6.py \
  > terrain/output/_logs/train_v6_$(date +%Y%m%d_%H%M).log 2>&1 &
```

日志正常增长就**不要 `StopCommand`**（此前误判过一次挂死）。

---

## 7. 回滚与产物隔离

| 产物 | 路径 | 说明 |
|---|---|---|
| 新权重 | `terrain/checkpoints/best_terrain_gan_v6.pth` | 新增文件，不覆盖 v5 |
| 旧权重 | `terrain/checkpoints/best_terrain_gan_v5.pth` | **只读，一个字节不动** |
| 训练日志 | `terrain/output/_logs/train_v6_*.log` | 含每 10 轮的 7 带比 |
| 试验出图 | `terrain/output/_seeds/gan_v6_20261031/` | 不写主产出 `terrain/output/` |
| 代码回滚 | `terrain/output/_rollback/` 预先快照 | 改动前备份 `terrain_engine.py` |

`torch.save` 的 `cfg` 字段需补充：

```python
                        'cfg': {'size': size, 'block': block,
                                'in_ch': 12, 'base': 32,
                                'w_spec': 4.0, 'spec_edges': [12, 18, 28, 40, 64, 128]},
```

保证半年后能复现当时的损失配置。

---

## 8. 与「大陆完善」的关系

用户确认大陆方向包含三项：**海洋波纹、群系占比与群系形态（现在是条形的）、轮廓**。

本节只做归档，**不在本次训练方案内执行**：

1. **群系占比**：荒漠 37.9% → 25%、温带草原 23.9% → 8%、雨林 1.7% → 13%。改 `whittaker_biome` 阈值，以 `terrain/supervisor/references.py` 的地球值为靶。
2. **群系形态「条形的」**：根因是温度场几乎纯纬度函数 + 降水场纬向条带，`smooth_biome_labels` 只做平滑不破对称。需要给温度/降水加经向扰动（`continentality_index` 已存在但未进入降水）。
3. **陆块轮廓**：`plate_sim` 的板块尺寸分布 + `polar_despeckle` / `reclaim_inland_water` / `add_island_arcs` 的强度，目标是减少极区碎岛、陆块长短轴比更接近地球。

三条子线都位于 `plate_sim` / `climate_module` / 后处理，与 `terrain_engine.train` 无交集，可以在训练跑的同时并行推进。

---

## 9. 附：本次方案依赖的既有事实

- 线上 checkpoint：`cfg = {'size': 256, 'block': 4, 'in_ch': 12, 'base': 32}`，`epoch=29`，`val_score=0.1242`
- 瓦片推理：`tb = 256 // block = 64` 粗胞、`ovb = 6`、`step = 58` 粗胞 = 232 世界px、`ovp = 24` 世界px；`W = 2048/4 = 512` → 9 行 → 81 瓦片
- 输入上采样：`np.repeat(ch, 4)` 的 4× 最近邻
- 训练/推理尺度：训练 patch 256px = 4.27°，推理瓦片 256px = 45.2°，失配约 10.6×（不解释谱超标，见 1.2）
- 验证带：`VAL_LON_LO, VAL_LON_HI = 130.0, 310.0`（太平洋 held-out）
