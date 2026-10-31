"""地形深度学习模块的全局配置：设备、路径、超参数、高程归一化。

约定
----
- 高程数据以 float32 numpy 数组存储，单位米，范围 [-12000, 9000]
- 网络输入归一化到 [-1, 1]：``x_norm = 2 * (x - zmin) / (zmax - zmin) - 1``
- 设备选择与 ``terrain/core/terrain_engine.py`` 保持一致：MPS > CUDA > CPU
- 路径遵循 ``.gitignore`` 已排除的 ``terrain/data`` / ``terrain/checkpoints`` / ``terrain/cache``
"""
from __future__ import annotations
from pathlib import Path
import torch

# ==================== 路径 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # 项目根（Fantasy_World_Simulator/）
TERRAIN_DIR = PROJECT_ROOT / "terrain"
DATA_DIR = TERRAIN_DIR / "data"
DEM_TILES_DIR = DATA_DIR / "dem_tiles"               # 无标签 DEM 切片（.npy）
PAIRED_DIR = DATA_DIR / "paired"                     # 少量配对样本（阶段 2）
CKPT_DIR = TERRAIN_DIR / "checkpoints"               # 模型权重
CACHE_DIR = TERRAIN_DIR / "cache"                    # 中间缓存（如参考统计量）
LOG_DIR = TERRAIN_DIR / "cache" / "logs"             # 训练日志（CSV / TensorBoard）

for _d in (DEM_TILES_DIR, PAIRED_DIR, CKPT_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ==================== 设备（与 terrain_engine.py:L37-42 一致） ====================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ==================== 高程归一化 ====================
ELEV_MIN = -12000.0   # 米，覆盖马里亚纳海沟
ELEV_MAX = 9000.0     # 米，覆盖珠峰
ELEV_RANGE = ELEV_MAX - ELEV_MIN


def elev_to_norm(x):
    """米 -> [-1, 1]。支持 numpy 与 torch。"""
    return 2.0 * (x - ELEV_MIN) / ELEV_RANGE - 1.0


def norm_to_elev(x):
    """[-1, 1] -> 米。支持 numpy 与 torch。"""
    return (x + 1.0) * 0.5 * ELEV_RANGE + ELEV_MIN


# ==================== 数据与模型超参 ====================
TILE_SIZE = 256            # 切片尺寸（像素）
IN_CHANNELS = 1            # 阶段 1 仅高程单通道
LATENT_CHANNELS = 4        # VAE 潜空间通道
LATENT_SIZE = TILE_SIZE // 8  # 32

# ==================== 训练超参 ====================
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 500         # 余弦调度预热步数
GRAD_CLIP = 1.0            # 梯度裁剪范数
EMA_DECAY = 0.9999         # 模型权重 EMA 衰减
USE_AMP = True             # 混合精度（CUDA/MPS 有效；CPU 自动退化为 fp32）

# ==================== 扩散超参 ====================
DIFFUSION_STEPS = 1000     # DDPM 训练步数 T
BETA_START = 1e-4          # 线性 β 调度下界
BETA_END = 2e-2            # 线性 β 调度上界
SAMPLING_STEPS = 50        # DDIM 推理采样步数（阶段 2 启用）

# ==================== 评估超参 ====================
EVAL_INTERVAL_EPOCHS = 5   # 每多少 epoch 跑一次完整评估
SAMPLE_NUM = 4             # 评估时生成的样本数
SAMPLE_SIZE = TILE_SIZE
