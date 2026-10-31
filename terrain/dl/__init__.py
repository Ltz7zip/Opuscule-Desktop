"""地形深度学习模块（terrain.dl）。

阶段 1：数据采集 + 无监督 DDPM 训练 + 评估指标
阶段 2：半监督条件扩散主生成器（LDM + DDIM 采样 + 大规模分块推理）
阶段 3：物理一致性监督子模块（河网/雪线/海岸线）
阶段 4：大规模推理优化（显存管理 + 瓦片调度 + 缓存 + 流水线）
阶段 5：端到端集成管线（LDM + 超分辨率 + 多模态条件 + 物理后处理）

子模块
------
config     : 设备 / 路径 / 超参 / 高程归一化
logger     : CSV 训练日志（可选 TensorBoard）
metrics    : 地形质量评估指标（高程曲线 / 功率谱 / 坡度 / FID / 条纹指数）
evaluator  : 训练中周期性评估器，封装 metrics 调用
vae        : 地形专用 VAE（256²×1 <-> 32²×4 潜空间）
unet       : DDPM 去噪 U-Net（带时间嵌入）
diffusion  : DDPM 前向加噪 / 反向采样调度器
dataset    : DEM 切片数据集与增强
train      : 训练入口（EMA / 余弦调度 / 混合精度 / 断点续训 / 周期采样）
infer_ldm  : LDM 推理接口（分块生成）
physics_nets : 物理监督网络（RiverNet/SnowlineNet/CoastNet）
physics_metrics : 物理评估指标
physics_dataset : 物理数据集
train_physics : 物理网络训练入口
infer_physics : 物理网络推理接口
large_infer : 大规模推理优化（MemoryManager/TileScheduler/PipelineInference）
sr_net     : DEM 超分辨率网络（ESRGAN）
multicond  : 多模态条件编码器（草图/高度/群落）
pipeline   : 端到端集成管线
"""
from .config import DEVICE  # noqa: F401

__all__ = ["DEVICE"]
