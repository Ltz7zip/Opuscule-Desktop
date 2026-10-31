"""阶段 3 物理监督网络评估指标。

三个任务各自的完整评估指标：

1. **河网（RiverNet）** —— 二值分割
   - IoU / Dice / Precision / Recall / F1
   - 连通性指数（预测河网 vs 标签河网的骨架重合度）
   - 拓扑正确性（源头-出口路径命中率）

2. **雪线（SnowlineNet）** —— 像素级回归
   - MAE / RMSE / R² / Bias
   - 按纬度分带误差（赤道/副热带/中纬/极地）
   - 陆地像素内的误差分布

3. **海岸线（CoastNet）** —— 三分类语义分割
   - 总体精度 OA / 平均精度 mAcc
   - 各类 IoU + 平均 mIoU
   - 混淆矩阵
   - 边缘锐化度（海岸线带宽）

所有指标接受 numpy 数组（米制或概率），返回 dict。
不依赖 PyTorch，可在 CPU 上高效计算。
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


# ==================== 1. 河网二值分割指标 ====================

def river_metrics(pred: np.ndarray, target: np.ndarray,
                  threshold: float = 0.5) -> dict:
    """河网二值分割评估指标。

    Parameters
    ----------
    pred : (H, W) float32  河网概率 [0, 1]
    target : (H, W) float32 or bool  河网二值标签 {0, 1}
    threshold : float  概率阈值

    Returns
    -------
    dict  含 iou/dice/precision/recall/f1/connectivity/topo_acc
    """
    pred_bin = (pred >= threshold).astype(np.bool_)
    tgt = target.astype(np.bool_)

    tp = np.logical_and(pred_bin, tgt).sum()
    fp = np.logical_and(pred_bin, ~tgt).sum()
    fn = np.logical_and(~pred_bin, tgt).sum()
    tn = np.logical_and(~pred_bin, ~tgt).sum()

    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "connectivity": float(_connectivity_score(pred_bin, tgt)),
        "topo_acc": float(_topo_accuracy(pred_bin, tgt)),
    }


def _connectivity_score(pred: np.ndarray, target: np.ndarray) -> float:
    """连通性评分：预测河网骨架与标签骨架的重合度。

    思路：对双方做形态学骨架化（thin），计算骨架像素的 Jaccard。
    高分 = 预测的河网拓扑结构与真实一致。
    """
    if not pred.any() or not target.any():
        return 0.0
    skel_p = _skeletonize(pred)
    skel_t = _skeletonize(target)
    inter = np.logical_and(skel_p, skel_t).sum()
    union = np.logical_or(skel_p, skel_t).sum()
    return float(inter / max(union, 1))


def _skeletonize(mask: np.ndarray, n_iter: int = 8) -> np.ndarray:
    """简易骨架化：交替腐蚀 + 开运算，保留中心线。

    用查找表法（LUT）的简化版：迭代取形态学细化。
    这里用 erode-dilate 差值近似骨架，避免依赖 skimage。
    """
    m = mask.astype(np.uint8)
    skel = np.zeros_like(m)
    for _ in range(n_iter):
        eroded = ndi.binary_erosion(m)
        opened = ndi.binary_dilation(eroded)
        skel = np.logical_or(skel, m & ~opened)
        m = eroded
        if not m.any():
            break
    return skel.astype(bool)


def _topo_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """拓扑正确性：预测河网与标签河网在源头-出口路径上的命中率。

    简化版：统计预测河网中与标签河网"邻近"（距离 ≤ 3 像素）的像素占比。
    高分 = 预测的河道落在真实河道附近（即使有轻微偏移也算正确）。
    """
    if not pred.any():
        return 0.0
    # 标签的距离变换
    dist = ndi.distance_transform_edt(~target)
    # 预测河网中距离标签 ≤ 3 的像素占比
    near = (dist[pred] <= 3.0).mean()
    return float(near)


# ==================== 2. 雪线回归指标 ====================

def snowline_metrics(pred: np.ndarray, target: np.ndarray,
                     lats: np.ndarray | None = None,
                     land_mask: np.ndarray | None = None) -> dict:
    """雪线高程回归评估指标。

    Parameters
    ----------
    pred : (H, W) float32  预测雪线高程（米）
    target : (H, W) float32  标签雪线高程（米）
    lats : (H,) optional  纬度数组，用于分带误差分析
    land_mask : (H, W) bool optional  陆地掩码；None 时全图参与

    Returns
    -------
    dict  含 mae/rmse/r2/bias + 分带误差
    """
    if land_mask is None:
        mask = np.ones(pred.shape, dtype=bool)
    else:
        mask = land_mask.astype(bool)

    p = pred[mask].astype(np.float64)
    t = target[mask].astype(np.float64)
    err = p - t

    if p.size == 0:
        # 空 mask 防除零
        return {
            "mae_m": 0.0, "rmse_m": 0.0, "r2": 0.0,
            "bias_m": 0.0, "n_pixels": 0,
        }

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # R²
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    bias = float(np.mean(err))

    result = {
        "mae_m": mae,
        "rmse_m": rmse,
        "r2": float(r2),
        "bias_m": bias,
        "n_pixels": int(mask.sum()),
    }

    # 分带误差（赤道/副热带/中纬/极地）
    if lats is not None:
        bands = [
            ("tropics", 0.0, 23.5),
            ("subtropics", 23.5, 35.0),
            ("midlat", 35.0, 60.0),
            ("polar", 60.0, 90.0),
        ]
        abs_lat = np.abs(lats)[:, None]
        for name, lo, hi in bands:
            band_mask = mask & (abs_lat >= lo) & (abs_lat < hi)
            if band_mask.any():
                band_err = pred[band_mask] - target[band_mask]
                result[f"mae_{name}"] = float(np.mean(np.abs(band_err)))
    return result


# ==================== 3. 海岸线三分类指标 ====================

def coast_metrics(pred: np.ndarray, target: np.ndarray,
                  n_classes: int = 3) -> dict:
    """海岸线三分类评估指标。

    Parameters
    ----------
    pred : (H, W) int  预测类别 ∈ {0, 1, 2}
    target : (H, W) int  标签类别 ∈ {0, 1, 2}
    n_classes : int  类别数，默认 3

    Returns
    -------
    dict  含 oa/macc/miou + 各类 iou + 混淆矩阵
    """
    pred = np.asarray(pred).astype(np.int64)
    target = np.asarray(target).astype(np.int64)
    valid = (target >= 0) & (target < n_classes)

    # 混淆矩阵
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = np.logical_and(pred[valid] == i, target[valid] == j).sum()
    # cm[i, j] = 预测 i / 标签 j 的像素数

    # 总体精度
    oa = float(np.trace(cm) / max(cm.sum(), 1))

    # 各类精度与 IoU
    ious = []
    accs = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[c, :].sum() - tp
        fn = cm[:, c].sum() - tp
        iou = tp / max(tp + fp + fn, 1)
        acc = tp / max(cm[:, c].sum(), 1)
        ious.append(float(iou))
        accs.append(float(acc))

    mIoU = float(np.mean(ious))
    mAcc = float(np.mean(accs))

    # 边缘锐化度：海岸线（class 0/1 边界）的过渡带宽度
    edge_sharpness = _edge_sharpness(pred, target)

    result = {
        "oa": oa,
        "macc": mAcc,
        "miou": mIoU,
        "confusion_matrix": cm.tolist(),
        "edge_sharpness": edge_sharpness,
    }
    # 三类时用语义化键名，其他类别数用通用键名
    if n_classes == 3:
        result.update({
            "iou_land": ious[0], "iou_shelf": ious[1], "iou_deep": ious[2],
            "acc_land": accs[0], "acc_shelf": accs[1], "acc_deep": accs[2],
        })
    else:
        for c in range(n_classes):
            result[f"iou_{c}"] = ious[c]
            result[f"acc_{c}"] = accs[c]
    return result


def _edge_sharpness(pred: np.ndarray, target: np.ndarray) -> float:
    """边缘锐化度：预测海岸线过渡带宽度 vs 标签过渡带宽度。

    计算陆地（class 0）到非陆地的距离变换，在海岸线附近（±5 像素）
    统计过渡带宽度。值越接近 1.0 表示边缘越锐利（与标签一致）。
    """
    # 标签的海岸线位置（class 0 与非 0 的边界）
    land_t = (target == 0)
    if not land_t.any() or land_t.all():
        return 1.0
    edge_t = np.logical_xor(land_t, ndi.binary_erosion(land_t))
    # 预测的海岸线
    land_p = (pred == 0)
    edge_p = np.logical_xor(land_p, ndi.binary_erosion(land_p))

    # 两条海岸线的 Hausdorff 距离（简化为平均距离）
    if not edge_p.any():
        return 0.0
    dist_t = ndi.distance_transform_edt(~edge_t)
    dist_p = ndi.distance_transform_edt(~edge_p)
    # 预测海岸线到标签海岸线的平均距离
    avg_dist = float(dist_t[edge_p].mean())
    # 归一化：距离越小越锐利（5 像素以内为好）
    return float(max(0.0, 1.0 - avg_dist / 5.0))


# ==================== 4. 批量评估器（在整个验证集上累积） ====================

class MetricAccumulator:
    """在整个验证集上累积指标。

    用法
    ----
    >>> acc = MetricAccumulator("river")
    >>> for batch in val_loader:
    ...     acc.update(pred_batch, target_batch)
    >>> metrics = acc.compute()
    """

    def __init__(self, task: str):
        """初始化累积器，记录任务名并清空缓存。"""
        self.task = task.lower()
        self.reset()

    def reset(self) -> None:
        """清空已累积的预测、标签与附加数据。"""
        self._preds: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._extras: dict = {}

    def update(self, pred: np.ndarray, target: np.ndarray,
               **extras) -> None:
        """累积一个 batch 的预测与标签。

        extras 可包含 lats / land_mask（雪线任务用）。
        """
        self._preds.append(np.asarray(pred))
        self._targets.append(np.asarray(target))
        for k, v in extras.items():
            self._extras.setdefault(k, []).append(np.asarray(v))

    def compute(self) -> dict:
        """计算整个验证集的累积指标。"""
        if not self._preds:
            return {}

        # 拼接所有 batch
        if self.task == "river":
            # 二值分割：逐 batch 累积 TP/FP/FN/TN
            tp = fp = fn = tn = 0
            conn_scores = []
            topo_scores = []
            for p, t in zip(self._preds, self._targets):
                m = river_metrics(p, t)
                tp += m["tp"]; fp += m["fp"]
                fn += m["fn"]; tn += m["tn"]
                conn_scores.append(m["connectivity"])
                topo_scores.append(m["topo_acc"])
            iou = tp / max(tp + fp + fn, 1)
            dice = 2 * tp / max(2 * tp + fp + fn, 1)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            return {
                "iou": float(iou),
                "dice": float(dice),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / max(precision + recall, 1e-8)),
                "connectivity": float(np.mean(conn_scores)),
                "topo_acc": float(np.mean(topo_scores)),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

        if self.task == "snowline":
            # 回归：沿 batch 维拼接，保持 2D 形状以匹配 lats
            preds = np.concatenate([p.reshape(-1, p.shape[-1])
                                    if p.ndim > 2 else p
                                    for p in self._preds], axis=0)
            tgts = np.concatenate([t.reshape(-1, t.shape[-1])
                                   if t.ndim > 2 else t
                                   for t in self._targets], axis=0)
            lats = None
            if "lats" in self._extras:
                # 所有 batch 的 lats 应一致，取第一个即可
                lats = self._extras["lats"][0]
                # 若拼接后的行数与 lats 不匹配（多 batch），广播 lats
                if preds.shape[0] != lats.shape[0]:
                    # 按重复 batch 数扩展 lats
                    n_batch_rows = preds.shape[0]
                    n_lat = lats.shape[0]
                    if n_batch_rows % n_lat == 0:
                        lats = np.tile(lats, n_batch_rows // n_lat)
            return snowline_metrics(preds, tgts, lats=lats)

        if self.task == "coast":
            # 分类：累积混淆矩阵
            cm = np.zeros((3, 3), dtype=np.int64)
            edge_scores = []
            for p, t in zip(self._preds, self._targets):
                m = coast_metrics(p, t)
                cm += np.array(m["confusion_matrix"], dtype=np.int64)
                edge_scores.append(m["edge_sharpness"])
            # 从累积混淆矩阵重算
            oa = float(np.trace(cm) / max(cm.sum(), 1))
            ious = []
            accs = []
            for c in range(3):
                tp = cm[c, c]
                fp = cm[c, :].sum() - tp
                fn = cm[:, c].sum() - tp
                ious.append(float(tp / max(tp + fp + fn, 1)))
                accs.append(float(tp / max(cm[:, c].sum(), 1)))
            return {
                "oa": oa,
                "macc": float(np.mean(accs)),
                "miou": float(np.mean(ious)),
                "iou_land": ious[0],
                "iou_shelf": ious[1],
                "iou_deep": ious[2],
                "edge_sharpness": float(np.mean(edge_scores)),
            }

        raise ValueError(f"未知任务: {self.task}")


# ==================== 5. 指标格式化打印 ====================

def format_metrics(task: str, metrics: dict) -> str:
    """格式化指标为单行字符串，便于训练日志打印。"""
    if task == "river":
        return (f"iou={metrics.get('iou', 0):.3f} "
                f"dice={metrics.get('dice', 0):.3f} "
                f"f1={metrics.get('f1', 0):.3f} "
                f"conn={metrics.get('connectivity', 0):.3f} "
                f"topo={metrics.get('topo_acc', 0):.3f}")
    if task == "snowline":
        s = (f"mae={metrics.get('mae_m', 0):.0f}m "
             f"rmse={metrics.get('rmse_m', 0):.0f}m "
             f"r2={metrics.get('r2', 0):.3f} "
             f"bias={metrics.get('bias_m', 0):+.0f}m")
        for band in ["tropics", "subtropics", "midlat", "polar"]:
            key = f"mae_{band}"
            if key in metrics:
                s += f" {band[:3]}={metrics[key]:.0f}m"
        return s
    if task == "coast":
        return (f"oa={metrics.get('oa', 0):.3f} "
                f"miou={metrics.get('miou', 0):.3f} "
                f"land={metrics.get('iou_land', 0):.3f} "
                f"shelf={metrics.get('iou_shelf', 0):.3f} "
                f"deep={metrics.get('iou_deep', 0):.3f} "
                f"edge={metrics.get('edge_sharpness', 0):.3f}")
    return str(metrics)
