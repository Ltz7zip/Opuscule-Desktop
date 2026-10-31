# -*- coding: utf-8 -*-
"""
visualize_context.py — 基于 WorldGeoContext 的地形可视化工具

=============================================================
功能（4 面板大图，默认输出到 terrain/output/geoctx_visual_seed{seed}.png）：
  7 项 KPI 卡片 + 高程/坡度/地理带三图层切换
  + 悬停查询 / 探针点联动 / 多边形统计 / 剖面 Canvas
  [0,0] 全球高程渲染（terrain 色卡）+ 探针点标注 + 可选多边形叠加
  [0,1] 坡度角度（度）渲染（陆地可见，海洋填深蓝色遮挡）
  [1,0] 地貌分区图（深海 / 浅海 / 平原 / 丘陵 / 山地 / 高山雪线）
  [1,1] 指定经纬度线段 A→B 的高程剖面（带坡度填充）

CLI 使用示例：
  # 1) 最简：seed1 的 2048 默认图
  python -m terrain.core.visualize_context

  # 2) 指定 seed/size + 自定义探针点 + 自定义剖面线段
  python -m terrain.core.visualize_context --seed 7 --size 2048 \
      --points 30,85,10,5,-20,-140,0,179.9 \
      --profile 50,-5,30,35 \
      --poly 45,5,55,5,55,20,45,20 \
      --output terrain/output/my_world.png

  # 3) 内嵌调用（Jupyter/IPython）：返回 Figure 对象自行保存
  from terrain.core.visualize_context import build_geo_figure
  fig, axes = build_geo_figure(seed=1, size=2048)
  fig.show()
=============================================================
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from typing import List, Tuple, Optional, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无显示环境安全；交互式时调用方自行切回
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon as MplPolygon

# 交互 HTML 报告需要的 I/O / 编码 / JSON
import io
import json
import html
import base64
import datetime as _dt
from scipy.ndimage import zoom as _ndi_zoom

# --- 同目录 geo_context 与 terrain_engine 导入（支持跨目录运行） ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from .geo_context import WorldGeoContext  # noqa: E402


# ============================================================================
# 色卡 & 分区（与 generate_world.py 渲染风格对齐，但不依赖 torch/训练代码）
# ============================================================================

def make_terrain_cmap():
    """复刻 terrain_engine.make_terrain_cmap（避免引入 torch）。"""
    colors = [
        (0.04, 0.12, 0.30),   # -10500m  深海沟
        (0.07, 0.22, 0.45),   # -6000m   深海平原
        (0.12, 0.36, 0.60),   # -3000m   洋中脊/深洋
        (0.25, 0.55, 0.70),   # -1000m   陆坡
        (0.55, 0.80, 0.80),   #  -200m   陆架/浅海
        (0.35, 0.55, 0.28),   #     0m   海滩/低地平原
        (0.55, 0.62, 0.30),   #   300m   草原/森林
        (0.68, 0.52, 0.30),   #   800m   丘陵/温带草原
        (0.55, 0.38, 0.28),   #  2000m   山地基带
        (0.80, 0.76, 0.72),   #  4000m   高海拔裸岩
        (1.00, 1.00, 1.00),   #  6000m+  雪线
    ]
    return LinearSegmentedColormap.from_list("dem", colors, N=512)


def make_slope_cmap():
    """坡度专用色卡（绿→黄→橙→红）。"""
    stops = [
        (0.00, (0.78, 0.90, 0.65)),  # 0%   极缓
        (0.10, (0.45, 0.75, 0.38)),  # 5%   缓坡
        (0.25, (0.95, 0.85, 0.30)),  # 20%  中等
        (0.50, (0.92, 0.52, 0.18)),  # 40%  陡坡
        (0.75, (0.80, 0.18, 0.18)),  # 70%  极陡
        (1.00, (0.30, 0.05, 0.10)),  # 100%+近乎垂直
    ]
    return LinearSegmentedColormap.from_list("slope", stops, N=256)


# 地貌分区边界（米） 与 颜色
# 注：±inf 被替换为 -11000/+9000（generate_world 已钳制 ±10500/8900），
#     这样 colorbar 的 BoundaryNorm ylim 不会报 ±inf。
_GEO_CLAMP_MIN, _GEO_CLAMP_MAX = -11000.0, 9000.0
_GEO_BOUNDS = [_GEO_CLAMP_MIN, -200, 0, 300, 800, 2000, 4000, _GEO_CLAMP_MAX]
_GEO_LABELS = ["深海", "浅海/陆架", "平原", "丘陵", "山地带", "高山", "雪线/极峰"]
_GEO_COLORS = [
    "#183860",  # 深海
    "#7EC8C8",  # 浅海
    "#8FB04D",  # 平原
    "#C8A25A",  # 丘陵
    "#8C6048",  # 山地
    "#CCC6C0",  # 高山
    "#FFFFFF",  # 雪线
]


def make_geozone_norm_and_cmap():
    """构建地理分区色卡与边界归一化器，返回 (cmap, norm)。"""
    cmap = ListedColormap(_GEO_COLORS)
    norm = BoundaryNorm(_GEO_BOUNDS, cmap.N)
    return cmap, norm


# ============================================================================
# 子函数：采样剖面 / 探针点 / 多边形叠加
# ============================================================================

def _sample_profile(ctx: WorldGeoContext,
                    lat0: float, lon0: float,
                    lat1: float, lon1: float,
                    n_pts: int = 800) -> dict:
    """沿 A→B 线段（等距圆柱上的直线段）等距采样 n_pts 个点。

    返回 dict：lats, lons, cum_dist_km, elevs, slopes
    """
    t = np.linspace(0, 1, n_pts)
    lats = lat0 + (lat1 - lat0) * t
    lons = lon0 + (lon1 - lon0) * t

    # 距离：用简化大圆（Haversine）累积
    dlat = np.diff(lats, prepend=lats[0])
    dlon = np.diff(lons, prepend=lons[0])
    mlon = np.deg2rad((lats + np.roll(lats, 1)) / 2)
    d_km = 111.32 * np.sqrt(dlat ** 2 + (dlon * np.cos(mlon)) ** 2)
    cum_km = np.cumsum(d_km) - d_km[0]  # 起点 0

    elevs = ctx.get_elevation(lats, lons)
    slopes = ctx.get_slope(lats, lons)
    return dict(lats=lats, lons=lons, cum_km=cum_km, elevs=elevs, slopes=slopes)


def _parse_points(s: Optional[str]) -> List[Tuple[float, float, str]]:
    """解析 CLI --points: lat0,lon0[,name0],lat1,lon1[,name1],... 为 (lat,lon,name) 列表。
    如果名字缺省就自动编号 P1/P2/..."""
    if not s:
        return [(30.0, 85.0, "P1"), (0.0, 5.0, "P2"),
                (-20.0, -140.0, "P3"), (0.0, 179.9, "P4")]
    toks = [x.strip() for x in s.split(",") if x.strip()]
    pts: List[Tuple[float, float, str]] = []
    i = 0
    pidx = 1
    while i < len(toks):
        if i + 1 >= len(toks):
            raise ValueError("--points 需要成对出现 lat,lon")
        lat, lon = float(toks[i]), float(toks[i + 1])
        name = None
        if i + 2 < len(toks):
            # 如果下一个 token 不像数字，当作名字
            try:
                float(toks[i + 2])
                # 是数字 → 不是名字
                i += 2
            except ValueError:
                name = toks[i + 2]
                i += 3
        else:
            i += 2
        if name is None:
            name = f"P{pidx}"
            pidx += 1
        pts.append((lat, lon, name))
    return pts


def _parse_poly(s: Optional[str]) -> Optional[List[Tuple[float, float]]]:
    """解析 --poly: lat0,lon0,lat1,lon1,... 为 [(lat, lon), ...]。"""
    if not s:
        return None
    toks = [x.strip() for x in s.split(",") if x.strip()]
    if len(toks) < 6 or len(toks) % 2 != 0:
        raise ValueError("--poly 至少需要 3 个 (lat,lon) 对")
    vals = [float(x) for x in toks]
    return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]


def _parse_profile(s: Optional[str]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """解析 --profile: lat0,lon0,lat1,lon1。缺省给一个横贯欧亚的剖面。"""
    if not s:
        return (55.0, 10.0), (25.0, 125.0)  # 北海 → 杭州湾（大致）
    toks = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(toks) != 4:
        raise ValueError("--profile 需要 4 个数：lat0,lon0,lat1,lon1")
    return (toks[0], toks[1]), (toks[2], toks[3])


# ============================================================================
# 主绘图函数
# ============================================================================

def build_geo_figure(
    seed: int = 1,
    size: int = 4096,
    probe_points: Sequence[Tuple[float, float, str]] = (),
    profile: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    polygon: Optional[Sequence[Tuple[float, float]]] = None,
    downsample_for_display: int = 1,
    figsize: Tuple[float, float] = (20, 14),
    dpi: int = 120,
) -> Tuple[plt.Figure, np.ndarray, WorldGeoContext, dict]:
    """构建 2×2 四面板地形可视化大图（公开 API，matplotlib Figure 级别）。

    四个面板布局：
        [0,0] 全球高程渲染（terrain 色卡）+ 探针点标注 + 多边形叠加 + 剖面线段 A/B
        [0,1] 坡度百分比渲染（海洋用深蓝遮罩，陆地使用 0~100% 的绿→黄→橙→红色卡）
        [1,0] 地貌分区图（深海 / 浅海 / 平原 / 丘陵 / 山地带 / 高山 / 雪线 7 类离散色）
        [1,1] 线段 A→B 的高程剖面（左 Y 轴高程、右 Y 轴坡度，含陆地/海洋填充与雪线/山脉基准线）

    Parameters
    ----------
    seed : int
        世界种子。传入 ``WorldGeoContext.from_seed``；决定加载哪份 DEM。
    size : {2048, 4096}
        DEM 单边像素数。若文件实际尺寸与请求不符会由单例自动打印警告并回退。
    probe_points : sequence of (lat, lon, name) tuples
        探针点三元组。空时退化为 ``_parse_points(None)`` 的 4 个默认点。
        叠加在 [0,0] 与 [0,1] 面板，同时写入返回值 ``info["probe_points"]``。
    profile : ((latA, lonA), (latB, lonB)) or None
        剖面线段端点。空时退化为横贯欧亚的默认线段（北海→杭州湾）。
    polygon : sequence of (lat, lon) tuples or None
        多边形顶点（自动闭合、支持凹多边形）。若给定，会在 [0,0] 面板叠加黄色半透明
        填充，并在左下角以文本框形式展示 ``get_region_stats`` 8 项统计。
    downsample_for_display : int
        绘图降采样步长。4096² DEM 传入 ``2`` 可在不丢失宏观结构的前提下显著加速；
        统计与探针点/剖面插值始终使用原始分辨率 DEM，不受此参数影响。
    figsize : (float, float)
        Figure 尺寸（英寸），默认 20×14。
    dpi : int
        Figure / 保存时的 DPI，默认 120（生成大图约 2400×1680 px）。

    Returns
    -------
    fig  : matplotlib.figure.Figure
        构建好的 Figure 对象。调用方自行 ``fig.savefig(...)`` 或 ``fig.show()``。
    axes : np.ndarray of Axes, shape (2, 2)
        四个子图的 Axes 数组（``axes[0,0]`` 为全球高程面板等）。
    ctx  : WorldGeoContext
        已加载的单例；调用方可以复用进行更多查询（避免重复加载）。
    info : dict
        辅助性汇总字典，包含以下键，便于后续生成 HTML / 导出 JSON：
            - ``probe_points``: list[dict]，每个探针点的 name/lat/lon/elev/slope/land
            - ``profile``: dict，剖面的 lats/lons/cum_km/elevs/slopes 数组
            - ``polygon_stats``: dict 或 None，多边形区域 8 项统计
            - ``S``: int，DEM 单边尺寸
            - ``land_ratio`` / ``mountain_ratio``: float，陆地/山地占比（0~1）
            - ``dem_min`` / ``dem_max``: float，DEM 极值（米）
    """
    t0 = time.time()
    ctx = WorldGeoContext.from_seed(seed=seed, size=size)
    S = ctx.size
    dem = ctx.dem
    lats = ctx.lat_grid
    lons = ctx.lon_grid

    # 对 4096 大网格默认降采样到 2048 级别显示，加速绘图
    if downsample_for_display and downsample_for_display > 1:
        step = downsample_for_display
        dem_d = dem[::step, ::step]
    else:
        dem_d = dem

    # --- 计算坡度场（用于渲染面板 2） ---
    from .geo_context import _compute_slope_field
    slope_field = _compute_slope_field(dem, lats, lons)
    slope_d = slope_field if downsample_for_display == 1 else slope_field[::step, ::step]
    land_d = dem_d >= 1.0

    # --- 图像（以 matplotlib imshow 的 origin 习惯：默认 upper → 我们要
    #     lats[-1] 在上方，lats[0] 在下方。因为 dem 第 0 行是南极附近。
    #     因此必须使用 origin="lower" 搭配 extent=(lons[0],lons[-1],lats[0],lats[-1])）。
    #     实际上更直接的做法：把 dem 上下翻转（np.flipud）后用 origin="upper" + 正常 extent，
    #     这样刻度更贴近大家习惯的"世界地图：左=-180 右=180 上=+90 下=-90"。
    flip = lambda a: np.flipud(a)
    dem_img = flip(dem_d)
    slope_img = flip(slope_d)
    land_img = flip(land_d)
    # 翻转后纬度范围仍然对；因为 extent 给出映射：imshow 像素 y=0（顶）→ extent 上值；像素 y=H-1（底）→ extent 下值
    # 所以我们翻转数组 + origin='upper' + extent=(左,右,-90,+90) → 顶=+90 底=-90 ✅
    # 但注意：翻转后 imshow 的 y 像素 0 对应"原最后一行（接近+90）"，extent 上界=+90
    #          imshow 的 y 像素 H-1 对应"原第 0 行（接近-90）"，extent 下界=-90
    # 这样完全正确。
    map_extent = (-180.0, 180.0, -90.0, 90.0)

    # --- 计算地貌分区 ---
    cmap_geo, norm_geo = make_geozone_norm_and_cmap()
    # BoundaryNorm + ListedColormap 需要连续值；钳制到有限范围避免 colorbar ylim=±inf
    dem_clamped = np.clip(dem_img, _GEO_CLAMP_MIN, _GEO_CLAMP_MAX)

    # ============ 画 2x2 图 ============
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    fig.suptitle(
        f"WorldGeoContext 可视化  |  Seed={seed}  Size={S}×{S}\n"
        f"陆地 {ctx.land_mask.mean() * 100:.2f}%  |  "
        f"山脉(≥1500m) {ctx.mountain_mask.mean() * 100:.2f}%  |  "
        f"峰 {dem.max():.0f}m  沟 {dem.min():.0f}m  |  "
        f"加载耗时 {time.time() - t0:.2f}s",
        fontsize=15, fontweight="bold", y=1.01,
    )
    cmap_dem = make_terrain_cmap()
    cmap_slope = make_slope_cmap()

    # --- [0,0] 全球高程渲染 ---
    ax00 = axes[0, 0]
    # 钳到 -10500~8900（与 generate_world 一致）
    im00 = ax00.imshow(dem_img, cmap=cmap_dem, vmin=-10500, vmax=8900,
                       extent=map_extent, origin="upper", aspect="auto")
    ax00.set_title("① 全球高程渲染（terrain 色卡）", fontsize=13, fontweight="bold")
    ax00.set_xlabel("经度 (°)", fontsize=10)
    ax00.set_ylabel("纬度 (°)", fontsize=10)
    ax00.set_xticks(np.arange(-180, 181, 30))
    ax00.set_yticks(np.arange(-90, 91, 30))
    ax00.grid(alpha=0.25, linestyle="--")
    plt.colorbar(im00, ax=ax00, fraction=0.04, pad=0.02, label="高程 (m)")

    # 叠加探针点
    if not probe_points:
        probe_points = _parse_points(None)
    probe_info = []
    for lat, lon, name in probe_points:
        e = ctx.get_elevation(lat, lon)
        s = ctx.get_slope(lat, lon)
        L = "LAND" if ctx.is_land(lat, lon) else "SEA"
        # 颜色：LAND=红，SEA=青
        color = "#E11D48" if L == "LAND" else "#0EA5E9"
        ax00.plot(lon, lat, marker="o", ms=8, mew=1.5, mec="black",
                  mfc=color, zorder=5)
        ax00.annotate(f"{name}\n{e:+.0f}m {L} {s:.0f}%",
                      xy=(lon, lat), xytext=(7, 7), textcoords="offset points",
                      fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                ec=color, alpha=0.92, lw=1.2))
        probe_info.append(dict(name=name, lat=lat, lon=lon,
                               elev=float(e), slope=float(s), land=L == "LAND"))

    # 叠加多边形（如果给了）
    poly_stats = None
    if polygon is not None:
        poly_lons = [p[1] for p in polygon] + [polygon[0][1]]
        poly_lats = [p[0] for p in polygon] + [polygon[0][0]]
        xy = np.column_stack([poly_lons, poly_lats])
        patch = MplPolygon(xy, closed=True, facecolor="#FACC15",
                           edgecolor="#B45309", linewidth=2.0,
                           alpha=0.28, zorder=4, linestyle="-")
        ax00.add_patch(patch)
        # 显示 get_region_stats
        st = ctx.get_region_stats(polygon)
        poly_stats = st
        text = (f"多边形区域统计:\n"
                f"  像素数     = {st['pixel_count']}\n"
                f"  平均海拔   = {st['mean_elevation']:+.0f} m\n"
                f"  最高峰     = {st['peak']:+.0f} m\n"
                f"  最低点     = {st['trough']:+.0f} m\n"
                f"  陆地占比   = {st['land_ratio'] * 100:.1f}%\n"
                f"  山脉占比   = {st['mountain_ratio'] * 100:.1f}%")
        ax00.text(0.02, 0.02, text, transform=ax00.transAxes,
                  fontsize=9.5,
                  verticalalignment="bottom",
                  bbox=dict(boxstyle="round,pad=0.4", fc="#0B1220",
                            ec="#FACC15", alpha=0.86, color="#FDE68A", lw=1.5))

    # --- [0,1] 坡度百分比渲染（海洋遮罩） ---
    ax01 = axes[0, 1]
    # 坡度超过 100% 按 100% 渲染
    slope_show = np.clip(slope_img, 0, 100)
    slope_masked = np.ma.masked_where(~land_img, slope_show)
    im01 = ax01.imshow(slope_masked, cmap=cmap_slope, vmin=0, vmax=100,
                       extent=map_extent, origin="upper", aspect="auto")
    # 海洋用深蓝色衬底
    ax01.set_facecolor("#0A1930")
    ax01.set_title("② 坡度百分比（陆地显示，海洋深蓝遮罩）",
                   fontsize=13, fontweight="bold")
    ax01.set_xlabel("经度 (°)", fontsize=10)
    ax01.set_yticks(np.arange(-90, 91, 30))
    ax01.set_xticks(np.arange(-180, 181, 30))
    ax01.grid(alpha=0.25, linestyle="--")
    cb = plt.colorbar(im01, ax=ax01, fraction=0.04, pad=0.02, label="坡度 (%)")
    cb.set_ticks([0, 10, 20, 40, 60, 80, 100])

    # 叠加探针点
    for lat, lon, name, *_ in [(p[0], p[1], p[2]) for p in probe_points]:
        ax01.plot(lon, lat, marker="o", ms=6, mec="white", mfc="#F59E0B", zorder=5)

    # --- [1,0] 地貌分区 ---
    ax10 = axes[1, 0]
    # ListedColormap + BoundaryNorm 要求数据为 bin 索引 0..N-1（np.digitize 输出正是）
    im10 = ax10.imshow(dem_clamped, cmap=cmap_geo, norm=norm_geo,
                       extent=map_extent, origin="upper", aspect="auto",
                       interpolation="nearest")
    ax10.set_title("③ 地貌分区（深海/浅海/平原/丘陵/山地/高山/雪线）",
                   fontsize=13, fontweight="bold")
    ax10.set_xlabel("经度 (°)", fontsize=10)
    ax10.set_ylabel("纬度 (°)", fontsize=10)
    ax10.set_xticks(np.arange(-180, 181, 30))
    ax10.set_yticks(np.arange(-90, 91, 30))
    ax10.grid(alpha=0.25, linestyle="--")

    # 自定义色条（类别标签）
    cbar = plt.colorbar(im10, ax=ax10, fraction=0.04, pad=0.02, spacing="proportional")
    # 把边界中点作为刻度位置
    mid = 0.5 * (np.asarray(_GEO_BOUNDS[:-1]) + np.asarray(_GEO_BOUNDS[1:]))
    # 替换 ±inf 为有限值以便显示
    mid[0] = (-10500 + (-200)) / 2
    mid[-1] = (4000 + 8900) / 2
    cbar.set_ticks(mid)
    cbar.set_ticklabels([f"{lab}" for lab in _GEO_LABELS])
    cbar.ax.tick_params(labelsize=9)

    # --- [1,1] 剖面线 ---
    ax11 = axes[1, 1]
    if profile is None:
        profile = _parse_profile(None)
    (latA, lonA), (latB, lonB) = profile
    prof = _sample_profile(ctx, latA, lonA, latB, lonB, n_pts=800)

    # 在 [0,0] 图上叠加剖面线段
    ax00.plot([lonA, lonB], [latA, latB], color="#FBBF24", linestyle="--",
              lw=2.2, zorder=6)
    ax00.plot([lonA, lonB], [latA, latB], "o", ms=7, color="#FBBF24",
              mec="black", zorder=7)
    ax00.annotate("A", xy=(lonA, latA), xytext=(-14, 10),
                  textcoords="offset points", fontweight="bold", color="#FBBF24")
    ax00.annotate("B", xy=(lonB, latB), xytext=(6, 10),
                  textcoords="offset points", fontweight="bold", color="#FBBF24")

    # 在 [0,1] 上也叠加剖面
    ax01.plot([lonA, lonB], [latA, latB], color="#FBBF24", linestyle="--", lw=2, zorder=6)

    # 双轴：左高程、右坡度
    ax11_r = ax11.twinx()
    xkm = prof["cum_km"]
    elev = prof["elevs"]
    slp = prof["slopes"]

    # 填充：陆地绿色，海洋蓝色
    ax11.fill_between(xkm, elev, 0, where=(elev >= 0), interpolate=False,
                      color="#86A852", alpha=0.70, label="陆地")
    ax11.fill_between(xkm, elev, 0, where=(elev < 0), interpolate=False,
                      color="#4A8BBF", alpha=0.75, label="海洋")
    # 基准零线
    ax11.axhline(0, color="black", lw=1.0, ls="-", alpha=0.8)
    # 雪线、山脉基准线
    ax11.axhline(1500, color="#888888", lw=0.9, ls=":", alpha=0.7, label="山脉阈值 1500m")
    ax11.axhline(4000, color="#B0B0B0", lw=0.9, ls=":", alpha=0.6, label="雪线 ~4000m")

    # 高程线
    ax11.plot(xkm, elev, color="#1F2937", lw=1.2, zorder=5, label="高程")
    ax11.set_xlabel(f"A({latA:.1f}°,{lonA:.1f}°) → B({latB:.1f}°,{lonB:.1f}°)  "
                    f"沿线距离 (km)  总长 ≈ {xkm[-1]:.0f} km", fontsize=10)
    ax11.set_ylabel("高程 (m)", fontsize=11, color="#1F2937")
    ax11.tick_params(axis="y", labelcolor="#1F2937")
    ax11.grid(alpha=0.35, axis="both", linestyle=":")

    # 坡度（右轴）
    ax11_r.plot(xkm, slp, color="#DC2626", lw=1.1, alpha=0.85, label="坡度 %")
    ax11_r.fill_between(xkm, slp, 0, alpha=0.12, color="#DC2626")
    ax11_r.set_ylabel("坡度 (%)", fontsize=11, color="#DC2626")
    ax11_r.tick_params(axis="y", labelcolor="#DC2626")
    ax11_r.set_ylim(0, max(15.0, float(np.nanpercentile(slp, 99)) * 1.2))

    ax11.set_title("④ 线段 A→B 的高程剖面与沿线坡度",
                   fontsize=13, fontweight="bold")

    # 合并双轴图例（不重复）
    h1, l1 = ax11.get_legend_handles_labels()
    h2, l2 = ax11_r.get_legend_handles_labels()
    ax11.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5,
                framealpha=0.9, ncol=2)

    info = dict(probe_points=probe_info, profile=prof,
                polygon_stats=poly_stats, S=S,
                land_ratio=float(ctx.land_mask.mean()),
                mountain_ratio=float(ctx.mountain_mask.mean()),
                dem_min=float(dem.min()), dem_max=float(dem.max()))
    return fig, axes, ctx, info


# ============================================================================
# 交互 HTML 报告导出（单文件，自包含 base64 图层 + 查询网格 + Canvas 剖面）
# ============================================================================

def _resize_2d(arr: np.ndarray, target_s: int) -> np.ndarray:
    """对 2D 数组各向同性缩放到 target_s × target_s（float64 中间精度，再还原 dtype）。"""
    if arr.shape[0] == target_s and arr.shape[1] == target_s:
        return arr
    factor = target_s / arr.shape[0]
    out = _ndi_zoom(arr.astype(np.float64), factor, order=1, mode="nearest")
    return out.astype(arr.dtype)


def _ndarray_to_png_b64(data: np.ndarray, cmap, vmin=None, vmax=None,
                       norm=None, mask_ocean: Optional[np.ndarray] = None,
                       ocean_rgb=(0x0A, 0x19, 0x30), target_s=1024) -> str:
    """2D 数据经 cmap/norm 着色 → (target_s,target_s,3) RGB → PNG bytes → base64。"""
    data_s = _resize_2d(data.astype(np.float32), target_s)
    if norm is not None:
        rgba = cmap(norm(data_s))
    else:
        v0 = float(np.nanmin(data_s)) if vmin is None else float(vmin)
        v1 = float(np.nanmax(data_s)) if vmax is None else float(vmax)
        if v1 == v0: v1 = v0 + 1e-6
        t = np.clip((data_s - v0) / (v1 - v0), 0.0, 1.0)
        rgba = cmap(t)
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    if mask_ocean is not None:
        mo = _resize_2d(mask_ocean.astype(np.uint8), target_s).astype(bool)
        rgb[~mo] = np.asarray(ocean_rgb, dtype=np.uint8)
    buf = io.BytesIO()
    import matplotlib.image as _mpimg
    _mpimg.imsave(buf, rgb, format="png", origin="upper")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _float32_grid_b64(arr: np.ndarray, target_s: int = 512) -> Tuple[str, int]:
    """2D → 缩放到 target_s² → float32 bytes → base64。返回 (base64, size)。"""
    data = _resize_2d(arr.astype(np.float32), target_s)
    return base64.b64encode(data.tobytes(order="C")).decode("ascii"), target_s


# ----------------------------------------------------------------------------
# HTML 模板（使用 Python %(KEY)s 字符串格式化；CSS+HTML+JS 单文件，完全离线可用）
# ----------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>%(TITLE)s</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{
  --bg:#0f172a; --panel:#111c34; --panel-2:#172445; --border:#27365e;
  --text:#e6edf7; --muted:#93a3c4; --brand:#a78bfa; --brand-2:#22d3ee;
  --ok:#34d399; --warn:#fbbf24; --err:#f87171;
  --land:#4ade80; --sea:#38bdf8;
  --radius:10px; --mono: ui-monospace,Menlo,Consolas,monospace;
  --sans: ui-sans-serif,-apple-system,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;
}
@media (prefers-color-scheme: light){:root{
  --bg:#f6f8ff; --panel:#ffffff; --panel-2:#eef2ff; --border:#d8deee;
  --text:#0f172a; --muted:#475569; --brand:#6d28d9; --brand-2:#0891b2;
}}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px}
.wrap{max-width:1400px;margin:0 auto;padding:20px 24px 48px}
h1{margin:4px 0 20px;font-size:22px;letter-spacing:.2px}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px;position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--bar,var(--brand))}
.kpi .l{font-size:12px;color:var(--muted);margin-bottom:4px}
.kpi .v{font-family:var(--mono);font-size:22px;font-weight:700}
.kpi .v small{font-size:12px;color:var(--muted);font-weight:500;margin-left:6px}
.layout{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
@media (max-width:960px){.layout{grid-template-columns:1fr}.kpi-row{grid-template-columns:repeat(2,1fr)}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:14px}
.card h2{margin:0 0 10px;font-size:15px;display:flex;align-items:center;gap:8px}
.card h2 .pill{font-size:11px;font-weight:500;padding:2px 8px;border-radius:999px;background:var(--panel-2);color:var(--muted)}
.map-wrap{position:relative;width:100%%;aspect-ratio:2/1;background:#000;border-radius:8px;overflow:hidden;border:1px solid var(--border);user-select:none}
.map-wrap .layer{position:absolute;inset:0;width:100%%;height:100%%;display:block;object-fit:cover;transition:opacity .15s ease;opacity:0;pointer-events:none}
.map-wrap .layer.on{opacity:1}
.map-wrap #tooltip{position:absolute;padding:6px 9px;border-radius:7px;background:rgba(15,23,42,.92);
  color:#fff;font-family:var(--mono);font-size:12px;line-height:1.5;border:1px solid var(--border);
  pointer-events:none;transform:translate(-50%%,calc(-100%% - 10px));white-space:nowrap;z-index:10;
  box-shadow:0 4px 18px rgba(0,0,0,.45);display:none}
.map-wrap #crosshair{position:absolute;width:14px;height:14px;pointer-events:none;display:none;z-index:9}
.map-wrap #crosshair::before,.map-wrap #crosshair::after{content:"";position:absolute;background:var(--warn);mix-blend-mode:difference;border-radius:1px}
.map-wrap #crosshair::before{left:50%%;top:0;bottom:0;width:1.5px;transform:translateX(-50%%)}
.map-wrap #crosshair::after{top:50%%;left:0;right:0;height:1.5px;transform:translateY(-50%%)}
.map-wrap .probe-marker{position:absolute;width:14px;height:14px;border-radius:50%%;transform:translate(-50%%,-50%%);
  border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,.6);pointer-events:none;z-index:8}
.map-wrap .probe-marker.land{background:#ef4444}
.map-wrap .probe-marker.sea{background:#0ea5e9}
.map-wrap .probe-label{position:absolute;transform:translate(-50%%,calc(-100%% - 10px));
  background:rgba(255,255,255,.95);color:#111;padding:1px 6px;border-radius:4px;font-size:11px;
  pointer-events:none;z-index:8;white-space:nowrap;font-weight:600}
.layer-tabs{display:inline-flex;background:var(--panel-2);border-radius:8px;padding:3px;margin-left:auto}
.layer-tabs button{all:unset;padding:6px 12px;border-radius:6px;font-size:12px;cursor:pointer;color:var(--muted)}
.layer-tabs button.on{background:var(--panel);color:var(--text);box-shadow:0 1px 0 rgba(0,0,0,.15) inset}
.heading{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.hud{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;font-family:var(--mono);font-size:12px;color:var(--muted)}
.hud .tag{background:var(--panel-2);border:1px solid var(--border);border-radius:6px;padding:4px 8px}
.hud .tag b{color:var(--text)}
table.probe{width:100%%;border-collapse:collapse;font-size:13px}
table.probe th,table.probe td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--border)}
table.probe th{color:var(--muted);font-weight:500;font-size:12px;background:var(--panel-2);position:sticky;top:0}
table.probe tbody tr:hover{background:var(--panel-2);cursor:pointer}
.ld{display:inline-block;width:8px;height:8px;border-radius:50%%;margin-right:6px;vertical-align:middle}
.ld.land{background:var(--land)}.ld.sea{background:var(--sea)}
.poly-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px 14px;font-family:var(--mono);font-size:13px}
.poly-grid div{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px dashed var(--border)}
.poly-grid div span:nth-child(1){color:var(--muted)}
.profile-wrap{position:relative;width:100%%;aspect-ratio:16/7;background:var(--panel-2);border-radius:8px;border:1px solid var(--border);overflow:hidden}
.profile-wrap canvas{display:block;width:100%%;height:100%%}
#pro-tip{position:absolute;top:6px;left:6px;padding:3px 8px;border-radius:6px;
  background:rgba(15,23,42,.85);color:#fff;font-family:var(--mono);font-size:11px;
  border:1px solid var(--border);pointer-events:none;display:none}
.footer{margin-top:22px;padding:12px 16px;background:var(--panel);border:1px solid var(--border);
  border-radius:var(--radius);color:var(--muted);font-family:var(--mono);font-size:12px;line-height:1.7}
.footer kbd{background:var(--panel-2);border:1px solid var(--border);padding:1px 6px;border-radius:4px;color:var(--text)}
.hidden{display:none !important}
</style>
</head>
<body>
<div class="wrap">
<h1>🌍 WorldGeoContext 交互地形报告 · <span style="color:var(--brand)">Seed %(SEED)s</span> <small style="color:var(--muted);font-weight:400;font-size:14px">(%(SIZE)s×%(SIZE)s, %(GEN_TIME)s)</small></h1>

<div class="kpi-row">
  <div class="kpi" style="--bar:var(--brand)"><div class="l">陆地占比</div><div class="v">%(LAND_RATIO).2f<small>%%</small></div></div>
  <div class="kpi" style="--bar:var(--brand-2)"><div class="l">山地占比</div><div class="v">%(MONT_RATIO).2f<small>%%</small></div></div>
  <div class="kpi" style="--bar:var(--ok)"><div class="l">最高峰</div><div class="v">%(PEAK).0f<small>m</small></div></div>
  <div class="kpi" style="--bar:var(--err)"><div class="l">最深沟</div><div class="v">%(TROUGH).0f<small>m</small></div></div>
  <div class="kpi" style="--bar:var(--ok-2,limegreen)"><div class="l">探针点数</div><div class="v">%(PROBE_COUNT)d<small>个</small></div></div>
  <div class="kpi" style="--bar:var(--purple,mediumpurple)"><div class="l">DEM 网格</div><div class="v">%(SIZE)d<small>×%(SIZE)d</small></div></div>
  <div class="kpi" style="--bar:var(--amber,goldenrod)"><div class="l">查询精度</div><div class="v">%(Q_SIZE)d<small>×%(Q_SIZE)d</small></div></div>
</div>

<div class="layout">
  <div class="card">
    <div class="heading">
      <h2>全球交互查询 <span class="pill">悬停查看地形 · 双击追加探针 · 切换图层</span></h2>
      <div class="layer-tabs" id="layerTabs">
        <button data-l="dem" class="on">高程</button>
        <button data-l="slope">坡度</button>
        <button data-l="geo">地理带</button>
      </div>
    </div>
    <div class="map-wrap" id="map">
      <img class="layer on"  id="layer-dem"   src="data:image/png;base64,%(LAYER_DEM)s"   alt="高程图层">
      <img class="layer"     id="layer-slope" src="data:image/png;base64,%(LAYER_SLOPE)s" alt="坡度图层">
      <img class="layer"     id="layer-geo"   src="data:image/png;base64,%(LAYER_GEO)s"   alt="地貌图层">
      <div id="tooltip"></div>
      <div id="crosshair"></div>
      %(PROBE_MARKERS_HTML)s
    </div>
    <div class="hud" id="hud">
      <span class="tag">纬度 <b id="h-lat">—</b></span>
      <span class="tag">经度 <b id="h-lon">—</b></span>
      <span class="tag">高程 <b id="h-elev">—</b></span>
      <span class="tag">坡度 <b id="h-slope">—</b></span>
      <span class="tag">陆海 <b id="h-land">—</b></span>
      <span class="tag">查询 <b style="color:var(--brand)">%(Q_SIZE)s²</b> 网格双线性</span>
    </div>
  </div>

  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="card">
      <h2>探针点清单 <span class="pill">点击行 → 地图定位 · <span style="color:var(--brand);margin-left:4px">%(PROBE_COUNT)d</span> 个</span></h2>
      <div style="max-height:290px;overflow:auto;border-radius:8px;border:1px solid var(--border)">
        <table class="probe" id="probe-table">
          <thead><tr><th style="width:22%%"></th><th>纬度</th><th>经度</th><th>海拔</th><th>坡度</th></tr></thead>
          <tbody>%(PROBE_TABLE_ROWS)s</tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h2>多边形区域统计 <span class="pill">%(POLY_PILL)s</span></h2>
      %(POLY_HTML_BODY)s
    </div>
  </div>
</div>

<div class="card" style="margin-top:16px">
  <h2>剖面 A→B <span class="pill">%(PROFILE_DESC)s · 总长 ≈ %(PROFILE_DIST).0f km · 鼠标悬停与地图联动</span></h2>
  <div class="profile-wrap" id="profile-wrap">
    <canvas id="profile-canvas" width="1600" height="700"></canvas>
    <div id="pro-tip"></div>
    <div id="pro-fallback" style="padding:24px;text-align:center;color:var(--muted)">正在渲染剖面…</div>
  </div>
</div>

<div class="footer">
  🛈 DEM 文件：<kbd>%(DEM_PATH)s</kbd><br>
  🛈 图层分辨率：<kbd>%(LAYER_SIZE)s²</kbd> · 查询分辨率：<kbd>%(Q_SIZE)s²</kbd>（高程/坡度，双线性插值）· 所有资源均内嵌，本报告离线 100%% 可用<br>
  🛈 生成命令：<kbd>%(CMD)s</kbd>
</div>
</div>

<script>
(function(){
'use strict';
const Q_SIZE = %(Q_SIZE)d;
const LAYER_SIZE = %(LAYER_SIZE)d;
const DEM_FULL = %(SIZE)d;
const LAT_MIN = -90 + 180/DEM_FULL/2;
const LAT_MAX =  90 - 180/DEM_FULL/2;
const LON_MIN = -180 + 360/DEM_FULL/2;
const LON_MAX =  180 - 360/DEM_FULL/2;

function b64ToBytes(b64){
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) out[i] = bin.charCodeAt(i);
  return out.buffer;
}
function decodeF32Grid(b64){ return new Float32Array(b64ToBytes(b64)); }
const DEM_Q = decodeF32Grid("%(Q_DEM_B64)s");
const SLP_Q = decodeF32Grid("%(Q_SLP_B64)s");

function query(lat, lon){
  lat = Math.max(LAT_MIN, Math.min(LAT_MAX, lat));
  lon = ((lon + 180) %% 360 + 360) %% 360 - 180;
  if(lon >= 180) lon -= 360;
  if(lon < -180) lon += 360;
  const rn = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN);
  const cn = (lon - LON_MIN) / (LON_MAX - LON_MIN);
  const r = Math.max(0, Math.min(Q_SIZE-1, rn * (Q_SIZE-1)));
  const c = Math.max(0, Math.min(Q_SIZE-1, cn * (Q_SIZE-1)));
  const r0 = Math.floor(r), c0 = Math.floor(c);
  const r1 = Math.min(Q_SIZE-1, r0+1), c1 = Math.min(Q_SIZE-1, c0+1);
  const fr = r - r0, fc = c - c0;
  function blin(g){
    const a=g[r0*Q_SIZE+c0],b=g[r0*Q_SIZE+c1],d=g[r1*Q_SIZE+c0],e=g[r1*Q_SIZE+c1];
    return (a*(1-fc)+b*fc)*(1-fr) + (d*(1-fc)+e*fc)*fr;
  }
  const e=blin(DEM_Q), s=Math.max(0,Math.min(1000,blin(SLP_Q)));
  return {elev:e, slope:s, land:e>=1};
}

const map = document.getElementById("map");
const tt = document.getElementById("tooltip");
const ch = document.getElementById("crosshair");
const layers = {
  dem:   document.getElementById("layer-dem"),
  slope: document.getElementById("layer-slope"),
  geo:   document.getElementById("layer-geo"),
};
document.getElementById("layerTabs").addEventListener("click", e=>{
  const btn = e.target.closest("button[data-l]");
  if(!btn) return;
  btn.parentElement.querySelectorAll("button").forEach(b=>b.classList.toggle("on", b===btn));
  const w = btn.dataset.l;
  Object.entries(layers).forEach(([k,el])=>el.classList.toggle("on", k===w));
});

function pxToLatLon(pctX, pctY){
  return {lon: -180 + pctX*360, lat: 90 - pctY*180};
}
function latLonToPct(lat, lon){
  return {
    px: Math.max(0,Math.min(1,(lon+180)/360)),
    py: Math.max(0,Math.min(1,(90-lat)/180)),
  };
}
const hLat=document.getElementById("h-lat"), hLon=document.getElementById("h-lon"),
      hElev=document.getElementById("h-elev"), hSlope=document.getElementById("h-slope"),
      hLand=document.getElementById("h-land");

function updateHUD(lat, lon){
  const q = query(lat, lon);
  hLat.textContent  = lat.toFixed(2)+"°";
  hLon.textContent  = lon.toFixed(2)+"°";
  hElev.textContent = (q.elev>=0?"+":"") + q.elev.toFixed(0) + "m";
  hSlope.textContent = q.slope.toFixed(1) + "%%";
  hLand.textContent  = q.land ? "陆地" : "海洋";
  hLand.style.color = q.land ? "var(--land)" : "var(--sea)";
  return q;
}
function moveTooltip(clientX, clientY){
  const rect = map.getBoundingClientRect();
  const pctX = Math.max(0,Math.min(1,(clientX-rect.left)/rect.width));
  const pctY = Math.max(0,Math.min(1,(clientY-rect.top)/rect.height));
  const {lat,lon} = pxToLatLon(pctX, pctY);
  const q = updateHUD(lat, lon);
  tt.style.display="block";
  tt.style.left=(pctX*100).toFixed(3)+"%%";
  tt.style.top =(pctY*100).toFixed(3)+"%%";
  const L = q.land?"🌲 陆地":"🌊 海洋";
  tt.innerHTML = `<b>${lat.toFixed(2)}°, ${lon.toFixed(2)}°</b><br>
    高程 <span style="color:var(--brand)">${(q.elev>=0?"+":"")+q.elev.toFixed(0)}m</span><br>
    坡度 <span style="color:var(--brand-2)">${q.slope.toFixed(1)}%%</span> · ${L}`;
  ch.style.display="block";
  ch.style.left=(pctX*100).toFixed(3)+"%%";
  ch.style.top =(pctY*100).toFixed(3)+"%%";
  if(typeof onMapMove === "function") onMapMove(lat, lon);
}
map.addEventListener("mousemove", e=>moveTooltip(e.clientX, e.clientY));
map.addEventListener("mouseleave", ()=>{tt.style.display="none";ch.style.display="none"});
map.addEventListener("dblclick", e=>{
  const rect = map.getBoundingClientRect();
  const {lat,lon}=pxToLatLon((e.clientX-rect.left)/rect.width, (e.clientY-rect.top)/rect.height);
  addProbeRow(lat, lon, "点击+");
});

const tbody = document.querySelector("#probe-table tbody");
function goTo(lat, lon){
  const {px,py}=latLonToPct(lat, lon);
  ch.style.display="block";
  ch.style.left=(px*100).toFixed(3)+"%%"; ch.style.top=(py*100).toFixed(3)+"%%";
  tt.style.display="block";
  tt.style.left=(px*100).toFixed(3)+"%%";
  tt.style.top =(py*100).toFixed(3)+"%%";
  const q=updateHUD(lat, lon);
  const L=q.land?"🌲 陆地":"🌊 海洋";
  tt.innerHTML=`<b>${lat.toFixed(2)}°, ${lon.toFixed(2)}°</b><br>
    高程 ${(q.elev>=0?"+":"")+q.elev.toFixed(0)}m<br>
    坡度 ${q.slope.toFixed(1)}%% · ${L}<br>
    <span style="color:var(--muted);font-size:10px">(定位高亮 · 2s 后消失)</span>`;
  setTimeout(()=>{ch.style.display="none";tt.style.display="none"},2000);
}
tbody.addEventListener("click", e=>{
  const tr=e.target.closest("tr[data-lat]"); if(!tr) return;
  goTo(parseFloat(tr.dataset.lat), parseFloat(tr.dataset.lon));
});
function addProbeRow(lat, lon, name){
  const q=query(lat, lon);
  const cls=q.land?"land":"sea";
  const tr=document.createElement("tr");
  tr.dataset.lat=lat.toFixed(3); tr.dataset.lon=lon.toFixed(3);
  tr.innerHTML=`<td><span class="ld ${cls}"></span>${name}</td>
    <td>${lat.toFixed(2)}°</td><td>${lon.toFixed(2)}°</td>
    <td style="font-family:var(--mono)">${(q.elev>=0?"+":"")+q.elev.toFixed(0)}m</td>
    <td style="font-family:var(--mono)">${q.slope.toFixed(1)}%%</td>`;
  tbody.appendChild(tr);
  const {px,py}=latLonToPct(lat,lon);
  const m=document.createElement("div");
  m.className="probe-marker "+cls;
  m.style.left=(px*100).toFixed(3)+"%%";
  m.style.top =(py*100).toFixed(3)+"%%";
  map.appendChild(m);
}

/* 剖面渲染（纯 Canvas，不依赖 CDN —— 完全离线） */
const profileData = %(PROFILE_JSON)s;
const profileA = %(PROFILE_A)s;
const profileB = %(PROFILE_B)s;
const profilePoints = %(PROFILE_POINTS)s;

const canvas = document.getElementById("profile-canvas");
const proTip = document.getElementById("pro-tip");
document.getElementById("pro-fallback").classList.add("hidden");

function drawProfile(hoverIdx){
  const ctx2d = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  canvas.width = cssW*dpr; canvas.height = cssH*dpr;
  ctx2d.setTransform(dpr,0,0,dpr,0,0);
  ctx2d.clearRect(0,0,cssW,cssH);
  const PAD_L=48, PAD_R=64, PAD_T=14, PAD_B=36;
  const plotW=cssW-PAD_L-PAD_R, plotH=cssH-PAD_T-PAD_B;
  const xs=profileData.cum_km, es=profileData.elevs, ss=profileData.slopes;
  const xMin=xs[0], xMax=xs[xs.length-1];
  let eMin=es[0], eMax=es[0]; for(let i=1;i<es.length;i++){if(es[i]<eMin)eMin=es[i];if(es[i]>eMax)eMax=es[i];}
  if(eMin===eMax){eMin-=10;eMax+=10;}
  const ePad=(eMax-eMin)*0.12; eMin-=ePad; eMax+=ePad;
  let sMax=ss[0]; for(let i=1;i<ss.length;i++) if(ss[i]>sMax) sMax=ss[i];
  sMax = Math.max(5, sMax*1.3);
  const sx = x => PAD_L + (x-xMin)/(xMax-xMin)*plotW;
  const syE= e => PAD_T + (1 - (e-eMin)/(eMax-eMin))*plotH;
  const syS= s => PAD_T + (1 - Math.max(0,s)/sMax)*plotH;

  ctx2d.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--border")||"#222";
  ctx2d.lineWidth=1;
  ctx2d.beginPath();
  for(let i=0;i<=5;i++){ const y=PAD_T+plotH*i/5; ctx2d.moveTo(PAD_L,y); ctx2d.lineTo(cssW-PAD_R,y); }
  for(let i=0;i<=8;i++){ const x=PAD_L+plotW*i/8;   ctx2d.moveTo(x,PAD_T); ctx2d.lineTo(x,cssH-PAD_B); }
  ctx2d.stroke();
  if(eMin<0 && eMax>0){
    ctx2d.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue("--text")||"#fff";
    ctx2d.setLineDash([4,4]); ctx2d.beginPath();
    ctx2d.moveTo(PAD_L,syE(0)); ctx2d.lineTo(cssW-PAD_R,syE(0)); ctx2d.stroke();
    ctx2d.setLineDash([]);
  }
  function fillRegion(cond, color){
    ctx2d.beginPath(); let started=false, lastX=0;
    for(let i=0;i<xs.length;i++){
      const c=!!cond(es[i]); const x=sx(xs[i]);
      if(c && !started){ ctx2d.moveTo(x, syE(0)); ctx2d.lineTo(x, syE(es[i])); started=true; }
      else if(c && started){ ctx2d.lineTo(x, syE(es[i])); }
      else if(!c && started){ ctx2d.lineTo(lastX, syE(0)); ctx2d.closePath(); started=false; }
      lastX=x;
    }
    if(started){ ctx2d.lineTo(lastX, syE(0)); ctx2d.closePath(); }
    ctx2d.fillStyle=color; ctx2d.globalAlpha=0.55; ctx2d.fill(); ctx2d.globalAlpha=1;
  }
  fillRegion(e=>e>=0, "#4ade80"); fillRegion(e=>e<0, "#38bdf8");
  ctx2d.strokeStyle="#0f172a"; ctx2d.lineWidth=1.4; ctx2d.beginPath();
  for(let i=0;i<xs.length;i++){ const x=sx(xs[i]),y=syE(es[i]); i?ctx2d.lineTo(x,y):ctx2d.moveTo(x,y); }
  ctx2d.stroke();
  ctx2d.strokeStyle="#dc2626"; ctx2d.lineWidth=1; ctx2d.globalAlpha=0.9; ctx2d.beginPath();
  for(let i=0;i<xs.length;i++){ const x=sx(xs[i]),y=syS(ss[i]); i?ctx2d.lineTo(x,y):ctx2d.moveTo(x,y); }
  ctx2d.stroke();
  ctx2d.globalAlpha=0.12; ctx2d.fillStyle="#dc2626"; ctx2d.beginPath();
  ctx2d.moveTo(sx(xs[0]), syS(0));
  for(let i=0;i<xs.length;i++) ctx2d.lineTo(sx(xs[i]), syS(ss[i]));
  ctx2d.lineTo(sx(xs[xs.length-1]), syS(0)); ctx2d.closePath(); ctx2d.fill(); ctx2d.globalAlpha=1;

  ctx2d.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted")||"#888";
  ctx2d.font="11px ui-monospace,Menlo,monospace";
  for(let i=0;i<=5;i++){
    const y=PAD_T+plotH*i/5; const val=eMax-(eMax-eMin)*i/5;
    ctx2d.fillText((val>=0?"+":"")+val.toFixed(0), 6, y+4);
  }
  for(let i=0;i<=5;i++){
    const y=PAD_T+plotH*i/5; const val=sMax-sMax*i/5;
    ctx2d.fillStyle="#dc2626"; ctx2d.fillText(val.toFixed(0)+"%%", cssW-PAD_R+6, y+4);
    ctx2d.fillStyle=getComputedStyle(document.documentElement).getPropertyValue("--muted")||"#888";
  }
  for(let i=0;i<=8;i++){
    const x=PAD_L+plotW*i/8; const val=xMin+(xMax-xMin)*i/8;
    ctx2d.fillText(val.toFixed(0)+"km", x-18, cssH-PAD_B+18);
  }
  if(hoverIdx!==undefined && hoverIdx>=0 && hoverIdx<xs.length){
    const x=sx(xs[hoverIdx]), y=syE(es[hoverIdx]), y2=syS(ss[hoverIdx]);
    ctx2d.strokeStyle="#f59e0b"; ctx2d.setLineDash([3,3]); ctx2d.beginPath();
    ctx2d.moveTo(x,PAD_T); ctx2d.lineTo(x,cssH-PAD_B); ctx2d.stroke(); ctx2d.setLineDash([]);
    ctx2d.fillStyle="#f59e0b";
    ctx2d.beginPath(); ctx2d.arc(x,y,4,0,Math.PI*2); ctx2d.fill();
    ctx2d.beginPath(); ctx2d.arc(x,y2,3.5,0,Math.PI*2); ctx2d.fill();
  }
}

let lastHoverIdx=-1;
function idxFromX(clientX){
  const rect=canvas.getBoundingClientRect();
  const PAD_L=48, PAD_R=64, cssW=rect.width;
  const plotW=cssW-PAD_L-PAD_R;
  const xs=profileData.cum_km; const xMin=xs[0], xMax=xs[xs.length-1];
  const lx = clientX-rect.left-PAD_L;
  if(lx<0 || lx>plotW) return -1;
  const xv=xMin + lx/plotW*(xMax-xMin);
  let lo=0, hi=xs.length-1;
  while(lo<hi){ const m=(lo+hi+1)>>1; if(xs[m]<=xv) lo=m; else hi=m-1; }
  if(lo+1<xs.length && xv-xs[lo] > xs[lo+1]-xv) lo++;
  return lo;
}
document.getElementById("profile-wrap").addEventListener("mousemove", e=>{
  const i=idxFromX(e.clientX);
  if(i<0){ proTip.style.display="none"; lastHoverIdx=-1; drawProfile(); return; }
  lastHoverIdx=i; drawProfile(i);
  proTip.style.display="block";
  proTip.textContent=`km ${profileData.cum_km[i].toFixed(0)} · 高程 ${(profileData.elevs[i]>=0?"+":"")+profileData.elevs[i].toFixed(0)}m · 坡度 ${profileData.slopes[i].toFixed(1)}%%`;
  const pp=profilePoints[i];
  if(pp){
    const {px,py}=latLonToPct(pp.lat, pp.lon);
    ch.style.display="block";
    ch.style.left=(px*100).toFixed(3)+"%%"; ch.style.top=(py*100).toFixed(3)+"%%";
    updateHUD(pp.lat, pp.lon);
  }
});
document.getElementById("profile-wrap").addEventListener("mouseleave", ()=>{
  proTip.style.display="none"; lastHoverIdx=-1; drawProfile();
});

function onMapMove(lat, lon){
  let best=-1, bestD=1e18;
  for(let i=0;i<profilePoints.length;i+=6){
    const p=profilePoints[i];
    const dL=p.lat-lat, dO=(p.lon-lon)*Math.cos(p.lat*Math.PI/180);
    const d=dL*dL+dO*dO; if(d<bestD){bestD=d; best=i;}
  }
  if(best>=0 && bestD<0.6){
    lastHoverIdx=best; drawProfile(best);
    proTip.style.display="block";
    proTip.textContent=`km ${profileData.cum_km[best].toFixed(0)} · ${(profileData.elevs[best]>=0?"+":"")+profileData.elevs[best].toFixed(0)}m · ${profileData.slopes[best].toFixed(1)}%%`;
  }
}
window.addEventListener("resize", ()=>drawProfile(lastHoverIdx));
drawProfile();
})();
</script>
</body>
</html>
"""


def _build_probe_markers_html(probe_info):
    """把探针点信息渲染为地图上的标记与名称标签 HTML 字符串。"""
    parts = []
    for p in probe_info:
        px = (p["lon"] + 180.0) / 360.0
        py = (90.0 - p["lat"]) / 180.0
        cls = "land" if p["land"] else "sea"
        safe_name = html.escape(str(p.get("name", "P")), quote=True)
        style_m = f"left:{px*100:.4f}%;top:{py*100:.4f}%"
        style_l = f"left:{px*100:.4f}%;top:{py*100:.4f}%;" + \
                  f"color:{'#b91c1c' if p['land'] else '#0369a1'}"
        parts.append(
            f'<div class="probe-marker {cls}" style="{style_m}"></div>'
            f'<div class="probe-label" style="{style_l}">{safe_name}</div>'
        )
    return "".join(parts)


def _build_probe_table_rows(probe_info):
    """把探针点信息渲染为表格行 HTML（含海拔/坡度），无探针时给出占位提示行。"""
    rows = []
    for p in probe_info:
        cls = "land" if p["land"] else "sea"
        safe_name = html.escape(str(p.get("name", "P")), quote=True)
        rows.append(
            f'<tr data-lat="{p["lat"]:.5f}" data-lon="{p["lon"]:.5f}">'
            f'<td><span class="ld {cls}"></span>{safe_name}</td>'
            f'<td>{p["lat"]:.2f}°</td><td>{p["lon"]:.2f}°</td>'
            f'<td style="font-family:ui-monospace,Menlo,monospace">{p["elev"]:+.0f}m</td>'
            f'<td style="font-family:ui-monospace,Menlo,monospace">{p["slope"]:.1f}%</td>'
            f"</tr>"
        )
    return "".join(rows) or \
        '<tr><td colspan="5" style="text-align:center;color:var(--muted);padding:16px">' \
        '暂无探针点；在地图上<b>双击</b>即可新增。</td></tr>'


def _build_poly_html(polygon_stats):
    """把多边形统计渲染为概要 pill 与统计网格 HTML，未提供多边形时返回提示文本。"""
    if polygon_stats is None:
        return ("未提供 --poly",
                '<div style="color:var(--muted);text-align:center;padding:20px 8px">'
                '调用时传入 polygon 或 CLI <code style="background:var(--panel-2);'
                'padding:2px 6px;border-radius:4px">--poly</code> 可在此显示 '
                '像素数/均值/峰值/陆地占比等 8 项统计。</div>')
    st = polygon_stats
    pill = f"{st['pixel_count']:,} px · 陆地 {st['land_ratio']*100:.1f}%"
    items = [
        ("像素总数",       f"{st['pixel_count']:,}"),
        ("陆地像素",       f"{st['land_pixel_count']:,}"),
        ("平均海拔",       f"{st['mean_elevation']:+.0f} m"),
        ("最高峰",         f"{st['peak']:+.0f} m"),
        ("最低点/海深",    f"{st['trough']:+.0f} m"),
        ("陆地占比",       f"{st['land_ratio']*100:.2f} %"),
        ("山脉占比",       f"{st['mountain_ratio']*100:.2f} %"),
        ("峰-谷极差",      f"{abs(st['peak'])+abs(st['trough']):,.0f} m"),
    ]
    rows = "".join(f"<div><span>{k}</span><span>{v}</span></div>" for k, v in items)
    return pill, f'<div class="poly-grid">{rows}</div>'


def export_html_report(
    ctx: WorldGeoContext,
    output_path: str,
    probe_points = (),
    profile = None,
    polygon = None,
    *,
    layer_size: int = 1024,
    query_size: int = 512,
    title: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """**公开接口**：从已加载的 WorldGeoContext 导出单文件、完全离线的交互 HTML 报告。

    HTML 报告结构（详见 ``_HTML_TEMPLATE``）：
        1. 顶部 KPI 行：陆地占比 / 山地占比 / 最高峰 / 最深海沟 / 探针点数 /
           DEM 网格尺寸 / 查询精度（共 7 张卡片）。
        2. 左栏 全球交互查询图：三图层 PNG（高程 / 坡度 / 地理带）切换 + 鼠标悬停
           实时 HUD（经纬度 / 高程 / 坡度 / 陆海） + 双击地图追加临时探针点。
        3. 右上 探针点清单表：点击任意行 → 地图十字线高亮 + tooltip 定位（2s 后消失）。
        4. 右中 多边形区域统计：传入 ``polygon`` 时展示 8 项像素级统计（像素总数 /
           陆地像素 / 平均海拔 / 最高峰 / 最低点 / 陆海占比 / 山脉占比 / 峰谷极差）。
        5. 下栏 剖面 A→B：纯 Canvas 绘制高程（填充）+ 坡度双轴折线，鼠标悬停与地图
           十字线双向联动（地图上移动靠近剖面线时自动在剖面画出对应的竖线）。
        6. 页脚：DEM 文件路径 / 图层分辨率 / 查询分辨率 / 生成命令行。

    所有资源均以 base64 内嵌：三图层 PNG（``layer_size²``）+ 两查询网格 float32
    （``query_size²``，供 JS 双线性内插）+ 剖面 JSON 数据 + CSS/HTML/JS，无需
    任何外部 CDN。默认配置下文件约 3.5~5 MB，双击 HTML 即可离线打开。

    Parameters
    ----------
    ctx : WorldGeoContext
        **已加载** 的单例实例。``dem / land_mask / mountain_mask / lat_grid / lon_grid``
        必须可用；推荐通过 ``build_html_report`` 创建以避免手漏。
    output_path : str
        HTML 写入路径。父目录不存在会自动创建。若写入时遭遇 ``PermissionError``
        （常见于 Excel/浏览器正占用文件），函数会在文件名后追加 ``_YYYYMMDD_HHMMSS``
        时间戳并写入备用路径，返回值会切换为备用路径，同时向 stderr 打印提示。
    probe_points : sequence of (lat, lon, name) tuples
        探针点三元组。空时使用默认 4 点。
    profile : ((latA, lonA), (latB, lonB)) or None
        剖面线段端点。空时使用默认剖面（北海→杭州湾）。
    polygon : sequence of (lat, lon) tuples or None
        多边形顶点；空时在「多边形区域统计」面板展示占位说明。
    layer_size : int, keyword-only
        HTML 内三图层（高程 / 坡度 / 地理带）的 PNG 分辨率，默认 1024²。
        越大越清晰，文件体积线性增长（1024 → 2048 ≈ ×4）。
    query_size : int, keyword-only
        JS 悬停双线性查询使用的 float32 网格分辨率，默认 512²。
        512 下误差 < 原始 DEM 的 0.1%，且可在低端浏览器流畅交互。
    title : str or None, keyword-only
        浏览器标签页标题与页头文字。空时自动构造为
        ``WorldGeoContext 交互地形报告 · Seed {seed}``。
    seed : int or None, keyword-only
        展示用种子编号（写入 KPI 标题/页脚）。空时从单例的 ``_instance_key`` 回读；
        若仍未知则回退为 "?"。

    Returns
    -------
    str
        实际写入的文件路径（若触发权限备份会返回备用路径）。

    Raises
    ------
    RuntimeError
        当模板 ``%(KEY)s`` 占位符与 payload 键不一致（模板被手工误改）时抛出，
        附带错误摘要与排查建议。

    Notes
    -----
    ``_HTML_TEMPLATE`` 是单文件 HTML/CSS/JS 三引号大模板，使用 Python 的
    ``%(KEY)s`` 语法替换。**所有字面量百分号必须写作 ``%%``**，否则会触发
    ``TypeError: not enough arguments for format string``（FAQ Q4 亦有说明）。
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not probe_points:
        probe_points = _parse_points(None)
    if profile is None:
        profile = _parse_profile(None)
    (latA, lonA), (latB, lonB) = profile

    dem = ctx.dem
    S = ctx.size
    land_ratio = float(ctx.land_mask.mean() * 100)
    mont_ratio = float(ctx.mountain_mask.mean() * 100)
    dem_max, dem_min = float(dem.max()), float(dem.min())

    # —— 三张图层 PNG base64 ——
    layer_dem_b64 = _ndarray_to_png_b64(
        dem, make_terrain_cmap(), vmin=-10500, vmax=8900, target_s=layer_size)
    from .geo_context import _compute_slope_field
    slope = _compute_slope_field(dem, ctx.lat_grid, ctx.lon_grid)
    land_mask = ctx.land_mask
    layer_slope_b64 = _ndarray_to_png_b64(
        slope, make_slope_cmap(), vmin=0, vmax=100,
        mask_ocean=land_mask, ocean_rgb=(0x0A, 0x19, 0x30), target_s=layer_size)
    dem_c = np.clip(dem, _GEO_CLAMP_MIN, _GEO_CLAMP_MAX)
    cmap_geo, norm_geo = make_geozone_norm_and_cmap()
    layer_geo_b64 = _ndarray_to_png_b64(
        dem_c, cmap_geo, norm=norm_geo, target_s=layer_size)

    # —— 查询网格（DEM + slope，float32）——
    q_dem_b64, _ = _float32_grid_b64(dem,               target_s=query_size)
    q_slp_b64, _ = _float32_grid_b64(slope.astype(np.float32), target_s=query_size)

    # —— 探针点详细信息 ——
    probe_info = [
        dict(name=name, lat=float(lat), lon=float(lon),
             elev=float(ctx.get_elevation(lat, lon)),
             slope=float(ctx.get_slope(lat, lon)),
             land=bool(ctx.is_land(lat, lon)))
        for lat, lon, name in probe_points
    ]

    # —— 剖面数据 ——
    pf = _sample_profile(ctx, latA, lonA, latB, lonB, n_pts=800)
    profile_json = dict(
        cum_km=[float(x) for x in pf["cum_km"]],
        elevs= [float(x) for x in pf["elevs"]],
        slopes=[float(x) for x in pf["slopes"]],
    )
    profile_points = [
        {"lat": float(la), "lon": float(lo)}
        for la, lo in zip(pf["lats"], pf["lons"])
    ]
    prof_dist_km = float(pf["cum_km"][-1])

    # —— 多边形统计 ——
    poly_stats = ctx.get_region_stats(polygon) if polygon else None
    poly_pill, poly_html_body = _build_poly_html(poly_stats)

    # —— 元数据 ——
    if seed is None:
        k = WorldGeoContext._instance_key
        seed = k[0] if k else "?"
    title = title or f"WorldGeoContext 交互地形报告 · Seed {seed}"
    gen_time = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dem_path = os.path.join(_THIS_DIR, "output", f"seed{seed}_dem.npy")
    if not os.path.exists(dem_path):
        dem_path = os.path.join(_THIS_DIR, "output", "world_dem.npy")
    cmd = (f"python -m terrain.core.visualize_context "
           f"--seed {seed} --size {S} --html")

    # —— 组装 HTML ——
    payload = dict(
        TITLE=html.escape(title, quote=True), SEED=seed, SIZE=S, GEN_TIME=gen_time,
        LAND_RATIO=land_ratio, MONT_RATIO=mont_ratio,
        PEAK=dem_max, TROUGH=dem_min,
        LAYER_DEM=layer_dem_b64, LAYER_SLOPE=layer_slope_b64, LAYER_GEO=layer_geo_b64,
        LAYER_SIZE=layer_size, Q_SIZE=query_size,
        Q_DEM_B64=q_dem_b64, Q_SLP_B64=q_slp_b64,
        PROBE_COUNT=len(probe_info),
        PROBE_MARKERS_HTML=_build_probe_markers_html(probe_info),
        PROBE_TABLE_ROWS=_build_probe_table_rows(probe_info),
        POLY_PILL=poly_pill, POLY_HTML_BODY=poly_html_body,
        PROFILE_DESC=f"A({latA:.1f}°,{lonA:.1f}°) → B({latB:.1f}°,{lonB:.1f}°)",
        PROFILE_DIST=prof_dist_km,
        PROFILE_JSON=json.dumps(profile_json, ensure_ascii=False, separators=(",", ":")),
        PROFILE_A=json.dumps([float(latA), float(lonA)]),
        PROFILE_B=json.dumps([float(latB), float(lonB)]),
        PROFILE_POINTS=json.dumps(profile_points, ensure_ascii=False, separators=(",", ":")),
        DEM_PATH=html.escape(dem_path, quote=True), CMD=cmd,
    )
    try:
        html_out = _HTML_TEMPLATE % payload
    except (ValueError, KeyError) as exc:
        raise RuntimeError(
            f"HTML 模板格式化失败: {exc}\n"
            "请检查模板中的 %%()s 占位符与 payload 键是否一致。") from exc

    # —— 写入 ——
    def _save(p):
        """把 HTML 内容写入路径 p。"""
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(html_out)
    try:
        _save(output_path)
    except PermissionError:
        stem, ext = os.path.splitext(output_path)
        backup = f"{stem}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext or '.html'}"
        _save(backup)
        print(f"[visualize/html] ⚠ 原路径被占用，已保存到备用路径: {backup}")
        output_path = backup
    return output_path


def build_html_report(
    seed: int = 1, size: int = 2048,
    probe_points=(), profile=None, polygon=None,
    output_path: Optional[str] = None,
    *, layer_size: int = 1024, query_size: int = 512,
    title: Optional[str] = None,
):
    """**CLI 友好工厂**：自动加载 DEM（单例缓存）→ 组装参数 → 导出 HTML 报告。

    与 ``export_html_report`` 的关系：
        - ``export_html_report(ctx, output_path, ...)`` 面向“已有单例 / 多份报告
          共享同一份 DEM / 想自定义输出路径”的编程场景。
        - ``build_html_report(seed, size, ...)`` 面向「一把梭」CLI / Jupyter 场景：
          内部先 ``WorldGeoContext.from_seed``，再补齐默认输出路径，最后调用前者。

    Parameters
    ----------
    seed : int
        世界种子，默认 1。命中 ``seed{seed}_dem.npy`` / ``world_dem.npy`` 顺序
        （完整查找链见 ``_resolve_dem_path``）。
    size : {2048, 4096}
        DEM 单边像素数，默认 2048。
    probe_points / profile / polygon
        叠加元素；格式与 ``export_html_report`` 完全一致。
    output_path : str or None
        HTML 输出路径。空时自动写入
        ``terrain/output/geoctx_report_seed{seed}.html``（父目录按需创建）。
    layer_size / query_size / title : keyword-only
        与 ``export_html_report`` 相同，含义不赘述。

    Returns
    -------
    tuple (output_path, ctx)
        - output_path (str): 实际写入的 HTML 文件路径（权限备份后会变更）。
        - ctx (WorldGeoContext): 已加载的单例，供调用方复用进行后续查询。
    """
    ctx = WorldGeoContext.from_seed(seed=seed, size=size)
    if output_path is None:
        out_dir = os.path.join(_THIS_DIR, "output")
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, f"geoctx_report_seed{seed}.html")
    out = export_html_report(
        ctx, output_path=output_path,
        probe_points=probe_points, profile=profile, polygon=polygon,
        layer_size=layer_size, query_size=query_size,
        title=title, seed=seed,
    )
    return out, ctx


# ============================================================================
# CLI 入口
# ============================================================================

def main() -> None:
    """``visualize_context.py`` 命令行入口。

    处理流程：
        1. 使用 ``argparse`` 解析所有参数（PNG 参数 + HTML 参数两大组）。
        2. 用 ``_parse_points / _parse_profile / _parse_poly`` 把字符串叠加元素转为
           结构化 Python 对象；失败时抛 ``ValueError`` 给用户提示正确格式。
        3. PNG 分支（``--no-png`` 除外）：调用 ``build_geo_figure`` → ``fig.savefig``
           保存；遭遇 ``PermissionError`` 自动以时间戳为后缀回退到备用文件名。
        4. HTML 分支（``--html`` 或 ``--html-output`` 任一启用）：调用
           ``build_html_report`` 生成交互报告，打印文件体积与耗时，并提示可
           双击 HTML 离线打开。
        5. 两项都开启时额外打印总耗时。

    完整 CLI 参数表见 ``terrain/TECHNICAL.md §19``。常用参数：
        --seed, --size, --output, --points, --profile, --poly, --downsample,
        --html, --html-output, --html-layer-size, --html-query-size, --no-png
    """
    ap = argparse.ArgumentParser(
        description="基于 WorldGeoContext 的 4 面板地形可视化工具 + 交互 HTML 报告")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", type=int, default=4096, choices=[2048, 4096])
    ap.add_argument("--output", type=str, default=None,
                    help="输出 PNG 路径（默认 terrain/output/geoctx_visual_seed{seed}.png）")
    ap.add_argument("--points", type=str, default=None,
                    help="探针点列表 lat0,lon0[,name0],lat1,lon1[,name1],...")
    ap.add_argument("--profile", type=str, default=None,
                    help="剖面线段：latA,lonA,latB,lonB")
    ap.add_argument("--poly", type=str, default=None,
                    help="多边形叠加：lat0,lon0,lat1,lon1,...,latN,lonN")
    ap.add_argument("--downsample", type=int, default=1,
                    help="PNG 显示降采样（4096 可设 2，加速 PNG 绘图）")
    ap.add_argument("--dpi", type=int, default=120)

    # —— HTML 交互报告参数 ——
    ap.add_argument("--html", action="store_true",
                    help="额外导出单文件 HTML 交互报告（默认路径见下）")
    ap.add_argument("--html-output", type=str, default=None,
                    help="显式指定 HTML 输出路径（自动启用 --html）。默认："
                         "terrain/output/geoctx_report_seed{seed}.html")
    ap.add_argument("--html-layer-size", type=int, default=1024,
                    help="HTML 内三图层 PNG 分辨率，默认 1024²（越大越清晰，文件越大）")
    ap.add_argument("--html-query-size", type=int, default=512,
                    help="HTML 内 JS 悬停查询网格分辨率，默认 512²（越大越精确）")
    ap.add_argument("--no-png", action="store_true",
                    help="仅输出 HTML（跳过 PNG 4 面板图生成；若只需要 HTML 报告可提速 60%%+）")
    a = ap.parse_args()

    # 解析参数
    probe_pts = _parse_points(a.points)
    profile = _parse_profile(a.profile)
    polygon = _parse_poly(a.poly)

    html_enabled = bool(a.html) or (a.html_output is not None)

    # PNG 默认路径
    if not a.no_png:
        if a.output is None:
            out_dir = os.path.join(_THIS_DIR, "output")
            os.makedirs(out_dir, exist_ok=True)
            a.output = os.path.join(out_dir, f"geoctx_visual_seed{a.seed}.png")
        else:
            parent = os.path.dirname(os.path.abspath(a.output))
            os.makedirs(parent, exist_ok=True)

    # HTML 默认路径
    html_out_path = None
    if html_enabled:
        if a.html_output:
            html_out_path = a.html_output
            parent = os.path.dirname(os.path.abspath(html_out_path))
            if parent: os.makedirs(parent, exist_ok=True)
        else:
            out_dir = os.path.join(_THIS_DIR, "output")
            os.makedirs(out_dir, exist_ok=True)
            html_out_path = os.path.join(out_dir, f"geoctx_report_seed{a.seed}.html")

    t0 = time.time()

    # ---------- PNG ----------
    if not a.no_png:
        print(f"[visualize/png ] seed={a.seed}  size={a.size}  输出 → {a.output}")
        fig, *_ = build_geo_figure(
            seed=a.seed, size=a.size,
            probe_points=probe_pts, profile=profile, polygon=polygon,
            downsample_for_display=a.downsample, dpi=a.dpi,
        )
        try:
            fig.savefig(a.output, dpi=a.dpi, bbox_inches="tight",
                        facecolor="#F8FAFC", edgecolor="none")
        except PermissionError:
            stem, ext = os.path.splitext(a.output)
            backup = f"{stem}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            fig.savefig(backup, dpi=a.dpi, bbox_inches="tight",
                        facecolor="#F8FAFC", edgecolor="none")
            print(f"[visualize/png ] ⚠ 原路径被占用，已保存到备用路径: {backup}")
            a.output = backup
        finally:
            plt.close(fig)
        print(f"[visualize/png ] 完成  耗时 {time.time() - t0:.1f}s  → {a.output}")

    # ---------- HTML 交互报告 ----------
    if html_enabled:
        t1 = time.time()
        print(f"[visualize/html] 生成交互报告 (图层 {a.html_layer_size}² / "
              f"查询 {a.html_query_size}²) → {html_out_path}")
        html_real_path, _ctx = build_html_report(
            seed=a.seed, size=a.size,
            probe_points=probe_pts, profile=profile, polygon=polygon,
            output_path=html_out_path,
            layer_size=a.html_layer_size, query_size=a.html_query_size,
        )
        size_kb = os.path.getsize(html_real_path) / 1024
        print(f"[visualize/html] 完成  耗时 {time.time() - t1:.1f}s  "
              f"({size_kb/1024:.1f} MB)  → {html_real_path}")
        print(f"[visualize/html] 💡 离线 100% 可用：直接双击 HTML 即可在浏览器中交互查看。")

    total = time.time() - t0
    if html_enabled and not a.no_png:
        print(f"[visualize     ] 全部产物完成  总耗时 {total:.1f}s")


if __name__ == "__main__":
    main()
