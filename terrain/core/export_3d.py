# -*- coding: utf-8 -*-
"""
export_3d.py — DEM → 3D 模型导出（OBJ + glTF 2.0）+ 桌面窗口预览
=============================================================

读取 ``world_dem.npy``（米制高程，行 0 ≈ 南极，末行 ≈ 北极），输出：
  * ``*.obj`` + ``*.mtl`` + ``*_tex.png``   — 经典格式（Blender / Maya / 3D 查看器）
  * ``*.gltf``                              — glTF 2.0 自包含单文件（纹理 base64 内嵌）
支持两种地形形态：
  * plane — 拉伸平面浮雕（X=经度·111.32km/°，Y=纬度·111.32km/°，Z=高程×exagg）
  * globe — 球体星球（半径 R0=6371km + 高程×exagg，带极冠补面）
垂直夸张 exagg 默认 ``auto`` 自适应：平面按地图幅宽（起伏约占 10%），
球体按半径（约占 15%），保证远景下起伏清晰；``--exagg-plane/--exagg-globe``
传数字可手动覆盖。``--coast-cut`` 将海洋钳到海平面（0 等高线锋利贴合，
适合展示海岸线；默认 False 保留海底起伏）；预览窗口默认开启平海。
表面纹理（与网格抽稀解耦，默认全分辨率）：
  * ``--texture auto``（默认）— 优先复用同目录 2D 渲染图（``seed*_map.png`` /
    ``world_map.png``，行 0 = 南极，与 3D UV 一致），否则用高程着色降级；
  * ``--texture map`` — 强制使用 2D 渲染图；``--texture elevation`` — 高程着色；
  * 高程着色已内建山影（hillshade）+ 坡度细节纹理 + 海岸 0 等高线描边，
    避免大范围平台同色"色块"，海岸在 3D 上清晰可读。

CLI 使用示例：
  # 1) 最简：加载默认 world_dem.npy，导出两种格式 + 两种形态，并打开预览窗口
  python -m terrain.core.export_3d --preview

  # 2) 只导出球体 OBJ，抽稀 8 倍
  python -m terrain.core.export_3d --dem terrain/output/world_dem.npy \
      --mode globe --export obj --decimate 8

  # 3) 用 2D 渲染图当 3D 贴图，看一眼预览
  python -m terrain.core.export_3d --texture map --preview

  # 4) 生成后直接导出（generate_world 内置钩子）
  python -m terrain.core.generate_world --seed 1 --size 2048 --export-3d --export-3d-mode both

  # 5) 样条平滑加密面数（--subdivide 2 → 每个 quad 拆 4 子 quad，面数约 ×4，
  #    更平滑更细密；OBJ/glTF 体积随面数近似 ×4，谨慎使用）
  python -m terrain.core.export_3d --subdivide 2 --export gltf

  # 6) 导出平海（海岸锋利贴合）+ 固定夸张 80 倍球体
  python -m terrain.core.export_3d --coast-cut --exagg-globe 80 --mode globe
=============================================================
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import json
import os
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# ============================================================================
# 本地复刻常量 / 工具（避免引入 torch 与 GAN 权重，与 geo_context 口径一致）
# ============================================================================

# terrain_engine.py:47-50 的全局「米制高程 ↔ 归一化 v∈[0,1]」分段线性映射
_M_KNOTS = np.array([-11000, -6000, -4000, -2500, -1000, -200, 0,
                     100, 400, 1000, 2000, 3500, 5000, 8849], np.float64)
_V_KNOTS = np.array([0.00, 0.08, 0.22, 0.34, 0.42, 0.47, 0.50,
                     0.53, 0.60, 0.70, 0.82, 0.92, 0.97, 1.00], np.float64)
_KM_PER_DEG = 111320.0        # 米/纬度 (°)，geo_context.py:47
_EARTH_R_M = 6371000.0        # 地球平均半径（米）

# exagg="auto" 自适应夸张的目标占比（relief 高度 / 地图幅宽·半径）
_AUTO_RELIEF_PLANE = 0.10     # 平面：起伏 ≈ 全幅宽(360°·111.32km) 的 10%
_AUTO_RELIEF_GLOBE = 0.15     # 球体：起伏 ≈ 半径的 15%

# glTF 2.0 枚举
_GLTF_CLAMP_TO_EDGE = 33071
_GLTF_REPEAT = 10497


def _m2v(m: np.ndarray) -> np.ndarray:
    """米制高程 → 归一化 [0, 1]（与 terrain_engine.m2v 逐点一致）。"""
    return np.interp(m, _M_KNOTS, _V_KNOTS).astype(np.float32)


def _make_dem_cmap():
    """复刻 terrain_engine.make_terrain_cmap（避免引入 torch），同 visualize_context.py。"""
    from matplotlib.colors import LinearSegmentedColormap
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


def _lat_lon_grids(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """像素中心对齐的纬度/经度网格（行 0 ≈ 南极，末行 ≈ 北极，同 geo_context.py:214）。"""
    lats = (-90.0 + (np.arange(h) + 0.5) * (180.0 / h)).astype(np.float64)
    lons = (-180.0 + (np.arange(w) + 0.5) * (360.0 / w)).astype(np.float64)
    return lats, lons


def _decimate(dem: np.ndarray, step: int) -> np.ndarray:
    """按 step 抽稀 DEM（线性插值；step<=1 时原样返回，保证 AC-2 逐点一致）。"""
    if step is None or step <= 1:
        return np.asarray(dem, np.float32)
    from scipy.ndimage import zoom
    return zoom(dem, 1.0 / step, order=1).astype(np.float32)


def _upsample_dem(dem: np.ndarray, factor: int) -> np.ndarray:
    """样条平滑加密 DEM：cubic spline 上采样至 (h·factor, w·factor)。

    factor<=1 时原样返回。cubic B-spline 在陡峭边界可能轻微振铃
    （超出原高程范围），裁剪回原始 [min, max]，避免海沟/雪线等
    极值被外推出物理上不可能的高程。
    """
    dem = np.asarray(dem, np.float32)
    if factor is None or factor <= 1:
        return dem
    lo, hi = float(dem.min()), float(dem.max())
    from scipy.ndimage import zoom
    dem = zoom(dem, (factor, factor), order=3).astype(np.float32)
    return np.clip(dem, lo, hi)


# ============================================================================
# 自适应垂直夸张（exagg="auto"）
# ============================================================================

def _auto_exagg_plane(dem: np.ndarray) -> float:
    """让平面模式的起伏高度 ≈ 全幅宽(360°·111.32km) 的 _AUTO_RELIEF_PLANE。"""
    span = float(dem.max() - dem.min())
    if span <= 1e-6:
        return 1.0  # 无起伏（全平），夸张无意义
    return max(1.0, 360.0 * _KM_PER_DEG * _AUTO_RELIEF_PLANE / span)


def _auto_exagg_globe(dem: np.ndarray, r0: float = _EARTH_R_M) -> float:
    """让球体模式的起伏高度 ≈ 半径 r0 的 _AUTO_RELIEF_GLOBE。"""
    span = float(dem.max() - dem.min())
    if span <= 1e-6:
        return max(1.0, r0 * _AUTO_RELIEF_GLOBE)
    return max(1.0, r0 * _AUTO_RELIEF_GLOBE / span)


def _resolve_exagg(e, dem: np.ndarray, mode: str, r0: float = _EARTH_R_M) -> float:
    """解析垂直夸张系数：float 原样透传；"auto"/None 按模式自适应计算。"""
    if e is None or (isinstance(e, str) and e.strip().lower() == "auto"):
        return (_auto_exagg_globe(dem, r0) if mode == "globe"
                else _auto_exagg_plane(dem))
    return float(e)


# ============================================================================
# 网格构建
# ============================================================================

def _build_faces(h: int, w: int, wrap: bool) -> np.ndarray:
    """行主序网格 (h,w) → 三角面（每 quad 拆 2 三角）。

    wrap=True 时列方向循环（球体经度无缝），wrap=False 时最右列不生成面（平面）。
    顶点索引 = i*w + j。
    """
    i0 = np.arange(h - 1)[:, None] * w
    j0 = np.arange(w)[None, :]
    a = i0 + j0
    b = a + w
    j1 = (j0 + 1) % w if wrap else j0 + 1
    c = i0 + j1
    d = c + w
    mask = np.broadcast_to(j1 < w, a.shape)  # 平面模式排除最右列
    t1 = np.stack([a, c, d], axis=-1)[mask]
    t2 = np.stack([a, d, b], axis=-1)[mask]
    return np.concatenate([t1, t2]).astype(np.int64)


def _pole_faces(ring_start: int, pole: int, w: int, reverse: bool) -> np.ndarray:
    """极冠扇形三角（ring 行 + 极顶顶点），绕序交给 _orient 统一纠正。"""
    js = np.arange(w)
    nxt = (js + 1) % w
    pole_v = np.full(w, pole, np.int64)
    if reverse:
        tris = np.stack([pole_v, ring_start + nxt, ring_start + js], axis=-1)
    else:
        tris = np.stack([pole_v, ring_start + js, ring_start + nxt], axis=-1)
    return tris.astype(np.int64)


def _orient(verts: np.ndarray, faces: np.ndarray,
            mode: str, center: np.ndarray) -> np.ndarray:
    """统一面绕序：平面朝 +Z，球体朝外（法线点积（质心-球心）≥0）。"""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if mode == "plane":
        outward = n[:, 2] >= 0
    else:
        c = tri.mean(axis=1)
        outward = np.einsum("ij,ij->i", n, c - center) >= 0
    flip = ~outward
    if flip.any():
        faces = faces.copy()
        faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces


def _vertex_normals(verts: np.ndarray, faces: np.ndarray,
                    center: Optional[np.ndarray] = None) -> np.ndarray:
    """面积加权顶点法线（跨面累加 + 归一化）。

    center=None（平面模式）时，法线长度 ≈0（如 coast_cut 后的平坦海面）
    兜底为 +Z；(0,0,1)。center 给定（球体模式）时兜底为径向外侧方向，
    避免平坦海面顶点法线为 0 导致渲染发黑。
    """
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], n)
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / np.maximum(ln, 1e-12)
    bad = ln.ravel() < 1e-9
    if bad.any():
        if center is None:
            vn[bad] = [0.0, 0.0, 1.0]
        else:
            d = verts[bad] - center
            vn[bad] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return vn


def _uvs(h: int, w: int, wrap: bool) -> np.ndarray:
    """顶点 UV。纹理数组行 0 = 南极（图像顶行），v=1 - (i+0.5)/h 与之对齐。"""
    if wrap:
        u = (np.arange(w) + 0.5) / w          # 球体：texel 中心对齐，经度无缝
    else:
        u = np.arange(w) / (w - 1) if w > 1 else np.array([0.5])
    v = 1.0 - (np.arange(h) + 0.5) / h
    U, V = np.meshgrid(u, v)
    uv = np.stack([U.ravel(), V.ravel()], axis=-1).astype(np.float32)
    return uv


def build_mesh(dem: np.ndarray, mode: str, exagg: float,
               r0: float = _EARTH_R_M, subdivide: int = 1,
               coast_cut: bool = False) -> Dict[str, np.ndarray]:
    """DEM → 网格 dict（verts / faces / uvs / normals）。

    mode='plane'：周期平面浮雕。网格列覆盖经度 -180..+180 含折缝列
                  （w+1 列，折缝列 = 首列），X=经度·111.32km、Y=纬度·111.32km、
                  Z=高程×exagg，跨 ±180° 子午线的地形在左右边缘无缝衔接。
    mode='globe'：半径 R=r0+高程×exagg 的球坐标，附带南/北极冠补面
                  （极顶高程 = 所在环行均值）。
    subdivide>1：先对 DEM 做样条平滑加密（每轴向 ×subdivide，面数约 ×subdivide²），
                 输出更平滑细密的网格；纹理 UV/法线随之在加密网格上重建。
    coast_cut=True：将海洋（≤0）钳位到海平面，使海陆交界成为锋利边缘，
                 与 2D 地图 0 等高线贴合（平面→z=0 平面；球体→r0 球面）。
                 顺序固定为「先 subdivide 再裁剪」，避免样条振铃把海岸拉回负值。
    """
    dem = np.asarray(dem, np.float32)
    if subdivide and subdivide > 1:
        dem = _upsample_dem(dem, subdivide)
    if coast_cut:
        dem = np.maximum(dem, 0.0).astype(np.float32)
    h, w = dem.shape
    lats, lons = _lat_lon_grids(h, w)

    if mode == "plane":
        # 周期平面（经度无缝）：网格列覆盖 -180..+180 含「折缝列」(w+1 列)。
        # 折缝列数据 = 首列（±180 是同一条子午线），左/右边缘折叠时几何与
        # 地形完全对齐，跨经度大陆不再被硬切断。
        dem_w = np.concatenate([dem, dem[:, :1]], axis=1)      # (h, w+1)
        lon_e = -180.0 + np.arange(w + 1) * (360.0 / w)        # 边缘经度
        X, Y = np.meshgrid(lon_e * _KM_PER_DEG, lats * _KM_PER_DEG)
        Z = dem_w * exagg
        verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
        faces = _build_faces(h, w + 1, wrap=False)
        faces = _orient(verts, faces, "plane", None)
        uvs = _uvs(h, w + 1, wrap=False)
        normals = _vertex_normals(verts, faces, center=None)
    else:  # globe
        phi = np.deg2rad(lats)[:, None]
        theta = np.deg2rad(lons)[None, :]
        r = r0 + dem * exagg
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.ones_like(r)
        z = -r * np.cos(phi) * np.sin(theta)
        verts = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
        center = np.zeros(3, np.float64)
        # 极顶：顶行（i=0，南极环）与末行（i=h-1，北极环）均值高程
        r_s = r0 + float(dem[0].mean()) * exagg
        r_n = r0 + float(dem[-1].mean()) * exagg
        verts = np.vstack([verts, [0.0, -r_s, 0.0], [0.0, r_n, 0.0]]).astype(np.float32)
        faces = _build_faces(h, w, wrap=True)
        faces = np.vstack([faces,
                           _pole_faces(0, h * w, w, reverse=False),
                           _pole_faces((h - 1) * w, h * w + 1, w, reverse=True)])
        faces = _orient(verts, faces, "globe", center)
        uvs = np.vstack([_uvs(h, w, wrap=True), [0.5, 1.0], [0.5, 0.0]]).astype(np.float32)
        normals = _vertex_normals(verts, faces, center=center)

    return {"verts": verts, "faces": faces, "uvs": uvs, "normals": normals}


def _elev_texture(dem: np.ndarray, shade: bool = True,
                  detail: bool = True, seed: int = 7,
                  coastline: bool = True) -> np.ndarray:
    """DEM 高程着色纹理（H,W,3）uint8：m2v 归一化 → 地形色卡。

    shade=True 时叠加山影（坡度越陡越暗）与坡度加权低频细节纹理，
    避免大片平台/台地海拔相同 → 整片同色"大色块"。
    coastline=True 时在 0 等高线两侧 1-2px 加深描边，让海岸在 3D 上
    清晰可读（确定性：仅二值形态学，无随机扰动）。
    """
    v = _m2v(dem)
    rgb = _make_dem_cmap()(v)[..., :3]
    if shade:
        from scipy.ndimage import gaussian_filter
        gy, gx = np.gradient(dem.astype(np.float64))
        sl = np.hypot(gy, gx)                 # 米/px 坡度
        sh = 1.0 / (1.0 + np.clip(sl / 60.0, 0.0, None) ** 1.3)
        if detail:
            rng = np.random.RandomState(seed)
            n = gaussian_filter(rng.uniform(size=dem.shape).astype(np.float32),
                                sigma=2.0).astype(np.float64)
            n = (n - n.mean()) / (n.std() + 1e-6)
            w = np.clip(sl / 160.0, 0.0, 1.0)   # 越陡细节越强，平原保持干净
            sh = sh * (1.0 + 0.10 * n * w)
        sh = np.clip(sh, 0.35, 1.25)
        rgb = np.clip(rgb * sh[..., None], 0.0, 1.0)
    if coastline:
        from scipy.ndimage import binary_dilation, binary_erosion
        land = dem >= 0
        er = binary_erosion(land)
        dl = binary_dilation(land, iterations=2)
        edge = (land & ~er) | (dl & ~land)      # 0 等高线两侧 1-2px 环带
        rgb[edge] = rgb[edge] * 0.55
    return (rgb * 255.0).astype(np.uint8)


def _resize_to(dem: np.ndarray, n: int) -> np.ndarray:
    """线性插值缩放 DEM 至 (n, n)（n == 原尺寸时原样返回）。"""
    dem = np.asarray(dem, np.float32)
    if dem.shape[0] == n and dem.shape[1] == n:
        return dem
    from scipy.ndimage import zoom
    f = n / dem.shape[0]
    return zoom(dem, (f, f), order=1).astype(np.float32)


def _load_map_texture(path: str, tex_size: int) -> np.ndarray:
    """加载 2D 渲染图（行 0 = 南极，与 3D UV 对齐）→ RGB uint8，必要时缩到 tex_size。"""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if tex_size and (im.width != tex_size or im.height != tex_size):
        im = im.resize((tex_size, tex_size), Image.BILINEAR)
    return np.asarray(im, np.uint8)


def resolve_map_png(dem_path: Optional[str]) -> Optional[str]:
    """按 DEM 路径找配套 2D 渲染图（同前缀 priority，兜底 world_map.png / 最新 seed*_map.png）。"""
    if not dem_path:
        return None
    out_dir = os.path.dirname(os.path.abspath(dem_path)) or "."
    stem = os.path.splitext(os.path.basename(dem_path))[0]   # world_dem / seed15_dem ...
    base = stem[:-4] if stem.endswith("_dem") else stem       # world / seed15 ...
    for cand in (f"{stem}_map.png", f"{base}_map.png", "world_map.png"):
        p = os.path.join(out_dir, cand)
        if os.path.exists(p):
            return p
    newest = sorted(glob.glob(os.path.join(out_dir, "seed*_map.png")),
                    key=os.path.getmtime, reverse=True)
    return newest[0] if newest else None


# ============================================================================
# OBJ 导出（.obj + .mtl + 纹理 PNG）
# ============================================================================

def write_obj(prefix: str, mesh: Dict[str, np.ndarray], tex: np.ndarray) -> Tuple[str, str, str]:
    """写 ``<prefix>.obj`` + ``<prefix>.mtl`` + ``<prefix>_tex.png``，返回三个路径。"""
    from PIL import Image
    verts, faces, uvs = mesh["verts"], mesh["faces"], mesh["uvs"]
    obj_path, mtl_path = prefix + ".obj", prefix + ".mtl"
    tex_name = os.path.basename(prefix) + "_tex.png"
    tex_path = os.path.join(os.path.dirname(os.path.abspath(prefix)) or ".", tex_name)
    Image.fromarray(tex).save(tex_path)

    lines = [
        "# Fantasy World Simulator export_3d",
        f"mtllib {os.path.basename(mtl_path)}",
        "o world",
    ]
    v, vt, vn = verts, uvs, mesh["normals"]
    for row in zip(v, vt, vn):
        lines.append("v %.6f %.6f %.6f" % (row[0][0], row[0][1], row[0][2]))
        lines.append("vt %.6f %.6f" % (row[1][0], row[1][1]))
        lines.append("vn %.6f %.6f %.6f" % (row[2][0], row[2][1], row[2][2]))
    lines.append("usemtl dem")
    for f in faces:
        lines.append("f %d/%d/%d %d/%d/%d %d/%d/%d" % (
            f[0] + 1, f[0] + 1, f[0] + 1,
            f[1] + 1, f[1] + 1, f[1] + 1,
            f[2] + 1, f[2] + 1, f[2] + 1))
    with open(obj_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    mtl = (
        "newmtl dem\n"
        "Ka 1.0 1.0 1.0\n"
        "Kd 1.0 1.0 1.0\n"
        "Ks 0.0 0.0 0.0\n"
        "d 1.0\n"
        "illum 2\n"
        f"map_Kd {tex_name}\n"
    )
    with open(mtl_path, "w") as fh:
        fh.write(mtl)
    return obj_path, mtl_path, tex_path


# ============================================================================
# glTF 2.0 导出（自包含单文件，纹理 base64 内嵌）
# ============================================================================

def write_gltf(path: str, mesh: Dict[str, np.ndarray], tex: np.ndarray,
               wrap_u: bool = True) -> str:
    """写自包含 glTF 2.0 单文件（POSITION/NORMAL/TEXCOORD_0 + uint32 indices + base64 PNG 纹理）。"""
    from PIL import Image
    verts, faces, uvs, normals = (mesh["verts"], mesh["faces"],
                                  mesh["uvs"], mesh["normals"])
    V, F = verts.shape[0], faces.shape[0]

    pos = np.ascontiguousarray(verts, dtype="<f4").tobytes()
    nrm = np.ascontiguousarray(normals, dtype="<f4").tobytes()
    uv = np.ascontiguousarray(uvs, dtype="<f4").tobytes()
    idx = np.ascontiguousarray(faces, dtype="<u4").tobytes()
    buf_parts = [pos, nrm, uv, idx]
    offsets: list = []
    off = 0
    for p in buf_parts:
        offsets.append(off)
        off += len(p)
    img_buf = io.BytesIO()
    Image.fromarray(tex).save(img_buf, format="PNG")
    png_bytes = img_buf.getvalue()
    offsets.append(off)  # image bufferView 起点
    total = off + len(png_bytes)

    pos_min = [float(verts[:, 0].min()), float(verts[:, 1].min()), float(verts[:, 2].min())]
    pos_max = [float(verts[:, 0].max()), float(verts[:, 1].max()), float(verts[:, 2].max())]

    gltf = {
        "asset": {"version": "2.0", "generator": "FantasyWorldSimulator/export_3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "world"}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3, "material": 0, "mode": 4}]}],
        "materials": [{
            "name": "dem", "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0, "roughnessFactor": 1.0,
            }}],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9729,
                      "wrapS": _GLTF_REPEAT if wrap_u else _GLTF_CLAMP_TO_EDGE,
                      "wrapT": _GLTF_CLAMP_TO_EDGE}],
        "images": [{"bufferView": 4, "mimeType": "image/png", "name": "tex"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": V, "type": "VEC3",
             "min": pos_min, "max": pos_max},
            {"bufferView": 1, "componentType": 5126, "count": V, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": V, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5125, "count": F, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(pos)},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(nrm)},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(uv)},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(idx),
             "target": 34963},
            {"buffer": 0, "byteOffset": offsets[4], "byteLength": len(png_bytes),
             "target": 34962},
        ],
        "buffers": [{"byteLength": total,
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(
                         pos + nrm + uv + idx + png_bytes).decode("ascii")}],
    }
    with open(path, "w") as fh:
        json.dump(gltf, fh)
    return path


# ============================================================================
# 桌面预览（PyVista/VTK GPU 渲染优先，matplotlib 回退）
# ============================================================================

def _face_colors(tri: np.ndarray, tri_uv: np.ndarray,
                 tex: np.ndarray) -> np.ndarray:
    """每面颜色 = 质心 UV 最近 texel。"""
    th, tw = tex.shape[:2]
    cu = np.clip(tri_uv.mean(axis=1)[:, 0] * tw - 0.5, 0, tw - 1).astype(int)
    cv = np.clip(tri_uv.mean(axis=1)[:, 1] * th - 0.5, 0, th - 1).astype(int)
    return tex[cv, cu] / 255.0


def _draw_mesh(ax, mesh: Dict[str, np.ndarray], tex: np.ndarray) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tri = mesh["verts"][mesh["faces"]]
    tri_uv = mesh["uvs"][mesh["faces"]]
    pc = Poly3DCollection(tri, facecolors=_face_colors(tri, tri_uv, tex),
                          edgecolors="none", linewidth=0)
    ax.add_collection3d(pc)
    v = mesh["verts"]
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())
    ax.set_box_aspect((1, 1, 1))


def _preview_matplotlib(dems: Dict[str, np.ndarray], texs: Dict[str, np.ndarray],
                        exagg_plane, exagg_globe,
                        r0: float = _EARTH_R_M,
                        coast_cut: bool = True) -> None:
    """matplotlib 回退预览（无 PyVista 时）。headless 环境自动降级为打印提示。"""
    import matplotlib
    matplotlib.use("MacOSX")  # 交互后端；不可用则回退默认
    import matplotlib.pyplot as plt
    modes = list(dems)
    fig = plt.figure(figsize=(7.5 * len(modes), 6.5))
    for k, md in enumerate(modes):
        dem = dems[md]
        exagg = _resolve_exagg(exagg_plane if md == "plane" else exagg_globe, dem, md, r0)
        mesh = build_mesh(dem, md, exagg, r0, coast_cut=coast_cut)
        ax = fig.add_subplot(1, len(modes), k + 1, projection="3d")
        tex = texs.get(md)
        if tex is None:
            tex = _elev_texture(dem)
        _draw_mesh(ax, mesh, tex)
        ax.set_title(f"mode={md}")
        ax.view_init(elev=45, azim=-60)   # 明显斜视角：露出陆地起伏
    try:
        plt.show()
    except Exception as e:  # noqa: BLE001
        print(f"[preview] 显示窗口失败（{e}）—— 无 GUI 会话？产物已导出，可跳过预览。", flush=True)


def _mesh_to_polydata(mesh: Dict[str, np.ndarray]):
    """mesh dict → pyvista.PolyData。

    VTK faces 布局为 [n, i0, i1, i2, ...]；tcoords 直接复用网格 UV。
    纹理数组行 0=南极，与 numpy/图像语义同向（v=0 ↔ 末行），无需翻转。
    """
    import pyvista as pv
    faces = mesh["faces"].astype(np.int64)
    f = np.empty(faces.shape[0] * 4, np.int64)
    f[0::4] = 3
    f[1::4] = faces[:, 0]
    f[2::4] = faces[:, 1]
    f[3::4] = faces[:, 2]
    pd = pv.PolyData(mesh["verts"].astype(np.float64), f)
    pd.active_texture_coordinates = mesh["uvs"].astype(np.float32)
    return pd


def _preview_pyvista(dems: Dict[str, np.ndarray], texs: Dict[str, np.ndarray],
                     exagg_plane, exagg_globe,
                     r0: float = _EARTH_R_M,
                     coast_cut: bool = True, subdivide: int = 2) -> None:
    """桌面窗口预览（PyVista/VTK，GPU 渲染）。单窗口多面板，平面顶视/球体等轴。

    coast_cut=True 时预览默认海平面裁剪，海岸锋利贴合 2D 地图 0 等高线。
    subdivide=2 对预览网格做样条平滑加密（≈1024² 顶点/约 2M 三角），
    使大陆轮廓与起伏细节不再呈块状马赛克。
    """
    import pyvista as pv
    pv.set_plot_theme("dark")
    pv.global_theme.multi_samples = 8          # MSAA：抗锯齿（幂等）
    pv.global_theme.background = "black"
    modes = list(dems)
    pl = pv.Plotter(shape=(1, len(modes)), window_size=[780 * len(modes), 680])
    # 定向"太阳光"（NW 上方）：让地形起伏被光影刻画（旧版本无 Light 则跳过）
    try:
        pl.add_light(pv.Light(position=(-0.9, 1.0, 0.45), focal_point=(0.0, 0.0, 0.0),
                              intensity=1.2, light_type="scene light"))
    except Exception:  # noqa: BLE001  旧版 pyvista 缺少 Light/add_light
        pass
    for k, md in enumerate(modes):
        pl.subplot(0, k)
        dem = dems[md]
        exagg = _resolve_exagg(exagg_plane if md == "plane" else exagg_globe, dem, md, r0)
        mesh = build_mesh(dem, md, exagg, r0, subdivide=subdivide, coast_cut=coast_cut)
        tex = texs.get(md)
        if tex is None:
            tex = _elev_texture(dem)
        pd = _mesh_to_polydata(mesh)
        texture = pv.Texture(tex) if tex is not None else None
        if md == "globe" and texture is not None:
            texture.wrap = pv.Texture.WrapType.REPEAT   # 经度无缝
        pl.add_mesh(pd, texture=texture, smooth_shading=True,
                    show_edges=False, lighting=True,
                    ambient=0.28, diffuse=0.75, specular=0.0, name=md)
        pl.add_text(f"mode={md} (exagg={exagg:.1f})", font_size=10)
        if md == "plane":
            pl.view_xy()
            pl.camera.zoom(1.5)                       # 平面长宽比大，拉近填满画面
            try:
                pl.camera.Elevation(30)               # 明显斜视角：露出陆地起伏
            except Exception:  # noqa: BLE001
                pass
        else:
            pl.view_isometric()
    pl.show()


def preview_desktop(dems: Dict[str, np.ndarray], texs: Dict[str, np.ndarray],
                    exagg_plane, exagg_globe,
                    r0: float = _EARTH_R_M,
                    coast_cut: bool = True,
                    preview_subdivide: int = 2) -> None:
    """桌面窗口预览：优先 PyVista（GPU，旋转流畅）；缺失或失败时回退 matplotlib。

    coast_cut=True：预览默认海平面裁剪（海岸锋利贴合）。
    preview_subdivide：预览样条加密倍数，仅 PyVista 生效；matplotlib 回退
    强制 1（Poly3DCollection 性能上限）。
    """
    exagg_plane = "auto" if exagg_plane is None else exagg_plane
    exagg_globe = "auto" if exagg_globe is None else exagg_globe
    try:
        import pyvista  # noqa: F401
    except ImportError:
        _preview_matplotlib(dems, texs, exagg_plane, exagg_globe, r0, coast_cut)
        return
    try:
        _preview_pyvista(dems, texs, exagg_plane, exagg_globe, r0,
                         coast_cut, subdivide=preview_subdivide)
    except Exception as e:  # noqa: BLE001
        print(f"[preview] PyVista 预览失败（{e}）—— 回退 matplotlib。", flush=True)
        _preview_matplotlib(dems, texs, exagg_plane, exagg_globe, r0, coast_cut)


# ============================================================================
# 编排与 CLI
# ============================================================================

def export_world(dem_m: np.ndarray, out_dir: str, mode: str = "both",
                 export: Sequence[str] = ("obj", "gltf"), decimate: int = 0,
                 exagg_plane="auto", exagg_globe="auto",
                 preview: bool = False, preview_decimate: int = 0,
                 r0: float = _EARTH_R_M, verbose: bool = True,
                 texture: str = "auto", tex_size: int = 0,
                 dem_path: Optional[str] = None,
                 subdivide: int = 1,
                 coast_cut: bool = False,
                 preview_coast_cut: bool = True,
                 preview_subdivide: int = 2) -> list:
    """主编排：抽稀 → 每形态构建网格 → 导出 OBJ/glTF →（可选）桌面预览。

    decimate<=0 时自适应：DEM≤2048 用 2（约 1024² 网格），4096 用 4（1024²），
    保证精细度的同时控制文件体积。纹理(默认全分辨率 2048²)与网格抽稀独立。

    exagg_plane/exagg_globe：垂直夸张，"auto" 按地图幅宽/半径自适应
    （起伏约占幅宽 10% / 半径 15%，远景下清晰可辨）；数值则原样透传。
    coast_cut：导出网格是否海平面裁剪（True → 海洋压平为 z=0 / r0 球面，
    海岸锋利贴合 0 等高线，但海底起伏在 3D 几何上消失）。默认 False 保留海底。
    preview_coast_cut：预览窗口是否默认海平面裁剪（默认 True）。
    preview_subdivide：预览网格样条加密倍数（默认 2，PyVista 生效；
    matplotlib 回退强制 1）。

    texture: 'auto'（优先 2D 渲染图，缺则高程着色）| 'map'（强制渲染图）|
             'elevation'（高程着色+山影）。
    dem_path: 用于定位配套渲染图；缺省时按 out_dir 内 world_map.png / 最新 seed*_map.png。
    subdivide: 导出细分加密倍数（1=不细分；>1 时对抽稀后 DEM 做样条平滑加密，
               每轴向 ×subdivide、面数约 ×subdivide²，输出更平滑细密的模型；
               OBJ/glTF 体积随面数近似 ×subdivide²，谨慎使用）。

    Returns: 生成的产物文件路径列表。
    """
    dem_m = np.asarray(dem_m, np.float32)
    modes = ["plane", "globe"] if mode == "both" else [mode]
    export_set = {"obj", "gltf"} if "both" in export else set(export)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    produced: list = []
    dems: Dict[str, np.ndarray] = {}
    texs: Dict[str, np.ndarray] = {}

    if decimate <= 0:
        decimate = 2 if dem_m.shape[0] <= 2048 else 4
    tex_src = min(tex_size or 2048, dem_m.shape[0])
    tex_src = max(tex_src, 256)  # 网格已抽稀时贴图也别低于 256²

    # ---- 纹理：全分辨率渲染，与网格抽稀解耦 ----
    map_png = None
    if texture in ("auto", "map"):
        map_png = resolve_map_png(dem_path) or resolve_map_png(
            os.path.join(out_dir, "world_dem.npy"))
        if not map_png and texture == "map":
            print("[export_3d] 警告：未找到 2D 渲染图（--texture map），降级为高程着色。")
    if map_png:
        tex = _load_map_texture(map_png, tex_src)
    else:
        tex = _elev_texture(_resize_to(dem_m, tex_src))
    if verbose and map_png:
        print(f"[export_3d] 贴图: {os.path.basename(map_png)} ({tex.shape[0]}²)")

    for md in modes:
        t0 = time.time()
        dem_d = _decimate(dem_m, decimate)
        exagg = _resolve_exagg(exagg_plane if md == "plane" else exagg_globe,
                               dem_d, md, r0)
        mesh = build_mesh(dem_d, md, exagg, r0, subdivide=subdivide,
                          coast_cut=coast_cut)
        dems[md] = dem_d
        texs[md] = tex
        if "obj" in export_set:
            produced.extend(write_obj(os.path.join(out_dir, f"world_3d_{md}"), mesh, tex))
        if "gltf" in export_set:
            produced.append(write_gltf(
                os.path.join(out_dir, f"world_3d_{md}.gltf"), mesh, tex,
                wrap_u=(md == "globe")))
        if verbose:
            print(f"[export_3d] {md}: {dem_d.shape} 网格 (exagg={exagg:.2f}) "
                  f"→ {len(produced)} 个文件 ({time.time()-t0:.2f}s)")

    if preview:
        try:  # 预览细分档位按渲染后端分流：GPU 512² 基底 + 加密，matplotlib ≤128²
            import pyvista  # noqa: F401
            _has_pv = True
        except ImportError:
            _has_pv = False
        pdm = {}
        ptex = {}
        for md, dem in dems.items():
            cap = 512 if (_has_pv and preview_subdivide >= 1) else 128
            step = preview_decimate or max(1, int(np.sqrt(dem.size) / cap))
            pdm[md] = _decimate(dem, step)
            ptex[md] = texs[md]
        preview_desktop(pdm, ptex, exagg_plane, exagg_globe, r0,
                        coast_cut=preview_coast_cut,
                        preview_subdivide=preview_subdivide)
    return produced


def _resolve_dem(seed: Optional[int], size: int) -> Optional[str]:
    """按 geo_context._resolve_dem_path 同优先级查找 DEM 文件（不自动生成）。"""
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    _dir = os.path.dirname(_file_dir) if os.path.basename(_file_dir) == "core" else _file_dir
    out_dir = os.path.join(_dir, "output")
    if seed is not None:
        for cand in (os.path.join(out_dir, f"world_dem_{seed}_{size}.npy"),
                     os.path.join(out_dir, f"seed{seed}_dem.npy")):
            if os.path.exists(cand):
                return cand
    p3 = os.path.join(out_dir, "world_dem.npy")
    if os.path.exists(p3):
        return p3
    candidates = (glob.glob(os.path.join(out_dir, "world_dem*.npy"))
                  + glob.glob(os.path.join(out_dir, "seed*_dem.npy")))
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="DEM → OBJ + glTF 导出 / 桌面预览")
    ap.add_argument("--dem", default=None,
                    help="world_dem.npy 路径（默认按 seed/size 或 output/ 最新文件解析）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--mode", choices=["plane", "globe", "both"], default="both")
    ap.add_argument("--export", choices=["obj", "gltf", "both", "none"], default="both")
    ap.add_argument("--decimate", type=int, default=0,
                    help="网格抽稀倍数（0=自适应：DEM≤2048 用 2，4096 用 4；"
                         "--decimate 1 输出全分辨率）")
    ap.add_argument("--subdivide", type=int, default=1,
                    help="网格样条平滑加密倍数（1=不加密；2=每 quad 拆 4 子 quad，"
                         "面数约 ×4；OBJ/glTF 体积随之近似 ×4，谨慎使用）")
    ap.add_argument("--texture", choices=["auto", "map", "elevation"], default="auto",
                    help="表面贴图来源（见模块 docstring；auto 默认优先 2D 渲染图）")
    ap.add_argument("--tex-size", type=int, default=0,
                    help="贴图边长（0=自动 min(DEM, 2048)，与网格抽稀解耦）")
    ap.add_argument("--exagg-plane", type=str, default="auto",
                    help="平面垂直夸张（'auto' 按幅宽自适应，数字原样生效）")
    ap.add_argument("--exagg-globe", type=str, default="auto",
                    help="球体高程夸张（'auto' 按半径自适应，数字原样生效）")
    ap.add_argument("--preview", action="store_true", help="打开桌面窗口预览")
    ap.add_argument("--preview-decimate", type=int, default=0,
                    help="预览专用抽稀（默认 GPU ≤512² 基底 / matplotlib ≤128²）")
    ap.add_argument("--preview-subdivide", type=int, default=2,
                    help="预览网格样条加密倍数（默认 2；matplotlib 回退强制 1）")
    ap.add_argument("--coast-cut", action="store_true",
                    help="导出网格海平面裁剪：海洋压平为海平面，海岸锋利贴合"
                         "（默认 False 保留海底起伏；预览默认开启平海）")
    ap.add_argument("--no-preview-coast", action="store_true",
                    help="关闭预览窗口的海平面裁剪（默认开启平海贴合海岸线）")
    ap.add_argument("--out", default=None, help="输出目录（默认 terrain/output）")
    a = ap.parse_args()

    if a.dem:
        dem_path = a.dem
    else:
        dem_path = _resolve_dem(a.seed, a.size)
    if not dem_path or not os.path.exists(dem_path):
        print("找不到 DEM 文件。请先运行："
              "`python -m terrain.core.generate_world --seed N --size S` 或显式 --dem 指定。")
        raise SystemExit(1)
    dem = np.load(dem_path).astype(np.float32)
    if dem.ndim != 2 or dem.shape[0] != dem.shape[1]:
        raise ValueError(f"DEM {dem_path} 不是二维方阵：shape={dem.shape}")
    dec = a.decimate if a.decimate >= 1 else (2 if dem.shape[0] <= 2048 else 4)
    grid = dem.shape[0] // dec
    out_v = (grid * a.subdivide) ** 2
    if out_v >= (2048 ** 2):
        print(f"[export_3d] 警告：有效网格 {grid}² × --subdivide {a.subdivide} "
              f"→ {out_v} 顶点，OBJ 可能 >1GB，建议增大 --decimate 或减小 --subdivide。")

    out_dir = a.out or os.path.dirname(os.path.abspath(dem_path))
    export_world(dem, out_dir, mode=a.mode, export=(a.export,),
                 decimate=a.decimate, subdivide=a.subdivide,
                 exagg_plane=a.exagg_plane,
                 exagg_globe=a.exagg_globe, preview=a.preview,
                 preview_decimate=a.preview_decimate,
                 preview_subdivide=a.preview_subdivide,
                 coast_cut=a.coast_cut,
                 preview_coast_cut=(not a.no_preview_coast),
                 texture=a.texture, tex_size=a.tex_size, dem_path=dem_path)
    print(f"[export_3d] 完成 → {out_dir}")


if __name__ == "__main__":
    main()
