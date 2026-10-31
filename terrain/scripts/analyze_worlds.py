# -*- coding: utf-8 -*-
# ======================================================================
# analyze_worlds.py —— 多种子生成图 vs 真实地球 七大地理维度统一指标
# 真实地球: ETOPO1(Ice) 降采样 2048×2048 + PB2002 构造场 (t1=俯冲/碰撞)
# 运行: python analyze_worlds.py   → 输出 output/analysis_results.json + 汇总表
# ======================================================================
import os
import sys
import json
import numpy as np
from scipy.ndimage import label as ndi_label, zoom as ndi_zoom
from scipy.io.netcdf import netcdf_file

# 允许直接 `python terrain/scripts/analyze_worlds.py` 运行：把项目根加入 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from terrain.core.terrain_engine import (m2v, snowline_m, d8_flow_accumulation,
                            fill_depressions, temperature_field,
                            precipitation_field, whittaker_biome,
                            periodic_fbm, warp_lonlat, _psd_slope,
                            _fractal_dim, ocean_current_effects)

_file_dir = os.path.dirname(os.path.abspath(__file__))
_dir = (os.path.dirname(_file_dir)
        if os.path.basename(_file_dir) in ("core", "scripts") else _file_dir)
S = 2048
LATS = (-90.0 + (np.arange(S) + 0.5) * (180.0 / S)).astype(np.float64)
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------- 工具 ----------
def fill_pd(v, ocean, eps=1e-6):
    """M6 Planchon-Darboux+ε 填洼: 无 eps 量化噪声的优先队列填洼。
    海洋像素即种子 (f=原值); 陆地抬升至能沿严格递增路径排入海。
    旧迭代填洼的 eps=4e-4(v域) 在 200-1000m 带产生 ~3.8m/px 伪梯度,
    淹没 M5 盆地倾斜面 (0.9m/px) → 流域指标失真; ε=1e-6 ≈ 0.01m/px。"""
    import heapq
    H, W = v.shape
    f = v.astype(np.float64).copy()
    coast = ocean & (~np.roll(ocean, 1, 0) | ~np.roll(ocean, -1, 0)
                     | ~np.roll(ocean, 1, 1) | ~np.roll(ocean, -1, 1))
    visited = ocean & ~coast                     # 海洋免处理 (陆流才计入指标),
    heap = []                                    # 海岸种子需弹出以扩展陆地邻居
    cand = np.full((H, W), np.inf)               # 候选水位 (首候选=原值也须入堆)
    for i, j in zip(*np.nonzero(coast)):
        cand[i, j] = float(v[i, j])
        heapq.heappush(heap, (float(v[i, j]), int(i), int(j)))
    D8N = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while heap:
        z, i, j = heapq.heappop(heap)
        if visited[i, j]:
            continue
        visited[i, j] = True
        f[i, j] = z                              # 弹出时即最终水位 (最小候选)
        for di, dj in D8N:
            ii, jj = i + di, j + dj
            if 0 <= ii < H and 0 <= jj < W and not visited[ii, jj]:
                zn = max(v[ii, jj], z + eps)
                if zn < cand[ii, jj]:
                    cand[ii, jj] = zn
                    heapq.heappush(heap, (zn, ii, jj))
    return f.astype(np.float32)


def label_periodic(mask):
    """连通域标注 (经度周期环绕合并)"""
    lab, n = ndi_label(mask)
    if n == 0:
        return lab, 0
    parent = list(range(n + 1))

    def find(a):
        """并查集查找根节点（带路径压缩）。"""
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    right = lab[:, -1]
    left = lab[:, 0]
    for r in np.unique(right):
        if r == 0:
            continue
        rows = np.nonzero(right == r)[0]
        ls = left[rows]
        ls = np.unique(ls[ls > 0])
        for l in ls:
            parent[find(int(r))] = find(int(l))
    lut = np.array([find(i) for i in range(n + 1)])
    lab2 = lut[lab]
    vals = np.unique(lab2)
    vals = vals[vals > 0]
    remap = np.zeros(lab2.max() + 1, np.int32)
    remap[vals] = np.arange(1, len(vals) + 1)
    return remap[lab2], len(vals)


def receivers(elev, rng):
    """D8 接收者索引 (-1=无/海), 与 d8_flow_accumulation 同逻辑"""
    h, w = elev.shape
    e = elev.astype(np.float32) + rng.uniform(0, 1e-4, (h, w)).astype(np.float32)
    best_s = np.full((h, w), 1e-9, np.float32)
    recv = np.full((h, w), -1, np.int32)
    for dy, dx in D8:
        dd = (dy * dy + dx * dx) ** 0.5
        shift = np.full((h, w), np.inf, np.float32)
        ys, xs = slice(max(dy, 0), h + min(dy, 0)), slice(max(dx, 0), w + min(dx, 0))
        yt, xt = slice(max(-dy, 0), h + min(-dy, 0)), slice(max(-dx, 0), w + min(-dx, 0))
        shift[yt, xt] = e[ys, xs]
        s = (e - shift) / dd
        m = s > best_s
        if m.any():
            yy, xx = np.nonzero(m)
            recv[yy, xx] = ((yy + dy) * w + (xx + dx)).astype(np.int32)
            best_s[m] = s[m]
    return recv


def block_mean_to(a, ny, nx):
    """任意尺寸栅格 → ny×nx 块均值 (行/列 reduceat)"""
    a = a.astype(np.float64)
    def red(x, nbins, axis):
        """沿指定轴把 x 均分成 nbins 段并做块均值。"""
        b = np.linspace(0, x.shape[axis], nbins + 1).round().astype(int)
        s = np.add.reduceat(x, b[:-1], axis=axis)
        c = np.diff(b).astype(np.float64)
        sh = [1] * x.ndim
        sh[axis] = nbins
        return s / c.reshape(sh)
    a = red(a, ny, 0)
    a = red(a, nx, 1)
    return a.astype(np.float32)


def load_earth():
    """ETOPO1(Ice) → 2048×2048 米制 DEM (南→北, 经度周期)"""
    path = os.path.join(_dir, "data", "ETOPO1_Ice_g_gmt4.grd")
    f = netcdf_file(path, "r", mmap=False)
    zv = f.variables["z"]
    z = np.array(zv.data)
    scale = float(np.atleast_1d(getattr(zv, "scale_factor", 1.0))[0])
    off = float(np.atleast_1d(getattr(zv, "add_offset", 0.0))[0])
    if z.shape[0] > z.shape[1]:
        z = z.T
    if "y" in f.variables:
        y = np.array(f.variables["y"][:], np.float64).ravel()
        if y[0] > y[-1]:
            z = z[::-1]
    f.close()
    if z.shape[1] % 21600 == 1:          # 0..360 重复端点列
        z = z[:, :-1]
    dem = block_mean_to(z * scale + off, S, S)
    return dem


def load_earth_conv():
    """PB2002 俯冲边界线 → 距离场 exp(-(d/300km)²), 2048×2048 (经度周期)"""
    from scipy.ndimage import distance_transform_edt
    js = json.load(open(os.path.join(_dir, "data", "PB2002_boundaries.json")))
    pts = []

    def add_line(coords):
        """把一条折线按经度解绕 + 加密采样为点列，追加到 pts。"""
        co = np.asarray(coords, np.float64)
        if len(co) < 2:
            return
        lon, lat = co[:, 0].copy(), co[:, 1]
        for i in range(1, len(lon)):
            d = lon[i] - lon[i - 1]
            if d > 180:
                lon[i:] -= 360
            elif d < -180:
                lon[i:] += 360
        step = np.abs(np.diff(lon)) + np.abs(np.diff(lat))
        n = np.maximum((step / 0.12).astype(int), 1)
        for i in range(len(lon) - 1):
            t = np.linspace(0, 1, n[i] + 1)
            pts.append(np.stack([lat[i] + (lat[i + 1] - lat[i]) * t,
                                 lon[i] + (lon[i + 1] - lon[i]) * t], 1))

    for f in js.get("features", []):
        if (f.get("properties") or {}).get("Type") != "subduction":
            continue
        g = f.get("geometry") or {}
        if g.get("type") == "LineString":
            add_line(g["coordinates"])
        elif g.get("type") == "MultiLineString":
            for ln in g["coordinates"]:
                add_line(ln)
    P = np.concatenate(pts, 0)
    mask = np.zeros((S, S), bool)
    rr = np.clip(((90.0 - P[:, 0]) / 180.0 * S).astype(int), 0, S - 1)
    cc = np.mod(((P[:, 1] + 180.0) / 360.0 * S).astype(int), S)
    mask[rr, cc] = True
    tiled = np.tile(mask, (1, 3))                       # 经度周期
    d_px = distance_transform_edt(~tiled, sampling=(180.0 / S, 360.0 / S))[:, S:2 * S]
    d_km = d_px * 111.32
    return np.exp(-(d_km / 300.0) ** 2).astype(np.float32)


def sim_conv(seed):
    """复现 generate_world 的域扭曲, 取 t1 俯冲/碰撞场"""
    d = np.load(os.path.join(_dir, "cache", f"plate_sim_v3_T512_seed{seed}.npz"))
    tect = d["tect"]
    rng = np.random.RandomState(seed + 5)
    wxT = periodic_fbm(512, 6.0, 3, rng) * 16.0 + periodic_fbm(512, 20.0, 2, rng) * 4.0
    wyT = periodic_fbm(512, 6.0, 3, rng) * 16.0 + periodic_fbm(512, 20.0, 2, rng) * 4.0
    c = warp_lonlat(tect[1].astype(np.float32), wxT, wyT, 1.0)
    return ndi_zoom(c, S / 512, order=1)[:S, :S]


# ---------- 主分析 ----------
def analyze(name, dem, conv):
    """对单个世界（name 标识）计算七大地理维度统一指标，返回指标字典 r。

    覆盖: 基础/尺度、地形地貌、海岸与岛屿、海洋地理、水系水文、
    气候-植被、板块构造耦合（conv 为俯冲/碰撞距离场，可为 None）。"""
    lats = LATS
    coslat = np.clip(np.cos(np.deg2rad(lats)), 0.0, 1.0)
    Wm = coslat[:, None].astype(np.float64)
    v01 = m2v(dem)
    land = dem > 0
    ocean = ~land
    r = {"name": name}

    # --- 基础/尺度 ---
    land_frac = float((land * Wm).sum() / (Wm.sum() * S))
    r["land_frac"] = land_frac
    r["mean_land_elev"] = float((dem * Wm)[land].sum() / (land * Wm).sum())
    r["mean_ocean_depth"] = float((dem * Wm)[ocean].sum() / (ocean * Wm).sum())
    r["peak"] = float(dem.max())
    r["deepest"] = float(dem.min())
    r["nh_land_share"] = float((land[:S // 2] * Wm[:S // 2]).sum()
                               / (land * Wm).sum())

    # --- 地形地貌 ---
    dy_m = 111320.0 * 180.0 / S
    dx_m = (111320.0 * 360.0 / S) * coslat
    gy, gx = np.gradient(dem.astype(np.float64))
    slope = np.rad2deg(np.arctan(np.hypot(gx / dx_m[:, None], gy / dy_m)))
    r["mean_land_slope_deg"] = float(slope[land].mean())
    r["psd_slope"] = _psd_slope(v01)
    bands = [(0, 200), (200, 1000), (1000, 2000), (2000, 3500), (3500, 9000)]
    for lo, hi in bands:
        m = land & (dem >= lo) & (dem < hi)
        r[f"land_{lo}_{hi}"] = float((m * Wm).sum() / (land * Wm).sum())
    plat = land & (dem > 1000) & (slope < 1.0)
    r["plateau_share"] = float((plat * Wm).sum() / (land * Wm).sum())
    # 山地带状性: >2000m 连通域 PCA 长宽比
    mtn = land & (dem > 2000)
    if mtn.any():
        lab, n = label_periodic(mtn)
        tot = float(mtn.sum())
        belt_a = 0.0
        ars, areas = [], []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            a = ys.size
            if a < 30:
                continue
            pts = np.stack([ys - ys.mean(), xs - xs.mean()])
            lam = np.linalg.eigvalsh(np.cov(pts))
            ar = float(np.sqrt(max(lam[1], 1e-9) / max(lam[0], 1e-9)))
            ars.append(ar)
            areas.append(a)
            if ar > 3.0:
                belt_a += a
        r["mtn_area_share_of_land"] = float((mtn * Wm).sum() / (land * Wm).sum())
        r["belt_share_of_mtn"] = belt_a / tot
        r["mtn_mean_AR"] = float(np.average(ars, weights=areas)) if ars else 0.0
        r["n_mtn_ranges"] = int(np.sum(np.array(areas) >= 100))
    # 海平面附近过渡带 (双峰性: 越小说明海陆二元越分明)
    trans = (dem >= -300) & (dem <= 300)
    r["sealevel_band_±300m"] = float((trans * Wm).sum() / (Wm.sum() * S))

    # --- 海岸与岛屿 ---
    oc_roll = land & (np.roll(ocean, 1, 0) | np.roll(ocean, -1, 0)
                      | np.roll(ocean, 1, 1) | np.roll(ocean, -1, 1))
    coast_len_km = float((oc_roll * dx_m[:, None]).sum() / 1000.0
                         + oc_roll.sum() * dy_m / 1000.0) / 2.0
    land_area_km2 = float((land * (dx_m[:, None] * dy_m)).sum() / 1e6)
    r["coast_density_m_per_km2"] = coast_len_km * 1000.0 / max(land_area_km2, 1e-9)
    r["coast_fd"] = _fractal_dim(v01)
    lab, n = label_periodic(land)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    shares = np.array([(lab == i).sum() for i in range(1, n + 1)]) / max(land.sum(), 1)
    cont = shares[shares > 0.01]
    r["n_continents"] = int(len(cont))
    r["largest_cont_share"] = float(shares.max())
    r["island_area_share"] = float(shares[shares <= 0.01].sum())
    px_area = (180.0 / S) * (360.0 / S) * 111.32 ** 2        # 赤道像素面积 km²
    isles = shares[(shares > 0.01 / 500) & (shares <= 0.01)]
    r["n_islands_gt2000km2"] = int(np.sum(isles * land.sum() * px_area > 2000))
    # 次大陆/大岛 (含在大陆成分内): 最大岛 share (非大陆成分中最大)
    r["largest_island_share"] = float(shares[shares <= 0.01].max()) if (shares <= 0.01).any() else 0.0

    # --- 海洋地理 ---
    obands = [(-200, 0), (-1000, -200), (-4000, -1000), (-6000, -4000), (-12000, -6000)]
    for lo, hi in obands:
        m = ocean & (dem >= lo) & (dem < hi)
        r[f"ocean_{lo}_{hi}"] = float((m * Wm).sum() / (ocean * Wm).sum())
    olab, on = label_periodic(ocean)
    osz = np.bincount(olab.ravel())
    osz[0] = 0
    osh = osz / max(ocean.sum(), 1)
    r["n_oceans_gt5pct"] = int(np.sum(osh > 0.05))
    r["largest_ocean_share"] = float(osh.max())

    # --- 水系水文 ---
    # 湖泊: 原始 v01 填洼差
    filled = fill_depressions(v01)
    lake = land & ((filled - v01) > 0.012)
    r["lake_share_of_land"] = float((lake * Wm).sum() / (land * Wm).sum())
    # 大陆尺度流域格局: 平滑表面 D8 (避免块均值海岸平原"撕碎"流域)
    # M6: PD+ε 填洼 (无量化噪声) — 旧 eps 填洼在 200-1000m 带 ~3.8m/px
    # 伪梯度淹没人工缓坡 → 流域拓扑失真
    from scipy.ndimage import gaussian_filter
    v01s = gaussian_filter(v01, 6.0)
    filled_s = fill_pd(v01s, ocean)
    acc = d8_flow_accumulation(filled_s, np.random.RandomState(31))
    recv = receivers(filled_s, np.random.RandomState(31))
    landN = float(land.sum())
    ri = recv[land]
    o_mask = np.zeros(S * S, bool)
    o_mask[np.nonzero(ocean.ravel())[0]] = True
    is_out = (ri >= 0) & o_mask[ri]
    idx = np.nonzero(land)
    outlet_idx = idx[0][is_out] * S + idx[1][is_out]
    r["n_outlets"] = int(outlet_idx.size)
    if outlet_idx.size:
        accs = acc.ravel()[outlet_idx]
        r["largest_basin_share"] = float(accs.max() / landN)
        r["n_basins_gt1pct_land"] = int(np.sum(accs > 0.01 * landN))
        r["n_basins_gt190kkm2"] = int(np.sum(accs > 1000))
    else:
        r["largest_basin_share"] = 0.0
        r["n_basins_gt1pct_land"] = 0
        r["n_basins_gt190kkm2"] = 0

    # --- 气候-植被 (同一气候模型, 同口径对比; M4 洋流/季风同口径) ---
    dT, dP = ocean_current_effects(dem, lats)
    T = temperature_field(lats, dem, dT)
    P = precipitation_field(lats, dem, np.random.RandomState(7), dP)
    bi = whittaker_biome(T, P)
    for i, bn in enumerate(["苔原", "北方针叶林", "温带草原", "温带落叶林",
                            "温带雨林", "亚热带荒漠", "稀树草原",
                            "热带季节林", "热带雨林"]):
        m = land & (bi == i)
        r[f"biome_{bn}"] = float((m * Wm).sum() / (land * Wm).sum())
    ice = land & ((T <= -12.0) | (dem > snowline_m(lats)[:, None]))
    r["perm_snow_share"] = float((ice * Wm).sum() / (land * Wm).sum())
    a2 = np.abs(lats)[:, None].repeat(S, 1)
    belt = land & (a2 >= 15) & (a2 < 35)
    r["desert_in_subtropics"] = float(((P < 250) & belt).sum() / max(belt.sum(), 1))
    trop = land & (a2 < 10)
    r["rainforest_in_tropics"] = float(((bi == 8) & trop).sum() / max(trop.sum(), 1))

    # --- 板块构造: 高山与俯冲/碰撞场耦合 ---
    if conv is not None:
        cl = conv[land].astype(np.float64)
        pct = cl.argsort().argsort() / max(cl.size - 1, 1)
        mt = (dem > 2000)[land]
        r["conv_ptile_at_mountains"] = float(pct[mt].mean()) if mt.any() else 0.5
        r["corr_conv_elev_land"] = float(np.corrcoef(cl, dem[land].astype(np.float64))[0, 1])
    return r


def main():
    """入口: 分析真实地球 + seed1..10 生成图, 写 analysis_results.json, 打印汇总表。"""
    results = {}
    print("加载真实地球 ETOPO1 ...")
    earth = load_earth()
    conv_e = load_earth_conv()
    results["earth"] = analyze("earth", earth, conv_e)
    print("earth 完成")
    for seed in range(1, 11):
        p = os.path.join(_dir, "output", f"seed{seed}_dem.npy")
        if not os.path.exists(p):
            print(f"seed{seed} 缺失, 跳过")
            continue
        dem = np.load(p).astype(np.float32)
        try:
            conv = sim_conv(seed)
        except Exception:
            conv = None
        results[f"seed{seed}"] = analyze(f"seed{seed}", dem, conv)
        print(f"seed{seed} 完成")

    out = os.path.join(_dir, "output", "analysis_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # ---- 汇总表: 各指标 earth | seeds 均值±std ----
    keys = [k for k in results["earth"] if k != "name"]
    print(f"\n{'指标':<28}{'真实地球':>14}{'生成图均值':>16}{'生成图±':>12}  评价")
    for k in keys:
        e = results["earth"][k]
        vs = [results[s][k] for s in results if s != "earth" and k in results[s]]
        if not vs:
            continue
        m, sd = float(np.mean(vs)), float(np.std(vs))
        rel = abs(m - e) / (abs(e) + 1e-9)
        flag = "≈" if rel < 0.25 else ("偏差" if rel < 0.7 else "显著偏差")
        print(f"{k:<30}{e:>14.4f}{m:>16.4f}±{sd:<11.4f} {flag}")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
