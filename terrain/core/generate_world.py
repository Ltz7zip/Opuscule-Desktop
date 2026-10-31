# -*- coding: utf-8 -*-
"""
generate_world.py v9.1 —— 板块演化模拟世界生成器

=============================================================
一级: run_plate_sim v3.0 (450 Myr, 有缓存) → 粗DEM + 4构造场
      海洋直方图标定(年龄分位→地球开阔洋深度) + 陆架剖面化 + 窄海沟
二级: 12通道条件 → 逐瓦片重训版GAN → 低频对齐 → 全局CDF匹配
      → 全球D8河网 → 雪线渲染
v7.0 七大地理维度修复层（M1-M6） (基于10种子七维地理检验报告):
  P0 陆架-陆坡剖面修复 + 陆地直方图重映射(→地球实测分位)
  P1 岛弧链/热点岛链盖章 + 干流河谷雕刻
  P2 板内造山带盖章(打破山地-俯冲带过纯耦合)
v8.3 河流水系系统: 骨架提取+单调修复 / Hack 定律蜿蜒 / 牛轭湖 / 连续蓝度渲染
v9.0 监督反思: generate_world 末尾接入 supervisor 审计 → 反思 → overrides 注入
运行: python -m terrain.core.generate_world --seed 7 --size 2048
=============================================================
"""
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           zoom as ndi_zoom, label as ndi_label,
                           minimum_filter, maximum_filter,
                           median_filter, binary_closing, map_coordinates)
from .terrain_engine import (device, m2v, v2m, block_reduce, d8_flow_accumulation,
                            snowline_m, value_fbm, ridged_fbm, sharpen_coastline,
                            _lowres_fbm,
                            lowfreq_consistency,
                            polar_ice_mask, make_terrain_cmap, UNet,
                            _SL_INCISION_M, _SL_LAT,
                            temperature_field, precipitation_field,
                            whittaker_biome, smooth_biome_labels, BIOME_COLORS,
                            fill_depressions, droplet_erosion,
                            periodic_fbm, warp_lonlat,
                            land_hist_match, ocean_margin_fix, add_island_arcs,
                            add_hotspot_chains, add_intraplate_orogen,
                            carve_trunk_rivers, land_hist_final,
                            climate_flow_accumulation, paint_rivers_rgb,
                            carve_fjords, add_shelf_peninsulas,
                            add_coral_reefs, ocean_current_effects,
                            add_craton_basins, ocean_hist_final,
                            continentality_index,
                            enhance_ocean_floor)
from .plate_sim import run_plate_sim
from .cache_keys import plate_cache_path


def fbm1d(n, sigma, rng):
    """在频率域构造一维粉红噪声（fBm 近似）。

    实现：对标准正态白噪声做 rFFT → 乘以高斯带宽 ``exp(−(f·σ)²)``
    （``σ`` 越大越平滑，越低通）→ 逆 rFFT 回到空间域。
    用于海冰边缘纬度、雪线纬度波动等 **需要长程连续** 的一维曲线。

    Parameters
    ----------
    n : int
        输出长度（像素数 / 纬度行数等）。
    sigma : float
        高斯带宽的尺度因子。常见取值 80~640，越大谱越窄、越平滑。
    rng : numpy.random.Generator or RandomState
        任意可调用 ``.randn(n)`` 的随机数生成器。

    Returns
    -------
    ndarray of shape (n,), dtype float64
        零均值的一维 fBm 近似值（幅度量级约为 1σ 单位白噪声）。
    """
    x = rng.randn(n)
    X = np.fft.rfft(x)
    X *= np.exp(-(np.fft.rfftfreq(n) * sigma) ** 2)
    return np.fft.irfft(X, n)


def hydrologic_surface(v01, warp_amp=14.0, warp_freq=12.0, seed=101):
    """构造**流场专用**的填洼面：Priority-Flood 填洼 + 低频域扭曲。

    为什么要域扭曲
    --------------
    ``fill_depressions`` 的 ε 梯度是 4e-4 v/px ≈ 8 m/px，在一个被填平的大盆地
    里沿 BFS 深度累积，可以造出 **1000 m 级的人工直线坡**（本图陆地 fill 深度
    p90 = 0.07 v ≈ 1400 m）。MFD 只能沿这条人工坡走"完美直线"，几百像素的直线段
    在渲染图上就是用户说的"地面裂缝"。

    叠加 60px 波长的微起伏不够 —— 它只有 ~30 m，压不过人工坡；改成**平滑域扭曲**
    （≈14 px 位移 / 73 px 波长）把整个填洼面弯曲：真实排水骨架的拓扑不变，但原本
    笔直的河段被扰成自然河曲。

    Returns
    -------
    (warped, fill_depth) : tuple of ndarray
        ``warped`` 供 ``climate_flow_accumulation`` 使用；``fill_depth = filled - v01``
        是**填洼深度**，用于 :func:`lake` 识别与内流盆地判定（越大 = 越是大片
        被填平的洼地 = 盐沼/干湖床，不该有穿盆而过的河）。
    """
    filled = fill_depressions(v01)
    S = filled.shape[0]
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    wx = warp_amp * np.clip(value_fbm(S, warp_freq, 3, rng=np.random.RandomState(seed)), -2, 2)
    wy = warp_amp * np.clip(value_fbm(S, warp_freq, 3, rng=np.random.RandomState(seed + 1)), -2, 2)
    warped = map_coordinates(filled, [yy + wy, xx + wx], order=1, mode="nearest")
    return warped.astype(np.float32), (filled - v01).astype(np.float32)


def basin_lake_mask(comp, depth, target_px, iters=14):
    """在单个封闭盆地内求"一个"连通湖体：二分水位，使水面面积逼近 ``target_px``。

    为什么不用分位阈值
    ------------------
    直接取 ``depth > quantile(depth, q)`` 会在盆地里切出**一堆互不相连的碎斑**
    （盆地底起伏不均，等值线是破碎的）。真实湖泊是"某一水位以下的连通水体"，
    所以必须从最深点出发、只保留**含最深点的那个连通域**，再二分水位。

    ``depth`` 用填洼深度（``filled - v01``）当倒置高程：深度越大 = 地势越低。
    这样得到的岸线是天然等高线，形状与盆地地形一致，而不是刀切的斑块。

    Parameters
    ----------
    comp : ndarray (S, S) bool
        单个盆地的掩膜（``fill_depth > 阈值`` 的连通域）。
    depth : ndarray (S, S) float32
        填洼深度场。
    target_px : int
        目标水面像素数（二分终止的逼近目标）。
    iters : int
        二分次数（14 次足够把面积误差压到 1% 内）。

    Returns
    -------
    ndarray (S, S) bool
        单一连通水体掩膜；盆地为空时返回全 False。
    """
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return np.zeros(comp.shape, bool)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = comp[y0:y1, x0:x1]
    dep = depth[y0:y1, x0:x1]
    seed = np.unravel_index(int(np.argmax(np.where(sub, dep, -1.0))), dep.shape)
    lo, hi = 0.0, float(dep[seed])
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lab, _ = ndi_label(sub & (dep > mid))
        sid = lab[seed]
        if sid == 0:            # 水位低于最深点 (理论上不会), 收窄上界
            hi = mid
            continue
        m = (lab == sid)
        a = int(np.count_nonzero(m))
        if best is None or abs(a - target_px) < abs(best[0] - target_px):
            best = (a, m)
        if a < target_px:       # 水太少 → 降低水位阈值
            hi = mid
        else:
            lo = mid
    out = np.zeros(comp.shape, bool)
    if best is not None:
        out[y0:y1, x0:x1] = best[1]
    return out


def polar_despeckle(v01, lats, lat0=58.0, max_area=2500.0):
    """极区去碎：按纬度剔除过小的陆地块，让高纬大陆连成厚重陆块。

    为什么需要
    ----------
    板块模拟在极区（等距圆柱投影的两端）会把大陆切成大量碎岛，实测极区陆地块
    占全图陆地近一半 —— 用户的 "大陆像被随机打碎的拼图/碎块" 与 "苔原占陆 44.5%"
    都是这个几何造成的：碎岛总面积巨大，但每一块的面积都小到无地理意义。
    真实地球在 60° 以北只剩格陵兰 + 加拿大北极群岛 + 西伯利亚岸线，都是成片陆块。

    规则：只删**小岛**（不填陆地内部的湖），阈值随纬度从 58° 的 0 线性升到
    ``max_area``（≈50×50 px ≈ 1000 km 见方）。每个半球面积最大的连通域永不删除，
    保证大陆核部不会被误伤。

    Returns
    -------
    ndarray (S, S) float32
        去碎后的归一化高程（被删的小岛以 9×9 最小值滤波腐蚀为浅海）。
    """
    v = v01.copy()
    lab, n = ndi_label(v >= 0.5)
    if n == 0:
        return v
    flat = lab.ravel()
    sz = np.bincount(flat, minlength=n + 1).astype(np.float64)
    sz[0] = 0.0
    # 每个连通域的 |纬度| 质心（按像素平均）
    latmap = np.repeat(np.abs(lats).astype(np.float64), v.shape[1])
    cy = np.bincount(flat, weights=latmap, minlength=n + 1) / np.maximum(sz, 1.0)
    thr = np.clip((cy - lat0) / 22.0, 0.0, 1.0) * max_area
    drop = (thr > 0) & (sz < thr)
    drop[0] = False
    # 每半球最大陆块永不删除
    for sign in (True, False):
        idx = np.nonzero((cy >= 0) == sign)[0]
        idx = idx[idx > 0]
        if idx.size:
            drop[idx[np.argmax(sz[idx])]] = False
    kill = drop[flat].reshape(v.shape)
    if kill.any():
        v[kill] = minimum_filter(v, 9)[kill]
        print(f"      [polar] 极区去碎: 剔除 {int(drop.sum())} 个碎岛 "
              f"/ {n} 块 ({int(kill.sum())}px)")
    return v.astype(np.float32)


def reclaim_inland_water(dem_m, keep_max=2, keep_min_px=12000, floor_m=45.0,
                         smooth_sigma=5.0, verbose=True):
    """把大陆内部的封闭水域回填成陆地，只保留极少数巨型内陆海。

    为什么需要
    ----------
    GAN 输出与 ``land_hist_final`` 秩重排只保证"陆地在海平面以上"，并不保证
    "陆地内部不出现封闭的负高程洼地"。实测 seed=1504334501 / 2048 层终稿：
    ``dem<=0`` 且不与图幅边界连通（= 被陆地完全包围）的连通水体有 **452 个 /
    156798px = 15.0% 陆地面积**，其中 <512px 的碎点 400+ 个。渲染时这些像素走
    的是海水色卡 → 用户看到的"大陆内部密布蓝色小斑点 / 在陆地上打洞"。

    物理上封闭水体只有里海、贝加尔、北美五大湖这类巨型水文单元（地球内陆水体
    合计 ≈2% 陆地，数量以十计），不会在大陆腹地撒满碎点。故在 DEM 终稿收编：
      · 与图幅边界连通 → 外海，完全不动；
      · 封闭水体按面积降序，只保留最大的 ``keep_max`` 个且面积 ≥ ``keep_min_px``；
      · 其余全部回填为陆地：高程取最近陆地邻域的归一化高斯均值，向盆地中心
        渐降到 ``floor_m``（形成缓盆而不是平台），岸边与原地形连续，无接缝。

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        米制高程终稿。
    keep_max, keep_min_px : int
        保留的内陆海个数上限 / 单个最小面积（px）。2048 层 12000px ≈ 0.36% 陆地
        （里海 = 0.4% 陆地），实测本图最大的两个封闭水体为 18768 / 14297 px，
        正好保留成两大内陆海，其余 450 个碎点全部回填。
    floor_m : float
        回填盆地底部的最低高程（米），必须 > 0 才能保证回填后是陆地。
    smooth_sigma : float
        周边陆地高程归一化卷积的高斯 σ（px），决定回填面的起伏尺度。

    Returns
    -------
    ndarray (S, S) float32
        回填后的米制高程。
    """
    sea = dem_m <= 0
    if not sea.any():
        return dem_m
    lab, n = ndi_label(sea)
    if n <= 1:
        return dem_m
    _b = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    _b = _b[_b > 0]                                  # 0 = 陆地标签, 去掉
    sz = np.bincount(lab.ravel()); sz[0] = 0
    inner = np.setdiff1d(np.arange(1, n + 1), _b)    # 封闭水体 id
    if inner.size == 0:
        return dem_m
    order = inner[np.argsort(sz[inner])[::-1]]
    keep = order[:keep_max]
    keep = keep[sz[keep] >= keep_min_px]
    drop = np.setdiff1d(inner, keep)
    if drop.size == 0:
        return dem_m
    fill = np.isin(lab, drop)

    # 回填高程: 只对陆地做归一化高斯卷积 (避免海水负值污染), 得到与岸线连续的面
    _land = ~sea
    _num = gaussian_filter(np.where(_land, dem_m, 0.0).astype(np.float32),
                           smooth_sigma)
    _den = gaussian_filter(_land.astype(np.float32), smooth_sigma)
    base = np.where(_den > 1e-3, _num / np.maximum(_den, 1e-3), floor_m)
    base = np.clip(base, floor_m, 1200.0)
    # 由岸向心挖成缓盆: 岸边 = 周边陆地高程, 中心比周边低 (最多 350m)
    # v3.3: 旧版中心下沉 0.55·(base-floor) 且 base 上不封顶 1200m —— 大干盆地中心的
    #   回填面与周边落差可达 600m+, 同时 σ=5 平滑让它成为一块平整的"塑料贴片".
    #   现改为"最大下沉 350m + fBm 起伏", 盆地中心是缓丘平原而不是台阶.
    # v3.3b: 出图复核发现回填区仍是一整片平整米色 (块内 RGB≈[212,194,150], 而全陆均值
    #   [162,161,129]): 实测中位高程只有 50-61m, 而 floor=45m 是**硬钳位**, 大半个盆
    #   被压死在 45m 同一条等高线上, 9×9 局部起伏中位仅 6.6m (全陆 28.7m). 两处修正:
    #     1) 盆底整体抬升 +60m (w 加权, 岸边仍与邻接陆地齐平) —— 让起伏有空间落地;
    #     2) 起伏由单一 ±45m/70px 改为三尺度叠加. 第二轮实测回填区 9×9 起伏 6.5m,
    #        而同高程带的自然低地是 17.4m (仍只有 1/3) —— 低频 ±55m/70px 在 9px 窗口
    #        里几乎不变, 缺的正是 10-25px 的中频起伏, 故补 ±50m/23px + ±25m/11px.
    #     3) 第三轮实测起伏仍只有 7.5m: 根源是"下沉 350m + 起伏"把大量像素推到
    #        floor=45m 以下又被**硬钳位**, 变成一条 45m 的平板等高线. 改为
    #        "台面 base+170m → 中心微凹 60m → 起伏 0.5·R (σ≈24m, 3σ≈72m)",
    #        最低点 ≈ base+110-72 ≥ 45m, 起伏不再触底.
    d_in = distance_transform_edt(fill).astype(np.float32)
    w = np.clip(d_in / 12.0, 0.0, 1.0)
    _fb1 = np.clip(value_fbm(dem_m.shape[0], 30.0, 4,
                             rng=np.random.RandomState(1701)), -2.0, 2.0)
    _fb2 = np.clip(value_fbm(dem_m.shape[0], 90.0, 3,
                             rng=np.random.RandomState(1702)), -2.0, 2.0)
    _fb3 = np.clip(value_fbm(dem_m.shape[0], 200.0, 2,
                             rng=np.random.RandomState(1703)), -2.0, 2.0)
    _fb4 = np.clip(value_fbm(dem_m.shape[0], 700.0, 2,
                             rng=np.random.RandomState(1704)), -2.0, 2.0)
    # 4) 实测 value_fbm 的 9×9 局部起伏系数随频率递增 (freq30=0.012 / 90=0.048 /
    #    200=0.126), 中低频再叠也不出细节 —— 故补一档 freq=700 (σ≈1.5px) 的细噪声,
    #    并把能量重心从 70px 挪到 10-30px, 使回填面读起来是"砂砾平原"而非塑料板.
    val = base + w * (170.0 - 60.0 * w
                      + 0.5 * (55.0 * _fb1 + 70.0 * _fb2
                               + 45.0 * _fb3 + 35.0 * _fb4))
    val = np.maximum(val, floor_m)
    out = np.where(fill, val, dem_m).astype(np.float32)
    # 回填面 σ=1.5 轻平滑只作用在 ≤3px 的岸缘 (抹掉与原地形的 1px 台阶);
    # 全域平滑会把 freq=700 那档细噪声削掉一半, 内部保持原始起伏.
    _sm = gaussian_filter(out, 1.5)
    out = np.where(fill & (d_in < 3.0), _sm, out).astype(np.float32)
    if verbose:
        _before = float((dem_m > 0).mean() * 100.0)
        print(f"      [inland] 回填内陆封闭水体: {drop.size} 个 / {int(fill.sum())}px"
              f" | 保留 {keep.size} 个内陆海 ({sz[keep].tolist()}) | 陆 "
              f"{_before:.1f}% → {float((out > 0).mean() * 100.0):.1f}%", flush=True)
    return out


def consolidate_continents(dem_m, min_island_px=280, strait_px=4,
                           smooth_sigma=6.0, verbose=True):
    """大陆固结：把"噪波裁切"式的碎大陆修成整块板块（v62）。

    改了什么
    --------
      1) **填窄海峡**：对海面做半径 ``strait_px`` 的形态学闭运算，宽度小于
         ``2·strait_px``（2048 图 ≈ 8px ≈ 160km）的水道被填成陆地 → 碎岛连成
         半岛/大陆，海岸线从高频锯齿回到大形态。
      2) **删碎岛**：面积 < ``min_island_px``（≈280px ≈ 10.7 万 km²，冰岛量级）
         的陆块改为浅海，只保留有地理意义的岛屿。
      3) 新造陆地的高程取**周边陆地高程的归一化高斯卷积**（只统计陆地，避免
         海水负值污染），再压到 0.55 倍并夹在 6~600m —— 沿岸与原地形连续，
         不会出现"贴上去的平地"。

    为什么
    ------
    用户第 2 条「大陆板块过于支离破碎」。实测 seed=1504334501 / 2048 终稿：
    陆块 157 个，其中 134 个 <1000px（合计仅 8703px），主大陆又被大量
    <20px 的浅海汊道切出密集的青色裂缝；1:1 复核整块大陆读作"被打碎的拼图"。
    地球在 2048 尺度上只有十余个陆块，海峡宽度多在 1~5px（直布罗陀 13km、
    白令 85km），碎汊道不是地理实体而是 GAN/直方图校准的残留。

    实测：闭运算 r=8 时陆块 157→39，r=12→31；本函数 r=4 + 删岛后见调用处的
    打印（陆块数 / 陆地占比）。
    """
    sea = dem_m <= 0
    if not sea.any():
        return dem_m
    # 圆形结构元（半径 strait_px）
    _yy, _xx = np.mgrid[-strait_px:strait_px + 1, -strait_px:strait_px + 1]
    _disk = ((_xx * _xx + _yy * _yy) <= strait_px * strait_px)
    # v77 修复「地图边缘出现奇怪竖线」的**根因**：
    #   改了什么: binary_closing 补 border_value=1（图外一律视为海）。
    #   为什么: 默认 border_value=0 会把图幅外当成"非海 = 陆", 闭运算的膨胀步骤
    #     在边界上无法向外扩张、随后的腐蚀却按"外面是海"回收, 净效果是整圈
    #     strait_px 宽的**海面边框被翻成陆地**。实测 64×64 全海图 r=6 时误判
    #     1392px = 64²−52² 的完整 6px 框; 加 border_value=1 后误判 0px。
    #   实测影响(2048 终稿): 左 6 列 / 右 5 列 / 上 5 行陆地占比 100%,
    #     均值 +221~+406m, 渲染后左边缘亮度 0.558 vs 内部 0.287 → 用户看到的竖线。
    _closed_sea = binary_closing(sea, _disk, border_value=1)
    new_land = (~_closed_sea) & sea
    out = dem_m.astype(np.float32).copy()

    if new_land.any():
        # 新陆地高程：只对陆地做归一化卷积，近岸继承邻接陆地的高度
        _land = out > 0
        _num = gaussian_filter(np.where(_land, out, 0.0).astype(np.float32),
                               smooth_sigma)
        _den = gaussian_filter(_land.astype(np.float32), smooth_sigma)
        base = np.where(_den > 1e-3, _num / np.maximum(_den, 1e-3), 120.0)
        val = np.clip(base * 0.55, 6.0, 600.0).astype(np.float32)
        # 边缘 2px 与原海岸做线性过渡，避免 1px 台阶
        _d = distance_transform_edt(new_land).astype(np.float32)
        _w = np.clip(_d / 2.0, 0.0, 1.0)
        _blend = out * (1.0 - _w) + val * _w
        # 新陆地必须 > 0，否则会被 land 掩膜判回海水（填海失败）
        out = np.where(new_land, np.maximum(_blend, 1.0), out).astype(np.float32)

    # 删碎岛（只删小陆块，不动任何大块）
    lab, n = ndi_label(out > 0)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        drop = (sz > 0) & (sz < min_island_px)
        if drop.any():
            kill = drop[lab]
            # 用周围海的 9×9 最小值腐蚀，形成浅海而不是深坑
            out[kill] = np.minimum(minimum_filter(out, 9)[kill], -18.0)
    _n2 = int(ndi_label(out > 0)[1])
    if verbose:
        print(f"      [v62] 大陆固结: 填窄海峡 {int(new_land.sum())}px / 删碎岛 "
              f"<{min_island_px}px → 陆块 {n}→{_n2} 个 | 陆 "
              f"{float((dem_m > 0).mean()) * 100:.1f}% → "
              f"{float((out > 0).mean()) * 100:.1f}%", flush=True)
    return out.astype(np.float32)


def heal_map_border(dem_m, frame_px=8, verbose=True):
    """修复图幅外圈的"假陆地边框"：用内侧镜像替换最外 ``frame_px`` 圈。

    改了什么
    --------
    对最外 ``frame_px`` 圈像素做**内侧镜像**填充（镜像轴在 ``frame_px-0.5``）：
    第 k 列 = 第 ``2f-1-k`` 列，第 k 行 = 第 ``2f-1-k`` 行，左右上下同理。
    镜像使 ``col[f-1] == col[f]``，接缝两侧数值完全相等，不会产生新的台阶。

    为什么
    ------
    v77 已修根因（``consolidate_continents`` 的 ``binary_closing`` 缺
    ``border_value=1``），但**历史 DEM 文件**仍然带着那圈边框，且边框内侧
    有一个 2800m 的硬台阶（实测 x=5→6 的 |dx| = 2804.4m，正常内部中位 7.98m）。
    渲染时这圈边框是一整列高亮陆地（左边缘亮度 0.558 vs 内部 0.287），
    在 2D 图上就是用户报告的"奇怪的竖线"。本函数是**兜底修复**：无论边框
    来自哪一步，收尾时都保证边缘与内侧地形连续。

    实测
    ----
    对 v76 终稿（2048²）应用后：第 0 列陆地占比 100% → 18.8%，
    左边缘亮度 0.558 → 0.28（与内部一致），底部 3 行常数 66.0m → 正常起伏。

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        米制高程（会先复制，不原地修改入参）。
    frame_px : int
        需要替换的外圈像素宽度；取 8 以覆盖 strait_px=6 造成的 6px 框 + 余量。

    Returns
    -------
    ndarray (S, S) float32
        边缘已与内侧连续的高程。
    """
    d = dem_m.astype(np.float32).copy()
    H, W = d.shape
    f = int(max(1, min(frame_px, min(H, W) // 4)))
    # 先列后行：行镜像读取的是已经修好的列边缘，不会把假边框再复制进来。
    for k in range(f):
        d[:, k] = d[:, 2 * f - 1 - k]
        d[:, W - 1 - k] = d[:, W - 2 * f + k]
    for k in range(f):
        d[k, :] = d[2 * f - 1 - k, :]
        d[H - 1 - k, :] = d[H - 2 * f + k, :]
    if verbose:
        print(f"      [v77] 边缘修复: 外圈 {f}px 内侧镜像 "
              f"(左列陆地 {float((dem_m[:, 0] > 0).mean()) * 100:.1f}% → "
              f"{float((d[:, 0] > 0).mean()) * 100:.1f}%)", flush=True)
    return d


def sculpt_continental_outline(dem_m, lats=None, rng=None, *,
                               fine_px=900, big_px=20000, far_px=110.0,
                               dir_w=0.55, huge_amp=0.16, verbose=True):
    """按"陆块规模 + 海岸类型 + 板块走向"分级重塑海岸线（v78/v79）。

    改了什么
    --------
    1) **尺度分级 + 分区粗糙度**：每个陆块先算基础半径
       ``r = clip(1.0 + 2.4·log10(area/400), 1.0, 12)``（2048 图），
       同时生成一个 σ≈80px（≈1600km）的低频"岸型场" ``rough∈[0,1]``：
       ``rough`` 低 → 该岸段用 ``2.2r`` 强平滑（沉积/沙滩岸，长而平缓），
       ``rough`` 高 → 用 ``0.3r`` 弱平滑（侵蚀/峡湾岸，保留锯齿）。
       于是同一大陆上同时存在"平滑长岸"和"破碎岸段"，不再是全域等锯齿。
    2) **板块式长直被动边缘（v79 新增）**：大块沿自身 PCA 主轴做**定向运动模糊**
       （等距平移平均，长度 3.2r），再弱平滑——这会把圆胖团块拉成有主走向的
       大陆，并沿走向形成长而直的被动大陆边缘（大西洋型），而不是各向同性墨迹。
       与各向同性平滑按 ``dir_w`` 混合。
    3) **大尺度轮廓**：三级低频抖动（σ≈45px 的 ±``huge_amp`` + σ≈20px 的 ±0.12
       + σ≈4px 的 ±0.05），使等值线出现 10³km 级的大湾/半岛，而不是椭圆。
    4) **岛屿成因化**：面积 < ``fine_px`` 且离最近大块 > ``far_px`` 的块删除
       （v78 用 150px 时距离 p90=135px，一个都删不掉；v79 收紧到 110px，
       把"撕碎纸片"式散落小岛清掉，只留下近岸残岛 = 大陆架淹没后的露头）。
    5) **细丝/窄峡治理**：``closing(r=2)`` 加宽 <4px 地峡并填噪声细缝，
       ``opening(r=1)`` 去 1px 毛刺；每个块重塑后只保留**最大连通体**，
       避免高斯阈值产生"回声岛"。
    6) 掩膜变化后重赋高程：新陆取邻域陆地归一化高斯 ×0.6（6~900m），
       新海压到 −60~−180m 浅水，交给下游 ``create_continental_shelf``
       重建陆架—陆坡剖面。

    为什么
    ------
    用户 v78 批评：「无论巨大主大陆还是极小岛屿，边缘锯齿程度几乎一样」
    「主大陆像一滩随机蔓延的墨迹」「岛屿像撕碎的纸片散落」「缺乏大尺度的
    线性构造（长断裂带、笔直海岸线）」。实测 v77 终稿（2048²）：主大陆
    814055px、海岸曲折度 twist 3.22；1000~5000px 的块 twist 仅 1.23 ——
    大块被高频 fBm 刻成同等尺度的锯齿，而小岛反而更圆。地球的尺度关系相反：
    大陆岸线由 10³km 级构造/沉积控制，小岛受波浪侵蚀更接近圆形。

    实测
    ----
    seed=7 / 2048²（v77 终稿叠加本函数 + 末次陆架）：
      大块 twist 3.22 → 2.02（三块 ≥20000px 的中位），
      5000~20000px 1.48 → 1.26，1000~5000px 1.23 → 1.08，
      陆地 28.34% → 27.9%（去碎岛 + 平滑净减）。
    见调用处打印（陆块数 / twist / 删除碎岛数 / 增删像素）。

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        米制高程（正值陆地）。
    lats : ndarray or None
        行纬度（本函数暂不使用，保留以匹配管线签名与未来极地处理）。
    rng : numpy.random.RandomState or None
        低频抖动/新海深度噪声的随机源。
    dir_w : float
        定向运动模糊的混合权重（0 = 纯各向同性，1 = 纯定向）。
    huge_amp : float
        最大尺度（σ≈45px ≈ 880km）抖动的幅度。

    Returns
    -------
    ndarray (S, S) float32
        轮廓重塑后的米制高程。
    """
    from scipy.ndimage import (binary_erosion, binary_opening,  # noqa: F401
                               binary_closing, shift as ndi_shift)
    out = dem_m.astype(np.float32).copy()
    land = out > 0
    S = out.shape[0]
    lab, n = ndi_label(land)
    if n == 0:
        return out
    sz = np.bincount(lab.ravel()); sz[0] = 0
    rng = rng if rng is not None else np.random.RandomState(0)

    def _disk(r):
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return (xx * xx + yy * yy) <= r * r

    big_ids = [i for i in range(1, n + 1) if sz[i] >= big_px]
    if not big_ids:
        big_ids = [int(np.argmax(sz))]
    big = np.isin(lab, big_ids)
    d_big = distance_transform_edt(~big)

    def _norm(a):
        a = a.astype(np.float32)
        return a / (float(np.max(np.abs(a))) + 1e-8)

    # 三级低频抖动：保留 10³km 级湾/半岛（否则高斯平滑会把大陆圆化成椭圆）
    jit_huge = gaussian_filter(_norm(value_fbm(S, 1.1, 3, rng=rng) - 0.5), 45.0)
    jit_big = gaussian_filter(_norm(value_fbm(S, 2.2, 3, rng=rng) - 0.5), 20.0)
    jit_mid = gaussian_filter(_norm(value_fbm(S, 8.0, 3, rng=rng) - 0.5), 4.0)
    # 岸型场：低频分区决定"平滑沉积岸 vs 破碎侵蚀岸"，σ≈80px ≈ 1600km
    rough = gaussian_filter(_norm(value_fbm(S, 1.6, 3, rng=rng)), 30.0)
    rough = np.clip((rough - rough.min()) / (rough.max() - rough.min() + 1e-8),
                    0.0, 1.0).astype(np.float32)
    _t = np.clip((rough - 0.32) / 0.36, 0.0, 1.0)
    w_rough = (_t * _t * (3.0 - 2.0 * _t)).astype(np.float32)   # smoothstep

    new = np.zeros_like(land)
    n_kill, r_big = 0, []
    for i in range(1, n + 1):
        area = int(sz[i])
        if area <= 0:
            continue
        m = (lab == i)
        if area < fine_px:
            if i not in big_ids and float(d_big[m].min()) > far_px:
                n_kill += 1
                continue                       # 远处孤立碎岛 → 变海
            new |= m                           # 近岸残岛原样保留
            continue
        r = float(np.clip(1.0 + 2.4 * np.log10(area / 400.0), 1.0, 12.0))
        sm_b = gaussian_filter(m.astype(np.float32), 2.2 * r)   # 平滑岸
        sm_f = gaussian_filter(m.astype(np.float32), max(1.0, 0.3 * r))
        base = (1.0 - w_rough) * sm_b + w_rough * sm_f
        if dir_w > 0.0 and area >= big_px:
            # 沿块的主轴做定向运动模糊 → 有主走向的大陆 + 长直被动边缘
            ys, xs = np.nonzero(m)
            _dy, _dx = ys - ys.mean(), xs - xs.mean()
            _cov = np.array([[np.mean(_dx * _dx), np.mean(_dx * _dy)],
                             [np.mean(_dx * _dy), np.mean(_dy * _dy)]])
            _ev, _vec = np.linalg.eigh(_cov)
            ang = float(np.arctan2(_vec[0, 1], _vec[1, 1]))
            acc = np.zeros_like(sm_b, np.float32)
            for _t in np.linspace(-0.5, 0.5, 11):          # 11 次等距平移平均
                acc += ndi_shift(m.astype(np.float32),
                                 (np.sin(ang) * 3.2 * r * _t,
                                  np.cos(ang) * 3.2 * r * _t),
                                 order=1, mode="constant", cval=0.0)
            dirn = gaussian_filter(acc / 11.0, 1.15 * r)
            base = (1.0 - dir_w) * base + dir_w * dirn
        fld = (base + huge_amp * jit_huge
               + 0.12 * jit_big + 0.05 * jit_mid)
        flat = fld.ravel()
        k = flat.size - area                   # 面积守恒分位阈值
        thr = np.partition(flat, k)[k]
        m2 = (fld >= thr)
        # 只保留最大连通体：高斯等值线会在远端生成"回声岛"
        lb, kb = ndi_label(m2)
        if kb > 1:
            _s = np.bincount(lb.ravel()); _s[0] = 0
            m2 = (lb == int(np.argmax(_s)))
        new |= m2
        if area >= big_px:
            r_big.append((area, r))

    # 细丝/窄峡治理：先加宽 <4px 地峡、填噪声细缝，再去 1px 毛刺
    new = binary_closing(new, _disk(2))
    new = binary_opening(new, _disk(1))

    add = new & ~land
    rem = land & ~new
    if add.any():
        num = gaussian_filter(np.where(land, out, 0.0).astype(np.float32), 8.0)
        den = gaussian_filter(land.astype(np.float32), 8.0)
        base = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 120.0)
        val = np.clip(base * 0.6, 8.0, 900.0).astype(np.float32)
        d = distance_transform_edt(add).astype(np.float32)
        w = np.clip(d / 3.0, 0.0, 1.0)
        out = np.where(add, np.maximum(out * (1.0 - w) + val * w, 2.0), out)
    if rem.any():
        nz = value_fbm(S, 12.0, 3, rng=np.random.RandomState(7781))
        out[rem] = (-60.0 - 120.0 * nz[rem]).astype(np.float32)

    if verbose:
        _lab2, _n2 = ndi_label(out > 0)
        print(f"      [v78] 轮廓重塑: 碎岛 -{n_kill} | 新陆 +{int(add.sum())}px "
              f"新海 -{int(rem.sum())}px | 陆块 {n}→{_n2} 个 | "
              f"大块平滑 r {[f'{a/1000:.0f}k:{r:.1f}' for a, r in r_big]} | "
              f"陆 {float((dem_m > 0).mean()) * 100:.2f}% → "
              f"{float((out > 0).mean()) * 100:.2f}%", flush=True)
    return out.astype(np.float32)


def _disk_selem(r):
    """半径 r 的圆形结构元（bool）。"""
    _yy, _xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (_xx * _xx + _yy * _yy) <= r * r


def _norm_sym(a):
    """去均值后压到 [-1, 1]（按绝对值最大者归一）。

    v81 实测：只归一不去均值时，``value_fbm−0.5`` 经大 σ 平滑后仍带 +0.1~0.3 的
    全局偏置，直接当位移场会把海岸整体外推 —— 首轮试跑 ``+160638px 新陆 vs
    −10297px 新海``，陆地 27.77%→31.35%（多出 3.6 个百分点）。去掉均值后位移场
    净面积近似守恒。
    """
    a = a.astype(np.float32)
    a = a - float(a.mean())
    return a / (float(np.max(np.abs(a))) + 1e-8)


def _znoise(S, sigma, rng, octaves=1):
    """按**目标特征尺度** sigma(px) 生成 z-score 噪声（std≈1）。

    改了什么：v81 的位移场一律走 ``_norm_sym(value_fbm(...))`` —— 先按 max|·| 归一，
    实测 std 只剩 ~0.26，于是"30px 振幅的大湾"实际只把海岸推移 ~8px，"1~2px 的
    侵蚀齿"实际 ~0.4px，等于没加；这正是「海岸像融化塑料」的量化来源。
    本函数改成按 std 归一，调用方给的振幅就是真实的像素位移 1σ。

    为什么不用 ``value_fbm`` 的 base_freq：base_freq 与像素尺度是倒数关系
    （sigma = S/(2·freq)），在函数里直接写 sigma 可读性更好，也不会在 2048/4096
    两种分辨率下出现不同的表现尺度。
    """
    s = max(float(sigma), 0.6)
    a = value_fbm(S, max(S / (2.0 * s), 0.6), max(int(octaves), 1), rng=rng)
    a = a.astype(np.float32)
    return (a - float(a.mean())) / (float(a.std()) + 1e-8)


def reshape_coast_scale_stratified(dem_m, lats=None, rng=None, verbose=True):
    """按陆块规模分层的海岸线重塑：大块出大湾/大半岛，小块保原样（v81）。

    改了什么
    --------
    1) **尺度场**：对每个连通陆块算等效半径 ``R=sqrt(area/π)``，铺回像素后
       σ=25px 平滑成连续场 ``L``；``w_macro = clip((L−14)/90,0,1)**0.8``。
       于是 70 万 px 的大陆 ``w≈1``、5000px 的岛 ``w≈0.25``、500px 的礁
       ``w≈0`` —— 大尺度形变**只作用在大陆上**。
    2) **宏观湾/半岛（波长 ~300px ≈ 6 000km）**：对**有符号距离场**
       ``phi = d_in − d_out`` 直接减去 ``w_macro·(30·n₁ + 9·n₂)`` 再取零等值线。
       相位于是在大块上整体外扩/内缩最多 30px（≈590km），形成宽阔深海湾与
       宏大半岛；小岛因为 ``w≈0`` 完全不动。
    3) **微观侵蚀齿**：``(1+1.2·L/40)·n₃(σ=1.4) + 0.7·n₄(σ=0.8)``，即 1~2px
       尺度的碎齿，给所有海岸（含大陆）加"被水切割"的粗糙边，而不是只在大陆
       上做大弯、边缘仍是一条矢量曲线。
    4) 二值化后 ``opening(disk 1)`` 去掉 1px 毛刺；新增陆地按离新岸距离
       18m/px 抬升、新增海域压到 −45~−155m（交给下游陆架重建剖面）。

    为什么
    ------
    用户 v81 批评："宏观的过度破碎与微观的极致平滑并存 —— 大陆边缘像融化的
    塑料/矢量曲线，没有侵蚀感；右侧巨型大陆与左下的极小岛屿是同一个规格，
    缺乏分形维度差异"。v79 的 ``sculpt_continental_outline`` 是"高斯平滑 + 面积
    守恒取分位"，它的输出一定是**等尺度**的光滑等值线：平滑半径只跟块大小
    有关（大块 σ≈26px），所有低于 20px 的结构被抹平，所以越大的块越像塑料。

    实测
    ----
    seed=7 / 2048²（接在 v80h 终稿上，单独调用本函数）：
      见调用处的打印（陆占比 / 增删像素 / twist 分级）。
    见 ``/tmp/probe_v81.py`` 的 bay-fill 谱（k=16/32/64 的湾填充面积）。

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        米制高程（正值陆地）。
    lats : ndarray or None
        行纬度，本函数不使用，保留以匹配管线签名。
    rng : numpy.random.RandomState or None
        低频/高频噪声源。

    Returns
    -------
    ndarray (S, S) float32
        重塑后的米制高程。
    """
    from scipy.ndimage import (binary_erosion, binary_opening,  # noqa: F401
                               label as _lbl)
    out = dem_m.astype(np.float32).copy()
    land = out > 0
    S = out.shape[0]
    rng = rng if rng is not None else np.random.RandomState(0)
    lab, n = _lbl(land)
    if n == 0:
        return out
    sz = np.bincount(lab.ravel()).astype(np.float32)
    sz[0] = 0.0
    rad = np.sqrt(sz / np.pi)
    radmap = rad[lab].astype(np.float32)
    L = gaussian_filter(radmap, 25.0)
    w_macro = np.clip((L - 14.0) / 90.0, 0.0, 1.0) ** 0.8

    d_in = distance_transform_edt(land).astype(np.float32)
    d_out = distance_transform_edt(~land).astype(np.float32)
    phi = d_in - d_out

    # v82: 真正的多倍频分形岸线（替代 v81 那套"名义 30px、实际 8px"的位移场）。
    #   改了什么：v81 的 n1~n4 都经 ``_norm_sym``（按 max|·| 归一），实测 std 只有
    #     ~0.26 → 30px·n₁ 实际只把海岸推 ~8px（波长还是 850px，即一个超长缓弯），
    #     1~2px 齿实际仅 0.4px，肉眼完全读不到。现改为 z-score（std=1）的五级
    #     级联 + 一级**脊线**场：
    #       σ=520px/12px  → 大洲级深海湾、大半岛（只作用在大陆 w_macro）
    #       σ=190px/7px   → 数百公里的次一级湾岬
    #       σ=70px/4px    → 海湾内的次级凹凸（~1300km 以下）
    #       σ=26px/2.2px  → 岬角上的小半岛、岸外礁
    #       σ=9px/1.1px   → 1~2px 的侵蚀碎齿（"被水切割"的readable detail）
    #       -|n₁₄₀|/4px   → 脊线（零交叉处是尖角），给岬角与湾口**棱角**，
    #                        避免正弦般的圆润弧 —— 用户说的"矢量曲线感"。
    #   为什么：分形维度 D≈1.1~1.2 要求振幅按波长近似等比衰减；v81 只有两级且
    #     有效振幅被压掉 4 倍，等于一条 850px 波长的缓弯，所以"远看碎屑、近看塑料"。
    #   实测：见调用处的 twist 分级打印 + /tmp 探针（改前 1.09~1.39）。
    o_mac = _znoise(S, 520.0, rng, 2)
    o_mid = _znoise(S, 190.0, rng, 2)
    o_fin = _znoise(S, 70.0, rng, 2)
    o_sub = _znoise(S, 26.0, rng, 1)
    o_teeth = _znoise(S, 9.0, rng, 1)
    # v82b: 像素级两级 —— 这是"融化塑料"的最后一刀。v82 首轮只做到 σ=9px，
    #   而 σ=9 的高斯随机场的等值线曲率半径 ≥9px，描出来仍是一条光滑曲线
    #   （实测离线试跑 /tmp/v82_mask_crops.png：只多了几个大起伏，近看还是塑料）。
    #   真正的海岸在 19.6km/px 尺度上仍有 1~3px 的碎齿与凹口，所以补 σ=3.5/0.9px
    #   与 σ=1.6/0.55px 两级，让边界在像素级"咬"起来。
    o_px1 = _znoise(S, 3.5, rng, 1)
    o_px2 = _znoise(S, 1.6, rng, 1)
    # 脊线：标准正态在零交叉处密度最高，-|n| 把这些位置变成尖峰（岬角/断层崖）
    rg = -(np.abs(_znoise(S, 140.0, rng, 2)) - 0.798)
    rg2 = -(np.abs(_znoise(S, 14.0, rng, 1)) - 0.798)
    kc = np.clip(L / 60.0, 0.15, 1.0)          # 小块整体缩幅（保持尺度分层）
    disp = w_macro * (14.0 * o_mac)
    disp = disp + kc * (8.0 * o_mid + 5.0 * o_fin + 3.6 * o_sub + 2.2 * o_teeth)
    disp = disp + (0.25 + 0.75 * kc) * (1.0 * o_px1 + 0.5 * o_px2)
    disp = disp + np.clip((L - 10.0) / 50.0, 0.0, 1.0) * (5.0 * rg + 2.2 * rg2)
    # 面积守恒：只在**海岸带**(|phi|<60px)上把位移场归零均值。全局去均值不够 ——
    #   低频噪声在沿海带的均值仍可能偏正，实测首轮 +160638px 新陆 / −10297px 新海。
    _band = np.abs(phi) < 60.0
    if _band.any():
        disp = disp - float(disp[_band].mean())

    new = (phi - disp) > 0.0
    # v81 修（实测）：原先这里用 binary_opening(new, disk1) 去毛刺，但它同时把
    #   disp 里的 1~2px 侵蚀齿全部啃掉 —— 首轮 twist 大块 2.42→1.97（反而更光滑），
    #   海岸总长 20159→18622px。改为**按连通块面积**删除 <6px 的数值毛刺，
    #   毫米级的齿与凹口全部保留。
    _lb, _nn = _lbl(new, structure=np.ones((3, 3), bool))
    _sz = np.bincount(_lb.ravel()); _sz[0] = 0
    _keep = np.zeros(_sz.size, bool)
    _keep[1:] = _sz[1:] >= 6
    # v82: 分形位移会把小块甩到岸外几十像素 —— 全新碎块（不接触原陆地）必须
    #   ≥80px 才留，否则就是新长出来的"芝麻"（用户第 3 条「随机撒芝麻」的反向版本）。
    _touches = np.bincount(_lb[land].ravel(), minlength=_sz.size) > 0
    _keep &= (_sz >= 80) | _touches
    new = _keep[_lb]

    add = new & ~land
    rem = land & ~new
    if add.any() or rem.any():
        d_new = distance_transform_edt(new).astype(np.float32)
    if add.any():
        nz = value_fbm(S, 14.0, 2, rng=np.random.RandomState(9182))
        val = np.clip(18.0 * d_new + 12.0 * nz, 4.0, 800.0).astype(np.float32)
        out = np.where(add, val, out)
    if rem.any():
        nz2 = value_fbm(S, 12.0, 3, rng=np.random.RandomState(7713))
        # v81b 修（实测）：原先把新海一律压成 −45~−155m，与离岸距离无关 ——
        #   实测这一步之后浅海(−170~0m) 从 9.1% 涨到 13.0%，且 `-170m 等深线到岸`
        #   的中位宽度被撑到 22px（≈430km）：新吞进来的大湾是整片 100m 深的浅水台，
        #   而下游 create_continental_shelf 只在 dist<d_tot(~10~25px) 内重铺剖面，
        #   大湾深处够不到，于是留下"越靠湾心越浅"的浅水平台 —— 正是用户
        #   「外围浅海像钢笔描边」的放大版。
        #   现按到新岸线的距离给一条真实陆坡剖面：d=1→−93m、3→−267m、6→−648m、
        #   10→−1380m、20→−4200m（夹到本地深海基底），再叠 ±60m 起伏。
        # v82 修（用户 v81 第 3 条「45° 直边楔形」）：depth 只是 d 的单调函数，
        #   而 d 是 EDT —— 两片陆地的**等距线在中间汇成一条直线中轴**，量化成
        #   200m 色带后就是那片刺眼的直边楔形（前几轮已定位、未修）。这里给距离
        #   叠一层 σ=55/18px、±3.2/1.4px 的低频位移（域扭曲），等距线被扭成不
        #   规则曲线，直边随之消失；剖面量级（30~4200m）不变。
        _wq = 3.2 * _znoise(S, 55.0, np.random.RandomState(6601), 2) \
            + 1.4 * _znoise(S, 18.0, np.random.RandomState(6602), 1)
        # v82c 修（实测回归）：位移必须按离岸距离渐入。v82 首版直接叠 ±3~4px，
        #   而这里 d=1→−93m、d=3→−267m，±3px 就等于 ±300m —— 出图
        #   (/tmp/v82_ab_crops.png) 的新海岸浅带被撕成"迷彩斑"。
        #   楔形在离岸 15~60px 的中轴处，近岸 14px 内根本不需要扭曲。
        _ramp = np.clip(d_new / 14.0, 0.0, 1.0) ** 1.5
        d_eff = np.maximum(d_new + _wq * _ramp, 0.4)
        dep = -(30.0 + 55.0 * d_eff + 8.0 * d_eff ** 2)
        dep = -np.minimum(-dep + 60.0 * nz2, 4200.0)
        out[rem] = np.minimum(dep[rem], -20.0).astype(np.float32)

    if verbose:
        _lb, _nn = _lbl(out > 0)
        print(f"      [v81 尺度分层] 陆 {float(land.mean()) * 100:.2f}% → "
              f"{float((out > 0).mean()) * 100:.2f}% | 新陆 +{int(add.sum())}px "
              f"新海 -{int(rem.sum())}px | 陆块 {n}→{_nn} | "
              f"w_macro p50={np.percentile(w_macro, 50):.2f} "
              f"p95={np.percentile(w_macro, 95):.2f}", flush=True)
    return out.astype(np.float32)


def carve_coastal_inlets(dem_m, lats=None, rng=None, n_inlets=150,
                         min_sep_px=26.0, min_block_px=3000, verbose=True):
    """沿岸峡湾/溺谷：从海岸向陆地方向刻出细长指状水道（v82）。

    改了什么
    --------
    v82 新增。在面积 ≥3000px 的陆块海岸上按贪心最小间距 ~26px（≈510km）
    取至多 ``n_inlets`` 个口门；每个口门沿"向内法线 + 每步随机偏转"的路径
    向陆推进 10~44px，途中把高程压到海平面以下：口门处宽 ~3px / 深 −25m（浅槛），
    中段深至 −330m（峡湾盆），谷头回升到 −120m 并收窄到 ~0.8px。路径逐像素相连，
    所以水道一定与海连通；一旦离"另一侧的海"不足 3px 就截断，不会把半岛切断。

    **必须放在 create_continental_shelf 之后**（与 carve_river_mouths 同层）：
    建架会按"到新岸线的距离"重铺 −25~−170m 剖面，1~3px 宽的水道距离算出来就是
    1~3px，全被抹成 −30~−130m 浅沟（v82 首版实测），峡湾读不出来。

    为什么
    ------
    用户 v81「大陆边缘像融化的塑料/矢量曲线 —— 没有水流切割的河口、没有
    断层棱角」。v81 只靠"位移场把岸线推歪"，而**高斯随机场的等值线本身
    就是光滑曲线**（曲率半径 ≥ 噪声尺度），再推也只是换个方向的圆滑弧。
    真实海岸（挪威、苏格兰西岸、缅因、智利峡湾区）的辨识特征恰恰是
    "1~3px 宽的指状水道切进陆地"——这是排水与冰川侵蚀留下的、非等值线
    形状的拓扑特征，必须单独刻。

    实测
    ----
    seed=42 / 2048²：见调用处打印（口门数 / 挖去的陆地像素 / 陆块数变化）。

    Parameters
    ----------
    n_inlets : int
        最多刻多少条（受可用岸点与间距约束，实际可能更少）。
    min_sep_px : float
        口门之间的最小间距（px）。26px ≈ 510km，与真实大河口/峡湾间距同量级。
    min_block_px : float
        只在面积不小于该值的陆块上刻 —— 小礁石没有流域，刻了就是碎屑。
    """
    from scipy.ndimage import (binary_erosion as _be, binary_dilation as _bd,
                               label as _lbl)

    out = dem_m.astype(np.float32).copy()
    land = out > 0
    if not land.any():
        return out
    _lab, _n = _lbl(land)
    if _n == 0:
        return out
    _sz = np.bincount(_lab.ravel()).astype(np.float32)
    _sz[0] = 0.0
    _big = _sz >= float(min_block_px)

    coast = land & ~_be(land, structure=np.ones((3, 3), bool))
    cy, cx = np.nonzero(coast)
    if cy.size == 0:
        return out
    keep = _big[_lab[cy, cx]]
    cy, cx = cy[keep], cx[keep]
    if cy.size == 0:
        return out

    # 向内法线：EDT 的梯度指向离海更远（=更靠内陆）的方向
    d_in = distance_transform_edt(land).astype(np.float32)
    gy, gx = np.gradient(d_in)
    # 低频扰动场：让水道"蜿蜒"，不是笔直插进去
    pn = _znoise(out.shape[0], 40.0, rng, 2)

    order = rng.permutation(cy.size)
    chosen = []
    used = np.zeros_like(land)
    carved_mask = np.zeros_like(land)
    S = out.shape[0]
    carved = 0
    for i in order:
        if len(chosen) >= int(n_inlets):
            break
        y0, x0 = int(cy[i]), int(cx[i])
        if used[y0, x0]:
            continue
        chosen.append((y0, x0))
        # 标记间距（用圆盘近似，半径 min_sep/2）
        _r = int(max(min_sep_px * 0.5, 1))
        used[max(0, y0 - _r):min(S, y0 + _r + 1),
             max(0, x0 - _r):min(S, x0 + _r + 1)] = True

        _ny, _nx = float(gy[y0, x0]), float(gx[y0, x0])
        ang = float(np.arctan2(_ny, _nx))
        _len = int(rng.randint(10, 45))
        y, x = float(y0), float(x0)
        for s in range(_len):
            t = s / max(_len - 1.0, 1.0)
            # 每步偏转：低频场 + 随机抖动，同时轻度拉回"向内"方向（避免沿岸走）
            ang += 0.55 * float(pn[min(max(int(y), 0), S - 1),
                                    min(max(int(x), 0), S - 1)]) * 0.35 \
                + float(rng.normal(0.0, 0.18))
            ang = 0.88 * ang + 0.12 * float(np.arctan2(_ny, _nx))
            y += np.sin(ang)
            x += np.cos(ang)
            iy, ix = int(round(y)), int(round(x))
            if not (0 <= iy < S and 0 <= ix < S):
                break
            # v82: 距离"另一侧海"不足 3px 就停 —— 否则水道会把半岛/地峡切断，
            #   实测首版把陆块 18 拆成 62，全是这种"一刀两断"。
            #   注意 s<5 时 d_in 本来就只有 1~3px（口门就在岸上），不能判。
            if s >= 5 and d_in[iy, ix] < 3.0:
                break
            w = max(5.0 * (1.0 - t) + 1.0 * t, 1.0)   # 口门 5px → 末端 1px
            # v83: 口门 3px→5px。2048 图上 1px≈19.6km，3px 口门只有 60km，
            #   描出来是一条"划痕"；5px(≈100km) 才读得出峡湾/溺谷的湾口。
            #   末端仍收成 1px，保证谷头尖锐（不是一根等宽的沟）。
            # 峡湾剖面：口门有浅槛 (−25m)、中段深盆 (−~330m)、谷头回升到 −120m。
            #   v82 首版是 −6→−45m 的浅沟，建架后被抹平且本身也读不出"峡湾"。
            dep = -(25.0 + 30.0 * t + 320.0 * np.exp(-((t - 0.62) / 0.34) ** 2))
            _rw = int(w * 0.5)
            ys = slice(max(0, iy - _rw), min(S, iy + _rw + 1))
            xs = slice(max(0, ix - _rw), min(S, ix + _rw + 1))
            blk = out[ys, xs]
            m = blk > 0
            if m.any():
                carved += int(m.sum())
                carved_mask[ys, xs] |= m
            blk[m] = np.minimum(blk[m], np.float32(dep))
            out[ys, xs] = blk

    # v82c: 清掉水道留下的碎屑。水道切穿窄地峡后会在原地剩 1~7px 的孤立碎点
    #   （实测 150 条水道把 <100px 的陆块从 12 个增到 35 个，其中 23 个是
    #   1~7px）。despeckle_islets_m 已经在**本函数之前**跑过，不会再兜底，
    #   所以必须在这里清。只清"贴着新水道"的碎屑，原地就有小礁石不动。
    if carved:
        _l3, _n3 = _lbl(out > 0)
        _s3 = np.bincount(_l3.ravel())
        _dust = (_s3 < 40)
        _dust[0] = False
        if _dust.any():
            _kill = _dust[_l3] & _bd(carved_mask, np.ones((3, 3), bool), iterations=2)
            if _kill.any():
                # 取 9×9 邻域最深值 —— 直接抹平成浅滩会在深海里成为亮斑
                out[_kill] = minimum_filter(out, 9)[_kill]

    if verbose:
        _l2, _n2 = _lbl(out > 0)
        print(f"      [v82 峡湾] 水道 {len(chosen)} 条 | 挖去陆地 {carved}px | "
              f"陆块 {_n}→{_n2} | 陆 {float((out > 0).mean()) * 100:.2f}%",
              flush=True)
    return out


def carve_river_mouths(dem_m, lats=None, rng=None, n_rivers=22,
                       min_sep_px=85.0, delta_frac=0.45, verbose=True):
    """在大陆海岸上刻出河口湾与三角洲（v81 水文学修复）。

    改了什么
    --------
    1) 候选岸点只取 **面积 ≥20000px 的大陆**（小岛没有大流域），再用
       贪心最小间距 85px（≈1700km）采样 ``n_rivers`` 个口门 —— 真实世界
       能在这个比例尺上读出来的大河口也就二十来个。
    2) 每个口门按 45% 概率分两支：
       - **三角洲**：向海 1.5px 外的扇形沉积体（半径 4~8px，高 3~9m，
         扇形半角随离口距离张开），把海岸推出去一个弧；
       - **河口湾**：向陆楔形下切 6~14px，宽度 3.0→0.6px，深度 −18→−73m，
         在轮廓图上就是一个漏斗状缺口（泰晤士/拉普拉塔型）。
    3) 两支都再向陆刻一条宽 ~1.2px、深 15~35m、长 40~90px 的河谷，
       让高度图上能读出"河流从哪里来"。

    为什么
    ------
    用户 v81："缺乏水文学（找不出大河的入海口和三角洲）"。此前整条管线只有
    GAN 通道里的低频河网，最终 DEM 上没有任何**河口形态**：海岸线是一条连续
    光滑曲线，没有一处凹口或沉积扇。

    实测
    ----
    seed=7 / 2048²：见调用处的 ``[v81 河口] N 个（三角洲 x / 河口湾 y）``。

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        米制高程。
    lats : ndarray or None
        未使用，保留以匹配签名。
    rng : numpy.random.RandomState or None
        口门位置/形态的随机源。
    n_rivers : int
        全球口门数量上限。
    min_sep_px : float
        两个口门的最小间距（px）。
    delta_frac : float
        三角洲（相对河口湾）的比例。

    Returns
    -------
    ndarray (S, S) float32
        刻完河口/三角洲的高程。
    """
    from scipy.ndimage import binary_erosion, label as _lbl  # noqa: F401
    out = dem_m.astype(np.float32).copy()
    land = out > 0
    S = out.shape[0]
    rng = rng if rng is not None else np.random.RandomState(0)
    lab, n = _lbl(land)
    if n == 0:
        return out
    sz = np.bincount(lab.ravel()); sz[0] = 0
    big_ids = [i for i in range(1, n + 1) if sz[i] >= 20000]
    if not big_ids:
        return out
    coast = land & ~binary_erosion(land) & np.isin(lab, big_ids)
    ys, xs = np.nonzero(coast)
    if ys.size == 0:
        return out
    picked = []
    for k in rng.permutation(ys.size):
        y, x = int(ys[k]), int(xs[k])
        if all((y - ay) ** 2 + (x - ax) ** 2 > min_sep_px ** 2 for ay, ax in picked):
            picked.append((y, x))
        if len(picked) >= n_rivers:
            break

    g = gaussian_filter(out, 3.0)
    gy, gx = np.gradient(g)
    n_delta = n_est = 0
    for (y, x) in picked:
        nx_, ny_ = float(gx[y, x]), float(gy[y, x])
        nl = float(np.hypot(nx_, ny_)) + 1e-9
        ux, uy = nx_ / nl, ny_ / nl          # 上坡 = 内陆
        ox, oy = -ux, -uy                     # 向海
        # 内陆河谷：宽 1.2px、深 15~35m、长 40~90px
        vlen = rng.uniform(40.0, 90.0)
        vdep = rng.uniform(15.0, 35.0)
        for t in np.arange(1.0, vlen, 1.0):
            iy = int(round(y + uy * t)); ix = int(round(x + ux * t)) % S
            if not (0 <= iy < S) or out[iy, ix] <= 0:
                break
            r = int(max(1, 2 - t / vlen))
            yy, xx = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
            # v81 修：先用未裁剪的坐标算距离，再裁剪/环绕后才做索引。否则 ix 贴近
            #   图幅右缘时 xx 会越界（首轮实测 IndexError: index 2048），贴左缘时
            #   环绕后的 xx 又会让 rr 虚高、漏刻半个河湾。
            rr = np.hypot(yy - iy, xx - ix)
            m = rr <= 1.2
            out[np.clip(yy, 0, S - 1)[m], (xx % S)[m]] -= (
                vdep * (1.0 - 0.5 * t / vlen)).astype(np.float32)
        if rng.rand() < delta_frac:
            R = rng.uniform(4.0, 8.0)
            cy, cx = y + oy * 1.5, x + ox * 1.5
            pad = int(R * 2.2) + 2
            y0, y1 = max(0, int(cy) - pad), min(S, int(cy) + pad + 1)
            x0 = max(0, int(cx) - pad); x1 = min(S, int(cx) + pad + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            along = (xx - cx) * ox + (yy - cy) * oy
            across = (xx - cx) * (-oy) + (yy - cy) * ox
            fan = np.abs(across) < (0.45 * R + 0.5 * np.clip(along + R, 0, 2 * R))
            rr = np.hypot(along * 0.85, across)
            h = rng.uniform(3.0, 9.0)
            blob = h * np.clip(1.0 - (rr / (1.2 * R)) ** 2, 0, 1) * fan
            out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1],
                                           blob.astype(np.float32))
            n_delta += 1
        else:
            L = rng.uniform(6.0, 14.0)
            W0 = rng.uniform(2.6, 3.6)
            for t in np.arange(0.0, L, 0.5):
                iy = int(round(y + uy * t)); ix = int(round(x + ux * t)) % S
                if not (0 <= iy < S):
                    break
                w = W0 * (1.0 - t / L) + 0.6
                r = int(w) + 1
                yy, xx = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
                rr = np.hypot(yy - iy, xx - ix)
                m = rr < w
                dep = -18.0 - 55.0 * (t / L)
                out[np.clip(yy, 0, S - 1)[m], (xx % S)[m]] = np.minimum(
                    out[np.clip(yy, 0, S - 1)[m], (xx % S)[m]],
                    np.float32(dep))
            n_est += 1

    if verbose:
        print(f"      [v81 河口] {len(picked)} 个（三角洲 {n_delta} / "
              f"河口湾 {n_est}）", flush=True)
    return out.astype(np.float32)


def _stamp_island_dome(out, cy, cx, L, W, peak, ty, tx, rng, slope=80.0):
    """在 (cy, cx) 盖一座**轮廓不规则**的椭圆穹顶岛（v81c）。

    改了什么 / 为什么 / 实测
    -----------------------
    改了什么：把 v81b 里岛弧与热点**各抄一份**的 ``peak·e^(−r²) − k·r²`` 提成本
    函数，并加三处形状扰动：
      1) 归一化半径叠两层零均值噪声（σ≈7px 担 ±26%、σ≈2.6px 担 ±8%，幅度夹在
         [0.60, 1.50]），扰动强度按 ``min(u, 1.4)`` 随半径增大 —— 岛心仍是单峰，
         只让**轮廓**起伏；向内收缩上限 40% 是为了不让窄岛被横向掐断；
      2) 峰顶沿长轴随机偏心 ±0.35L（真实火山岛/背斜岛的最高点多半偏在一侧）；
      3) 整块叠一条 ±16m 的线性倾斜，使一侧陆架窄、另一侧宽（不再同心）。
      另：L ≥ 24px 的大岛有 45% 概率再挂一个 0.42L 的偏心小瓣（岬角/次火山锥）。

    为什么：v81b 出图目视复核（热点链裁切）—— 每座岛都是**完美同心椭圆**：奶油色
    椭圆 + 一圈等宽浅蓝，整条链像一排鸡蛋，比不画岛更"算法"。真实火山岛的
    轮廓有岬角、海湾，分形维数远高于椭圆（理想椭圆的 twist 恒为 1.00）。

    实测（seed=7 / 2048）：改前 27 座岛、`<800px` 碎块 8 个（几乎全是理想椭圆，
    单岛 twist 0.98）；改后 27 座岛、`<800px` 碎块 37 个（多出来的是主岛旁的
    小屿/岬角，可读作"列岛"），单岛 twist 1.04、9 颗叠加 twist 1.00→1.16。
    注意：twist 只看"周长/等面积圆周长"，对**椭圆 vs 不规则**不敏感（椭圆本来就是
    1.0 附近），真正的差别在轮廓不再是同心椭圆 —— 出图裁切可见。

    Returns
    -------
    ndarray (S, S) float32
        就地改写的 ``out``（调用方持有同一数组）。
    """
    S = out.shape[0]
    # v82: 掩膜半径 r2<3 对应 u<1.73；u 被 f（≤1.50）放大后实际可达 2.6L，
    #   而旧 pad=2.2L+5 的矩形窗口先截断了掩膜 → 岛的外缘被**轴对齐矩形**硬切，
    #   出图就是"直角缺口 / 方形台面"。pad 必须 ≥ 2.6L + 噪声余量。
    pad = int(L * 3.0) + 8
    y0, y1 = max(0, cy - pad), min(S, cy + pad + 1)
    x0, x1 = max(0, cx - pad), min(S, cx + pad + 1)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return out
    h, w = y1 - y0, x1 - x0
    size = max(h, w)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    along = (xx - cx) * tx + (yy - cy) * ty
    across = (xx - cx) * (-ty) + (yy - cy) * tx
    along = along - rng.uniform(-0.35, 0.35) * L          # 峰顶偏心
    u = np.sqrt((along / L) ** 2 + (across / W) ** 2)
    k = np.minimum(u, 1.4)                                # 岛心不受扰动
    n1 = value_fbm(size, 7.0, 3, rng=rng)[:h, :w]
    n2 = value_fbm(size, 2.6, 2, rng=rng)[:h, :w]
    # 幅度夹在 [0.60, 1.50]：向内收缩不能超过 40%，否则窄岛（W 只有 3~6px）会被
    #   噪声在横向"掐断"成两三个碎块（v81c 首轮实测：27 座岛炸成 112 个 <800px 块）。
    f = np.clip(1.0 + (0.26 * np.clip(n1, -1.5, 1.5)
                       + 0.08 * np.clip(n2, -1.5, 1.5)) * k, 0.60, 1.50)
    u = u * f
    r2 = u * u
    th = rng.uniform(0.0, 2.0 * np.pi)
    dome = (peak * np.exp(-r2) - slope * r2
            - 16.0 * (along * np.cos(th) + across * np.sin(th)) / max(L, 1.0))
    m = r2 < 3.0
    sub = out[y0:y1, x0:x1]
    # v82: dome 在 r2=3 处 ≈ −240m，周边深海 −4700m —— 直接硬切回 sub 会留下
    #   4400m 的 1px 断崖。改为 r2∈[1,3] 内把 dome 平滑混合回原地形（smoothstep），
    #   使岛裙与海底连续过渡，断崖消失。
    _tb = np.clip((r2 - 1.0) / 2.0, 0.0, 1.0)
    _tb = _tb * _tb * (3.0 - 2.0 * _tb)
    dome = dome * (1.0 - _tb) + sub * _tb
    out[y0:y1, x0:x1] = np.where(m, np.maximum(sub, dome), sub).astype(np.float32)
    if L >= 24.0 and rng.rand() < 0.45:                   # 偏心小瓣（岬角）
        ang = rng.uniform(0.0, 2.0 * np.pi)
        ux = np.cos(ang) * tx + np.sin(ang) * (-ty)
        uy = np.cos(ang) * ty + np.sin(ang) * tx
        lo = rng.uniform(0.50, 0.95) * L
        L2, W2 = 0.42 * L, max(2.0, 0.62 * W)
        a2 = (xx - (cx + ux * lo)) * tx + (yy - (cy + uy * lo)) * ty
        c2 = (xx - (cx + ux * lo)) * (-ty) + (yy - (cy + uy * lo)) * tx
        u2 = np.sqrt((a2 / L2) ** 2 + (c2 / W2) ** 2)
        n3 = value_fbm(size, 4.0, 2, rng=rng)[:h, :w]
        u2 = u2 * (1.0 + 0.45 * np.clip(n3, -1.5, 1.5) * np.minimum(u2, 1.4))
        r22 = u2 * u2
        dome2 = 0.55 * peak * np.exp(-r22) - slope * r22
        sub = out[y0:y1, x0:x1]
        _tb2 = np.clip((r22 - 1.0) / 2.0, 0.0, 1.0)
        _tb2 = _tb2 * _tb2 * (3.0 - 2.0 * _tb2)
        dome2 = dome2 * (1.0 - _tb2) + sub * _tb2
        out[y0:y1, x0:x1] = np.where(r22 < 3.0, np.maximum(sub, dome2),
                                     sub).astype(np.float32)
    return out


def add_island_arc_chains(dem_m, lats=None, rng=None, mcls=None, n_arcs=3,
                          verbose=True):
    """在**主动边缘**外海生成岛弧链（日本列岛/千岛型）（v81）。

    改了什么
    --------
    1) 岸段筛选：``mcls``（边缘类型系数，小 = 窄陆架 = 俯冲岸）最低 30% 分位
       的岸线 → 二值膨胀 9px 连成岸段 → 只保留 ≥80px 的段，随机取 ``n_arcs`` 段。
    2) 弧链定位：从段内随机一点起步，用 ``cKDTree`` 在海岸点集上**沿切线**
       每步走 62~105px 再投影回岸线，逐岛取外法向（d_out 梯度）偏移
       60~135px → 弧随海岸弯曲，而不是一条直线。
    3) 单岛：长 15~34px（≈300~670km，日本本州 ≈1300km、九州 ≈450km 取中段）、
       宽 3.2~6.5px 的椭圆（长轴沿弧切向），峰高 260~1250m；岛与岛之间间距
       62~105px，**有序**排列（不随机撒盐）。整条弧下方加 σ=14px、+500m 的
       弧脊，使岛链坐在一条可读的海底隆起上。

    为什么
    ------
    用户 v81："岛屿的散落像随机撒下的芝麻，没有形成岛弧（如日本列岛）"。
    原 ``terrain_engine.add_island_arcs`` 的偏移量是 9~17px、岛长 4~11px，
    那是给 512 网格写的；搬到 2048 图上只有 1~2px，视觉上等于没生成。

    实测
    ----
    seed=7 / 2048²：见调用处 ``[v81 岛弧] N 条 / M 座``。

    Returns
    -------
    ndarray (S, S) float32
    """
    from scipy.ndimage import binary_erosion, binary_dilation, label as _lbl
    from scipy.spatial import cKDTree
    out = dem_m.astype(np.float32).copy()
    land = out > 0
    S = out.shape[0]
    rng = rng if rng is not None else np.random.RandomState(0)
    coast = land & ~binary_erosion(land)
    if not coast.any():
        return out
    if mcls is not None:
        q = float(np.quantile(mcls[coast], 0.30))
        act = coast & (mcls < q)
        if act.sum() < 200:
            act = coast
    else:
        act = coast
    d_out = distance_transform_edt(~land).astype(np.float32)
    gd = gaussian_filter(d_out, 3.0)
    gy, gx = np.gradient(gd)               # 指向离岸（距离增大）方向

    seg = binary_dilation(act, np.ones((9, 9), bool))
    slab, nseg = _lbl(seg)
    cand = []
    for i in range(1, nseg + 1):
        ys, xs = np.nonzero(act & (slab == i))
        if ys.size >= 80:
            cand.append((ys, xs))
    if not cand:
        return out
    rng.shuffle(cand)

    ys_all, xs_all = np.nonzero(coast)
    tree = cKDTree(np.c_[ys_all.astype(np.float64), xs_all.astype(np.float64)])
    n_arc = 0; n_isl = 0; n_drop = 0
    anchors = []          # v90: 已用弧链锚点（行, 列），用于弧链间隔离
    for ys, xs in cand:
        if n_arc >= n_arcs:
            break
        p = int(rng.randint(ys.size))
        cy0, cx0 = float(ys[p]), float(xs[p])
        # v90: 两条弧链落在同一岸段/同一片海时，岛会互相穿插成一团"芝麻"。
        #   要求新弧链锚点距已有锚点 ≥260px（经度方向按周期回绕）。
        if any(np.hypot(cy0 - ay, min(abs(cx0 - ax), S - abs(cx0 - ax))) < 260.0
               for ay, ax in anchors):
            continue
        # v90: 弧链间距 62~105 → 46~72px。真实弧（千岛/琉球）在 2048 尺度上岛间
        #   只剩几 px 海峡，旧间距让整条弧读成"虚线"。
        gap = rng.uniform(46.0, 72.0)
        sgn = 1.0 if rng.rand() < 0.5 else -1.0
        base_d = rng.uniform(60.0, 135.0)
        n_islands = int(rng.randint(4, 8))
        # v90: 整条弧先盖在草稿 buf 上，凑够 3 座才提交 —— 旧版中途 break 时
        #   已经盖下的 1~2 座会留在图上（没有弧脊、没有邻居），就是"散岛"。
        buf = out.copy()
        swell = np.zeros_like(land)
        placed = 0
        for k in range(n_islands):
            iy, ix = int(round(cy0)), int(round(cx0)) % S
            if not (0 <= iy < S):
                break
            ny_, nx_ = float(gy[iy, ix]), float(gx[iy, ix])
            nl = float(np.hypot(nx_, ny_)) + 1e-9
            ny_, nx_ = ny_ / nl, nx_ / nl
            dist = base_d + rng.normal(0.0, 8.0)
            sy = cy0 + ny_ * dist
            sx = (cx0 + nx_ * dist) % S
            syi, sxi = int(round(sy)), int(round(sx))
            if not (0 <= syi < S) or buf[syi, sxi] > -260.0 or d_out[syi, sxi] < 25:
                # 太浅/太贴岸 → 换成更深的位置（外移 25%）
                sy = cy0 + ny_ * dist * 1.25
                sx = (cx0 + nx_ * dist * 1.25) % S
                syi, sxi = int(round(sy)), int(round(sx))
                if not (0 <= syi < S) or buf[syi, sxi] > -260.0:
                    break
            # v81b 放大（目视复核）：v81 首版 L 8~20 / W 2.2~4.5 / peak 120~900 在
            #   1024 预览里只剩 4~10px 的"芝麻点"，读不出"列岛"。按日本/千岛弧
            #   实测（本州 1300km ≈ 66px）取中段放大到 L 15~34、W 3.2~6.5，
            #   峰高抬到 260~1250m（屋久岛 1935m、富士 3776m 的低配版）。
            # v81e 修（目视复核 /tmp/v81f_overview.png + /tmp/v81g_needle.png）：
            #   上一版 L 15~34 / W 3.2~6.5 的长短轴比 = 4.6~5.2，出水的等深线椭圆
            #   就是 **2~5px 宽、24~54px 长的"刀片"**——放大裁切里整条弧链像一排
            #   平行刀刃，比"撒芝麻"更像算法。真实岛弧单岛（九州 450km×220km、
            #   台湾 394×144km）长宽比只有 2.0~2.8。现改为 L 14~28 / W 5.2~9.0
            #   （比 ≈ 2.2~3.2），出水区仍是椭圆但两端变钝、可读作岛。
            L = rng.uniform(16.0, 30.0) * (1.0 - 0.22 * k / max(n_islands - 1, 1))
            W = rng.uniform(5.2, 9.0)
            peak = rng.uniform(260.0, 1250.0)
            ty, tx = -nx_, ny_                     # 切向
            # v81 修（实测）：原先用「海底深度 + 高斯幅值」的加法，出水区只在峰顶
            #   一小片（exp(-r²)=|depth|/(|depth|+peak) → 仅 35% 半径），再被海底
            #   ±110m 起伏切成碎片：3 条弧 14 座岛竟生成 45 个独立陆块。改成**绝对
            #   高程穹顶** peak·e^(−r²) − 80·r²，与海底取 max：出水区≈设计椭圆。
            # v81c：穹顶本身改由 _stamp_island_dome 生成 —— v81b 出的岛全是完美同心
            #   椭圆（目视裁切像一排鸡蛋），该函数给轮廓叠双尺度噪声 + 偏心 + 小瓣。
            _stamp_island_dome(buf, syi, sxi, L, W, peak, ty, tx, rng, slope=80.0)
            # v81d 修（实测）：swell 的窗口仍沿用穹顶的 pad=L*1.8+4 —— 而 swell 的
            #   沿轴半宽是 (L+14)，在窗口边缘还剩 exp(−(1.8L+4)²/(L+14)²) ≈ 0.25，
            #   归一化后是 **125m 的方形陡坎**（探针 /tmp/probe_straight.py 量到
            #   row1078 一条 162px 长、+39.9m 的横向直边；渲染色阶上就是方块）。
            #   改为单独取 3.6 倍半宽的外框，边缘幅值 <1e-6。
            spad = int(3.6 * max(L + 14.0, 12.0)) + 4
            sy0, sy1 = max(0, syi - spad), min(S, syi + spad + 1)
            sx0, sx1 = max(0, sxi - spad), min(S, sxi + spad + 1)
            salong = (np.arange(sx0, sx1)[None, :] - sxi) * tx + \
                     (np.arange(sy0, sy1)[:, None] - syi) * ty
            sacross = (np.arange(sx0, sx1)[None, :] - sxi) * (-ty) + \
                      (np.arange(sy0, sy1)[:, None] - syi) * tx
            swell[sy0:sy1, sx0:sx1] = np.maximum(
                swell[sy0:sy1, sx0:sx1],
                np.exp(-(sacross / 12.0) ** 2 - (salong / (L + 14.0)) ** 2))
            placed += 1
            # 沿切向前进一步，再投影回岸线
            qy = cy0 + ty * gap * sgn
            qx = cx0 + tx * gap * sgn
            _, idx = tree.query([qy, qx])
            cy0, cx0 = float(ys_all[idx]), float(xs_all[idx])
        if placed >= 3:
            n_arc += 1
            n_isl += placed
            anchors.append((cy0, cx0))
            sw = gaussian_filter(swell.astype(np.float32), 3.0)
            sw = 500.0 * sw / (float(sw.max()) + 1e-8)
            # v81 修（实测）：σ=3 归一化后 sw 峰 +500m，直接加到 −201m 的海底会
            #   −201+500 = +299m → 弧脊两侧凭空冒出 215 个 1px 陆点（+45 个
            #   <800px 块）。弧脊是**海底隆起**，不许越过 −80m，故加完后统一钳位。
            # v81d：旧的 np.where(out < −200, sw, 0) 在 −200m 等深线上是**硬台阶**
            #   （sw 在此若仍有几十~上百米，就沿等深线绕出一圈"墙"）。改为 300m
            #   宽的线性闸门：−200m 处 0、−500m 及更深 1，过渡连续。
            _gate = np.clip((-200.0 - buf) / 300.0, 0.0, 1.0).astype(np.float32)
            _add = (sw * _gate).astype(np.float32)
            buf = buf + _add
            buf = np.where(_add > 0.0, np.minimum(buf, -80.0), buf).astype(np.float32)
            out = buf
        else:
            n_drop += 1                # v90: 不足 3 座的半条弧直接丢弃

    if verbose:
        print(f"      [v81 岛弧] {n_arc} 条 / {n_isl} 座 "
              f"(丢弃半链 {n_drop})", flush=True)
    return out.astype(np.float32)


def add_hotspot_island_chain(dem_m, lats=None, rng=None, n_chains=2,
                             verbose=True):
    """热点岛链（夏威夷型）：沿直线排列、尺寸与高度沿链单调衰减（v81）。

    改了什么
    --------
    在**开阔洋**（离任何陆地 >150px、水深 <−3200m）随机取起点，沿固定方位
    放 5~7 座岛，间距 30~48px；主岛长 13~22px、峰高 750~1300m，沿链按
    (1−0.62f)/(1−0.80f) **单调衰减**（热点固定、板块漂移 → 老岛被侵蚀变小）；
    链下方加 σ=8px、+450m 的宽缓隆起（热点隆起）。

    为什么
    ------
    用户 v81："没有热点岛链（如夏威夷）"。原 ``add_hotspot_chains`` 的岛半径
    1.6~3.2px、间距 4~8px，在 2048 图上是一串 2px 的点，肉眼只能看到
    "一排噪点"，读不出"链"。

    实测
    ----
    seed=7 / 2048²：见调用处 ``[v81 热点] N 链 / M 座``。

    Returns
    -------
    ndarray (S, S) float32
    """
    out = dem_m.astype(np.float32).copy()
    land = out > 0
    S = out.shape[0]
    rng = rng if rng is not None else np.random.RandomState(0)
    d_land = distance_transform_edt(~land).astype(np.float32)
    n_done = 0; n_isl = 0; n_drop = 0
    tries = 0
    anchors = []          # v90: 已提交链的全部岛心（行, 列），用于链间隔离
    while n_done < n_chains and tries < 400:
        tries += 1
        y = int(rng.randint(160, S - 160))
        x = int(rng.randint(0, S))
        if land[y, x] or d_land[y, x] < 150.0 or out[y, x] > -3200.0:
            continue
        # v90: 旧版两条热点链可以在同一片开阔洋里相距 ~150px 平行铺开，热点隆起
        #   (σ=30px) 与岛上小瓣互相穿插，900px 总览上是一团白点而不是两条链。
        #   新链起点距任何已放岛心 ≥300px（经度按周期回绕）。
        if any(np.hypot(y - ay, min(abs(x - ax), S - abs(x - ax))) < 300.0
               for ay, ax in anchors):
            continue
        ang = rng.uniform(0.0, 2.0 * np.pi)
        dy, dx = float(np.sin(ang)), float(np.cos(ang))
        n_islands = int(rng.randint(5, 8))
        gap = rng.uniform(30.0, 48.0)
        # v81b 放大（与岛弧同因）：L0 9~14 / P0 500~900 在 1024 预览里是 4~7px 的点。
        #   夏威夷大岛 150km ≈ 8px、茂宜 75km ≈ 4px —— 真实值本就小，但整条链要
        #   在图上"成串可读"，故主岛取 13~22px（250~430km，接近大岛+毛伊合并视觉），
        #   衰减放缓到 0.62/0.80，老年岛才不会缩成 1px。
        L0 = rng.uniform(13.0, 22.0)
        P0 = rng.uniform(750.0, 1300.0)
        # v90: 草稿缓冲 —— 凑不够 3 座的半条链不提交（否则留在图上的是没有
        #   热点隆起、没有邻居的孤立"鸡蛋"，正是"撒芝麻"的来源）。
        buf = out.copy()
        swell = np.zeros_like(land)
        placed = 0
        pts = []
        for k in range(n_islands):
            f = k / max(n_islands - 1, 1)
            # v81c: 间距也上抖动（真实热点链受洋壳走滑/海山叠置影响，不会等距）
            jit = 0.0 if k == 0 else float(rng.normal(0.0, 0.13 * gap))
            cy = int(round(y + dy * (gap * k + jit)))
            cx = int(round(x + dx * (gap * k + jit))) % S
            if not (0 <= cy < S) or land[cy, cx] or d_land[cy, cx] < 40.0:
                break
            L = L0 * (1.0 - 0.62 * f)
            # v81e 修（同 add_island_arc_chains，见该处注释）：0.50·L 的长宽比同样
            #   把热点岛压成"刀片"。夏威夷大岛 150×110km（比 1.36）、茂宜 75×40km
            #   （比 1.9），改 0.62·L 且下限抬到 4.2px，老年岛仍是圆丘而不是尖刺。
            W = max(4.2, 0.62 * L)
            peak = P0 * (1.0 - 0.80 * f)
            # v90: 旧版 ty,tx = (−dx, dy) 是**垂直**于链方向 —— 每座岛被拉成
            #   横放在链上的"鸡蛋"，一串鸡蛋在总览上就是一团白点（实测裁切
            #   /tmp/p12_chain_zoom.png）。热点链的岛是板块沿链向漂移拉长的，
            #   长轴必须**沿链**：ty,tx = (dy, dx)。
            ty, tx = dy, dx
            # v81d 修（同 add_island_arc_chains）：热点隆起的 σ=30px，旧窗口只有
            #   L*2.2+5 ≈ 33px，框边还剩 30% 幅值 → 归一化后 ~135m 的方形陡坎。
            #   外框改 3.6σ=110px，边缘幅值 <1e-6。
            spad = 110
            y0, y1 = max(0, cy - spad), min(S, cy + spad + 1)
            x0 = max(0, cx - spad); x1 = min(S, cx + spad + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            # 与岛弧同一修法（见 add_island_arc_chains 的实测注释）：
            #   绝对高程穹顶 + max，保证每座岛是一个完整连通的小陆块。
            # v81c：改由 _stamp_island_dome 生成 —— v81b 的热点链是一排完美同心
            #   椭圆（目视裁切像鸡蛋），这里是"算法感"最刺眼的一处。
            _stamp_island_dome(buf, cy, cx, L, W, peak, ty, tx, rng, slope=150.0)
            swell[y0:y1, x0:x1] = np.maximum(
                swell[y0:y1, x0:x1],
                np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (30.0 ** 2))))
            placed += 1
            pts.append((cy, cx))
        if placed >= 3:
            n_done += 1
            n_isl += placed
            anchors.extend(pts)
            sw = gaussian_filter(swell.astype(np.float32), 8.0)
            sw = 450.0 * sw / (float(sw.max()) + 1e-8)
            # 同 add_island_arc_chains：热点隆起不许穿出海面（否则 −200~−450m 的
            #   海底被 +450m 顶成散点陆块）；闸门同样改 300m 宽线性过渡（v81d）。
            _gate = np.clip((-200.0 - buf) / 300.0, 0.0, 1.0).astype(np.float32)
            _add = (sw * _gate).astype(np.float32)
            buf = buf + _add
            buf = np.where(_add > 0.0, np.minimum(buf, -80.0), buf).astype(np.float32)
            out = buf
        else:
            n_drop += 1                # v90: 不足 3 座的半条链直接丢弃
    if verbose:
        print(f"      [v81 热点] {n_done} 链 / {n_isl} 座 "
              f"(丢弃半链 {n_drop})", flush=True)
    return out.astype(np.float32)


def despeckle_islets_m(dem_m, min_px=40, fill_m=-18.0, verbose=True):
    """删除米制 DEM 上 < ``min_px`` 的孤立陆块（海岸单像素噪点）。

    改了什么 / 为什么 / 实测
    -----------------------
    改了什么：v79 在 ``create_continental_shelf`` 之后新增一次孤立小陆块清理，
    把面积 < ``min_px`` 的连通陆块压成 ``fill_m``（陆架内缘 -18m，与周边浅水齐平）。

    为什么：v78/v79 的海岸重塑（各向同性高斯 + 定向运动模糊 + 面积守恒阈值）
    在岸线拐点处会漏下 **1~7px 的单像素孤点**。实测 v77→v79 的 ``<800px`` 陆块
    数从 4 涨到 18，其中 14 个只有 1~4px、到最近大块距离 1.4px —— 它们不是岛，
    是二值化在平滑面上的数值毛刺。用户第 1/2 条批评的「碎纸片」正是这类东西：
    真实小岛（本图最小 622px）没有 1px 的。

    实测（seed=7, 2048）：删掉 15 块 / 24px，``<800px`` 陆块 18→4（回到 v77 水平），
    陆地占比 27.90%→27.90%（这 24px 占全图 0.0006%），海岸线像素 -38px。
    不用 ``minimum_filter`` 腐蚀：单像素点周围是陆架浅水，腐蚀会把 -18m 换成更深
    的值，形成假坑；直接赋 ``fill_m`` 才是"该点是海"的正确写法。

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        米制高程（陆 > 0）。
    min_px : int
        删除阈值（像素）。40px ≈ 0.001% 陆地，远小于本图最小真实岛（622px）。
    fill_m : float
        被删像素的新高程（米），应 > -200（仍在陆架带内）。

    Returns
    -------
    ndarray (S, S) float32
    """
    land = dem_m > 0
    lab, n = ndi_label(land)
    if n == 0:
        return dem_m
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    drop_ids = np.nonzero((sz > 0) & (sz < min_px))[0]
    if drop_ids.size == 0:
        return dem_m
    kill = np.isin(lab, drop_ids)
    out = dem_m.astype(np.float32).copy()
    out[kill] = fill_m
    if verbose:
        print(f"      [despeckle] 删除 {drop_ids.size} 个 <{min_px}px 孤立陆块 "
              f"({int(kill.sum())}px)", flush=True)
    return out


def roughen_coastline(dem_m, lats, rng, amp_px=3.2, verbose=True):
    """沿岸带做**平滑域扭曲**，消除"钢笔描出来的矢量海岸线"（用户 v82 第 1 条）。

    改了什么
    --------
    取海陆符号距离 ``sd = +edt(海) on 陆 / −edt(陆) on 海``，只在 ``|sd| ≲ 25px``
    的岸带内用 ``exp(−0.5·(sd/12)²)`` 加权，把整块 DEM 沿两个正交方向各平移
    ``±amp_px``（多尺度 fBm：51px 大摆 + 17px 岩石岬角 + 6px 碎裂）。
    用**重采样**而不是给高程加噪声：拓扑与剖面梯度都保持不变，只把 0m
    等值线本身扭弯，因此不会凭空造出孤立小岛或内陆湖（加噪声版会）。

    为什么
    ------
    用户：「海岸线像用钢笔工具拉出来的平滑矢量曲线，缺乏粗糙的岩石感 /
    沙滩的平缓感 / 断层切割的锋利感」。实测终稿 2048 的海岸线在 10px 窗内
    曲率几乎恒定（这正是 ``carve_coastal_inlets`` 只能做 150 条长水道的
    原因：它改的是"有没有水道"，没改"岸线本身的曲率分布"）。

    Returns
    -------
    ndarray (S, S) float32
        扭曲后的米制高程（内陆/远洋像素保持不变）。
    """
    S = dem_m.shape[0]
    land = dem_m > 0
    if land.all() or not land.any():
        return dem_m
    # 注意 distance_transform_edt 求的是「到最近的 0 的距离」：
    #   陆地像素自身为 1 → 到海的距离 = edt(land)；
    #   海洋像素自身为 0 → 到陆的距离 = edt(~land)。
    # （v85 首版把两支写反，edt 在两个分支上都返回 0 → sd≡0 → 权重恒为 1，
    #   整幅图被 ±3.2px 无差别扭曲，实测改动 55% 全图。这里修正。）
    sd = np.where(land, distance_transform_edt(land),
                  -distance_transform_edt(~land)).astype(np.float32)
    w = np.exp(-0.5 * (sd / 12.0) ** 2).astype(np.float32)   # ±25px 内有效
    if float(w.max()) < 0.5:
        return dem_m

    def _n(f, seed):
        v = value_fbm(S, f, 3, rng=np.random.RandomState(seed))
        return np.clip(v / 1.6, -1.6, 1.6)

    nx = 0.55 * _n(60.0, 9911) + 0.30 * _n(180.0, 9912) + 0.15 * _n(360.0, 9913)
    ny = 0.55 * _n(60.0, 9914) + 0.30 * _n(180.0, 9915) + 0.15 * _n(360.0, 9916)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    out = map_coordinates(dem_m.astype(np.float32),
                          [yy + amp_px * nx * w, xx + amp_px * ny * w],
                          order=1, mode="nearest").astype(np.float32)
    if verbose:
        moved = int((np.abs(out - dem_m) > 5.0).sum())
        print(f"      [coast-rough] 沿岸域扭曲 ±{amp_px:.1f}px，"
              f"改动 {moved}px（{moved / max(1, S * S) * 100:.2f}% 全图）", flush=True)
    return out


def despike_coastline(dem_m, passes=2, verbose=True):
    """削掉海岸线上的**单像素针尖**与单像素凹缺（用户 v82 第 2 条）。

    改了什么
    --------
    对全图做 ``passes`` 轮 3×3 中值判类：只要某像素的 3×3 中值高程与它自身
    分属海/陆两侧，就把该像素高程替换为中值。单像素凸起（周围 8 邻全是海）
    与单像素针孔（周围 8 邻全是陆）都会被一次抹平；高程取邻域中值，因此不会
    留下突兀的孤立高程。

    为什么
    ------
    用户：「放大看大陆边缘，有很多极其尖锐、像针一样的三角形小突起……
    像劣质 3D 建模没处理好的接缝」。实测终稿 2048：海岸 1px 凸起 300+ 处、
    1px 针孔 200+ 处，全部来自上游 ``sculpt/reshape`` 的高斯阈值等值线与
    ``carve_coastal_inlets`` 的 1~3px 水道尖角 —— 前者在 σ≈4px 的平滑场上取
    0m 等值线，锐角处会漏出单像素；后者在入海口切开 1px 缝，缝尖就是一个针尖。
    真实海岸的细碎来自**数十像素尺度的岬角/礁石**，没有 1px 的针。

    实测（seed=42 / 2048）：1px 凸起 318 → 0，1px 针孔 214 → 3，
    陆地占比变化 < 0.002%。
    """
    out = dem_m.astype(np.float32).copy()
    n_fix = 0
    # v88（用户 v82 第 2 条「海岸线上有很多像针一样的三角形小突起」仍可见）：
    #   3×3 中值只能清**孤立单像素**。3~5px 长的针其中段在 3×3 窗口里有 3 个陆地
    #   邻居 → 中值判类仍是陆地 → 整根针存活（实测针尖计数 156→169，不降反升）。
    #   这里先用 5×5 陆地计数做形态学去针：计数 ≤6 的陆地像素（1px 宽、长度 ≲4px
    #   的细针）直接判海；计数 ≥19 的海像素（单像素针孔）判陆。只在 ±3.5px 岸带内
    #   生效，避免啃掉真实岬角与内陆细长半岛。最后仍保留原中值轮收口。
    from scipy.ndimage import distance_transform_edt as _edt
    from scipy.ndimage import uniform_filter as _uf
    land0 = out > 0
    if land0.any() and not land0.all():
        _sd = np.where(land0, _edt(land0), -_edt(~land0)).astype(np.float32)
        near = np.abs(_sd) <= 3.5
        cnt = _uf(land0.astype(np.float32), size=5, mode="nearest") * 25.0
        med3 = median_filter(out, size=3, mode="nearest")
        spike = land0 & near & (cnt <= 6.0)
        hole = (~land0) & near & (cnt >= 19.0)
        k = int(spike.sum() + hole.sum())
        if k:
            out[spike] = np.minimum(med3[spike], np.float32(-30.0))
            out[hole] = np.maximum(med3[hole], np.float32(30.0))
            n_fix += k
    for _ in range(int(passes)):
        land = out > 0
        med = median_filter(out, size=3, mode="nearest")
        flip = (med > 0) != land
        k = int(flip.sum())
        if k == 0:
            break
        out[flip] = med[flip]
        n_fix += k
    if verbose and n_fix:
        print(f"      [despike] 海岸针尖/针孔修复 {n_fix}px", flush=True)
    return out


def sink_isolated_shoals(dem_m, max_px=3000, min_dist_px=2.0, taper_px=8.0,
                         min_target_m=-600.0, verbose=True,
                         shallow_max_m=-200.0):
    """把**不贴岸**的孤立浅水台地（-200~0m）压进周边海底，消除"海上碎渣"。

    改了什么 / 为什么 / 实测
    -----------------------
    改了什么：v79b 在 ``despeckle_islets_m`` 之后新增一步：找出 ``-200 < dem <= 0``
    且到最近陆地 > ``min_dist_px``、面积 < ``max_px`` 的连通浅水块，逐块压到
    "自身外环 8px 的海底中位深度"（不低于 ``min_target_m``），外围用 ``taper_px``
    平滑过渡，不留浅色细环。

    为什么：轮廓图上左下与中部有一堆漂浮在深蓝里的浅蓝碎斑。实测 2048 终稿：
    ``-200~0m`` 的浅水块共 84 个，其中 **71 个不贴岸**（合计 23857px = 全图
    0.57%，最大 2014px ≈ 1200km 见方），到最近陆地 9~153px。真实海洋里浅水台地
    几乎只贴着大陆（陆架），孤立的只有凯尔盖朗/福克兰等个位数海底高原；这 71 个
    是板块模拟的微陆块残留，在二值渲染下正是用户第 1 条说的"被撕碎的纸片 /
    掉落的碎渣"。

    实测（seed=7, 2048）：孤立浅水块 71→14（剩下的是面积 ≥3000px 的真实台地），
    浅水总面积 8.07%→7.72%，陆地占比不变（只动海）。

    v80g: 新增 ``shallow_max_m``（默认 −200m，与旧行为一致）。第二次调用时传
    −1200m + ``max_px`` 26000 + ``min_dist_px`` 30 + ``min_target_m`` −4200，
    用来清掉用户第 1 条里的**宽阔**无源浅台：实测开阔洋（离陆 >60px）仍有
    14 块 2000~21000px、顶 −0~−600m 的浅色台地（最大一块 21319px 顶 −0m），
    深度色阶上就是"深蓝里的一块浅青斑"。第一次调用只处理 −200~0m 且
    >2px 贴岸的小碎渣，管不到这种又宽又深的台地；而且旧 ``min_target_m``
    −600m 会把台地只压到 −600m（仍然明显偏浅），故第二次单独传 −4200。
    """
    from scipy.ndimage import binary_dilation

    land = dem_m > 0
    shoal = (dem_m > shallow_max_m) & ~land
    if not shoal.any():
        return dem_m
    lab, n = ndi_label(shoal)
    if n == 0:
        return dem_m
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    dland = distance_transform_edt(~land)
    dmin = np.full(n + 1, np.inf, np.float64)
    np.minimum.at(dmin, lab.ravel(), dland.ravel())
    ids = [i for i in range(1, n + 1)
           if 0 < sz[i] < max_px and dmin[i] > min_dist_px]
    if not ids:
        return dem_m
    out = dem_m.astype(np.float32).copy()
    nz = value_fbm(out.shape[0], 12.0, 3, rng=np.random.RandomState(791))
    n_px = 0
    for i in ids:
        m = lab == i
        ring = binary_dilation(m, iterations=8) & ~m
        if not ring.any():
            continue
        tgt = max(min(float(np.median(out[ring])), -600.0), min_target_m)
        # 权重: 块内 1 → 块外 taper_px 内线性衰减到 0（不留浅色细环）
        d_out = distance_transform_edt(~m)
        ext = binary_dilation(m, iterations=int(taper_px))
        w = np.clip(1.0 - d_out / float(taper_px), 0.0, 1.0)
        w = w * w * (3.0 - 2.0 * w)
        blend = tgt + 60.0 * np.clip(nz, -1.5, 1.5)
        out[ext] = (out[ext] * (1.0 - w[ext])
                    + blend[ext].astype(np.float32) * w[ext]).astype(np.float32)
        n_px += int(m.sum())
    if verbose:
        print(f"      [shoals] 压深 {len(ids)} 个孤立浅水台地 ({n_px}px) "
              f"→ 海底中位深度", flush=True)
    return out


def add_seamount_chains(dem_m, lats, rng, n_chains=3, verbose=True):
    """沿"热点轨迹"在深海放置线性海山链（夏威夷 / 土阿莫土型）。

    改了什么 / 为什么 / 实测
    -----------------------
    改了什么：v80 新增。在深水（``dem<-3400`` 且离陆 >130px）随机取起点与
    走向，沿直线每隔 22~34px 放一座高斯锥海山（半径 5~11px）；锥顶高程沿链
    递减（热点最新的海山最高、越老越被侵蚀沉降），每 4 座做一次 ±0.35rad
    的走向折转，模拟洋壳走滑段。

    为什么：用户 v80 第 1/5 条「深海散落大量无源浅蓝斑块，分布随机且密集，
    缺海底火山链的线性排列特征」「碎岛群带强烈随机生成意味，没有经历板块
    碰撞或张裂」。真实海山 90% 以上集中在热点轨迹与岛弧上，呈**等间距线性
    排列**；此前的浅水孤斑是板块微陆块残留，既不线性也不成链。

    实测（seed=7 / 2048）：新增 3 条链 / 26 座海山，相邻间距 26~38px
    （≈510~745km，与夏威夷—天皇海山链同量级），锥顶 −210~−1300m。

    Parameters
    ----------
    n_chains : int
        海山链条数；每条 6~9 座。
    """
    S = dem_m.shape[0]
    land = dem_m > 0
    dland = distance_transform_edt(~land).astype(np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    out = dem_m.astype(np.float32).copy()
    lat_ok = (np.abs(lats)[:, None] < 60.0) if lats is not None \
        else np.ones((S, 1), bool)
    cand = np.argwhere((dem_m < -3400) & (dland > 130) & lat_ok)
    if cand.size == 0:
        return out
    n_made = 0
    # v80c: 旧版"每条链只掷一次起点"，撞上洋中脊/浅台地就 miss≥3 提前 break，
    #   实测 3 条链只落下 9 座（每链 3 座，链长 <80px，视觉上仍是散点）。
    #   现改为：每次尝试先攒成 cone 列表，落够 6 座才算一条链；不够就换起点重掷
    #   （每链最多 8 次）。只有达标的链才真正写进 DEM，避免失败尝试留下零散锥体。
    cones = []
    for _ in range(n_chains):
        best = []
        for _attempt in range(8):
            p = cand[rng.randint(len(cand))].astype(np.float64)
            th = rng.uniform(0.0, 2.0 * np.pi)
            n_seam = rng.randint(6, 10)
            # v80h: 间距 22~34 → 26~38px，半径 5~10 → 7~12px，锥顶 −900~−220 →
            #   −1250~−450m。实测 v80g 的 crop_A 里每座海山是"亮青圆点 + 双圈光环"
            #   （锥顶 −220m 比洋底浅 4800m，在 800m 连续色带上跨了 6 档，等于
            #   一盏灯）。真实热点海山多数峰顶仍在 −1000~−2000m，只有少数盖约特
            #   接近海面；放大半径 + 压低峰顶后读作"海底隆起串"，而不是发光弹珠。
            step = rng.uniform(26.0, 38.0)
            r0 = rng.uniform(7.0, 12.0)
            top0 = rng.uniform(-1250.0, -450.0)    # 链首（最新）锥顶
            got = []
            miss = 0
            # v80h: 间距/半径/峰顶逐座抖动（步长 ±22%、半径 0.72~1.30×、
            #   峰顶 ±120m）。实测 v80h 之前的 crop_A 里整条链是**等距等大的圆点
            #   串珠**（步长恒定 26~38px、半径恒定、峰顶按 12% 线性衰减），
            #   一眼就是程序生成的。真实热点轨迹的火山大小/间距/高度都随岩浆
            #   供给起伏，绝不会有 8 座同尺寸同间距的锥体。
            cy_acc, cx_acc = float(p[0]), float(p[1])
            for k in range(n_seam):
                if k and k % 4 == 0:
                    th += rng.normal(0.0, 0.35)
                st = step * rng.uniform(0.78, 1.22)
                cy_acc += np.sin(th) * st
                cx_acc += np.cos(th) * st
                yi, xi = int(round(cy_acc)), int(round(cx_acc))
                if not (0 <= yi < S and 0 <= xi < S):
                    break
                # 跳过落在岛链附近或已经变浅的洋底，连续 3 次才终止
                if dland[yi, xi] < 90 or dem_m[yi, xi] > -2600:
                    miss += 1
                    if miss >= 3:
                        break
                    continue
                miss = 0
                # 只抬不压：锥顶 = 沿链线性衰减（越老越深）
                got.append((cy_acc, cx_acc,
                            top0 * (1.0 - 0.12 * k) + rng.normal(0.0, 120.0),
                            r0 * rng.uniform(0.72, 1.30)))
            if len(got) > len(best):
                best = got
            if len(best) >= 6:
                break
        cones.extend(best)
    for (cy, cx, top, r0) in cones:
        # v80f: 旧版 ``rad = int(r0) + 5`` 把高斯锥**硬截断**在轴对齐方框边缘 ——
        #   实测 r0=5 时框边处 w=exp(−0.5·(10/5)²)=0.135，而框外立刻归零，
        #   于是高原相差 4300m 的锥体在框边留下 **±580m 的方形陡坎**；
        #   出图按 2000m 分色后，每座海山都套着一个"圆点 + 方形光晕"
        #   （/tmp/crop_A_topleft.png 里最明显）。现改为 rad≈3.2·r0，
        #   框边 w=exp(−0.5·3.2²)=0.006（≈26m），再减去边值归一化，
        #   使锥体在框边严格归零、过渡连续 —— 方形光晕消失。
        rad = int(3.2 * r0) + 2
        yi, xi = int(round(cy)), int(round(cx))
        y0, y1 = max(0, yi - rad), min(S, yi + rad + 1)
        x0, x1 = max(0, xi - rad), min(S, xi + rad + 1)
        dy = yy[y0:y1, x0:x1] - cy
        dx = xx[y0:y1, x0:x1] - cx
        # v81c: 纯径向高斯在任何色阶下都必然是**一圈圈同心圆**（出图目视就是
        #   "靶心/弹珠"，比没有海山更假）。给归一化半径叠一层 σ≈4px 的零均值噪声
        #   （±32%，按 min(u,2) 随半径增强），光环被撕成不规则闭合线，读作"海底隆起"。
        hh, ww = dy.shape
        nsm = value_fbm(max(hh, ww), 4.0, 2, rng=rng)[:hh, :ww]
        u = np.sqrt(dy * dy + dx * dx) / max(r0, 1e-6)
        u = u * (1.0 + 0.32 * np.clip(nsm, -1.5, 1.5) * np.minimum(u, 2.0))
        w = np.exp(-0.5 * u * u).astype(np.float32)
        w_edge = float(np.exp(-0.5 * (rad / r0) ** 2))
        w = np.clip((w - w_edge) / (1.0 - w_edge), 0.0, 1.0)
        # v82: w_edge 归一化只对**未扭曲**的 u 成立；噪声扭曲（最低 −0.48·u）后
        #   窗口边缘 w 仍可达 0.25，窗口被图幅边界裁断时更严重 —— 实测列 2022
        #   有 58px 长、跳变 4501m 的轴对齐垂直台阶。给窗口边补 10px smoothstep
        #   渐隐，强制锥体在矩形框边归零。
        _ay = np.minimum(np.arange(hh), hh - 1 - np.arange(hh)).astype(np.float32)
        _ax = np.minimum(np.arange(ww), ww - 1 - np.arange(ww)).astype(np.float32)
        _ty = np.clip(_ay / 10.0, 0.0, 1.0)
        _tx = np.clip(_ax / 10.0, 0.0, 1.0)
        _ty = _ty * _ty * (3.0 - 2.0 * _ty)
        _tx = _tx * _tx * (3.0 - 2.0 * _tx)
        w = w * _ty[:, None] * _tx[None, :]
        sub = out[y0:y1, x0:x1]
        sub += np.maximum(top - sub, 0.0) * w
        n_made += 1
    if verbose:
        print(f"      [seamount] 热点海山链 {n_chains} 条 / {n_made} 座", flush=True)
    return out


def add_ocean_ridge_system(dem_m, lats, rng, n_ridges=2, verbose=True):
    """在开阔洋中部添加洋中脊系统（对称隆起 + 轴部裂谷 + 转换断层错断）。

    改了什么 / 为什么 / 实测
    -----------------------
    改了什么：v80 新增。随机游走生成 1~2 条 1200~1800px 的脊轴折线
    （每 8px 一步、航向 ±3.2°，每 18 步做一次 8~22px 的**垂直错断**模拟
    转换断层），栅格化成 1px 掩膜后取欧氏距离场，叠加：
      侧翼隆起 ``+2400m·exp(-(d/95)²)``、轴部裂谷 ``-1000m·exp(-(d/9)²)``。
    只作用于离陆 >55px、自身 <−1200m 的深水，并在 60px 内渐隐；脊顶硬性
    钳在 −900m（不露出海面）。

    为什么：用户第 2 条「深海与次深海交界非常破碎，纯二维噪声切成几段，
    没有洋中脊/海沟这类线性构造」。地球洋中脊全长 6.5 万 km、宽 1000~4000km、
    脊顶比深海平原浅 2~3km，是大洋里唯一的一级构造线；轴部还有 1~2km 深的
    裂谷。缺少它，海洋读起来就只能是"噪点染色"。

    实测（seed=7 / 2048，n_ridges=3）：脊轴 3 条 / 合计 4448px，脊顶 −2410m，
    同纬度深海平原 −5120m，轴部裂谷比脊顶再低 980m。

    Parameters
    ----------
    n_ridges : int
        脊轴条数；每条长度 1200~1800px（≈2.4~3.5 万 km，地球的 1/2~1/3）。
    """
    S = dem_m.shape[0]
    land = dem_m > 0
    dland = distance_transform_edt(~land).astype(np.float32)
    lat_ok = (np.abs(lats)[:, None] < 62.0) if lats is not None \
        else np.ones((S, 1), bool)
    _pad = 200
    _inner = np.zeros((S, S), bool)
    _inner[_pad:S - _pad, _pad:S - _pad] = True
    cand = np.argwhere((dem_m < -3600) & (dland > 120) & lat_ok & _inner)
    if cand.size == 0:
        return dem_m
    axis = np.zeros((S, S), bool)
    ridges = []                       # 每条脊的折线顶点 (y, x)，供转换断层取向
    n_pts = 0
    n_made = 0
    # v80h: 把 for 改成 while —— 走废了就换一个起点重来。
    #   改了什么：每条脊的随机游走失败（落点不足 65%）后不再直接放弃，而是重新
    #     抽起点重试，直到够 n_ridges 条或试满 n_ridges×8 次。
    #   为什么：v80g 在它前面加了第二次 sink_isolated_shoals（宽阈值压深），洋底
    #     整体变深 → `cand`（<−3600m 且离陆 >120px）的集合变了 → rng 抽到的起点
    #     不同 → 其中一条脊一路贴到岛链（dland<45 连续 25 步）提前收尾被丢弃。
    #     实测日志从 v80f「3 条 / 脊轴 4440px / 脊顶带 137974px / 基底 −5192m」
    #     掉到 v80g「2 条 / 2480px / 48583px / −5568m」——洋底变深、脊顶带缩水，
    #     等于把用户第 2 条（"缺绵延数万公里的大洋中脊"）又还回去了。
    #   重试后条数只取决于"有没有地方放得下"，不再取决于一次抽样的运气。
    tries = 0
    max_tries = n_ridges * 8
    while n_made < n_ridges and tries < max_tries:
        tries += 1
        p = cand[rng.randint(len(cand))].astype(np.float64)
        th = rng.uniform(0.0, 2.0 * np.pi)
        n_steps = rng.randint(180, 260)          # ×8px = 1440~2080px
        pts = []
        miss = 0
        for k in range(n_steps):
            th += rng.normal(0.0, 0.032)
            p = p + np.array([np.sin(th), np.cos(th)]) * 8.0
            # v80b: 越界不再 break（旧版一条脊常在中途撞到图边，实测只剩 73px），
            #        改为沿边界反射，让脊轴能横贯整幅图。
            if not (0 <= p[0] < S - 1 and 0 <= p[1] < S - 1):
                p[0] = np.clip(p[0], 1.0, S - 2.0)
                p[1] = np.clip(p[1], 1.0, S - 2.0)
                th += np.pi
            yi, xi = int(round(p[0])), int(round(p[1]))
            if dland[yi, xi] < 45:
                miss += 1                        # 短暂穿到岛链附近则绕行
                if miss >= 25:
                    break
                th += 0.6
                continue
            miss = 0
            pts.append((p[0], p[1]))
            if k % 18 == 17:                     # 转换断层错断
                off = rng.choice([-1.0, 1.0]) * rng.uniform(8.0, 22.0)
                p = p + np.array([np.cos(th), -np.sin(th)]) * off
        # v80c: 门槛从写死的 200px 改成"占计划步数的 65% 且至少 120px"。
        #   实测 seed=7 → RandomState(809)：ridge0 got=153/n_steps=180、
        #   ridge1 got=181/n_steps=181，两条都被 200 阈值丢弃 → n_pts=0 →
        #   整个函数早返回，日志里连 [ridge] 都不打印（用户第 2 条批评完全没被
        #   修复）。相对门槛既保住"够长才算一条脊"，又不会因 n_steps 随机偏小
        #   而全盘失败。
        if len(pts) >= max(120, int(0.65 * n_steps)):
            n_made += 1
            n_pts += len(pts)
            ridges.append(np.asarray(pts, np.float64))
    if not ridges:
        return dem_m
    # v80h: 脊轴折线先做 3 次 [0.25,0.5,0.25] 平滑（等效 σ≈10px≈200km）。
    #   改了什么：随机游走每 8px 取一个顶点，航向每步 ±3.2° 抖动，每 18 步还有
    #     一次 8~22px 的垂直错断；栅格化后脊轴是一串 45°/90° 的折角。
    #   为什么：距离场绕这些折角生成的等深带在出图上读作"楼梯 + 横档"
    #     （crop_A 里那条浅色折线），而真实洋中脊是长波弯曲的连续隆起，
    #     转换断层的错断只在**等深线**上表现为横向错开，脊轴本身不断。
    #   实测：平滑后拐点曲率半径由 0（折角）变成 ~20px，楼梯感消失。
    for _i, _pts in enumerate(ridges):
        if len(_pts) >= 9:
            _sm = _pts.copy()
            for _ in range(3):
                _sm[1:-1] = (0.25 * _sm[:-2] + 0.5 * _sm[1:-1]
                             + 0.25 * _sm[2:])
            ridges[_i] = _sm

    # v80e: 折线**逐段插值**后再栅格化。
    #   改了什么：旧版每走 8px 只把端点写进 axis，1px 宽的掩膜于是变成"串珠"，
    #     出图上是断断续续的对角虚线（实测 3 条脊只拿到 4440 个孤立点，取距离场
    #     后脊顶带被拉成锯齿）。现按 0.5px 步长在相邻顶点间线性插值填满。
    #   为什么：洋中脊是连续的一级构造线，脊顶带必须是一条不断开的等深带。
    for pts in ridges:
        for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.hypot(y1 - y0, x1 - x0) * 2.0))
            ys = np.rint(np.linspace(y0, y1, n)).astype(np.int32)
            xs = np.rint(np.linspace(x0, x1, n)).astype(np.int32)
            axis[np.clip(ys, 0, S - 1), np.clip(xs, 0, S - 1)] = True
    d = distance_transform_edt(~axis).astype(np.float32)
    # ---- v80e: 转换断层（fracture zone）----
    # 改了什么：新增 20~40 条**垂直于脊轴**的线性构造，长 140~380px。
    # 为什么：用户第 2 条「海洋深浅分布没有海底地形的起伏逻辑」。真实洋底除了
    #   脊轴，还有大量近于垂直脊轴、长达上千公里的转换断层：它把洋壳切成一条条
    #   年龄不等的"条带"，在海图上就是一条条笔直的深度陡坎（大西洋的 Vema /
    #   Romanche 断层落差 >2km）。只靠年龄-深度场只能得到沿脊平行的一维渐变，
    #   断层是唯一能给出"横向构造线"的机制。
    #
    # v80f 重写：旧版把断层写成**半平面阶跃**（``step − 0.5`` 在整个
    #   2·(half+6) ≈ 150~390px 的方形框内一次抬/降 ±300~700m），实测 22 条
    #   断层在洋底铺出几十块 380×380px 的轴对齐矩形色块 —— 出图上就是
    #   /tmp/crop_A_topleft.png 中那种"弥散污渍"和阶梯状直角边界。
    #   真实转换断层在海底地形上的**可视线索是窄槽**（宽 2~5px、深 500~2000m），
    #   两侧的年龄错断只体现为缓变、不是硬台阶。故改为：
    #     ① 窄槽 ``−amp·exp(−0.5·(s/1.8)²)``（s = 垂直断层的坐标，σ=1.8px）；
    #     ② 残留错断幅度压到 ±180m（只为让槽两侧略有高差，不跨色阶）。
    fz = np.zeros((S, S), np.float32)
    n_fz = 0
    for pts in ridges:
        # v84（用户 v82 第 3 条「深蓝海洋里隐约可见十字交叉的笔直细线」）：
        #   断层条数 len//24（一条脊 20~30 条）→ len//40。22 条 140~380px 的
        #   完全笔直的窄槽在深海里纵横交错，出图上就是规则网格；减到 ~12 条后
        #   稀疏得读不出"网格"，但仍保留横向构造线。
        k_fz = max(2, len(pts) // 40)
        picks = rng.choice(len(pts), size=min(k_fz, len(pts)), replace=False)
        for i in picks:
            i0, i1 = max(int(i) - 2, 0), min(int(i) + 2, len(pts) - 1)
            ty, tx = pts[i1] - pts[i0]
            tn = float(np.hypot(ty, tx))
            if tn < 1e-6:
                continue
            ty, tx = ty / tn, tx / tn
            ny, nx = -tx, ty                     # 垂直于脊轴 = 断层走向
            cy, cx = pts[i]
            half = rng.uniform(70.0, 190.0)
            # v80h: 槽深 600~1600 → 380~950，σ 1.8 → 2.6，残留错断 80~180 → 55~115。
            #   实测 v80g 出图（/tmp/crop_A_topleft.png）里这些断层是**贯穿半幅图
            #   的黑色划痕**：1.8px 宽的 −1600m 硬槽在 1024 预览里不足 1px，
            #   采样后只剩一条纯黑细线，比真实转换断层（宽 2~5px、深 500~2000m，
            #   且两侧有 1km 级缓坡）刺眼得多。加宽 + 变浅后它读作"海底沟槽"
            #   而不是"划痕"，同时仍能在等深线上留下横向构造线。
            # v84: v83 的 σ=2.6px / −380~950m 窄槽在 2048 图上仍只有 5px 宽，
            #   出图读作"铅笔划痕"（用户第 3 条「笔直的细线像坐标网格」）。
            #   真实转换断层在地形图上是**宽缓的沟槽**（10~30px）而非刻线：
            #   槽深降到 120~260m（不足一条 800m 等深线）、σ 2.6→9.0px、
            #   两侧渐隐 22→34px。这样它只在连续色带上留下极淡的横向明暗，
            #   不再产生可辨认的直线。
            amp = rng.uniform(120.0, 260.0)      # 宽缓沟槽深度
            step_amp = rng.uniform(30.0, 55.0) * rng.choice([-1.0, 1.0])
            # v83 修（用户 v82「深海直上直下的矩形像素块 / 十字格网」）：
            #   旧版把残留错断写成半平面阶跃 ``0.5·tanh(s/6)``，而窗口 ``win``
            #   只沿断层走向 t 收尾、沿 s 方向**没有收尾** → 每个断层在
            #   ``r = half+6`` 的**轴对齐方框**里铺出一整块 ±55~115m 的恒定高差
            #   平台，方框四边在海底被硬切成 2·half（140~380px）长的笔直陡坎。
            #   22 条断层叠出来就是用户看到的"竖直条纹 + 矩形色块 + 十字格网"，
            #   而且因为窗口是 axis-aligned，这些边永远平行于图像坐标轴。
            #   现改为：① 沿 s 方向也用 smoothstep 收尾（半宽 half_s=22px），
            #   阶跃在 s=±22px 内平滑归零，不再有硬边；② 阶跃幅度压到 35~70m
            #   且乘 win_s（两侧趋零），只剩"槽两边略有高差"的读感；
            #   ③ 包围盒改为覆盖 (half_t, half_s) 的最小外接圆，避免方形裁切。
            half_s = 34.0
            r = int(np.hypot(half, half_s)) + 6
            y0b, y1b = max(0, int(cy) - r), min(S, int(cy) + r + 1)
            x0b, x1b = max(0, int(cx) - r), min(S, int(cx) + r + 1)
            gy, gx = np.mgrid[y0b:y1b, x0b:x1b]
            gy = gy - cy
            gx = gx - cx
            s = gx * ny + gy * nx                # 垂直断层走向的距离
            t = gx * tx + gy * ty                # 沿断层走向的距离
            win = np.clip(1.0 - (t / half) ** 2, 0.0, 1.0)
            win = win * win * (3.0 - 2.0 * win)
            win_s = np.clip(1.0 - (s / half_s) ** 2, 0.0, 1.0)
            win_s = win_s * win_s * (3.0 - 2.0 * win_s)
            trough = np.exp(-0.5 * (s / 9.0) ** 2)
            step = 0.5 * np.tanh(np.clip(s / 14.0, -3.0, 3.0))
            fz[y0b:y1b, x0b:x1b] += (
                (-amp * trough + step_amp * step) * win * win_s
            ).astype(np.float32)
            n_fz += 1
    # v80d: 用"海底扩张年龄—深度关系"重建开阔洋基底，而不再是在噪声上叠一条脊。
    #   改了什么：删掉旧的 +2400m 高斯侧翼 / −1000m 轴部叠加，改成把"离陆 >70px
    #     且水深 >400m"的海域整体替换为
    #       z = −(2550 + 3300·(1−e^(−√t/7.5)))，t = d_px·19.6km ÷ 25km/Myr
    #         ≈ 0.784·d_px（百万年，Parsons & Sclater 1977 半空间冷却 + 平台期）。
    #   为什么：用户第 2 条原话「海洋深浅分布纯粹是二维噪声图被切成了几段，没有
    #     海底地形的起伏逻辑」。地球洋底深度是**单调的年龄函数**：脊顶 −2500m，
    #     100px（≈2000km ≈ 80Ma）处 −5100m，250px 以外稳定在 −6000m 深海平原。
    #     只有按这个单调场重建，洋中脊 → 侧翼 → 深海平原 → 大陆隆才各就各位；
    #     同时它顺手清掉了外海残留的无源浅水斑块（用户第 1 条）。
    #   v80e 关键修正：旧版把轴部裂谷写成 σ=11px（≈215km）的 −880m，恰好在脊轴上
    #     把脊顶从 −2550m 压到 −3430m —— 实测整条脊只比洋底浅 ~700m，落进同一个
    #     2000m 色阶，出图上就是一条看不见的虚线（用户第 2 条等于没修）。现改为
    #     σ=1.4px（≈30km，与东太平洋海隆裂谷同量级）的 −160m 细沟，脊顶回到 −2710m。
    # v84（用户 v82 第 3 条「深海里隐约的十字交叉笔直细线」）：脊顶过窄。
    #   旧版 age = 0.784·d 在 d=0 处斜率无穷大（√age），脊顶 −2550m 到 8px 外
    #   −3480m 掉了 930m —— 一条 8~10px 宽的亮带被等深线勾成一圈，出图上就是
    #   一根笔直的亮线。现改为 age = 0.784·√(d²+25)，脊顶 −2550 → −3320m，
    #   但半宽从 ~10px 展宽到 ~35px（≈700km，东太平洋海隆的真实尺度），
    #   出图读作"一条宽缓的浅色隆起带"，不再是刻线。
    age = 0.784 * np.sqrt(d * d + 25.0)
    zt = -(2550.0 + 3300.0 * (1.0 - np.exp(-np.sqrt(age) / 7.5)))
    zt = zt - 220.0 * np.exp(-0.5 * (d / 3.0) ** 2)                  # 轴部裂谷
    zt = zt + fz                                                     # 转换断层
    # v80e: 盆地尺度深度起伏 ±390m（σ=320px ≈ 6300km）。没有它，年龄-深度场在
    #   远离所有脊的地方收敛到同一个 −5800m 平台，整片开阔洋是一块纯色；真实
    #   太平洋深海平原 −5500~−6000、印度洋 −4000~−5000、大西洋 −4500~−5500。
    # v80f: 细尺度起伏从 ±280m（σ=22px）压到 ±110m（σ=40px）。
    #   改前实测：洋底基底 −5160m，±280m 让大片像素在 −5200m 分色线的两侧来回
    #   翻越，出图上整片大洋是均匀"电视雪花"（/tmp/crop_A_topleft.png 第一版）；
    #   真实深海丘陵只有 50~250m 起伏、波长 10~30px（200~600km），
    #   压到 ±110m 后仍保留质感但不再跨色阶。
    zt = zt + 260.0 * np.clip(
        value_fbm(S, 320.0, 3, rng=np.random.RandomState(8813)), -1.5, 1.5)
    zt = zt + 55.0 * np.clip(
        value_fbm(S, 40.0, 3, rng=np.random.RandomState(8812)), -2.0, 2.0)
    # 大陆隆约束：离陆 <160px 的海域至少 −4200~−4800m（地球陆坡坡脚 500km≈25px
    #   就进入深海平原），保证"环绕大陆的深海平原"，也避免洋脊贴着海岸隆起。
    near_land = np.clip(1.0 - (dland - 40.0) / 120.0, 0.0, 1.0)
    zt = np.where(near_land > 0.02,
                  np.minimum(zt, -4200.0 - 600.0 * near_land), zt)
    zt = np.clip(zt, -5850.0, -2400.0).astype(np.float32)
    w_open = (np.clip((dland - 70.0) / 80.0, 0.0, 1.0)
              * np.clip((-dem_m - 400.0) / 900.0, 0.0, 1.0)).astype(np.float32)
    out = (dem_m * (1.0 - w_open) + zt * w_open).astype(np.float32)
    out = np.where(dem_m > 0, dem_m, out).astype(np.float32)   # 陆地像素零改动
    if verbose:
        # v80c: n_pts 记的是"落点步数"（每步 8px），旧版日志把它当像素数直出，
        #   2 条脊显示成 "338px"，实际轴长约 338×8 ≈ 2700px。这里换算成像素。
        _oo = (dland > 120) & (out < 0)
        _c = (d < 24.0) & (out < 0) & (dland > 120)
        print(f"      [ridge] 洋中脊 {n_made} 条 / 脊轴约 {n_pts * 8}px | "
              f"转换断层 {n_fz} 条 | 脊顶带(离轴<24px) {int(_c.sum())}px "
              f"中位 {np.median(out[_c]):.0f}m | 开阔洋基底 "
              f"{np.median(out[_oo]):.0f}m（年龄-深度场，最浅 "
              f"{out[_oo].max():.0f}m）", flush=True)
    return out.astype(np.float32)


def carve_margin_trenches(dem_m, lats, rng, max_depth_m=-10600.0, verbose=True,
                          mcls=None):
    """沿**主动大陆边缘**外侧雕刻窄而深的海沟（俯冲带），与窄陆架同址。

    改了什么 / 为什么 / 实测
    -----------------------
    改了什么：v80 新增。取 ``margin_class_map`` 的系数（越小 = 越主动），
    映射成 0~1 的门（<0.62 全开），σ=18px 平滑保证沿岸连续，再乘一个
    σ=60px 的低频门把海沟切成**分段**（真实海沟被转换断层/海岭截断）。
    在离岸 7~40px、高程 <−700m 的带内下切
    ``4200m·exp(-0.5·((d−17)/6)²)·gate``，下限 ``max_depth_m``。

    为什么：用户第 2 条「海洋缺线性延展的深海海沟（通常在岛弧外侧）」，
    以及第 1 条「太平洋边缘几乎没有大陆架」。海沟是主动边缘的标志性
    构造：宽 40~120km（2~6px）、深 6~11km，紧贴陆坡坡脚。

    实测（seed=7 / 2048）：岸带取点 320458px / 下切 44833px，最深 −9344m，
    所在岸段陆架宽度 p50 = 2.7px（被动岸段 p50 = 11.4px）。

    Parameters
    ----------
    max_depth_m : float
        海沟底高程下限（地球马里亚纳 −10994m）。
    """
    S = dem_m.shape[0]
    land = dem_m > 0
    sea = ~land
    dland = distance_transform_edt(~land).astype(np.float32)
    # v80g: 与 create_continental_shelf 共用冻结场（管线内由 generate_world 传入），
    #   保证"窄陆架"与"海沟"必然贴着同一段主动边缘；None 时退回现场估算。
    mcls = margin_class_map(dem_m) if mcls is None else np.asarray(mcls, np.float32)
    # v80b: 门限 0.95→0.62、宽度 0.33→0.22。旧门限把"中等偏主动"的岸段也
    #   算进来，实测 <−7000m 从 1841px 暴涨到 21208px / 71 段（碎成筛子）。
    # v80h: 绝对门限改成**分位数**门限。mcls 的绝对值依赖全图深海占比（本世界
    #   重标后 p50≈0.97），写死 0.62 依然会在半数岸线上开沟 —— 实测海沟带
    #   334257px（16% 全图），crop_B 里每条海岸都套一圈深蓝描边，这是比"外发光"
    #   更假的新描边。真实地球只有环太平洋/印尼/加勒比/斯科舍等少数俯冲带，
    #   占洋面 <2%。现取沿岸 40px 带内 mcls 的 22% 分位当门限、向上 0.18 过渡，
    #   保证"只切最主动的那 ~1/5 岸段"，且不随全球深度标定漂移。
    _coast = (dland > 0) & (dland < 40)
    if _coast.any():
        q_lo = float(np.quantile(mcls[_coast], 0.22))
    else:
        q_lo = 0.62
    q_hi = q_lo + 0.18
    gate = np.clip((q_hi - mcls) / max(q_hi - q_lo, 1e-3), 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    gate = gaussian_filter(gate.astype(np.float32), 26.0)
    # 分段门：σ=200px 的低频场做 smoothstep 开关，只切出 3~5 段长海沟，
    # 而不是把海沟打成 60 多个小碎段。
    seg = value_fbm(S, 200.0, 3, rng=rng)
    sg = np.clip((seg + 0.15) / 0.50, 0.0, 1.0)
    gate = np.clip(gate * (sg * sg * (3.0 - 2.0 * sg)), 0.0, 1.0)
    # v86（用户 v84/v85「海岸外一圈近黑描边」→ v85 改完变成「沿岸成团的墨渍」）：
    #   v85 把海沟峰值按**离岸距离**外移到 dland=20 / σ=9、band 扩到 46px 之后，
    #   在弯曲海岸上必然失败：陆架宽的岸段坡脚在 30px 外、窄岸段在 8px 内，
    #   同一圈 dland 落点忽深忽浅，再乘"本来就深"的深度门，海沟就碎成一串斑块
    #   （实测 /tmp/ab_coast_outline.png 下半幅：离岸 20~40px 的深色团块）。
    #   现改为**按深度定位**：海沟轴固定在 −2200m（= 陆坡坡脚，真实海沟位置），
    #   σ=650m ≈ 2.2px。等深线在空间上天然连续 → 海沟是沿坡脚的一整条窄槽；
    #   外侧 prof 在 −4200m 已衰减到 1%，槽壁自然回升，形成真正的 V 形而非洼地。
    band = (dland > 2.0) & (dland < 90.0)
    n_px = int(band.sum())
    if n_px == 0:
        return dem_m
    prof = np.exp(-0.5 * ((dem_m + 2200.0) / 650.0) ** 2)
    # v84（用户 v82 第 4 条「右侧大陆中间偏下有一个近乎直角的向内凹陷缺口」）：
    #   历史（v84）：旧版 band 的内边界是硬阈值 ``dland>6`` —— 而 prof 在 dland=6 处
    #   仍有 exp(-0.5·(-8/5.5)²)=0.35，即**贴着这条硬边界一次下切 1260m**。实测
    #   y=590 / x≈1572 的剖面是 ``… −6093  0  0  0 …``：一侧被挖到 −6km 、
    #   隔 1px 就是平坦的 0m 陆架，出图上就是刀切的直角缺口与方形台面。
    #   v86 的 _rin 已在离岸 2→8px 内渐变到 0，band 内也不再有任何硬深度门。
    _rin = np.clip((dland - 2.0) / 6.0, 0.0, 1.0)
    _rin = _rin * _rin * (3.0 - 2.0 * _rin)
    _rout = np.clip((90.0 - dland) / 30.0, 0.0, 1.0)
    _rout = _rout * _rout * (3.0 - 2.0 * _rout)
    # v86: 深度门 _dg 已删 —— 定位改由 prof 的深度高斯完成（浅陆架上 prof≈0.007，
    #   下切量 <30m，无需额外的硬门；硬门只会在 −800m 等深线上留下一条台阶）。
    drop = (3600.0 * prof * gate * _rin * _rout).astype(np.float32)
    out = dem_m.astype(np.float32).copy()
    out[band] = out[band] - drop[band]
    out = np.maximum(out, np.float32(max_depth_m))
    # 被下切的区域用 1.2px 低通抹掉 1px 台阶（海沟内部保持陡壁）
    sm = gaussian_filter(out, 1.2)
    w = np.clip(drop / 1500.0, 0.0, 1.0)
    out = (out * (1.0 - 0.3 * w) + sm * (0.3 * w)).astype(np.float32)
    # v80c: 1.2px 低通把邻近陆地正值混进 band，且 band 外侧 w 虽小但 sm 可达数千米，
    #   实测仍凭空多出 5 个陆块（25 → 30，<800px 由 1 → 6）。这里对"雕刻前是海水"
    #   的全部像素钳回海平面以下 —— 海沟只会下切，任何抬升都不该造陆。
    out[sea] = np.minimum(out[sea], np.float32(-1e-3))
    if verbose:
        deep = int((out < -7000).sum())
        # v80h: 旧日志打的是 band 大小（离岸 6~34px 且已在 −600m 以下的所有像素），
        #   与真正被下切的面积无关 —— 实测 334257px 会让人误以为海沟占了 8% 全图。
        n_cut = int((drop > 800.0).sum())
        print(f"      [trench] 实际下切(>800m) {n_cut}px / 岸带 {n_px}px，"
              f"最深 {out.min():.0f}m，<−7000m {deep}px", flush=True)
    return out


def despeckle(v01, min_area=48):
    """去斑：清除面积过小的陆地块与海洋孔洞（联通分量域）。

    原理：
        1. 对陆地（v ≥ 0.5）做连通域标注，若某连通域像素数 < ``min_area``，
           则以 9×9 最小值滤波覆盖（把小岛腐蚀成海洋，不产生洞）。
        2. 再对海洋（v < 0.5）做同等处理，用 9×9 最大值滤波回填（清除
           陆地内部的小淡水湖 / 盐沼噪点）。

    Parameters
    ----------
    v01 : ndarray (S, S), float32, 值域 [0, 1]
        归一化高程（v01 = m2v(dem_m)）。
    min_area : int
        最小联通域面积（像素）。v6.6 默认 48 → 去斑后为 32（保住岛弧小岛）。

    Returns
    -------
    ndarray (S, S), float32
        去斑后的归一化高程。
    """
    v = v01.copy()
    land = v >= 0.5
    lab, n = ndi_label(land)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        drop = np.isin(lab, np.nonzero(sz < min_area)[0])
        v[drop] = minimum_filter(v, 9)[drop]
    lab, n = ndi_label(~land)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        fill = np.isin(lab, np.nonzero(sz < min_area)[0])
        v[fill] = maximum_filter(v, 9)[fill]
    return v.astype(np.float32)


def _hi_stat(dem_m, tag, beltM=None):
    """开发诊断: >2000m 高山占陆地比 / 连通块 PCA 拉长比 AR / 构造带归属率。

    只打印, 不参与任何计算。地球参考: >2000m ≈ 8% 陆地, 大块 AR ≈ 6.4。
    """
    hi = dem_m > 2000.0
    tot = int((dem_m > 0).sum())
    n = int(hi.sum())
    if n == 0 or tot == 0:
        print(f"      [hi] {tag}: >2000m 0", flush=True)
        return
    lab, _k = ndi_label(hi)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    big = np.nonzero(sz >= 400)[0]
    ars = []
    for i in big:
        ys, xs = np.nonzero(lab == i)
        ys = ys - ys.mean(); xs = xs - xs.mean()
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]))))
        ars.append(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
    ar = float(np.mean(ars)) if ars else 0.0
    share = f" | belt_share {float(beltM[hi].mean()):.2f}" if beltM is not None else ""
    print(f"      [hi] {tag}: >2000m {n / tot * 100:.2f}% 陆地 | 块 {len(ars)} "
          f"| AR {ar:.2f}{share}", flush=True)


def global_envelope_match(v01, bm, bstd, bmin, bmax, glac, block=8,
                          blend=0.60, seed=0, cap_m=8200.0):
    """全局一次 CDF 匹配 + 逐像素块包络裁剪 + 行星峰值截断 + 冰盖低 blend 保纹理。

    GAN 生成的原始 ``[0, 1]`` 归一化高程会有整体分布失真（峰不够高、洋不够深）
    的问题。本函数把它按分位对齐到一个更接近地球的"参考分布"，同时在每个粗块
    上用 ``bmin/bmax`` 做软包络约束（避免 GAN 在单块上产生跳变极端值），并在
    冰川覆盖区降低 blend 比例（避免把冰盖内部纹理磨平）。

    算法简述：
        1. 以像素为单位从 ``bm/bstd/bmin/bmax`` 采样一个"合理值"：
           ``s = Normal(bm, bstd)`` → 夹到 ``[bmin, bmax]``（每像素独立样本）。
        2. 把 ``s`` 与 ``v01`` 同时按排序（ranked）配对 → 得到基于 CDF 的
           ``matched``（单调，保留 v01 空间结构）。
        3. 块包络（上采样后的 bmin/bmax）软夹一次 + 行星峰值 ``cap_m`` 硬钳。
        4. 与原始 v01 按 blend 线性混合（冰盖区 blend 下调 0.30）。

    Parameters
    ----------
    v01 : ndarray (H, W), float32, [0, 1]
        GAN 原始输出。
    bm, bstd, bmin, bmax : ndarray (W//block, H//block), float32
        粗块均值 / 标准差 / 最小值 / 最大值（来自 ``world_to_channels``）。
    glac : ndarray (同粗块尺寸)
        每块冰川覆盖率（0~1），用于降低高冰盖区的 blend。
    block : int
        块大小（训练时 cfg.block；v5=8）。与粗块尺寸匹配的上采样基。
    blend : float
        线性混合权重（1 = 完全匹配替换，0 = 完全保留 GAN 输出）。v6.6 默认 0.60。
    seed : int
        用于生成"合理值"参考样本的随机种子。
    cap_m : float
        行星绝对峰值（米），v6.6 取 8200m（先 m2v 再钳制）。

    Returns
    -------
    ndarray (H, W), float32, [0, 1]
        匹配 + 夹取 + 混合后的归一化高程。
    """
    rng = np.random.default_rng(seed)
    h, w = v01.shape
    n = h * w
    idx = rng.integers(0, bm.size, n)
    s = rng.normal(bm.ravel()[idx], bstd.ravel()[idx] + 1e-4)
    s = np.clip(s, bmin.ravel()[idx], bmax.ravel()[idx])
    ts = np.sort(s)
    order = np.argsort(v01, axis=None)
    ranked = np.empty(n, np.int64); ranked[order] = np.arange(n)
    matched = ts[ranked].reshape(h, w)
    bmin_up = np.clip(ndi_zoom(bmin, block, order=1)[:h, :w], 0, 1)
    bmax_up = np.clip(ndi_zoom(bmax, block, order=1)[:h, :w], 0, 1)
    matched = np.minimum(np.clip(matched, bmin_up, bmax_up), m2v(cap_m))
    glac_up = ndi_zoom(glac, block, order=1)[:h, :w]
    bl = np.clip(blend - 0.30 * glac_up, 0.30, blend)
    return np.clip(bl * matched + (1 - bl) * v01, 0, 1).astype(np.float32)


# ===========================================================================
# v55: Terrain Web 美术常量（atlas 风格）
# ---------------------------------------------------------------------------
# 改了什么: 新增 Terrain Web 的 atlas 调色板与两条 LUT（陆地 256 项高程色带 /
#   海洋 256 项深度色带）。数值逐字节照抄参考实现，仅把 RGB 整数换成 0~1 float。
# 来源: Opuscule/backend/apps/maps/terrain_web.py 的 _PALETTES["atlas"]、
#   _build_elevation_lut()(L817) 与 ocean_lut 构建(L1283-1290)；前端着色器
#   Opuscule/frontend/src/components/MapTerrainCanvas.tsx 的 factor 公式一致。
# 为什么: 用户对比 Terrain Web 后要求"使用一样的美术效果"。旧管线用 Whittaker
#   ​群系离散色（9 类）再叠 5 类地貌纹理，全陆主色 >6 种且相邻群系像素级硬切；
#   Terrain Web 全陆只有 beach→grass→forest→rock→snow 一条连续色带，山链因此
#   天然连成整体。
# ===========================================================================
_TW_COLORS = {
    # v62: 暗端整体提亮（deep 7,34,64→10,48,88；abyss 3,17,36→5,26,52）。
    # 实测旧值把 pd 中位（-4234m）整片压进 abyss，900 预览里大洋是一块黑板，
    # 洋中脊/海山/海沟全糊成同一个黑色；提亮后四档色阶的实际亮度差从 ~6 提到 ~14。
    # v78: deep (11,54,100) → (28,92,148)、abyss (7,36,72) → (14,54,102)、
    #   trench (3,18,40) → (5,24,54)、mid (16,78,124) → (26,104,152)。
    # 为什么: 用户两次反馈「海洋底部没什么起伏」。实测（world_map × DEM，开阔洋
    #   离岸>60px、|lat|<55°）亮度 p10/p50/p90 = 24.2/30.4/39.2 —— **整片大洋只占
    #   15 个亮度级**，而 DEM 在 64px 窗口内的真实高差中位数是 1462m。问题不在几何
    #   （DEM 侧 σ64 高通 std=633m），而在配色：底色中位仅 30/255，±20% 的晕渲
    #   只有 ±6 级，10~40px 的海底丘陵在 8bit 上根本落不到一个可辨刻度。
    #   提亮深海端后同一套晕渲/色调可拿到 2~3 倍的绝对对比（模拟 lum std 5.0→9.3，
    #   σ20 起伏响应 6.0→10.8 lum/1000m）。浅海端 shallow 不动，避免岸带发亮。
    "deep":    np.array([28, 92, 148], np.float32) / 255.0,
    # v76: abyss (5,26,52) → (7,38,74)、trench (2,12,26) → (3,18,40)。
    # 为什么: 1:1 实测（probe8）开阔洋面的"霉斑"本体是 deep (10,48,88) 的
    #   **大块等色区**，背景是 deep→abyss 的混色 (6,32,60)。也就是说霉斑的
    #   观感来自 deep/abyss 两档之间 **1.8 倍的亮度台阶** 落在 100~300px 的
    #   平滑过渡上 —— 这正是"云"的读法。把两档的台阶压到 1.25 倍后，
    #   海底起伏改由 v76 的细颗粒晕渲承担（见下方 rel_fine），分层设色只负责
    #   "越深越静的"整体倾向，不再切出可见的色盘。
    "abyss":   np.array([14, 54, 102], np.float32) / 255.0,
    "trench":  np.array([5, 24, 54], np.float32) / 255.0,
    "mid":     np.array([26, 104, 152], np.float32) / 255.0,
    # v63: 近岸浅青压暗降饱和（27,112,145 → 22,86,114）。
    # 改了什么: 只动 shallow 与 foam 权重。为什么: 1:1 复核时每条海岸都是一圈
    #   **等宽发光青边**（用户「像贴纸描边」）—— (27,112,145) 与陆地绿的明度差太大，
    #   且 pd<0.035 的色段铺满整个 0~250m 陆架，于是整条岸线变成一条霓虹灯带。
    #   压暗后浅海仍比深海亮一档（海底地形可读），但不再从陆地上"跳出来"。
    "shallow": np.array([22, 86, 114], np.float32) / 255.0,
    "foam":    np.array([180, 221, 220], np.float32) / 255.0,
    # v76: (232,213,166) → (214,199,158)。沙色亮度过高时，一条 6~10px 的岸线
    #   与深海底色并置就是"霓虹描边"；压 8% 后仍比草地亮，但不再自发光。
    "beach":   np.array([214, 199, 158], np.float32) / 255.0,
    "grass":   np.array([124, 166, 83], np.float32) / 255.0,
    "forest":  np.array([57, 112, 62], np.float32) / 255.0,
    "rock":    np.array([145, 130, 105], np.float32) / 255.0,
    "rock_hi": np.array([104, 100, 94], np.float32) / 255.0,
    "snow":    np.array([245, 247, 242], np.float32) / 255.0,
    "river":   np.array([30, 112, 140], np.float32) / 255.0,
}

_TW_LUT_CACHE = {}

# v65: 海洋深度 → LUT 索引 pd 的映射节点（米 → 0~1）。
# 改了什么: 水深归一不再用"除以单一尺度"，改为按地球海底四带定节点插值。
# 为什么: 实测本图海洋深度 p50=4228m / p75=4835 / p90=5134（world_dem.npy 口径）。
#   v64 的 clip((d-30)/4200)**0.72 让**一半以上洋面直接钳到 1.0**，整片落进 LUT
#   最暗段 —— 出图海面主色实测 (1,9,20) 占 13.0%、(8,19,32) 占 12.1%，缩略图上
#   大洋是一块黑板，海底地形（用户第 3 条）与极区冰缘（第 4 条）都被拖累。
# 节点含义: 30m 岸带 / 400m 陆架缘 / 1500m 陆坡 / 3200m 洋中脊顶 / 4400m 深海平原 /
#   5200m 盆地底 / 6200m 海沟轴。
_OC_DEPTH_KNOTS = np.array([30.0, 400.0, 1500.0, 3200.0, 4400.0, 5200.0, 6200.0],
                           np.float32)
_OC_PD_KNOTS = np.array([0.00, 0.10, 0.22, 0.36, 0.50, 0.64, 0.82], np.float32)


def _tw_smoothstep(e0, e1, x):
    """Terrain Web 的 smoothstepf: 0..1 之间的 Hermite 三次过渡。"""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _tw_mix(a, b, w):
    """标量权重或 (H,W) 权重下的两色混合（w 为逐像素场时自动加通道轴）。"""
    if np.ndim(w) == 0:
        return a * (1.0 - float(w)) + b * float(w)
    w = np.asarray(w, np.float32)[..., None]
    return a * (1.0 - w) + b * w


def _tw_put(rgb, xs, ys, col):
    """把一组散点像素刷成 ``col``（越界自动丢弃），供 v56 的 stipple 使用。

    改了什么: 新增向量化落点函数，取代参考实现里逐像素的 ``_draw_pixel``。
    为什么: 参考的树点/岩线是在 310×310 粗格上 Python 循环画的（900px 时约 3.4 万次
      落笔）；本图粗格 682×682、树点约 2.6 万棵 × 5 笔，纯 Python 循环要百万次
      ``rgb[y, x] = c``，实测会多花 2s+。散点已由 numpy 算好，这里一次花式索引
      写完；越界点先用布尔掩膜滤掉，避免负索引绕回行尾。
    """
    S = rgb.shape[0]
    ok = (xs >= 0) & (xs < S) & (ys >= 0) & (ys < S)
    rgb[ys[ok], xs[ok]] = col


# v59 色层胞体的可调参数（集中一处，便于 A/B 扫描时只改这里）
_VC_VEG_CELL = 68.0     # 植被胞宽（输出 px，2048 下 ≈30 胞/边）
_VC_VEG_AMP = 0.12      # 湿度偏移幅度（±_VC_VEG_AMP/2）；0 = 关闭
_VC_VEG_GAIN = 0.6      # 森林浓度按胞体分档的增益（乘子 1±GAIN/2）；0 = 关闭
_VC_SAND_CELL = 165.0   # 沙滩胞宽（输出 px，沙袋更长更连续）
_VC_SAND_AMP = 0.020    # 近岸 t 偏移幅度（±_VC_SAND_AMP/2）；0 = 关闭


def _tw_voronoi_cells(S, cell_px, seed, gv=512):
    """抖动网格 Voronoi 胞体值场：(S,S) float32，每胞一个常数，胞界线性过渡。

    改了什么: 新增**只服务颜色层**的胞体噪声（v59）。用户提议"用 Voronoi 胞体做底"，
      这里的落地方式是只把它接到纯色层（植被斑块 / 沙滩宽度），不碰高程与光照。
    为什么: 把 Voronoi 用在高程/光照上会引入多边形明暗台阶（胞内常数、胞界突变），
      会把上一轮刚对齐的相邻格 |Δshade| p50 0.033/p90 0.127 推回 0.3 量级，
      等于把 v58 修掉的"霉斑"请回来；而用在颜色上，得到的正是参考图那种
      有机斑块边界（现在 _moist 是 σ=8 高斯场，森林边界是一整片模糊色域）。
    实现: 抖格（每格一个胞心，抖动 ±0.42 格宽）+ 3×3 邻域找最近胞心，
      逐胞哈希取 [0,1] 值。先在 gv 分辨率上算，再双线性放大 —— 放大自带的
      4px 线性过渡就是天然的软边，胞界因此是水彩式过渡而不是硬多边形线。
    cell_px: 输出分辨率下的目标胞宽（2048 下 68px ≈ 33 胞/边）。
    """
    W = min(int(gv), int(S))
    if W < 16 or cell_px <= 0:
        return np.full((S, S), 0.5, np.float32)
    _step = S / float(W)
    c = max(2.0, float(cell_px) / _step)          # 胞宽（工作分辨率像素）
    n = int(np.ceil(W / c)) + 2
    rs = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    jx = rs.uniform(-0.42, 0.42, (n, n)).astype(np.float32)
    jy = rs.uniform(-0.42, 0.42, (n, n)).astype(np.float32)
    gyi, gxi = np.mgrid[0:n, 0:n]
    sx = (gxi.astype(np.float32) + 0.5 + jx) * c
    sy = (gyi.astype(np.float32) + 0.5 + jy) * c
    yyi, xxi = np.mgrid[0:W, 0:W]
    xx = xxi.astype(np.float32)
    yy = yyi.astype(np.float32)
    bx = np.floor(xx / c).astype(np.int32)
    by = np.floor(yy / c).astype(np.int32)
    best = np.full((W, W), np.float32(1e30), np.float32)
    val = np.full((W, W), 0.5, np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx = np.clip(bx + ox, 0, n - 1)
            cy = np.clip(by + oy, 0, n - 1)
            d = (xx - sx[cy, cx]) ** 2 + (yy - sy[cy, cx]) ** 2
            h = ((cx.astype(np.int64) * 73856093)
                 ^ (cy.astype(np.int64) * 19349663)
                 ^ (int(seed) * 83492791))
            v = ((h % 10007).astype(np.float32)) / 10006.0
            m = d < best
            best = np.where(m, d, best)
            val = np.where(m, v, val)
    val = gaussian_filter(val, 1.2).astype(np.float32)
    if W != S:
        val = ndi_zoom(val, int(round(S / float(W))), order=1)[:S, :S]
    return np.ascontiguousarray(val, dtype=np.float32)


def _terrain_web_luts():
    """返回 (elev_lut(256,3), ocean_lut(256,3))，首次调用后缓存。"""
    if _TW_LUT_CACHE:
        return _TW_LUT_CACHE["elev"], _TW_LUT_CACHE["ocean"]
    C = _TW_COLORS
    elev = np.zeros((256, 3), np.float32)
    for i in range(256):
        t = i / 255.0
        # v76: 沙滩带 t<0.05 → t<0.020，过渡起点 0.006 → 0.002。
        # 为什么: [v59] 日志实测「沙滩带 12.00% 陆地」—— t 是陆地高程分位，
        #   t<0.05 对应最低 14.8 百分位，也就是**整块沿海平原**都被涂成沙色
        #   (232,213,166)。900px 预览下每条海岸是一圈 6~10px 的亮奶油色描边，
        #   与深海藏青并置读作"发光描边"。收到 0.020 后沙滩只占最低 ~6% 陆地，
        #   岸线亮带从"描边"退回"一线沙滩"。
        if t < 0.020:
            c = _tw_mix(C["beach"], C["grass"], _tw_smoothstep(0.002, 0.020, t))
        elif t < 0.34:
            c = C["grass"]
        elif t < 0.54:
            c = _tw_mix(C["grass"], C["forest"], _tw_smoothstep(0.34, 0.54, t))
        elif t < 0.70:
            # v74: forest→rock 段从 (0.54,0.68)×0.55 拉长到 (0.54,0.70)×0.72。
            c = _tw_mix(C["forest"], C["rock"], _tw_smoothstep(0.54, 0.70, t) * 0.72)
        elif t < 0.88:
            # v74: 岩带 0.68~0.82 → 0.70~0.88。
            c = _tw_mix(C["rock"], C["rock_hi"], _tw_smoothstep(0.70, 0.88, t))
        else:
            # v62: 雪带 0.82~0.94 → 0.88~1.00，最高只混到 0.88 的白。
            # 改了什么: 旧版 t>0.94 的整条山脊都是纯雪白，1:1 里是"白色蛇形胶带"
            #   （用户截图东侧那条连续白山脊）。抬阈值 + 封顶后，只有顶峰附近有雪，
            #   山肩仍是 rock_hi 的灰褐，山体读作"岩壁 + 残雪"而不是一条白带。
            # v74: 雪线下限 0.88 → 0.90、白度 0.88 → 0.80，岩带下段同步加宽。
            # 为什么: 出图调色板最近邻统计（world_map.png 口径）实测草地 59.1% /
            #   森林 12.4% / rock 仅 2.5%，而 rock_hi 6.1% + snow 13.8% —— 山体是
            #   "顶端一大片灰白 + 几乎没有中灰岩壁"，900px 上整条山链读作一条白蛇。
            #   加宽岩带把 rock_hi 的占比换给 rock，山体第一次有"岩壁-残雪"分层。
            # A1: 删除「按陆地高程分位起混雪」—— t 是百分位排名，t>=0.90 等于
            #   「全图最高的 ~2.5% 陆地」都上雪，与纬度/绝对高程无关，于是雪帽是
            #   散点而不是雪线。雪现在由 render_world 的 A1 段按
            #   「绝对高程 vs 纬度雪线」单独合成，这里只保留最高档岩石色。
            c = C["rock_hi"]
        elev[i] = c
    # 海洋: v55c 改回参考实现的「到岸距离」分级。
    # 改了什么: 旧版按深度占比 pd=clip(-elev/4500) 取色, 实测我方大陆架极宽, 大片
    #   海域落在 0~200m → 整片浅青色, 出图海洋均值 RGB(69,97,121), 而参考海洋
    #   主色 (6,33,62) 占 45%(=纯 deep)。现按到岸距离 d(参考粗格单位, 1 格 = 2.9px)
    #   映射: 0~1.7 格 浅青→中蓝, 1.7~12 格 中蓝→藏青, >12 格 全 deep。
    # 为什么: 参考的观感是"深藏青大洋 + 细窄亮色岸带"; 若按真实水深上色, 我方宽陆架
    #   会把 40% 的海面染成浅青, 整体读作"淡蓝背景板", 与参考差距明显。
    # 索引单位: pd = d_cells / 12 ∈ [0,1]。
    # v56: 一格的实际像素宽由 6.6 改回 2.9（= 参考 build_fields 的 width//2.9）。
    #   改了什么: 旧版按 6.6px/格 折算, 浅海带被拉宽 2.27 倍(0~11px 浅青 + 11~79px
    #   中蓝), 出图海洋均值 RGB(69,98,124), 大片海面是"发光的淡青背景板"。
    #   为什么: 参考在 2048 输出下 gw=clamp(2048//2.9,·,700)=700, 每格恒 2.9 输出px,
    #   12 格 = 35px 就到全 deep。改回 2.9 后浅海带只有 0~5px、中蓝到 0~35px,
    #   大洋立刻回到参考的深藏青 (deep 7,34,64) 基调。
    # 海洋: v61b 改成「水深主导」的四段分层设色（GEBCO 读法）。
    # 改了什么: v55c~v61 的 LUT 里 mid→deep 段从 1.7/12 一直拉到 1.0，也就是索引
    #   pd>0.30 之后**全部收敛到 deep(7,34,64)**。实测深海平原(-4234m) 与海沟
    #   (-5200m) 出图亮度只差 0.012，海底地形在配色阶段就被抹平。
    # 为什么: 用户第 3 条「海洋之类的地形没有了，未来 3D 地形需要这个」。GEBCO 式
    #   海底图靠**分层设色**读地貌：大陆架浅青 → 陆坡中蓝 → 深海平原藏青 → 海沟近黑。
    #   现在把 0~6200m 均匀铺成四段 (0/0.075/0.34/0.78/1.0)，让洋中脊(-2500m)、
    #   深海平原(-4200m)、海沟(-5200m) 各落在不同色段，无晕渲也能读出脊轴走向。
    ocean = np.zeros((256, 3), np.float32)
    for i in range(256):
        pd = i / 255.0
        # v62: 色段边界重排 + 暗端提亮。实测 pd 中位 0.67（-4234m）全落在旧
        #   abyss(3,17,36) 的 0.34~0.78 段里 → 一半洋面是近黑色，"海底地形"被压成
        #   一块黑板。现改成 0.035/0.22/0.62/0.95 分段，abyss 提前到 0.62 结束、
        #   深海平原落在 deep→abyss 的中间过渡，洋中脊到盆地到海沟能读出四档灰度。
        # v63: 浅海带 0.020 → 0.060 渐变，且浅青再压暗一档。
        # 为什么: 实测海岸扫线 x=1444~1451 八列像素逐点恒为 (27,103,136) —— 这是一条
        #   8px 宽的**等色硬边带**，两侧分别是奶油色沙滩 (232,213,166) 和藏青深海，
        #   于是整条海岸读作霓虹描边。把浅→中的过渡拉宽到 pd 0~0.06 后，同一段岸线
        #   变成 8→40px 的渐隐浅色，不再是硬边色带。
        # v75: 深海两段过渡拉长 —— deep→abyss (0.24,0.62) → (0.24,0.74)、
        #   abyss→trench (0.62,0.95) → (0.74,1.00)。
        # 为什么: 900px 复核里洋面是成片 30~80px 的淡蓝团块（"水里长霉斑"）。根因不是
        #   噪声，而是 DEM 海底有大量 1500~3000m 的**圆丘状浅台**，而 deep→abyss 只用了
        #   0.38 的 pd 跨度 —— 同一片台地在 0.1 的 pd 抖动下就跨完两档色，团块边缘被
        #   画成一条柔和的明暗过渡带，读作"气泡"。把过渡摊到 0.50 后，同样的起伏
        #   只换来 3/4 的色阶变化，团块边界退成"水下台地的缓坡"而不是霉斑。
        c = _tw_mix(C["shallow"], C["mid"], _tw_smoothstep(0.00, 0.060, pd))
        c = _tw_mix(c, C["deep"], _tw_smoothstep(0.060, 0.24, pd))
        c = _tw_mix(c, C["abyss"], _tw_smoothstep(0.24, 0.74, pd))
        c = _tw_mix(c, C["trench"], _tw_smoothstep(0.74, 1.00, pd))
        c = _tw_mix(c, C["foam"], 0.010 * (1.0 - _tw_smoothstep(0.0, 0.012, pd)))
        ocean[i] = c
    _TW_LUT_CACHE["elev"], _TW_LUT_CACHE["ocean"] = elev, ocean
    return elev, ocean


def render_world(v01, lats, landS, cmap, rng, tect_ctx=None, *, _profile=False):
    """渲染入口；_profile=True 时打开逐段计时打印（开发用，不影响结果）。

    v55 美术管线 = Terrain Web（Opuscule/backend/apps/maps/terrain_web.py）风格：
        1. 陆地配色：一条连续高程色带 beach→grass→forest→rock→rock_hi→snow
           （``_terrain_web_luts()`` 的 256 项 LUT），t = 高程 / 陆地 p99.5。
        2. 光照：单一 NW 光源 ``shade=clip(0.62+0.56·light, 0.44, 1.28)``；
           detail 项（fBm 起伏 × mountain）同时扰动色带位置与光照 → 山体不是平面色块。
        3. 森林：真实降水当 moisture，m>0.48 且 t<0.72 的区域混 38% forest。
        4. 海洋：三段式深度 LUT（shallow→mid→deep）+ 岸缘 foam 带 + 海底晕渲（±6%）。
        5. 湖泊 / 河流：湖用 shallow→mid，河用 river 青蓝三档（与海同色相）。
        6. 极地冰盖 / 融水 / 海冰 / 极点平滑：物理特征，与参考实现无关，保留。
        7. v59 色层 Voronoi 胞体：只调制「森林浓度分档」与「近岸沙带宽度」两处颜色，
           高程场与光照场完全不动（理由见 _tw_voronoi_cells 与调用处注释）。
    旧管线的 Whittaker 群系离散色、5 类地貌纹理、独立雪线层已在 v55 删除
    （理由见各段注释：大陆主色过多且相邻群系像素级硬切）。

    Parameters
    ----------
    v01 : ndarray (S, S), float32, [0, 1]
        最终归一化高程（v6.6 全部后处理后的结果）。
    lats : ndarray (S,)
        行对应的纬度（像素中心对齐，递增）。
    landS : ndarray (S, S) bool
        陆地掩膜（提前缓存，避免多次计算）。
    cmap : matplotlib Colormap
        v55 起**未使用**（配色全部来自 Terrain Web LUT）。保留形参只为不破坏
        6 处既有调用点。
    rng : numpy.random.RandomState
        随机数生成器（供 fBm / 浮冰 / 降水场采样）。
    tect_ctx : dict, optional
        v55 起**未使用**（地貌纹理已删除）。保留形参以兼容既有调用点。

    Returns
    -------
    ndarray (S, S, 3), float32, [0, 1]
        RGB 图像。直接 ``plt.imsave`` 或 ``PIL.Image.fromarray((rgb*255).astype(uint8))``
        均可得到与 ``world_map.png`` 一致的 PNG 图像。
    """
    import time as _time
    _t = [0.0]
    def _mark(label):
        """性能打点：profile 开启时打印标签耗时并记录新时间戳。"""
        if _profile:
            print(f"    [render] {label}: {_time.time()-_t[-1]:.2f}s")
        _t.append(_time.time())

    S = v01.shape[0]
    land = v01 >= 0.5
    elev_m = v2m(v01)
    _t.append(_time.time())
    # v9.x: 预计算 elev_climate_smooth / hf_norm / gate (temperature_field 与
    # precipitation_field 都需要, 原本各自重复 3 个 gaussian_filter, 共 ~2s×2=4s 浪费)
    elev_pos = np.maximum(elev_m, 0.0)
    elev_climate_smooth = gaussian_filter(elev_pos, 2.2)
    hf = elev_pos - gaussian_filter(elev_pos, 0.9)
    hf = gaussian_filter(hf, 0.5)
    hf_norm = np.clip(hf / np.maximum(np.max(np.abs(hf)), 1.0), -1.0, 1.0)
    gate = np.clip((elev_climate_smooth - 800.0) / 3200.0, 0.0, 1.0)
    _mark("elev_climate_precompute")
    # v9.x: 预计算 d_sea_km (continentality_index 与 precipitation_field 共用, 省 ~1s)
    kmpp = 360.0 / max(S, 1) * 111.32
    d_sea_km = distance_transform_edt(~(elev_m < 0.0)) * kmpp
    _mark("d_sea_km_precompute")
    ci = continentality_index(elev_m, lats, _d_sea_km=d_sea_km)  # v9.x: 复用预计算
    _mark("continentality_index")
    dT, dP = ocean_current_effects(elev_m, lats)         # M4 洋流/季风气候效应
    _mark("ocean_current_effects")
    T = temperature_field(lats, elev_m, dT, ci=ci,
                           _elev_climate_smooth=elev_climate_smooth,
                           _hf_norm=hf_norm, _gate=gate)  # v9.x: 复用预计算
    _mark("temperature_field")
    P = precipitation_field(lats, elev_m, rng, dP, ci=ci,
                            _elev_climate_smooth=elev_climate_smooth,
                            _hf_norm=hf_norm, _gate=gate,
                            _d_sea_km=d_sea_km)  # v9.x: 复用预计算
    _mark("precipitation_field")
    # ---- v55: Terrain Web 美术 —— 单一 NW 光照 + 连续 hypsometric 色带 ----
    # 改了什么: 删除 Whittaker 群系离散色 / 群系边界软混合 / _hA·_hB 高程色带,
    #   改为参考实现的「一条连续高程 LUT × 单一光照 × detail 起伏调制」。
    # 为什么: 用户要求"使用 Terrain Web 一样的美术效果"。旧管线同一座山会被切成
    #   迎风坡雨林(绿) + 背风坡草原(黄) 两个色块, 再叠 5 类地貌噪声 → 全陆主色
    #   >6 种, 1024 预览里读作"随机色斑"; 参考实现全陆只有 5 个锚点一条色带,
    #   山链/盆地因高程连续而自然连成整体。
    # 光照: 严格照抄 MapTerrainCanvas.tsx —— dx*18 / dy*16 + NW 光,
    #   shade = clamp(0.62+0.56*light, 0.44, 1.28) (旧双尺度晕渲只有 ±24% 且方向
    #   随尺度漂移; 这里是 −31%..+53% 的单一方向 → 山脉走向第一次全图一致)。
    P_s = gaussian_filter(P.astype(np.float32), 1.5)
    _relief_v = max(float(np.percentile((v01 - 0.5)[land], 99.5)), 1e-3)
    _surf = ((v01 - 0.5) / _relief_v).astype(np.float32)
    # v55c: 光照改在 256 粗格上求梯度, 再双线性上采样（与参考实现的 310 粗格同构）。
    # 改了什么: 旧版在 2048 全分辨率逐像素求梯度, 每像素高差 ~0.003 → dx≈0.06,
    #   light 恒在 0.72 附近, shade≈1.02 → factor≈1.18 全图一刀切; 实测算出来的
    #   成品陆地是一片平亮绿(主色 (160,208,104) 7.9%), 与参考的斑驳质感差很远。
    # 为什么: 参考在 310×310 粗格(每格 2.9 输出px)上求梯度, 实测 shade 分位
    #   p10=0.446/p25=0.652/p50=0.893/p75=1.061, 且有 30% 触底 0.44 —— 它的地形
    #   在"每格"尺度上就有真实高差, 光照因此是亮暗交错的。降到 256 格均值后
    #   实测本图 shade p25=0.583/p50=0.814, 与参考同量级(略偏暗), factor 中位
    #   随之从 1.18 降到 ~0.80, 陆地才回到参考的深绿基调。
    # v58: 光照粗格 256→512、求梯度前先对 _surf 做 σ=S/256(2048→8px) 平滑。
    #   改了什么: 旧版直接对 256 格(_f=8px)的块平均求梯度。
    #   为什么: 用 /tmp/probe_shade_ours.py 与参考 build_fields(2048) 对拍，同一口径
    #     （粗格上相邻格 shade 差）实测:
    #       参考 700 格: shade std 0.142 / factor clip 底 6.2% 顶 21.4%
    #                    / 相邻|Δshade| p50 0.034 p90 0.152
    #       我方 256 格: shade std 0.231 / clip 底 24.2%（1/4 陆地压在暗端）
    #                    / 相邻|Δshade| p50 0.109 p90 0.450 —— 粒度粗 3 倍
    #     参考的 surface 是 Perlin 平滑场，2.9px 一格的相邻高差很小，shade 因此是
    #     "细颗粒"；我方的 8px 块之间高差可达数十米 → factor 大量撞到 0.60/1.28
    #     两个硬钳位，出图读作**成片的暗绿霉斑 + 亮米色斑块**（1:1 裁图复核）。
    #     改 512 格(4px) + σ=8 预平滑后实测相邻|Δshade| p50 0.033 p90 0.127、
    #     clip 底 24.2%→14.1%，与参考(0.034/0.152、6.2%)同一量级。
    _G = min(512, S)
    _f = max(S // _G, 1)
    if _f >= 2:
        _G = S // _f
        _small = gaussian_filter(_surf, max(1.0, S / 256.0)).reshape(
            _G, _f, _G, _f).mean(axis=(1, 3))
        _gx_s = np.zeros_like(_small)
        _gy_s = np.zeros_like(_small)
        _gx_s[:, 1:-1] = (_small[:, 2:] - _small[:, :-2]) * 18.0
        _gy_s[1:-1, :] = (_small[2:, :] - _small[:-2, :]) * 16.0
        _light_s = ((0.72 - 0.58 * _gx_s - 0.44 * _gy_s)
                    / np.sqrt(1.0 + _gx_s * _gx_s + _gy_s * _gy_s))
        shade = np.clip(0.62 + 0.56 * _light_s, 0.44, 1.28)
        shade = ndi_zoom(shade.astype(np.float32), _f, order=1)[:S, :S]
        shade = np.ascontiguousarray(shade, dtype=np.float32)
    else:
        _gy, _gx = np.gradient(_surf)
        _dx = (_gx * 18.0).astype(np.float32)
        _dy = (_gy * 16.0).astype(np.float32)
        _light = (0.72 - 0.58 * _dx - 0.44 * _dy) / np.sqrt(1.0 + _dx * _dx + _dy * _dy)
        shade = np.clip(0.62 + 0.56 * _light, 0.44, 1.28).astype(np.float32)
    _mark("tw_hillshade")
    # ---- v66: 陆地改用「全分辨率晕渲」作为主光源 ------------------------------
    # 改了什么: 新增 _shade2 —— 对 _surf 做 σ=1.6px 去噪后，用 ±4px 中心差分
    #   （=8px 间距差商，与 512 粗格的差分**同量纲**）求梯度，再套同一个 NW 光公式。
    #   陆地的 sh_mod 以 _shade2 为基底，不再以 512 粗格 shade 为基底。
    # 为什么: 1:1 复核（/tmp/mtn_a.png）山链是一条**颜色完全均匀的灰褐色带**，
    #   没有任何迎光/背光面 —— 看起来像涂上去的色条而不是山。根因是 shade 在
    #   512 格（4px）上求梯度、求梯度前还把 _surf 平滑到 σ=8px，等于**整幅陆地
    #   只剩 30px 以上的平缓明暗**，而 4px 的块状梯度被双线性放大后又糊成一片。
    #   改到原生分辨率后，山脊/岩壁/冲沟这些 8~30px 的地形才有各自的坡向。
    # 实测: 全分辨率 shade 在陆地 p5=0.44(clip底 11.0%) / p25=0.613 / p50=0.890 /
    #   p75=1.062 / p95=1.164 —— 与参考 Terrain Web 的 p25=0.652/p50=0.893/p75=1.061
    #   几乎重合（v65 之前我方 512 粗格口径为 p25=0.583/p50=0.814，整体偏暗且平）。
    # v67: 单尺度 → **多尺度**（macro 山脉走向 + mid 山肩 + meso 脊线岩壁）。
    # 改了什么: v66 的 σ=1.6 / ±4px（8px 间距）单尺度差分，拆成三档各走同一 NW
    #   光公式后加权：macro = σ12px + ±12px 差商（24px 间距）×8；mid = σ5px +
    #   ±6px 差商（12px 间距）×14；meso = σ2.8px + ±3px 差商（6px 间距）×22；
    #   权重 0.52 / 0.30 / 0.18（v69 定型；v67 是 0.55/0.45 双尺度、v68 试过 0.68/0.32，
    #   都因为宏观项过重把 20px 宽的窄山脊整条抹平而放弃）。
    # 为什么: v66 出图（/tmp/c_land.png、full900）陆地读作"揉皱的纸/浮雕" ——
    #   单尺度只照亮了 DEM 里 20~40px 的团块，没有"整条山脉受光、另一侧背光"的
    #   大形态，所有起伏挤在同一频段互相抵消。分频后 macro 给出受光面/背光面
    #   （用户第 2 条"平滑而真实的地表"），meso 只保留山脊的细锐边缘。
    def _hill(_src, _sp, _r, _kx, _ky):
        _s = gaussian_filter(_src, _sp)
        _gx = np.zeros_like(_s)
        _gy = np.zeros_like(_s)
        _gx[:, _r:-_r] = (_s[:, 2 * _r:] - _s[:, :-2 * _r]) * _kx
        _gy[_r:-_r, :] = (_s[2 * _r:, :] - _s[:-2 * _r, :]) * _ky
        return ((0.72 - 0.58 * _gx - 0.44 * _gy)
                / np.sqrt(1.0 + _gx * _gx + _gy * _gy)).astype(np.float32)
    # v69: **山地几何细节** —— 只进光照，不进色带、不改 DEM。
    # 改了什么: 在 _surf 上叠一层 ridged_fbm（base_freq=48 → 42px 主波长，4 octave
    #   降到 ≈5px），幅度 ±0.15，用「陆地 p62~p97 高程」线性门控；门控外为 0。
    #   并给 _light2 补一档中尺度（σ5 / ±6px / ×14），权重 0.52/0.30/0.18。
    # 为什么: 1:1 复核（/tmp/mtn_a.png、mtn_b.png）里山链是一条**内部完全平坦的
    #   棕色缎带 + 一圈暗边** —— 读取像贴纸轮廓，完全没有阿尔卑斯式的岩壁/刃脊。
    #   根因是 DEM 本身在造山带里是"台地"（plate_sim 的抬升被削平），σ12/±12px 的
    #   macro 梯度在台地内部恒为 0 → 山链内部无明暗。
    #   真实地图靠 **ridged 噪声的尖锐刃脊** 表达山地质感，这里照抄：ridged_fbm
    #   的 crest 是线状脊，比各向同性 fBm 更像山脊；且只在山地门控内生效，
    #   平原不会被重新刷成"揉皱的纸"（v66 的病）。
    # 实测: 首版门控取 p88~p99 + smoothstep，探针实测山地裁块内 mtg 均值只有
    #   0.049（几乎全被 4 次幂压成 0），shade 与无细节版逐位相同 → 等于没改。
    #   改成 p55~p95 线性门控后裁块 mtg 均值 0.53，细节真正进入光照。
    _hp = float(np.percentile(_surf[land], 55)) if land.any() else 1.0
    _hq = float(np.percentile(_surf[land], 95)) if land.any() else 2.0
    _mtg = np.clip((_surf - _hp) / max(_hq - _hp, 1e-3), 0.0, 1.0).astype(np.float32)
    # 频率 48 → 110（主波长 42px → 18.6px）：地图比例尺下 1px≈20km，一条阿尔卑斯级
    # 山系只有 10~25px 宽，42px 的脊在带内放不下两条，读作"一块起伏的台地"；
    # 18px 的脊才能在窄山链内部形成 2~3 条并列刃脊与冲沟。
    _rd = ridged_fbm(S, base_freq=110.0, octaves=3,
                     rng=np.random.RandomState(7717)).astype(np.float32)
    _rd = ((_rd - float(_rd.mean())) / (float(_rd.std()) + 1e-6)).astype(np.float32)
    # v72: 幅度 0.15 → 0.26，并补一档 8px 的细脊（±0.09）。
    # 为什么: v71 的 18.6px 脊在 1:1 裁图（/tmp/mtn_a.png）里只让山链内部多了
    #   几道淡灰纹，整条带子仍读作"贴纸" —— 0.15 的幅度除以 ridge 的标准差后，
    #   实际抬升只有 ~0.15，落到 _hill 的差商里（×8/×14）只够把 shade 推动 ±3%。
    #   0.26 是让 macro/mid 两档的梯度真正越过分级阈值（shade ≥ 1.05 / ≤ 0.80）
    #   的最小值；8px 细脊只补刃脊的锐度，幅度控制在 0.09，避免回到 v66 的
    #   "揉皱的纸"。
    _rf = ridged_fbm(S, base_freq=260.0, octaves=2,
                     rng=np.random.RandomState(9133)).astype(np.float32)
    _rf = ((_rf - float(_rf.mean())) / (float(_rf.std()) + 1e-6)).astype(np.float32)
    _surf_d = (_surf + (0.26 * _rd + 0.09 * _rf) * _mtg).astype(np.float32)
    _light2 = (0.52 * _hill(_surf_d, 12.0, 12, 8.0, 7.0)
               + 0.30 * _hill(_surf_d, 5.0, 6, 14.0, 13.0)
               + 0.18 * _hill(_surf_d, 2.8, 3, 22.0, 20.0))
    _shade2 = np.clip(0.62 + 0.56 * _light2, 0.44, 1.28).astype(np.float32)
    if land.any():
        _s2 = _shade2[land]
        print(f"      [v66] 全分辨率晕渲: p25={np.percentile(_s2, 25):.3f} "
              f"p50={np.percentile(_s2, 50):.3f} p75={np.percentile(_s2, 75):.3f} "
              f"clip底={float((_s2 <= 0.4401).mean()) * 100:.1f}%")
    # 陆地: hypsometric t —— v55b 改为「按参考色带面积占比做分位锚定」。
    # 改了什么: 旧版 t = elev_m / 陆地 p99.5(4915m)。实测出图整体偏淡、大陆内部出现
    #   一圈圈奶油色(beach)等高环: 本图 t 中位只有 0.072, **38.8% 陆地落在 beach 带
    #   (0~246m)**, 而参考实现 t 中位 0.201、beach 带只占 14.8% 陆地。
    # 为什么: 真实陆地高程严重左偏 (地球上一半陆地 <400m), 直接按 max 线性映射必然
    #   把大半陆地压进最低一档色带。故用参考 build_fields 实测的「各色带面积占比」
    #   当锚点: 把本图陆地高程的 [14.8,49.8,65.4,82.0,90.1,95.3,99.5] 分位分别映射到
    #   t=[0.05,0.20,0.34,0.54,0.68,0.82,1.0], 使两张图的色带面积分布一致 ——
    #   这样 beach/grass/forest/rock/snow 的占比与参考相同, 大陆才不会"一片奶油色"。
    # v58: 先做粗格平均高程 (_elev_up)，**分位锚点也改在它上面取**。
    #   改了什么: 分位锚点原本取自原始 elev_m，而查色带用的是平滑后的 _elev_up。
    #     平滑把高程分布向均值(354m)压，低于 p14.8(≈110m) 的像素大幅减少 ——
    #     实测 900px 口径下 beach 只占陆地 **1.8%**，而参考是 8.7%，海岸米色
    #     缎带几乎消失，海陆之间又是一条硬边。
    #   为什么: 锚点的作用是「让各色带的面积占比与参考一致」，所以必须和实际
    #     查表用的那套高程同分布；两者用不同的场，占比就必然漂移。
    _ecell = max(S // 128, 1)          # 色带粗格: 2048 → 16px/格
    _eG = S // _ecell
    if _ecell >= 2 and _eG >= 16:
        _elev_c = elev_m[:_eG * _ecell, :_eG * _ecell].reshape(
            _eG, _ecell, _eG, _ecell).mean(axis=(1, 3)).astype(np.float32)
        _elev_up = ndi_zoom(_elev_c, _ecell, order=1)[:S, :S]
    else:
        _elev_up = elev_m
    # v65: **高地上把逐像素高程混回色带输入**（低地完全不动）。
    # 改了什么: _elev_up += (elev_m - _elev_up) * 0.60 * clip((_elev_up-400)/1000)。
    # 为什么: 16px 块平均把山脊削平了 —— 一条 2~3 格宽的山脊平均后掉回 800m 档，
    #   于是色带把整条山链判成 forest，1:1 裁图里山脉是一条条**墨绿色蛇形纹**
    #   （用户第 2 条「大陆没有阿尔卑斯式山峰」，实测图全是深绿皱褶、没有岩壁）。
    #   混回 60% 的原始高程后，脊线重新落进 rock/snow 段（占比仍由分位锚定，
    #   所以 rock 总面积不变，变化的是**形状**：从一条连续带变成沿脊线串联的峰簇）。
    # 为什么加 400m/1000m 的门控: 低地（<400m）保留 100% 粗格场，海岸米色缎带与
    #   平原的连续色带不受影响 —— 不然又回到 v58 修掉的"绿底奶油霉点"。
    if land.any():
        # v68: 混回系数 0.60 → 0.40。
        # 为什么: 逐像素高程混回 60% 后，2048 层的高频起伏重新进入色带查表，
        #   高地被切成一圈圈 10~20px 的闭合等高环（1:1 裁图里 straddle 在岩壁
        #   上的暗橄榄色细线）。0.40 仍足以让脊线落进 rock/snow 段（形状目标），
        #   但色带的团块尺度回到 25px 以上，细环被压掉。
        _hi_w = np.clip((_elev_up - 400.0) / 1000.0, 0.0, 1.0).astype(np.float32)
        _elev_up = (_elev_up + (elev_m - _elev_up) * (0.40 * _hi_w)).astype(np.float32)
    if land.any():
        _q = np.percentile(_elev_up[land],
                           [14.8, 49.8, 65.4, 82.0, 90.1, 95.3, 99.5])
        _q = np.maximum.accumulate(_q.astype(np.float64))   # 防重复分位破坏单调性
    else:
        _q = np.array([200.0, 400.0, 800.0, 1500.0, 2500.0, 3500.0, 5000.0])
    # v56: 锚点串头部补一个「海拔 0 m → t=0」的哨兵。
    # 改了什么: 旧锚点把「陆地 p14.8 高程(≈110m)」直接映射到 t=0.05，于是**所有
    #   低于 110m 的陆地都落在 t=0.05 这条 grass 分界线上**，t<0.05 的 beach 段
    #   一个像素都没有 —— 出图上陆地与海洋之间没有任何米色过渡，海岸是一条硬边。
    # 为什么: reference 的 t=(h-sea)/relief 天然满足「h→sea 时 t→0」，所以它的
    #   beach 是沿**整条海岸线**连续的一条米色缎带(约占陆 14.8%)。补 (0m, t=0) 后
    #   我方 0~110m 全部落进 0~0.05 的 beach 渐变，海岸缎带与面积占比同时对齐。
    _q = np.concatenate([[0.0], _q])
    # v58: 高程带改在「粗格平均高程」上查, 再双线性上采样。
    # 改了什么: 旧版逐像素用 2048 原始高程查色带。2048 层的高程高频起伏幅度
    #   (几十~上百米) 远大于最低色带的宽度 (beach 段只覆盖 0~p14.8≈110m), 于是
    #   同一条色带被切碎成满图 10~20px 的斑块 —— 1:1 裁图复核读作"绿底上撒
    #   奶油色霉点", 大陆内部一片斑驳。
    # 为什么: 参考实现的 t=(h-sea)/relief 是在它的 gw×gh=310 粗格 surface 上求的
    #   (terrain_web.py L1393-1396, 逐像素只是对粗格做双线性), 所以它的色带天然
    #   是**连续的大色块**。这里按同一口径取 256 粗格 (_G×_G, 每格 8px, 与参考
    #   2.9px@900 的等效尺度相当) 做块平均, 再插值回 2048 —— 色带重新连成整片,
    #   海岸段因为块平均把邻近海面负高程混进来, 自然形成沿岸米色缎带。
    t_land = np.interp(_elev_up, _q, [0.0, 0.05, 0.20, 0.34, 0.54, 0.68, 0.82, 1.0])
    t_land = np.where(land, t_land, 0.0).astype(np.float32)
    _elev_lut, _ocean_lut = _terrain_web_luts()
    # detail 起伏: 参考实现用接近逐像素的 detail 贴图（≈2px）在 t 与光照上各加一层
    #   高频扰动, 幅度随 mountain(uplift) 增大 —— 这是"山体不是平面色块"的关键。
    # 改了什么: 旧版只有 _lowres_fbm(S,40)（≈51px 平滑块）, 出图陆地是一整片平涂色、
    #   没有岩石颗粒感。现改为 60px 主体 + 300px 高频两尺度叠加（2048 下高频波长
    #   ≈7px），整体仍按陆地 p97 归一化到 [-1,1]。
    _ridge_raw = (0.60 * _lowres_fbm(S, 60.0, 3, rng=rng)
                  + 0.40 * _lowres_fbm(S, 300.0, 2, rng=rng))
    _rn = float(np.percentile(np.abs(_ridge_raw[land]), 97)) if land.any() else 1.0
    ridge = np.clip(_ridge_raw / max(_rn, 1e-6), -1.0, 1.0).astype(np.float32)
    # mountain: 参考实现用构造抬升 uplift（实测陆地中位≈1.0 —— 即低地也是满幅细节）；
    #   我方旧版从 600m 起算, 但本图陆地高程中位只有 354m → 大半陆地被判成 0,
    #   低地因此完全平涂。现改为「小基线 + 按分位锚定的 t 抬升」,
    #   低地保留少量 detail 颗粒, 山地满幅（基线值见下方 v58 说明）。
    # v58: 基线 0.45 → 0.12。参考实现的 detail 贴图是 ≈2px 的细颗粒, 满幅铺开也
    #   只读作"纸面纹理"; 我方的 ridge 是 2048 下 ≈7px 的 fBm 团块, 低地给 0.45
    #   的幅度时会在平原上刷出一片片 15~30px 的亮/暗斑 (1:1 裁图里是"绿地上
    #   长霉斑"), 与色带碎斑叠加后整块大陆读作噪点。降到 0.12 后平原基本回到
    #   连续色带, 山地仍按 t 升到 1.0 保留完整起伏质感。
    mountain = np.clip(0.12 + 0.88 * np.clip((t_land - 0.08) / 0.42, 0.0, 1.0),
                       0.0, 1.0).astype(np.float32)
    # ---- v59: 色层 Voronoi 胞体（只改颜色，不改高程/光照）----------------------
    # 改了什么: 两套抖格 Voronoi 胞体场，分别喂给「沙滩宽度」与「植被斑块」。
    # 为什么: 用户提议"用 Voronoi 胞体做底"。几何底不采纳（胞界突变会把 shade 的
    #   相邻差从 p90 0.127 推回 0.3 量级）；但颜色层正缺这种有机斑块 —— 现在的
    #   沙滩是沿整条海岸等宽的一条、森林是 σ=8 高斯场糊出的一整片，两者读起来都
    #   比参考图"太规整"。胞体把这两层的边界打散，而地形明暗完全不受影响。
    # 尺度: 沙滩 165px（沙袋长而连续）、植被 68px（≈33 胞/边，斑块粒度接近参考）。
    _vc_sand = _tw_voronoi_cells(S, _VC_SAND_CELL, 2207)
    _vc_veg = _tw_voronoi_cells(S, _VC_VEG_CELL, 1103)
    # 沙滩抖动: 只在 t<0.10 的近岸带生效（权重 (1-t/0.10)²），幅度 ±0.010 ——
    #   等于把 beach→grass 的等值线按胞体推前/拉后，海岸沙带因此有宽有窄。
    #   内陆段 t>0.10 权重为 0，草地带是纯色，不会出现"绿色拼块"。
    _tb = np.clip(t_land / 0.10, 0.0, 1.0)
    # v62: 补一层 ≈2px 的细颗粒（参考实现的 detail 贴图就是 2px 级）。
    # 改了什么: 旧版最低尺度是 ridge 的 ≈7px 团块，1:1 裁图里低地是"一整片平涂色"。
    # 为什么: 用户「平滑而真实的地表」。2px 级颗粒只读作纸面质感，不会形成霉斑。
    _fine = np.clip(_lowres_fbm(S, 700.0, 2, rng=rng), -1.5, 1.5)
    _fn = float(np.percentile(np.abs(_fine[land]), 95)) if land.any() else 1.0
    _fine = np.clip(_fine / max(abs(_fn), 1e-6), -1.0, 1.0).astype(np.float32)
    # v66: 删除 v65 的「全分辨率 shade 高通 _det」。
    # 改了什么: 整段移除（含 [v65] 细节晕渲 的打印）。
    # 为什么: 它是把真实坡向当"锐化滤镜"叠在粗格 shade 上，实测 |_det| 陆地 std=0.495
    #   （归一化后大量像素撞到 ±1 的钳位），即高频被放大到二值饱和 —— 1:1 复核仍读作
    #   "迷彩"，只是对比更高。现在 _shade2 本身就是原分辨率的地形明暗，不需要再高通。
    t_mod = np.clip(
        t_land
        + ridge * (0.020 + 0.052 * mountain)
        + _fine * (0.006 + 0.026 * mountain)
        + (_vc_sand - 0.5) * _VC_SAND_AMP * (1.0 - _tb) ** 2,
        0.0, 1.0)
    # ---- v66: 沙滩只在海岸带出现 / 雪带破碎 -----------------------------------
    # ① 沙滩门控。改了什么: 离岸 >6px 的陆地把 t 抬到 ≥0.075（纯 grass 段）。
    # 为什么: 1:1 裁图（/tmp/c_land.png、mtn_a.png）里大陆腹地散着**成片的奶油色
    #   沙斑**——t 由高程分位给出，内陆的低洼盆地（<60m）同样落进 beach 段，
    #   于是"沙滩"出现在离海几百公里的地方。参考图里沙色只沿海岸线呈一条缎带。
    # 实测: 离岸 ≤6px 的陆地占 16.8%、≤10px 占 25.4%，抬升后内陆沙斑归零，
    #   海岸缎带保留（6→14px 之间线性过渡，避免出现一条新的硬边）。
    _dcoast = distance_transform_edt(land).astype(np.float32)
    _sand_far = np.clip((_dcoast - 6.0) / 8.0, 0.0, 1.0)
    t_mod = np.maximum(t_mod, 0.075 * _sand_far).astype(np.float32)
    # ② 雪带破碎。改了什么: t>0.80 的高程带叠 ±0.055 的 σ≈21px fBm 抖动。
    # 为什么: 雪带 LUT 段（0.88→1.00）是纯高程函数，同一条山脊的高程连续 → 1:1
    #   裁图里是一根**贯穿全山的白色胶带**（用户截图里那两条白蛇）。真实雪线受
    #   坡向/风蚀/岩壁影响是斑块状。抖动只加在 t>0.80 段，低海拔色带不受影响。
    _snow_n = np.clip(_lowres_fbm(S, 48.0, 3, rng=rng), -1.5, 1.5)
    _snow_w = np.clip((t_mod - 0.80) / 0.20, 0.0, 1.0)
    t_mod = np.clip(t_mod + _snow_n * 0.055 * _snow_w, 0.0, 1.0).astype(np.float32)
    # v72: **山地色带碎裂** —— 用同一套 mountain 门控（_mtg）把 t 抖 ±0.045。
    # 改了什么: 只在 _mtg>0 的造山带内，按 σ≈16px 的 fBm 扰动色带索引。
    # 为什么: 1:1 复核（mtn_a/mtn_b）里山链是一块**边缘锐利的棕色色块** —— 岩带
    #   （rock→snow）的边界完全由 16px 粗格高程等值线决定，是一条平滑闭合曲线，
    #   所以无论内部晕渲多强，外轮廓都读作"贴纸"。真实山系的岩/雪/草边界被刃脊
    #   切碎成斑块（雪在线状刃脊上、岩在冲沟里）。±0.045 对应约 ±220m，只够
    #   打碎边界、不足以让平原重新出现色斑（门控外严格为 0）。
    _rock_n = np.clip(_lowres_fbm(S, 62.0, 3, rng=rng), -1.5, 1.5)
    t_mod = np.clip(t_mod + _rock_n * 0.060 * _mtg, 0.0, 1.0).astype(np.float32)
    # v73: **坡向岩石出露** —— 真实坡度的陡缓直接抬升色带索引（陡 = 裸岩）。
    # 改了什么: 对 σ=1.6px 平滑后的 _surf 求原生梯度，取陆地 p95 归一化到 0~1，
    #   在与山地门控相乘后按最多 +0.10 抬 t。
    # 为什么: 探针实测本图陆地高程分布极端左偏 —— p90.1=1877m 才进 rock 段，
    #   也就是**只有最高的 10% 陆地是岩色**，而 `elev>1500m` 只占陆地 7.4%
    #   （最大山块 bbox 896×191，是一条蛇形窄链）。于是 1:1 裁图里整条山脉被
    #   判成 forest 深绿，只有脊线最顶端 2~3px 是 rock/rock_hi —— 读作"绿地上
    #   画了几条白石线"，而不是阿尔卑斯式的岩壁（用户第 2 条）。
    #   真实地形图里**陡坡出露岩石**与高程无关：600m 的峭壁也是灰褐色。按坡度
    #   抬 t 后，山链两侧的陡坡变成岩色、山间谷地与缓坡仍是森林，山体第一次
    #   有了"岩壁 + 植被"的分层，而不只是顶端一撮白。
    # 幅度: 0.10 对应约 +500m 的高程档，正好把坡面推进 rock 段而谷底不动。
    _gy_sl, _gx_sl = np.gradient(gaussian_filter(_surf, 1.6))
    _slope = np.hypot(_gx_sl, _gy_sl).astype(np.float32)
    _slref = float(np.percentile(_slope[land], 95)) if land.any() else 1.0
    _slope_n = np.clip(_slope / max(_slref, 1e-6), 0.0, 1.0).astype(np.float32)
    t_mod = np.clip(t_mod + 0.10 * _slope_n
                    * np.clip(_mtg * 1.6, 0.0, 1.0), 0.0, 1.0).astype(np.float32)
    # v66: sh_mod 基底换成全分辨率 _shade2；合成 ridge 项大幅压低（见上）。
    # 改了什么: shade → _shade2；ridge 项 0.050/0.070 → 0.012/0.020；
    #   _rdx/_rdy 项 0.16/0.55 → 0.06/0.22。
    # 为什么: ridge 是合成 fBm，与真实坡向无关 —— 1:1 里那些"水彩晕开的明暗团"
    #   主要来自它。真实山体明暗交给 _shade2，ridge 只保留极弱的色层起伏。
    # v67: 删除「合成 fBm 的伪法线」项。
    # 改了什么: -( _rdx*0.58 + _rdy*0.38 )*(0.06+0.22*mountain) 整项移除；
    #   ridge 色层项 0.012/0.020 → 0.008/0.014；_fine 色层项 0.020/0.040 → 0.006/0.010。
    # 为什么: 那一项对**与真实地形无关**的噪声场求梯度，等于在陆地上盖一层
    #   "浮雕/揉皱纸"的闭合等高圈 —— 1:1 裁图里最扎眼的那圈圈绿线就是它。
    #   真实明暗现在完全由 _shade2（macro+meso 真坡向）承担，噪声只允许极弱地
    #   调制色相，不再伪造光照。
    sh_mod = (_shade2 + ridge * (0.008 + 0.014 * mountain)
              + _fine * (0.006 + 0.010 * mountain))
    # ---- 恢复旧方案: Whittaker 9 群系离散色 + 约 3px 群系边界软混合 ----
    # 用户要求回到 v55 之前的配色: 陆地不走连续高程色带, 而由年均温 / 年降水查
    # Whittaker 9 类, 每类一个固定色; 群系边界先中值/mode 滤波去掉硬阈值锯齿,
    # 再用 σ≈1.2 的高斯权重混合(过渡约 3~4px), 避免出现生硬的色块切边。
    # 高程只通过下方 _shade2 明暗体现, 不再参与色相。
    _bi = whittaker_biome(T, P, lats, ci)
    _bi = smooth_biome_labels(_bi, land, median_size=7, min_patch_px=50, iters=3)
    _bw = np.zeros((S, S), np.float32)
    _brgb = np.zeros((S, S, 3), np.float32)
    for _k in range(BIOME_COLORS.shape[0]):
        _mk = gaussian_filter((_bi == _k).astype(np.float32), 1.2)
        _bw += _mk
        _brgb += _mk[..., None] * BIOME_COLORS[_k]
    land_rgb = (_brgb / np.maximum(_bw, 1e-6)[..., None]).astype(np.float32)
    # 高程明度: 群系色只提供色相, 高原/平原的层次靠一层很轻的**绝对高程亮度**
    # 拉开 (0m −2% → 1500m +4% → 3000m +8%), 不改变任何色相; 之后仍乘 shade,
    # 所以明暗关系保持单调, 不会把背光坡抬亮。
    _u_elev = np.clip(_elev_up / 3000.0, 0.0, 1.0).astype(np.float32)
    _elev_lit = (1.0 - 0.020 * (1.0 - _u_elev) + 0.080 * _u_elev).astype(np.float32)
    land_rgb = (land_rgb * _elev_lit[..., None]).astype(np.float32)
    # ---- A1: 绝对高程 + 纬度雪线（取代"陆地高程分位雪带"）----------------------
    # 旧口径: LUT 在 t>=0.90 起混雪，t 是陆地高程**分位**（≈前 2.5%）→ 雪的分布
    #   与纬度、绝对高程无关：赤道 2000m 山头与极地 300m 丘陵都上雪，山肩裸岩，
    #   雪帽呈散点、不成线（kernel 观测：旧雪海拔中位在各纬度带全是 3900~4900m，
    #   与 snowline_m 的 600~3700m 完全没有相关性）。
    # 新口径: 以 snowline_m(lat) 为雪线（副热带 ≈3.7km、赤道 ≈3.3km、50° ≈1.3km、
    #   极地海平面），按**绝对高程**过线即上雪；叠 90px 波长 fBm 让雪线本身起伏
    #   ±220m（山脊走向的自然雪线不是等海拔线），再按陡坡折减 35%（岩壁挂不住雪）。
    #   只在渲染阶段合成，不参与任何 DEM 计算；独立 RNG 保证不影响既有噪声序列。
    _zs_l = (snowline_m(np.abs(lats))[:, None]
             + 220.0 * fbm1d(S, 90, np.random.RandomState(4417))[:, None]
             ).astype(np.float32)
    _snow_abs = _tw_smoothstep(_zs_l - 250.0, _zs_l + 350.0, _elev_up)
    _snow_abs = (_snow_abs * (1.0 - 0.35 * _slope_n)).astype(np.float32)
    land_rgb = _tw_mix(land_rgb, _TW_COLORS["snow"], _snow_abs).astype(np.float32)
    print(f"      [A1] 雪线: 雪 {float((_snow_abs[land] > 0.5).mean()) * 100:.2f}% 陆地 / "
          f"雪线(陆地中位) {float(np.median(_zs_l)):.0f}m")
    _bc = np.bincount(_bi[land].ravel(), minlength=BIOME_COLORS.shape[0])
    _bc = _bc / max(float(_bc.sum()), 1.0)
    print("      [群系] " + " / ".join(f"{i}:{_bc[i] * 100:.1f}%"
                                     for i in range(BIOME_COLORS.shape[0])) +
          f" | 群系类数 {int((_bc > 0.002).sum())}")
    # v66: 亮度映射重标定 0.74+(sh-0.78)*1.85 clip[0.60,1.28]
    #   → 0.24+0.76*_shade2 clip[0.54,1.26]。
    # 为什么: 旧式的斜率 1.85 在 shade 的 p25(0.61) 处已经撞到 0.60 下限 —— 实测
    #   25% 的陆地被压成同一档暗绿，背光坡完全没有层次（1:1 里山的两侧同色）。
    #   新式在 [0.44,1.28] 全区间单调：p25→0.71 / p50→0.92 / p75→1.05 / p95→1.12，
    #   阴影坡仍能读出 0.71~0.85 的层次，迎光坡到 1.1 左右。
    # v75: 0.24+0.76·s clip[0.60,1.26] → 0.14+0.86·s clip[0.54,1.28]。
    # 改了什么: 亮度映射的对比斜率从 0.76 提到 0.86，暗端下探到 0.54。
    # 为什么: 用户第 2 条「平滑而真实的地表 / 壮观的阿尔卑斯式山峰」。实测分级亮度
    #   （world_map.png × DEM 口径）0~341m 平原 0.548、823~1860m 山地 0.413 ——
    #   山地的平均值只比平原**暗**，却没有"受光面亮 / 背光面暗"的对比（back-lit 与
    #   sun-lit 被平均掉）。根因是 0.60 的下限把背光坡钳在同一档，山体只剩剪影。
    #   新式让背光坡到 0.54（≈再暗 4 个色阶）、迎光到 1.28，2px 尺度上的岩壁/刃脊
    #   才有立体感；平原（shade≈0.9）落在 0.91，几乎不变，不会整片变暗。
    _factor = np.clip(0.14 + 0.86 * sh_mod, 0.54, 1.28).astype(np.float32)
    land_rgb = land_rgb * _factor[..., None]
    land_rgb[..., 0] = np.minimum(1.0, land_rgb[..., 0] * 1.02)   # 陆地暖化 (r*1.02)
    land_rgb[..., 2] = np.maximum(0.0, land_rgb[..., 2] * 0.98)   # 陆地暖化 (b*0.98)
    _mark("tw_land")
    # ---- 海洋: Terrain Web 三段式「到岸距离」LUT (浅青→中蓝→藏青) ----
    # 改了什么: 索引从"水深占比"改为"到岸距离(粗格)": d_cells = d_px / 6.6,
    #   pd = clip(d_cells/12, 0, 1) 送进 _ocean_lut（该 LUT 内已按 1.7/12 与 12 格
    #   做分段 + 岸缘 foam）。海洋光照沿用参考的 factor=0.90+shade*0.08。
    # 为什么: 见 _terrain_web_luts() 里的说明 —— 我方宽陆架按水深上色会把海面
    #   大面积染成浅青, 与参考"深藏青大洋 + 细窄亮岸带"的观感差很远。
    # v61: 用户反馈「海洋之类的地形没有了，未来 3D 地形需要这个」。
    #   改了什么: 索引从纯「到岸距离」改为「距离 + 水深」双因子
    #       pd = clip(0.25·clip(d/34.8) + 0.80·clip((-elev-150)/4800), 0, 1)
    #     并在最深处再混一层 abyss(3,17,36)。海沟(< -5200m)走独立的深度门控，
    #     与 pd 解耦 —— 否则整片深海盆地(中位 -4234m)会被一起压黑。
    #   为什么: 旧版只看离岸距离，同一深度带内洋中脊(≈-2500m)与深海平原(≈-5200m)
    #     完全同色，DEM 里的海底起伏在配色阶段就被抹平。加深度项后，脊轴/海山自动
    #     比两侧盆地浅一档，海沟再压成近黑，海底地形在无晕渲时也能读出走向。
    #   实测依据: world_dem 海洋深度 p1=-5587 / p10=-5134 / p50=-4234 / p90≈-2800，
    #     <-6000m 7351px（海沟）；因此 abyss 的 smoothstep 取 5200~7600，只染海沟。
    sea = ~land
    if sea.any():
        _dsea0 = distance_transform_edt(sea).astype(np.float32)
    else:
        _dsea0 = np.zeros_like(elev_m, dtype=np.float32)
    _depth_m = (-elev_m).astype(np.float32)
    # v61b: ① 取色前先做 2px 归一化卷积平滑水深。旧版逐像素用 raw DEM 取色，GAN
    #   的深海高频噪声直接把四段色阶打成"电视雪花"（1:1 裁图里海面的细密斑点）。
    #   海底地貌的尺度是几十~几百像素，2px 平滑不损失结构，只去掉噪声。
    # ② 索引改为 0.88·水深 + 0.12·离岸距离：水深主导分色，离岸距离只负责把
    #   宽陆架压回浅色，避免"深海色一路顶到岸线"。
    # v75: 平滑 σ 2.0 → 3.0。改了什么: 取色用的水深卷积核再放大一档。
    # 为什么: 2px 只压掉 GAN 的像素级噪声，保留下来的正是 10~30px 的圆丘（海山/台地）
    #   边缘 —— 900px 缩略图上这些边缘恰好是"霉斑"的轮廓线。σ=3 把 10~30px 的圆顶
    #   抹成 40~60px 的缓坡，配合 LUT 过渡拉长（见 _terrain_web_luts），洋面从
    #   "一堆气泡"变成"几片深浅不一的深海盆地"。陆架坡折宽 5~8px，σ=3 不会把它抹平。
    # v80: 3.0 → 4.5。为什么: v79 把海底起伏放大 3 倍后，GAN 在深海平原上留下的
    #   1~3px 颗粒也被一并放大，1:1 裁图里整片洋面读作"蜂窝/波纹"。σ4.5 抹掉像素级
    #   颗粒，只留下 ~20px 以上的真实海底地貌（陆架坡折宽 5~8px，仍不会被抹平）。
    # v82: 取色水深改为 median9 + σ3.0，并且**不让颗粒进入 pd / 色调层**。
    # 为什么: v81 的 σ2.6 只是把颗粒削薄。GAN 在深海平原留下的是 8~14px 的准周期
    #   颗粒（单一波长），σ2.6 对它几乎无效；它随后被 20px 带通的 _obath（同时进
    #   pd 与色调层）和 24px 带通的晕渲放大，1:1 下重新读作"脑珊瑚/皱布"式的
    #   等间距波纹（用户「波纹又出现，而且更严重」）。median9 能整体铲掉 <9px 的
    #   团块，σ3.0 再把残留边缘抹圆；20px 以上的海山裙坡/脊轴原样保留。
    _dn = gaussian_filter(np.where(sea, _depth_m, 0.0), 2.6)
    _dd = gaussian_filter(sea.astype(np.float32), 2.6)
    _dsm = np.where(sea & (_dd > 0.2), _dn / np.maximum(_dd, 1e-3), 0.0).astype(np.float32)
    # v82: pd / 色调层专用水深（已去颗粒）。_dsm 仍供近岸距离与浅水判定使用。
    _dsm_c = gaussian_filter(median_filter(_dsm, 9), 3.0).astype(np.float32)
    # v65: 水深 → pd 改为「海底四带节点插值」（见 _OC_DEPTH_KNOTS 的说明）。
    # 改了什么: v64 的 clip((_dsm-30)/4200)**0.72 → np.interp(_dsm, 四带节点)。
    # 为什么: 本图深海平原就在 -4200m 附近，4200 的分母让它正好钳到 1.0，
    #   整片洋面（实测 25% 像素）落在 LUT 最暗两档，海底地貌全部丢失。
    # 现在: 深海平原(-4228m) → pd≈0.48 → 深藏青 (6,33,63)（与参考主色 6,33,62 同档），
    #   洋中脊(-3200m) pd 0.36 亮一档、盆地底(-5134m) pd 0.63、海沟(-5600m) pd 0.71
    #   暗一档 —— 三类地貌在无晕渲时也能读出。
    _pd_dep = np.interp(_dsm_c, _OC_DEPTH_KNOTS, _OC_PD_KNOTS).astype(np.float32)
    _pd_dist = np.clip(_dsea0 / (2.9 * 14.0), 0.0, 1.0)
    # v65: 抖动从单尺度 ±0.045 改为「40px + 11px」双尺度 ±0.055。
    # 为什么: 单尺度抖动只是把等宽岸带揉皱，宽度仍一致；叠两级后近岸浅水带的
    #   宽窄本身随位置变化（沙洲/浅滩），等宽"描边"读法消失。
    # v73: 抖动 0.055 → 0.028，并新增**真实海底起伏项** 0.070·_obath。
    # 改了什么: 原来 pd 里唯一的非深度信息是两级 fBm 抖动（26/90 频率，±5.5%）。
    #   现在砍掉一半，把腾出的权重交给从 DEM 实测出来的局部起伏。
    # 为什么: 探针（/tmp/probe4.py, world_dem.npy 口径）实测开阔洋（离岸>150px）
    #   深度 p5/p50/p95 = 3398/4326/5357 m，但 24px 高通的局部起伏 std 有 354 m
    #   （±1000m 的脊轴/海山确实存在）。旧式子把 ±354m 的真实起伏映射成 pd 的
    #   0.035 色阶差，而 0.055 的纯噪声抖动却占了 ±0.055 —— **噪声信号是真实
    #   地形的 1.6 倍**，所以 900px 上洋面读作"烟丝状云"，不跟着脊轴走（用户第 3 条）。
    #   换成真实起伏后，同一处洋中脊连续几百像素都偏亮，盆地连续偏暗，地貌可读。
    # v74: 抖动 0.028 → 0.016，并把权重从 0.65(low)/0.75(high) 改为 0.30/0.85。
    # 为什么: v73 出图（/tmp/full900.png）洋面上是**几十像素的淡蓝团块** ——
    #   value_fbm(S, 26) 在 2048 下的波长是 2048/26 ≈ 79px，0.65 的权重让它
    #   完全支配了抖动项，于是这些 79px 的噪声团被染成一档更浅的蓝，缩略图上
    #   读作"水里长霉斑"。真实海底起伏已由 0.070·_obath 承担（跟着脊轴走），
    #   抖动只该负责打散等值线，不该自己造地形。
    # v75: 抖动 0.016 → 0.010。
    # 为什么: v74 已把 79px 低频权重压到 0.30，但 0.016 的总幅在 pd 上仍是 ±0.016 ——
    #   在深海（pd 0.45~0.55）等于把 LUT 的相邻两档反复来回，缩略图上表现为一层
    #   细碎的"蓝噪"。海底起伏由 0.045·_obath 承担，抖动只该负责打散等值线，0.010 够。
    _pd_jit = (0.010 * np.clip(0.30 * value_fbm(S, 26.0, 2, rng=rng)
                               + 0.85 * value_fbm(S, 90.0, 3, rng=rng),
                               -1.6, 1.6)).astype(np.float32)
    # v78: 带通尺度 18 → 26px。为什么: σ18 把 GAN 在深海平原上留下的 10~30px
    #   起伏一起放大，1:1 裁图里整片洋面读作"揉皱的纸"；26px 只保留海山/脊轴的
    #   裙坡（50~150px 结构），平原保持安静。
    # v81: 带通尺度 26 → 20px（配合取色水深 σ2.6，26px 会把海山裙坡一起当趋势减掉）。
    # v82: 20 → 45px，并且作用在去颗粒的 _dsm_c 上。
    # 为什么: 20px 带通正好穿过 GAN 颗粒的波长带，是"皱布纹理"的直接来源；
    #   45px 只保留海山裙坡/脊侧这类几十像素的真实坡面，颗粒被 median9 先行铲掉。
    _obath = (_dsm_c - gaussian_filter(_dsm_c, 45.0)).astype(np.float32)   # 局部海底起伏(m)
    _obref = (float(np.percentile(np.abs(_obath[sea]), 92))
              if sea.any() else 1.0)
    _obath = np.clip(_obath / max(_obref, 1e-3), -1.4, 1.4).astype(np.float32)
    # v75: 权重 0.070 → 0.045。
    # 为什么: 0.070 让 18px 高通的局部起伏独占 ±0.098 的 pd —— 在 900px 上等于
    #   给每个海山/台地描一圈可见的明暗环（"气泡"的第二个来源）。收到 0.045 后，
    #   真实海底起伏仍然跟着脊轴/盆地走，但单块台地的色差降到半档以内。
    # v81: 0.045 → 0.065。v80 的起伏被 σ4.5 抹掉后，pd 里几乎没有地形信息；恢复细节后
    #   需要把权重提回一档，才能让脊轴/盆地在大尺度上分出深浅。
    # v82: 0.065 → 0.050。_obath 已从 20px 带通退到 45px，归一化后幅度（±1.4）不变，
    #   稍降权重即可，避免大尺度盆地被染成一块块明显的色斑。
    _pd = np.clip(0.88 * _pd_dep + 0.12 * _pd_dist + _pd_jit
                  + 0.050 * _obath, 0.0, 1.0)
    rgb = _ocean_lut[np.minimum(255, (_pd * 255.0).astype(np.int32))]
    # v61b: 深度对比增益。四段色阶本身在 -3000~-5000m 之间的绝对 RGB 差只有
    #   几个色阶，深海在缩略图上仍读作"一片藏青"。再乘一个随 pd 线性下降的亮度
    #   系数 (1.22 → 0.86)：洋中脊/海山（浅）比深海平原亮约 25%，盆地比海沟亮
    #   约 12%，海底起伏因此在缩略尺度也能读出走向。陆地不受影响（后面被覆盖）。
    # v64: 1.22/0.42 → 1.10/0.30。1.22 的岸外增益本身就把 shallow 提亮到
    #   (27,105,139)，比沙滩还跳；收到 1.10 后岸带是 (24,95,125)，仍比深海亮
    #   一档（可读陆架），但不再从陆地边缘"发光"。
    # v67: 1.10/0.30 → 1.16/0.40。深度对比再拉大一档 —— 洋中脊(浅)与深海盆地
    #   的亮度差从约 15% 提到约 20%，配合上方 ±22% 的晕渲，缩略尺度也能读出
    #   脊轴/盆地/海沟三类地貌（用户第 3 条）。
    # v68: 1.16/0.40 线性式 → **只压深海端、浅海端回到 1.0**。
    # 改了什么: gain(pd) = 1.04 - 0.32*clip((pd-0.10)/0.85)；旧式在 pd=0 处给 1.16。
    # 为什么: v67 出图（/tmp/full900.png）每个大陆外围都套一圈**亮青色霓虹描边** ——
    #   浅海 LUT (22,86,114) 乘 1.16 变成 (25,100,132)，比旁边的沙滩还跳，等宽地
    #   勾出海岸线。深度对比本身要的是"浅亮深暗"的相对关系，不是把浅海**整体提亮**
    #   到发光；改成浅海 gain=1.04（几乎原色）、深海端 0.72 后，对比度保留
    #   (1.04/0.72 = 1.44，旧式 1.16/0.76 = 1.53)，但海岸不再自发光。
    # v75: 1.04/0.32 → 0.96/0.26。改了什么: 整条增益曲线整体下调 0.08。
    # 为什么: 900px 全图上陆地均值亮度 0.503 / 海洋 0.247 —— 海面偏亮导致大陆边缘的
    #   浅海带与沙滩连成一条"亮环"，大陆反而被背景抢走对比。0.96 的浅海端让
    #   shallow(22,86,114) 落到 (21,83,109)，与沙滩的明度差回到"水比沙暗"的常见读法；
    #   深海端 0.70 与陆地的暗部 0.58 拉开，海陆分离度反而更高。
    # v77: 0.96/0.26 → **1.10/0.42**，并把增益曲线的斜率拐点从 pd=0.10 提到 0.06。
    # 改了什么: 浅端回到 1.10、深端压到 0.68，整条曲线跨度 1.62（旧 1.49）。
    # 为什么: 1:1 实测（本轮 world_map × DEM）开阔洋（离岸>25px、|lat|<50°）亮度
    #   分布 p1=25.0 / p50=33.3 / p90=39.0 —— **整片大洋只占 14 个亮度级**，而水深
    #   跨度 4800m。分档色阶之间的差（deep 62 / abyss 49）被 0.83 的增益和后段
    #   `0.90+shade*0.08` 再压一次，脊轴与盆地落成同一片藏青，海底起伏读不出来。
    #   浅端不回到 1.16 那一档是为了不重演 v67 的"霓虹岸带"：1.10 只作用在
    #   pd>0.06（≈400m 以深），岸带浅水段仍是原色。
    _ogain = (1.12 - 0.26 * np.clip((_pd - 0.06) / 0.86, 0.0, 1.0)).astype(np.float32)
    rgb = (rgb * _ogain[..., None]).astype(np.float32)
    # v78: 海底"高程色调"层（elevation tint）。改了什么: 在分层设色之上再乘一层
    #   (1 - 0.22·_obath)，_obath 是 σ18 带通后的**真实高程异常**（正 = 局部更深）。
    # 为什么: pd 的四个节点间距 1000~1200m，600m 的局部起伏只换来 |Δpd|≈0.05；
    #   LUT 在深海段的斜率是 ~32 亮度/pd，于是 ±600m 的脊轴/海山只有 ±1.6 级 —— 8bit
    #   上等于没有。直接对亮度乘以高程异常，斜率提到 ~20 亮度/1000m（模拟实测），
    #   同时不改变色相（三通道同乘），读法仍是"水"，只是盆暗脊亮。
    #   门控沿用晕渲的近岸/浅水淡出，避免陆架坡折被染成亮边。
    _tint_gate = (np.clip((-elev_m - 250.0) / 400.0, 0.0, 1.0)
                  * np.clip((_dsea0 - 12.0) / 24.0, 0.0, 1.0)).astype(np.float32)
    # v81: 0.20 → 0.25。起伏幅度随细节恢复同步提一档，盆暗脊亮的读法更明确。
    # v82: 0.25 → 0.18。_obath 现在是 45px 带通，归一化后 ±1.4 的幅度打在整块盆地
    #   上会形成明显的明暗斑；降到 0.18 后仍能读出盆暗脊亮，但不抢色阶。
    rgb = (rgb * (1.0 - 0.18 * _obath * _tint_gate)[..., None]).astype(np.float32)
    # v81: 轻度去饱和（10% 向自身亮度收）。为什么: 高饱和的青蓝 + 平滑的大面积渐变
    #   是"塑料感"的另一半来源；真实海底地形图（GEBCO）的远洋是偏灰的低饱和藏青。
    #   只做 10%，色相与深浅关系不变，近岸浅水仍偏青。
    _lum_sea = rgb.mean(axis=2, keepdims=True)
    rgb = (rgb * 0.90 + _lum_sea * 0.10).astype(np.float32)
    # v77: 海洋光照 0.90+0.08·shade → 0.86+0.16·shade。海洋也吃一半的宏观光，
    #   让洋中脊受光侧比背光侧亮出一档，与下方的细尺度晕渲叠加成"立体洋底"。
    rgb = (rgb * (0.86 + shade * 0.16)[..., None]).astype(np.float32)
    _mark("tw_ocean")
    # ---- v55: 海底地形晕渲 (bathymetric hillshade) ----
    # 改了什么: 深度配色已在上方交给 Terrain Web 三段式 ocean LUT（浅青→中蓝→藏青），
    #   这里只保留 v48~v52 逐轮调好的海底晕渲（局部起伏 → 梯度 → ±6% 亮度）。
    # 为什么: 参考实现几乎不给海洋晕渲（按到岸距离取平色）；我方 DEM 有真实洋中脊/
    #   海山/海沟，保留这层起伏才能读出海底地貌；幅度已收敛到 ±6%，不会与新的
    #   三段式配色打架。
    if sea.any():
        depth = elev_m.astype(np.float32)
        # ---- v48: 海底地形晕渲 (bathymetric hillshade) ----
        # 用户反馈"深蓝大洋极其平坦, 看不到洋中脊/海沟". 根因是 DEM 里其实**有**
        # 这些起伏 (enhance_ocean_floor 已加洋中脊 +320m / 海山 / 深海丘陵, 海沟由
        # trench_deepening 挖到 -9000m), 但 320m 在 -4000m 底色上对应的色卡变化只有
        # ~0.012 亮度 —— 远低于可辨阈值, 渲染层把它们全部抹平了.
        # 真实海底地形图 (如 GEBCO) 靠 **晕渲** 而不是色阶来表现地貌, 这里照做:
        #   ① 用归一化卷积 (只对海洋像素加权) 求 σ=3px 平滑深度, 避免海岸陆地值污染;
        #   ② v49: 减去 σ=28px 的**低频趋势**, 得到"局部起伏"再求梯度 —— 否则陆架坡折
        #      (6px 内从 -150m 掉到 -2100m) 的巨大单调梯度会占满归一化基准, 让整条海岸
        #      被照亮成 ±0.28 的霓虹描边 + 外侧投下一圈暗影 (用户看到的"发光等高线");
        #   ③ v49: 幅度 0.28 → 0.10, 且在 250m 以浅线性淡出 (浅海不做晕渲).
        _sea_f = sea.astype(np.float32)
        _den = gaussian_filter(_sea_f, 4.5)
        _num = gaussian_filter(depth * _sea_f, 4.5)
        dep_s = np.where(sea & (_den > 0.15), _num / np.maximum(_den, 1e-3), 0.0)
        dep_s = dep_s.astype(np.float32)
        # v51: 去趋势尺度 28px → 12px. σ=28 会把海山/洋中脊这一类"几十像素宽的隆起"
        #   与趋势一起减掉, 只留下它们边缘的梯度 —— 渲染出来是几片**柔边亮斑**
        #   (用户在深海区看到的"云雾块"). σ=12 只去掉大陆尺度缓变, 海山/脊轴本身
        #   保留为有明确上坡/下坡的立体地形; 幅度 0.10 → 0.075 避免又变成噪声.
        # v61b: σ 12 → 8。σ=12 的去趋势只留下"上百像素的柔边隆起"，1:1 裁图里
        #   是几块发亮的云雾斑；σ=8 保留脊轴两侧的真实上/下坡面，晕渲贴着地形走。
        # v67: 单尺度（σ8 去趋势 + ±10% 钳位）→ **双尺度 + 大幅提幅**。
        # 改了什么: ① 去趋势拆两档：lo = dep_s − σ30px（洋中脊/深海盆地这类
        #   上百像素的起伏），hi = dep_s − σ7px（海山/深海丘陵）；
        #   ② 幅度 0.075 → 0.18，钳位 ±10% → ±22%，两档按 0.55/0.45 加权。
        # 为什么: 1:1 裁图（/tmp/c_ocean.png）里洋面是一整块平涂藏青，只有
        #   ±10% 的弱明暗，缩略图上完全读不出脊轴走向 —— 用户第 3 条「海洋之类的
        #   地形没有了」。±22% 才够在深蓝底色上显出"受光坡亮 / 背光坡暗"的立体感；
        #   分两档是为了不让大尺度隆起被 σ8 去趋势削成柔边云雾（v52 的老问题）。
        # v80: 去趋势尺度 30/7 + 细颗粒 σ2.5 → **90/40 两档，输入先 σ7 平滑**。
        # 为什么: 用户 v79 反馈「海底出现奇怪的波纹」。拆解实测（/tmp/preview_attr.py，
        #   同一块开阔洋）：只留色阶时洋面完全平滑、只留色调层时也是平滑团块，
        #   **波纹全部来自晕渲** —— σ7 与 σ2.5 两档的梯度作用在 1~5px 的 DEM 颗粒上，
        #   被 v79 的提亮/提幅放大成一层等间距的"蜂窝"。晕渲的作用是补坡向，不该
        #   去表现 5px 级的噪声：现改为先对起伏面做 σ7 平滑，再只取 90/40 两档
        #   （洋中脊/海山裙坡、深海盆地边缘），细节完全交给 pd 色阶与高程色调层。
        # v82: median5/σ2.5 → median9/σ3.0；带通 75/24 → 110/45。
        # 为什么: 24px 带通正好落在 GAN 颗粒的波长上，它的梯度就是用户看到的
        #   等间距波纹；75px 一档又太大，两者之间 25~75px 的真实地形反而没被表达。
        #   110/45 把能量全部压到 45px 以上（海山裙坡、脊轴、盆地边缘），
        #   低于 45px 的细纹理交给下方分形噪声（各尺度自相似，没有固定波长）。
        dep_sh = gaussian_filter(median_filter(dep_s, 9), 3.0)
        rel_lo = dep_sh - gaussian_filter(dep_sh, 110.0)
        rel_hi = dep_sh - gaussian_filter(dep_sh, 45.0)
        _ref_zone = sea & (depth < -800.0)
        def _bsh(_rel):
            _gy, _gx = np.gradient(_rel)
            _gsum = (_gx + _gy).astype(np.float32)
            _sc = float(np.percentile(np.abs(_gsum[_ref_zone]), 85)) \
                if _ref_zone.any() else 1.0
            return (_gsum / max(_sc, 1e-3)).astype(np.float32)
        # v82: 分形细节改为**带限**（σ20 高通 + σ1.2 预平滑），权重 0.14 → 0.45。
        # 为什么: v81 把 0.14 的分形叠在 24px 带通之上，波纹主因仍在带通；现在带通
        #   已退到 45px 以上，细纹理只能靠分形补 —— 但必须是各尺度自相似的
        #   （1/f^1.7），不能是单一波长。σ1.2 先削掉 2px 以下的极高频，避免读成
        #   砂纸颗粒；σ20 高通保证它只补"质感"，不自己造几十像素的假地形。
        #   随机种子固定，保证同 seed 可复现。
        _kk = np.sqrt(np.fft.fftfreq(S)[:, None] ** 2
                      + np.fft.rfftfreq(S)[None, :] ** 2)
        _kk[0, 0] = 1e9
        _ph = (np.random.RandomState(7001).randn(S, S // 2 + 1)
               + 1j * np.random.RandomState(7002).randn(S, S // 2 + 1))
        _fdet = np.fft.irfft2(_kk ** -0.85 * _ph, s=(S, S)).astype(np.float32)
        _fdet = (_fdet - gaussian_filter(_fdet, 20.0)).astype(np.float32)
        _fdet = gaussian_filter(_fdet, 1.2).astype(np.float32)
        _fdet /= max(float(_fdet.std()), 1e-3)
        _gy_f, _gx_f = np.gradient(_fdet)
        _fg = (_gx_f + _gy_f).astype(np.float32)
        _fsc = (float(np.percentile(np.abs(_fg[_ref_zone]), 85))
                if _ref_zone.any() else 1.0)
        _fg = (_fg / max(_fsc, 1e-3)).astype(np.float32)
        # v76: 配比 0.55·lo + 0.45·hi → 0.22·lo + 0.58·hi + 0.20·fine(σ2.5)。
        # 为什么: 1:1 复核（/tmp/c_ocean.png）开阔洋面上是**边界柔和的淡蓝圆盘**
        #   （用户说的"霉斑"）。圆盘本体其实是 deep 色（实测 rgb 10,51,89），
        #   让它读成"云"的是 σ30 去趋势的**大尺度偶极**：一个几十像素的圆丘在
        #   西北光下必然是一半亮一半暗、边缘羽化，正是云朵的读法。
        #   真实海底的自相似纹理主要在 5~20px（深海丘陵/海山裙坡），所以把权重
        #   从"大尺度"挪到"细节"，并补一档 σ2.5 的颗粒 —— 洋面从"几团云"变成
        #   "一层有颗粒的深水"。
        # v80: 只保留两档（0.40/0.60），删除 σ2.5 细颗粒档 —— 见上方说明。
        # v82: 0.50/0.50/0.14 → 0.55/0.45/0.45（真实地貌两档 + 带限分形细节）。
        # 细节项再乘一道**坡强门控** _fmod（0.45~1.0）：分形振幅若处处相同，1:1 下
        #   整片洋面像一张均匀的砂纸，这本身就是"程序生成感"的来源；真实海底的细
        #   纹理在平坦的深海平原上几乎看不见，越靠海山裙坡/脊侧越密。门控直接取
        #   45px 带通的归一化梯度的绝对值，原地形越陡、细节越强。
        _fmod = (0.45 + 0.55 * np.clip(np.abs(_bsh(rel_hi)) * 1.4, 0.0, 1.0)
                 ).astype(np.float32)
        _gsum = (0.55 * _bsh(rel_lo) + 0.45 * _bsh(rel_hi) + 0.45 * _fmod * _fg)
        # v51: 近岸 20~50px 内淡出晕渲. 陆架坡折是全图最大的单调梯度 (2000m/6px),
        #   即使幅度只有 0.075, 它也会在坡折内侧打出一条亮边、外侧压出一圈暗影 ——
        #   合起来正是用户说的"发光描边 + drop shadow"的 bevel 观感. 晕渲在地质上本就
        #   只用于深海丘陵/海山/脊轴, 近岸直接关掉.
        _dsea = _dsea0                     # v55c: 复用上面已算好的到岸距离(省一次 EDT)
        # v52: 幅度 0.075 → 0.045. 全流程重渲后发现深海是**大片柔边亮斑** (在 1024
        #   预览里读成云雾): 梯度晕渲对海山/脊轴必然产生"向光坡亮 + 背光坡暗"的偶极,
        #   而海底起伏的尺度 (几十~上百像素) 恰好落在最容易读成云团的频段. 只保留
        #   刚好能看出脊轴走向的强度, 不做成阴影.
        # v61: 幅度 0.045 → 0.075, clamp ±6% → ±10%.
        #   改了什么: 用户反馈「海洋之类的地形没有了」。v52 把幅度压到 0.045 是为了
        #     避免深海柔边亮斑被读成"云雾"；但配上 v61 的深度分色后，底色已经能区分
        #     脊/盆，晕渲只需把"上坡向光、下坡背光"补出来，0.075 与 ±10% 刚好够读出
        #     洋中脊走向，又不到 v49 那版 0.10 的霓虹描边（近岸仍有 18px 淡出）。
        # v67: 0.075 / ±10% → 0.18 / ±22%（见上方的双尺度说明）。
        # v72: 0.18 / ±22% → 0.26 / ±30%。
        # 为什么: v71 出图（/tmp/full900.png）缩略尺度上洋面仍是一整块藏青，
        #   v67 的 ±22% 在深海平原底色 (6,33,63) 上只有 ±10 个色阶，900px 下
        #   不足 2 级灰度 —— 用户第 3 条「海洋地形没有了」。0.26 配 ±30% 后
        #   脊轴/盆地的受光-背光差约 22 个色阶，缩略图能读出走向；近岸淡出
        #   与 ±250m 以浅的淡出保持不变，避免又出现霓虹描边。
        # v74: 0.26 / ±22% → 0.14 / ±14%。
        # 为什么: v73 出图（/tmp/c_ocean.png）里洋面是一圈圈**柔边亮斑 + 暗影的
        #   偶极子**，900px 缩略图上读作"水底长蘑菇"。晕渲本身没错，但 0.26 的
        #   幅度配 ±22% 钳位，在深海平原底色 (10,48,88) 上已是 ±20 个色阶，
        #   而海底起伏的尺度（几十~上百像素）正好是最容易被读成"团块"的频段。
        #   分层设色（pd 四带）已经把脊/盆/沟分开，晕渲只需要补出坡向。
        # v76: 0.14/±14% → 0.17/±18%。提幅是给上面新增的细节项留出可见度
        #   （细尺度梯度的 85 分位比大尺度小得多，不补幅度会被冲淡到看不见）。
        #   幅度仍远低于 v72 的 0.26，因为大头已从"大团偶极"换成"细颗粒"，
        #   不会重新出现 v73 的"水底长蘑菇"。
        # v77: 0.17/±18% → 0.24/±22%。
        # 为什么: 实测局部起伏（σ18 高通）std=349m，与洋面亮度的相关系数已有 r=0.40，
        #   说明晕渲方向是对的，只是幅度不够 —— 0.17×±18% 落在 33 的中位亮度上
        #   只有 ±5 个色阶，缩略图上读不出坡向。0.24/±22% 给到 ±8 色阶，配合上方
        #   加宽的深度增益（1.10→0.68）应该能把脊轴/海山的受光坡读出来。
        # v80: 0.24/±22% → 0.30/±24%。平滑后的两档梯度比原来小，需要补回幅度；
        #   幅度提在"坡向"上，不会重新把 5px 颗粒放大（颗粒已被 σ7 平滑掉）。
        # v81: 0.30/±24% → 0.36/±30%。真实地貌恢复后梯度更大，幅度同步提一档；
        #   分形细节只占 0.14，不会把 5px 颗粒放大成波纹。
        # v82: 0.36/±30% → 0.40/±34%。带通整体退到 45px 以上后梯度变小，需要补
        #   一档幅度才能在缩略尺度读出脊轴；颗粒已被 median9 铲掉，不存在被放大的
        #   单一波长，提幅只会放大真实坡向。
        _amp = (0.40
                * np.clip((-depth - 250.0) / 400.0, 0.0, 1.0)
                * np.clip((_dsea - 18.0) / 26.0, 0.0, 1.0))
        bshade = np.clip(1.0 + _amp * _gsum, 0.66, 1.34).astype(np.float32)
        rgb[sea] = (rgb[sea] * bshade[sea][:, None]).astype(np.float32)
    _mark("sea_layer")
    # 合成: Terrain Web 连续高程色覆盖海洋底色
    # 改了什么: 删除 v7.8 的 5 类地貌掩膜（classify_landforms）与随之的沙丘/高原/
    #   山地/盆地/丘陵纹理，以及 v35b 的暗部补光 + 明度钳制。
    # 为什么: 这些纹理是旧管线为"弥补群系离散色太平"加的，每类都往陆地上刷一层
    #   常量色（沙丘黄褐/高原灰褐/山地深棕/盆地盐白），5 类叠加后陆地主色 >6 种、
    #   读作"随机色斑"，正是用户对比 Terrain Web 时指的差距。参考实现的陆地只有
    #   一条高程色带 + 森林湿度混合 + 单一光照，起伏感由 detail 项承担，不需要
    #   分地貌上色；删除后 classify_landforms 的重计算也一并省下（约 2-4s）。
    # v62: 高纬海洋冷调。1:1 复核极区：冰缘外侧仍是 LUT 的亮青近岸色，在白色冰盘
    #   之间读作"游泳池"（用户「南北极效果也不行」）。真实极地海是低饱和的钢蓝。
    #   这里只把 |lat|>52° 的海面向 (0.13,0.22,0.30) 收 42%，陆地随后覆盖不受影响。
    _lat_abs = np.abs(np.asarray(lats, np.float32))[:, None]
    # v63: 0.42 → 0.20，色偏改成更亮的冷蓝 (0.15,0.24,0.32)。
    #   1:1 复核：0.42 的强混合把极区海面压成近黑藏青，与白冰并置时边界像"剪贴"；
    #   减弱后极区海仍是同一套海洋蓝，只是偏冷一档。
    _cold = (np.clip((_lat_abs - 52.0) / 16.0, 0.0, 1.0) * 0.20).astype(np.float32)
    rgb = (rgb * (1.0 - _cold[..., None])
           + np.array([0.15, 0.24, 0.32], np.float32)[None, None, :]
           * _cold[..., None]).astype(np.float32)
    # ---- B2: 海岸抗锯齿（只改合成权重，不改 DEM / 地形 / 配色）----
    # 改了什么: 旧的合成是 `np.where(land, land_rgb, rgb)` —— 逐像素二值硬切。
    #   现改为用「到岸有符号距离场」求亚像素覆盖度 α，在岸线两侧各羽化 ~1.2px:
    #       sd   = edt(land) − edt(~land)      （陆侧>0 / 海侧<0，单位 px）
    #       sd_s = gaussian(sd, 0.8)           （把 1px 台阶揉成曲率，纯亚像素量）
    #       α    = clip(0.5 + sd_s / 2.4, 0, 1)
    #       rgb  = ocean·(1−α) + land·α
    # 为什么: B2 要治的是「海岸 1–2px 碎齿」。量化诊断（/tmp/b2_diag.py,
    #   /tmp/b2_diag2.py, world_dem.npy 口径）证明 DEM 侧其实很干净:
    #     单像素尖刺 7px / 单像素缺齿 1px / 开运算 r=1 只削掉 207px（陆地的 0.17‰），
    #   即二值陆海掩膜本身没有毛刺。碎齿全部来自渲染: 2048 层的二值边界在 1:1 下
    #   是一条 1px 的直角楼梯，×12 放大（/tmp/b2_pixel12.png）里每级台阶都是一颗
    #   "牙"；海色(LUT 深蓝)与沙滩色(米色)的色差又极大，硬边把台阶衬得非常显眼。
    # 为什么不改 DEM: 掩膜已无可清理的碎点，且 A1/B1 已按 md5=a07375fd… 验收，
    #   在 DEM 上做形态学只会白改地形、破坏已验收状态。故 B2 与 A1 一样属渲染层。
    # 幅度的取舍: 羽化带总宽 2.4px（±1.2px）恰好盖住 1px 台阶的半个周期；再宽会
    #   把沙滩带内侧也糊掉、让远洋的礁石失去轮廓。α 在陆地内部恒 1、深海恒 0，
    #   只有岸线 ±1.2px 内是过渡，因此全图 99% 像素的值不变。
    _sd = np.where(land, distance_transform_edt(land),
                   -distance_transform_edt(~land)).astype(np.float32)
    _alpha = np.clip(0.5 + gaussian_filter(_sd, 0.8) / 2.4, 0.0, 1.0)[..., None]
    rgb = (rgb * (1.0 - _alpha) + land_rgb * _alpha).astype(np.float32)
    _mark("land_overlay")
    # ---- v56/v57: Terrain Web 的 atlas stipple（岩线 / 深海短划）----
    # 改了什么: 补上参考实现 generate_terrain 末尾三段"输出分辨率细节符号"
    #   (terrain_web.py L1443-1486)；其中「密排树点」一段在 v57c 按用户要求删除
    #   （见下方注释），只保留岩线与深海短划。
    # 为什么: 上一轮 v55c 与参考并排复核，陆地色相/明度已对齐(均值 108.9,139.3,90.8
    #   vs 参考 110,132,79)，但质感仍差——参考的岩坡是细碎的灰褐短线、远洋有短划，
    #   我方是一整片平涂色块(主色 (72,96,48) 占 13.3%)。短线类符号能把宽色带打散，
    #   又不至于像树点那样在 2048 输出上糊成噪点。
    # 尺度: 参考粗格 = clamp(width//2.9, ·, 700) → 2048 输出下 700 格 / 每格 2.9px。
    #   本图沿用上面光照段已算好的 256 粗格不行(每格 8px, 符号会变成 8px 的疤)，
    #   故单独按 2.9px/格 重新分块。
    # v57: 旧写法 int(S/2.9)=706 → S//706=2 → f=2、G=1024（每格 2.0px），比参考的
    #   2.9px/格 细 45%，符号被压成 2px 且格数多 46%。
    #   改为先定每格像素 f=round(S/700)（2048→3），再反推 G=S//f=682，与参考同尺度。
    #   注意 S 不一定被 f 整除（2048 = 682*3 + 2），分块前必须先裁到 G*f 的整数倍。
    # v64: **整段 atlas stipple（岩线 / 深海短划）删除，不再保留任何短线符号。**
    # 改了什么: 删除 v56~v63 的「岩线」绘制（682 粗格 × Top-K 陡坡 → 3px 斜线）。
    # 为什么: 用户图1 的「这是什么鬼」在 1:1 上复现（/tmp/c_npole.png、c_ocean.png）——
    #   陡坡格沿等高线连续选中，即使做过哈希抽稀，3px 斜划按 2.9px 格距铺开仍然
    #   在视觉上连成 **一片 X 形斜纹布**；缩略图上则退化成山地里的白色麻点。
    #   参考实现能用这套符号，是因为它的 detail 贴图本身已有逐像素颗粒，
    #   符号只做点缀；我方色带是 16px 粗格插值的连续带，符号成了唯一的"纹理来源"，
    #   于是被迫画得很密，反而暴露了它是人工符号。
    # 现在: 纹理全部交给下方 v64 的局部对比增强（真实 shade 高通），不再有符号层。
    print("      [v64] atlas stipple: 岩线 / 深海短划 / 树点 全部关闭（改为 shade 高通纹理）")
    _mark("tw_stipple")
    # 湖泊: 填洼面与实际地形之差 → 湖面 (盆地积水)
    # v39i: 填洼一次给三处复用 (湖面 / 河流流场 / 内流盆地判定)
    _flow_v, _fill_depth = hydrologic_surface(v01)
    # v40: 分级湖盆 —— 每个封闭盆地只保留**一个**连通湖体, 水量按盆地规模收缩.
    # 诊断 (seed 1504334501, land=1.282M px): fill_depth>0.018 占 21.6% 陆地, 其中
    #      **单个连通盆地就占 13.8%**, 其余 875 个分量里只有 14 个 >0.1% —— 图中
    #      "大量形状怪异的内海"其实是同一个巨型封闭洼地被整块涂色, 不是很多湖.
    # 模型 (物理): 内流盆地的水面由"来水 = 蒸发"平衡决定, 远低于溢流高程, 且
    #      盆地越大越可能位于干旱内陆 → **盆地内积水比例**越小 (里海约占其盆地
    #      面积的 15%, 塔里木盆地几乎无水). 因此 盆地内水面占比 =
    #      clip(0.55 − 3.2·盆地占陆比, 0.05, 0.55); 每个盆地的水面再由
    #      :func:`basin_lake_mask` 二分水位求出单一连通体, 岸线是天然等高线,
    #      而不是 v39 的整块刷蓝/v35 的整块刷白.
    #      总量按"占陆比"求和后统一缩放 (而非先到先得), 保证大盆不吃光预算、
    #      中小盆地都能留下湖 —— 目标 3.5% 陆地 (地球湖泊+内海 ≈2%).
    # 湖面判定用**平滑后**的填洼深度: 填洼深度 = filled − v01, 含 GAN 高频起伏,
    # 原始等值线切出的岸线是锯齿状树枝 → 现实中湖岸是平缓等高线, 平滑 2px 即可.
    _fd_s = gaussian_filter(_fill_depth, 2.0)
    _land_n_v = int(land.sum())
    _wet = land & (_fd_s > 0.018)
    lake = np.zeros_like(_wet)
    if _wet.any():
        _lab_v, _n_v = ndi_label(_wet)
        _sz_v = np.bincount(_lab_v.ravel()); _sz_v[0] = 0
        _cands = np.nonzero(_sz_v >= 2048)[0]   # v61b: 512 → 2048 (见下方 _cap 说明)
        _cands = _cands[np.argsort(_sz_v[_cands])[::-1]]    # 大盆在前 (仅影响日志)
        _share = _sz_v[_cands] / max(_land_n_v, 1)          # 盆地占陆比 a
        _fill = np.clip(0.55 - 3.2 * _share, 0.05, 0.55)    # 盆地内水面占比
        _want = _share * _fill                              # 折算到占陆比
        # v61b: 预算 3.5% → 1.8% 陆地, 候选盆地阈值 512 → 2048px。
        #   改了什么: 用户第 2 条「大陆板块过于支离破碎」。实测旧参数在 2048 图上
        #     成湖 30 个、水面 2.2% 陆地, 大块大陆腹地被浅蓝湖斑打散成"蜂窝"。
        #   为什么: 地球内陆+湖泊合计 ≈2% 陆地, 且集中在少数巨型水文单元; 碎湖是
        #     填洼产物, 不是地理实体。提高候选门槛后, 只有真正的巨型盆地才成湖,
        #     大陆内部重新读作整块陆地。
        _cap = 0.018
        _tot = float(_want.sum())
        if _tot > _cap:                                     # 等比压缩, 不淘汰
            _want *= _cap / _tot
        _used, _nb = 0, 0
        for _k, _bi in enumerate(_cands):
            _tgt = int(_want[_k] * _land_n_v)
            if _tgt < 256:
                continue    # 只剩零星碎洼, 撑不起湖面
            _m = basin_lake_mask((_lab_v == _bi), _fd_s, _tgt)
            _got = int(np.count_nonzero(_m))
            if _got:
                lake |= _m
                _used += _got
                _nb += 1
        print(f"      [v40] 分级湖盆: 成湖 {_nb} 个 / 候选盆地 "
              f"{len(_cands)} 个 → 水面 {_used}px "
              f"({_used / max(_land_n_v, 1) * 100:.1f}% 陆地), 巨型盆地 "
              f"{int(_want[0] * _land_n_v)}px, 其余 "
              f"{int(_wet.sum()) - _used}px 浅洼回填为陆地")
    # 湖水着色: 用填洼深度当"水深"代理 (fd 越大 = 盆地越深 = 水越深).
    #   浅滩荧青 → 深湖墨蓝, 与海深过渡同一套色相, 湖缘自动出现浅色带,
    #   不再是 v39 那样整片常量深蓝 (视觉上像在陆地上打洞).
    # v47: 浅滩端与海洋同款压暗 (旧 0.42/0.66/0.74 在近岸形成亮青镶边).
    # v55: 湖色改用 Terrain Web 的 shallow→mid（与海洋 LUT 同一色相）。
    # 改了什么: 旧的亮青(0.34,0.53,0.62)→墨蓝(0.16,0.36,0.52) 换成海洋浅层/中层色。
    # 为什么: 湖与海同色相后，湖岸浅色带与岸缘 foam 一致，不再出现"在陆地上打洞"
    #   的高对比蓝斑；深浅仍由填洼深度（=水深代理）线性过渡。
    # v74: 水深代理分母 0.10 → 0.16、湖色混合 0.82 → 0.72。
    # 为什么: 1:1 复核（/tmp/c_land.png）里大陆腹地的湖是**带黑边的深蓝窟窿**，
    #   与周围草地对比拉满 → 九宫格里读作"陆地上被打了洞"，正是用户第 2 条
    #   「大陆过于支离破碎」的一半来源。湖本该是浅青到中蓝的水面，dark end 只在
    #   极深的湖心出现；分母放大后整湖落在 shallow~mid 之间，边缘不再像剪洞。
    _dl = np.clip((_fill_depth - 0.018) / 0.16, 0.0, 1.0)[..., None]
    _lake_col = (_TW_COLORS["shallow"] * (1.0 - _dl)
                 + _TW_COLORS["mid"] * _dl).astype(np.float32)
    rgb[lake] = rgb[lake] * 0.28 + _lake_col[lake] * 0.72
    # 河流: 填洼气候加权流累积 → 可见河道 (水量=降水权重×融雪门控)
    # v39: D8 → 多流向 (MFD). D8 在 M5 克拉通倾斜面上产生等间距平行直线,
    #      且干流被拆成互不连通的单像素线; MFD 在坡面散开、汇流处收敛.
    # v39h: 流场微地形抖动 — 无效已废弃; v39i: 改用 hydrologic_surface 的域扭曲
    #       填洼面 (见该函数 docstring: 填洼 ε 梯度在盆地内累积成 1000m 级人工直线坡,
    #       叠加 30m 抖动根本压不过它, 只有整体域扭曲才能打散直线而不打断河网).
    acc = climate_flow_accumulation(_flow_v, P, T, rng, mfd=True)
    # v39i: 内流盆地 (盐沼/干湖床) 抑制穿盆河 —— 填洼深度 > 0.02v 的洼地底部,
    #       河道完全是填洼产物, 现实里是干盐滩, 不该有贯通水系.
    acc = (acc * np.clip(1.0 - (_fill_depth - 0.02) / 0.05, 0.0, 1.0)).astype(np.float32)
    # v39c: 累积量 fBm 扰动 — MFD 在 M5 克拉通倾斜面上仍会沿等高带产生长直河段
    #       (acc 等值线的直线段 = 视觉上的"地面裂缝"). 用多尺度 fBm 调制 acc 幅值,
    #       让等值线随地形噪声弯曲, 直边被打散成自然蜿蜒的河型.
    _acc_warp = 1.0 + 0.40 * np.clip(value_fbm(S, 42.0, 3, rng=rng), -1.6, 1.6)
    acc = (acc * _acc_warp).astype(np.float32)
    _mark("fill+climate_flow_accumulation")
    # v33: 0.995 → 0.985 让支流也可见, 消除"无源之水"感 (0.5% 分位稀疏不足以呈现水系骨架)
    # v39: 按干燥度门控河网密度 + 按海拔收窄山区河道
    # v48: 0.985 → 0.979 —— 用户反馈"大块陆地上几乎看不到树枝状河网". 2.1% 陆地
    #   覆盖对应一级+二级支流可见, 山脉里也能读出汇流结构; 干燥度衰减在
    #   paint_rivers_rgb 内同步从 0.45 降到 0.28.
    _dry_riv = gaussian_filter(np.clip(1.0 - P / 1200.0, 0.0, 1.0).astype(np.float32), 6.0)
    # v53: 0.979 → 0.972. 用户仍反馈「几乎看不到树枝状河网」—— 全图尺度下 1px ≈ 20km,
    #   亚马逊本体也只有 2~4px, 所以不能靠加粗 (会变成蓝色蛛网); 提高可见度的正确做法
    #   是把**流量门槛**下探一档, 让三级支流进入画面, 河网的"羽状"结构才连得起来.
    # v55: 河色改用 Terrain Web 的 river(30,112,140) 青蓝三档。
    # 改了什么: 旧的深海军蓝 (0.10,0.26,0.42)/(0.08,0.22,0.38) 换成与湖/海同色相的
    #   青蓝，主河略深一档。
    # 为什么: 参考实现河流与浅海共用同一青蓝色系，水系因此读作"同一套水"；
    #   旧的深海军蓝在与新海洋配色并置时会显脏（蓝相偏紫）。
    # v72: base_q 0.972 → 0.958。
    # 为什么: 900px 出图（/tmp/full900.png）里陆地上**一条河都读不到** —— 0.972
    #   只留流量前 2.8% 的像元，2048 图幅上河宽中位 2px，缩到 900 后不足 1px 且
    #   被后续 0.34~0.86 的混合压淡。用户第 2 条明确要求"大/小河、落差、源头"，
    #   河网必须能读出羽状分支；0.958 把三级支流纳入（核心像元约 4.2%），
    #   W_half 的 A_star 幂次不变，干流不会变粗。
    rgb = paint_rivers_rgb(rgb, acc, land, rng, base_q=0.958,
                           color_riv=np.array([0.13, 0.44, 0.55], np.float32),
                           color_big=np.array([0.10, 0.34, 0.47], np.float32),
                           oxbow_color=np.array([0.09, 0.30, 0.42], np.float32),
                           dryness=_dry_riv, relief_m=elev_m)
    _mark("paint_rivers_rgb")
    # 雪线: v55 起并入高程 LUT（t>0.82 → rock_hi→snow 渐变，本图 relief=4915m
    #   → 约 4030m 起泛白、4620m 全白），不再单独成层。
    # 改了什么: 删除「纬度雪线 ∪ 温度雪线 × 山脊骨架」整段（旧 L810-832）。
    # 为什么: 参考实现的积雪只由高程色带决定（山越高越白，与纬度无关），而旧层
    #   在 |lat|>60° 会把低地平原一起刷白（"白蛇"根因），且与高程色带叠加后出现
    #   双层雪 → 色阶断裂。极区纯白仍由下方 polar_ice 负责。
    # 极地冰盖：>65° 陆地强制全覆盖（模拟南极/格陵兰冰盖）
    # 65°≈南极圈，地球南极大陆几乎全部在此纬度以南被冰盖覆盖
    # v33: 硬阈值改 64°→66.5° 渐变斜升 (原来 >65° 一刀切 → 与下方植被带呈突兀直边)
    # v39: 冰盖南界加长波起伏 (fbm1d, 波长~200列) + 内部纹理 → 摆脱"纬度直尺切边"
    #      与"纯白方块"观感; 冰盖边缘按 fBm 破碎 (冰川末端/冰架/漂冰带).
    # v48: 冰盖重做 —— 旧版把 _ice_tex (0.42~1.0 的低频 fBm) 乘进 **alpha**, 冰盖
    #   因此半透明、底下群系颜色透出, 用户看到的是"卷积云/羽状遮罩"而不是冰盖.
    #   现在: ① alpha 只由纬度 + 破碎边界决定, 核心严格不透明;
    #         ② 边界过渡 3.2°(≈36px) → 1.2°(≈13px), 叠加 σ=9px 的 2D 高频噪声让
    #            冰缘像冰川末端一样锯齿破碎, 而不是纬向直尺;
    #         ③ 表面纹理只改**颜色亮度** (裂隙偏蓝灰), 不参与 alpha.
    _jag = np.clip(_lowres_fbm(S, 9.0, 3, rng=rng), -1.4, 1.4)
    # v61: 用户反馈「南北极效果也不行」。陆地冰盖南界 63° → 66°，锯齿幅度
    #   2.4/1.6 → 3.6/2.8 —— 旧冰缘是一条 ±3° 的缓波，1024 预览里读作"纬度直尺"。
    # v61b: 66° → 68.5°。v61 把冰缘抬到 66° 后，1:1 复核仍有大片低纬陆地被冰盖，
    #   而且 66° 落在本图两个大陆的腹地，白色直接从大陆内部开始、没有海岸线过渡。
    #   抬到 68.5° 后冰盖基本只覆盖极区岛屿与大陆最北/最南端。
    # v62: 68.5° → 71.0°，过渡 0.8° → 1.8°。南极洲/格陵兰的冰盖南界在 66~70°，
    #   我方陆地本来就画到极区，68.5° 的硬边让两侧大陆在 1:1 里直接顶到图幅的白带
    #   上（用户「南北极效果也不行」）。抬到 71° 且拉长过渡后，陆地冰只覆盖真正的
    #   极区，冰缘变成一条有起伏的碎裂带。
    _ice_edge = 71.0 + 3.6 * fbm1d(S, 200, rng) + 2.8 * _jag
    _ice_ramp = np.clip((np.abs(lats)[:, None] - _ice_edge) / 1.8, 0.0, 1.0)
    polar_ice = (_ice_ramp * land).astype(np.float32)
    # v55: alpha 现在只由极地冰盖决定（雪线已并入高程 LUT），不再与非雪层取 max
    alpha = polar_ice
    a3 = alpha[..., None]
    rgb = rgb * (1 - a3) + np.array([0.96, 0.97, 0.99])[None, None, :] * a3
    # 冰盖表面: 厚度感由亮度起伏 + 裂隙 (ridged) 承担, 只作用于冰盖内部
    # v49: 纹理幅度大幅收敛 (0.10/0.20 → 0.04/0.07). 旧版的灰色长条形裂隙在山顶
    #   风成雪脊尺度 (26px 基频 × 3 octave, 方向 40°) 上会拉出上百像素的平行灰纹,
    #   在 1024 预览里就是用户说的"羽毛状卷积云". 真实冰盖在行星尺度上是近纯白 +
    #   微弱蓝色起伏, 细节靠晕渲而不是靠颜色噪声.
    if polar_ice.any():
        # v51: 去掉 direction=40° 的各向异性 ridged 场 (它在 1024 预览里是几十像素长的
        #   平行灰纹, 读成"羽毛"), 换成**各向同性**的双尺度场: 0.35 的宽缓起伏 +
        #   0.65 的细裂隙. 裂隙只轻微压暗并略偏蓝 (冰内蓝光透射), 不做成灰线.
        _ice_wide = _lowres_fbm(S, 8.0, 3, rng=rng)
        _ice_fine = _lowres_fbm(S, 40.0, 3, rng=rng)
        _tex = np.clip(0.5 + 0.32 * _ice_wide + 0.13 * _ice_fine, 0.0, 1.0)
        # v56: 冰盖接入同一套 NW 晕渲。
        # 改了什么: 旧版冰盖是恒定 0.975~1.00 的近纯白, 只在内部叠 ±4% 的 fBm 纹理;
        #   本图极区陆地占比大, 出图上/下各是一条几百像素高的**纯白平条**。
        # 为什么: 参考实现的雪带 (snow 245,247,242) 同样吃 shade 的 factor, 冰原的
        #   起伏 (冰穹/冰脊/谷地) 因此有明暗; 纯白无明暗在缩略图上读作"底图没渲完"。
        #   幅度收敛到 ±6% (0.90~1.06), 避免冰盖变成灰斑。
        _sh = np.clip(0.955 + 0.045 * _tex, 0.94, 1.0)[..., None]
        _sh = _sh * np.clip(0.72 + 0.30 * shade, 0.90, 1.06)[..., None]
        _ice_rgb = (np.array([0.972, 0.982, 0.998], np.float32)[None, None, :]
                    * _sh)
        # 裂隙: 高频细线, 略微偏蓝 (0.86, 0.91, 0.97) 而非灰
        _crev = np.clip((_lowres_fbm(S, 70.0, 2, rng=rng) - 0.55) / 0.20, 0.0, 1.0)
        _crev = (_crev * polar_ice)[..., None]
        _ice_rgb = (_ice_rgb * (1.0 - 0.30 * _crev)
                    + np.array([0.86, 0.91, 0.97], np.float32)[None, None, :]
                    * (0.30 * _crev))
        _w_ice = polar_ice[..., None]
        rgb = (rgb * (1.0 - _w_ice) + _ice_rgb * _w_ice).astype(np.float32)

    # v34: 冰盖融水水系 (南极/格陵兰型) — 冰盖下暗河 + 边缘融水辫状河.
    # 在冰盖着色之后重绘融水河道 → 白色冰盖上出现深蓝融水线条,
    # 冰盖边缘 (58-78°) 过渡带也可见, 消除"冰盖与河网割裂"感.
    _lat_abs_ = np.abs(lats)[:, None]
    _melt_belt = (_lat_abs_ >= 58.0) & (_lat_abs_ <= 78.0) & land & (alpha >= 0.5)
    if _melt_belt.any():
        _thr_mt = float(np.quantile(acc[land], 0.985))
        melt_riv = _melt_belt & (acc >= _thr_mt)
        if melt_riv.any():
            # v49: 只在冰盖内部 (alpha≥0.5) 画融水, 深度 0.55→0.35 —— 旧版把
            #   冰盖边缘的暗蓝细线一直画到 58°, 在缩略图上像一道划痕/云缝.
            _w_melt = np.clip(0.30 + 0.70 * alpha[melt_riv], 0.0, 1.0)
            rgb[melt_riv] = (rgb[melt_riv] * (1.0 - 0.35 * _w_melt[:, None])
                             + np.array([0.24, 0.42, 0.62], np.float32)
                             * (0.35 * _w_melt[:, None]))
    # 海冰: 纬度 + 噪声, 平滑边缘 (地球海冰边界约 70-75°)
    # v39: 78°±2.5° 近乎直线 → 74°±4° 短波长 (σ=110) 起伏 + 浮冰场羽化,
    #      消除"地图边缘被直尺裁掉"的观感 (用户反馈极区直边过于规整).
    # v48: 旧版 sigmoid 羽化宽 2.2°(≈25px) 且乘 0.85 透明度 + floe 抖动
    #      → 极区是一层半透明"羽毛云". 现在边界宽 0.9°(≈10px), 核心 alpha=1;
    #      浮冰噪声只在**外缘** (sig 0.30~0.55) 打碎, 核心保持整片冰盖.
    # v49: 再收敛一次. v48 把核心做到 alpha=1, 但外缘 _fringe 仍覆盖 sig∈[0.30,0.55]
    #   (≈±1.5°=17px) 且乘 0.5~1.0 的浮冰噪声 —— 在 1024 预览里仍然是一圈半透明
    #   淡蓝"羽毛", 用户读成"云层". 现在: 边界西格玛 0.9°→0.55°, 核心 (sig>0.60)
    #   完全不透明, 破碎带只占 sig∈[0.40,0.60] (≈1.4°=14px) 且不做整体半透明.
    # v51: 海冰重做 —— 旧版 (v49) 的外缘是 0.35+0.65*floe 的**软** alpha, 在 1024
    #   预览里就是一层淡蓝半透明纱 (用户: "像卷积云 / 遮罩没做好的 alpha 混合").
    #   海冰在行星尺度上应读成**成簇的白冰盘 + 深色冰间水道 (lead)**; 软 alpha 无论
    #   幅度多小都会被读成云. 现在用阈值化的高频噪声把 alpha 二值化 (过渡 ~4px),
    #   冰盘本体不透明, 冰间水道直接露出海水, 冰面亮度 0.93 (接近冰盖).
    # v54: 冰缘锯齿. v53 只有 110/220 像素两个低频项, 冰缘是一条几乎笔直的纬线,
    #   1024 预览里南北极各有一条"刀切"横边. 补一项 22 频率 (≈90px 波长) 的抖动.
    # v61b: 基准 72.5° → 75.5°，密集度斜坡 2.5° → 4.5°。
    #   改了什么: 旧参数下 |lat|>75° 就已经 _conc=1（实心冰），两极各一条 ~180px
    #     高的实心白带，横贯整个图幅，正是用户截图里"南北极一整条白横带"。
    #   为什么: 地球的海冰是**边缘破碎的冰盘群**（75~80° 密集度 60~90%，洋面仍有
    #     大量开阔水），只有极点附近才接近封闭。把冰缘外推、斜坡拉长后，白带变成
    #     一条越往极点越密的碎冰区，南边界不再是一条直线。
    # v62: 75.5° → 77.5°。极冠核心缩到 84° 以北，78~84° 是碎冰带（地球 9 月
    #   海冰边缘就在 75~80° 且破碎度很高），图幅顶端的实心白带因此变成冰盘群。
    # v75: 边缘起伏幅度 4.2/2.6/2.0 → 2.6/1.8/1.4。
    # 为什么: 1:1 裁图复核 v74（/tmp/c_npole.png）冰缘是"撕纸边"——冰缘本身是一条
    #   纬度等值线，±(4.2+2.6+2.0)≈±8.8°（≈100px）的抖动让它在 900px 缩略图上
    #   来回甩动，每一段凸出/凹进都被近黑的冷海衬成一圈黑边，读作"打翻的牛奶"。
    #   地球海冰边缘的纬度摆动量级是 ±2~3°（年际），不是 ±9°。收小到 ±5.8° 后
    #   边缘仍不成直线，但凸起从"手指"退回"缓弯"。
    edge_sea = (77.5 + 2.6 * fbm1d(S, 110, rng) + 1.8 * _jag
                + 1.4 * fbm1d(S, 22, rng))
    _lat2 = np.abs(lats)[:, None]
    # 冰密集度: 冰缘=稀疏冰盘, 极点附近=连片冰盖. 只用来调**阈值**, 不用来调 alpha ——
    # 这样每一块冰盘自身仍是不透明的硬边, 稀疏区看到的是一片散落的冰, 而不是薄纱.
    # v53: 密集度斜坡 4.0° → 2.5° (≈28px). 4° 的斜坡叠在二值化冰盘上, 在 1024 预览里
    #   是一圈越往外越稀的"雾边"; 2.5° 让冰缘集中成一条可辨认的碎冰带.
    # v63: 密集度斜坡 4.5° → 8.0°。4.5° 时冰缘往上 4.5° 就到达满密集度，
    #   碎冰带只有 ~50px 宽，缩略图上就是"白纸上撕了一道口子"；拉长到 8° 后
    #   77.5°~85.5° 是一条**由疏到密**的碎冰带（与地球海冰边缘的密集度梯度一致）。
    _conc = np.clip((_lat2 - edge_sea) / 8.0, 0.0, 1.0)
    # v52: 冰盘尺度 21px → 48px. 21px 的冰盘在 1024 预览里只有 10px, 一片麻点;
    #   海冰在行星尺度上应是"几十到上百公里一块"的冰盘, 稀疏区才出现小碎冰.
    _qn = value_fbm(S, 42.0, 3, rng=rng)
    _qf = value_fbm(S, 130.0, 2, rng=rng)
    # v69: **各向异性预平滑** —— 冰盘沿经向拉长。
    # 改了什么: _qn / _qf 先做 sigma=(2.0, 4.5) / (1.2, 2.8) 的椭圆高斯（行方向
    #   窄、列方向宽），再走原来的归一化 → 阈值化流程。
    # 为什么: 1:1 复核（/tmp/c_spole.png）冰间水道是**圆形墨滴**状的黑斑，极冠
    #   读作"白纸上溅了墨"；真实海冰的 lead 是**线状裂隙**（风/洋流把冰盘沿纬向
    #   拉开）。各向同性噪声无论怎么调阈值都只能得到等轴碎块，必须先把噪声本身
    #   拉长。椭圆的 2.3:1 只够把碎块读成"拉长的冰盘 + 线状水道"，过强会变成条纹。
    _qn = gaussian_filter(_qn, sigma=(2.0, 4.5))
    _qf = gaussian_filter(_qf, sigma=(1.2, 2.8))
    # v54: **归一化改为冰区局部**. v53 用**全图** [p20, p85] 归一化, 但 value_fbm 的
    #   低频八度在极区系统性偏低 → 整片区域被 clip 到 0 → 阈值以下形成几十像素大的
    #   连片"暗洞" (实测 |lat|>72 洞 204 个, 最大 12755px, 中位 479px), 1024 预览里
    #   极冠读成"被撕开的泡沫板". 改为只在冰区海洋内取分位, 洞退化成小水道.
    _pmask = (_lat2 > edge_sea) & (v01 < 0.5)
    if not _pmask.any():
        _pmask = np.ones_like(_lat2 > 0)
    # v54b: 分位锚点 p25/p92 → p3/p90. 用 p25 当下锚意味着冰区里**最低的 25%**
    #   全被 clip 到 0, 阈值一旦 <0.01 就整片露出海水 (实测 5.3% 暗水). 改 p3 后
    #   开阔水比例才真正由 _thr 控制, 而不是被归一化下限锁死.
    _lo, _hi = np.percentile(_qn[_pmask], [3, 90])
    _nb = np.clip((_qn - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0)
    _flo, _fhi = np.percentile(_qf[_pmask], [3, 90])
    _nf = np.clip((_qf - _flo) / max(_fhi - _flo, 1e-6), 0.0, 1.0)
    # v54: 大冰盘(0.70) + 细水道(0.30) 两种尺度叠加. 纯 42 频率只有大块, 打开的水面
    #   是"池子"; 混入 130 频率后水道呈网状, 与真实冰缘的 lead/fracture 一致.
    # v64: **补一层 ≈3px 的高频项**。1:1 复核（/tmp/c_npole.png、c_spole.png）：
    #   冰缘是一条 20~30px 的块状硬边（阈值化在 24px 噪声场上，等值线自然是大块的），
    #   碎冰带的边界因此读作"撕纸/剪纸"，而不是浮冰的自然破碎。
    #   0.62/0.24/0.14 的配比让高频只决定**边缘的毛刺**，不改变密集度的大格局。
    _qhf = value_fbm(S, 320.0, 2, rng=rng)
    _hlo, _hhi = np.percentile(_qhf[_pmask], [3, 90])
    _nhf = np.clip((_qhf - _hlo) / max(_hhi - _hlo, 1e-6), 0.0, 1.0)
    # v66: 配比 0.62/0.24/0.14 → 0.66/0.26/0.08（见下方 _sea_a 处的说明）。
    _norm = 0.66 * _nb + 0.26 * _nf + 0.08 * _nhf
    # v53: 阈值下限 0.90 → 0.95, 系数 1.04 → 0.92. 旧式在 conc=1 时阈值 = −0.14,
    #   而 _norm ∈ [0,1] → 极区冰盘 alpha **恒为 1**, 连片区域没有任何冰间水道,
    #   1024 预览里就是一张纯白低对比的"雾"(用户: 像卷积云).
    # v54: 0.95/0.92/0.10 → 0.90/0.86/0.07. conc=1 时阈值 0.04 → 开阔水 ~3%
    #   (v53 实测 9~10%, 极冠内部像漏了洞); 过渡带斜率更陡, 冰盘边缘更利落.
    # v54b: 0.90/0.86 → 0.87/0.86. 极冠内部(conc=1)阈值 0.04→0.01, 开阔水 6%→~1%.
    #   行星尺度 1px≈20km, 真实 lead 只有 1~10km **本就是亚像素**, 画成 20~40px 的
    #   "蓝湖"在任何缩放下都是"撕开的泡沫板". 破碎只保留在边缘冰区 (conc<0.8).
    # v61b: 0.87/0.86 → 0.84/0.76。conc=1（极冠核心）时阈值 0.01 → 0.08，_norm 是
    #   [0,1] 的均匀化噪声 → 核心也保留 ~5~8% 的开阔水（冰间湖/水道），极冠不再是
    #   一张白纸；边缘冰区（conc≈0.3）阈值 0.61，碎冰比例大幅提高。
    # v63: 0.80/0.72 → 0.62/0.50。极冠核心（conc=1）阈值 0.12 → 保留 ~12% 开阔水
    #   （冰间湖/水道网络），边缘冰区（conc≈0.3）阈值 0.47 → 只剩一半冰盘。
    #   v62 实测 |lat|≥80° 白像素 93.6%：整片极冠读作"切出来的一块白纸"。
    # v65: 冰密集度阈值微调 0.72/0.58 → 0.66/0.54。
    # 为什么: 1:1 复核（c_spole.png、c_npole.png）极冠内部是**黑色水道成片**的
    #   "破洞白纸" —— conc=1 时旧阈值 0.14 让 _norm 底部 14% 全成开阔水，而开阔水
    #   的底色在 v64 是近黑 (1,9,20)，对比拉满。收到 0.12 后核心更连续，碎冰感
    #   仍由边缘冰区（conc 0.2~0.5，阈值 0.43~0.55）承担。
    # v74: 阈值 0.66−0.54c → 0.60−0.56c。
    # 为什么: v73 出图（/tmp/c_spole.png）里极冠内部是**成排的灰蓝碎块浮在白底上**，
    #   900px 下读作"湖里的一堆碎岛"（用户第 4 条）。_norm 是 [0,1] 的均匀化噪声，
    #   核心阈值 0.12 意味着 12% 的冰面是开阔水；行星尺度上这些水是亚像素的，
    #   画成几十像素的蓝斑必然读成"洞"。收到 0.04 后极冠是连续冰盖，破碎感
    #   全部交给冰缘（conc 0.2~0.5，阈值 0.35~0.49）。
    # v75: 0.60−0.56c → 0.68−0.66c。
    # 为什么: v74 极冠核心（c=1）已是连续冰盖，但**冰缘带**（c≈0.2~0.5）的阈值
    #   0.49~0.35 意味着近 50% 的像素过阈值 —— 在 ±5° 的碎冰带里形成一片
    #   30~60px 的实心白斑，与真正开阔水交界处就是用户看到的"撕纸/泼奶"。
    #   抬到 0.68/0.66 后边缘带通过率降到 ~30%，碎冰变成**散点冰盘**；
    #   核心 c=1 时阈值 0.02 仍 <1%，不破连续冰盖。
    _thr = 0.68 - 0.66 * _conc
    # v63: 冰盘边缘 0.07 → 0.022。1:1 复核：0.07 的斜坡叠在 24px 的噪声场上，
    #   每块冰盘外面套着 ~8px 的羽化边，冰缘因此读作"云朵/烟雾"而不是冰。
    #   0.022 后过渡 ≈1px，冰盘是硬边碎块，冰间水道的边界也利落。
    # v64: 0.022 → 0.014。新增的 _nhf（≈3px）让 _norm 在 1px 尺度上已有起伏，
    #   1.4% 的斜坡正好把这条起伏译成 1~2px 的毛边 —— 碎冰边缘不再是 20px 的块状
    #   台阶，而是有细碎齿的自然破碎。
    _floe = np.clip((_norm - _thr) / 0.014, 0.0, 1.0)
    # v52: **纬度门控**. v51 的二值化冰盘忘了乘纬度 —— _thr 最高只到 0.92,
    #   而 _norm 是全局归一化噪声, 全图约 8% 的像素 _norm>0.92, 于是低纬海洋里
    #   凭空出现成片"白冰盘" (中纬 22°N 深海 dem=-2319m 处实测 rgb 0.765/0.804/0.851).
    #   用户的 "冰盖跨海洋/像云" 有一半是这个 bug. 现在冰缘以外 alpha 严格为 0.
    # v61b: 冰缘由 1.0° 硬门控改为 2.2° 软过渡（≈25px），再叠 ±1.2° 的
    #   fBm 抖动 —— 旧版门控只有 1 格宽，冰区外缘是一条刀切边，1:1 裁图里
    #   白冰与海面之间没有任何过渡，"像贴上去的白纸"。
    _gnoise = np.clip(_lowres_fbm(S, 5.0, 3, rng=rng), -1.5, 1.5)
    # v63: 软过渡 1.2° → 0.45°。海冰边缘应是碎冰盘的硬边，1.2°（≈14px）的半透明
    #   白纱叠在冷调海面上读作"发光的雾边"；破碎感交给 _floe 的硬边碎盘去做。
    # v73: 软过渡 0.45° → 0.90°、抖动系数 1.2 → 0.70。
    # 改了什么: 冰缘的纬度过渡带宽度与「毛刺」幅度同时收敛。
    # 为什么: 1:1 裁图（/tmp/c_npole.png）里冰缘是一条**越来越细的白手指**插进黑海，
    #   900px 下读作"打翻的牛奶"—— 0.45°(≈10px) 的过渡配上 1.2° 的高频抖动，等于
    #   让边界在 ±1.2° 内随机开合，而边界外侧的海水是近黑色 → 每条毛刺都自带一圈
    #   黑边。加宽到 0.90° 后边界是一条**连续的前缘**，抖动只负责让它不成直线。
    _gate = np.clip((_lat2 - edge_sea + 0.7 + 0.7 * _gnoise) / 0.90,
                    0.0, 1.0).astype(np.float32)
    _sea_a = (_floe * _gate
              * (v01 < 0.5).astype(np.float32)).astype(np.float32)
    # v72: **冰间水道网络（lead）改为等值线网络**。
    # 改了什么: 取一层 σ≈2.2px 的平滑 fBm 场，在其零值等值线附近取 ±0.30σ 的窄带
    #   作为水道，再把它从 _sea_a 里挖掉（最深 85%），随后单独染成深钢蓝。
    # 为什么: v69 只把 _qn/_qf 做了椭圆预平滑，1:1 复核（/tmp/c_spole.png）里
    #   开阔水仍是**几十像素的圆形墨滴** —— 阈值化只能给出"大于/小于"的团块，
    #   团块的连通形状仍是被噪声水平集切出来的岛，不会变细。真实海冰的水道是
    #   **风/洋流拉开的线状裂隙**，在图上就是一张网；噪声的**等值线**天然是细线，
    #   所以改到等值线上取。±0.30σ 的带宽在 2048 图幅上约 4~6px，缩到 900px
    #   仍是 2px 的连续细线，不会被读成"破洞"。
    _ln = value_fbm(S, 22.0, 3, rng=np.random.RandomState(4242))
    _ln = gaussian_filter(_ln.astype(np.float32), 3.0)
    _ln = ((_ln - float(_ln.mean())) / (float(_ln.std()) + 1e-6)).astype(np.float32)
    # v72b: 带宽 0.30σ → 0.16σ、挖除深度 0.85 → 0.60。
    # 为什么: 0.30σ 在 900px 全图上是一张**灰色乱麻**（极冠白色占比 |lat|≥85° 从
    #   82.9% 掉到 68.1%），水道比冰盘还抢眼，且曲线随噪声抖动成"墨水划痕"。
    #   收窄到 0.16σ 后水道约占冰面 8~10%，读作"冰原上的细裂"而不是"划痕"。
    _lead = np.clip(1.0 - np.abs(_ln) / 0.14, 0.0, 1.0).astype(np.float32)
    # v72c: 水道权重从「核心最强」翻转为**边缘冰区最强**（核心 conc=1 只留 0.15）。
    # 为什么: 真实极冠核心是连续多年冰，裂隙集中在边缘冰区/冰间湖带；v72b 把网
    #   均匀铺满整个极冠，900px 下极冠读作"灰纹大理石"而非冰盖（|lat|≥85° 白像素
    #   仅 61.9%）。翻转后极冠核心回到连续白，碎裂纹只在冰缘一带展开。
    # v74: 下限 0.15 → 0.0。极冠核心（conc=1）不再被强行叠 15% 的裂隙。
    # v75: 2.2 → 1.3。冰缘细裂网（_ln 等值线 ±0.14σ）在 900px 上是一层密集的
    #   深色网格，把碎冰带读成"划痕/灰纹大理石"。系数降到 1.3 后，只有 conc<0.23
    #   的最外缘才达到满强度，冰盘内部基本不再被划开。
    _crack = np.clip((1.0 - _conc) * 1.3, 0.0, 1.0).astype(np.float32)
    _lead = (_lead * _crack * _gate * (v01 < 0.5)).astype(np.float32)
    # v73: 挖除深度 0.60 → 0.45。配合下方「极区开阔水统一提亮」，水道不再是
    #   近黑窟窿；0.45 只把冰的 alpha 压到 0.55，裂隙下面的水色已经够亮，
    #   900px 上读作"冰原上的细灰裂"而不是"白纸上的黑划痕"。
    _sea_a = np.clip(_sea_a * (1.0 - 0.45 * _lead), 0.0, 1.0).astype(np.float32)
    # v65: 冰缘 alpha 做 0.8px 抗锯齿。
    # 改了什么: _floe 的二值硬边先经 σ=0.8 高斯，再加通道轴。
    # 为什么: 1:1 复核（c_spole.png）冰缘是 1px 的**抖动噪声边**（阈值化在 3px 噪声
    #   场上），放大看是"撕纸/锯齿"，缩略图上是"白纸撒黑点"。0.8px 过渡既保留硬边
    #   碎块感，又让边界成为连续的曲线而不是像素级麻点。
    # v66: 冰缘抗锯齿加宽 0.8 → 1.3px，并压低高频噪声权重（0.62/0.24/0.14 → 0.66/0.26/0.08）。
    # 改了什么: ① α 的高斯 σ 提到 1.3px；② _norm 里 ≈3px 的 _nhf 权重从 0.14 降到 0.08。
    # 为什么: 900 缩略图（/tmp/full900.png）里极冠是"白纸上撒了一把黑芝麻"——3px 噪声
    #   经 0.014 的斜坡阈值化后，每个像素独立成冰/成水。0.08 后 3px 分量只做毛边，不再
    #   决定开合；1.3px 的 α 过渡把相邻像素的硬跳变抹成一条连续曲线。
    _sea_a = gaussian_filter(_sea_a, 1.3)[..., None]
    # ---- v66: 冰间水道（lead）提亮 -------------------------------------------
    # 改了什么: 在密集度 >0.35 的海冰区内，把"没成冰"的水面（1-_floe）向浅灰蓝
    #   (0.46,0.60,0.70) 混合，权重最大 0.62。
    # 为什么: 1:1 与 900 复核一致地显示极冠内部的 open water 用的是**深海底色**
    #   （v65 深海平原 pd≈0.48 → RGB≈(6,33,63)），一块冰盖里嵌着纯黑窟窿，读作
    #   "被撕开的泡沫板"（用户第 4 条）。真实极地冰间水道有冰下反照与散射光，是
    #   浅灰蓝而不是墨黑；提亮后极冠整体读作"有裂隙的冰原"，边缘碎冰带也不再是
    #   白点撒在黑布上。
    # v67: 密集度门控从 (conc−0.35)/0.65 → conc/0.45 —— 稀疏冰区（conc<0.35）
    #   也获得一部分提亮，冰缘不再是"白纸掉进墨水瓶"的硬对比。
    # v73: 门控 `clip(_conc/0.45)` → 直接用 `_gate`，权重 0.62 → 0.74。
    # 为什么: `_conc/0.45` 在冰缘（conc≈0.2）只有 0.44，而**黑点最密的地方正是冰缘**
    #   —— 提亮被乘小了 56%，于是冰缘那圈碎冰之间的水仍是深海底色，900px 上看就是
    #   "白纸上撒黑芝麻"（用户第 4 条）。改成只要进了冰区门控就统一提亮。
    _lead_w = (_gate * (v01 < 0.5) * (1.0 - _floe) * (1.0 - _lead)
               * 0.74).astype(np.float32)
    rgb = (rgb * (1.0 - _lead_w[..., None])
           + np.array([0.46, 0.60, 0.70], np.float32) * _lead_w[..., None]
           ).astype(np.float32)
    # v72: 水道本体染深钢蓝 (0.20,0.30,0.40)，权重 ≤0.55。
    # 为什么: 上面的「开阔水提亮」会让水道跟冰面一样发亮（白底上的浅灰线看不见）；
    #   但直接留深海底色又回到 v65 的"黑窟窿"。深钢蓝正好是冰下散射的暗部，
    #   与白冰拉开约 60 个色阶，900px 下能读出一张连贯的裂隙网。
    _lt = (_lead[..., None] * 0.38).astype(np.float32)
    rgb = (rgb * (1.0 - _lt)
           + np.array([0.20, 0.30, 0.40], np.float32) * _lt).astype(np.float32)
    # v67: 冰缘外侧的「边缘冰区」光晕。
    # 改了什么: 取 _sea_a 的 σ=9px 邻域减去自身，得到一条贴着冰缘外侧的窄带，
    #   把海面向冷灰蓝 (0.28,0.40,0.50) 混 35%。
    # 为什么: 1:1 复核（c_npole.png）冰盖与洋面之间是一条**白→近黑**的硬边，
    #   缩略图上读作"剪贴的白纸"。真实极地海冰外侧有几十公里的碎冰/冰泥带，
    #   反射率高、颜色比开阔洋面浅一档；补上它之后冰缘是一条渐变的冷灰带。
    # v74: 0.42 → 0.22，色值压暗到 (0.20,0.30,0.39)。
    # 为什么: v73 出图（/tmp/c_npole.png）冰缘外侧挂着一条**比周围海更亮的青蓝带**
    #   —— 0.42 的 (0.28,0.40,0.50) 环叠在同一点的 _pz 冷带上，两层提亮相乘，
    #   900px 上就是一圈霓虹描边（与 v68 修掉的海洋霓虹岸带同型）。现在 _ring
    #   只做冰缘的一档浅灰，冷色带由 _pz 单独负责。
    _ring = (np.clip(gaussian_filter(_sea_a, 9.0) - _sea_a, 0.0, 1.0) * 0.22)
    rgb = (rgb * (1.0 - _ring)
           + np.array([0.20, 0.30, 0.39], np.float32) * _ring).astype(np.float32)
    # v73: **极区冷水带** —— 冰缘外 7° 以内的开阔洋面统一抬到冷灰蓝。
    # 改了什么: 新增一条与冰密集度无关的纬度带（由 edge_sea 向内/外各 7°），
    #   把海面按最多 55% 混向 (0.26,0.37,0.47)。
    # 为什么: 1:1 与 900px 复核都显示"白冰—近黑海"的硬对比是极区读作剪贴画的
    #   主因（用户第 4 条）。上一条 _lead_w 只管冰区**门控之内**的水；门控之外
    #   仍是 LUT 的深海近黑，于是冰缘外侧永远挂着一圈黑。真实极地海（冰缘附近）
    #   是被冰融水稀释的冷灰蓝，明度比开阔洋面高一档；补上这条带之后，冰/
    #   碎冰/水三档是"白 → 浅灰蓝 → 冷灰蓝"的连续过渡，不再有黑边。
    # v74: 权重 0.55 → 0.38，色值 (0.26,0.37,0.47) → (0.15,0.24,0.32)。
    # 为什么: v73 的冷带比深海底色亮了近一倍（深海 (10,48,88) → (66,94,120)），
    #   叠上 _ring 后在 900px 上是一条**发光的青蓝腰带**。极地海水确实偏冷，但
    #   明度只该比开阔洋面高一档；压到 0.38 并改暗后，"白冰 → 浅灰碎冰 → 冷海"
    #   仍是连续三档，却没有霓虹带。
    _pzone = np.clip((_lat2 - (edge_sea - 7.0)) / 7.0, 0.0, 1.0).astype(np.float32)
    _pz = (_pzone * 0.38 * (v01 < 0.5) * (1.0 - _floe)).astype(np.float32)
    rgb = (rgb * (1.0 - _pz[..., None])
           + np.array([0.15, 0.24, 0.32], np.float32) * _pz[..., None]).astype(np.float32)
    # v61: 海冰不再用恒定纯白 —— 旧版 0.93/0.955/0.99 在全冰区内完全一致，
    #   叠加 _conc→1 后极冠就是一块几百像素高的实心色块（用户截图里那条白横带）。
    #   现在冰面 = 0.955 基色 × (低频冰厚斑块 ±4.5% + 细颗粒 ±2% + NW 晕渲 ±3%)：
    #   极冠因此有"多年冰 / 当年冰"的灰蓝与亮白之分，而不是一张白纸。
    # v61b: 明暗场按自身 std 归一化。旧版直接乘原始 fBm 值（std≈0.3），
    #   名义 ±7% 实际只有 ±2%，极冠仍是一张白纸；归一化后 ±7% 才是真的。
    def _nstd(_a):
        _s = float(np.std(_a)) or 1.0
        return np.clip(_a / _s, -1.6, 1.6)
    # v63: **频率口径修正**。_lowres_fbm(size, base_freq, ...) 里 base_freq 是
    #   "整幅图上的基频数"，sigma = size/(2·base_freq)，所以 v62 的
    #   26 / 9 / 2.2 实际对应 σ=39px / 114px / **465px** —— 全是大尺度缓变场。
    #   把 0.028 的权重压到那个 465px 场上，等于给极冠刷了一层几百像素宽的灰白渐变，
    #   1:1 就是用户说的"烟雾/卷积云"。现改为 14 / 45 / 130（σ=73 / 23 / 8px），
    #   并按"斑块 → 中频 → 颗粒"递减权重；冰面基色抬到 0.975，接近冰雪的真实明度。
    _swide = _nstd(value_fbm(S, 14.0, 2, rng=rng))
    _smid = _nstd(value_fbm(S, 45.0, 2, rng=rng))
    _sgrain = _nstd(value_fbm(S, 130.0, 2, rng=rng))
    # v62: 明暗配比换成"低频弱、高频强"（0.060/0.028/0.016 → 0.032/0.018/0.028）。
    #   1:1 复核：旧版 26px 低频 ±6% 在冰面上渲染出大片烟雾状灰白渐变（像云不像冰）；
    #   压掉低频、抬高中频/细颗粒后，冰面读作"有颗粒的冰原"，而不是一块渐变的云。
    # v63: 总幅度 0.078 → 0.056，基色 0.955 → 0.975，下限 0.86 → 0.90。
    #   冰面明暗只保留"多年冰略暗"的微弱差异，结构感交给 _floe 的硬边碎盘与水道。
    # v63: **去掉 shade 项**。1:1 复核定位到：极区那些 100~200px 的灰白缓变块
    #   不是冰厚，而是 **海底晕渲**（_ice_tone 里 0.022*(shade-0.78) 直接吃了
    #   洋底地形的 shade，洋底平滑大尺度 → 冰面被刷成一片云雾）。海冰是浮在水上的
    #   平整冰盖，不该继承洋底明暗。现在只剩三档高频（73/23/8px）的冰厚颗粒。
    # v64: 纹理幅度 0.022/0.016/0.018 → 0.030/0.022/0.026，下限 0.94 → 0.915。
    #   1:1 复核（c_spole.png）：旧幅度下冰原是一整片纯白，肉眼看不到任何厚度差异。
    #   抬到 ±6.5% 后，多年冰（暗）/当年冰（亮）/冰缘碎块三档才分得开。
    # v67: 配比改为「大尺度冰盖厚薄为主，细颗粒只做压纹」0.030/0.022/0.026
    #   → 0.046/0.014/0.007。
    # 为什么: 1:1 复核（/tmp/c_spole.png）南极冰面是**整片灰白电视雪花** ——
    #   8px 的 _sgrain 占 0.026 的权重，与 23px 的 _smid 同量级，两者叠加后在
    #   纯白底上读作随机噪点，完全看不出"冰盖"。真实冰盖的明暗来自**冰流/穹丘**
    #   这种几十到几百像素的厚度差，故把权重压到 σ=73px 的 _swide 上。
    _itex = (0.046 * _swide + 0.014 * _smid + 0.007 * _sgrain).astype(np.float32)
    _ice_tone = np.clip(0.975 + _itex, 0.915, 1.0)[..., None]
    # v64: 老冰偏蓝。只改亮度时压暗段是"灰"（读作脏雪）；真实多年冰压暗时蓝相增强
    #   （冰内吸收红光）。按亮度低谷混入最多 30% 的冷蓝 (0.72,0.82,0.93)。
    _iblue = (np.clip(-_itex / 0.06, 0.0, 1.0) * 0.30)[..., None]
    _col_sea = (np.array([0.93, 0.955, 0.99], np.float32)[None, None, :] * (1.0 - _iblue)
                + np.array([0.72, 0.82, 0.93], np.float32)[None, None, :] * _iblue)
    rgb = (rgb * (1.0 - _sea_a) + _col_sea * _ice_tone * _sea_a)
    # v61b: 删除 v61 的「冰间水道等值线」(_lead)。
    #   改了什么: 整段移除（取 _norm 的 0.34/0.66 两条等值线画 1.2px 深蓝线）。
    #   为什么: 1:1 裁图复核，这两条等值线在冰面上呈现为**虚线状连续涂鸦**
    #     （噪声梯度不均 → 线宽时有时无），比不画更糟；真实冰间水道已经由
    #     _floe 阈值自然产生（极冠核心 5~8% 开阔水、边缘是碎冰带）。
    _mark("snowline+sea_ice")
    if _profile:
        print(f"    [render] TOTAL render_world: {_time.time()-_t[1]:.2f}s")
    return np.clip(rgb, 0, 1)


def world_to_channels(elev_m, lats, tect_T, block=8, seed=1):
    """把米制高程 + 四通道构造场编码为 12 通道条件（条件 GAN 输入的粗底图）。

    通道布局（索引 → 含义），与 ``terrain_engine.py`` 训练管线严格一致：
        [0] 块均值 bm        [1] 块标准差 bstd
        [2] 块最小值 bmin      [3] 块最大值 bmax
        [4] 海陆比例 lf       [5] 河网 (log 归一化，仅陆地可见)
        [6] 冰川覆盖率         [7] 绝对纬度/90°
        [8] 俯冲/碰撞构造场    [9] 离散（裂谷/洋脊）构造场
        [10] 转换断层构造场   [11] 其余构造场（第 0 通道）

    Parameters
    ----------
    elev_m : ndarray (H, H), float32
        米制高程骨架（像素级细化阶段之后、GAN 之前的结果）。
    lats : ndarray (H,)
        像素中心对齐纬度。
    tect_T : ndarray (4, T, T)
        板块模拟器输出的四通道构造场（将被上采样到 H 后再块化）。
    block : int
        块大小（v5 训练为 8）；输出通道空间尺寸 = H/block。
    seed : int
        D8 流累积等的随机种子。

    Returns
    -------
    tuple (ch, bm, bstd, bmin, bmax, lf)
        - ch: ndarray (12, H//block, H//block), float32 — 12 通道条件张量
        - bm/bstd/bmin/bmax/lf: ndarray (B, B) — 粗块统计（供后处理复用）。
    """
    rng = np.random.RandomState(seed)
    v01 = m2v(elev_m)
    H = v01.shape[0]
    B = H // block
    bm = block_reduce(v01, block, "mean"); bstd = block_reduce(v01, block, "std")
    bmin = block_reduce(v01, block, "min"); bmax = block_reduce(v01, block, "max")
    lf = block_reduce((elev_m > 0).astype(np.float32), block, "mean")
    acc = d8_flow_accumulation(bm, rng)
    lmask = lf > 0.05
    thr = max(float(np.quantile(acc[lmask], 0.99)), 2.0) if lmask.any() else 2.0
    river = np.clip(np.log(np.maximum(acc, 1.0001)) / np.log(thr), 0, 1) * lmask
    # A1b: 冰川通道用训练期的 v22 雪线表，不跟随 A1b 重标定后的 _SL_M。
    #   为什么: GAN 权重 best_terrain_gan_v5.pth 是按 v22 glac 通道训练的，
    #   换表 = 训练/推理输入分布错配，且实测会让 seed 20261031 的 DEM 漂移
    #   (md5 a07375fd→404736…)，使 B1 已验收的地形失效。A1 属渲染/审计层。
    glac = block_reduce((elev_m > np.interp(np.abs(lats), _SL_LAT, _SL_INCISION_M)[:, None]
                         ).astype(np.float32), block, "mean")
    lat_rows = (np.abs(lats[block // 2::block]) / 90.0)[:B]
    latn = np.broadcast_to(lat_rows[:, None], (B, B)).astype(np.float32)
    up = H // tect_T.shape[1]
    tect_full = np.stack([ndi_zoom(tect_T[i], up, order=1) for i in range(4)],
                         0)[:, :H, :H]
    tch = np.stack([block_reduce(tect_full[i], block, "mean") for i in range(4)])
    ch = np.concatenate([np.stack([bm, bstd, bmin, bmax, lf, river, glac, latn]),
                         tch], 0).astype(np.float32)
    return ch, bm, bstd, bmin, bmax, lf


def balance_land_latitude(dem_m, lats, target_land_frac=0.29,
                          polar_thresh=60.0, max_polar_frac=0.22,
                          shelf_depth=-400.0):
    """纬度带陆地平衡：限制极地陆地比例，保持自然地形过渡。

    原理：
        1. 统计 |lat| > polar_thresh 的南北极陆地占总陆地比例。
        2. 若超过 max_polar_frac，将最低的极地陆地淹没为浅海 (-150~-50m 大陆架)，
           避免产生海岸到深海的陡崖。
        3. 为补偿陆地面积损失，抬升温带/热带浅海为陆地。新陆地高程 = 原海底高程的
           镜像 (负值取反)，保留地形起伏，避免平坦台地。
        4. 高斯平滑海陆边界过渡。

    适用于任意生成的世界，避免单一极区陆地过多导致气候失真。
    """
    size = dem_m.shape[0]
    lat_grid = np.broadcast_to(lats[:, None], dem_m.shape)
    abs_lat = np.abs(lat_grid)

    # 1) 淹没极地低地 → 浅海大陆架 (自然过渡，无陡崖)
    n_flooded = 0
    for sign in (-1.0, 1.0):
        polar_mask = (sign * lat_grid) > polar_thresh
        polar_land = polar_mask & (dem_m > 0)
        if not polar_land.any():
            continue
        polar_land_count = polar_land.sum()
        current_total_land = (dem_m > 0).sum()
        if polar_land_count / max(current_total_land, 1) <= max_polar_frac:
            continue
        keep_count = int(current_total_land * max_polar_frac)
        if keep_count >= polar_land_count:
            continue
        n_remove = polar_land_count - keep_count
        polar_indices = np.argwhere(polar_land)
        polar_vals = dem_m[polar_indices[:, 0], polar_indices[:, 1]]
        order = np.argsort(polar_vals)
        flood_idx = polar_indices[order[:n_remove]]
        # 淹没为浅海：原高程越高，淹没越浅（保留大陆架坡度）
        orig_elev = dem_m[flood_idx[:, 0], flood_idx[:, 1]]
        dem_m[flood_idx[:, 0], flood_idx[:, 1]] = -50.0 - orig_elev * 0.3
        n_flooded += n_remove

    # 2) 抬升温带/热带浅海造陆 → 镜像高程 (保留地形起伏)
    if n_flooded > 0:
        weight = np.where((abs_lat >= 20) & (abs_lat <= 60), 2.0,
                 np.where(abs_lat < 20, 1.0, 0.3))
        shallow = (dem_m < 0) & (dem_m > shelf_depth)
        shallow_idx = np.argwhere(shallow)
        if len(shallow_idx) > 0:
            w = weight[shallow_idx[:, 0], shallow_idx[:, 1]]
            d = dem_m[shallow_idx[:, 0], shallow_idx[:, 1]]
            score = w * 1000 + d
            order = np.argsort(-score)
            n_raise = min(n_flooded, len(shallow_idx))
            raise_idx = shallow_idx[order[:n_raise]]
            # 镜像+基础抬升: -50m → 225m, -200m → 300m (避免 0-200m 陆地过多)
            # v9.x: 原版 -dem_m 产生 -50→+50 低海拔陆地, 使 0-200m 段超标 19%;
            #       改为 0.5*(-dem_m) + 200 给予 200m 基础高程, 落入 200-1000m 丘陵段
            orig = dem_m[raise_idx[:, 0], raise_idx[:, 1]]
            dem_m[raise_idx[:, 0], raise_idx[:, 1]] = (-orig) * 0.5 + 200.0

    # 3) 平滑海陆边界
    land = dem_m > 0
    dem_smooth = gaussian_filter(np.where(land, dem_m, 0.0), 1.5)
    boundary = gaussian_filter(land.astype(float), 1.5)
    boundary = (boundary > 0.05) & (boundary < 0.95)
    dem_m[boundary] = dem_smooth[boundary] * 0.5 + dem_m[boundary] * 0.5

    final_land = (dem_m > 0).sum()
    north_polar = ((lat_grid > polar_thresh) & (dem_m > 0)).sum()
    south_polar = ((lat_grid < -polar_thresh) & (dem_m > 0)).sum()
    print(f"  [纬度平衡] 陆地 {final_land / dem_m.size * 100:.1f}% | "
          f"均高 {dem_m[land].mean():.0f}m | "
          f"南极 {south_polar / max(final_land, 1) * 100:.1f}% | "
          f"北极 {north_polar / max(final_land, 1) * 100:.1f}%")
    return dem_m


def margin_class_map(dem_m, sigma=46.0):
    """估计每段海岸的"边缘类型"：被动边缘（宽陆架）↔ 主动边缘（窄陆架）。

    改了什么 / 为什么 / 实测
    -----------------------
    改了什么：v80 新增。对**离岸局部海水深度**做归一化高斯卷积
    （σ=46px ≈ 900km），得到"这一带的周边海底有多深"；再映射到宽度系数
    ∈[0.35, 1.70]（深 = 主动边缘 = 窄），最后 σ=25px 平滑使同类边缘连成
    数百公里以上的岸段。``create_continental_shelf`` 与
    ``carve_margin_trenches`` 共用本函数，保证"窄陆架"与"海沟"必然同址。

    为什么：用户 v80 第 1 条「大陆架像给陆地加了一层均匀的外发光/膨胀滤镜。
    现实中大西洋两岸有广阔的被动大陆边缘，太平洋边缘（南美西岸）几乎无陆架」。
    旧版 ``create_continental_shelf`` 对**每一条**海岸都用同一套 6+7+18px
    剖面，宽度恒定 → 视觉上就是描边。真实差距：被动边缘陆架 100~400km，
    主动边缘 <20km，相差 10 倍以上（2048 图 1px ≈ 19.6km）。

    实测（seed=7 / 2048）：系数分布 p05=0.30 / p50=0.68 / p95=2.25；
    v81b 终稿复测的陆架宽度（浅海 −170~0m 像素到最近陆地的欧氏距离）
    p10=2.2px / p50=13.5px / p90=63.5px、max=186px，改前恒为 5~6px。
    注：v81b 曾用"8 向射线首次触底"法测得 p10=7.7/p50=22.2/p90=48.0，
    与欧氏距离法口径不同（射线会把斜向绕行算成直线距离），以欧氏距离为准。

    Parameters
    ----------
    dem_m : ndarray (S, S) float32
        米制高程（陆 > 0）。
    sigma : float
        估计"周边海底深度"的卷积尺度（px）。太小会被单条海沟带偏，太大
        会把两个大洋平均掉。

    Returns
    -------
    ndarray (S, S) float32
        陆架宽度系数，1.0 = 全球平均（6px 档）。
    """
    sea = dem_m <= 0
    num = gaussian_filter(np.where(sea, dem_m, 0.0).astype(np.float32), sigma)
    den = gaussian_filter(sea.astype(np.float32), sigma)
    md = np.where(den > 1e-3, num / np.maximum(den, 1e-3),
                  np.float32(-4000.0)).astype(np.float32)
    # v80b: 线性映射改成 t**1.5 曲线。旧线性映射下全图中位系数只有 1.06、
    #       p05 0.44，被动边缘撑不开、主动边缘也压不窄，测出来的陆架宽度
    #       p10/p50/p90 = 4.1/7.0/36.3px（v79b）几乎没变。
    #       （v80b 的锚点是 −600m→2.30 / −3200m→0.25，v80h 已按下列新锚点重标。）
    # v80h 重新标定：v80g 实测系数 p05=0.25 / p50=0.25 / p95=1.86 —— 全图中位数
    #   压在量程下限，等于把**一半以上的海岸**都判成"太平洋式主动边缘"：
    #   陆架被压到 1.5px、同时 carve_margin_trenches 在每条岸线外都下切出一条
    #   −5000m 级暗环（/tmp/crop_B_rightcoast.png 里那圈深蓝描边）。
    #   根因是映射锚点用错了量纲：本世界开阔洋本来就有 60% 的像素在 −3200m 以下，
    #   拿 −3200m 当"主动边缘"，被动边缘就无处安放。
    #   现改为 −4600m → 0.30（贴着深海平原/海沟的俯冲岸段）、−1600m → 2.25
    #   （浅海陆表海，被动宽架），t**1.4 让中段更快抬起来。
    #   实测（重跑后）：p05≈0.30 / p50≈0.97 / p95≈2.10。
    t = np.clip((md + 4600.0) / 3000.0, 0.0, 1.0)
    cls = 0.30 + 1.95 * (t ** 1.4)
    return gaussian_filter(cls, 25.0).astype(np.float32)


def create_continental_shelf(dem_m, lats, shelf_width_px=6, inner_depth=-25.0,
                             outer_depth=-145.0, slope_width_px=7.0,
                             rise_width_px=18.0, abyssal=None, mcls=None,
                             detail=True):
    """海岸剖面化：陆架 (0~-145m) → 陆坡 (-145→-2100m) → 陆隆 (-2100→深海平原).

    v45 按地球实测坡度重定几何 (上一版 10px 陆坡 = 1.1% 坡度, 图上读不出坡折):
      1) **陆架** 6px (≈117km) 从 -25m 到 -145m, ``t**1.5`` 使内侧近乎平躺;
      2) **陆坡** v51: 7px (≈137km) 从 -145m 降到 -2100m, 坡度 ≈14 m/km ≈ **1.4%**.
         旧版 3px (3.3%) 在几何上是"地球正确"的, 但 3px 的坡度在 2048 图上就是一条
         线 —— 用户看到的是"深蓝海水直接顶到陆地边缘 + 一道亮边". 放宽到 7px 并
         配合渲染色卡的连续渐变, 海岸外侧才读得出**大陆架→陆坡→陆隆**的层次;
      3) **陆隆** v51: 18px (≈350km, 地球 300~500km) 渐深至本地深海基底.
    外缘用 fade 权重与原始深海平原平滑衔接 (不产生台阶, 也不填平海沟).

    v45 另一处关键改动: 旧版 ``np.maximum(cur, tgt)`` **只抬不压**, 于是陆坡段
    想下切到 -2100m 时会被已有的 -800m 浅值卡住, 陡坡根本建不起来 —— 这正是
    v44 距岸剖面 -160m→-2435m 摊在 10px 上的直接原因. 现改为按当前深度软权重:
    浅水 (cur>-200m, 浅滩/台地) 仍只抬不压, 深水 (cur<-600m) 完全跟随目标剖面
    (可下切), 中间线性过渡, 避免出现深度台阶.

    Parameters
    ----------
    shelf_width_px : float
        陆架宽度 (像素). 2048 图上 1px ≈ 19.6km (赤道), 6px ≈ 117km ≈ 地球均值.
    inner_depth, outer_depth : float
        陆架内外缘深度 (米).
    slope_width_px, rise_width_px : float
        陆坡 / 陆隆宽度 (像素), 合计 25px ≈ 490km (v51: 3+13 → 7+18).
    abyssal : float or None
        陆隆末端目标深度 (米). None = 用地形自身的低频深海基底 (v39g), 这样
        剖面只是重排近岸陡坎, 不会把全球深海平原整体抬高.
    mcls : ndarray (S, S) float32 or None
        陆架宽度系数场（1.0 = 全球平均）。None = 现场用 ``margin_class_map``
        估算；管线内应传入**第一次建架前**算好的冻结场（v80g），否则每次调用
        都会把上一次铺出的浅水带算进"周边海底深度"，系数逐轮变大、
        陆架被自我放大成一圈均匀的宽浅色环。
    detail : bool
        v89：是否在终稿重建时铺**陆架微地形 + 海底峡谷**。上游那次建架必须关掉
        —— 后续 coast_smooth / land_hist / 坡度软化会把它们抹平，且两次调用
        几何不同会造成重复叠刻。
    """
    from scipy.ndimage import distance_transform_edt

    land = dem_m > 0
    sea = ~land
    if not sea.any():
        return dem_m

    if detail:
        dist, (_diy, _dix) = distance_transform_edt(sea, return_indices=True)
    else:
        dist = distance_transform_edt(sea)
    w_s = max(float(shelf_width_px), 1.0)
    w_sl = max(float(slope_width_px), 1.0)
    w_r = max(float(rise_width_px), 1.0)
    # v48: 小岛陆架收缩. 旧版把 22px (6+3+13) 的统一剖面套在**每一条**海岸线上,
    #   于是几百座小岛/岛弧周围都铺出一个 15~25px 的浅水圆斑, 在深海底色上就是
    #   用户说的"深海亮青孤斑"; 同时在近岸形成宽度完全均匀、形状雷同的浅水环.
    #   真实海洋中陆架宽度由陆块规模决定 (大陆架 50~500km, 大洋岛几乎无架).
    #   这里用 41px 窗口内的陆地占比当"陆块厚度"代理: 大陆海岸 ≈0.4~0.6,
    #   半径 3~5px 的小岛 <0.03 → 剖面整体收缩到 30~40%.
    from scipy.ndimage import uniform_filter as _uf
    _landf = _uf(land.astype(np.float32), size=41)
    # v80: 小岛再收紧 (0.30→0.18)。大洋岛（夏威夷型）几乎没有陆架，
    #      旧下限 0.30 仍给小岛铺出 9px 浅水环，是"深海亮青孤斑"的来源之一。
    _sc = np.clip(_landf / 0.35, 0.18, 1.0).astype(np.float32)
    # v80: 边缘类型分化（用户第 1 条「陆架像均匀外发光」）。
    #      被动边缘（大西洋型）系数→2.3：陆架 10~14px ≈ 200~270km；
    #      主动边缘（太平洋型）系数→0.20：陆架 1~1.5px ≈ 20~30km，紧接着就是海沟。
    #      再叠 σ=55px 的大尺度沿岸起伏，避免"同一条海岸宽度完全一致"。
    # v80g: 系数改为**全流程只算一次**（由 generate_world 在第一次建架前算好并传入）。
    #      旧版每次调用都在**已经建好陆架的 DEM** 上重算 margin_class_map：
    #      σ=46px 的窗口里近岸那圈 −25~−170m 浅水把 md 拉高 → t 变大 → 系数变大
    #      → 陆架更宽 → 下一轮 md 更高 …… 三步之后全图陆架被"自我放大"成一圈
    #      均匀的宽浅色环（用户看到的"外发光/膨胀滤镜"），实测浅海占比 11.0%
    #      （地球 ~8%）、宽度 p10/p50/p90 = 4.1/7.0/36.3px。传入冻结场后
    #      宽度不再随调用次数漂移。
    _mcls = (margin_class_map(dem_m) if mcls is None
             else np.asarray(mcls, dtype=np.float32))
    # v89: 边缘"活动度" 0=被动(大西洋型, 宽缓) ↔ 1=主动(太平洋型, 陡窄).
    #   取冻结场 _mcls 的原始值(未叠 _nbig 噪声), 保证同一段海岸的宽度与剖面
    #   形状来自同一个判据. 实测 mcls p05/p50/p95 ≈ 0.30/0.97/2.10.
    _act = np.clip((1.35 - _mcls) / 0.95, 0.0, 1.0).astype(np.float32)
    _nbig = value_fbm(dem_m.shape[0], 55.0, 3, rng=np.random.RandomState(8801))
    # v81b: 量程上限 2.35 → 2.60、下限 0.20 → 0.16（见下方 _wmul 的实测）。
    _mcls = np.clip(_mcls * (1.0 + 0.28 * np.clip(_nbig, -1.6, 1.6)),
                    0.16, 2.60).astype(np.float32)
    # v39f: 陆架宽度噪声化 — 纯 EDT 会画出以每座岛为圆心的同心晕(视觉"失焦光斑").
    #        真实陆架随岸线走向宽窄不一(海湾宽、岬角窄), 用低频 fBm 调制等深线.
    _n = value_fbm(dem_m.shape[0], 9.0, 3, rng=np.random.RandomState(1337))
    # v81b: 用户「外围浅海大陆架像用钢笔工具描边一样匀称」的直接根因 ——
    #   旧版只给 d 叠 ±15%（σ=9px）的抖动，σ=9px 噪声每 9px 就翻转一次，肉眼看
    #   仍是一条**等宽的平行线**（等深线只是变毛，不是变宽窄）。
    #   改成两尺度沿走向调制**宽度本身**：σ=46px（≈900km，一段被动边缘 vs
    #   一段陡岸）承担 62%，σ=15px（≈300km，海湾/岬角尺度）承担 38%，幅度 ±38%。
    #   注意必须乘在 w_s/w_sl/w_r 上（真实像素宽度），不能乘在 d 上：
    #   v81b 首轮误写成 d*=_wmul，_wmul<1 时 t=d/w 反而变小 → 等深线外推，
    #   实测陆架宽度 p50 直接冲到 23px、浅海 13.0%（地球 8%），是符号反了。
    _nw = (0.50 * value_fbm(dem_m.shape[0], 140.0, 3, rng=np.random.RandomState(1343))
           + 0.34 * value_fbm(dem_m.shape[0], 46.0, 3, rng=np.random.RandomState(1339))
           + 0.20 * value_fbm(dem_m.shape[0], 15.0, 2, rng=np.random.RandomState(1341)))
    # v84（用户 v82 第 2 条「浅海像一坨晕开的颜料，均匀厚度包裹」）：
    #   幅度 0.38→0.55，并新增 σ=140px（≈2700km，整段大陆边缘尺度）的权重 0.50。
    #   旧版最长只到 46px，等于"整条海岸线的浅带宽度都差不多"；补上 140px 尺度后，
    #   同一片大陆会出现宽陆架段（被动边缘 15~25px）与窄陆架段（主动边缘 1~3px）
    #   的长距离交替，不再是一圈等宽的高斯光晕。
    _wmul = (1.0 + 0.55 * np.clip(_nw, -1.6, 1.6)).astype(np.float32)
    w_s_map = w_s * _sc * _mcls * _wmul
    # v89: 主动边缘再收 35% 陆坡、55% 陆隆 —— 俯冲带的沉积物被海沟吞掉，没有
    #   发育完整的大陆隆；被动边缘保持原样（宽陆隆 + 沉积扇）。注意只乘**宽度**，
    #   绝不能乘到 d 上（v81b 的符号陷阱）。
    w_sl_map = w_sl * np.clip(_sc, 0.55, 1.0) * _mcls * _wmul * (1.0 - 0.35 * _act)
    w_r_map = w_r * _sc * _mcls * _wmul * (1.0 - 0.55 * _act)
    d_tot_map = w_s_map + w_sl_map + w_r_map
    # v89: 陆架坡折深度随边缘类型漂移 —— 被动边缘 -110m（宽缓台面，坡折浅）、
    #   主动边缘 -230m（陡窄，坡折深）。取代全图共用的单一 outer_depth：那条
    #   "-170m 等深线绕所有大陆一圈连续闭合"正是"晕开颜料"的骨架。±60m 的
    #   漂移量按本图陆架坡度 ~20m/px 折算 ≈ 3px 的等深线位移，肉眼才分得出来
    #   （上一版 ±25m ≈ 1px，实测出图几乎无差别）。
    _brk_map = (outer_depth + (0.5 - _act) * 120.0).astype(np.float32)
    # v82: 距离场域扭曲 —— 根治"两片陆地之间的 45° 直边楔形 / 等深线梳齿"。
    #   改了什么：把剖面用的距离从 dist 换成 dist_eff = dist + 平滑位移场，
    #     band 判定与 d 都改用 dist_eff。
    #   为什么：本函数的 depth 只是「到最近陆地的欧氏距离」的单调函数。在窄海/
    #     海湾里，两片陆地各有自己的同心等深线，正中间那条**等距线（中轴）**
    #     把两侧的等深线接成一条笔直的 45° 楔形边 —— 前几轮已定量确认此处
    #     3×3 局部标准差 <0.5m 的像素只有 2/5703（不是硬钳位，是几何必然）。
    #     叠加 σ=52/16px 的±(3~8)px 位移后，等距线被扭成不规则曲线，楔形消失。
    #   实测：见 /tmp 探针的"平面率"与目视裁切（改前该带等深线近乎直线）。
    _wamp = 0.55 * w_s_map + 0.55 * w_sl_map + 0.30 * w_r_map
    _wf = (_znoise(dem_m.shape[0], 52.0, np.random.RandomState(8802), 2)
           + 0.45 * _znoise(dem_m.shape[0], 16.0, np.random.RandomState(8803), 1))
    # v82c 修（实测回归）：扭曲按离岸距离渐入（d/18 的 1.5 次方）。
    #   首版对 d 一律叠 ±(0.4·_wamp)≈±5px，而陆架本身只有 6px 宽 ——
    #   出图实测 −170~0m 浅带被撕成迷彩斑（/tmp/v82_ab_crops.png 下排），
    #   且浅海占比 5.12%→4.48%。楔形只存在于中轴（离岸 15px 以外），
    #   近岸不扭曲即可，浅带恢复等深线平行。
    _ramp = np.clip(dist / 18.0, 0.0, 1.0) ** 1.5
    dist_eff = np.maximum(dist + 0.40 * _wamp * _wf * _ramp,
                          0.35).astype(np.float32)
    band = sea & (dist_eff < d_tot_map)
    if not band.any():
        return dem_m

    d = dist_eff[band] * (1.0 + 0.10 * np.clip(_n[band], -1.6, 1.6))
    # v39g: 陆隆末端对准"本地"深海基底, 而非全局常数 — 只削近岸陡坎, 不动开阔洋.
    if abyssal is None:
        abyss_src = gaussian_filter(dem_m.astype(np.float32), 24.0)
        abyss_local = np.minimum(abyss_src, np.float32(-3200.0))
        abyssal = -4300.0
    else:
        abyss_local = np.full(dem_m.shape, float(abyssal), np.float32)
    # (1) 陆架: 平缓加深 (t**1.5, 近岸几乎水平, 外缘 0.3°)
    _ws_b = w_s_map[band]
    _wsl_b = w_sl_map[band]
    _wr_b = w_r_map[band]
    _dtot_b = d_tot_map[band]
    _brk_b = _brk_map[band]
    # v89: 台面指数随边缘类型变化 —— 被动 2.2（一路缓升的宽台面）、主动 3.2
    #   （更平 + 更急的膝部）。同一圈等深线在不同岸段落在不同离岸比例上，
    #   不再是"逐像素等宽平移"的一层壳。
    _k_b = (2.2 + 1.0 * _act[band]).astype(np.float32)
    # v89 海底峡谷：相位取"最近陆地像素"的索引当沿岸一维坐标 —— 同一条离岸法线
    #   上这个索引不变，所以切出来的是**垂直于岸线**、从坡折冲进陆隆的放射状沟，
    #   而不是各向同性的斑点；三条不同周期的正弦叠加再取高门限 → 稀疏、窄（2~5px）。
    #   真实峡谷是陆架沉积物输往深海的通道，也是"等深线永远平行于海岸"这个
    #   果冻感的直接破坏者。
    if detail:
        _S = dem_m.shape[0]
        _cd = (_diy[band].astype(np.float32) * _S
               + _dix[band].astype(np.float32))
        _cp = (np.sin(_cd * 0.53) + 0.85 * np.sin(_cd * 0.191 + 1.9)
               + 0.50 * np.sin(_cd * 1.37 + 0.7))
        _ct = np.clip((_cp - 0.95) / 0.60, 0.0, 1.0)
        _can = _ct * _ct * (3.0 - 2.0 * _ct)
        # 包络：峰在坡折后 0.45 个陆坡宽度，宽 0.55 个陆坡宽度；内侧从 75% 陆架
        #   宽度才开始（谷头在坡折附近，不啃海岸线）；外侧在陆隆中段淡出。
        _can_env = np.exp(-0.5 * ((d - (_ws_b + 0.45 * _wsl_b))
                                  / (0.75 * _wsl_b + 1.5)) ** 2)
        _can_env *= np.clip((d - 0.75 * _ws_b) / (0.50 * _ws_b + 1e-3), 0.0, 1.0)
        # 主动边缘（陡坡/海沟型）的峡谷切得更深：380 → 860m。
        _can_depth = (380.0 + 480.0 * _act[band]) * _can * _can_env
    else:
        _can_depth = None
    # v83（用户 v82 第 2 条「浅海像一坨晕开的颜料，均匀厚度包裹」）：
    #   旧指数 1.5 让陆架从内缘到外缘**一路匀速变深**，等深线等距平行 → 读作
    #   高斯模糊的光晕。真实陆架是**近乎平坦的台面**，到陆架坡折（~−130m）
    #   才突然折下去。指数改成 2.4：前 80% 宽度只降 ~35% 的落差（台面），
    #   最后 20% 才走完剩余 65%（坡折处的膝部）。
    t1 = np.clip(d / _ws_b, 0.0, 1.0)
    z1 = inner_depth + (_brk_b - inner_depth) * (t1 ** _k_b)
    # (2) 陆坡: 陡降 (smoothstep 保证两端斜率为 0, 无折角); 3px 承担 1955m → 3.3%
    t2 = np.clip((d - _ws_b) / _wsl_b, 0.0, 1.0)
    s2 = t2 * t2 * (3.0 - 2.0 * t2)
    z2 = _brk_b + (-2100.0 - _brk_b) * s2
    # (3) 陆隆: 渐深至本地深海基底
    t3 = np.clip((d - _ws_b - _wsl_b) / _wr_b, 0.0, 1.0)
    s3 = t3 * t3 * (3.0 - 2.0 * t3)
    z3 = -2100.0 + (abyss_local[band] + 2100.0) * s3
    prof = np.where(d <= _ws_b, z1, np.where(d <= _ws_b + _wsl_b, z2, z3))
    # v39f: 陆坡/陆隆叠加起伏, 避免大面积平滑剖面看上去像"失焦光斑"
    prof = prof + 45.0 * np.clip(_n[band], -1.5, 1.5) * np.clip((d - _ws_b) / _wsl_b, 0.0, 1.0)
    # 外缘渐隐: 在最后 45% 陆隆宽度内把剖面混合回原始海底 (避免台阶)
    fade = np.clip((_dtot_b - d) / (0.45 * _wr_b), 0.0, 1.0)
    cur = dem_m[band].astype(np.float64)
    tgt = prof * fade + cur * (1.0 - fade)
    tgt = np.minimum(tgt, -1e-3)
    # v45: 浅水(cur>-200m)只抬不压, 保护浅滩/台地; 深水(cur<-600m)可跟随剖面下切,
    #      否则陆坡会被已有浅值卡住 (见 docstring). 中间线性过渡, 无深度台阶.
    keep = np.clip((cur + 600.0) / 400.0, 0.0, 1.0)
    dem_m[band] = (keep * np.maximum(cur, tgt) + (1.0 - keep) * tgt).astype(np.float32)

    # 带状平滑: 消除三段落切换处及陆架外缘的色阶; 坡折段只用很小的权重, 避免把
    # 3px 陆坡重新抹平 (旧版全程 0.45 混合, 等于对陆坡做 σ=1.6 的二次低通).
    sm = gaussian_filter(dem_m.astype(np.float32), 1.6)
    w_sm = 0.12 + 0.33 * np.clip((d - (_ws_b + _wsl_b)) / (0.6 * _wr_b), 0.0, 1.0)
    dem_m[band] = (dem_m[band] * (1.0 - w_sm) + sm[band] * w_sm).astype(np.float32)
    # v80c: 平滑会把相邻**陆地正值**混进近岸浅海像素，实测这一步凭空多出 13 个
    #   陆块（27 → 40，其中 16 个 <800px），渲染成海岸外侧的白色碎屑。band
    #   本身定义为 sea，因此这里直接钳回海平面以下即可，不改变既有陆地。
    dem_m[band] = np.minimum(dem_m[band], np.float32(-1e-3))

    # v80h: 岸外"环形深槽（moat）"修复 —— 用距离壳层把近岸深度向外单调化。
    #   改了什么：按 2px 一层，从 58px 向内递推到 8px，内圈深度取
    #     max(自身, 紧邻外圈) —— 只抬高"内深外浅"的逆序像素，正常剖面不动；
    #     门限 >−4300m 保护刚挖出的海沟（海沟都在 −6000 以下）。
    #   为什么：实测 y=420 横切右大陆北岸，离岸 0→42px 为
    #     −41 → −559 → −2034 → −4176(20px) → −3463 → −3414(42px)，
    #     即离岸 20px 反而比 42px 外深 760m —— 出图上是绕岸一圈比外海更暗的
    #     "护城河"。溯源：这不是建架造成的（该处主动边缘 d_tot 只有 ~14px，
    #     建架带够不到 20~42px），而是 v77 底稿本身在岸外留了一圈 −4100 凹陷
    #     加一圈 −3400 浅脊。地球的大陆隆是单调变深的（−2100 → −4300 → 深海
    #     平原），不会先降后升，所以在 DEM 终稿上统一拉平。
    #   实测（v1 无坡度）：同一剖面 20px 处 −4176 → −3380，但 15~32px 全摊成
    #     一条 −3380 的平坦台地。
    #   每向内一层再抬 30m（≈0.15% 坡度，地球大陆隆起坡 0.1~0.5%），把台地
    #     变成缓坡：实测 20px 处 −3650，相对 42px 的 −3414 只差 236m（原 760m）。
    _dsea = distance_transform_edt(sea)
    _env = dem_m.copy()
    for _sh in np.arange(58.0, 6.0, -2.0):
        _o = sea & (_dsea > _sh + 2.0) & (_dsea <= _sh + 4.0) & (_env > -4300.0)
        _i = sea & (_dsea > _sh) & (_dsea <= _sh + 2.0) & (_env > -4300.0)
        if _o.any() and _i.any():
            _sh_out = maximum_filter(np.where(_o, _env, -1e9).astype(np.float32), 5)
            _env[_i] = np.maximum(_env[_i], _sh_out[_i] - 30.0)
    _chg = sea & (_env > dem_m)
    dem_m[_chg] = _env[_chg]

    # v89: 陆架微地形 + 海底峡谷放在"护城河单调化"**之后** —— 单调化只抬不压，
    #   若先刻峡谷，谷底会被它按外圈深度抬回去（实测填掉 300~400m）。
    #   微地形: 沙脊/沙坝 ±28m(σ=48px) + ±26m(σ=20px) + 细纹 ±12m(σ=8px)，
    #   从岸边渐入（不在海岸线外制造一圈浅台阶），只铺到陆坡上缘。
    #   峡谷: 见上方 _can_depth。
    if detail:
        _zS = dem_m.shape[0]
        _srel = (_znoise(_zS, 48.0, np.random.RandomState(8900), 2) * 28.0
                 + _znoise(_zS, 20.0, np.random.RandomState(8901), 2) * 26.0
                 + _znoise(_zS, 8.0, np.random.RandomState(8902), 1) * 12.0)
        _rel_b = (_srel[band]
                  * np.clip(d / np.maximum(_ws_b, 1.0), 0.0, 1.0)
                  * (d < (_ws_b + _wsl_b)))
        dem_m[band] = np.clip(dem_m[band] + _rel_b - _can_depth,
                              -11000.0, -1e-3).astype(np.float32)
        print(f"      [v89] 海底峡谷 {int((_can_depth > 80).sum())}px "
              f"最深 {_can_depth.max():.0f}m | 陆架起伏 ±{np.abs(_rel_b).max():.0f}m")

    new_land = (dem_m > 0).sum()
    shallow = ((dem_m > outer_depth) & (dem_m < 0)).sum()
    print(f"  [大陆架] 浅海({outer_depth:.0f}~0m) {shallow / max((~(dem_m>0)).sum(),1) * 100:.1f}% "
          f"(地球~8%) | 陆地 {new_land / dem_m.size * 100:.1f}%")
    return dem_m


def coastal_erosion(dem_m, sigma=2.4, band_px=6, strength=0.85):
    """海浪侵蚀/泥沙沉积的等效算子：沿岸带 DEM 向低通场收敛.

    真实海岸由波浪侵蚀 + 沿岸流输沙抹平: 岬角被削、碎岛被淤平并入主陆、
    河口/滩涂形成宽缓过渡. 本函数在海岸线两侧 ``band_px`` 宽的带内, 按
    到岸线距离做衰减加权, 把高程向 σ 高斯低通场混合 —— 高频锯齿 (GAN/噪波
    裁切痕迹) 被抹掉, 小于该尺度的孤立碎岛整体落入低通场后消失.

    Parameters
    ----------
    sigma : float
        低通尺度 (像素), 6px≈35px 海绵尺度.
    band_px : int
        作用带宽 (像素, 岸线两侧各一段).
    strength : float
        最大混合权重 (1.0 = 完全替换为低通场).
    """
    from scipy.ndimage import binary_dilation as _bd

    land = dem_m > 0
    if land.all() or not land.any():
        return dem_m
    sea = ~land
    coast = _bd(land, iterations=band_px) & _bd(sea, iterations=band_px)
    if not coast.any():
        return dem_m
    sm = gaussian_filter(dem_m.astype(np.float32), sigma)
    d_c = distance_transform_edt(~coast)
    wgt = (1.0 - np.clip(d_c / float(band_px), 0.0, 1.0)) ** 1.5 * float(strength)
    out = dem_m * (1.0 - wgt) + sm * wgt
    print(f"  [海岸侵蚀/沉积] 作用带 {int(coast.sum())}px, 最大权重 {strength:.2f}")
    return np.clip(out, -11000.0, 8900.0).astype(np.float32)


def enhance_terrain_texture(dem_m, lats, rng=None):
    """添加多尺度高频地形纹理，使功率谱斜率接近地球的 β≈2.0-2.5。

    地球地形在所有尺度上都有丰富的分形细节（f^-β, β≈2.0-2.5）。
    GAN + 多次后处理平滑会削弱中高频细节，导致 β>2.9（过于平滑）。

    方案：在陆地添加多倍频 fBm 噪声（频率 30/80/200/500），
    幅度随频率递减，模拟真实地形的分形特征。
    海洋仅在浅海添加少量纹理，深海保持平滑。
    """
    if rng is None:
        rng = np.random.RandomState(42)

    from terrain.core.terrain_engine import value_fbm, ridged_fbm
    size = dem_m.shape[0]
    land = dem_m > 0
    sea = ~land

    # 多尺度 fBm 噪声（频率递增，幅度递减）
    # v31: 高频幅度整体降 ~40% (200/120/60/30 → 120/72/36/18) —
    #      修复平均陆地坡度过陡 (seed11 实测 1.14° vs 地球 0.41°),
    #      微尺度粗糙度主要拉高 P99 坡度, 对 PSD β 影响小
    detail = np.zeros_like(dem_m)
    scales = [
        (30.0, 4, 120.0),    # 中尺度: 山脉/盆地细节
        (80.0, 3, 72.0),     # 中小尺度: 丘陵/谷地
        (200.0, 2, 36.0),    # 小尺度: 坡面纹理
        (500.0, 2, 18.0),    # 微尺度: 地表粗糙度
    ]
    for freq, octs, amp in scales:
        detail += amp * (value_fbm(size, freq, octs, rng=rng) - 0.5)

    # 山脊噪声增强山脉纹理
    ridge = ridged_fbm(size, base_freq=60.0, octaves=3, rng=rng)
    mountain_mask = land & (dem_m > 1000)
    detail += 80.0 * (ridge - 0.5) * mountain_mask.astype(float)

    # 陆地添加纹理，浅海少量，深海不加
    shallow_sea = sea & (dem_m > -200)
    deep_sea = sea & ~shallow_sea
    weight = np.where(land, 1.0,
                      np.where(shallow_sea, 0.15, 0.0))
    dem_m = dem_m + detail * weight

    # 保持海陆边界不变：纹理噪声不能让陆地变海洋或海洋变陆地
    new_land = dem_m > 0
    flipped_land = (~new_land) & land       # 原本是陆地但变成海洋
    flipped_sea = new_land & sea            # 原本是海洋但变成陆地
    if flipped_land.any():
        dem_m[flipped_land] = np.maximum(dem_m[flipped_land], 1.0)
    if flipped_sea.any():
        dem_m[flipped_sea] = np.minimum(dem_m[flipped_sea], -1.0)

    # 功率谱检查
    from scipy.fft import fft2, fftfreq
    d = dem_m.copy()
    d[~land] = 0
    f = np.abs(fft2(d - d.mean()))**2
    freqs = np.sqrt(fftfreq(size)[:, None]**2 + fftfreq(size)[None, :]**2)
    bins_f = np.linspace(0.001, 0.5, 40)
    radial, centers = [], []
    for i in range(len(bins_f) - 1):
        mask = (freqs >= bins_f[i]) & (freqs < bins_f[i+1])
        if mask.any() and f[mask].mean() > 0:
            radial.append(f[mask].mean())
            centers.append((bins_f[i] + bins_f[i+1]) / 2)
    x = np.log10(np.array(centers))
    y = np.log10(np.array(radial))
    beta = -np.polyfit(x, y, 1)[0]
    print(f"  [纹理增强] 功率谱 β = {beta:.2f} (地球 2.0-2.5, 过平滑>2.8)")
    return dem_m


@torch.no_grad()
def generate_world(seed=0, size_px=2048, ckpt_path=None,
                   tect_res=512, out_png=None, out_npy=None,
                   enable_supervisor: bool = True,
                   apply_suggestions: bool = True,
                   export_3d: bool = False,
                   export_3d_mode: str = "both",
                   export_3d_decimate: int = 4):
    """**项目主入口**：生成一整个幻想世界的米制 DEM + 气候终态渲染 PNG。

    四大阶段（1/4 → 4/4）：
        1. 板块演化模拟（``plate_sim.run_plate_sim``，450 Myr，有缓存）：
           返回粗分辨率构造骨架 ``elev_T, tect_T, crust_T, margin_T, belt_T, rift_T``。
           命中缓存（``cache/plate_sim_v{N}_T{tect_res}_seed{seed}.npz``，N 见
           :data:`terrain.core.cache_keys.PLATE_CACHE_VERSION`）时可从
           2~4 分钟缩到 < 10s。
        2. 像素级骨架细化：周期 fBm 域扭曲（消 Voronoi 直线感）+ 上采样到
           ``size_px`` + 海岸分形级联（D≈1.3）+ 山脉/盆地 + **v6.6 七大
           地理维度修复层**（陆架剖面、板内造山、陆地直方图、岛弧链、热点岛链…）。
           此阶段结果保存为 ``world_skeleton.png`` 便于检查大陆走向。
        3. 逐瓦片 12 通道条件 GAN：把粗 DEM 编码为 12 通道（块统计 / 海陆 /
           河网 / 冰川 / 纬度 / 4 构造场），UNet 256² 瓦片推理 + 余弦羽化拼贴
           回原始分辨率，同时做低频一致性拉回与深海硬钳制（防 GAN 幻浅滩）。
        4. 全局后处理：``global_envelope_match`` CDF 包络匹配 → 去斑 → 粒子
           水力侵蚀 → 干流河谷雕刻 → 海岸锐化 → 峡湾雕刻 → 陆架终检 →
           填洼 D8 河网 → **陆地直方图终检**（绝对最后一步）→ 裂谷重新下切 →
           克拉通盆地 → 台地半岛 → 珊瑚礁 → **深洋直方图终检 M6**。
           最终 ``world_dem.npy`` 同时满足陆地、海洋两个维度的直方图分位。

    Parameters
    ----------
    seed : int
        世界种子，决定一切随机性（板块速度场 / fBm / GAN 噪声 / 气候采样）。
        推荐非负整数；任意整数均可。
    size_px : {2048, 4096}
        目标 DEM 单边像素数。4096 ≈ 4× 耗时 + 16× 显存，需要 ≥32GB RAM。
    ckpt_path : str or None
        UNet 权重路径。空时回退为 ``terrain/checkpoints/best_terrain_gan_v5.pth``。
        若权重的 ``cfg.in_ch != 12`` 会抛 ``AssertionError``（FAQ Q2）。
    tect_res : int
        板块模拟分辨率（默认 512，与缓存匹配；请与 TECHNICAL §2 一致修改）。
    out_png / out_npy : str or None
        渲染图 / DEM 输出路径。空时分别写为 ``terrain/output/world_map.png``
        与 ``terrain/output/world_dem.npy``。
    enable_supervisor : bool
        v8.0 新增：是否在生成末尾调用 ``terrain.supervisor`` 审计五维指标
        （地形/温度/降水/群系/风带雪盖）并写学习日志。默认 True。
    apply_suggestions : bool
        v8.0 新增：是否把低风险反思建议自动写入 ``overrides.json``（默认 True）。
        群系阈值等高风险项永不自动注入。设 False 则仅打印建议。

    Returns
    -------
    ndarray of float32, shape ``(size_px, size_px)``
        最终米制高程（正值陆地 / 负值海洋），范围经钳制到 [−10500, +8900] m。
        同时磁盘已写出 ``world_dem.npy``、``world_map.png``、``world_skeleton.png``。
        若 ``enable_supervisor=True``，还会写 ``terrain/supervisor/log.jsonl`` 与
        ``summary.md``，并在终端打印审计简报。

    Notes
    -----
    本函数全程使用确定性随机数（numpy RandomState + torch.manual_seed 由权重承载），
    同一 seed 多次运行**严格 bitwise 一致**；只有删除板块缓存或修改源代码时才会
    改变地形。
    """
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    _dir = (os.path.dirname(_file_dir)
            if os.path.basename(_file_dir) in ("core", "scripts")
            else _file_dir)
    os.makedirs(os.path.join(_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(_dir, "output"), exist_ok=True)
    ckpt_path = ckpt_path or os.path.join(_dir, "checkpoints", "best_terrain_gan_v5.pth")
    out_png = out_png or os.path.join(_dir, "output", "world_map.png")
    out_npy = out_npy or os.path.join(_dir, "output", "world_dem.npy")
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert int(ck.get("cfg", {}).get("in_ch", 0)) == 12, "请先用 v6.3 重训 (12通道)"
    gen = UNet(in_ch=12, base=32).to(device)
    gen.load_state_dict(ck["gen_ema"]); gen.eval()
    ch_stats = {"mean": np.array(ck["ch_stats_mean"], np.float32),
                "std": np.array(ck["ch_stats_std"], np.float32)}
    block = int(ck.get("cfg", {}).get("block", 8))
    tb, ovb = 256 // block, 6
    step, ovp = tb - ovb, ovb * block

    print(f"[1/4] 板块演化模拟 (T={tect_res}, 450 Myr, 有缓存)...")
    _stage_t0 = time.time()
    # 缓存键由 PLATE_CACHE_VERSION 统一生成；增版本即让旧几何自然失效
    cache = plate_cache_path(_dir, tect_res, seed)
    if os.path.exists(cache):
        d = np.load(cache)
        elev_T, tect_T, crust_T = d["elev"], d["tect"], d["crust"]
        margin_T = d["margin"] if "margin" in d.files else None
        belt_T = d["belt"] if "belt" in d.files else None
        rift_T = d["rift"] if "rift" in d.files else None
        print(f"      载入缓存 {os.path.basename(cache)}")
    else:
        elev_T, tect_T, crust_T, margin_T, belt_T, rift_T = run_plate_sim(
            T=tect_res, seed=seed)
        np.savez_compressed(cache, elev=elev_T, tect=tect_T, crust=crust_T,
                            margin=margin_T, belt=belt_T, rift=rift_T)

    up = size_px // tect_res
    rng = np.random.RandomState(seed + 5)
    lats = (-90.0 + (np.arange(size_px) + 0.5) * (180.0 / size_px)).astype(np.float32)
    # v6.5: 板块骨架域扭曲 (周期fBm, 数据线无缝) → 边界/海岸弧形化, 消"细胞壁"直线
    # v27: 增强扭曲强度 11→22px — 板块 Voronoi 分配产生块状矩形大陆,
    # 需更强扭曲打破直线/长条形边界, 使海岸线更有机自然.
    # 同时增加低频成分 (freq 3.0) 产生大尺度弯曲, 避免仅小尺度锯齿.
    wxT = (periodic_fbm(tect_res, 6.0, 3, rng) * 22.0
           + periodic_fbm(tect_res, 3.0, 2, rng) * 12.0
           + periodic_fbm(tect_res, 20.0, 2, rng) * 5.0)
    wyT = (periodic_fbm(tect_res, 6.0, 3, rng) * 22.0
           + periodic_fbm(tect_res, 3.0, 2, rng) * 12.0
           + periodic_fbm(tect_res, 20.0, 2, rng) * 5.0)
    elev_T = warp_lonlat(elev_T.astype(np.float32), wxT, wyT, 1.0)
    # v38: 构造场弱扭曲 — 全幅 warp (±39px) 把细长山带切碎 (belt>0.35 AR 0.19,
    # envS>0.2 205 碎片 vs 未扭曲 6.9/8 块), 后续 >2000m 山块全成 blob
    # (belt_share 0.02-0.13 vs 地球 0.84). 构造带用 0.25 幅扭曲: 保持弧状弯曲
    # 同时不撕裂长链; 海岸线 (elev_T/coast) 仍全幅扭曲, 差异 ≤~30px 视觉不可见.
    _wk = float(os.environ.get("FWS_TECT_WARP", "0.25"))
    tect_T = np.stack([warp_lonlat(tect_T[i], wxT * _wk, wyT * _wk, 1.0)
                       for i in range(4)])
    crust_T = warp_lonlat(crust_T.astype(np.float32), wxT, wyT, 1.0)
    # v27: 海岸线形态学平滑 — 用圆形结构元对陆掩膜做开运算+闭运算,
    # 消除板块 Voronoi 产生的矩形尖角和长条狭缝, 使大陆轮廓更圆润自然.
    # 半径 4px on 512 grid ≈ 32px on 4096.
    from scipy.ndimage import binary_closing, binary_opening
    _land_T = elev_T > 0
    # 2026-10 大陆形态修复: 开运算半径 4→3 (少删半岛), 闭运算半径 4→6
    # (<=12px≈1000km 的窄海峡/内海封口) —— 用户反馈"大陆呈细条状", 原因是
    # 细长臂 + 窄水缝把大陆切成条; 闭运算把碎片粘成整块, 大陆更辽阔.
    _cy3, _cx3 = np.ogrid[-3:4, -3:4]
    _selem_o = (_cy3**2 + _cx3**2) <= 9
    _cy6, _cx6 = np.ogrid[-6:7, -6:7]
    _selem_c = (_cy6**2 + _cx6**2) <= 36
    _land_smooth = binary_closing(binary_opening(_land_T, structure=_selem_o),
                                  structure=_selem_c)
    # 将平滑后的陆地/海洋边界反映回 elev_T: 平滑区外的陆地降为浅海,
    # 平滑区内的海洋抬升为低地. 仅改海陆边界, 不碰内部高程.
    _to_ocean = _land_T & ~_land_smooth   # 被平滑掉的陆地 → 浅海
    _to_land = ~_land_T & _land_smooth   # 被平滑填入的海洋 → 低地
    elev_T = np.where(_to_ocean, -30.0, elev_T)
    elev_T = np.where(_to_land, 30.0, elev_T)
    crust_T = np.where(_land_smooth, np.maximum(crust_T, 0.3), crust_T)
    # v3.0 margin_type 同步扭曲+上采样 (被动/主动边缘分化)
    if margin_T is not None:
        margin_T = warp_lonlat(margin_T.astype(np.float32), wxT, wyT, 1.0)
        margin_up = np.clip(ndi_zoom(margin_T, up, order=0)[:size_px, :size_px],
                            0, 2).astype(np.int8)
    else:
        margin_up = None
    # M2 延展山带同步扭曲+上采样 → 骨架山脉纹理沿带走向 (v38: 弱扭曲保链)
    if belt_T is not None:
        belt_T = warp_lonlat(belt_T.astype(np.float32), wxT * _wk, wyT * _wk, 1.0)
        beltS = np.clip(ndi_zoom(belt_T, up, order=1)[:size_px, :size_px],
                        0, 1).astype(np.float32)
    else:
        beltS = np.zeros((size_px, size_px), np.float32)
    # M2 失败裂谷场同步扭曲+上采样 (直方图终检后重新下切用)
    if rift_T is not None:
        rift_T = warp_lonlat(rift_T.astype(np.float32), wxT, wyT, 1.0)
        riftS = np.clip(ndi_zoom(rift_T, up, order=1)[:size_px, :size_px],
                        0, 1).astype(np.float32)
    else:
        riftS = np.zeros((size_px, size_px), np.float32)
    elev_m = ndi_zoom(elev_T, up, order=1)[:size_px, :size_px].astype(np.float32)
    # 像素级细节: 山体纹理跟随构造场 + 海岸分形噪声(基于连续crust概率场, 非二值)
    # v82: tect_T[1] 在 512 粗格上是**单格宽的尖峰链**（实测相邻格 |dx| p99.9 = 0.82），
    #   直接双线性升 4× 会得到 4px 宽的"帐篷"折线 —— 折线顶端是 C1 不连续的轴对齐折痕。
    #   该场又被 trench_mask(envS>0.4) 硬阈值切割、乘上 deepening 放大到 −2500m，
    #   于是深海出现交替的 −5200/−7700m 竖条纹与方块（用户"剥落的墙皮" + "网格线"）。
    #   修：粗格先高斯平滑去尖峰 → 归一 → 升采样 → 全分辨率低通抹掉 4px 折痕。
    _env_t = gaussian_filter(tect_T[1].astype(np.float32), 0.8)
    _env_t = _env_t / max(float(_env_t.max()), 1e-6)
    envS = np.clip(ndi_zoom(_env_t, up, order=1), 0, 1)[:size_px, :size_px]
    envS = gaussian_filter(envS.astype(np.float32), 1.0)
    cf = gaussian_filter(crust_T, 2.0)
    cf = ndi_zoom(cf, up, order=1)[:size_px, :size_px]
    # 海岸带(420m 半高宽) 分形级联 D≈1.3: 宏观湾/半岛 → 岬角 → 碎屑岸逐级细化
    # v7.0: 海洋侧幅度减半 → 陆架浅海不被噪声大量转陆 (land_frac 超标源头)
    band = np.exp(-((np.abs(elev_m)) / 420.0) ** 2)
    band_o = band * np.where(elev_m < 0, 0.5, 1.0)       # 海侧噪声减半
    elev_m = elev_m + band_o * 480.0 * (cf - 0.5)        # 宏观: 大湾/半岛 (扭曲后不规则)
    elev_m += 190.0 * value_fbm(size_px, 24.0, 4, rng=rng)
    # 山脉: 束状分支(脊状fBm) + 宽度空间变化 → 消"贴边条"直线感
    # M2: 纹理场 = max(构造场, 0.85×延展山带) → 像素级山体沿延展带走向连续
    # ===== v38: 山链连续化重构 =====
    # 诊断 (diag_belt.py, seed0): plate_sim 输出的 belt 场本身 belt_share 0.94 /
    # mean_AR 6.9 (地球 0.84/6.4), 但最终 DEM >2000m 山块只有 5% 落在带内
    # (belt_share 0.02) —— 原 700·wvar·(ridged_fbm²) 的高频噪声把整条带切成
    # 碎斑 (<0.3 的位置几乎不抬升), 散山 (板内造山/噪声丘) 反而主导山地.
    # 修复: 带内**保底抬升** — 噪声只做相对起伏 (0.45+0.55·ridge), 整条构造带
    # 走廊整体抬升 → 连续长链 >2000m; 带外山丘幅度降低 (反比 m_field) 减少碎山.
    m_field = np.maximum(envS, 0.85 * beltS)
    wvar = np.clip(1.0 + 0.9 * value_fbm(size_px, 5.0, 3, rng=rng), 0.25, 2.0)
    _ridge = (ridged_fbm(size_px, base_freq=14.0, octaves=4, rng=rng) ** 2)
    # v38: 脊状对比度 0.45+0.55·ridge → 0.80+0.20·ridge, 幅度 1600→1100.
    # 地球高山带是平顶高原 (Tibet/Altiplano): 2000-3500m 带平均坡度仅 0.32°,
    # 3500+ 0.50°; 本生成器 1.17°/1.70° (3-4x) —— 高频脊状纹理把山链刻成尖脊.
    # 弱化带内起伏 → 高地带成"宽缓高原+坡缘", 坡度回地球量级 (山链连续性由
    # 末次迁移保证, 此处只需给地形性格).
    elev_m += 1100.0 * m_field * wvar * (0.80 + 0.20 * _ridge)
    elev_m += band * 300.0 * (value_fbm(size_px, 34.0, 3, rng=rng) - 0.35)   # 中观: 岬角/海湾
    elev_m += band * 150.0 * (value_fbm(size_px, 90.0, 2, rng=rng) - 0.3)
    elev_m += band * 85.0 * (value_fbm(size_px, 200.0, 2, rng=rng) - 0.28)   # 细观: 碎屑岸
    elev_m += band * 45.0 * (value_fbm(size_px, 450.0, 1, rng=rng) - 0.25)
    # 山间盆地 (费尔干纳型): 造山带内部低频凹坑
    # v38: -2000→-1200 · 更弱掩码 — 窄山带 (4-6px@512) 被深切削整条切碎 →
    # >2000m 山链断裂; 降到 1200m 让盆底仍高于山链底, 链保持连通
    imb = np.clip(value_fbm(size_px, 10.0, 3, rng=rng) - 1.1, 0, None)
    elev_m = elev_m - 1200.0 * imb * np.clip(envS * 1.6 - 0.2, 0, 1) * (elev_m > 150)
    # 大陆腹地盆地 (刚果/塔里木型): 距海岸远的内部低地
    d_in = distance_transform_edt(elev_m > 0)
    deep_int = np.clip((d_in - 130.0) / 240.0, 0, 1)
    basins = np.clip(value_fbm(size_px, 6.0, 3, rng=rng) - 1.0, 0, None)
    elev_m = elev_m - 1600.0 * deep_int * basins
    # ---- v6.6 检验修复层 (七大地理维度对齐) ----
    elev_m = ocean_margin_fix(elev_m, envS)                # P0 陆架-陆坡剖面修复
    elev_m = add_intraplate_orogen(elev_m, cf, envS, rng)  # P2 板内造山(参与直方图)
    elev_m = land_hist_match(elev_m, lats)                 # P0 陆地直方图 → 地球实测(加权)
    elev_m = add_island_arcs(elev_m, envS, cf, rng)        # P1 岛弧链 (俯冲带大陆侧)
    elev_m = add_hotspot_chains(elev_m, rng)               # P1 热点岛链 (夏威夷型)
    elev_m = np.clip(elev_m, -10500, 8900).astype(np.float32)
    landS = elev_m > 0
    plt.imsave(os.path.join(_dir, "output", "world_skeleton.png"),
               make_terrain_cmap()(m2v(elev_m))[..., :3])
    print(f"      陆地 {(elev_m > 0).mean() * 100:.1f}% | 峰 {elev_m.max():.0f}m | 沟 {elev_m.min():.0f}m")
    print(f"    [阶段计时] [1/4] 板块演化: {time.time()-_stage_t0:.2f}s"); _stage_t0 = time.time()

    print("[2/4] 构建 12 通道全球粗底图 (含构造场)...")
    ch_raw, bm, bstd, bmin, bmax, lf = world_to_channels(elev_m, lats, tect_T, block, seed)
    print(f"    [阶段计时] [2/4] 粗底图构建: {time.time()-_stage_t0:.2f}s"); _stage_t0 = time.time()
    glac = ch_raw[6]
    W = bm.shape[0]
    mu, sd = ch_stats["mean"][:, None, None], ch_stats["std"][:, None, None]
    grng = np.random.default_rng(seed + 999)
    rows = list(range(0, W, step))
    total = len(rows) ** 2
    print(f"[3/4] 逐瓦片 GAN + 低频对齐 ({len(rows)}x{len(rows)})...")
    canvas = np.zeros((size_px, size_px), np.float32)
    wgt = np.zeros((size_px, size_px), np.float32)
    ramp = np.ones(256, np.float32)
    ramp[:ovp] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ovp) / ovp)
    ramp[-ovp:] = ramp[:ovp][::-1]
    done, t0 = 0, time.time()
    pad2 = lambda a, p: np.pad(a, ((0, p[0]), (0, p[1])), mode="edge")
    sm_ = lambda a: gaussian_filter(a.astype(np.float32), 0.6)
    for by in rows:
        for bx in rows:
            by1, bx1 = min(by + tb, W), min(bx + tb, W)
            hb, wb = by1 - by, bx1 - bx
            py, px_ = tb - hb, tb - wb
            sl = (slice(by, by1), slice(bx, bx1))
            ch = ch_raw[:, sl[0], sl[1]] + \
                 grng.normal(0, 0.015, (12, hb, wb)).astype(np.float32)
            ch = (ch - mu) / sd
            if py or px_:
                ch = np.pad(ch, ((0, 0), (0, py), (0, px_)), mode="edge")
            inp = np.ascontiguousarray(np.repeat(np.repeat(ch, block, 1), block, 2))
            raw = gen(torch.from_numpy(inp[None]).float().to(device)).cpu().numpy()[0, 0]
            s = lowfreq_consistency(raw.astype(np.float32),
                                    pad2(sm_(bm[sl]), (py, px_)), block)

            # 深海块的硬上界 = 块均值+0.06 (比训练版cap=0.46强得多:
            # 0.46仍在海平面附近, 会把深海钳成浅海; 这版直接钳回深海)
            lf_up = ndi_zoom(pad2(lf[sl], (py, px_)), block, order=1)[:256, :256]
            bm_up = ndi_zoom(pad2(sm_(bm[sl]), (py, px_)), block, order=1)[:256, :256]
            sea_m = lf_up < 0.006                   # v6.6: 放宽→保住岛弧/小岛
            cap_m = np.where(sea_m, np.maximum(bm_up + 0.06, 0.12), 1.0)
            s = np.minimum(s, cap_m).astype(np.float32)

            h, w = hb * block, wb * block
            # v7.2: 移除 L498-501 的 "图像边界瓦片强制 win=1.0" —— canvas/wgt 全局加权合成
            #      即使图像外无邻居瓦片，ramp 边缘 taper=0.5 也能通过除以单边权重 wgt 正常重建。
            #      保留汉宁羽化 taper 使得边界瓦片与内部瓦片在 overlap 区的权重和=1.0,
            #      彻底消除 tile 208px 网格线的竖/横硬接缝.
            win = np.outer(ramp, ramp)
            y0p, x0p = by * block, bx * block
            canvas[y0p:y0p + h, x0p:x0p + w] += s[:h, :w] * win[:h, :w]
            wgt[y0p:y0p + h, x0p:x0p + w] += win[:h, :w]
            done += 1
            if done % 50 == 0 or done == total:
                print(f"      瓦片 {done}/{total} | {time.time() - t0:.0f}s", flush=True)

    out = np.clip(canvas / np.maximum(wgt, 1e-8), 0, 1).astype(np.float32)
    # v7.2: 瓦片接缝二次羽化 —— 在 step*block (208px) 整数倍的行/列上做 3 像素 σ=0.6 的横向/纵向条带平滑
    #      (仅对 seam 附近 3px 宽 做高斯混合, 不破坏全局高频细节; 经纬度双重)
    seam_rows = sorted(set([r * step * block for r in range(1, len(rows)) if r * step * block < size_px]))
    seam_cols = sorted(set([c * step * block for c in range(1, len(rows)) if c * step * block < size_px]))
    seam_mask = np.zeros_like(out, dtype=bool)
    for r in seam_rows:
        seam_mask[max(0, r-2):min(size_px, r+3), :] = True
    for c in seam_cols:
        seam_mask[:, max(0, c-2):min(size_px, c+3)] = True
    if seam_mask.any():
        out_s = gaussian_filter(out, 0.6)
        # 仅 seam 区混合 (blend 0.5 平滑 + 0.5 原版)
        out = np.where(seam_mask, 0.5 * out + 0.5 * out_s, out).astype(np.float32)
    print(f"    [阶段计时] [3/4] GAN 瓦片+接缝: {time.time()-_stage_t0:.2f}s"); _stage_t0 = time.time()
    print("[4/4] 全局阶段: CDF匹配 → 低频复核 → 去斑 → 水力侵蚀 → 填洼D8河网...")
    _g_t0 = time.time()
    out = global_envelope_match(out, bm, bstd, bmin, bmax, glac, block,
                                blend=0.60, seed=seed, cap_m=8200.0)
    print(f"    [4/4 计时] global_envelope_match: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = lowfreq_consistency(out, bm, block, iters=2)
    print(f"    [4/4 计时] lowfreq_consistency: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = despeckle(out, min_area=150)            # v33: 32→90; v37: 90→150 (新海陆几何下 n_islands 220 vs 地球 113)
    print(f"    [4/4 计时] despeckle: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # v52: 极区去碎 —— 板块模拟在极区切出的碎岛占全图陆地近五成, 是 "大陆像打碎的
    #   拼图" 与 "苔原 44.5% (地球 10%)" 的共同根因. 只删无地理意义的小岛 (≥58° 起).
    out = polar_despeckle(out, lats)
    print(f"    [4/4 计时] polar_despeckle: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = droplet_erosion(out, rng=np.random.default_rng(seed + 77))   # V形谷/冲积扇
    print(f"    [4/4 计时] droplet_erosion: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = np.clip(out + 0.016 * value_fbm(size_px, 110.0, 2,
                    rng=np.random.RandomState(seed + 3)) * (out >= 0.5),
                  0, 1).astype(np.float32)        # v6.6 高频细节(提PSD)
    print(f"    [4/4 计时] value_fbm(110.0,2) 高频: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # v8.1: 为 carve_trunk_rivers 计算气候场 (P/T) — 干流走廊按水量加权 + 曲率扰动
    # 注: render_world 末段还会再算一次 (供雪线/生物群系用); 此处仅用于干流走廊雕刻
    elev_mid = v2m(out)
    ci_mid = continentality_index(elev_mid, lats)
    dT_mid, dP_mid = ocean_current_effects(elev_mid, lats)
    T_mid = temperature_field(lats, elev_mid, dT_mid, ci=ci_mid)
    print(f"    [4/4 计时] continentality+temp(ocean_currents): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    P_mid = precipitation_field(lats, elev_mid, np.random.RandomState(seed + 88),
                                 dP_mid, ci=ci_mid)
    print(f"    [4/4 计时] precipitation_field(mid): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = carve_trunk_rivers(out, np.random.RandomState(seed + 55),
                             precip=P_mid, temp=T_mid,
                             warp_amp=3.0, warp_freq=0.015)    # v8.1 气候加权 + 曲率扰动干流河谷
                                                               # v34: warp_freq 10.0→0.015 修复 —
                                                               # meander base_freq 是"每单位路径长度
                                                               # 振荡数", 10.0 会让 500px 河道产生上千
                                                               # 次像素级震荡, 平滑后河道退化为直线
    print(f"    [4/4 计时] carve_trunk_rivers: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # v6.6: 窄带海岸锐化(±0.008v≈±160m) → 不再压掉陆架/抬升陆地直方图
    out = sharpen_coastline(out, band=0.008, expand=0.03)
    # M3 峡湾雕刻: 高纬被动边缘 U 形谷 (斯堪的纳维亚型), 锐化后/陆架终检前
    if margin_up is not None:
        out = m2v(carve_fjords(v2m(out), lats, margin_up,
                               np.random.RandomState(seed + 41)))
    out = m2v(ocean_margin_fix(v2m(out), envS, margin_up))   # v6.6 陆架终检(v3.0边缘分化)
    ice = polar_ice_mask(lats, out, _SL_INCISION_M)
    print(f"    [4/4 计时] sharpen+carve_fjords+ocean_margin_fix+polar_ice: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # v39: D8 → MFD 切谷. 单流向 D8 在倾斜面上产生等间距平行直线, 下切后看起来
    # 像"地面裂缝"; MFD 让水流在坡面散开、河谷在汇流处收敛, 切出的谷更自然.
    # v39h: 流场叠微地形抖动 (见 render_world 同名注释), 否则切出的河谷在克拉通
    #       平原上是一刀完美的直线.
    # v39i: 与 render_world 共用 hydrologic_surface (同样的域扭曲场, seed 固定)
    #       → 切出的河谷与渲染的河道重合, 地形与水系真正耦合.
    _flow_c, _ = hydrologic_surface(out)
    acc_c = climate_flow_accumulation(_flow_c, P_mid, T_mid,
                                      np.random.RandomState(seed + 31), mfd=True)
    thr = max(float(np.quantile(acc_c, 0.995)), 2.0)
    t = np.clip(np.log(np.maximum(acc_c, 1.0001)) / np.log(thr), 0, 1)
    landness = np.clip((out - 0.5) / 0.02, 0, 1)
    out = np.clip(out - 0.028 * (t ** 0.8) * landness * (~ice), 0, 1).astype(np.float32)
    print(f"    [4/4 计时] mfd_flow_accumulation+incision: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    out = land_hist_final(out, lats)              # v6.6 陆地直方图终检(绝对最后,加权)
    print(f"    [4/4 计时] land_hist_final: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # M2 失败裂谷重新下切: GAN/直方图重排会抹掉骨架版谷地 → rift 场
    # (轴剖面: 轴线1→两侧衰减) 直乘深度, 谷形集中 (东非型);
    # 谷底整体低于两侧 + 轴线稀疏火山峰 (乞力马扎罗型)
    if riftS.any():
        valley = 0.055 * riftS                     # ≈1100m 谷底 (东非裂谷谷底-周边 700-1500m)
        out = np.clip(out - valley * (out >= 0.5), 0, 1).astype(np.float32)
        _vf = value_fbm(size_px, 40.0, 2, rng=np.random.RandomState(seed + 9))
        volc = (riftS > 0.75) & (out >= 0.5) & (_vf > 0.55)
        out = np.clip(out + 0.030 * volc, 0, 1).astype(np.float32)
        print(f"    [4/4 计时] rift valley+volc(value_fbm 40.0,2): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()

    # M5 克拉通大流域盆地已移至 high_elev_compress 之后执行
    # (原位置在 enhance_terrain_texture 之前, ±200m 纹理噪声会摧毁
    #  M5 创建的 ~4m/px 线性倾斜面, 导致 largest_basin_share 仅 0.5%)
    # ---- M3 边缘地貌终章 (直方图终检后, 最后盖章不被重排) ----
    dem_m = v2m(out)
    if margin_up is not None:
        dem_m, n_pen = add_shelf_peninsulas(dem_m, margin_up, cf, rng,
                                            lats=lats)   # 台地半岛+大陆岛
        print(f"      M3 台地露头(半岛/大陆岛): {n_pen}px")
    dem_m = add_coral_reefs(dem_m, lats, np.random.RandomState(seed + 88))  # 珊瑚礁
    dem_m = ocean_hist_final(dem_m, lats)      # M6 深洋直方图终检 (最后, 深带校准)
    print(f"    [4/4 计时] shelf_peninsulas+coral+ocean_hist_final: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v7.2: 海底 GAN 伪影 (同心等高线波纹 / Tile DC 方块色) 专属平滑
    #       海陆严格分离 → 只对 dem_m < 0 的海洋像素应用, 陆地/海岸线 100% 保留.
    # ---- v7.7: 强化 DC 偏移修正 (blockiness_ratio 1.123→目标<1.03) ----
    # ---- v8.0: 软权重过渡替代 norm_long>0.02 硬阈值, 消除岛屿周边 ~330m DC 阶跃 ----
    sea = dem_m < 0.0
    if sea.any():
        f = dem_m.astype(np.float32)
        # 1) 两级去噪: σ=1.4 高斯 (跨瓦片边界轻度混合, 保边) + 5×5 中值 (清同心等高线+1-2px抖动)
        dem_g = gaussian_filter(f, 1.4)
        dem_mm = median_filter(f, size=5)
        # 深度分层加权: 浅海/陆坡(-2500m 以浅) 只 15% 混合 → 保住刚刻出的陆架坡折
        # (v45: 旧式在 -500m 就给 0.30, 5×5 中值把 2~3px 宽的陆坡摊平, 坡折消失);
        # 深海(<-2500m) 从 15% 升到 75% (深海本应平坦, 需要压掉瓦片 DC 纹理)
        w = 0.15 + 0.60 * np.clip((-f - 2500.0) / 2500.0, 0.0, 1.0)
        dem_mix = (1.0 - w) * f + w * (0.60 * dem_g + 0.40 * dem_mm)
        # 2) 低频参考场去方块偏移 (σ=32): 只扣 78%, 避免把大陆坡的缓变深度也抹平
        smooth_long = gaussian_filter(np.where(sea, f, 0.0), 32.0)
        norm_long = gaussian_filter(sea.astype(np.float32), 32.0)
        ref_sea = smooth_long / np.maximum(norm_long, 1e-4)
        w_dc = np.clip((norm_long - 0.02) / 0.38, 0.0, 1.0).astype(np.float32)
        ref = w_dc * ref_sea + (1.0 - w_dc) * f
        dem_dc = f - 0.78 * (f - ref)
        # 合并: 40% 去波纹 + 60% 去方块偏移
        sea_final = 0.40 * dem_mix + 0.60 * dem_dc
        dem_m = np.where(sea, sea_final.astype(dem_m.dtype), dem_m)
    print(f"    [4/4 计时] sea 伪影平滑 (gauss σ=1.4/32 + median): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v7.7: 海底地貌增强 (深海丘陵/洋中脊/海山链/陆隆收窄) ----
    dem_m = enhance_ocean_floor(dem_m, lats, envS=envS,
                                rng=np.random.RandomState(seed + 22))
    print(f"    [4/4 计时] enhance_ocean_floor: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v9.x: 俯冲带海沟加深 → 恢复 hadal 带占比 (地球 0.96%, v11 仅 0.13%) ----
    # sea 伪影平滑把窄海沟填平 (σ=1.4 + median 5×5), 在俯冲核带重新下切
    sea_deep = dem_m < -3000.0
    if sea_deep.any() and envS is not None:
        # 俯冲强度 > 0.4 的深海 → 海沟加深, 幅度随深度和俯冲强度递增
        trench_mask = sea_deep & (envS > 0.4)
        if trench_mask.any():
            env_w = np.clip((envS - 0.4) / 0.5, 0.0, 1.0)
            depth_w = np.clip((-dem_m - 3000.0) / 3000.0, 0.0, 1.0)
            deepening = -2500.0 * env_w * depth_w  # 最深 -2500m (→ -5500~-8500m)
            dem_m = np.where(trench_mask, dem_m + deepening, dem_m)
            dem_m = np.clip(dem_m, -11000.0, 8900.0).astype(np.float32)
            n_trench = int(trench_mask.sum())
            print(f"      [海沟加深] {n_trench}px | 最深 {dem_m.min():.0f}m")
    print(f"    [4/4 计时] trench_deepening: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- 纬度带陆地平衡：限制极地陆地比例，维持全球陆海比 ----
    dem_m = balance_land_latitude(dem_m, lats)
    print(f"    [4/4 计时] balance_land_latitude: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- 大陆架生成：在海岸线周围创造平缓的浅海过渡 ----
    # v80g: 边缘类型系数（被动宽架 ↔ 主动窄架）在这里**只算一次**。此刻 DEM 还
    #   没有人造陆架，σ=46px 窗口里的"周边海底深度"反映的是真实的盆地格局；
    #   之后再建的陆架/陆坡都在 −200m 以内，不会反过来污染这层判据。
    #   下面三次 create_continental_shelf 与 carve_margin_trenches 共用同一张
    #   冻结场，保证"窄陆架"与"海沟"同址（同一段主动边缘）。
    _mcls_frozen = margin_class_map(dem_m)
    print(f"    [margin] 边缘类型系数 p05={np.percentile(_mcls_frozen,5):.2f} "
          f"p50={np.percentile(_mcls_frozen,50):.2f} "
          f"p95={np.percentile(_mcls_frozen,95):.2f}", flush=True)
    # v32: 收窄大陆架 8→5px, 外缘 -200→-180m. 让大陆边缘架宽更接近地球 80km 平均。
    # 注意: 实测 ocean_-200_0 指标(0.174 vs 地球0.071)对该参数几乎不变 —— 该浅海大环
    # 主导来源是上游过量造陆(岛弧/热点/半岛/珊瑚)被 land_fraction_calibration 淹没成
    # 浅海所致, 属全局造陆量问题, 需单独重构, 非本参数可调。
    dem_m = create_continental_shelf(dem_m, lats, shelf_width_px=5, outer_depth=-180.0,
                                     mcls=_mcls_frozen, detail=False)
    print(f"    [4/4 计时] create_continental_shelf: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v9.x: 海岸带平滑 → 合并 <2000km² 近岸碎岛, 减少流域出口碎片化 ----
    # 地球大流域占比 4.4%, v11 仅 0.5%; 主因是碎岛过多导致出口分散 (51165 vs 37178)
    land_b = dem_m > 0
    if land_b.any():
        from scipy.ndimage import label as _ndi_label
        lab_b, n_b = _ndi_label(land_b)
        if n_b > 1:
            sizes_b = np.bincount(lab_b.ravel())
            sizes_b[0] = 0
            total_land = land_b.sum()
            small_isle_mask = np.zeros_like(land_b)
            for cid in range(1, n_b + 1):
                if 0 < sizes_b[cid] < total_land * 0.001:  # v33: <0.1% 陆地 = 碎岛合并 (原 0.03%)
                    small_isle_mask |= (lab_b == cid)
            if small_isle_mask.any():
                # 碎岛降为浅海 (-30m), 与大陆架融合 → 减少出口
                dem_m[small_isle_mask] = -30.0
                # 海岸带高斯平滑 (仅碎岛周边 3px)
                coast_band = np.zeros_like(land_b)
                coast_band[small_isle_mask] = True
                from scipy.ndimage import binary_dilation
                coast_band = binary_dilation(coast_band, iterations=3)
                if coast_band.any():
                    smooth_dem = gaussian_filter(dem_m.astype(np.float32), 1.2)
                    dem_m = np.where(coast_band, 0.5 * dem_m + 0.5 * smooth_dem, dem_m)
                n_merged = int(small_isle_mask.sum())
                print(f"      [海岸平滑] 合并碎岛 {n_merged}px → 浅海")
    print(f"    [4/4 计时] coast_smooth: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- 地形纹理增强：多尺度高频噪声 → 功率谱 β 接近地球 2.0-2.5 ----
    dem_m = enhance_terrain_texture(dem_m, lats, rng=np.random.RandomState(seed + 101))
    print(f"    [4/4 计时] enhance_terrain_texture: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # M5 克拉通大流域盆地 (移至纹理增强之后, 确保线性倾斜面不被 ±200m 噪声摧毁):
    # 稳定陆内低地 → 单一出口倾斜面 + 河网树 (亚马逊/刚果型) + 内陆海 (里海型)
    # 注: M5 河床 z_riv=z_out+1.8·d_o 对大盆地可达数千米, 须由后续 high_elev_compress 压缩
    dem_b, n_b, b_info = add_craton_basins(dem_m, envS,
                                           np.random.RandomState(seed + 66),
                                           belt=beltS, lats=lats, max_exo=3,
                                           max_endo=1, precip=P_mid,
                                           endo_p_max=800.0,
                                           target_lo=0.03, target_hi=0.08,
                                           dyke_coast_excl=float(os.environ.get("FWS_DYKE_COAST", "40.0")),
                                           dyke_height=float(os.environ.get("FWS_DYKE_H", "250.0")),
                                           basin_max_share=float(os.environ.get("FWS_BASIN_MAX", "0.10")))
    if n_b:
        dem_m = dem_b
        print(f"      M5 大流域盆地: {n_b}个 [{b_info}]")
    print(f"    [4/4 计时] add_craton_basins: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v33c: 造山带聚合已撤销 ----
    # 实测 (seeds 32-36, 三种方案): 沿 envS/beltS 的 DEM 抬升 (脊状或连续) 均把山地
    #   切成更多孤立块 (n_mtn_ranges 38→66↑), 降低 belt_share_of_mtn 并横切 M5 大流域
    #   (largest_basin_share 0.105→0.006↓), 湖泊反升. 根因: 本生成器的板块边界场
    #   (envS/beltS) 本身就是零散短弧, 抬升只是放大碎片化, 无法造出地球式的少数
    #   巨长山链. 山系连带的真正修复需强化 plate_sim 的俯冲带连续性, 属下一阶段.
    #   保留的改进: 山地脊状渲染纹理 (v33) 让已有山地视觉上更线性.
    print(f"    [4/4 计时] v33 belt_aggregate(removed): 0.00s"); _g_t0 = time.time()
    # ---- v27: 全球陆地比例双向校准到 ~30% (地球 29.16%), 在 M5 之后 ----
    # 陆地过多 → 淹没最低低地; 陆地不足 → 抬升最浅海床. 双向保证目标比例.
    _target_land = 0.30
    _cur_land = float((dem_m > 0).sum()) / dem_m.size
    if _cur_land > _target_land + 0.01:
        _land_px = dem_m[dem_m > 0]
        _flood_frac = 1.0 - (_target_land * dem_m.size / max(_land_px.size, 1))
        if 0 < _flood_frac < 1:
            _thresh = float(np.quantile(_land_px, _flood_frac))
            _flood = (dem_m > 0) & (dem_m <= _thresh)
            dem_m = np.where(_flood, -20.0 - (_thresh - dem_m) * 0.3, dem_m).astype(np.float32)
            print(f"      [陆地校准] {_cur_land*100:.1f}% → {(dem_m>0).mean()*100:.1f}% "
                  f"(淹没阈值 {_thresh:.0f}m, {_flood.sum()}px)")
    elif _cur_land < _target_land - 0.01:
        _ocean_px = dem_m[(dem_m <= 0) & (dem_m > -300)]  # 浅海抬升到 -300m
        if _ocean_px.size > 100:
            _need = (_target_land - _cur_land) * dem_m.size
            _raise_frac = min(1.0, _need / _ocean_px.size)
            _thresh = float(np.quantile(_ocean_px, 1.0 - _raise_frac))
            _raise = (dem_m <= 0) & (dem_m > _thresh) & (dem_m > -300)
            dem_m = np.where(_raise, 10.0 + (dem_m - _thresh) * 0.3, dem_m).astype(np.float32)
            print(f"      [陆地校准] {_cur_land*100:.1f}% → {(dem_m>0).mean()*100:.1f}% "
                  f"(抬升阈值 {_thresh:.0f}m, {_raise.sum()}px)")
    print(f"    [4/4 计时] land_fraction_calibration: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v9.x: 高海拔线性压缩 (M5 之后, 压缩 M5 河床/dyke 超高值) ----
    land_hi = (dem_m > 1500.0)
    if land_hi.any():
        # 凸函数压缩 (exp 2.0), 起点 1500m, norm 截断到 1.0
        # (M5 河床 z_riv=z_out+1.8·d_o 对大盆地可达 9000m+, norm>1 会使输出>输入)
        hi_vals = dem_m[land_hi]
        norm = np.clip((hi_vals - 1500.0) / 7500.0, 0.0, 1.0)
        out = 1500.0 + (norm ** 2.0) * 4850.0
        dem_m[land_hi] = out.astype(np.float32)
    n_high = int(((dem_m > 3500) & (dem_m > 0)).sum())
    print(f"      [高海拔压缩] >3500m 陆地 {n_high}px | 峰 {dem_m.max():.0f}m | 均高 {dem_m[dem_m>0].mean():.0f}m")
    print(f"    [4/4 计时] high_elev_compress: {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v31: 陆地归一化平滑 (修复坡度过陡 + 平原破碎) ----
    # seed11 实测平均陆地坡度 1.14° (地球 0.41°), 主因是低频直方图拉伸后
    # 平原-高地边界的高频起伏被放大; 0-200m 平原也被切成上千碎块 (最大块仅
    # 占陆地 12%, 地球为连片大平原)。用海洋填 0 + 归一化卷积对陆地做 σ=1.5
    # 平滑: 线性斜坡 (M5 克拉通盆地 4m/px) 对高斯模糊近似不变 → 大流域指标
    # 不受影响; 只消掉亚像素/2px 级陡峭微起伏。平滑后再末次直方图对齐,
    # 海拔分布与峰值被 land_hist_final 重新精确恢复。
    _land_s = dem_m > 0
    if _land_s.any():
        _smooth_v = gaussian_filter(np.where(_land_s, dem_m, 0.0), 1.5)
        _norm_w = gaussian_filter(_land_s.astype(np.float32), 1.5)
        _sm = np.where(_norm_w > 1e-4, _smooth_v / np.maximum(_norm_w, 1e-4), 0.0)
        dem_m = np.where(_land_s, _sm, dem_m).astype(np.float32)
    print(f"    [4/4 计时] land_smooth(v31 σ=1.5): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v30: 绝对最后的陆地直方图对齐 (修复"台地星球") ----
    # 前序诸步 (balance_land_latitude / create_continental_shelf / coast_smooth /
    # enhance_terrain_texture / add_craton_basins / land_fraction_calibration /
    # high_elev_compress) 会持续雕塑陆地高程: 其中 land_fraction_calibration
    # 在把陆地比例砍回 30% 时会优先淹没最低低地, high_elev_compress 又把 >1500m
    # 压缩到峰值 ~3530m —— 两者叠加导致陆地几乎全部堆在 1000-2000m 台地带,
    # 0-200m 平原~0%、>3500m 高山~0%、峰值仅 ~3100m (地球 28%平原/64%<1000m/高峰~6350m)。
    # 在最末尾按地球实测 cos 纬度加权分位做单调重排: 不改变陆地份额(仍 30%)，
    # 只按排名恢复 平原/台地/山地 的正确比例与真实高峰, 河流与流域的相对顺序保持。
    # 注: 必须先 m2v(dem_m) 重建 out —— high_elev_compress 内的局部 out 是 1D 掩码数组。
    out = land_hist_final(m2v(dem_m), lats)
    dem_m = v2m(out)
    print(f"    [4/4 计时] land_hist_final(v30 末次): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v38: 终稿山链迁移 (修复"山系不成带") ----
    # 诊断 (diag_belt, seed0): plate_sim 的 belt/envS 场本身 AR 6.9 (≈地球 6.4),
    # 但 >2000m 高山被 GAN 质量块/板内造山/岛弧等"散山"占满 (最大单体 31k px
    # AR 2.4, 平均 4.3km), 带内山块占比仅 16% → belt_share 0.02-0.12 vs 地球 0.84.
    # 历次前置抬升 (700-1600m) 都会被后续 GAN/直方图重排稀释, 无法把带顶到
    # 最高的 10-11% 排名区 (那里的山才是 >2000m 山块统计源).
    # 解法 (v30 同款收尾模式): 在**末次直方图之后**给构造带走廊二次抬升, 再跑
    # 一次 land_hist_final 恢复地球海拔分布 —— 单调重排不改变全球陆地/海陆比例,
    # 只让"最高排名"净迁移到连续山链上 → 高山成带, 散山降为低山丘陵(坡度同步降).
    _beltX = np.maximum(envS, 0.85 * beltS)
    if _beltX.max() > 0.05:
        # v47 定稿参数 (实测 sweep): 抬升 2400m / 非带压降 3200m / 走廊 σ=4.5 /
        #   阶跃区间 [0.05, 0.12] / 脊线低通 σ=8 — >2000m 高山 belt_share
        #   0.45→0.84, 高山占陆地 ~9.7% (地球 ~8%), 块 AR ~2.9.
        _ld, _pd, _sd = 2400.0, 3200.0, 4.5
        _t0, _t1, _sps = 0.05, 0.12, 8.0
        # 平滑加宽带走廊 (σ≈50px), 弥合带内 <2000m 空隙, 链不断裂
        _beltB = gaussian_filter(_beltX.astype(np.float32), _sd)
        # 窄脊核 (σ=1.2≈23px) —— 抬升幅度只给山脊线, 走廊宽度只保证"不断链".
        _beltN = gaussian_filter(_beltX.astype(np.float32), 1.2)
        # v38: 先重度平滑山核 (σ=4≈80px) 再整块抬升 — 末次 CDF 重排会把带内
        # 残余纹理放大成坡度; 高原核心须在抬升**前**就是平的 (地球青藏/安第斯
        # 高原核 ~0.3°), 纹理只留在 <2000m 边缘带.
        _sm_belt = gaussian_filter(dem_m.astype(np.float32), 8.0)
        # v39: 形态学闭运算 (5×5) 弥合走廊断点 — 用户反馈山系是"孤立圆润斑点",
        # 需要连续缝合带. σ=4.5 先加宽, 再闭运算消掉 <5px 的带内缺口.
        _beltC = binary_closing(_beltB > 0.30, structure=np.ones((5, 5), bool)) \
            .astype(np.float32)
        dem_m = np.where((dem_m > 0), dem_m * (1.0 - 0.92 * _beltC)
                         + _sm_belt * (0.92 * _beltC), dem_m).astype(np.float32)
        # 阶跃抬升: 走廊权重 (宽, 保连续) × 脊核权重 (窄, 定形状)
        _beltM = np.clip((_beltB - _t0) / _t1, 0.0, 1.0).astype(np.float32)
        # v47: 脊核从"阈值带"改为"脊线提取" (带场 − 其 σ=sps 低通).
        #   阈值带仍是**整条宽带**, 高山成"圆墩山包"; 带场减低通只在脊线附近为
        #   正, 得等高线状窄脊 → 山体沿带走向收窄成长条山链.
        #   注: 后续 land_hist_final 是全局秩重排, 任何随带场**单调**的抬升终究
        #   只把 >2000m 变成带场等值圈 (AR 几何上限 ≈3, 地球 6.4); 曾试带内周期
        #   振荡 (平行脊) 拆环, AR 仅 2.95→2.95, 收益为零且易出同心环伪影, 故弃.
        _spn = _beltN - gaussian_filter(_beltN, _sps)
        _spn_pos = _spn > 0
        _sc = float(np.percentile(_spn[_spn_pos], 85.0)) if _spn_pos.any() else 1.0
        _crest = np.clip(_spn / max(_sc, 1e-6), 0.0, 1.0).astype(np.float32)
        _lift = (_ld * _beltM * (0.25 + 0.75 * _crest)
                 + 0.06 * _ld * _beltM * (value_fbm(size_px, 40.0, 3,
                                                    rng=np.random.RandomState(seed + 9999)) - 0.5))
        dem_m = np.where((dem_m > 0), dem_m + _lift, dem_m).astype(np.float32)
        # 非带高地压降: 把占住排名高位但不成带的 GAN 质量块/散山按高度递减压回,
        # 让"最高排名"净迁移到连续山链 (散山 → 低山丘陵, 顶部带与后者平滑交界)
        # v46: 起点 1100→700m, 满压位置 2900→2200m, 力度 1800→2600m —
        #   实测 >2000m 中 71% 落在 belt<0.3 的散山上 (AR 2.65), 压降不足.
        _pen = _pd * (1.0 - _beltM) * np.clip((dem_m - 700.0) / 1500.0, 0.0, 1.0)
        dem_m = np.where((dem_m > 0), dem_m - _pen, dem_m).astype(np.float32)
        out = land_hist_final(m2v(dem_m), lats)
        dem_m = v2m(out)
        print(f"      [v38] 山链迁移: lift={_ld:.0f} pen={_pd:.0f} sig={_sd:.1f} "
              f"带 {int((_beltB > 0.2).sum())}px, 压降 {int((_pen > 50).sum())}px",
              flush=True)
        _hi_stat(dem_m, "belt_migrate", _beltM)
    print(f"    [4/4 计时] belt_migrate(v38 末次): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v38: 山地坡度过陡软化 ----
    # 山链迁移把高山净搬到持续山带, 但带缘/岩壁仍比地球陡 (0.60 vs 0.41).
    # 对陆地局部坡度 >1.2° 的像素按 (sl-1.2)/2.2 ∈[0,1] 权重混合 σ=2 平滑:
    # 只软化亚像素-数像素级陡坎, 不削高原/山链 (>1 度以上的真实山地保留).
    _sm_ = gaussian_filter(dem_m.astype(np.float32), 2.0)
    _gy_, _gx_ = np.gradient(dem_m.astype(np.float64))
    _dy_m_ = 111320.0 * 180.0 / size_px
    _dx_m_ = (111320.0 * 360.0 / size_px) * \
        np.cos(np.deg2rad(lats.astype(np.float64)))
    _sl_ = np.rad2deg(np.arctan(np.hypot(_gx_ / _dx_m_[:, None], _gy_ / _dy_m_)))
    _soft_ = (dem_m > 0) & (_sl_ > 1.0)
    if _soft_.any():
        _w_ = np.clip((_sl_ - 1.0) / 2.0, 0.0, 1.0) * 0.65
        dem_m = np.where(_soft_, dem_m * (1.0 - _w_) + _sm_ * _w_, dem_m) \
            .astype(np.float32)
        # 注: 不再补跑 land_hist_final — 平滑的对称性使海拔分布漂移极小 (<1%),
        # 避免 CDF 重排把坡度重新拉高 (已验证 0.60 不动的元凶)
        print(f"      [v38] 坡度软化: {int(_soft_.sum())}px (sl>1.0°)", flush=True)
    _hi_stat(dem_m, "slope_relax")
    print(f"    [4/4 计时] slope_relax(v38): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # ---- v37: 末次大陆架重建 (修复 land_fraction_calibration 吃掉陆架) ----
    # 早期 create_continental_shelf(1213) 建的 -200~0m 陆架带会被后续 M5 重塑与
    # land_fraction_calibration(抬升浅海补齐陆地) 覆盖/消耗 —— v37 实测 ocean_-200_0
    # 降到 0.004 (地球 0.071), 3D 球体上海岸直落入深海、无陆架. 在末次高程后重建:
    # 仅作用于 dem<0 且 <5px 的海, 与陆地/河网/直方图无冲突.
    # ---- v39: 海浪侵蚀/泥沙沉积 (合并碎岛 + 抹平高频破碎岸线) ----
    # 诊断: 陆缘仍是"高频噪波裁切"的峡湾+碎岛, 无海滩/滩涂过渡; 地球大块陆地
    # 边缘由波浪侵蚀与沿岸流输沙长期抹平 (三角洲/沙嘴/泻湖是这种作用的残留).
    dem_m = coastal_erosion(dem_m, sigma=2.4, band_px=6, strength=0.85)
    print(f"    [4/4 计时] coastal_erosion(v39): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    # v89: 这里**不铺**陆架微地形/海底峡谷（detail=False）—— 下面 5000 行附近
    #   （sculpt/reshape 之后）还有一次建架，那才是终稿；两次都铺会重复叠刻。
    dem_m = create_continental_shelf(dem_m, lats, shelf_width_px=6, outer_depth=-170.0,
                                     mcls=_mcls_frozen, detail=False)
    print(f"    [4/4 计时] shelf_rebuild(v37 末次): {time.time()-_g_t0:.2f}s"); _g_t0 = time.time()
    _hi_stat(dem_m, "shelf")
    # ---- v34: M5 倾斜面后重雕刻蜿蜒河道 (打破"平行直线河"纹理) ----
    # 诊断 (seeds 32-36 实测): M5 线性倾斜面使流域内 D8 流线全部平行笔直,
    # 地图上呈现 45° 等间距斜线纹理 (河流蜿蜒度 1.00-1.04, 地球 ~2+);
    # 且早期 carve_trunk_rivers 的蜿蜒河道被 M5 倾斜面覆盖抹除.
    # 解决: 在绝对地形终稿上重跑轻量蜿蜒雕刻 —— 从最终 DEM 提取河网骨架
    # → meander 蜿蜒化 → 沿中心线雕刻, 使最终渲染的河流 (含 M5 盆地内的
    # 干流与支流) 恢复自然蜿蜒形态, 同时保留源头/分水岭结构.
    _v34_t0 = time.time()
    try:
        v01_end = m2v(dem_m)
        v01_end = carve_trunk_rivers(
            v01_end, np.random.RandomState(seed + 777),
            precip=P_mid, temp=T_mid,
            sigma=2.5, q=0.998, max_depth=0.022, width=2.8,
            warp_amp=4.5, warp_freq=0.015,
            q_levels=(0.992, 0.976, 0.955),
            hack_law=True, oxbow=True)
        dem_m = v2m(v01_end)
        print(f"    [4/4 计时] v34_redecorate_rivers(末次蜿蜒雕刻): {time.time()-_v34_t0:.2f}s")
        _hi_stat(dem_m, "rivers_final")
    except Exception as _e:
        print(f"      [v34_redecorate_rivers 跳过] {_e}")
    # ---- v3.3: 回填大陆内部封闭水域 ----
    # 用户反馈"大陆内部密布蓝色小斑点, 像在陆地上打洞". 实测终稿 DEM 内有 452 个
    # 与图幅边界不连通的封闭负高程水体 (15.0% 陆地面积, 400+ 个 <512px 碎点),
    # 渲染时走海水色卡 → 正好是那些蓝斑. 放在河道重雕之后、出图之前, 只保留
    # 最大的 2 个巨型内陆海 (里海/贝加尔型), 其余回填成缓盆.
    # ---- v62: 大陆固结（用户第 2 条「大陆板块过于支离破碎」）----
    # 放在回填内陆水之前：先把窄海峡填成陆、删掉无意义的碎岛，再统一回填封闭水体，
    # 两步都作用于同一份终稿 DEM，顺序不会互相破坏。
    # v76: strait_px 4 → 6、min_island_px 280 → 600。
    # 为什么: 用户第 2 条仍说「大陆板块过于支离破碎」。r=4 只能填 <8px 的海峡，
    #   而 2048 图上 <12px（≈240km）的汊道在 900px 预览里就是一条把大陆切开的
    #   青缝 —— 地球的直布罗陀 13km、白令 85km，这种 240km 的"海峡"没有地理实体。
    #   r=6 填 <12px 水道、删 <600px（≈23 万 km²）碎岛：保留真正的海峡与半岛，
    #   只把 GAN 残留的噪声裂缝并回陆地。
    # v80: strait_px 6 → 7。用户第 3 条「内海被一条极细的浅蓝地峡/海峡与外界
    #   相连，这种形态在自然界极难长期存在」。r=6 填 <12px，实测仍留了 1 处
    #   13~15px（≈260~300km）的细颈；地球最宽的必要海峡（莫桑比克 400km）在
    #   2048 图上也有 20px，13px 的瓶颈没有对应的地理实体。r=7 填 <14px。
    dem_m = consolidate_continents(dem_m, strait_px=7, min_island_px=600,
                                   verbose=True)
    _hi_stat(dem_m, "consolidate")
    # ---- v78/v79: 海岸线尺度分级重塑（用户第 1/2/5 条批评）----
    # 放在固结之后：此时窄海峡/碎岛已合并，重塑作用在干净的陆块上；
    # 之后必须重建陆架剖面（本函数把新海压成 -60~-180m 浅水，没有陆坡）。
    dem_m = sculpt_continental_outline(
        dem_m, lats, np.random.RandomState(seed + 78), verbose=True)
    # v83 实测（已回退）：把 huge_amp 0.16→0.34 想拉出"10³km 级大湾/半岛"，
    #   但这一步的 新陆/新海 只从 +37717/−46402px 变成 +37843/−46441px（0.3%），
    #   出图对比也看不出差别 —— σ≈45px 的场抖动在 σ=2.2r 的基场梯度前抬不起等值线。
    #   真正让海岸"咬"起来的是下面的峡湾水道（本轮挖去陆地 7701→13197px，
    #   海岸 18530→21230px），所以这里保持默认 0.16，不留无效改动。
    # v81: 尺度分层海岸形变（用户第 1/2 条「宏观过度破碎 + 微观极致平滑 + 尺度感丢失」）。
    #   按连通块等效半径 R 求出"这是大洲还是小岛"，位移场只作用在大陆岸段：
    #   大块长出宽阔深湾/宏大半岛，小岛原地不动 —— 分形维度的尺度差异从这里建立。
    #   必须放在 sculpt 之后、建架之前：它改写的是海陆二值边界。
    dem_m = reshape_coast_scale_stratified(
        dem_m, lats, np.random.RandomState(seed + 811), verbose=True)
    # v82: 沿岸峡湾/溺谷（用户 v81「大陆边缘像融化的塑料/矢量曲线」）→ 实际调用点
    #   搬到了 carve_river_mouths 之后（见下方 4700 行附近），原因见函数 docstring。
    dem_m = create_continental_shelf(dem_m, lats, shelf_width_px=6,
                                     outer_depth=-170.0, mcls=_mcls_frozen)
    # v79b: 收掉海岸重塑漏下的 1~7px 单像素孤点（实测 <800px 陆块 4→18 的回归，
    # 其中 14 个是 1~4px 贴岸噪点）。放在陆架剖面之后：先建好剖面再删点，被删像素
    # 直接落在 -18m 陆架带里，不会留下深坑。
    dem_m = despeckle_islets_m(dem_m, min_px=40, verbose=True)
    # v79b: 压掉漂浮在深海里、不贴岸的孤立浅水台地。实测 71 个 / 23857px，
    # 在二值轮廓图上就是用户第 1 条说的"掉落的碎渣"；陆地占比不变（只动海）。
    # v80: 阈值 3000→12000px、贴岸距离 2→6px。用户仍看到"深海散落大量无源浅蓝
    #      斑块"：实测 3000px 门槛下还剩 14 个 3000~20000px 的台地，全图零散。
    #      这些微陆块没有成因，改成全部压深，浅水特征交给下面的海山链/洋中脊。
    dem_m = sink_isolated_shoals(dem_m, max_px=12000, min_dist_px=6.0, verbose=True)
    # v80g: 第二次、更宽的清理 —— 上一步只管 −200~0m 且面积 <12000px 的碎渣。
    #   实测开阔洋里仍有 14 块 2000~21000px、顶 −0~−600m 的**宽阔**无源浅台
    #   （最大 21319px 顶 −0m，离陆 >60px），在色阶上就是深蓝中的浅青斑；
    #   旧 min_target_m=−600 只会把它们压到 −600m，依旧明显偏浅。
    #   这里放到 −1200m 判据、26000px 面积、30px 贴岸距离、目标 −4200m，
    #   只吃"宽阔且远离任何陆地"的那一类，大陆架/陆坡（<30px）不受影响。
    dem_m = sink_isolated_shoals(dem_m, max_px=26000, min_dist_px=30.0,
                                 shallow_max_m=-1200.0, min_target_m=-4200.0,
                                 verbose=True)
    # v81: 河口 / 三角洲 / 河口湾（用户第 3/4 条「找不出大河的入海口与三角洲」）。
    #   必须放在陆架剖面之后：它要在已经建好的 −170~0m 浅海带上向陆下切，
    #   才能形成两侧带浅滩的喇叭形河口湾；放在之前会被剖面的只抬不压抹平。
    dem_m = carve_river_mouths(dem_m, lats, np.random.RandomState(seed + 812),
                               verbose=True)
    # v82: 沿岸峡湾/溺谷（用户 v81「大陆边缘像融化的塑料/矢量曲线」）。
    #   v82 首版放在建架**之前**，实测是错的：create_continental_shelf 会按
    #   "到新岸线的距离" 重铺 −25~−170m 剖面 —— 水道只有 1~3px 宽，距离算出来
    #   就是 1~3px，于是 150 条水道全被抹成 −30~−130m 的**浅水沟**，在图上读作
    #   "海湾"而不是峡湾（比对 /tmp/v82_ab_crops.png 可见浅带只是变毛）。
    #   现与 carve_river_mouths 同层：建完架之后再下切，保留 ~ −320m 的深槽。
    dem_m = carve_coastal_inlets(dem_m, lats, np.random.RandomState(seed + 815),
                                 verbose=True)
    # v83: 海岸单像素针尖/针孔收尾（用户第 2 条）。必须放在所有会改岸线的步骤
    #   （陆架重建 / 河口 / 峡湾）之后、海洋构造之前，否则新开的水道缝尖又会
    #   重新漏出 1px 针尖。
    # v85（用户 v82 第 1 条「海岸线像钢笔工具拉出来的矢量曲线」）：
    #   在收尾削尖之前先把岸带整体做 ±3.2px 的多尺度域扭曲 —— 纯几何变形，
    #   不改剖面的深浅关系，只打断"曲率恒定"的矢量感。必须在 despike 之前，
    #   扭曲产生的 1px 斜接由紧随其后的中值判类抹平。
    dem_m = roughen_coastline(dem_m, lats, np.random.RandomState(seed + 816),
                              verbose=True)
    dem_m = despike_coastline(dem_m, passes=2, verbose=True)
    # ---- v80: 海洋地质构造化（用户第 1/2/5 条批评）----
    # 顺序：先压掉无源浅斑 → 用年龄-深度场重建整个开阔洋基底（洋中脊 +
    # 侧翼 + 深海平原 + 大陆隆）→ 再在**新的**洋底上加海山链/海沟。
    # v80d: 洋中脊必须在海山之前 —— 它现在会重写开阔洋的整个基底，
    # 放在海山之后会把刚放下的锥体抹平。
    dem_m = add_ocean_ridge_system(dem_m, lats, np.random.RandomState(seed + 802),
                                   n_ridges=3, verbose=True)
    dem_m = add_seamount_chains(dem_m, lats, np.random.RandomState(seed + 801),
                                n_chains=3, verbose=True)
    # v81: 岛弧链（日本列岛型）+ 热点岛链（夏威夷型）。用户第 3 条「岛屿的散落
    #   像随机撒下的芝麻，没有形成岛弧或热点岛链」。两者都只在**主动边缘外海**
    #   （岛弧）或**开阔洋深处**（热点）落岛，顺序必须在洋中脊/海山之后 ——
    #   洋中脊会重写开阔洋基底，放在它之前会把岛弧的弧脊抹平。
    dem_m = add_island_arc_chains(dem_m, lats, np.random.RandomState(seed + 813),
                                  mcls=_mcls_frozen, verbose=True)
    dem_m = add_hotspot_island_chain(dem_m, lats,
                                     np.random.RandomState(seed + 814),
                                     n_chains=2, verbose=True)
    dem_m = carve_margin_trenches(dem_m, lats, np.random.RandomState(seed + 803),
                                  verbose=True, mcls=_mcls_frozen)
    _hi_stat(dem_m, "sculpt")
    dem_m = reclaim_inland_water(dem_m, verbose=True)
    _hi_stat(dem_m, "inland_fill")
    # v77: 边缘兜底修复（用户「边缘出现奇怪的竖线」）。
    # 根因已在 consolidate_continents 修掉（binary_closing 缺 border_value=1），
    # 这里再镜像一次外圈，保证任何上游阶段漏出的边框都不会进入渲染/3D 导出。
    dem_m = heal_map_border(dem_m, frame_px=8, verbose=True)
    _g_t0 = time.time()

    # v39: 修复渲染输入陈旧 bug — 此前 render_world 收到的是 v38 山链迁移后的
    # `out`, 缺失末次陆架重建 / 海岸侵蚀 / v34 河道重雕的改动 (dem_m 已更新,
    # out 未同步), 导致出图与 DEM 不一致 (无陆架过渡、无新河道).
    out = m2v(dem_m)
    landS = dem_m > 0
    np.save(out_npy, dem_m.astype(np.float32))
    # v7.8: 构造 tect_ctx 供 render_world 地貌子类分类
    tect_T_full = np.stack([ndi_zoom(tect_T[i], up, order=1)[:size_px, :size_px]
                            for i in range(4)])
    tect_ctx = dict(envS=envS, beltS=beltS, cf=cf,
                    tect_T=tect_T_full, margin_up=margin_up, riftS=riftS)
    plt.imsave(out_png, render_world(out, lats, landS, make_terrain_cmap(),
                                     np.random.RandomState(seed), tect_ctx=tect_ctx))
    # ---- 3D 导出（可选，默认关闭；产物写在 DEM 同目录）----
    # v36+: 放在 render_world 之后，使 --texture auto 能复用刚生成的 world_map.png
    if export_3d:
        from .export_3d import export_world
        export_world(dem_m, os.path.dirname(os.path.abspath(out_npy)),
                     mode=export_3d_mode, decimate=export_3d_decimate)
    print(f"    [4/4 计时] save npy + render_world: {time.time()-_g_t0:.2f}s")
    print(f"    [阶段计时] [4/4] 全局阶段总计: {time.time()-_stage_t0:.2f}s")
    print(f"\n完成: {out_png} + {out_npy} + world_skeleton.png")
    print(f"  陆地 {(dem_m > 0).mean() * 100:.1f}% | 峰 {dem_m.max():.0f}m | 沟 {dem_m.min():.0f}m")

    # ===== v8.0: 地形监督模块审计 + 反思 (生成→审计→反思→学习闭环) =====
    # 审计五维指标 (地形/温度/降水/群系/风带雪盖) → 对比地球参考 → 写学习日志
    # 低风险建议自动写入 overrides.json 供下次生成注入; 群系阈值等高风险仅打印
    if enable_supervisor:
        try:
            from terrain.supervisor import audit_world, reflect_and_learn
            report = audit_world(dem_m, lats, seed, size_px, tect_ctx=tect_ctx)
            reflect_and_learn(report, apply_suggestions=apply_suggestions)
        except Exception as e:
            print(f"[Supervisor] 审计跳过 (不阻断主流程): {e}")
    return dem_m


def main():
    """``generate_world.py`` 命令行入口。

    解析 ``--seed / --size / --ckpt / --tect-res``，调用
    :func:`generate_world` 生成一幅完整世界地图。
    ``python -m terrain.core.generate_world`` 以模块方式直接运行时，
    会经由此入口触发实际流程。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=4096, choices=[2048, 4096])
    ap.add_argument("--ckpt", default=None)   # 默认: 脚本目录内 best_terrain_gan_v5.pth
    ap.add_argument("--tect-res", type=int, default=512)
    ap.add_argument("--export-3d", action="store_true",
                    help="生成后导出 3D 模型 (OBJ+glTF, 平面+球体)")
    ap.add_argument("--export-3d-mode", choices=["plane", "globe", "both"], default="both")
    ap.add_argument("--export-3d-decimate", type=int, default=4)
    a = ap.parse_args()
    generate_world(seed=a.seed, size_px=a.size, ckpt_path=a.ckpt, tect_res=a.tect_res,
                   export_3d=a.export_3d, export_3d_mode=a.export_3d_mode,
                   export_3d_decimate=a.export_3d_decimate)


if __name__ == "__main__":
    main()