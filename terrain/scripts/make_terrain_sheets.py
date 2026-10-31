# -*- coding: utf-8 -*-
"""导出地形三视图：高度图 / 大陆轮廓图 / 等高线图。

用法:
    PYTHONPATH="." .venv/bin/python terrain/scripts/make_terrain_sheets.py

输入: terrain/output/world_dem.npy  (2048×2048 float32, 米制高程)
输出:
    terrain/output/world_heightmap.png  线性归一化灰度（供 3D 采样）
    terrain/output/world_outline.png    海陆 + 海底分层 + 海岸线 + 极地冰
    terrain/output/world_contours.png   陆地 250m / 海洋 1000m 等值线

坐标约定（v79 修正）:
    DEM 的 row0 = 纬度 −90°（南极），row S−1 = +90°（北极）。
    画图时必须 ``flipud`` 才能"北在上"，否则整张图南北倒置。
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import (binary_dilation, binary_erosion,
                           distance_transform_edt as edt, gaussian_filter,
                           label as ndi_label, zoom as ndi_zoom,
                           map_coordinates)
from matplotlib.collections import LineCollection

from terrain.core.terrain_engine import value_fbm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM = os.path.join(ROOT, "output", "world_dem.npy")
DPI = 100


def _fig(size_px):
    """建一个 1:1 像素、无留白的画布（imshow 默认 origin='upper'，row0 在顶部）。"""
    fig = plt.figure(figsize=(size_px / DPI, size_px / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(-0.5, size_px - 0.5)
    ax.set_ylim(size_px - 0.5, -0.5)
    return fig, ax


def _periodic_wave(size_px, rng, kmax=6):
    """沿经度的**周期**扰动（0 均值、幅度≈1），用于冰缘/极冠边界。

    用正弦叠加而不是随机插值，保证 x=0 与 x=S−1 首尾相接、图上无接缝。
    """
    lon = np.arange(size_px) / size_px * 2.0 * np.pi
    w = np.zeros(size_px, np.float64)
    for k in range(1, kmax + 1):
        w += rng.uniform(0.3, 1.0) / k * np.sin(k * lon + rng.uniform(0, 2 * np.pi))
    w -= w.mean()
    return w / (np.abs(w).max() + 1e-9)


def _smoothstep(x, a, b):
    """a→b 的 smoothstep（a<b）。"""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lat_field(size_px):
    """构造 (S, S) 纬度场（北在上，用于冰盖）。"""
    lat = (-90.0 + (np.arange(size_px) + 0.5) * (180.0 / size_px))[::-1]  # 北在上
    return np.repeat(lat.astype(np.float32)[:, None], size_px, axis=1)


def _ice_margin(size_px):
    """极地冰缘的**逐列位移场**（度；+ = 冰舌伸向赤道，− = 崩解楔凹入地面）。

    v90（用户 v89 第 6 条「极地冰盖边缘仍是一条平滑波浪线，没有崩解楔 / 峡湾冰舌」）：
      ``_ice_weight`` 里所有纵向摆动都在 1/8 分辨率上生成（low≈256），再 order=3
      三次插值放大 —— 能表达的最小尺度就是 ~8 全分辨率 px 的平滑起伏，而且
      wob/big/mid/notch 四条都是**正弦/高斯**叠加，等值线必然是圆滑的波浪。
      实测 v89 极区 900px 取景里冰缘只有 1 个峰、2 个谷，最长直线段 116px。
      真实冰盖边缘由**溢出冰川**控制：冰舌向海凸出、冰舌之间被崩解楔（尖 V 形
      湾）咬进去，所以这里在**全分辨率**上直接构造逐列位移：
        ① 三尺度**脊状噪声**（波长 ≈205~512 / 68~146 / 23~46px，幅度
           40 / 32 / 8px）：峰是宽缓冰穹、折点是尖 V 崩解楔，位置与宽窄天然
           不均匀。v90e 曾用 260/95 两级 + 4px 细尺度，实测极区 900px 取景里
           只有 5 个峰、又退回"波浪线"，故把主尺度挪到 68~146px；
        ② 2~4 条深峡湾（窄 V，半宽 8~18px、深 70~140px）+ 3~5 条溢出冰川舌
           （宽 25~55px、高 30~55px）—— 给边缘几个可辨识的地标；
        ③ ±3px 高频周期扰动 —— 崩解碎屑级毛边。
      位移以度返回（1° = size_px/180 px），由 ``_ice_weight`` 沿行重采样。
    """
    rng = np.random.RandomState(1303)
    x = np.arange(size_px, dtype=np.float64)

    def _dx(cx):
        d = np.abs(x - cx)
        return np.minimum(d, size_px - d)          # 经向周期

    def _ridged(k0, k1):
        """1D 脊状噪声：正弦叠加后取 −|w| —— 峰是宽缓的冰穹，折点是尖 V 崩解楔。

        用整数波数（k 个周期跨越整幅）保证 x=0 与 x=S−1 首尾相接，和 _periodic_wave
        同样的周期约定；不用等间距 tent，避免上一版 v90c/d 那种"梳齿"节奏。
        """
        w = np.zeros(size_px, np.float64)
        for k in range(k0, k1 + 1):
            w += (rng.uniform(0.6, 1.0) / np.sqrt(k)
                  * np.sin(2 * np.pi * k * x / size_px + rng.uniform(0, 2 * np.pi)))
        w -= w.mean()
        w /= (np.abs(w).max() + 1e-9)
        r = -np.abs(w)
        r -= r.mean()
        return r / (np.abs(r).max() + 1e-9)

    # 三个尺度：260 / 95 / 32px 波长的脊状起伏，越细幅度越小（冰缘是分形的）
    m = 40.0 * _ridged(4, 10) + 32.0 * _ridged(14, 30) + 8.0 * _ridged(45, 90)
    # 再补 2~4 条**深峡湾**（窄而深的 V，半宽 8~18px、深 70~140px）与
    # 3~5 条**溢出冰川舌**（宽 25~55px、高 30~55px），给边缘几个突出的地标，
    # 否则纯噪声生成的边缘全局"平均"、没有记忆点。
    for _ in range(rng.randint(2, 5)):
        cx, wid, dep = rng.uniform(0, size_px), rng.uniform(8, 18), rng.uniform(70, 140)
        m -= dep * np.maximum(0.0, 1.0 - _dx(cx) / wid)
    for _ in range(rng.randint(3, 6)):
        cx, wid, amp = rng.uniform(0, size_px), rng.uniform(25, 55), rng.uniform(30, 55)
        m += amp * np.exp(-0.5 * (_dx(cx) / wid) ** 2)
    m += 3.0 * _periodic_wave(size_px, rng, kmax=40)
    return (m * 180.0 / size_px).astype(np.float32)


def _ice_weight(size_px, rng, margin=None):
    """南极冰盖权重场 ∈[0,1]（北极为开阔洋，不画海冰）。

    为什么只画南极:
      本世界 +70°~+90° 陆地占比为 0（纯洋面），−90°~−80° 有 22.95% 陆地
      （= 一块南极型大陆）。给北极画一整条海冰会把画面顶部压成一条水平带，
      反而更像"边框"；南极大陆才是真正需要"冰盖化"来解决"被一刀切"的地方。

    v80 改了什么 / 为什么 / 实测:
      旧版在全分辨率上算 ``smoothstep + 0.12·fBm(24px)``，冰缘的起伏尺度只有
      24px（≈470km），再被 0.12 的幅度放大成逐像素抖动 —— 用户第 4 条
      「底部白色与海洋的交界是高频噪声产生的锯齿，像撕纸」。
      真实冰缘由**冰架**主导（数百~上千公里、近乎平直的前缘），尺度远大于 24px。
      现改为在 1/8 分辨率（256²）上生成：±3° 经度周期摆动 + ±2.5° 大尺度
      起伏（fBm σ=6 低分辨率 px ≈ 94 全分辨率 px），再 σ=2.5 低通、order=3
      三次插值放大，最后只保留 0.05 幅度的细噪声。
      实测（2048）：冰缘 1px 锯齿占比（相邻列边界行差 >1px）由 62% → 7%，
      冰缘平均起伏波长由 ≈24px 提升到 ≈340px。
    """
    low = max(64, size_px // 8)
    latl = _lat_field(low)
    wob = _periodic_wave(low, rng, kmax=7) * 3.0          # ±3° 长波摆动
    big = value_fbm(low, 6.0, 3, rng=rng) * 2.5           # ±2.5° 大尺度冰缘起伏
    # v84（用户 v82 第 4 条「冰盖边缘像一块白布盖在地图底部，没有冰架崩解/峡湾」）：
    #   上一版只有 ±3° 长波 + ±2.5° 大尺度，两条都是"整体平滑"的，于是冰缘是
    #   一条**单调的波浪线**。真实南极冰缘是「冰架前缘近乎平直 + 冰舌伸入海 +
    #   崩解湾深深凹入」的锯齿状轮廓。这里补两层：
    #     ① mid: 中尺度（20 低分辨率 px ≈ 78 全分辨率 px）±1.6° 起伏 —— 冰舌/湾；
    #     ② notch: 5~8 个**周期性高斯凹口**（深 3~9°、宽 6~24px）—— 崩解湾，
    #        让冰缘出现真正"咬进去"的深缺口，而不是正弦摆动。
    mid = value_fbm(low, 20.0, 3, rng=np.random.RandomState(915)) * 1.6
    notch = np.zeros(low, np.float64)
    for _ in range(rng.randint(5, 9)):
        cx = rng.uniform(0.0, low)
        wid = rng.uniform(6.0, 24.0)
        dep = rng.uniform(3.0, 9.0)
        dx = np.minimum(np.abs(np.arange(low) - cx), low - np.abs(np.arange(low) - cx))
        notch -= dep * np.exp(-0.5 * (dx / wid) ** 2)
    edge = -(latl + wob + big + mid + notch)
    base = _smoothstep(edge, 62.0, 72.0)
    base = gaussian_filter(base.astype(np.float32), 2.5)
    up = ndi_zoom(base, size_px / low, order=3)[:size_px, :size_px]
    # v84: 细纹理 0.05@18px → 0.07@11px，让冰缘有"崩解碎屑"级别的毛边，
    #   但仍远小于冰舌尺度，不会回到 v79 的"撕纸"感。
    fine = value_fbm(size_px, 11.0, 2, rng=rng)
    ice = up + 0.07 * fine
    if margin is not None:
        # v90: 逐列位移必须落在**纬度**上，不能直接加到权重上 —— 冰盖权重在远离
        #   冰缘的地方恒为 0，直接 +0.4 权重会把整列（从赤道到极点的 2048px）一起
        #   抬进 edge 带，出图上就是一条贯穿全图的竖直白条（实测 v90a）。
        #   正确做法是按位移量沿行重采样：ice(r) ← ice(r + Δpx)，Δpx>0 时阈值
        #   交点北移 = 冰舌向海凸出。
        from scipy.ndimage import map_coordinates
        dpx = (margin * (size_px / 180.0))[None, :]
        _yy = np.arange(size_px, dtype=np.float32)[:, None] + dpx
        _xx = np.broadcast_to(np.arange(size_px, dtype=np.float32)[None, :], _yy.shape)
        ice = map_coordinates(ice, [_yy, _xx], order=1, mode="nearest")
    return np.clip(ice, 0.0, 1.0).astype(np.float32)


def heightmap(dem, out):
    """(1) 高度图：全局 min→0 / max→255 的线性灰度，供 3D 直接采样。

    用线性归一化（不是直方图均衡），因为 3D 需要真实比例：
    0m 海平面落在 −lo/(hi−lo) 处，与 M_KNOTS/V_KNOTS 的 v=0.50 一致到 1 个灰阶内。
    不加极地冰：高度图要保留真实高程，冰盖会污染 3D 采样。
    """
    lo, hi = float(dem.min()), float(dem.max())
    g = np.clip((dem - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
    fig, ax = _fig(dem.shape[0])
    ax.imshow(rgb, interpolation="nearest")
    fig.savefig(out, dpi=DPI, facecolor="black")
    plt.close(fig)
    print(f"  heightmap  {out}  range {lo:.0f}~{hi:.0f}m  "
          f"sea-level gray={-lo/(hi-lo)*255:.0f}")


def outline(dem, out, rng):
    """(2) 大陆轮廓图：海底分层 + 海陆 + 海岸线 + 极地冰盖。

    为什么要分层（用户第 3 条「大陆架和深浅水区无体现」）:
      真实大陆边缘是 陆架(0~−200m) → 陆坡(−200~−2000m) → 深海平原(−2000~−6000m)
      → 海沟；只看二值海陆时，这些结构全被压成一块纯蓝，视觉上像"边缘垂直切断"。
      这里按深度带分区着色，并把 1px 海岸线描黑，轮廓才有"大陆+被动边缘"的层次。
    为什么加极地冰（用户第 4 条「图幅边缘一刀切」）:
      等距圆柱投影的上下边就是南北极，地球地图同样在此"切断"南极洲。
      按纬度画冰盖/海冰（边缘带经度周期扰动）后，上下边读作"极冠"而不是"被切的大陆"。
    """
    land = dem > 0
    coast = land ^ binary_erosion(land)
    S = dem.shape[0]

    # ---- 海底着色：连续渐变 + 800m 等深线 ----
    # v80f 改了什么 / 为什么：
    #   ① 旧版是硬分档（`dem<=-4000` 直接填一个纯色）。实测终稿开阔洋
    #      （离陆 >200px）深度中位 −5192m，而分档线正好在 −5200m —— 洋底
    #      平台上有 53.65% 的像素卡在这条线两侧，±138m 的深海丘陵噪声把它们
    #      来回翻越，出图上整片大洋是均匀的"电视雪花"（第一版 crop_A）。
    #   ② 改为**连续色带插值**（每个 800m 分档给出一个锚点色，像素在锚点间
    #      线性插值），再在分档边界处画 1px **等深线**。这样"分层"仍在
    #      （等深线清晰可分），但不再有量化跳变。
    #   实测：雪花像素占比从 100%（分档跳变处）降到只占 0.6%（等深线本身）。
    # v80h: 等深线由「压暗 18%」改成「提亮 +20」。实测 crop_B（右侧海湾）里
    #   陆坡 10~30px 内挤了 −400/−1200/−2000/−2800 四条线，压暗后它们并成一条
    #   绕岸一圈的**黑描边**——比用户批评的"外发光"更像人工描边。真实海图上
    #   等深线是比底色略亮的细线，故改为加常量提亮（深蓝 +20 ≈ 更浅一档）。
    # v85（用户 v84「海岸外一圈近黑的深蓝描边」）：
    #   旧锚点最深只到 −6000m，而本图海沟到 −10123m —— 于是 −4000m 以下整段被
    #   压进 (10,28,52) 的近黑色，任何离岸 10~20px 的陡坡都读成"人工描边"。
    #   现把色带延伸到 −11000m，并把 −2000~−5000m 的锚点整体提亮约 20%：
    #   陆坡是中等蓝、只有真正的海沟轴才接近墨黑，与真实海图的色阶一致。
    _edges = np.array([-11000, -8500, -7000, -5600, -4400, -3400,
                       -2500, -1700, -1100, -500, -180, 0], np.float32)
    _cols = np.array([[6, 18, 38], [10, 28, 54], [14, 36, 66],
                      [20, 46, 80], [27, 58, 95], [36, 72, 112],
                      [46, 88, 126], [58, 102, 140], [72, 118, 156],
                      [88, 136, 172], [102, 156, 186], [112, 168, 196]],
                     np.float32)
    _mid = _edges                                  # 颜色锚点 = 等深线本身
    rgbf = np.stack([np.interp(dem, _mid, _cols[:, c]) for c in range(3)],
                    axis=-1)
    # v87（用户「深海垂直条纹 / 矩形像素块 / 底色网格线外露」）：
    #   旧版直接对逐像素 dem 做 ``searchsorted`` 求所属色带，在**平坦深海**里
    #   ±100m 的深海丘陵噪声就能让像素在 800m 锚点两侧来回翻越 → 整片洋底布满
    #   5~10px 的闭合小圈（视觉上就是"电视雪花"），而 GAN 瓦片留下的低频块状
    #   起伏又让这些小圈的**疏密按矩形块变化**，于是整块整块的"剥落墙皮"、
    #   甚至沿瓦片边界的十字直线一起被凸显出来。
    #   两条修正：
    #     ① 求色带序号前先 σ=0.7 轻度平滑（只影响判带，不改颜色插值本身），
    #        消掉逐像素翻越；
    #     ② 只在**局部坡度足够大**（σ=1.6 平滑后的 |∇dem| > 18m/px）处画等深线：
    #        真实陆坡/海沟/洋中脊坡度远大于此，平坦深海平原上的伪闭合圈全部消失。
    _dsm = gaussian_filter(dem, 2.0)
    _bi = np.searchsorted(_mid, _dsm).astype(np.int16)
    _iso = np.zeros(dem.shape, bool)
    _iso[1:, :] |= _bi[1:, :] != _bi[:-1, :]
    _iso[:, 1:] |= _bi[:, 1:] != _bi[:, :-1]
    _gy, _gx = np.gradient(_dsm)
    _slope = gaussian_filter(np.hypot(_gx, _gy).astype(np.float32), 1.6)
    _iso &= _slope > 20.0
    # v87b: 深海丘陵是 10~20px 尺度的起伏，跨 800m 锚点时必然生成**孤立的
    #   小闭合圈**。坡度门只能删掉最平的那部分，剩下的在 900px 缩览图上仍然
    #   是满屏"网点"。真实等深线是**长程连续的线**，故再按连通域面积过滤：
    #   面积 <60px 的碎圈全部丢弃（实测这一步才真正消掉网点纹理）。
    _lab, _nl = ndi_label(_iso)
    _sz = np.bincount(_lab.ravel())
    _iso &= _sz[_lab] >= 60
    rgbf[_iso & ~land] += 20.0                            # 等深线（提亮）
    # v89: 浅海浮雕（shaded relief）—— 修「浅海像一坨晕开的颜料」。
    #   色带锚点每 500~800m 一个，而整个陆架落差只有 ~180m，所以 DEM 里无论铺了
    #   多少沙脊/海底峡谷，颜色都几乎不变，浅海就是一块平涂的薄蓝。真实海图靠
    #   **浮雕**表达浅海底形：对 σ=1.6 平滑后的高程按西北光源求明暗，只作用于
    #   −900~0m（陆架/陆坡，深海不碰，避免把 v87 刚压掉的"电视雪花"翻出来），
    #   离岸 60m 内渐入以避免给海岸线描一道暗边。
    _shy, _shx = np.gradient(gaussian_filter(dem, 1.6))
    _rel = np.clip((_shx * (-0.7071) + _shy * (-0.7071)) / 55.0, -1.0, 1.0)
    _wsh = (np.clip((dem + 900.0) / 400.0, 0.0, 1.0)
            * np.clip(-dem / 60.0, 0.0, 1.0) * (~land))
    rgbf *= (1.0 + 0.20 * _rel * _wsh)[..., None]
    rgb = rgbf.astype(np.uint8)
    rgb[land] = (232, 226, 212)                          # 陆地

    _ice_mg = _ice_margin(dem.shape[0])
    ice = _ice_weight(dem.shape[0], rng, margin=_ice_mg)
    # v80 冰盖分层（用户第 4 条：边缘「撕纸」+ 缺冰架/浮冰过渡带）
    #   ① 冰盖 solid：只画陆地且冰权重 >0.58，边界来自 1/8 分辨率的平滑场；
    #   ② 冰盖前缘 edge：陆地上 0.34~0.58 的过渡带，单独一档浅灰蓝，避免
    #      "白/米色" 在 1px 内硬切；
    #   ③ 冰架 shelf：用 1/4 分辨率陆地掩膜做 σ=9（≈36 全分辨率 px）高斯扩散，
    #      得到完全脱离海岸细碎形状的**平滑外缘**，只在海侧、纬度 <−63°、冰权重
    #      >0.55 处铺白 —— 这就是南极罗斯/龙尼冰架那种大片平直冰台；
    #   ④ 浮冰群 floes：冰架之外再用 1/4 分辨率 fBm 打碎成斑块，限制在离岸
    #      <110px、冰权重 >0.45，形成"海冰—浮冰—开阔洋"的渐变过渡。
    S = dem.shape[0]
    d_sea = np.where(land, 0.0, edt(~land)).astype(np.float32)
    # v87（用户「极地冰盖边缘是一条平滑的波浪线，没有冰架崩解 / 峡湾冰川质感」）：
    #   ``_ice_weight`` 是 1/8 分辨率上算的纬度场，起伏尺度 ≈340px，只有 0.07 的
    #   细噪声 —— 阈值 0.58 切出来的陆冰边缘必然是一条大尺度波浪线，读作"白布"。
    #   真实冰盖边缘由**溢出冰川 / 峡湾**控制：几十公里的冰舌伸入海、其间是峡湾。
    #   这里把阈值改成"冰权重 + 双尺度位移场"：σ≈110px 的中频给它 5~15px 的
    #   冰舌/湾，σ≈15px 的高频补 2~5px 的锯齿。阈值本身不动，只让等值线变糙。
    _fj_mid = value_fbm(S, 9.0, 3, rng=np.random.RandomState(1301))
    _fj_fin = value_fbm(S, 60.0, 2, rng=np.random.RandomState(1302))
    ice_eff = np.clip(ice + 0.100 * _fj_mid + 0.050 * _fj_fin, 0.0, 1.0)
    ice_land = ice_eff * land.astype(np.float32)
    solid = ice_land > 0.58
    edge = land & ~solid & (ice_land > 0.50)
    low4 = S // 4
    # v87（用户「极地冰盖边缘像一块白布」—— 上一版修了 floes 仍有一大块白）：
    #   v85/v86 的 shelf 主体是 ``gaussian_filter(land, 36px) > 0.40``。在地图
    #   最南端（南极大陆横贯整幅底边）这个模糊场沿纬度方向衰减极慢，阈值一过
    #   就是一条**几百 px 宽、边缘近乎直线**的白色色块 —— 实测 /tmp/c_ice.png
    #   里那块"白布"左上角是笔直的斜边、上沿是水平线，正是模糊场等值线的形状。
    #   真实冰架前缘不由"离岸模糊值"决定，而由**离岸距离 + 崩解**决定：
    #   罗斯/龙尼冰架的外缘离岸 400~800km（≈20~40px），前缘是一条被崩解湾
    #   咬出缺口的弧线。故改用 d_sea 直接定义前缘：
    #     score = 1.10·(1 − d_sea/60) + 0.16·nz_mid + 0.09·nz_fin
    #             + 2.0·bays + 0.10·nz_big
    #   d_sea=0 时 score≈1.10，d_sea=60px 时 score≈0；阈值 0.70 → 基准前缘在
    #   离岸 ~21px 处，再被中/高频噪声（±0.25）啃成锯齿、被 bays（±0.4）咬出
    #   深缺口，得到真正的"冰架前缘"而不是一块模糊白斑。范围硬限 d_sea<60。
    # v85（用户 v84「冰盖/冰架边缘仍是一块白布，没有崩解与峡湾」）：
    #   旧版 shelf_f*(1+0.18·nz)>0.30 —— 单一 340px 尺度的低幅摆动，阈值又低，
    #   结果冰架外缘依旧是一条平滑的等距线。现补三个层次：
    #     ① 三尺度噪声（340 / 110 / 45px）直接把前缘"啃"成不规则锯齿；
    #     ② 6~9 个周期性**崩塌湾**（宽 20~70px、深 0.10~0.22），让冰缘有真正
    #        咬进去的深缺口，而不是正弦摆动；
    #     ③ 阈值 0.30→0.40，配合噪声正负两支还能长出几处伸向海的**冰舌**。
    nz_big = ndi_zoom(value_fbm(low4, 3.0, 3, rng=np.random.RandomState(77)),
                      4.0, order=3)[:S, :S]
    nz_mid = ndi_zoom(value_fbm(low4, 9.0, 3, rng=np.random.RandomState(78)),
                      4.0, order=3)[:S, :S]
    nz_fin = ndi_zoom(value_fbm(low4, 22.0, 2, rng=np.random.RandomState(79)),
                      4.0, order=3)[:S, :S]
    _col = np.arange(low4)
    _xb = np.zeros(low4, np.float64)
    _rng = np.random.RandomState(81)
    for _ in range(int(_rng.randint(6, 10))):
        _cx = _rng.uniform(0.0, low4)
        _wid = _rng.uniform(5.0, 18.0)
        # 正负混合：负 = 崩解湾（冰缘后退），正 = 冰舌（伸入海）
        _dep = _rng.uniform(0.10, 0.22) * _rng.choice([-1.0, 1.0])
        _dx = np.minimum(np.abs(_col - _cx), low4 - np.abs(_col - _cx))
        _xb += _dep * np.exp(-0.5 * (_dx / _wid) ** 2)
    bays = ndi_zoom(np.repeat(_xb[None, :], low4, axis=0), 4.0, order=1)[:S, :S]
    lat_f = _lat_field(S)
    shelf_edge = (1.10 * np.clip(1.0 - d_sea / 60.0, 0.0, 1.0)
                  + 0.10 * nz_big + 0.16 * nz_mid + 0.09 * nz_fin
                  + 2.0 * bays)
    shelf = ((~land) & (lat_f < -63.0) & (ice_eff > 0.42) & (d_sea < 60.0)
             & (shelf_edge > 0.70))
    # v86（用户「极地冰盖边缘像一块白布」延伸出的更硬的问题）：
    #   旧版 floe_f = blur(fBm(2.2 cycles) > 1.05, σ1.1) —— 低频（波长 ≈930px）
    #   阈值化后被 σ 抹圆，得到的是**几块 200~250px 的实心大饼**，其中一块横在
    #   南纬 60° 的开阔洋面上，边缘还是被 ice>0.45 等值线切出来的直线，读作
    #   "漂浮的奶油块"（/tmp/ab86_wedge.png）。真实浮冰是**密集的小碎块**。
    #   现改为「大尺度密集区 × 小碎块」两层：5 cycles 的低频场决定"哪里有冰群"，
    #   140 cycles（波长 ≈15px）的高频场把它打成一粒粒；阈值 0.35 让碎块之间
    #   留出海面，不再连成实心片。范围也收到离岸 70px 内（真实海冰贴着冰架）。
    _dens = ndi_zoom(value_fbm(low4, 5.0, 2, rng=np.random.RandomState(1043)),
                     4.0, order=3)[:S, :S]
    _spec = value_fbm(S, 140.0, 2, rng=np.random.RandomState(1042))
    floes = ((~land) & ~shelf & (ice_eff > 0.45) & (d_sea < 70.0)
             & (_dens > 0.10) & (_spec > 0.35))

    rgb[solid] = (247, 250, 253)                         # 冰盖
    rgb[edge] = (231, 238, 245)                          # 冰盖前缘过渡
    # v87: 冰架与冰盖原来只差 11 个灰阶（247,250,253 vs 236,245,251），出图上
    #   两者糊成**一整块白**，用户读作"一块白布"。改为明显偏蓝的冰台色，并在
    #   前缘加 1px 深色描边，冰架才有"浮在海上的冰台"的分离感。
    rgb[shelf] = (223, 236, 247)                         # 冰架
    rgb[floes] = (196, 219, 236)                         # 浮冰群
    _front = binary_dilation(shelf) & ~shelf & ~land     # 冰架前缘（崩解线）
    rgb[_front] = (170, 199, 224)
    # v87: 冰盖内部原本是纯平色块（一张白纸）。用 σ≈230px 的低频场做 ±4% 明暗
    #   调制，读作"冰面起伏 / 冰流带"，而不是一块布。
    _tex = (1.0 + 0.040 * _fj_mid).astype(np.float32)
    _im = solid | shelf | edge
    rgb[_im] = np.clip(rgb[_im].astype(np.float32) * _tex[_im][:, None],
                       0.0, 255.0).astype(np.uint8)
    # v90: 冰盖内部"冰流线"—— 空白一片的主因是整块纯色。生成 1D 低频条纹后按行
    #   横向摆动（±26px、周期 ≈440px）再铺满，条纹顺着冰川流向弯而不是一张竖条码；
    #   幅度收到 1.5% 并用 nz_mid 做包络，只在部分区段显出来（flow stripe 本就是
    #   断续的）。v90g 的 2.5% / σ=2.2 在近白色上仍读作"窗帘"。
    _cl = gaussian_filter(np.random.RandomState(1304).randn(S).astype(np.float32), 7.0)
    _cl = np.clip(_cl / (_cl.std() + 1e-8), -2.2, 2.2)
    _rowi = np.arange(S, dtype=np.float32)[:, None]
    _warp = (np.arange(S)[None, :] + (26.0 * np.sin(_rowi / 70.0))).astype(np.int64) % S
    _streak = _cl[_warp] * (0.40 + 0.60 * nz_mid)
    rgb[_im] = np.clip(rgb[_im].astype(np.float32)
                       * (1.0 + 0.015 * _streak[_im])[:, None],
                       0.0, 255.0).astype(np.uint8)
    # v90: 冰缘内侧 45px 的蓝冰带（消融区）+ 零星冰原岛峰（裸岩）。
    #   蓝冰带让"白"到"米"之间多一层过渡，岛峰是真实冰盖边缘最抓眼的地标。
    _din = edt(solid).astype(np.float32)
    _ring = solid & (_din < 45.0)
    _wr = (np.clip(1.0 - _din[_ring] / 45.0, 0.0, 1.0) * 0.50)[:, None]
    rgb[_ring] = np.clip(rgb[_ring].astype(np.float32) * (1.0 - _wr)
                         + np.array([214, 231, 244], np.float32) * _wr,
                         0.0, 255.0).astype(np.uint8)
    _nun = solid & (_din < 90.0) & (_fj_fin > 1.95)
    rgb[_nun] = (203, 197, 185)
    # v90: 冰盖外缘加 1px 冰崖线。旧版 solid→edge→陆地 三层都是接近白的浅色，
    #   V 形崩解楔被 24px 宽的 edge 带填平，只剩"啃过的纸边"；edge 收到 0.50 后
    #   仍需要一条明确的冰崖边界，否则白色与米色在噪声里渐变。
    _cliff = binary_dilation(solid) & ~solid & ~shelf & ~floes & ~edge
    rgb[_cliff] = (176, 198, 218)
    # v91（P2-2：海岸线从"一根等宽黑描边"改成按海岸类型分带）
    #   旧版整条岸线都是 ``coast`` 上的 1px (12,24,38) —— 一条用笔画出来的等宽
    #   描边，既是最强的"人工感"来源，也抹掉了海图上最直观的岸线差异：低平
    #   沉积海岸是沙滩，山地直插海里的是岩岸/悬崖。判据只用**岸后陆地的
    #   高程与坡度**（σ=2 平滑后），不引入新的随机场，保证可复现：
    #     · 沙滩：岸后 dem<60m 且 |∇dem|<8 m/px → 内侧 2px 沙色带，
    #       岸线本身也换成浅沙色（黑线会把沙滩切成"贴了边的纸片"）；
    #     · 岩岸：岸后 dem>250m 或 |∇dem|>22 m/px → 岩灰岸线 + 内侧 2px 峭壁暗面；
    #     · 其余（缓丘/中等海岸）→ 保留 1px 深色线。
    #   注意 1px ≈ 19.6km：这里的"沙滩"是地图符号（真实滩宽在这个比例尺下
    #   不足 1 个像素），作用是把"低平海岸"和"山地海岸"在视觉上分开。
    _dland = edt(land).astype(np.float32)                # 陆地像素到海的距离(px)
    # v91b: 坡度必须**只在陆侧**求 —— dem 含海底（岸外立刻是 −2000m 级），
    #   直接 gaussian_filter(dem) 会让每个岸点都跨着"海平面断层"，∇ 恒为
    #   数千 m/px，实测 86% 岸线被判成岩岸（阈值 22）。改为归一化陆地卷积：
    #   σ=2 内只用陆地像素求均值，岸点得到的是"岸后高程"而不是海陆落差。
    _den = gaussian_filter(land.astype(np.float32), 2.0)
    _num = gaussian_filter(np.where(land, dem, 0.0).astype(np.float32), 2.0)
    _demg = np.where(_den > 1e-3, _num / np.maximum(_den, 1e-3), 0.0)
    _gy, _gx = np.gradient(_demg)
    _grd = np.hypot(_gx, _gy) * land                     # 海上梯度无意义，置 0
    _lowflat = (_demg < 60.0) & (_grd < 8.0)
    _rockish = (_demg > 250.0) | (_grd > 22.0)
    _sandline = coast & _lowflat & ~solid & ~edge & ~_cliff
    _sand = (land & (_dland > 1.5) & (_dland <= 4.5) & _lowflat
             & ~solid & ~edge & ~_cliff)
    rgb[_sand] = (230, 207, 163)                         # 沙滩带
    rgb[_sandline] = (200, 178, 136)                     # 沙质岸线
    _rock = coast & _rockish & ~solid & ~edge & ~_cliff
    _rk = (binary_dilation(_rock, iterations=2) & land & ~_sand
           & ~solid & ~edge & ~_cliff)
    rgb[_rk & ~_rock] = (128, 122, 110)                  # 峭壁暗面（岸线本身不盖）
    rgb[_rock] = (62, 60, 56)                            # 岩岸岸线
    rgb[coast & ~solid & ~edge & ~_sandline & ~_rock] = (12, 24, 38)  # 一般海岸线

    fig, ax = _fig(dem.shape[0])
    ax.imshow(rgb, interpolation="nearest")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    shelf_pct = float(((dem > -200) & ~land).mean()) * 100
    _nc = max(1, int(np.count_nonzero(coast & ~solid & ~edge)))
    print(f"  outline    {out}  land {land.mean()*100:.2f}%  coast {int(coast.sum())}px  "
          f"shelf {shelf_pct:.1f}%  ice {ice_land.mean()*100:.2f}%  "
          f"iceshelf {shelf.mean()*100:.3f}%  floes {floes.mean()*100:.3f}%  "
          f"岸线[沙 {_sandline.sum()/_nc*100:.0f}% 岩 {_rock.sum()/_nc*100:.0f}%] "
          f"沙带 {int(_sand.sum())}px")


def contours(dem, out, rng):
    """(3) 等高线图：陆地 250m 一档、海洋非等距 5 档（见下方 A2 说明），极区淡冰底。

    陆地间距 250m 是因为平原/丘陵（0~1000m）占陆地一半以上，500m 一档会把
    大片平原画成空白；海洋改非等距（700/1500/2600/4200/6000m）后，陆架坡折带
    只剩 1 条线，不再挤成与岸线平行的等距彩带；深海仍保留 4 档分辨。
    """
    land = dem > 0
    lev_land = np.arange(250, 9000, 250)
    # A2: 海洋等深线改为**非等距** + 近岸 350m 内不画线。
    # 旧版 1000m 一档，在陆架坡折（200~2000m）上只有两条线，且整条线沿理论等深
    # 面与岸线严格平行 —— 900px 缩略图读作"一圈圈等距彩带"。非等距后坡折处线距
    # 拉开，深海段仍保留 1000m 级分辨；0~350m 的浅水区不再描边。
    lev_sea = np.array([-6000, -4200, -2600, -1500, -700], np.float64)
    # A3: 海洋等深线场做 ±6px 低频域扭曲。等深线"等距"的根因是海底坡面在径向上
    # 均匀（ocean_margin_fix 的剖面），仅改档位无法消除平行感；扭曲让等深线随
    # 低频起伏前后摆动，同一条线的离岸距离因此有宽有窄。
    _S = dem.shape[0]
    _yy, _xx = np.mgrid[0:_S, 0:_S].astype(np.float32)
    _wx = 6.0 * np.clip(value_fbm(_S, 34.0, 3, rng=np.random.RandomState(2027)),
                        -2.0, 2.0)
    _wy = 6.0 * np.clip(value_fbm(_S, 41.0, 3, rng=np.random.RandomState(2028)),
                        -2.0, 2.0)
    _sea_f = gaussian_filter(dem, 1.6)
    _sea_f = map_coordinates(_sea_f, [_yy + _wy, _xx + _wx],
                             order=1, mode="nearest")
    fig, ax = _fig(dem.shape[0])
    ax.imshow(land.astype(np.uint8), cmap=ListedColormap([(0.87, 0.90, 0.94),
                                                          (0.97, 0.95, 0.90)]),
              interpolation="nearest", vmin=0, vmax=1)
    # v87: 海洋等值线同样吃"深海丘陵跨 1000m 档位"的亏 —— 原始 dem 直接 contour
    #   会在洋底画满 5~10px 的碎圈。先 σ=1.6 平滑再抽海洋线（陆地线保持原始
    #   dem，避免岸线被平滑移位）。
    _cs = ax.contour(_sea_f, levels=lev_sea,
                     colors=[(0.45, 0.56, 0.70)],
                     linewidths=0.30, alpha=0.85)
    # A3: 闭合小环（面积 < 300px²）直接丢弃 —— 深海丘陵每个圆顶都会闭合成一个
    # 5~10px 的"马蹄"，300px² 相当于直径 ~20px，保留的是真正的盆地/海山海沟。
    _segs, _dropped = [], 0
    for _lv in _cs.allsegs:
        for _seg in _lv:
            if _seg.shape[0] < 4:
                _dropped += 1
                continue
            if np.allclose(_seg[0], _seg[-1]):
                _x, _y = _seg[:, 0], _seg[:, 1]
                _area = 0.5 * abs(np.dot(_x, np.roll(_y, -1))
                                  - np.dot(_y, np.roll(_x, -1)))
                if _area < 300.0:
                    _dropped += 1
                    continue
            _segs.append(_seg)
    _cs.remove()
    if _segs:
        ax.add_collection(LineCollection(_segs, colors=[(0.45, 0.56, 0.70)],
                                         linewidths=0.30, alpha=0.85))
    ax.contour(dem, levels=lev_land, colors=[(0.45, 0.33, 0.22)],
               linewidths=0.32)
    ice = _ice_weight(dem.shape[0], rng, margin=_ice_margin(dem.shape[0]))
    ov = np.zeros(dem.shape + (4,), np.float32)
    ov[..., :3] = 0.92
    # 同 outline：只给陆地铺淡冰，否则整幅底边会被压成一条白带。
    # v80: 用平滑后的 ice_land 与更宽的过渡带，避免等高线上出现"撕纸"白边。
    ice_land = ice * land.astype(np.float32)
    ov[..., 3] = np.clip((ice_land - 0.30) * 1.9, 0.0, 0.55)
    ax.imshow(ov, interpolation="nearest")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  contours   {out}  land {len(lev_land)} levels / sea {len(lev_sea)} levels"
          f" (小环丢弃 {_dropped} 条, 保留 {len(_segs)} 条)")


def main():
    """读 DEM（翻转成北在上），依次产出三张图。"""
    dem = np.load(DEM).astype(np.float32)
    dem = np.flipud(dem)                      # row0=南极 → 画面北在上
    print(f"DEM {dem.shape} {dem.dtype}  land {(dem > 0).mean()*100:.2f}%  "
          f"range {dem.min():.0f}~{dem.max():.0f}m")
    rng = np.random.RandomState(2026)
    od = os.path.dirname(DEM)
    heightmap(dem, os.path.join(od, "world_heightmap.png"))
    outline(dem, os.path.join(od, "world_outline.png"), rng)
    contours(dem, os.path.join(od, "world_contours.png"), np.random.RandomState(2026))


if __name__ == "__main__":
    main()
