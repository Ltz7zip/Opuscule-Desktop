# -*- coding: utf-8 -*-
"""engine — Terrain Lite 的算法权威实现（numpy / CPU，无 torch）。

Lite 不做全球模拟：它把 Pro 预计算的 SeedPack 当作骨架，在设备上只生成
“最后 10% 的局部细节”。流程（对应规格 §13.2 / §15.2）：

    base（双线性采样 v01 → 米制）
      + ridged fBm 细节（按坡度、海岸距离调制，|detail| 在海岸衰减到 0）
      + 海岸亚像素带符号距离（等值线一阶近似 + 低频扰动 → 抗锯齿岸线、
        沙滩带、窄浅水带；<10px² 的噪声小岛按海处理）
      + 2–3 轮轻侵蚀（坡度限制：超阈值的坡降搬给邻居）
      + 河网叠加（读 pack.hydro 折线，宽度 1+2*level）
      + 气候 LUT / 9 群系（pack 有 T/P/biome 就直接采样）
      → 风格着色（hilshade + 分层设色）

确定性：所有随机性由 ``np.random.RandomState(pack.seed ^ hash(params))`` 决定，
同参数两次渲染 bitwise 一致。内存随 ``size`` 线性（2048² 约 16MB/场）。
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from scipy.ndimage import (distance_transform_edt, gaussian_filter, label,
                           median_filter)

from ..pack import SeedPack, v2m
from ..pack.schema import default_style, m2v

# ==================== 1. 参数 ====================
QUALITY = {
    # octaves：fBm 层数；erosion：轻侵蚀轮数；amp：细节振幅（米）
    "fast": {"octaves": 3, "erosion": 2, "amp": 700.0},
    "balanced": {"octaves": 5, "erosion": 2, "amp": 900.0},
    "high": {"octaves": 6, "erosion": 3, "amp": 1100.0},
}
SEA_LEVEL = 0.5           # v01 海平面（与 pack 的 m2v(0) 一致）
COAST_FADE_PX = 8.0       # 海岸细节衰减带（px，按 size/512 缩放）
SLOPE_REF = 2000.0        # 坡度归一化参考（m/px @512），随 size 反比缩放
SDF_RADIUS = 24           # 海岸过渡带（细节衰减）宽度上限（px）


class LiteParams:
    """一次渲染的参数。``style`` 缺省取 pack 自带的 style.json。

    ``time_myr`` 为 None 时渲染终态基场；pack 含时间切片时给出 Myr 值，
    Lite 会在相邻两帧之间线性插值（Plan v13 §4.2 / §5.2）。
    """

    def __init__(self, size=512, zoom=1.0, cx=0.5, cy=0.5,
                 quality="balanced", style=None, time_myr=None):
        self.size = int(size)
        if not 64 <= self.size <= 4096:
            raise ValueError(f"size 应在 64..4096，实际 {size}")
        self.zoom = float(zoom)
        if not 0.05 <= self.zoom <= 64.0:
            raise ValueError(f"zoom 应在 0.05..64，实际 {zoom}")
        self.cx = float(np.clip(cx, 0.0, 1.0))
        self.cy = float(np.clip(cy, 0.0, 1.0))
        if quality not in QUALITY:
            raise ValueError(f"quality 应为 {sorted(QUALITY)}，实际 {quality!r}")
        self.quality = quality
        self.style = None if style is None else dict(style)
        if time_myr is not None:
            time_myr = float(time_myr)
            if not 0.0 <= time_myr <= 10000.0:
                raise ValueError(f"time_myr 应在 0..10000，实际 {time_myr}")
        self.time_myr = time_myr

    def to_dict(self):
        d = {"size": self.size, "zoom": round(self.zoom, 6),
             "cx": round(self.cx, 6), "cy": round(self.cy, 6),
             "quality": self.quality}
        if self.time_myr is not None:
            d["time_myr"] = round(self.time_myr, 3)
        return d

    def digest(self):
        """参数指纹：决定随机流，保证同参数渲染 bitwise 一致。

        ``time_myr`` 不参与指纹：拖动时间滑杆时细节噪声保持不变，只插值基场。
        """
        d = self.to_dict()
        d.pop("time_myr", None)
        s = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8],
                              "little")

    def __repr__(self):
        d = " ".join(f"{k}={v}" for k, v in self.to_dict().items())
        return f"<LiteParams {d}>"


# ==================== 2. 采样工具 ====================
def _axis(count, zoom, center):
    """输出像素索引 → 归一化世界坐标 [0,1]（0.5 在中心，zoom=1 铺满全球）。"""
    t = (np.arange(count, dtype=np.float32) + 0.5) / float(count) - 0.5
    return (center + t / zoom).astype(np.float32)


def _bilinear(field, wx, wy):
    """在 field（H×W，行=y，列=x）上按归一化坐标双线性采样。"""
    h, w = field.shape
    fx = np.clip(wx * w - 0.5, 0.0, w - 1.0).astype(np.float32)
    fy = np.clip(wy * h - 0.5, 0.0, h - 1.0).astype(np.float32)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = (fx - x0)[None, :]
    ty = (fy - y0)[:, None]
    a = field[np.ix_(y0, x0)]
    b = field[np.ix_(y0, x1)]
    c = field[np.ix_(y1, x0)]
    d = field[np.ix_(y1, x1)]
    return ((a * (1 - tx) + b * tx) * (1 - ty)
            + (c * (1 - tx) + d * tx) * ty).astype(np.float32)


def _nearest(field, wx, wy):
    h, w = field.shape
    ix = np.clip(np.rint(wx * w - 0.5).astype(np.int32), 0, w - 1)
    iy = np.clip(np.rint(wy * h - 0.5).astype(np.int32), 0, h - 1)
    return field[np.ix_(iy, ix)]


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _value_noise(size, freq, rng, offset=0.0):
    """freq×freq 随机值栅格上的平滑双线性插值，返回 [0,1] 噪声。"""
    g = rng.rand(freq + 2, freq + 2).astype(np.float32)
    t = (np.arange(size, dtype=np.float32) + 0.5) * (freq / size) + offset
    i0 = np.floor(t).astype(np.int32)
    f = _smoothstep(t - i0)
    i0 = np.clip(i0, 0, freq)
    i1 = i0 + 1
    fy = f[:, None]
    fx = f[None, :]
    v00 = g[np.ix_(i0, i0)]
    v01 = g[np.ix_(i0, i1)]
    v10 = g[np.ix_(i1, i0)]
    v11 = g[np.ix_(i1, i1)]
    return (((v00 * (1 - fx) + v01 * fx) * (1 - fy))
            + ((v10 * (1 - fx) + v11 * fx) * fy)).astype(np.float32)


def _ridged_fbm(size, octaves, rng, freq0=12.0, lacunarity=2.03, gain=0.55):
    """多尺度 ridged 噪声：每层 1-|2n-1|，归一化到 [0,1]。"""
    total = np.zeros((size, size), np.float32)
    amp, norm, freq = 1.0, 0.0, freq0
    for k in range(octaves):
        n = _value_noise(size, int(min(256, max(2, round(freq)))), rng)
        r = 1.0 - np.abs(2.0 * n - 1.0)
        total += amp * r
        norm += amp
        amp *= gain
        freq *= lacunarity
    return (total / max(norm, 1e-6)).astype(np.float32)


# ==================== 3. 样式 / 群系 ====================
def _hex(c):
    c = str(c).lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def _style_colors(style):
    """把 style.json 解析成可用的调色板（缺字段回落到 default_style）。"""
    st = default_style()
    if style:
        st.update({k: v for k, v in style.items() if k != "palette"})
    pal = dict(default_style()["palette"])
    pal.update((style or {}).get("palette") or {})
    names = (style or {}).get("biome_colors") or st["biome_colors"]
    names = list(names)[:9]
    names += default_style()["biome_colors"][len(names):]
    hs = dict(default_style()["hillshade"])
    hs.update((style or {}).get("hillshade") or {})
    return {
        "palette": {k: _hex(v) for k, v in pal.items()},
        "biome": np.stack([_hex(c) for c in names]).astype(np.float32),
        "hillshade": hs,
    }


def _biome_lut(temperature, precipitation):
    """T(℃)/P(mm) → 9 群系（与 default_style 的 biome_colors 顺序一致）。"""
    t = np.asarray(temperature, np.float32)
    p = np.asarray(precipitation, np.float32)
    b = np.full(t.shape, 2, np.uint8)                     # 温带草原
    cold = t < -5.0
    sub = (t >= -5.0) & (t < 8.0)
    mild = (t >= 8.0) & (t < 20.0)
    warm = t >= 20.0
    b[cold] = 0
    b[sub & (p < 250.0)] = 0
    b[sub & (p >= 250.0)] = 1
    b[mild & (p < 350.0)] = 2
    b[mild & (p >= 350.0) & (p < 900.0)] = 3
    b[mild & (p >= 900.0)] = 4
    b[warm & (p < 300.0)] = 5
    b[warm & (p >= 300.0) & (p < 900.0)] = 6
    b[warm & (p >= 900.0) & (p < 1800.0)] = 7
    b[warm & (p >= 1800.0)] = 8
    return b


# ==================== 4. 河网栅格化 ====================
def _stamp_disc(mask, py, px, radius):
    """在布尔图上盖一个半径约 radius 的圆点（超采样 3×3 采样保证小半径可见）。"""
    h, w = mask.shape
    r = max(0.5, float(radius))
    y0 = max(0, int(np.floor(py - r)))
    y1 = min(h - 1, int(np.ceil(py + r)))
    x0 = max(0, int(np.floor(px - r)))
    x1 = min(w - 1, int(np.ceil(px + r)))
    if x1 < x0 or y1 < y0:
        return
    yy = np.arange(y0, y1 + 1, dtype=np.float32)[:, None] + 0.5
    xx = np.arange(x0, x1 + 1, dtype=np.float32)[None, :] + 0.5
    if r <= 0.7:
        mask[y0:y1 + 1, x0:x1 + 1] |= ((yy - py) ** 2 + (xx - px) ** 2) <= 0.25
    else:
        mask[y0:y1 + 1, x0:x1 + 1] |= ((yy - py) ** 2 + (xx - px) ** 2) <= r * r


def _fill_polygon_aa(ys, xs, out_size, ss=4):
    """闭合折线 → 抗锯齿填充覆盖率（ss×ss 超采样后盒式降采样）。

    直接用圆点盖章再 ``binary_fill_holes`` 会把大湖画成扇贝/直角色块；
    扫描线在超采样网格上填充能给出真实多边形边界，降采样后自带 1/ss px
    级抗锯齿。
    """
    alpha = np.zeros((out_size, out_size), np.float32)
    if len(ys) < 3:
        return alpha
    y0 = max(0, int(np.floor(ys.min())) - 1)
    y1 = min(out_size, int(np.ceil(ys.max())) + 2)
    x0 = max(0, int(np.floor(xs.min())) - 1)
    x1 = min(out_size, int(np.ceil(xs.max())) + 2)
    if y1 <= y0 or x1 <= x0:
        return alpha
    h, w = y1 - y0, x1 - x0
    # 闭合折线：显式补上首尾边，避免最后一个顶点与第一个之间缺一条边
    ay = np.append((ys - y0) * ss, (ys[0] - y0) * ss)
    ax = np.append((xs - x0) * ss, (xs[0] - x0) * ss)
    ey0, ey1 = ay[:-1], ay[1:]
    ex0, ex1 = ax[:-1], ax[1:]
    ey_lo = np.minimum(ey0, ey1)
    ey_hi = np.maximum(ey0, ey1)
    edy = np.where(ey1 == ey0, 1e-9, ey1 - ey0)
    m = np.zeros((h * ss, w * ss), bool)
    for r in range(h * ss):
        yc = r + 0.5
        hit = (ey_lo <= yc) & (yc < ey_hi)
        if not hit.any():
            continue
        xi = np.sort(ex0[hit] + (yc - ey0[hit])
                     * (ex1[hit] - ex0[hit]) / edy[hit])
        for a, b in zip(xi[0::2], xi[1::2]):
            ia = max(0, int(np.ceil(a - 0.5)))
            ib = min(w * ss, int(np.ceil(b - 0.5)))
            if ib > ia:
                m[r, ia:ib] = True
    cov = m.reshape(h, ss, w, ss).mean(axis=(1, 3)).astype(np.float32)
    alpha[y0:y1, x0:x1] = cov
    return alpha


def _raster_hydro(hydro, size, zoom, cx, cy, out_size):
    """把 pack 折线（[y,x]，pack 像素坐标）栅格到输出分辨率。

    返回 ``(mask, lake_a)``：``mask`` 是河/湖二值几何（用于下切、水系图），
    ``lake_a`` 是湖泊的抗锯齿覆盖率（用于平滑上色）。
    """
    mask = np.zeros((out_size, out_size), bool)
    lake_a = np.zeros((out_size, out_size), np.float32)
    if not hydro:
        return mask, lake_a
    k = zoom * out_size / float(size)                  # pack px → 输出 px
    half = out_size / 2.0 - 0.5

    def to_out(pts):
        a = np.asarray(pts, np.float32)
        y = (a[:, 0] / float(size) - cy) * zoom * out_size + half
        x = (a[:, 1] / float(size) - cx) * zoom * out_size + half
        return y, x

    for river in hydro.get("rivers") or []:
        if len(river) < 2:
            continue
        level = min(3, int(len(river) // 30))           # 长度近似 acc 量级
        radius = max(0.5, (1.0 + 2.0 * level) * k * 0.5)
        ys, xs = to_out(river)
        for i in range(len(ys)):
            _stamp_disc(mask, float(ys[i]), float(xs[i]), radius)

    for lake in hydro.get("lakes") or []:
        if len(lake) < 3:
            continue
        ys, xs = to_out(lake)
        lake_a = np.maximum(lake_a, _fill_polygon_aa(ys, xs, out_size))
    # 湖岸来自 pack 栅格轮廓，天然带 1px 直角台阶；轻微羽化把台阶抹圆
    lake_a = np.clip(gaussian_filter(lake_a, sigma=0.8, mode="nearest"),
                     0.0, 1.0).astype(np.float32)
    mask |= lake_a > 0.5
    return mask, lake_a


# ==================== 5. 引擎 ====================
class LiteEngine:
    """设备端轻量渲染：SeedPack 基场 + 程序化残差（numpy 参考实现）。"""

    def __init__(self, pack):
        f = _unpack(pack)
        self.seed = f["seed"]
        self.size = f["size"]
        self.base = f["base"]
        self.dem = f["dem"]
        self.temperature = f["temperature"]
        self.precipitation = f["precipitation"]
        self.biome = f["biome"]
        self.hydro = f["hydro"]
        self.style = f["style"]
        self.slices = f["slices"]
        self.hints = f["hints"]
        self.crs = f["crs"]

    # ---- 基场采样（测试与 Studio 共用同一映射）----
    def sample_base(self, params):
        p = _as_params(params)
        wx = _axis(p.size, p.zoom, p.cx)
        wy = _axis(p.size, p.zoom, p.cy)
        return np.clip(_bilinear(self.base, wx, wy), 0.0, 1.0)

    @property
    def times_myr(self):
        """包内时间切片的时间点（Myr）；无切片返回 ``[]``。"""
        return [float(fr["time_myr"]) for fr in self.slices]

    def sample_time(self, time_myr, wx, wy):
        """按时间插值基场，返回 ``(base01, sea_level_v01)``。

        无切片或 ``time_myr`` 为 None → 终态基场 + ``SEA_LEVEL``。
        时间超出切片范围时钳制到首/末帧。
        """
        if not self.slices or time_myr is None:
            return (np.clip(_bilinear(self.base, wx, wy), 0.0, 1.0),
                    SEA_LEVEL)
        t = float(time_myr)
        frames = self.slices
        if t <= frames[0]["time_myr"]:
            fr = frames[0]
            return (np.clip(_bilinear(fr["base"], wx, wy), 0.0, 1.0),
                    float(m2v(fr["sea_level"])))
        if t >= frames[-1]["time_myr"]:
            fr = frames[-1]
            return (np.clip(_bilinear(fr["base"], wx, wy), 0.0, 1.0),
                    float(m2v(fr["sea_level"])))
        for i in range(len(frames) - 1):
            a, b = frames[i], frames[i + 1]
            if a["time_myr"] <= t <= b["time_myr"]:
                span = max(b["time_myr"] - a["time_myr"], 1e-9)
                alpha = (t - a["time_myr"]) / span
                va = _bilinear(a["base"], wx, wy)
                vb = _bilinear(b["base"], wx, wy)
                v = (1.0 - alpha) * va + alpha * vb
                sea_m = (1.0 - alpha) * a["sea_level"] + alpha * b["sea_level"]
                return np.clip(v, 0.0, 1.0), float(m2v(sea_m))
        fr = frames[-1]                                   # 理论上不可达
        return (np.clip(_bilinear(fr["base"], wx, wy), 0.0, 1.0),
                float(m2v(fr["sea_level"])))

    # ---- 主入口 ----
    def render(self, params=None):
        t0 = time.perf_counter()
        p = _as_params(params)
        S = p.size
        q = QUALITY[p.quality]
        rng = np.random.RandomState((self.seed ^ p.digest()) & 0x7FFFFFFF)
        scale = S / 512.0
        wx = _axis(S, p.zoom, p.cx)
        wy = _axis(S, p.zoom, p.cy)

        # 1) 基场：v01 双线性；米制骨架优先用 dem_m（与 v01 同源、更平滑）
        use_time = bool(self.slices) and p.time_myr is not None
        base01, sea_v = self.sample_time(p.time_myr, wx, wy)
        if self.dem is not None and not use_time:
            height = _bilinear(self.dem, wx, wy)
        else:
            height = np.asarray(v2m(base01), np.float32)

        # 2) 海岸：亚像素带符号距离 + 低频海岸扰动
        #    (z-sea)/|∇z| 是一阶近似的「到等值线的像素距离」。基场在近岸常常接近
        #    水平，梯度趋零时这个商会在 0 附近剧烈跳动，海岸因此出现沙/水交错的
        #    麻点；所以先用 0.8px 高斯把基场磨掉 1px 抖动，再给梯度加一个「海岸带
        #    典型梯度」的下限，让 dpx 保持平滑。
        base_c = gaussian_filter(base01, sigma=0.8, mode="nearest")
        gy_b, gx_b = np.gradient(base_c)
        grad_b = np.hypot(gx_b, gy_b)
        band = np.abs(base_c - sea_v) < 0.03
        g_typ = float(np.median(grad_b[band])) if band.any() else 1e-3
        dpx = np.clip((base_c - sea_v) / np.maximum(grad_b, 0.5 * g_typ + 1e-6),
                      -48.0, 48.0)
        #    再叠加两段低频位移（≈S/64、S/24 波长，约 ±2.6px）打散等值线沿粗网格
        #    的 45° 台阶；波长必须远大于 1px，否则岸线会碎成孤立沙点。
        jitter = np.zeros((S, S), np.float32)
        for freq_div, amp_px in ((64.0, 2.0), (24.0, 0.6)):
            n = _value_noise(S, max(2, int(round(S / freq_div))), rng) - 0.5
            jitter = jitter + n * (2.0 * amp_px)
        dpx = np.clip(dpx + jitter, -48.0, 48.0).astype(np.float32)
        #    3×3 中值把「海平面附近的 1px 水/陆棋盘」抹平（平坦三角洲最容易出现），
        #    再用 0.7px 高斯把等值线的直角拐点磨圆——只磨 dpx 场，不位移岸线
        dpx = median_filter(dpx, size=3, mode="nearest").astype(np.float32)
        dpx = gaussian_filter(dpx, sigma=0.7, mode="nearest").astype(np.float32)
        land = dpx >= 0.0
        #    极小岛屿（<30px²@512，多为基场噪声）会在海面撒下一片沙色麻点，
        #    直接按海处理；世界面积阈值随分辨率缩放，放大后小岛会自然出现。
        min_isl = max(2, int(round(30.0 * scale * scale)))
        lab_i, n_i = label(land)
        if n_i:
            areas = np.bincount(lab_i.ravel())
            areas[0] = 0
            drop = areas < min_isl
            drop[0] = False                        # 标签 0 是海，不是岛
            if drop.any():
                dpx = np.where(drop[lab_i], np.minimum(dpx, -1.0),
                               dpx).astype(np.float32)
                land = dpx >= 0.0
        #    反向同理：内陆 1–2px 的水坑噪声会变成陆地上的深色麻点，按陆填平。
        lab_w, n_w = label(~land)
        if n_w:
            areas = np.bincount(lab_w.ravel())
            areas[0] = 0
            fill = areas < min_isl
            fill[0] = False                        # 标签 0 是陆，不是水坑
            if fill.any():
                dpx = np.where(fill[lab_w], np.maximum(dpx, 5.0),
                               dpx).astype(np.float32)
                land = dpx >= 0.0
        cov = _smoothstep(dpx / 1.6 + 0.5)      # 1.6px 抗锯齿的海陆覆盖率
        coast_eps = float(self.hints.get("coast_eps", COAST_FADE_PX))
        slope_ref = float(self.hints.get("slope_ref", SLOPE_REF))
        coast_px = min(float(SDF_RADIUS), max(1.0, coast_eps * scale))
        coast_w = _smoothstep(np.abs(dpx) / coast_px)     # 贴着海岸 → 0

        # 3) 细节：ridged fBm，振幅由坡度与海岸距离调制，海岸处衰减到 0
        gy, gx = np.gradient(height)
        snorm = np.clip(np.hypot(gx, gy) / (slope_ref * 512.0 / S), 0.0, 1.0)
        detail = (_ridged_fbm(S, q["octaves"], rng) - 0.5) * 2.0
        amp = q["amp"] * (0.25 + 0.75 * snorm) * coast_w
        height = height + detail * amp * cov

        # 4) 轻侵蚀：坡度限制（超过阈值的坡降搬给邻居），仅作用在陆地上
        limit = 600.0 * np.sqrt(512.0 / S) / max(p.zoom, 0.25)
        for _ in range(q["erosion"]):
            up = np.roll(height, 1, 0)
            dn = np.roll(height, -1, 0)
            lf = np.roll(height, 1, 1)
            rt = np.roll(height, -1, 1)
            avg = 0.25 * (up + dn + lf + rt)
            steep = np.maximum.reduce([np.abs(up - height), np.abs(dn - height),
                                       np.abs(lf - height), np.abs(rt - height)])
            move = np.clip((steep - limit) / limit, 0.0, 1.0)
            m = (land & (height > 0.0)).astype(np.float32) * move * 0.35
            height = (height + (avg - height) * m).astype(np.float32)

        # 5) 河网：预计算折线 → 局部河宽/下切/蓝度
        hydro, lake_a = _raster_hydro(self.hydro, self.size, p.zoom,
                                      p.cx, p.cy, S)
        width_px = np.maximum(1.0, 1.0 + 2.0 * p.zoom * S / self.size)
        carved = hydro & land
        height = height - np.where(carved, 35.0 * width_px, 0.0)

        # 6) 气候：有 T/P/biome 就采样，否则按纬度/高度做 LUT 兜底
        if self.biome is not None:
            biome = _nearest(self.biome, wx, wy).astype(np.uint8)
        else:
            if self.temperature is not None:
                temp = _bilinear(self.temperature, wx, wy)
            else:
                temp = 30.0 - 55.0 * np.abs(2.0 * wy - 1.0) - np.maximum(height, 0) / 60.0
            if self.precipitation is not None:
                precip = _bilinear(self.precipitation, wx, wy)
            else:
                precip = np.full((S, S), 900.0, np.float32)
            biome = _biome_lut(temp, precip)

        # 7) 着色：分层设色 + 海岸过渡 + 雪线/山体 + hillshade + 河网
        colors = _style_colors(p.style if p.style is not None else self.style)
        pal, hill = colors["palette"], colors["hillshade"]
        sea_m = float(v2m(sea_v))

        # 海水：水深以「离岸距离」为主控——近岸浅、远海深，几像素内平滑过渡。
        # 包内海底地形是 512 网格且近岸直接掉到 -4000m，直接映射会让海水在岸线
        # 处瞬间变深（海岸像直接抠出来的）；这里只取它很弱的大尺度分量做远海
        # 色差，近岸完全交给距离。
        depth = np.maximum(0.0, sea_m - height)
        bathy = np.clip(gaussian_filter(depth, sigma=6.0 * scale,
                                        mode="nearest") / 2500.0, 0.0, 1.0)
        #     浅水带宽度按「到陆地的真实像素距离」算：dpx 是等值线一阶近似，
        #     基场在岸边下跌很快，用它会在 2–3px 内就饱和成深海色（岸线像抠图）。
        water_px = distance_transform_edt(~land).astype(np.float32)
        #     浅水提亮必须「缓而长」：振幅压到 0.26、衰减长度 7px@512，让近岸
        #     比深海只亮一点点并平滑延伸到十几像素外；振幅一大就会在岸线外
        #     形成一圈高对比的亮青光晕。
        shelf_px = max(4.0, 7.0 * scale)
        depth_t = 1.0 - 0.26 * np.exp(-water_px / shelf_px)
        depth_t = np.clip(depth_t + 0.10 * (bathy - 0.5), 0.0, 1.0)
        sea = (pal["ocean_shallow"] * (1.0 - depth_t)[..., None]
               + pal["ocean_deep"] * depth_t[..., None])

        # 沙滩：不贴水线铺沙（贴着水线会在海岸描一圈浅色硬边），而是从岸线
        # 内侧 1px 起向内渐隐，且不做到满强度——满强度的沙色会把每段海岸
        # 描成一条奶油色轮廓线；低海拔海岸才略有沙色
        beach_px = max(1.5, 1.8 * scale)
        in_px = max(1.0, 1.0 * scale)
        beach_w = ((_smoothstep(np.clip(dpx / in_px, 0.0, 1.0))
                    * (1.0 - _smoothstep(
                        np.clip((dpx - in_px) / beach_px, 0.0, 1.0))))
                   * np.clip(1.0 - (height - 300.0) / 400.0, 0.0, 1.0)
                   * 0.55 * cov)[..., None]
        land_col = colors["biome"][np.clip(biome, 0, 8)]
        land_col = land_col * (1.0 - beach_w) + pal["beach"] * beach_w

        mount = (_smoothstep(np.clip((height - 900.0) / 1400.0, 0.0, 1.0))
                 * _smoothstep(np.clip(snorm * 1.6, 0.0, 1.0)))[..., None]
        land_col = land_col * (1.0 - 0.55 * mount) + pal["mountain"] * (0.55 * mount)

        snow = (_smoothstep(np.clip((height - 2600.0) / 1400.0, 0.0, 1.0))
                * land)[..., None]
        land_col = land_col * (1.0 - snow) + pal["snow"] * snow

        # 按覆盖率亚像素混合：海岸线不再有 1px 台阶与溢出的深色描边
        rgb = land_col * cov[..., None] + sea * (1.0 - cov)[..., None]

        az = np.deg2rad(float(hill.get("azimuth", 315.0)))
        alt = np.deg2rad(float(hill.get("altitude", 45.0)))
        strength = float(hill.get("strength", 0.6))
        h_light = np.maximum(height, sea_m)         # 水下截平，去掉岸坡断崖
        ny_, nx_ = np.gradient(h_light / 1000.0)
        nscale = 0.15                                  # 固定光照陡度（跨尺寸一致）
        norm = np.sqrt(nx_ ** 2 + ny_ ** 2 + nscale ** 2)
        dot = ((-nx_ * np.cos(alt) * np.sin(az)
                - ny_ * np.cos(alt) * np.cos(az)
                + nscale * np.sin(alt)) / np.maximum(norm, 1e-6))
        shade = np.clip(0.75 + strength * 0.45 * dot, 0.55, 1.35)
        rgb = rgb * (1.0 + (shade - 1.0) * cov)[..., None]

        river_col = pal.get("river", pal["ocean_shallow"])
        # 河流用 1px 羽化；湖泊本身已是超采样覆盖率，边缘天然抗锯齿
        river_a = np.clip(gaussian_filter(hydro.astype(np.float32), sigma=0.7,
                                          mode="nearest") * 1.4, 0.0, 1.0)
        river_a = np.maximum(river_a, lake_a)
        rgb = (rgb * (1.0 - river_a)[..., None]
               + river_col * river_a[..., None])
        rgb_u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

        rivers_rgb = np.zeros((S, S, 3), np.uint8)
        rivers_rgb[hydro] = (np.clip(river_col * 255.0 + 0.5, 0, 255)
                             .astype(np.uint8))
        ms = (time.perf_counter() - t0) * 1000.0
        stats = {
            "size": S, "quality": p.quality, "zoom": round(p.zoom, 4),
            "cx": round(p.cx, 4), "cy": round(p.cy, 4),
            "octaves": q["octaves"], "erosion_rounds": q["erosion"],
            "seed": self.seed, "param_digest": p.digest(),
            "pack_size": self.size,
            "time_myr": (None if p.time_myr is None else round(p.time_myr, 3)),
            "slices": len(self.slices),
            "sea_level_v01": round(float(sea_v), 4),
            "land_frac": round(float(land.mean()), 4),
            "river_pixels": int(hydro.sum()),
            "biome_count": int(np.unique(biome).size),
            "height_min": round(float(height.min()), 1),
            "height_max": round(float(height.max()), 1),
            "ms": round(ms, 1),
        }
        return {"rgb": rgb_u8, "height": height.astype(np.float32),
                "biome": biome, "rivers_rgb": rivers_rgb, "stats": stats}


# ==================== 6. 载入 / 便捷函数 ====================
def _unpack(pack):
    """把 SeedPack 或同名字典规整成统一字段。"""
    hints = {"coast_eps": COAST_FADE_PX, "slope_ref": SLOPE_REF,
             "hydro_min_acc": 300.0}
    if isinstance(pack, SeedPack):
        if getattr(pack, "lite_hints", None):
            hints.update({k: float(v) for k, v in pack.lite_hints.items()})
        return {"seed": pack.seed, "size": pack.size, "base": pack.v01,
                "dem": pack.dem_m, "temperature": pack.temperature,
                "precipitation": pack.precipitation, "biome": pack.biome,
                "hydro": pack.hydro, "style": pack.style,
                "slices": list(getattr(pack, "slices", None) or []),
                "hints": hints, "crs": getattr(pack, "crs", None)}
    if not isinstance(pack, dict):
        raise TypeError(f"LiteEngine 需要 SeedPack 或 dict，实际 {type(pack)!r}")
    base = pack.get("base", pack.get("v01", pack.get("global/base.npy")))
    if base is None:
        raise ValueError("pack 缺少基场 base/v01")
    base = np.ascontiguousarray(np.asarray(base, np.float32))
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError(f"base 必须是正方形二维数组，实际 {base.shape}")
    size = int(pack.get("size", base.shape[0]))
    seed = int(pack.get("seed", 0))

    def _f(names):
        for n in names:
            if pack.get(n) is not None:
                return np.ascontiguousarray(np.asarray(pack[n], np.float32))
        return None

    if isinstance(pack.get("lite_hints"), dict):
        hints.update({k: float(v) for k, v in pack["lite_hints"].items()})
    return {"seed": seed, "size": size, "base": base,
            "dem": _f(("dem", "dem_m", "global/dem_m.npy")),
            "temperature": _f(("temperature", "climate/temperature.npy")),
            "precipitation": _f(("precipitation",
                                 "climate/precipitation.npy")),
            "biome": (None if pack.get("biome") is None else
                      np.ascontiguousarray(np.asarray(pack["biome"], np.uint8))),
            "hydro": pack.get("hydro"), "style": pack.get("style"),
            "slices": list(pack.get("slices") or []),
            "hints": hints, "crs": pack.get("crs")}


def _as_params(params):
    if params is None:
        return LiteParams()
    if isinstance(params, LiteParams):
        return params
    if isinstance(params, dict):
        return LiteParams(**params)
    raise TypeError(f"params 需要 LiteParams 或 dict，实际 {type(params)!r}")


def load_pack(path):
    """读取 ``.seedpack``（等价于 ``SeedPack.open``）。"""
    return SeedPack.open(path)


def render_png(pack_path, out_png, size=512, **kw):
    """便捷函数：读包 → 渲染 → 写 PNG，返回 stats（附 ``png`` 路径）。"""
    from PIL import Image
    engine = LiteEngine(load_pack(pack_path))
    res = engine.render(LiteParams(size=size, **kw))
    Image.fromarray(res["rgb"], "RGB").save(str(out_png), "PNG")
    stats = dict(res["stats"])
    stats["png"] = str(out_png)
    return stats


