# -*- coding: utf-8 -*-
"""
tectonics.py —— 真实板块边界 → GAN 构造条件通道 (Bird 2003)

=============================================================
通道语义 (训练/模拟/生成三端共用, 不可更改):
  t0 一般边界邻近  exp(-(d/40')²)
  t1 俯冲/碰撞     exp(-(d/16')²)   (SUB+OCB+CCB)
  t2 离散(脊/裂谷) exp(-(d/16')²)   (OSR+CRB)
  t3 转换断层      exp(-(d/16')²)   (OTF+CTF)
d = 到最近对应类边界的距离(角分)。cell_arcmin 参数化 → 板块模拟器复用同一函数。
=============================================================
"""
import os
import json
import urllib.request
import numpy as np
from scipy.ndimage import (distance_transform_edt, gaussian_filter,
                            zoom as ndi_zoom)

# 数据文件目录 = 本脚本所在目录 (terrain/)
_DIR = os.path.dirname(os.path.abspath(__file__))

GEOJSON_URLS = [
    "https://raw.githubusercontent.com/fraxen/tectonicplates/master/PB2002_boundaries.json",
    "https://raw.githubusercontent.com/fraxen/tectonicplates/main/PB2002_boundaries.json",
    "https://cdn.jsdelivr.net/gh/fraxen/tectonicplates@master/PB2002_boundaries.json",
    "https://cdn.jsdelivr.net/gh/fraxen/tectonicplates@main/PB2002_boundaries.json",
    "https://ghproxy.net/https://raw.githubusercontent.com/fraxen/tectonicplates/master/PB2002_boundaries.json",
]
_DS = 8   # 距离场计算分辨率 (1 cell = 8 arcmin)

_TYPE2GROUP = {"SUB": 1, "OCB": 1, "CCB": 1, "OSR": 2, "CRB": 2, "OTF": 3, "CTF": 3}

# GeoJSON 下载大小上限 (~50MB), 防止异常大响应耗尽内存
_GEOJSON_MAX_BYTES = 50 * 1024 * 1024


def _safe_urlopen(url, timeout=90):
    """带 scheme 校验 + UA 头的 urlopen, 仅允许 http/https。"""
    from urllib.parse import urlparse
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"不允许的 URL scheme: {scheme!r} (仅支持 http/https)")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FantasyWorldSimulator/1.0 (+https://github.com)"},
    )
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310 - scheme 已在上行白名单校验


def proximity_fields(masks, cell_arcmin, r_general=40.0, r_type=16.0):
    """masks: [any, conv, div, trans] bool → (4,H,W) 邻近场, 距离物理单位=角分"""
    out = np.zeros((4,) + masks[0].shape, np.float32)
    for i, m in enumerate(masks):
        if m.any():
            d = distance_transform_edt(~m) * cell_arcmin
            out[i] = np.exp(-(d / (r_general if i == 0 else r_type)) ** 2)
    return out


def download_tectonic(dest=None):
    """优先: 环境变量 TECT_JSON / 本地文件; 其次: 多镜像下载(带JSON有效性校验)"""
    dest = dest or os.path.join(_DIR, "data", "PB2002_boundaries.json")
    env = os.environ.get("TECT_JSON")
    if env and os.path.exists(env):
        print(f"  使用 TECT_JSON 指定的边界文件: {env}")
        return env
    if os.path.exists(dest):
        return dest
    import json as _json
    last = None
    for url in GEOJSON_URLS:
        try:
            print(f"下载板块边界数据 ({url.split('/')[2]}) ...")
            with _safe_urlopen(url, timeout=90) as r:
                data = r.read(_GEOJSON_MAX_BYTES + 1)
            if len(data) > _GEOJSON_MAX_BYTES:
                raise ValueError(f"响应超过大小上限 {_GEOJSON_MAX_BYTES/1e6:.0f}MB")
            js = _json.loads(data)                      # 校验是合法GeoJSON而非错误页
            if "features" not in js or len(js["features"]) == 0:
                raise ValueError("响应不是有效的 GeoJSON")
            with open(dest, "wb") as f:
                f.write(data)
            print(f"  已保存 {dest} ({len(data) / 1e6:.1f} MB)")
            return dest
        except Exception as e:
            last = e
    print(f"  下载失败: {last} → 将使用地形梯度代理边界")
    return None



def _load_boundary_points(path):
    """从 GeoJSON 边界文件提取板块边界采样点，返回 (N,3) 数组 [lat, lon, grp]。"""
    js = json.load(open(path, "r", encoding="utf-8"))
    chunks = []

    def add_line(coords, grp):
        """把一条 LineString 坐标按约 7km 步长插值采样为点序列。"""
        co = np.asarray(coords, np.float64)
        if len(co) < 2:
            return
        lon, lat = co[:, 0].copy(), co[:, 1]
        for i in range(1, len(lon)):                      # dateline 展开
            d = lon[i] - lon[i - 1]
            if d > 180:
                lon[i:] -= 360
            elif d < -180:
                lon[i:] += 360
        step = np.abs(np.diff(lon)) + np.abs(np.diff(lat))
        n = np.maximum((step / 0.12).astype(int), 1)      # ~7km 步长
        for i in range(len(lon) - 1):
            t = np.linspace(0, 1, n[i] + 1)
            chunks.append(np.stack([lat[i] + (lat[i + 1] - lat[i]) * t,
                                    lon[i] + (lon[i + 1] - lon[i]) * t,
                                    np.full(n[i] + 1, grp, np.float64)], 1))

    for f in js.get("features", []):
        g = f.get("geometry") or {}
        props = f.get("properties") or {}
        grp = _TYPE2GROUP.get(props.get("type") or props.get("Type") or "", 0)
        if g.get("type") == "LineString":
            add_line(g["coordinates"], grp)
        elif g.get("type") == "MultiLineString":
            for ln in g["coordinates"]:
                add_line(ln, grp)
    if not chunks:
        return np.zeros((0, 3), np.float32)
    return np.concatenate(chunks, 0).astype(np.float32)


class TectField:
    """全球构造条件场 (4 通道)。加载/裁剪全防御化:
    - 缓存读取任何异常 → 自动重算
    - 缓存名含数据来源(bird/proxy) → 拿到真实数据后不会被旧代理缓存污染
    - patch 对任意 y0/x0/size 安全(裁剪+边缘填充兜底)"""

    def __init__(self, ny, nx, bjson=None, dem_for_proxy=None, cache=None):
        """``__init__``: 构建或从缓存加载 4 通道全球构造条件场，异常时自动重算。"""
        shp = ((ny + _DS - 1) // _DS, (nx + _DS - 1) // _DS)
        src = "bird" if bjson else "proxy"
        cache = cache or os.path.join(_DIR, "cache", f"tect_cache_{ny}x{nx}_{src}.npz")
        self.ch = None
        if os.path.exists(cache):
            try:
                d = np.load(cache)
                key = "ch" if "ch" in d.files else d.files[0]
                if d[key].shape[1:] == shp:
                    self.ch = d[key].astype(np.float32)
                    print(f"  构造场缓存加载 ({shp[1]}x{shp[0]}, {src})")
            except Exception as e:
                print(f"  缓存读取失败({e}) → 重新计算")
                self.ch = None
        if self.ch is None:
            masks = [np.zeros(shp, bool) for _ in range(4)]
            ok = False
            if bjson:
                P = _load_boundary_points(bjson)
                if len(P):
                    la, lo, grp = P[:, 0], P[:, 1], P[:, 2].astype(int)
                    rr = np.clip(((90.0 - la) / 180.0 * ny / _DS).astype(np.int64),
                                 0, shp[0] - 1)
                    cc = np.mod(((lo + 180.0) / 360.0 * nx / _DS).astype(np.int64),
                                shp[1])
                    masks[0][rr, cc] = True
                    for gval, gi in ((1, 1), (2, 2), (3, 3)):
                        sel = grp == gval
                        if sel.any():
                            masks[gi][rr[sel], cc[sel]] = True
                    ok = True
                    print(f"  板块边界栅格化: {len(P)} 采样点 → 俯冲/离散/转换掩膜")
            if not ok:
                if dem_for_proxy is None:
                    raise RuntimeError("无边界数据且无代理 DEM")
                print("  使用地形梯度代理边界 (离线兜底, 保真度有限)")
                s = gaussian_filter(dem_for_proxy.astype(np.float32), 3.0)
                gy, gx = np.gradient(s)
                m = np.hypot(gx, gy) > np.quantile(np.hypot(gx, gy), 0.986)
                ridge = m & (s > gaussian_filter(s, 10.0) + 150) & (s > -4000) & (s < -800)
                conv = m & ((s < -2000) | (s > 1500))
                masks = [m, conv, ridge, m & ~conv & ~ridge]
            self.ch = proximity_fields(masks, cell_arcmin=_DS)
            try:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.savez_compressed(cache, ch=self.ch.astype(np.float16))
                print(f"  构造场已缓存 → {cache}")
            except Exception as e:
                print(f"  缓存写入失败(不影响运行): {e}")

    def patch(self, y0, x0, size):
        """裁剪放大出 (y0,x0) 处 size×size 的构造场块，越界时边缘填充兜底。"""
        ny8, nx8 = self.ch.shape[1:]
        c0y = y0 // _DS
        c0x = x0 // _DS
        c1y = min((y0 + size + _DS - 1) // _DS + 1, ny8)
        c1x = min((x0 + size + _DS - 1) // _DS + 1, nx8)
        r0, c0 = y0 - c0y * _DS, x0 - c0x * _DS
        out = np.zeros((4, size, size), np.float32)
        for i in range(4):
            z = ndi_zoom(self.ch[i][c0y:c1y, c0x:c1x].astype(np.float32),
                         _DS, order=1)
            z = z[r0:r0 + size, c0:c0 + size]
            if z.shape != (size, size):                  # 边界兜底: 边缘填充补齐
                z = np.pad(z, ((0, max(0, size - z.shape[0])),
                               (0, max(0, size - z.shape[1]))), mode="edge")[:size, :size]
            out[i] = z
        return out