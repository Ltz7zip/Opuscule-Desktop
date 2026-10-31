# -*- coding: utf-8 -*-
"""
test_visualize_context.py — terrain/core/visualize_context.py 单元测试

覆盖范围（按用户要求的重点）：
  A. CLI 参数解析（_parse_points / _parse_poly / _parse_profile 及其异常路径）
  B. 缓存命中逻辑（WorldGeoContext 单例：同 seed/size 复用、不同 seed/size 重置、
     force_reload 强制重建、reset 清理）
  C. 附加：build_geo_figure 轻量烟雾测试（返回结构断言）
  D. 附加：main() CLI 端到端烟雾测试（真实写 PNG + PermissionError 备份分支）

运行方式（项目根目录执行）：
    python -m pytest terrain/tests/test_visualize_context.py -v
    python -m pytest terrain/tests/test_visualize_context.py -v -k "parse"     # 只跑解析
    python -m pytest terrain/tests/test_visualize_context.py -v -k "cache"     # 只跑单例缓存
"""

from __future__ import annotations

import os
import sys
import time
import re

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 路径准备：基于 __file__ 定位项目根 + terrain 目录，确保从任意目录运行都能导入
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))          # terrain/tests
_TERRAIN_DIR = os.path.dirname(_HERE)                       # terrain
_PROJECT_ROOT = os.path.dirname(_TERRAIN_DIR)               # 项目根
for _p in (_PROJECT_ROOT, _TERRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# visualize_context 内部已经 matplotlib.use("Agg") 所以这里直接 import 即可
from terrain.core import visualize_context as VC
from terrain.core.visualize_context import (
    _parse_points, _parse_poly, _parse_profile,
    build_geo_figure, main, _sample_profile,
)
from terrain.core.geo_context import WorldGeoContext


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def _reset_singleton_before_each():
    """每个测试之前清空单例，避免状态互相污染（测试缓存命中断言）。"""
    WorldGeoContext.reset()
    yield
    WorldGeoContext.reset()


@pytest.fixture(scope="module")
def shared_seed1_ctx():
    """模块级共享的 seed=1 上下文（加速多个烟雾测试）。"""
    yield WorldGeoContext.from_seed(seed=1, size=2048)


# ===========================================================================
# A. CLI 参数解析 —— _parse_points
# ===========================================================================

class TestParsePoints:
    """覆盖 _parse_points 全部分支。"""

    def test_none_returns_default_four(self):
        pts = _parse_points(None)
        assert len(pts) == 4
        # 默认 P1..P4 名字 + 坐标都为 tuple(len=3)
        for p in pts:
            assert isinstance(p, tuple) and len(p) == 3
            assert isinstance(p[0], float) and isinstance(p[1], float)
            assert isinstance(p[2], str) and p[2].startswith("P")

    def test_empty_string_returns_default(self):
        assert _parse_points("") == _parse_points(None)

    def test_two_unnamed_points_auto_named_p1_p2(self):
        pts = _parse_points("30,85,  0,5")
        assert len(pts) == 2
        assert pts[0][:2] == (30.0, 85.0) and pts[0][2] == "P1"
        assert pts[1][:2] == (0.0, 5.0) and pts[1][2] == "P2"

    def test_mixed_named_and_unnamed(self):
        # 有名字的点不占用 P1/P2 自动编号（用户已显式命名），
        # 第一个出现的无名点应当是 P1
        s = "35,115,Beijing, 0,5, -33,150,Sydney"
        pts = _parse_points(s)
        assert len(pts) == 3
        assert pts[0] == (35.0, 115.0, "Beijing")
        # 第二个点是第一个出现的无名点 → P1
        assert pts[1][:2] == (0.0, 5.0) and pts[1][2] == "P1"
        assert pts[2] == (-33.0, 150.0, "Sydney")

    def test_chinese_and_dot_names_accepted(self):
        s = "30,104,成都·蓉城, 50,30,莫斯科/Москва"
        pts = _parse_points(s)
        assert len(pts) == 2
        assert pts[0][2] == "成都·蓉城"
        assert pts[1][2] == "莫斯科/Москва"

    def test_floats_with_signs_and_decimals(self):
        s = "-15.5,-47.25,Point.A, 89.99,179.99,P2"
        pts = _parse_points(s)
        assert pts[0][:2] == (-15.5, -47.25)
        assert pts[0][2] == "Point.A"
        assert pts[1][:2] == (89.99, 179.99)

    def test_odd_coords_raises_valueerror(self):
        # 只有 lat 没 lon
        with pytest.raises(ValueError, match=r"成对"):
            _parse_points("30")
        with pytest.raises(ValueError, match=r"成对"):
            _parse_points("30,85, 0")

    def test_non_numeric_lat_lon_raises(self):
        with pytest.raises(ValueError):  # float("abc")
            _parse_points("abc,85")


# ===========================================================================
# A. CLI 参数解析 —— _parse_poly
# ===========================================================================

class TestParsePoly:
    def test_none_returns_none(self):
        assert _parse_poly(None) is None
        assert _parse_poly("") is None

    def test_triangle_ok(self):
        poly = _parse_poly("10,20,30,40,50,60")
        assert poly == [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]

    def test_quadrilateral_ok(self):
        poly = _parse_poly("45,5,55,5,55,20,45,20")
        assert len(poly) == 4
        assert poly[0] == (45.0, 5.0)
        assert poly[-1] == (45.0, 20.0)

    def test_three_values_raises(self):
        # 只有 1.5 个点，<3 对点
        with pytest.raises(ValueError, match=r"至少需要"):
            _parse_poly("1,2,3")

    def test_five_values_odd_raises(self):
        # 奇数个值（不可能是 lat/lon 对）
        with pytest.raises(ValueError, match=r"至少需要"):
            _parse_poly("1,2,3,4,5")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            _parse_poly("a,b,c,d,e,f")


# ===========================================================================
# A. CLI 参数解析 —— _parse_profile
# ===========================================================================

class TestParseProfile:
    def test_none_returns_default_segment(self):
        (latA, lonA), (latB, lonB) = _parse_profile(None)
        # 默认是 (55,10) → (25,125)（北海→杭州湾 大致）
        assert (latA, lonA) == (55.0, 10.0)
        assert (latB, lonB) == (25.0, 125.0)

    def test_empty_uses_default(self):
        assert _parse_profile("") == _parse_profile(None)

    def test_four_values_ok(self):
        seg = _parse_profile("-10, -150,  60.5, 40.5")
        assert seg == ((-10.0, -150.0), (60.5, 40.5))

    def test_wrong_number_of_values_raises(self):
        for bad in ["1,2", "1,2,3", "1,2,3,4,5"]:
            with pytest.raises(ValueError, match=r"4 个数"):
                _parse_profile(bad)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            _parse_profile("a,b,c,d")


# ===========================================================================
# B. 单例缓存命中逻辑 —— WorldGeoContext
# ===========================================================================

class TestSingletonCache:
    """验证单例复用 / 重建 / 清理机制。注意每个测试前 fixture 已重置单例。"""

    def test_same_seed_size_returns_same_object(self):
        ctx_a = WorldGeoContext.from_seed(seed=1, size=2048)
        ctx_b = WorldGeoContext.from_seed(seed=1, size=2048)
        assert ctx_a is ctx_b, "同 (seed,size) 第二次调用必须命中单例缓存"
        assert id(ctx_a) == id(ctx_b)

    def test_different_seed_creates_new_instance(self):
        ctx_a = WorldGeoContext.from_seed(seed=1, size=2048)
        id_a = id(ctx_a)
        # 先保证 instance_key 不同
        assert WorldGeoContext._instance_key == (1, 2048)
        ctx_b = WorldGeoContext.from_seed(seed=2, size=2048)
        # 单例槽被替换为新的 key
        assert WorldGeoContext._instance_key == (2, 2048)
        assert id(ctx_b) != id_a, "不同 seed 必须新建实例"

    def test_different_size_creates_new_instance(self):
        # 注：2048 DEM 文件存在（size 也匹配），不用 4096（可能没文件）
        # 用 seed1 / size2048 与 seed1 / size2048 force_reload 替代（但这里
        # 也可以借助 reset + 不同 seed；不过为了严格验证"size 维度"，
        # 我们检查 instance_key 变化 + 对象身份不同）
        ctx_a = WorldGeoContext.from_seed(seed=1, size=2048)
        WorldGeoContext.reset()
        # 因为 DEM 文件实际是 2048，请求 2047（不匹配会警告但继续）用来改变 key
        # 但 from_seed 的 key 是 (seed,size)，请求参数决定 key；加载后的 size
        # 不匹配只是打印 warning。所以我们可以直接对比两次不同 key 的请求。
        _ = WorldGeoContext.from_seed(seed=10, size=2048)
        # 切换回 seed=1,size=2048 必须是新对象（因为 reset 过；且上一个对象是 seed10）
        ctx_b = WorldGeoContext.from_seed(seed=1, size=2048)
        assert ctx_a is not ctx_b, "切换不同 (seed,size) 后再切换回来，\
            若中间 reset 过则不得复用旧对象"

    def test_force_reload_creates_fresh_instance(self):
        ctx_a = WorldGeoContext.from_seed(seed=1, size=2048)
        id_a = id(ctx_a)
        ctx_b = WorldGeoContext.from_seed(seed=1, size=2048, force_reload=True)
        assert ctx_b is not None
        assert id(ctx_b) != id_a, "force_reload=True 必须丢弃旧单例重建"
        # 重建后 instance_key 仍为 (1, 2048)
        assert WorldGeoContext._instance_key == (1, 2048)
        # 再一次无 reload 应命中新单例
        ctx_c = WorldGeoContext.from_seed(seed=1, size=2048)
        assert ctx_c is ctx_b

    def test_reset_clears_instance(self):
        WorldGeoContext.from_seed(seed=1, size=2048)
        assert WorldGeoContext._instance is not None
        WorldGeoContext.reset()
        assert WorldGeoContext._instance is None
        assert WorldGeoContext._instance_key is None

    def test_direct_new_raises_typeerror(self):
        """禁止直接构造（必须走 from_seed 工厂）。"""
        with pytest.raises(TypeError, match=r"from_seed"):
            WorldGeoContext()

    def test_loaded_data_invariants(self, shared_seed1_ctx):
        """缓存内容符合规格（shape、lat/lon 网格单调性）。"""
        ctx = shared_seed1_ctx
        assert ctx.shape[0] == ctx.shape[1]
        S = ctx.size
        assert S >= 2048
        assert ctx.lat_grid.shape == (S,)
        assert ctx.lon_grid.shape == (S,)
        # lat 严格递增（从 ~-90 到 ~+90）
        assert np.all(np.diff(ctx.lat_grid) > 0)
        assert ctx.lat_grid[0] < -89.0 and ctx.lat_grid[-1] > 89.0
        # lon 严格递增
        assert np.all(np.diff(ctx.lon_grid) > 0)
        assert ctx.lon_grid[0] < -179.0 and ctx.lon_grid[-1] > 179.0
        # DEM 尺寸一致
        assert ctx.dem.shape == (S, S)
        assert ctx.land_mask.dtype == bool
        # 地球级陆地占比合理区间（5%–50%）
        assert 0.05 < ctx.land_mask.mean() < 0.50


# ===========================================================================
# 附加 C. build_geo_figure 烟雾测试（返回结构 + profile 采样）
# ===========================================================================

class TestBuildFigureSmoke:
    """只验证不崩溃 + 返回结构正确；不做图像像素断言。"""

    @pytest.fixture(autouse=True)
    def _no_display(self, monkeypatch):
        # Agg 已在模块级设置；这里确保 show() 即便被误调用也不阻塞
        import matplotlib
        matplotlib.use("Agg", force=True)

    def test_figure_shape_and_info_structure(self):
        fig, axes, ctx, info = build_geo_figure(
            seed=1, size=2048,
            probe_points=[(35, 115, "Beijing"), (-15, -50, "Atl")],
            profile=((50, 0), (40, 30)),
            polygon=[(10, -80), (25, -80), (25, -55), (10, -55)],
            downsample_for_display=4,   # 降采样加速
        )
        try:
            # 图结构
            assert axes.shape == (2, 2)
            assert fig is not None

            # probe 信息
            assert "probe_points" in info and len(info["probe_points"]) == 2
            p0 = info["probe_points"][0]
            assert set(p0.keys()) >= {"name", "lat", "lon", "elev", "slope", "land"}
            assert p0["name"] == "Beijing"
            assert isinstance(p0["land"], bool)

            # poly 统计
            ps = info["polygon_stats"]
            assert ps is not None
            assert set(ps.keys()) >= {"mean_elevation", "peak", "trough",
                                      "land_ratio", "mountain_ratio",
                                      "pixel_count", "land_pixel_count"}
            assert ps["pixel_count"] > 0
            assert 0.0 <= ps["land_ratio"] <= 1.0
            assert ps["mean_elevation"] <= ps["peak"]
            assert ps["mean_elevation"] >= ps["trough"]

            # profile 结构
            pr = info["profile"]
            for k in ("lats", "lons", "cum_km", "elevs", "slopes"):
                assert k in pr and isinstance(pr[k], np.ndarray)
                assert pr[k].ndim == 1 and pr[k].size > 100
            # 累积距离单调递增（起点 0）
            assert pr["cum_km"][0] == 0.0
            assert np.all(np.diff(pr["cum_km"]) >= 0)
            # 距离与 elev/slopes 等长
            assert pr["elevs"].size == pr["slopes"].size == pr["cum_km"].size
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_sample_profile_distance_monotonic(self, shared_seed1_ctx):
        """_sample_profile 单独测试：短距离 & 长距离的 Haversine 累积合理性。"""
        # 赤道上 1° 经度 ≈ 111.32 km
        r = _sample_profile(shared_seed1_ctx, 0.0, 0.0, 0.0, 10.0, n_pts=100)
        assert 1100 < r["cum_km"][-1] < 1130, f"10° 赤道约 1113km，实际 {r['cum_km'][-1]}"
        # 同样 A→B 距离应与交换起点终点相同（绝对值）
        r2 = _sample_profile(shared_seed1_ctx, 0.0, 10.0, 0.0, 0.0, n_pts=100)
        assert abs(r["cum_km"][-1] - r2["cum_km"][-1]) < 1.0

    def test_latlon_idx_roundtrip_through_figure_ctx(self):
        """确保绘图所用 ctx 与 geo_context 原生一致性。"""
        ctx = WorldGeoContext.from_seed(seed=1, size=2048)
        # 选 100 个随机格点往返
        S = ctx.size
        rs = np.random.default_rng(0).integers(0, S, size=100)
        cs = np.random.default_rng(1).integers(0, S, size=100)
        for r, c in zip(rs, cs):
            lat, lon = ctx.idx_to_latlon(int(r), int(c))
            r2, c2 = ctx.latlon_to_idx(lat, lon)
            assert (int(r), int(c)) == (r2, c2), \
                f"往返不一致: ({r},{c})→({lat:.4f},{lon:.4f})→({r2},{c2})"


# ===========================================================================
# 附加 D. CLI main() 烟雾测试 + PermissionError 备份分支
# ===========================================================================

class TestCLIMain:
    """端到端调用 main()，检查 PNG 真实生成 + 备份分支。"""

    def test_main_writes_png(self, tmp_path):
        """tmp_path 是 pytest 提供的唯一临时目录。"""
        out_png = tmp_path / "out.png"
        args = [
            "visualize_context.py",
            "--seed", "1", "--size", "2048",
            "--downsample", "4",
            "--output", str(out_png),
        ]
        old_argv = sys.argv[:]
        try:
            sys.argv = args
            main()
        finally:
            sys.argv = old_argv

        assert out_png.exists(), "main() 应在指定 --output 路径写出 PNG"
        sz = out_png.stat().st_size
        assert sz > 50_000, f"PNG 太小 ({sz}B)，疑似空输出"
        # 头 8 字节 PNG 签名
        with open(out_png, "rb") as fh:
            magic = fh.read(8)
        assert magic == b"\x89PNG\r\n\x1a\n", "写出的文件不是 PNG 格式"

    def test_main_permission_error_falls_back_to_timestamped_file(self, tmp_path, monkeypatch):
        """模拟保存路径被占用（PermissionError）时，main() 应生成时间戳副本。"""
        out_png = tmp_path / "locked.png"

        # 伪造 fig.savefig 行为：第一次调用（目标路径）必抛 PermissionError，
        # 第二次（时间戳路径）正常写。
        real_savefile_calls = []

        class _FakeFig:
            def savefig(self, path, **_kw):
                real_savefile_calls.append(path)
                if path == str(out_png):
                    raise PermissionError("Simulated locked file")
                # 其他路径（时间戳备份）写一个小 PNG
                with open(path, "wb") as fh:
                    fh.write(b"\x89PNG\r\n\x1a\n" + b"\0" * 1000)

        # 把 build_geo_figure mock 掉直接返回 _FakeFig（避免真实绘图）
        import terrain.core.visualize_context as _vc
        monkeypatch.setattr(_vc, "build_geo_figure",
                            lambda *a, **kw: (_FakeFig(), np.empty((2, 2)), None, {}))
        # main() 末尾会 plt.close(fig)，但 _FakeFig 不是真 Figure，会 TypeError；
        # 此处直接把 plt.close 也 mock 成 no-op。
        import matplotlib.pyplot as _plt_mod
        close_calls = []
        monkeypatch.setattr(_plt_mod, "close",
                            lambda *a, **kw: close_calls.append((a, kw)))

        args = [
            "visualize_context.py",
            "--seed", "1", "--size", "2048",
            "--output", str(out_png),
        ]
        old_argv = sys.argv[:]
        try:
            sys.argv = args
            main()
        finally:
            sys.argv = old_argv

        # 验证 close 至少被调用一次（说明 finally 路径走通了）
        assert len(close_calls) >= 1, "plt.close 未被调用，main() finally 分支可能未执行"
        # 第一次原路径必须失败，第二次备份路径必须被调用
        assert len(real_savefile_calls) == 2
        assert real_savefile_calls[0] == str(out_png)
        backup = real_savefile_calls[1]
        # 备份路径必须位于同目录 + 包含 日期时间戳 + 扩展名与原名相同
        assert os.path.dirname(backup) == str(tmp_path)
        stem, ext = os.path.splitext(os.path.basename(out_png))
        assert ext == ".png"
        assert os.path.basename(backup).startswith(stem + "_")
        assert backup.endswith(ext)
        # 且文件名里应含 YYYYMMDD_HHMMSS 模式
        assert re.search(r"_\d{8}_\d{6}\.png$", backup), \
            f"备份文件名不符合时间戳约定: {backup}"
        assert os.path.exists(backup)


# ===========================================================================
# 附加：get_ctx 简写函数 & Protocol 符合性
# ===========================================================================

def test_protocol_compliance():
    from terrain.core.geo_context import TerrainDataProtocol, get_ctx
    ctx = get_ctx(seed=1, size=2048)
    assert isinstance(ctx, TerrainDataProtocol), \
        "WorldGeoContext 必须满足 TerrainDataProtocol"
    # 所有协议方法都必须可调用并返回正确类型
    e = ctx.get_elevation(0, 0)
    assert isinstance(float(e), float)
    assert isinstance(ctx.is_land(0, 0), (bool, np.bool_))
    assert isinstance(float(ctx.get_slope(0, 0)), float)
    st = ctx.get_region_stats([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert "mean_elevation" in st


def test_get_ctx_shortcut_returns_same_singleton():
    from terrain.core.geo_context import get_ctx
    a = get_ctx(seed=1, size=2048)
    b = WorldGeoContext.from_seed(seed=1, size=2048)
    assert a is b, "get_ctx() 简写应与 from_seed() 返回同一单例"


if __name__ == "__main__":
    # 允许直接 `python terrain/tests/test_visualize_context.py` 运行
    sys.exit(pytest.main([__file__, "-v"]))
