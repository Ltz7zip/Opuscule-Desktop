# -*- coding: utf-8 -*-
"""test_export_3d.py — 3D 导出 / 预览模块结构测试（免 torch，64² 合成 DEM）。"""

import base64
import json
import os
import struct

import numpy as np
import pytest

from terrain.core.export_3d import (
    _decimate, _elev_texture, _upsample_dem, build_mesh, export_world,
    write_gltf, write_obj, _m2v, _EARTH_R_M, _resolve_exagg,
)


@pytest.fixture(scope="module")
def dem64():
    rng = np.random.RandomState(0)
    dem = rng.uniform(-8000, 6000, (64, 64)).astype(np.float32)
    dem = np.where(np.abs(dem) < 200, 200.0, dem)  # 避免海陆边界极端梯度
    return dem


@pytest.fixture(scope="module")
def plane(dem64):
    return build_mesh(dem64, "plane", exagg=40.0)


@pytest.fixture(scope="module")
def globe(dem64):
    return build_mesh(dem64, "globe", exagg=50.0)


# ---- AC-2：OBJ 顶点与 DEM 一致（decimate=1 逐点） ----

def test_plane_mesh_z_matches_dem(dem64):
    dem = _decimate(dem64, 1)
    mesh = build_mesh(dem, "plane", exagg=40.0)
    h, w = dem.shape
    # 周期平面：w+1 列（含折缝列，数据 = 首列，±180° 同一条子午线）
    assert mesh["verts"].shape == (h * (w + 1), 3)
    z = mesh["verts"][:, 2].reshape(h, w + 1)
    expect = np.concatenate([dem, dem[:, :1]], axis=1) * 40.0
    assert np.max(np.abs(z - expect)) < 1e-2
    # 顶点/面数正确（相邻列连接 → 面数与原来一致）
    assert mesh["faces"].shape[0] == (h - 1) * w * 2
    assert mesh["uvs"].shape == (h * (w + 1), 2)
    assert mesh["normals"].shape == (h * (w + 1), 3)


# ---- AC-4：几何正确性（无 NaN / 半径 / 法线朝外 / 无退化面） ----

def test_no_nan(plane, globe):
    for mesh in (plane, globe):
        for k in ("verts", "faces", "uvs", "normals"):
            assert np.isfinite(mesh[k]).all(), f"{k} 含 NaN/Inf"


def test_globe_radius(dem64):
    dem = _decimate(dem64, 1)
    mesh = build_mesh(dem, "globe", exagg=50.0)
    r_grid = np.linalg.norm(mesh["verts"][: dem.size], axis=1)
    # float32 在 6.37e6 半径处 ulp≈0.76m，容差取 3m
    expect = (_EARTH_R_M + dem.ravel() * 50.0).astype(np.float32)
    assert np.max(np.abs(r_grid - expect)) < 3.0
    assert r_grid.min() >= _EARTH_R_M + dem.min() * 50.0 - 3.0


def test_faces_oriented_outward(dem64):
    for mode, exagg in (("plane", 40.0), ("globe", 50.0)):
        dem = _decimate(dem64, 1)
        mesh = build_mesh(dem, mode, exagg)
        tri = mesh["verts"][mesh["faces"]]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(n, axis=1)
        assert (area > 1e-6).all(), f"{mode} 存在退化面"
        if mode == "plane":
            assert (n[:, 2] >= 0).all()
        else:
            c = tri.mean(axis=1)
            assert (np.einsum("ij,ij->i", n, c) >= 0).all(), f"{mode} 存在朝内面"


def test_globe_pole_caps(dem64):
    dem = _decimate(dem64, 1)
    mesh = build_mesh(dem, "globe", exagg=50.0)
    h, w = dem.shape
    n_face = (h - 1) * w * 2 + 2 * w  # 网格面 + 两极 fan
    assert mesh["faces"].shape[0] == n_face
    assert mesh["verts"].shape[0] == h * w + 2  # 网格 + 两极顶点
    # 极顶位于 ±y 轴
    assert abs(mesh["verts"][-1, 0]) < 1e-3 and mesh["verts"][-1, 1] > 0
    assert abs(mesh["verts"][-2, 0]) < 1e-3 and mesh["verts"][-2, 1] < 0


# ---- 纹理 ----

def test_texture(dem64):
    dem = _decimate(dem64, 2)
    tex = _elev_texture(dem)
    assert tex.shape == dem.shape + (3,)
    assert tex.dtype == np.uint8
    # 高程着色单调性：高海拔应比深海亮（蓝→白）
    v = _m2v(dem)
    assert tex[v > 0.9].mean() > tex[v < 0.2].mean()


def test_decimate_shape(dem64):
    assert _decimate(dem64, 1).shape == (64, 64)
    assert _decimate(dem64, 4).shape == (16, 16)


# ---- 细分加密（--subdivide）----

def test_upsample_dem_shape_and_range(dem64):
    up = _upsample_dem(dem64, 2)
    assert up.shape == (128, 128)
    # 样条可能轻微振铃 → 必须裁剪回原始高程范围
    assert up.min() >= dem64.min() and up.max() <= dem64.max()
    assert _upsample_dem(dem64, 1).shape == (64, 64)   # factor<=1 原样


def test_subdivide_increases_face_count(dem64):
    dem = _decimate(dem64, 1)
    base = build_mesh(dem, "globe", exagg=50.0)
    fine = build_mesh(dem, "globe", exagg=50.0, subdivide=2)
    # zoom(×2) → 每轴向翻倍：面数/顶点数约 ×4
    assert fine["faces"].shape[0] > 3.9 * base["faces"].shape[0]
    assert fine["verts"].shape[0] > 3.9 * base["verts"].shape[0]
    assert fine["uvs"].shape == (fine["verts"].shape[0], 2)
    assert fine["normals"].shape == fine["verts"].shape
    assert fine["faces"].max() < fine["verts"].shape[0]   # 索引不越界


def test_subdivide_z_matches_upsampled_dem(dem64):
    dem = _decimate(dem64, 1)
    mesh = build_mesh(dem, "plane", exagg=40.0, subdivide=2)
    expect = _upsample_dem(dem, 2)
    he, we = expect.shape
    assert mesh["verts"].shape[0] == he * (we + 1)   # 周期平面含折缝列
    z = mesh["verts"][:, 2].reshape(he, we + 1)
    truth = np.concatenate([expect, expect[:, :1]], axis=1) * 40.0
    assert np.max(np.abs(z - truth)) < 1e-2


def test_subdivide_surface_smooth_and_accurate():
    """光滑解析高程上细分后，网格应精确逼近细网格上的解析真值（平滑、无振铃误差）。"""
    h = 64
    yy, xx = np.mgrid[0:h, 0:h]
    u = xx / (h - 1) * 3.0
    v = yy / (h - 1) * 3.0
    dem = (np.sin(u) * np.cos(v) * 8000.0).astype(np.float32)
    mesh = build_mesh(dem, "plane", exagg=1.0, subdivide=2)
    fine = 2 * h
    fy, fx = np.mgrid[0:fine, 0:fine]
    tu = fx / 2.0 / (h - 1) * 3.0       # zoom(×2) 边缘对齐：细网格中心 = 原网格 j/2
    tv = fy / 2.0 / (h - 1) * 3.0
    truth = (np.sin(tu) * np.cos(tv) * 8000.0).reshape(fine, fine)
    z = mesh["verts"][:, 2].reshape(fine, fine + 1)
    rms = float(np.sqrt(((z[:, :fine] - truth) ** 2).mean()))
    assert rms < 250.0, f"细分网格偏离解析真值过大: rms={rms:.1f}m"


def test_subdivide_deterministic(tmp_path, dem64):
    dem = _decimate(dem64, 2)
    mesh = build_mesh(dem, "plane", exagg=40.0, subdivide=2)
    tex = _elev_texture(dem)
    p1 = str(tmp_path / "a.gltf")
    p2 = str(tmp_path / "b.gltf")
    write_gltf(p1, mesh, tex, wrap_u=False)
    write_gltf(p2, mesh, tex, wrap_u=False)
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        assert f1.read() == f2.read()


def test_export_world_subdivide(tmp_path, dem64):
    out = str(tmp_path)
    produced = export_world(dem64, out, mode="globe", export=("gltf",),
                            decimate=1, subdivide=2, verbose=False)
    gltf = os.path.join(out, "world_3d_globe.gltf")
    assert gltf in produced
    g = json.load(open(gltf))
    acc = g["accessors"]
    n_pos = acc[g["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert n_pos["count"] > 3.9 * dem64.size  # 加密后顶点数约为原 4×(globe 含极项更略多)


# ---- AC-3：OBJ / glTF 结构性 ----

def test_obj_write(tmp_path, dem64, plane):
    dem = _decimate(dem64, 2)
    tex = _elev_texture(dem)
    prefix = str(tmp_path / "world_3d_plane")
    obj, mtl, tex_png = write_obj(prefix, plane, tex)
    assert os.path.exists(obj) and os.path.exists(mtl) and os.path.exists(tex_png)
    with open(obj) as fh:
        lines = fh.read().splitlines()
    n_v = sum(1 for l in lines if l.startswith("v "))
    n_vt = sum(1 for l in lines if l.startswith("vt "))
    n_f = sum(1 for l in lines if l.startswith("f "))
    assert n_v == plane["verts"].shape[0] and n_vt == n_v
    assert n_f == plane["faces"].shape[0]
    assert any(l.startswith("mtllib ") for l in lines)
    with open(mtl) as fh:
        assert "map_Kd" in fh.read()


def test_gltf_structural(tmp_path, dem64, globe):
    dem = _decimate(dem64, 2)
    tex = _elev_texture(dem)
    path = write_gltf(str(tmp_path / "world_3d_globe.gltf"), globe, tex, wrap_u=True)
    assert os.path.exists(path)
    g = json.load(open(path))
    assert g["asset"]["version"] == "2.0"
    prim = g["meshes"][0]["primitives"][0]
    acc = g["accessors"]
    n_pos = acc[prim["attributes"]["POSITION"]]
    n_idx = acc[prim["indices"]]
    assert n_pos["count"] == globe["verts"].shape[0]
    assert n_idx["count"] == globe["faces"].shape[0]
    assert "min" in n_pos and "max" in n_pos  # POSITION min/max 必填
    # buffer 解码长度 = bufferViews 之和 + 图片
    buf = g["buffers"][0]
    raw = base64.b64decode(buf["uri"].split(",", 1)[1])
    assert len(raw) == buf["byteLength"]
    # indices 全部在顶点范围内
    idx_view = g["bufferViews"][n_idx["bufferView"]]
    n_idx_count = n_idx["count"]
    start = idx_view["byteOffset"]
    idx_arr = np.frombuffer(raw[start:start + n_idx_count * 4], dtype="<u4")
    assert idx_arr.max() < globe["verts"].shape[0]
    # 图片 bufferView 存在（PNG magic）
    img_view = g["bufferViews"][4]
    assert raw[img_view["byteOffset"]:img_view["byteOffset"] + 8] == b"\x89PNG\r\n\x1a\n"
    # 自包含：无外部 uri
    assert all(not v.get("uri", "").startswith(("http", "file")) for v in g["buffers"])


# ---- AC-7：确定性 ----

def test_deterministic(tmp_path, dem64):
    dem = _decimate(dem64, 2)
    mesh = build_mesh(dem, "plane", exagg=40.0)
    tex = _elev_texture(dem)
    p1 = str(tmp_path / "a.gltf")
    p2 = str(tmp_path / "b.gltf")
    write_gltf(p1, mesh, tex, wrap_u=False)
    write_gltf(p2, mesh, tex, wrap_u=False)
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        assert f1.read() == f2.read()


# ---- export_world 编排 ----

def test_export_world_produces_files(tmp_path, dem64):
    out = str(tmp_path)
    produced = export_world(dem64, out, mode="both", export=("obj", "gltf"),
                            decimate=2, verbose=False)
    names = {os.path.basename(p) for p in produced}
    assert names == {
        "world_3d_plane.obj", "world_3d_plane.mtl", "world_3d_plane_tex.png",
        "world_3d_plane.gltf",
        "world_3d_globe.obj", "world_3d_globe.mtl", "world_3d_globe_tex.png",
        "world_3d_globe.gltf",
    }
    # glTF 球体应 wrapS=REPEAT（经度无缝）
    g = json.load(open(os.path.join(out, "world_3d_globe.gltf")))
    assert g["samplers"][0]["wrapS"] == 10497
    g = json.load(open(os.path.join(out, "world_3d_plane.gltf")))
    assert g["samplers"][0]["wrapS"] == 33071


# ---- PyVista 桌面预览（GPU 渲染后端） ----
# pyvista 为可选依赖：每个测试内单独 importorskip，缺失时跳过该测试而非整组


def test_pyvista_polydata_structure(plane):
    """_mesh_to_polydata：顶点/面/tcoords 与 mesh dict 一致（VTK faces 布局 [3, i0, i1, i2]）。"""
    pytest.importorskip("pyvista")
    from terrain.core.export_3d import _mesh_to_polydata
    pd = _mesh_to_polydata(plane)
    assert pd.n_points == plane["verts"].shape[0]
    assert pd.n_cells == plane["faces"].shape[0]
    f = np.asarray(pd.faces).reshape(-1, 4)
    assert (f[:, 0] == 3).all()                      # 全三角
    assert (f[:, 1:] == plane["faces"]).all()        # 顶点索引与原始面一致
    t = np.asarray(pd.active_texture_coordinates)
    assert t.shape == plane["uvs"].shape             # tcoords = uvs


def test_pyvista_texture_orientation(dem64):
    """离屏渲染：纹理行 0=南极(v≈1) 与网格 UV 对齐，顶视图下屏幕上方=北=纹理末行。"""
    pytest.importorskip("pyvista")
    import pyvista as pv
    from terrain.core.export_3d import _mesh_to_polydata
    dem = _decimate(dem64, 2)                        # 32²
    mesh = build_mesh(dem, "plane", exagg=40.0)
    h = dem.shape[0]
    tex = np.zeros((h, h, 3), np.uint8)
    tex[: h // 2] = (0, 40, 220)                     # 上半=蓝（南）
    tex[h // 2:] = (230, 200, 0)                     # 下半=黄（北）
    pd = _mesh_to_polydata(mesh)
    try:
        pl = pv.Plotter(off_screen=True, window_size=(400, 400))
    except Exception as e:  # noqa: BLE001 无渲染上下文则跳过
        pytest.skip(f"无 PyVista 渲染上下文: {e}")
    pl.add_mesh(pd, texture=pv.Texture(tex), lighting=False)
    pl.view_xy()                                     # 顶视图：屏幕上方=+Y=北
    pl.camera.zoom(3)                                # 拉近使平面充满画面
    img = pl.screenshot(return_img=True)
    pl.close()
    top = img[: 60].reshape(-1, 3).mean(axis=0)
    bottom = img[-60:].reshape(-1, 3).mean(axis=0)
    assert top[1] > top[2] + 40, f"顶部应偏黄(北半球, 纹理末行), 实得 {top}"
    assert bottom[2] > bottom[0] + 40, f"底部应偏蓝(南半球, 纹理首行), 实得 {bottom}"


def test_preview_routing_pyvista_first(monkeypatch, dem64):
    """preview_desktop 优先走 PyVista；pyvista 缺失时回退 matplotlib。"""
    import sys
    from terrain.core import export_3d
    pytest.importorskip("pyvista")
    calls = []
    monkeypatch.setattr(export_3d, "_preview_pyvista",
                        lambda *a, **k: calls.append("pv"))
    monkeypatch.setattr(export_3d, "_preview_matplotlib",
                        lambda *a, **k: calls.append("mpl"))
    dems = {"plane": _decimate(dem64, 4)}
    real_py = sys.modules["pyvista"]
    monkeypatch.setitem(sys.modules, "pyvista", None)   # 模拟未安装 → 回退
    export_3d.preview_desktop(dems, {}, 40.0, 50.0)
    assert calls == ["mpl"], f"pyvista 缺失应回退 matplotlib, 实得 {calls}"
    calls.clear()
    monkeypatch.setitem(sys.modules, "pyvista", real_py)  # 恢复已安装
    export_3d.preview_desktop(dems, {}, 40.0, 50.0)
    assert calls == ["pv"], f"pyvista 存在应优先使用, 实得 {calls}"


# ---- 精细化：自适应夸张 auto exagg ----

def test_resolve_exagg_auto_ratio(dem64):
    """auto 夸张：plane 起伏约占幅宽 7-14%，globe 约占半径 10-20%；数字原样透传。"""
    dem = _decimate(dem64, 1)
    span = float(dem.max() - dem.min())
    e_p = _resolve_exagg("auto", dem, "plane")
    assert 0.07 < span * e_p / (360.0 * 111320.0) < 0.14, \
        f"plane 起伏占比异常: {span*e_p/(360*111320):.3f}"
    e_g = _resolve_exagg("auto", dem, "globe")
    rho = span * e_g / _EARTH_R_M
    assert 0.10 < rho < 0.20, f"globe 起伏占比异常: {rho:.3f}"
    assert _resolve_exagg(40.0, dem, "plane") == 40.0
    assert _resolve_exagg("40", dem, "plane") == 40.0
    assert _resolve_exagg(None, dem, "globe") == e_g


# ---- 精细化：coast_cut 海平面裁剪 ----

def test_coast_cut_plane_sea_flat(dem64):
    """coast_cut 平面：海洋裁剪到 z=0，陆地保持 高程×exagg；折缝列=首列裁剪。"""
    dem = _decimate(dem64, 1)
    mesh = build_mesh(dem, "plane", exagg=40.0, coast_cut=True)
    h, w = dem.shape
    z = mesh["verts"][:, 2].reshape(h, w + 1)
    sea = dem <= 0
    assert np.allclose(z[:, :w][sea], 0.0, atol=1e-3), "海洋应裁剪到 z=0"
    assert np.allclose(z[:, :w][~sea], (dem * 40.0)[~sea], atol=1e-2), "陆地保持原高程×exagg"
    assert np.allclose(z[:, w], np.maximum(dem[:, 0], 0.0) * 40.0, atol=1e-2), \
        "折缝列应为首列裁剪结果（±180° 同一条子午线）"
    assert (z >= -1e-3).all(), "coast_cut 后平面不应出现负高程顶点"


def test_coast_cut_globe_min_radius(dem64):
    """coast_cut 球体：网格顶点半径 ≥ r0，极顶 = r0（海面为完美球面）。"""
    dem = _decimate(dem64, 1)
    mesh = build_mesh(dem, "globe", exagg=50.0, coast_cut=True)
    r_grid = np.linalg.norm(mesh["verts"][: dem.size], axis=1)
    assert (r_grid >= _EARTH_R_M - 3.0).all(), "coast_cut 后网格半径不应低于 r0"
    poles = mesh["verts"][-2:]
    assert np.linalg.norm(poles[0]) >= _EARTH_R_M - 2.0
    assert np.linalg.norm(poles[1]) >= _EARTH_R_M - 2.0


def test_coast_cut_no_nan(dem64):
    """coast_cut 网格无 NaN、无退化面。"""
    for mode, exagg in (("plane", 40.0), ("globe", 50.0)):
        dem = _decimate(dem64, 1)
        mesh = build_mesh(dem, mode, exagg, coast_cut=True)
        for k in ("verts", "faces", "uvs", "normals"):
            assert np.isfinite(mesh[k]).all(), f"{mode}/{k} 含非有限值"
        tri = mesh["verts"][mesh["faces"]]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        assert (area > 1e-6).all(), f"{mode} 存在退化面"


def test_smooth_normals_fallback_flat_sea():
    """全海 DEM + coast_cut：法线兜底（平面 +Z / 球体径向外侧），海面不发黑。"""
    dem = np.full((16, 16), -500.0, np.float32)
    m_plane = build_mesh(dem, "plane", exagg=40.0, coast_cut=True)
    assert np.allclose(m_plane["normals"], [0, 0, 1], atol=1e-4)
    m_globe = build_mesh(dem, "globe", exagg=50.0, coast_cut=True)
    v, n = m_globe["verts"], m_globe["normals"]
    assert (np.einsum("ij,ij->i", n, v) > 0).all(), "球体平海法线应径向外侧"
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-4)


def test_elev_texture_coastline_band(dem64):
    """coastline=True 在 0 等高线两侧产生更暗的描边像素。"""
    dem = _decimate(dem64, 2)
    t0 = _elev_texture(dem, coastline=False).astype(np.float32)
    t1 = _elev_texture(dem, coastline=True).astype(np.float32)
    assert (t1 != t0).any(), "coastline 描边应产生像素差异"
    from scipy.ndimage import binary_dilation, binary_erosion
    land = dem >= 0
    edge = (land & ~binary_erosion(land)) | \
           (binary_dilation(land, iterations=2) & ~land)
    darker = (t1[edge].mean(axis=1) < t0[edge].mean(axis=1)).mean()
    assert darker > 0.9, f"仅 {darker:.2%} 的海岸边界像素变暗"
