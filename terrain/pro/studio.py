# -*- coding: utf-8 -*-
"""studio — Terrain Studio v2（PySide6）：本地生成工作站。

一个窗口装下四件事（Plan v13 §6）：

    左    项目树：输出目录 → seed N → DEM / SeedPack / MapPack（双击即打开）
    中    QTabWidget：2D 画布（可缩放/平移）+ 时间滑杆；3D 视图占位（M2 后段）
    右    参数面板：seed / tier / size / quality / zoom / cx / cy
          「渲染 Lite」「导出 PNG」；Pro 分组：「运行 Pro 作业」「取消」
    底    任务队列：名称 / 状态 / 进度 / 耗时 + 总进度条 + 取消

线程模型（§2.2 / §6.3）：

    * Lite 渲染走 ``QThreadPool`` + ``QRunnable``，结果回主线程转 QPixmap；
      每次请求自增版本号，过期结果直接作废（拖动滑杆不会排长队）。
    * Pro 生产走 ``QProcess`` 子进程 ``python -m terrain.pro.produce --jsonl``，
      逐行解析 JSON 进度事件；取消 = ``terminate()``，UI 不会被 torch 拖住。

**无显示环境安全**：模块顶层不 import PySide6、不创建 QApplication，Qt 只在
``_build_window`` / ``main`` 内部引入，因此 headless / CI 下
``import terrain.pro.studio`` 不会炸。``QT_QPA_PLATFORM=offscreen`` 或
``--autoclose`` 时窗口展示后立刻退出（冒烟用）。

冒烟：

    QT_QPA_PLATFORM=offscreen python -m terrain.pro.studio terrain/output/smoke.seedpack
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


# ==================== 非 Qt 部分（可单测 / headless） ====================
def _png_array(data):
    import io

    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), np.uint8)


def load_pack(path):
    """读取包 → ``{path, kind, preview, meta}``（只解压预览图，不解全包）。"""
    from terrain.pack import PackReader, dem_to_rgb, info
    from terrain.pack.schema import FORMAT_MAPPACK

    path = os.path.abspath(str(path))
    meta = info(path)
    entries = meta.get("entries") or []
    with PackReader(path) as r:
        fmt = r.manifest["format"]
        if fmt == FORMAT_MAPPACK:
            kind = "mappack"
            if r.has("map/rgb.png"):
                preview = _png_array(r.read("map/rgb.png"))
            else:
                preview = dem_to_rgb(r.read_npy("map/dem_m.npy"))
        else:
            kind = "seedpack"
            if r.has("global/preview.png"):
                preview = _png_array(r.read("global/preview.png"))
            elif r.has("global/dem_m.npy"):
                preview = dem_to_rgb(r.read_npy("global/dem_m.npy"))
            else:
                preview = None            # 只有 v01 的极简种子包
    gen = meta.get("generator") or {}
    info_d = {
        "path": path,
        "kind": kind,
        "format": meta.get("format"),
        "seed": meta.get("seed"),
        "size": meta.get("size"),
        "tier": (gen.get("tier") if isinstance(gen, dict) else None),
        "entries": len(entries),
        "entry_bytes": int(sum(e.get("bytes") or 0 for e in entries)),
        "pack_hash": None,
        "filesize": os.path.getsize(path),
        "slices": meta.get("slices") or 0,
        "version": meta.get("version"),
    }
    with PackReader(path) as r:
        info_d["pack_hash"] = r.manifest.get("pack_hash")
    return {"path": path, "kind": kind, "preview": preview, "meta": info_d}


def export_lods(path, out_dir, levels=(512, 1024, 2048)):
    """把包内 DEM 重建为 GLB LOD 并写到 ``out_dir``，返回文件路径列表。"""
    from terrain.pack import MapPack, SeedPack

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.abspath(str(path))
    if p.lower().endswith(".seedpack"):
        sp = SeedPack.open(p)
        dem, rgb = sp.dem_m, sp.preview
    else:
        mp = MapPack.open(p)
        dem, rgb = mp.dem_m, mp.rgb
    if dem is None:
        raise ValueError(f"{os.path.basename(p)} 内没有 dem_m，无法重建 3D")

    from .lod import build_lods
    lods = build_lods(dem, rgb, levels=tuple(levels), mode="plane")
    out = []
    for lv in sorted(lods):
        fp = os.path.join(os.path.abspath(out_dir), f"lod{int(lv)}.glb")
        with open(fp, "wb") as fh:
            fh.write(lods[lv])
        out.append(fp)
    return out


def scan_output(root):
    """扫描输出目录 → ``{seed: {"dem": p, "seedpack": p, "mappack": p}}``。"""
    import re

    groups = {}
    if not os.path.isdir(root):
        return groups
    pat = re.compile(r"^seed(\d+)(_dem\.npy|\.seedpack|\.mappack)$")
    for name in sorted(os.listdir(root)):
        m = pat.match(name)
        if not m:
            continue
        sd, suffix = int(m.group(1)), m.group(2)
        if suffix == "_dem.npy":
            key = "dem"
        elif suffix == ".seedpack":
            key = "seedpack"
        else:
            key = "mappack"
        groups.setdefault(sd, {})[key] = os.path.join(root, name)
    return dict(sorted(groups.items()))


# ==================== Qt 部分（只在窗口构造时引入） ====================
def _build_window(path=None, out_dir=None):
    """构造 Studio 主窗口。返回 ``(window, api)``；Qt 在此函数内 import。"""
    import numpy as np
    from PySide6.QtCore import (QObject, QRectF, QRunnable, Qt, QThreadPool,
                                QTimer, Signal)
    from PySide6.QtGui import QAction, QImage, QPixmap
    from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                                   QFileDialog, QFormLayout, QGraphicsScene,
                                   QGraphicsView, QGroupBox, QHBoxLayout,
                                   QLabel, QLineEdit, QMainWindow, QMessageBox,
                                   QPlainTextEdit, QProgressBar, QPushButton,
                                   QScrollArea, QSlider, QSpinBox, QSplitter,
                                   QTableWidget, QTableWidgetItem, QTabWidget,
                                   QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                                   QWidget)

    from terrain.lite import LiteEngine, LiteParams
    from terrain.pack import SeedPack
    from terrain.pro.tiers import TIERS
    from .produce import DEFAULT_OUT

    root_out = os.path.abspath(str(out_dir or DEFAULT_OUT))
    win = QMainWindow()
    win.resize(1280, 820)
    win.setWindowTitle("Terrain Studio — v13")

    # ---------- 局部类（QRunnable / 画布） ----------
    class _Sig(QObject):
        done = Signal(int, object)
        failed = Signal(int, str)

    class _Job(QRunnable):
        """在线程池里跑一次 Lite 渲染，结果用信号送回主线程。"""

        def __init__(self, req, fn):
            super().__init__()
            self.req = req
            self.fn = fn
            self.sig = _Sig()

        def run(self):
            try:
                self.sig.done.emit(self.req, self.fn())
            except Exception as exc:              # noqa: BLE001
                self.sig.failed.emit(self.req, f"{type(exc).__name__}: {exc}")

    class _Canvas(QGraphicsView):
        """可滚轮缩放的图像画布。"""

        def __init__(self):
            super().__init__()
            self.setScene(QGraphicsScene())
            self.setStyleSheet("background:#1b1d21;border:0;")
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._item = self.scene().addPixmap(QPixmap())
            self._fitted = False

        def set_image(self, pix):
            self._item.setPixmap(pix)
            self.scene().setSceneRect(QRectF(pix.rect()))
            if not self._fitted and not pix.isNull():
                self.fit()

        def fit(self):
            pix = self._item.pixmap()
            if pix.isNull():
                return
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fitted = True

        def wheelEvent(self, ev):                 # noqa: N802
            f = 1.15 if ev.angleDelta().y() > 0 else 1.0 / 1.15
            self.scale(f, f)

        def resizeEvent(self, ev):                # noqa: N802
            super().resizeEvent(ev)
            if not self._fitted:
                self.fit()

    # ---------- 顶部菜单 ----------
    menubar = win.menuBar()
    m_file = menubar.addMenu("文件")
    m_prod = menubar.addMenu("生产")
    m_help = menubar.addMenu("帮助")

    # ---------- 左：项目树 ----------
    tree = QTreeWidget()
    tree.setHeaderHidden(True)
    tree.setMinimumWidth(210)

    # ---------- 中：2D 画布 + 时间滑杆 / 3D 占位 ----------
    canvas = _Canvas()
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 1000)
    slider.setValue(1000)
    slider.setEnabled(False)
    time_label = QLabel("无时间切片")
    time_label.setMinimumWidth(150)

    tab2d = QWidget()
    v2d = QVBoxLayout(tab2d)
    v2d.setContentsMargins(4, 4, 4, 4)
    v2d.addWidget(canvas, 1)
    hbar = QHBoxLayout()
    hbar.addWidget(QLabel("时间"))
    hbar.addWidget(slider, 1)
    hbar.addWidget(time_label)
    v2d.addLayout(hbar)

    tab3d = QWidget()
    v3d = QVBoxLayout(tab3d)
    lab3d = QLabel("3D 视图在 M2 后段接入（QOpenGLWidget）；"
                   "当前可用「生产 → 导出 3D」得到 GLB LOD。")
    lab3d.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lab3d.setWordWrap(True)
    v3d.addWidget(lab3d)

    tabs = QTabWidget()
    tabs.addTab(tab2d, "2D")
    tabs.addTab(tab3d, "3D")

    # ---------- 右：参数面板 ----------
    right = QWidget()
    rv = QVBoxLayout(right)
    rv.setContentsMargins(8, 8, 8, 8)
    pack_label = QLabel("未打开包")
    pack_label.setWordWrap(True)
    pack_label.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse)
    rv.addWidget(pack_label)

    form = QFormLayout()
    seed = QSpinBox()
    seed.setRange(0, 2 ** 31 - 1)
    seed.setValue(7)
    tier = QComboBox()
    tier.addItems([k for k in TIERS])
    tier.setCurrentText("draft")
    size = QSpinBox()
    size.setRange(64, 2048)
    size.setSingleStep(64)
    size.setValue(512)
    quality = QComboBox()
    quality.addItems(["fast", "balanced", "high"])
    quality.setCurrentText("balanced")
    zoom = QDoubleSpinBox()
    zoom.setRange(0.05, 64.0)
    zoom.setSingleStep(0.1)
    zoom.setValue(1.0)
    cx = QDoubleSpinBox()
    cx.setRange(0.0, 1.0)
    cx.setSingleStep(0.01)
    cx.setValue(0.5)
    cy = QDoubleSpinBox()
    cy.setRange(0.0, 1.0)
    cy.setSingleStep(0.01)
    cy.setValue(0.5)
    for w, lab in ((seed, "seed"), (tier, "tier"), (size, "size"),
                   (quality, "quality"), (zoom, "zoom"), (cx, "cx"),
                   (cy, "cy")):
        form.addRow(lab, w)
    rv.addLayout(form)

    btn_row = QHBoxLayout()
    btn_render = QPushButton("渲染 Lite")
    btn_fit = QPushButton("适应窗口")
    btn_png = QPushButton("导出 PNG")
    btn_row.addWidget(btn_render)
    btn_row.addWidget(btn_fit)
    rv.addLayout(btn_row)
    rv.addWidget(btn_png)

    stats_label = QLabel("—")
    stats_label.setWordWrap(True)
    stats_label.setMinimumHeight(90)
    stats_label.setAlignment(Qt.AlignmentFlag.AlignTop)
    stats_label.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse)
    rv.addWidget(stats_label)

    # ---- Pro 分组 ----
    grp = QGroupBox("Pro 生产（子进程）")
    gv = QVBoxLayout(grp)
    out_edit = QLineEdit(root_out)
    out_edit.setToolTip("输出目录（默认 terrain/output/pro）")
    gv.addWidget(out_edit)
    opt_row = QHBoxLayout()
    want_3d = QCheckBox("3D LOD")
    want_3d.setChecked(True)
    want_tiles = QCheckBox("2D 瓦片")
    want_tiles.setChecked(True)
    opt_row.addWidget(want_3d)
    opt_row.addWidget(want_tiles)
    gv.addLayout(opt_row)
    pro_row = QHBoxLayout()
    btn_pro = QPushButton("运行 Pro 作业")
    btn_stop = QPushButton("取消")
    btn_stop.setEnabled(False)
    pro_row.addWidget(btn_pro)
    pro_row.addWidget(btn_stop)
    gv.addLayout(pro_row)
    rv.addWidget(grp)
    rv.addStretch(1)

    right_scroll = QScrollArea()
    right_scroll.setWidgetResizable(True)
    right_scroll.setWidget(right)
    right_scroll.setMinimumWidth(300)
    right_scroll.setMaximumWidth(380)

    # ---------- 底部：任务队列 ----------
    bottom = QWidget()
    bv = QVBoxLayout(bottom)
    bv.setContentsMargins(6, 4, 6, 4)
    table = QTableWidget(0, 4)
    table.setHorizontalHeaderLabels(["任务", "状态", "进度", "耗时"])
    table.horizontalHeader().setStretchLastSection(True)
    table.setMaximumHeight(150)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    bv.addWidget(table)
    prog_row = QHBoxLayout()
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    prog_row.addWidget(bar, 1)
    bv.addLayout(prog_row)

    log = QPlainTextEdit()
    log.setReadOnly(True)
    log.setMaximumHeight(90)
    log.setPlaceholderText("Pro 子进程日志…")
    bv.addWidget(log)

    # ---------- 组装 ----------
    vsplit = QSplitter(Qt.Orientation.Vertical)
    hsplit = QSplitter(Qt.Orientation.Horizontal)
    hsplit.addWidget(tree)
    hsplit.addWidget(tabs)
    hsplit.addWidget(right_scroll)
    hsplit.setSizes([220, 700, 340])
    vsplit.addWidget(hsplit)
    vsplit.addWidget(bottom)
    vsplit.setSizes([620, 200])
    win.setCentralWidget(vsplit)

    # ---------- 状态 ----------
    st = {
        "path": None, "kind": None, "engine": None, "times": [],
        "time": None, "req": 0, "rows": {}, "jobs": {},
        "proc": None, "proc_row": None, "proc_t0": 0.0, "suppress": False,
        "last_rgb": None, "out": root_out,
    }
    pool = QThreadPool.globalInstance()
    debounce = QTimer(win)
    debounce.setSingleShot(True)
    debounce.setInterval(150)

    def _msg(text):
        win.statusBar().showMessage(text, 8000)

    # ---- 任务表 ----
    def task_add(name):
        row = table.rowCount()
        table.insertRow(row)
        for c, txt in enumerate((name, "运行中", "0%", "0.0s")):
            it = QTableWidgetItem(txt)
            if c >= 2:
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, c, it)
        table.scrollToBottom()
        return row

    def task_set(row, **kw):
        if row is None or row >= table.rowCount():
            return
        for col, key in ((0, "name"), (1, "status"), (2, "pct"), (3, "secs")):
            if key in kw:
                table.item(row, col).setText(str(kw[key]))

    # ---- 参数 ----
    def make_params():
        return LiteParams(size=size.value(), zoom=zoom.value(),
                          cx=cx.value(), cy=cy.value(),
                          quality=quality.currentText(), time_myr=st["time"])

    # ---- 渲染 ----
    def start_render():
        if st["engine"] is None:
            _msg("当前没有可渲染的 SeedPack（MapPack 只能预览）")
            return
        st["req"] += 1
        req = st["req"]
        p = make_params()
        row = task_add(f"Lite {p.size}² {p.quality}")
        st["rows"][req] = row
        eng = st["engine"]

        def work():
            t0 = time.perf_counter()
            return eng.render(p), time.perf_counter() - t0

        job = _Job(req, work)
        job.sig.done.connect(on_done)
        job.sig.failed.connect(on_failed)
        st["jobs"][req] = job
        bar.setValue(5)
        pool.start(job)

    def on_done(req, payload):
        row = st["rows"].pop(req, None)
        st["jobs"].pop(req, None)
        if req != st["req"]:
            task_set(row, status="已作废", pct="—", secs="—")
            return
        res, secs = payload
        rgb = np.ascontiguousarray(res["rgb"])
        h, w = rgb.shape[:2]
        st["last_rgb"] = rgb
        canvas.set_image(QPixmap.fromImage(
            QImage(rgb.tobytes(), w, h, 3 * w,
                   QImage.Format.Format_RGB888)))
        btn_png.setEnabled(True)
        s = res["stats"]
        stats_label.setText(
            f"{s['size']}² · {s['quality']} · {s['ms']:.0f} ms\n"
            f"陆地 {s['land_frac'] * 100:.1f}% · 群系 {s['biome_count']}\n"
            f"高程 {s['height_min']:.0f}…{s['height_max']:.0f} m\n"
            f"time={s['time_myr']} Myr · slices={s['slices']}\n"
            f"海平面 v01={s['sea_level_v01']}")
        task_set(row, status="完成", pct="100%", secs=f"{secs:.2f}s")
        bar.setValue(100)
        _msg(f"Lite 渲染完成 {s['ms']:.0f} ms")

    def on_failed(req, message):
        row = st["rows"].pop(req, None)
        st["jobs"].pop(req, None)
        task_set(row, status="失败", pct="—", secs="—")
        bar.setValue(0)
        _msg(f"渲染失败：{message}")

    def request_render(delay=0):
        if st["engine"] is None:
            return
        if delay:
            debounce.start(delay)
        else:
            start_render()

    # ---- 打开包 ----
    def open_pack(path_in):
        try:
            pack = load_pack(path_in)
        except Exception as exc:                  # noqa: BLE001
            QMessageBox.warning(win, "打开失败", str(exc))
            return False
        st.update(path=pack["path"], kind=pack["kind"], engine=None,
                  times=[], time=None, last_rgb=None)
        st["req"] += 1                            # 作废旧渲染
        meta = pack["meta"]
        pack_label.setText(
            f"{os.path.basename(pack['path'])}\n"
            f"{pack['kind']} · v{meta.get('version')} · "
            f"seed={meta.get('seed')} · {meta.get('size')}²\n"
            f"{meta.get('filesize', 0) / 1048576:.2f} MB · "
            f"slices={meta.get('slices')}")
        if pack["preview"] is not None:
            prev = np.ascontiguousarray(pack["preview"])
            h, w = prev.shape[:2]
            canvas.set_image(QPixmap.fromImage(
                QImage(prev.tobytes(), w, h, 3 * w,
                       QImage.Format.Format_RGB888)))

        st["suppress"] = True
        if pack["kind"] == "seedpack":
            try:
                sp = SeedPack.open(pack["path"])
                eng = LiteEngine(sp)
                st["engine"] = eng
                seed.setValue(int(sp.seed))
                st["times"] = eng.times_myr
            except Exception as exc:              # noqa: BLE001
                _msg(f"Lite 引擎加载失败：{exc}")
        times = st["times"]
        if times:
            slider.setEnabled(True)
            slider.setValue(slider.maximum())
            time_label.setText(f"{times[-1]:.1f} Myr（终态）")
        else:
            slider.setEnabled(False)
            time_label.setText("无时间切片")
        st["suppress"] = False

        btn_png.setEnabled(True)
        win.setWindowTitle(
            f"Terrain Studio — {os.path.basename(pack['path'])}")
        _msg(f"已打开 {os.path.basename(pack['path'])}")
        return True

    def choose_pack():
        fn, _ = QFileDialog.getOpenFileName(
            win, "打开地图包", st["out"],
            "地图包 (*.seedpack *.mappack);;所有文件 (*)")
        if fn:
            open_pack(fn)

    # ---- 时间滑杆 ----
    def slider_time():
        times = st["times"]
        if not times:
            return None
        v = slider.value() / slider.maximum()
        t = times[0] + (times[-1] - times[0]) * v
        return t

    def on_slider(_value):
        if st["suppress"] or not st["times"]:
            return
        t = slider_time()
        st["time"] = t
        time_label.setText(f"{t:.1f} Myr" if t is not None else "—")
        request_render(150)

    # ---- 导出 PNG ----
    def export_png():
        rgb = st["last_rgb"]
        if rgb is None:
            _msg("没有可导出的渲染结果，请先「渲染 Lite」")
            return
        stem = os.path.splitext(os.path.basename(st["path"] or "lite"))[0]
        fn, _ = QFileDialog.getSaveFileName(
            win, "导出 PNG", os.path.join(st["out"], f"{stem}_lite.png"),
            "PNG (*.png)")
        if not fn:
            return
        try:
            from PIL import Image
            Image.fromarray(rgb, "RGB").save(fn, "PNG")
            _msg(f"已导出 {fn}")
        except Exception as exc:                  # noqa: BLE001
            QMessageBox.warning(win, "导出失败", str(exc))

    # ---- Pro 子进程 ----
    def handle_event(line):
        import json
        try:
            ev = json.loads(line)
        except ValueError:
            log.appendPlainText(line)
            return
        row = st["proc_row"]
        name = ev.get("event")
        pct = ev.get("pct")
        if pct is not None:
            bar.setValue(int(pct))
            task_set(row, pct=f"{int(pct)}%")
        if name == "stage":
            task_set(row, status=f"阶段 {ev.get('stage')}")
            log.appendPlainText(
                f"stage={ev.get('stage')} {ev.get('seconds')}s "
                f"rss={ev.get('peak_rss_mb')}MB")
        elif name == "reuse":
            task_set(row, status="复用已有包")
        elif name in ("done", "error"):
            log.appendPlainText(line)

    def on_pro_stdout():
        proc = st["proc"]
        data = bytes(proc.readAllStandardOutput())
        buf = st.setdefault("_buf", b"") + data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                handle_event(line.decode("utf-8", "replace").strip())
        st["_buf"] = buf

    def on_pro_stderr():
        proc = st["proc"]
        text = bytes(proc.readAllStandardError()).decode("utf-8", "replace")
        if text.strip():
            log.appendPlainText(text.rstrip())

    def on_pro_finished(code, _status):
        row = st["proc_row"]
        secs = time.perf_counter() - st["proc_t0"]
        ok = int(code) == 0
        task_set(row, status=("完成" if ok else f"失败({code})"),
                 pct="100%" if ok else "—", secs=f"{secs:.1f}s")
        bar.setValue(100 if ok else 0)
        btn_pro.setEnabled(True)
        btn_stop.setEnabled(False)
        st["proc"] = None
        _msg(f"Pro 作业结束 code={code} {secs:.1f}s")
        refresh_tree()

    def run_pro():
        from PySide6.QtCore import QProcess
        if st["proc"] is not None:
            _msg("已有 Pro 作业在跑")
            return
        out = out_edit.text().strip() or root_out
        args = ["-m", "terrain.pro.produce",
                "--seed", str(seed.value()),
                "--tier", tier.currentText(),
                "--out", out,
                "--jsonl"]
        if not want_3d.isChecked():
            args.append("--no-3d")
        if not want_tiles.isChecked():
            args.append("--no-tiles")
        proc = QProcess(win)
        proc.setWorkingDirectory(_ROOT)
        proc.setProgram(sys.executable)
        proc.setArguments(args)
        proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.SeparateChannels)
        st["proc"] = proc
        st["_buf"] = b""
        st["proc_row"] = task_add(
            f"Pro seed{seed.value()} {tier.currentText()}")
        st["proc_t0"] = time.perf_counter()
        bar.setValue(1)
        log.appendPlainText(f"$ {' '.join([sys.executable] + args)}")
        proc.readyReadStandardOutput.connect(on_pro_stdout)
        proc.readyReadStandardError.connect(on_pro_stderr)
        proc.finished.connect(on_pro_finished)
        proc.start()
        btn_pro.setEnabled(False)
        btn_stop.setEnabled(True)
        _msg("Pro 作业已启动")

    def cancel_all():
        from PySide6.QtCore import QProcess
        st["req"] += 1                            # 作废 Lite 结果
        proc = st["proc"]
        if proc is not None and proc.state() != QProcess.ProcessState.NotRunning:
            proc.terminate()
            _msg("已请求终止 Pro 子进程")
        btn_stop.setEnabled(False)

    # ---- 项目树 ----
    def refresh_tree():
        tree.clear()
        root_item = QTreeWidgetItem([os.path.basename(st["out"]) or st["out"],
                                     str(len(scan_output(st["out"])))])
        root_item.setToolTip(0, st["out"])
        tree.addTopLevelItem(root_item)
        for sd, files in scan_output(st["out"]).items():
            node = QTreeWidgetItem([f"seed {sd}", ""])
            for key, label in (("dem", "DEM"), ("seedpack", "SeedPack"),
                               ("mappack", "MapPack")):
                fp = files.get(key)
                if not fp:
                    continue
                child = QTreeWidgetItem(
                    [f"{label} · {os.path.basename(fp)}", ""])
                child.setData(0, Qt.ItemDataRole.UserRole, fp)
                node.addChild(child)
            root_item.addChild(node)
        root_item.setExpanded(True)

    def on_tree_open(item, _col=0):
        fp = item.data(0, Qt.ItemDataRole.UserRole)
        if fp:
            open_pack(fp)

    def choose_out():
        d = QFileDialog.getExistingDirectory(win, "选择输出目录", st["out"])
        if d:
            st["out"] = os.path.abspath(d)
            out_edit.setText(st["out"])
            refresh_tree()

    def export_3d():
        if not st["path"]:
            _msg("先打开一个包")
            return
        out = QFileDialog.getExistingDirectory(win, "选择 3D 导出目录")
        if not out:
            return
        try:
            files = export_lods(st["path"], out)
            QMessageBox.information(
                win, "导出完成",
                "已写出：\n" + "\n".join(os.path.basename(f) for f in files))
        except Exception as exc:                  # noqa: BLE001
            QMessageBox.warning(win, "导出失败", str(exc))

    # ---- 菜单装配 ----
    act_open = QAction("打开包…", win)
    act_open.triggered.connect(choose_pack)
    act_out = QAction("切换输出目录…", win)
    act_out.triggered.connect(choose_out)
    act_quit = QAction("退出", win)
    act_quit.triggered.connect(win.close)
    m_file.addAction(act_open)
    m_file.addAction(act_out)
    m_file.addSeparator()
    m_file.addAction(act_quit)

    act_pro = QAction("运行 Pro 作业", win)
    act_pro.triggered.connect(run_pro)
    act_stop = QAction("取消当前任务", win)
    act_stop.triggered.connect(cancel_all)
    act_lod = QAction("导出 3D LOD…", win)
    act_lod.triggered.connect(export_3d)
    m_prod.addAction(act_pro)
    m_prod.addAction(act_stop)
    m_prod.addSeparator()
    m_prod.addAction(act_lod)

    act_about = QAction("关于 Terrain Studio", win)
    act_about.triggered.connect(lambda: QMessageBox.about(
        win, "关于", "Terrain Studio v13\n本地生成工作站：Lite 线程池 + "
                     "Pro 子进程，共用 SeedPack/MapPack v2 契约。"))
    m_help.addAction(act_about)

    # ---- 信号接线 ----
    btn_render.clicked.connect(lambda: request_render(0))
    btn_fit.clicked.connect(canvas.fit)
    btn_png.clicked.connect(export_png)
    btn_pro.clicked.connect(run_pro)
    btn_stop.clicked.connect(cancel_all)
    slider.valueChanged.connect(on_slider)
    debounce.timeout.connect(start_render)
    tree.itemDoubleClicked.connect(on_tree_open)
    quality.currentIndexChanged.connect(lambda _i: request_render(0))
    size.valueChanged.connect(lambda _v: request_render(200))
    zoom.valueChanged.connect(lambda _v: request_render(200))
    cx.valueChanged.connect(lambda _v: request_render(200))
    cy.valueChanged.connect(lambda _v: request_render(200))

    refresh_tree()
    if path:
        if os.path.exists(path):
            open_pack(path)
        else:
            print(f"[studio] 包不存在：{path}", file=sys.stderr)

    api = {
        "open": open_pack, "refresh_tree": refresh_tree, "state": st,
        "render": request_render, "canvas": canvas, "slider": slider,
        "run_pro": run_pro, "cancel": cancel_all,
    }
    return win, api


def main(argv=None):
    """Studio 入口：``python -m terrain.pro.studio [包路径] [--out DIR]``。

    只在函数内部 import PySide6 / 创建 QApplication，因此 headless 环境下
    ``import terrain.pro.studio`` 完全安全；``QT_QPA_PLATFORM=offscreen``
    或 ``--autoclose`` 时窗口展示后立刻退出（冒烟用，不等用户操作）。
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    args = list(sys.argv[1:] if argv is None else argv)
    autoclose = ("--autoclose" in args
                 or os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen")
    out_dir = None
    if "--out" in args:
        i = args.index("--out")
        if i + 1 < len(args):
            out_dir = args[i + 1]
            del args[i:i + 2]
    rest = [a for a in args if not a.startswith("--")]
    path = rest[0] if rest else None

    app = QApplication.instance() or QApplication([sys.argv[0]])
    win, _api = _build_window(path, out_dir)
    win.show()
    if autoclose:
        print(f"[studio] headless 冒烟：已加载 "
              f"{os.path.basename(path) if path else '(空窗口)'}，立即退出",
              flush=True)
        QTimer.singleShot(0, app.quit)
    return int(app.exec())


if __name__ == "__main__":
    if __package__:                              # python -m terrain.pro.studio
        raise SystemExit(main())
    # 直接 `python path/to/studio.py` 时本文件是裸脚本，相对导入无处可依；
    # 把它当包模块重新执行一次，两种启动方式行为一致。
    import runpy

    sys.path.insert(0, _ROOT)
    runpy.run_module("terrain.pro.studio", run_name="__main__")


__all__ = ["main", "load_pack", "export_lods", "scan_output"]
