# -*- coding: utf-8 -*-
"""版本源、缓存键与资产清单的一致性测试。"""
import hashlib
import json
import re
from pathlib import Path

import terrain
from terrain._version import __version__
from terrain.core.cache_keys import PLATE_CACHE_VERSION, plate_cache_path

_ROOT = Path(__file__).resolve().parents[2]

# X.X.X-dev.N / -alpha.N / -beta.N / -rc.N / X.X.X(GA)
_VERSION_RE = re.compile(
    r"(\d+\.\d+\.\d+)(?:-(dev|alpha|beta|rc)\.(\d+))?")


def test_version_matches_scheme():
    """版本号必须符合既定方案，且阶段名只允许 dev/alpha/beta/rc。"""
    assert _VERSION_RE.fullmatch(__version__), f"非法版本号: {__version__}"


def test_package_exports_same_version():
    """``terrain.__version__`` 与唯一来源保持一致（防止再次写死）。"""
    assert terrain.__version__ == __version__


def test_plate_cache_path_single_source():
    """缓存路径只由 PLATE_CACHE_VERSION 决定，不出现硬编码版本。"""
    p = plate_cache_path("/tmp/root", 512, 7)
    name = f"plate_sim_v{PLATE_CACHE_VERSION}_T512_seed7.npz"
    assert Path(p).name == name
    assert Path(p).parent.as_posix() == "/tmp/root/cache"


def test_no_hardcoded_plate_cache_version():
    """仓库内不应再有写死的 plate_sim_v* 字面量（文档与测试除外）。"""
    offenders = []
    for path in _ROOT.joinpath("terrain").rglob("*.py"):
        if "__pycache__" in str(path) or path.name == "cache_keys.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"plate_sim_v\d", line):
                offenders.append(f"{path.relative_to(_ROOT)}:{i}")
    assert not offenders, "发现硬编码缓存版本: " + ", ".join(offenders)


def test_assets_lock_matches_files():
    """assets.lock.json 必须与实际权重逐字节一致。"""
    lock = json.loads(_ROOT.joinpath("assets.lock.json").read_text(encoding="utf-8"))
    assert lock["version"] == __version__
    for item in lock["assets"]:
        p = _ROOT / item["path"]
        if not p.exists():
            continue          # 权重未下载时跳过，不制造假失败
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == item["sha256"], f"{item['path']} 哈希不匹配"
