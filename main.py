# -*- coding: utf-8 -*-
"""Terrain Studio 应用入口。

    python main.py                 # 空窗口
    python main.py terrain/output/smoke.seedpack
    python main.py --out terrain/output/pro

默认输出目录为 ``terrain/output/pro``；Qt 在 ``terrain.pro.studio.main()``
内部才 import，因此本入口在无显示环境下也不会在导入期失败。
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from terrain.pro.studio import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
