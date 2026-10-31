# -*- coding: utf-8 -*-
"""terrain.scripts — 批处理、种子分析、拼图等运维脚本。

各脚本以 CLI 为主（``python terrain/scripts/batch_gen.py``），不强制作为库使用。
terrain/__init__.py 已把本目录预注入 sys.path，``from analyze_worlds import …``
等旧相对路径导入在 ``import terrain`` 后仍然有效。
"""
