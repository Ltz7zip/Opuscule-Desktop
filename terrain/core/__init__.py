# -*- coding: utf-8 -*-
"""terrain.core — 地形核心模块（引擎 / 板块 / 构造 / 生成 / 查询 / 可视化 / 重渲染）。

路径注入：terrain/__init__.py 会把本目录预注入 sys.path，
因此 ``import generate_world`` 等裸导入也可命中本目录下的真实实现。

v12 说明：terrain/ 根目录的兼容薄封装已删除，正式代码请统一使用
``terrain.core.*`` 导入本包内的模块。
"""
