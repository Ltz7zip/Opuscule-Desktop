"""缓存键名的单一来源。

``PLATE_CACHE_VERSION`` 表示板块模拟缓存的数据契约版本：
算法一旦改变输出几何（如 v4 的 ``_reshape_continents``），必须递增，
使旧缓存自然失效。写缓存与读缓存都必须走 ``plate_cache_path``，
避免同一产物出现多个文件名。
"""

import os

# 板块模拟缓存契约版本。改输出几何/字段时必须 +1。
PLATE_CACHE_VERSION = 4


def plate_cache_path(root: str, T: int, seed: int) -> str:
    """返回板块模拟缓存路径（``<root>/cache/plate_sim_v{N}_T{T}_seed{seed}.npz``）。"""
    return os.path.join(
        root, "cache", f"plate_sim_v{PLATE_CACHE_VERSION}_T{T}_seed{seed}.npz")
