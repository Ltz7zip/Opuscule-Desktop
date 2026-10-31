# 地形引擎模块化架构文档

> 版本：1.0.0 · 适用范围：`terrain/modules/` 包及其编排器 `terrain/modules/engine.py`

***

## 1. 设计目标

将原先集中在 `terrain/core/terrain_engine.py` 中的物理/气候/水系/渲染逻辑拆分为**可独立优化**的功能模块，同时保持对旧代码的向后兼容：

| 目标     | 实现方式                                          |
| ------ | --------------------------------------------- |
| 模块独立优化 | 每个领域的物理实现独占一个 `.py` 文件，修改其内部不影响其他模块           |
| 解耦     | 模块间不直接 import 彼此的实现，仅通过 `WorldContext` 字典交换数据 |
| 依赖声明   | 模块用 `dependencies` 列表声明前置模块，引擎按拓扑序执行          |
| 可扩展    | 新增模块继承 `BaseModule` 并 `engine.register()` 即可  |
| 向后兼容   | `terrain_engine.py` 末尾再导出所有搬迁函数，旧调用方无需修改      |

***

## 2. 目录结构

```
terrain/
├── modules/
│   ├── __init__.py          # 包入口，导出所有公开符号
│   ├── base.py              # WorldContext 共享上下文 + BaseModule 基类
│   ├── engine.py            # WorldEngine 编排器（拓扑排序 + 地图生成）
│   ├── terrain_module.py    # 地形模块：高程/坡度/地貌/地形图
│   ├── hydro_module.py      # 水系模块：流量累积/河流/湖泊/水系图
│   ├── temperature_module.py# 气温模块：大陆度/洋流/温度场/气温图
│   └── climate_module.py    # 气候模块：降水/生物群系/气候图
└── core/
    └── terrain_engine.py    # 通用工具 + 向后兼容再导出层
```

### 通用工具归属

以下跨模块通用工具仍保留在 `terrain_engine.py`，模块按需 import：

- `m2v` / `v2m`：米制高程 ⇄ 归一化 \[0,1] 高程

- `snowline_m`：纬度雪线海拔

- `value_fbm`：分形布朗运动噪声

- `make_terrain_cmap`：地形色卡

- `classify_landforms`：地貌分类

***

## 3. 核心抽象

### 3.1 `WorldContext` — 共享上下文

模块间数据交换的唯一通道。构造时注入已生成的米制 DEM，模块通过 `ctx.get(key)` / `ctx.set(key, value)` 读写产物。

```
WorldContext(dem_m, lats, rng=None)
  ├── dem_m: np.ndarray (S, S)   米制高程（正=陆地，负=海洋）
  ├── lats:  np.ndarray (S,)     像素中心纬度 (-90° → +90°)
  ├── land_mask: bool (S, S)     陆地掩膜（dem_m > 0）
  ├── S: int                     方阵边长
  ├── rng: RandomState           全局随机数生成器
  └── data: Dict[str, Any]       模块输出字典（键见各模块契约）
```

### 3.2 `BaseModule` — 模块基类

```python
class BaseModule:
    name: str = "base"               # 模块唯一标识
    dependencies: List[str] = []     # 依赖的模块名

    def compute(self, ctx: WorldContext) -> None: ...  # 写入 ctx.data
    def render(self, ctx: WorldContext) -> np.ndarray: ...  # 返回 (S,S,3) RGB
```

***

## 4. 模块清单与数据契约

### 4.1 地形模块 `TerrainModule` (name=`terrain`)

| 项   | 内容                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------------------ |
| 依赖  | 无                                                                                                                  |
| 输入  | `ctx.dem_m`                                                                                                        |
| 输出键 | `terrain.v01` 归一化高程 \[0,1] float32`terrain.slope` 坡度百分比 float32`terrain.landforms` 地貌分类标签 int`terrain.rgb` 地形图 RGB |
| 渲染  | 地形色卡 + NW 方向山体阴影                                                                                                   |

### 4.2 水系模块 `HydroModule` (name=`hydro`)

| 项   | 内容                                                                                                         |
| --- | ---------------------------------------------------------------------------------------------------------- |
| 依赖  | `terrain`                                                                                                  |
| 输入  | `terrain.v01`, `ctx.land_mask`                                                                             |
| 输出键 | `hydro.flow_acc` D8 流量累积 float32`hydro.river_mask` 河道掩膜 bool`hydro.lake_mask` 湖泊掩膜 bool`hydro.rgb` 水系图 RGB |
| 渲染  | 深蓝海洋底 + 连续蓝度河道 + 青色湖泊                                                                                      |

### 4.3 气温模块 `TemperatureModule` (name=`temperature`)

| 项   | 内容                                                                                                                                                                  |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 依赖  | `terrain`                                                                                                                                                           |
| 输入  | `ctx.dem_m`, `ctx.lats`                                                                                                                                             |
| 输出键 | `temperature.ci` 大陆度 \[0,1] float32`temperature.dT` 洋流温度异常 ℃ float64`temperature.dP` 洋流降水异常 mm/yr float64`temperature.temp` 年均气温 ℃ float32`temperature.rgb` 气温图 RGB |
| 渲染  | 蓝→青→绿→黄→红 温度色卡，海洋叠加深蓝色                                                                                                                                              |

### 4.4 气候模块 `ClimateModule` (name=`climate`)

| 项   | 内容                                                                                           |
| --- | -------------------------------------------------------------------------------------------- |
| 依赖  | `terrain`, `temperature`                                                                     |
| 输入  | `temperature.ci`, `temperature.dP`, `temperature.temp`, `ctx.dem_m/lats`                     |
| 输出键 | `climate.precip` 年降水 mm/yr float32`climate.biomes` Whittaker 9 类群系 int8`climate.rgb` 气候图 RGB |
| 渲染  | 海洋深蓝 + 陆地按 BIOME\_COLORS 配色                                                                  |

***

## 5. 执行流程

`WorldEngine.run(dem_m, lats, rng)` 的内部流程：

```
构造 WorldContext
    │
    ▼
拓扑排序（依据各模块 dependencies）
    │
    ▼
依次执行模块.compute(ctx)
    terrain → hydro → temperature → climate
    │         │         │              │
    │         ▼         ▼              ▼
    │    写入 hydro.*  写入 temp.*   写入 climate.*
    │
    ▼
返回填充完毕的 ctx（所有产物在 ctx.data 中）
```

- **拓扑排序**：Kahn 算法，缺依赖或存在环时抛 `ValueError`。

- **确定性**：给定相同 `(dem_m, lats, rng)`，`ctx.data` 逐位可复现。

***

## 6. 地图生成 API

### 一站式生成四张图

```python
from terrain import WorldEngine
import numpy as np

dem = np.load("terrain/output/world_dem.npy")
S = dem.shape[0]
lats = -90.0 + (np.arange(S) + 0.5) * (180.0 / S)

maps = WorldEngine().generate_world_maps(dem, lats, rng=np.random.RandomState(1))
# maps = {"terrain": rgb, "hydro": rgb, "temperature": rgb, "climate": rgb}
```

### 分步执行 + 单独查询产物

```python
ctx = WorldEngine().run(dem, lats, rng=np.random.RandomState(1))
T   = ctx.get("temperature.temp")      # 气温场 ℃
P   = ctx.get("climate.precip")         # 降水场 mm/yr
riv = ctx.get("hydro.river_mask")       # 河道掩膜
bi  = ctx.get("climate.biomes")         # 群系标签
```

### 单独渲染某张图

```python
rgb = WorldEngine().render(ctx, "temperature")
```

***

## 7. 扩展指南：添加新模块

以「风场模块」为例，演示三步接入：

### 步骤 1：继承 `BaseModule` 并声明依赖

```python
# terrain/modules/wind_module.py
from .base import BaseModule, WorldContext

class WindModule(BaseModule):
    name = "wind"
    dependencies = ["temperature"]   # 依赖气温模块先执行

    def compute(self, ctx: WorldContext) -> None:
        T = ctx.get("temperature.temp")
        # ... 基于温度梯度计算风速 ...
        ctx.set("wind.speed", wind_speed)

    def render(self, ctx: WorldContext) -> np.ndarray:
        # 返回 (S, S, 3) RGB
        ...
```

### 步骤 2：注册到引擎

```python
from terrain import WorldEngine
from terrain.modules.wind_module import WindModule

eng = WorldEngine()           # 默认注册 terrain/hydro/temperature/climate
eng.register(WindModule())    # 追加风场模块
ctx = eng.run(dem, lats)
maps = eng.render_all(ctx)    # 现在包含 "wind"
```

### 步骤 3（可选）：在 `terrain/modules/__init__.py` 中导出

```python
from .wind_module import WindModule
__all__ = [..., "WindModule"]
```

**注意事项**：

- 依赖的模块名必须已注册，否则 `run()` 时抛 `ValueError`。

- 输出键建议使用 `"<模块名>.<产物名>"` 前缀，避免命名冲突。

- 模块内部使用的第三方库（numpy/scipy/matplotlib）需在文件顶部显式 import。

***

## 8. 向后兼容策略

`terrain_engine.py` 末尾通过再导出维持旧接口：

```python
# terrain_engine.py 末尾（节选）
from terrain.modules.temperature_module import (
    continentality_index, ocean_current_effects, temperature_field)
from terrain.modules.climate_module import (
    precipitation_field, whittaker_biome, smooth_biome_labels,
    BIOME_NAMES, BIOME_COLORS)
from terrain.modules.hydro_module import (
    d8_flow_accumulation, climate_flow_accumulation, fill_depressions,
    carve_trunk_rivers, paint_rivers_rgb, ...)
```

- 旧代码 `from terrain.core.terrain_engine import temperature_field` 仍然有效。

- `generate_world` / `render_world` / `supervisor.metrics` 等无需修改。

- 新代码推荐直接 `from terrain.modules.xxx_module import ...`。

***

## 9. C++ 原生加速层（terrain/native/, v10 起）

模块内部最耗时的数值内核（降水扫描 / 山体遮蔽 / 滤波 / 流量累积 / 洼地填充 / 河流蜿蜒 / 碎片清理等）已下沉到 C++ 扩展 `fws_native`（pybind11），与模块的 Numba 实现**逐位一致**。

### 接入约定

各模块顶部统一做「C++ 优先、Numba 回退」接入，调用点用 `if _native is not None` 分支：

```python
try:
    from ..native import fws_native as _native
except ImportError:
    _native = None        # 缺失/导入失败 → 全部走 Numba 实现
```

### 内核分布

| 模块文件 | native 内核 |
| --- | --- |
| `climate_module.py` | `scan_ew_step` / `scan_ns_step` / `mountain_shield` / `median_filter_njit` / `mode_filter_njit` |
| `hydro_module.py` | `flow_acc_kahn` / `fill_depressions_bf` / `trace_centerline` / `monotonic_repair` / `carve_path_dig` / `carve_oxbow` / `meander_single_river` / `rasterize_polylines` / `smooth_small_patches` |
| `temperature_module.py` | `ocean_run_lengths` |

### 工程约束（改 native/ 前必读）

- **零拷贝**：大数组参数用 `py::array + dtype 分派模板`零拷贝直读；`py::array_t<T, forcecast>` 会无条件拷贝（4096² 曾导致全流程 +34s）。
- **GIL**：绑定一律不加 `py::call_guard<py::gil_scoped_release>()`（无 GIL 下构造 numpy 返回数组会段错误；OpenMP 线程是原生线程不需要 GIL）。
- **OpenMP**：macOS 需 `brew install libomp`；链接必须解析到 torch 自带的 `libomp.dylib`（`-L<torch>/lib -Wl,-rpath,<torch>/lib -lomp`），否则双 runtime 触发 `OMP Error #15` 段错误。
- **Numba 语义**：Numba 整数 `/` 是浮点除法（int/int→float64），C++ 必须转 double 除；`numpy arange` 是乘法生成，表值须同法生成保证逐位一致。

### 对拍验证

- `terrain/native/tests/cmp_native.py`（基础内核）+ `cmp_adv.py`（进阶内核）：逐位对比 C++ 与 Numba 输出。
- v11 起 `fill_depressions` Python 侧已切换 `fill_depressions_bf`（Barnes 2014 堆+FIFO，堆与 FIFO 按水位交错推进才能逐位一致）。

***

## 10. 验证状态

| 验证项                 | 结果                      |
| ------------------- | ----------------------- |
| 现有测试套件              | 512 passed              |
| 真实 DEM 四图生成         | 2048×2048，37.9s，9 类群系齐全 |
| 旧 `render_world` 管线 | 正常出图                    |
| 模块化引擎确定性            | 同种子 bitwise 可复现         |
| native 对拍（cmp_native / cmp_adv） | 双分支输出逐位一致（v10–v12）  |
| v12 全流程（4096²）      | native 139.9s vs Numba 回退 146.3s |

