# Terrain Engine 升级计划 v13

**Pro + Lite 双线 · 全本地生成 · PySide6 工作站交付**

> 依据：`Terrain 引擎深拆（论文版）：算法、LaTeX 公式、源码与优化路线.html`、`世界构建系统的多尺度架构：Matrees、VVD 与 Terrain 引擎的综合研究.html`
> 基线代码：`terrain/`（v12）
> 版本：v13 / 2026-10-30

---

## 0 · 决策摘要

**一句话定位**：Terrain 是一个**本地运行的 4D 世界模拟工作站**——Pro 在工作站上离线跑完整物理链路并产出可分发资产包，Lite 在同一台机器的进程内做秒级派生与预览，两者共用一套 Pack 契约，统一由 PySide6 窗口操作与展示。

### 0.1 相对旧方案的三处硬变更

| # | 旧方案（两份文档） | v13 方案 | 原因 |
|---|---|---|---|
| 1 | 三产品线 Pro / Lite / **Web**，Web 追 8 秒在线生成，2C2G 控制面 + GPU worker | **取消 Web 线**：不建服务端生成、不租 GPU worker、不做 2C2G 部署 | 8 秒只在"骨架已缓存 + 2048 局部 2D"成立，收益全部来自预计算；同样的预计算在本地产出零边际成本、零并发、零运维 |
| 2 | Lite = 浏览器设备端生成（WebGPU 优先 / WASM 回退），`terrain/web/` 为交付形态 | **Lite = 本地进程内生成**（numpy/CPU，可选 ONNX 蒸馏模型），`terrain/web/` 降级为历史产物，不再维护 | Lite 的权威实现本来就是 Python（`terrain/lite/engine.py`），JS 只是渲染壳；换壳成本低于双端维护成本 |
| 3 | Studio 是"打开成品包看一眼"的只读查看器 | **Studio v2 是生成工作站**：任务队列、进度、参数扫描、2D/3D 预览、Pro↔Lite 对比、导出 | 生成能力已经在，缺的是把它变成可操作的产品层 |

### 0.2 关键架构决策

| 决策 | 选择 | 放弃 | 理由 |
|---|---|---|---|
| 算力位置 | 全部本地（用户机器） | 云端 GPU / 2C2G 控制面 | 隐私、可复现、零运维；单用户专业工具 |
| 交付形态 | PySide6 桌面应用 | 浏览器页面 / Web 服务 | 本地生成需要直接读写大文件、长任务、3D 预览 |
| 模型族 | Pro 参考模型 + Lite 摊销近似（两族） | 一套模型硬适配所有硬件 | 旧文档第 15 节的结论仍然成立：两者不是同一套公式 |
| 进程模型 | UI 主进程 + Lite 线程池 + Pro 子进程（QProcess） | 单进程跑到底 | torch/GAN 的内存与崩溃不能污染 UI；子进程可取消、可续跑 |
| 数据契约 | SeedPack / MapPack **v2**（新增时间切片、泥沙预算、provenance） | 各模块自定义中间文件 | 契约是架构的宪法，Pro 产物必须能被 Lite 与 Studio 直接消费 |
| 时间 | 模拟切片 + 插值 + 滑杆 | 只输出终态 | 450 Myr 已经算出来了，不暴露等于浪费 |

---

## 1 · 现状审计

### 1.1 已具备的能力（v12）

- **板块演化**：球面欧拉旋转、后向映射、边界分类、聚合期速度场、终局高程装配（`terrain/core/plate_sim.py`），450 Myr / 720 步。
- **细节生成**：GAN 瓦片合成 + 全局 CDF 校准（`terrain/core/generate_world.py`，权重 `terrain/checkpoints/best_terrain_gan_v5*.pth`）。
- **气候**：温度/降水/Whittaker 群系/雪线（`terrain/modules/climate_module.py`）。
- **水文**：D8 汇流 + 粒子侵蚀（`terrain/modules/hydro_module.py`）。
- **渲染/3D**：卫星风格 2D、OBJ/glTF 导出（`terrain/core/export_3d.py`）。
- **封装层**：Pack 读写与校验（`terrain/pack/`）、Pro 单种子生产（`terrain/pro/produce.py`）、Lite 引擎与 CLI（`terrain/lite/`）、3D LOD（`terrain/pro/lod.py`）、瓦片（`terrain/pro/tiles.py`）、基准（`terrain/bench/`）。

### 1.2 五个结构性不足（沿用深拆结论，按 v13 重新排序）

| 优先级 | 不足 | 现状证据 | v13 处理 |
|---|---|---|---|
| P0 | **时间被浪费**：450 Myr 只输出终态 | `run_plate_sim` 只回传末帧 | 时间切片 4D（§4.2） |
| P0 | **质量/能量不守恒**：侵蚀无全局泥沙预算，海平面固定 | 粒子侵蚀局部沉积，无收支约束 | Exner + 动态海平面（§4.3） |
| P1 | **没有岩石圈力学**：无 flexure/Airy，海岸线偏"规则装配" | `plate_sim` 无载荷补偿 | 轻量均衡（§4.4） |
| P1 | **耦合单向**：气候不改侵蚀，侵蚀不改均衡 | 模块串行、无回馈 | 季节气候 + 反馈项（§4.5），先弱耦合 |
| P2 | **Earth 偏置**：GAN/CDF 把世界拉回地球 | 全局直方图校准 | GAN 物理约束 + 风格条件（§4.6），只做小权重 |

### 1.3 代码地图与缺口

| 模块 | 职责 | 现状态 | v13 缺口 |
|---|---|---|---|
| `terrain/core/` | 模拟、GAN、气候装配、3D 导出 | 可用，单次全链路 | 缺时间切片、守恒、均衡；缺可中断的阶段钩子 |
| `terrain/pack/` | SeedPack/MapPack 读写、校验、CLI | v1 契约 | 需升 v2（切片/泥沙/provenance） |
| `terrain/pro/` | `produce` 单种子生产、`studio` 只读查看、`tiles`、`lod` | 生产可跑；Studio 仅能开包看预览与导出 3D | Studio 需变成生成工作站；produce 需任务化、进度化、可恢复 |
| `terrain/lite/` | numpy 版轻量渲染（权威实现）+ CLI | 可用 | 需参数扫描、切片插值、对比模式、误差报告 |
| `terrain/web/` | 浏览器 Lite demo（JS/WGSL） | 历史实现 | **冻结**，不新增功能；文档标注"已由 PySide6 取代" |
| `main.py` | — | 空文件 | 作为应用入口：`python main.py` 启动 Studio |

### 1.4 结论

引擎"能生成像世界的地形"已经成立；缺的是**时间、守恒、反馈**（模型层）与**一个能驱动它们的工作站**（产品层）。v13 的全部工作围绕这两件事，不再扩张部署形态。

---

## 2 · 目标架构（本地单机）

### 2.1 分层

```
┌───────────────────────── PySide6 Studio（主进程） ─────────────────────────┐
│  Project / Seed 管理 │ 参数面板 │ 2D 画布 │ 时间滑杆 │ 3D 视图 │ 任务队列 │ 对比 │
└──────────▲───────────────────────────▲───────────────────────────────────┘
           │ Qt Signal（进度/结果）     │ QProcess JSONL 事件
┌──────────┴───────────┐     ┌─────────┴──────────────────────────────────┐
│ Lite Worker          │     │ Pro Worker（独立子进程）                    │
│ QThreadPool          │     │ `python -m terrain.pro.produce --jsonl`     │
│ terrain.lite.engine  │     │ 板块 → GAN → 气候 → 水文 → 切片 → 瓦片/3D   │
└──────────▲───────────┘     └─────────▲──────────────────────────────────┘
           └──────────── SeedPack / MapPack v2（磁盘契约）──────────────────┘
                                 │
                       terrain/output/<world>/…
```

三条不变量：

1. **契约唯一**：进程之间只通过 Pack 文件与 JSONL 事件通信，不共享内存对象。
2. **可复现**：任何产物都能由 `seed + tier + 参数 + 模型版本` 重建；`provenance` 写进包内。
3. **可取消**：Pro 任意阶段可中止，已完成的切片/瓦片落盘，下次续跑。

### 2.2 进程 / 线程模型

| 角色 | 载体 | 约束 |
|---|---|---|
| UI | 主线程 | 只做绘制与事件；禁止调用任何 >50ms 的生成函数 |
| Lite 渲染 | `QThreadPool`（numpy，单任务） | 同一时刻只跑一个全尺寸渲染；旧任务用版本号作废 |
| Pro 生产 | `QProcess` 子进程 | `--jsonl` 输出阶段事件；内存超限/崩溃不拖垮 UI；`terminate()` 可取消 |
| 文件监视 | `QFileSystemWatcher` | 产物落盘即刷新产物树 |

### 2.3 数据流

```
seed + tier
   └─ Pro：DEM(2048/4096) → 切片 → SeedPack v2（骨架/气候/水文/风格）
                                   └─ Lite：局部 512–2048 细节 + 风格着色 → PNG
                                   └─ Pro：MapPack v2（瓦片 + 3D LOD + 印刷 PDF）
```

### 2.4 存储布局

```
terrain/output/<world_id>/
  dem/world.npy               # 终态 DEM
  slices/slice_000.npz …      # 时间切片（v2 新增）
  packs/<world>.seedpack
  packs/<world>.mappack
  tiles/z0..z3/
  lods/lod512.glb …
  reports/timing.json  quality.json
```

---

## 3 · 共享契约：Pack v2

### 3.1 SeedPack v2 新增字段

| 字段 | 类型 | 用途 | 兼容 |
|---|---|---|---|
| `slices[]` | `{time_myr, base.npy, sea_level}` | Lite 时间滑杆插值 | 缺省 = 只有终态 |
| `sediment` | `float32 (H,W)` | 侵蚀—沉积预算可视化、守恒校验 | 可选 |
| `provenance` | `{model_version, seed, tier, params_hash, git}` | 复现与误差归因 | 必有 |
| `lite_hints` | `{coast_eps, slope_ref, hydro_min_acc}` | Lite 侧参数对齐 Pro | 必有 |
| `crs` | `{projection, lat0, lon0, res_deg}` | 局部重投影 | 必有 |

### 3.2 规则

- v1 包**只读兼容**（加载时补默认值，写回必升 v2）。
- `pack_hash` 覆盖 `manifest.json` 与所有 entry 的 sha256；任何字段变更都必须改版本号。
- `python -m terrain.pack verify <pack>` 是 CI 与 Studio 的共同闸门。

---

## 4 · Terrain Pro 升级（工作站级离线生成）

> 定位：**工作站上的离线生产**。不追交互时延，只追真实度上限与可复现；单次分钟到小时级，允许排队、断点、批处理。

### 4.1 档位

| 档位 | 模型/渲染 | 时间 | 硬件 |
|---|---|---|---|
| Draft | 现有骨架 + 简化 GAN/气候；2048 全球；低 LOD 3D | 1–3 min | 8 核 / 16GB |
| Standard | 骨架 + 时间切片 + 局部 2048 + 河湖 + 2K 3D | 3–10 min | 8–12 核 / 16GB |
| Ultra | 守恒侵蚀 + flexure + 季节气候 + 4096 + GAN 约束 + PBR | 10–60 min | 12–16 核 / 32GB / 12GB+ GPU |
| Research | 8192、集合/敏感性、可选全 GCM | 小时—天 | 24+ 核 / 64GB / 24GB GPU |

### 4.2 P0-1 · 时间切片（4D）

**目标**：把已有 450 Myr 变成可看、可绑定、可对比的资产。

数据模型（新增 `terrain/core/evolution.py`）：

```python
EvolutionState:
  time_myr: float
  elevation: (H,W) float32
  crust:     (H,W) int8
  plates:    (H,W) int16
  orog:      (H,W) float32
  sediment:  (H,W) float32     # v2
  sea_level: float             # v2
  climate:   {T, P, biome, snow}
  provenance: {model_version, seed, stage}
```

实现要点：

1. 在 `run_plate_sim` 主循环按 `t ∈ {0, 150, 260, 450} Myr` 落盘 `npz`（int16 高程 + zlib），并同步调用气候/水文快照。
2. `generate_world` 接收 `slices` 输出目录；GAN 只对终态做全量，早期帧用骨架 + Lite 着色（避免 4× GAN 成本）。
3. Studio 时间滑杆按 `z(t) = (1-α) z_i + α z_{i+1}` 插值；标注"插值 ≠ 演化"。

**验收**：四帧可复现（同 seed 两次 bitwise）；切片间海岸线位移连续（无跳变）；单帧 <200MB、加载 <1s。

### 4.3 P0-2 · 质量守恒侵蚀 + 动态海平面

$$\frac{\partial \eta}{\partial t} = U - \nabla\cdot q_s,\qquad q_s = K Q^m S^n$$

- 把 `droplet_erosion` 的局部侵蚀/沉积改为**泥沙收支**：`Σ侵蚀 = Σ沉积 + Σ出口`，误差 <1%。
- 海平面由冰量/洋盆体积驱动：`sea_level = f(ice_volume, basin_capacity)`，河流基准面随海平面下降。
- 影响面：三角洲、河口、陆架、内陆盆地；需要重调 `_style_colors` 的海岸带配色。

**验收**：泥沙预算误差 <1%；海平面变化可解释（有曲线图）；三角洲面积与河流等级正相关。

### 4.4 P1-1 · flexure / Airy 均衡

$$D\nabla^4 w + \rho_m g w = q,\qquad D = \frac{E T_e^3}{12(1-\nu^2)}$$

- 先做 Airy（`w = q/ρ_m g` 的局部补偿）一版，再接弹性板挠曲（FFT 频域求解，低成本）。
- 用于山脉山根/盆地补偿；对 2048 分辨率足够。

**验收**：山—盆地壳厚度/高程关系合理；现有大陆形态不崩坏（与 v12 的 RMSE 上有界）。

### 4.5 P1-2 · 季节化气候（1D EBM + 水汽）

$$C\frac{\partial T}{\partial t} = S(1-\alpha) - (A + BT) + D\nabla^2 T$$

- 12 月/4 季太阳辐射；水汽输送 + 海洋混合层；ITCZ 迁移。
- 不做全 GCM：用查找表 + 代理近似，先保证季风/雨影/雪线振幅合理。

**验收**：全球平均温度无年际漂移；季风区降水季节差 > 阈值；雪线随季节移动可视化正确。

### 4.6 P1-3 · GAN 物理约束

$$L = L_{adv} + \lambda_{rec}L_{rec} + \lambda_s L_{slope} + \lambda_f L_{flow} + \lambda_t L_{tile}$$

- `L_slope` 约束坡向/坡度，`L_flow` 约束河网连续性，`L_tile` 约束重叠区（stride 208 / overlap 48）。
- 权重从小开始（λ_s、λ_f ≈ 0.05），避免压平纹理；风格条件另加 embedding。

**验收**：100% 缩放接缝下降 >30%；河网/海岸不被破坏；盲评不低于 v12。

### 4.7 Pro 在 PySide6 中的形态

- **任务队列**：`seed × tier × 参数` 组成作业单元，串行/并行 1–2 个（受内存约束）。
- **进度**：`produce --jsonl` 每阶段输出 `{stage, pct, eta, mem_mb}`，UI 直接消费；不做假进度条。
- **断点续跑**：`--reuse-dem`、切片级跳过、瓦片级跳过；失败任务保留中间产物。
- **产物树**：包/切片/瓦片/LOD/报告分层展示，双击即预览。
- **导出**：MapPack、PNG、PDF（300 DPI）、glTF/OBJ、`timing.json`。

### 4.8 Pro 验收

| 项 | 指标 |
|---|---|
| 2048 Standard | <10 min（8 核 / 16GB） |
| 4096 Ultra | <30 min（12 核 / 32GB / GPU） |
| 3D | LOD 无裂缝；1080p 60fps |
| 复现 | 同参两次 pack_hash 一致 |
| 可中断 | 任意阶段取消后无残留临时文件 |

---

## 5 · Terrain Lite 升级（本地秒级派生）

### 5.1 定位变化

旧：手机/核显浏览器里"设备端生成"。新：**同一台工作站上、Studio 进程内的秒级预览与派生**——用途是调参、比对、快速出图、给 Pro 任务做预检，而不是替代 Pro。

### 5.2 算法（保持模型族不变）

$$z_L(x,t) = B_s(x,t) + \sum_{k=1}^{K} a_k N_k(x) + R_\varphi(x)$$

$$d_s(x) = \operatorname{signedDist}(x, \mathrm{coast}),\qquad w(x)=\mathrm{clip}\!\left(\frac{d_s(x)}{\varepsilon},0,1\right)$$

- `B_s`：SeedPack 基场双线性/样条采样；`N_k`：多尺度 fBm/ridged；`R_φ`：可选蒸馏残差。
- 海岸 SDF 过渡 + 2–3 轮坡度限制侵蚀 + 预计算河网叠加 + 气候 LUT / 9 群系。
- 时间：`z(t)=(1-α)z_i+αz_{i+1}`，滑杆实时。

### 5.3 本地性能目标（需实测）

| 尺寸 | 目标 | 说明 |
|---|---|---|
| 512² | ≤0.4 s | 滑块拖动实时 |
| 1024² | ≤1.5 s | 调参主用 |
| 2048² | ≤4 s | 出图用；拖动时降级到 512 预览 |

用 `terrain/bench` 固化三档基准，Studio 内显示实测耗时与峰值内存（现 CLI 已输出）。

### 5.4 质量边界与误差

$$e(x) = z_A(x) - z_P(x),\qquad E_{\mathrm{RMSE}} = \sqrt{\frac{1}{N}\sum_x e^2(x)}$$

- 保证：同 Pack 的海岸/山脉/气候/河网一致；局部观感接近 Pro Standard。
- 不保证：新 seed 的 450 Myr、4096 细节、全 PBR、全量 GAN 纹理。
- 对比模式：Studio 左右分屏同区域渲染 Pro 与 Lite，输出 RMSE / SSIM / 海岸线偏移 / 河流连续性四项指标。

### 5.5 验收

| 项 | 指标 |
|---|---|
| 确定性 | 同参两次渲染 bitwise 一致 |
| 时延 | 512² ≤0.4s、1024² ≤1.5s、2048² ≤4s |
| 内存 | 2048² 峰值 <1GB |
| 误差 | 相对 Pro Standard：SSIM ≥0.85，海岸线偏移 ≤2px |

---

## 6 · PySide6 工作站（Studio v2）

现状：`terrain/pro/studio.py` 只能开包、看预览、导出 3D（约 150 行）。v13 把它扩成主应用，入口 `main.py`。

### 6.1 窗口结构

```
┌───────────────────────────────────────────────────────────────────────┐
│ 菜单：文件 / 世界 / 任务 / 视图 / 帮助                                   │
├───────────────┬───────────────────────────────────────┬───────────────┤
│ 左：项目树     │            中央：画布                  │ 右：检查器     │
│  World        │  ┌───────────────┬───────────────┐    │ 参数面板       │
│   ├ dem       │  │ 2D 地图        │ 3D 视图        │    │  seed/tier    │
│   ├ slices    │  │ (QGraphicsView)│(QOpenGLWidget)│    │  气候/水文     │
│   ├ packs     │  └───────────────┴───────────────┘    │  风格/配色     │
│   ├ tiles     │  [时间滑杆] 0 ──●────── 450 Myr        │  ─────────    │
│   └ lods      │  状态：Lite 1.2s · 512² · 峰值 380MB   │ [生成][对比]   │
├───────────────┴───────────────────────────────────────┴───────────────┤
│ 底部：任务队列（Pro 子进程 / Lite 渲染 / 导出） 进度 · ETA · 取消        │
└───────────────────────────────────────────────────────────────────────┘
```

### 6.2 功能清单

| 模块 | 功能 | 依赖 | 优先级 |
|---|---|---|---|
| 项目树 | 世界/包/切片/瓦片/LOD 浏览，拖拽打开 | Pack v2 | P0 |
| 画布 | pan/zoom、瓦片渐进加载、双缓冲 | QGraphicsView | P0 |
| 时间滑杆 | 切片插值、播放、帧导出 | slices | P0 |
| 参数面板 | seed/tier/质量/风格；改动即 Lite 预览（防抖 200ms） | lite.engine | P0 |
| 任务队列 | 提交 Pro 作业、进度、取消、续跑、日志 | QProcess | P0 |
| 对比视图 | Pro vs Lite 分屏 + 指标条 | 误差模块 | P1 |
| 3D 视图 | LOD mesh、法线、光照、glTF 导出 | lod/export_3d | P1 |
| 批量扫描 | seed 区间/参数网格，产物矩阵 | produce | P1 |
| 报告 | 生成 HTML/PDF 质检报告 | matplotlib | P2 |
| 快捷键/主题 | 浅深色、键位表 | QSS | P2 |

### 6.3 交互与线程规则

- 所有生成走 worker；UI 只收 `progress/finished/failed` 信号。
- Lite 渲染用"请求版本号"：新请求到来即作废旧结果，避免拖动滑杆时排队。
- 大图传输用 `QImage` 零拷贝（`ndarray.data` + `Format_RGB888`），在 worker 内完成后切到主线程 `setPixmap`。
- Pro 子进程统一 `--jsonl` 协议，UI 侧只解析 JSON 行，不解析人类可读日志。

### 6.4 3D 方案取舍

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| `QOpenGLWidget` + 自写 shader | 无额外依赖、可控 | 需自己写相机/材质 | **首选**（P1） |
| Qt Quick 3D | 现成材质/光照 | 引入 QML 与新概念 | 备选 |
| 嵌入外部查看器 | 省事 | 破坏"单窗口"体验 | 不做 |

### 6.5 打包分发

- PyInstaller / py2app 打包为 `.app`；torch 体积大，Draft/Standard 档可做"按需加载权重"。
- 权重与示例 Pack 作为可选资源包，首次启动引导选择。
- macOS 需签名与公证；开发期用 `QT_QPA_PLATFORM=offscreen` 做冒烟（现有 Studio 已支持）。

---

## 7 · 路线图（单人，约 14–18 周）

| 里程碑 | 周期 | 交付 | 验收 | 依赖 |
|---|---|---|---|---|
| **M0 契约与基座** | 2 周 | Pack v2 schema + `verify/info` 升级；`main.py` 启动 Studio v2 骨架；进程/线程框架 | v1 包可读、v2 包可写；UI 空壳可启动 | — |
| **M1 Lite 本地闭环** | 2–3 周 | Studio 内选 seed/Pack → Lite 生成 → 2D 预览 → 导出 PNG；参数面板防抖 | 512² ≤0.4s、1024² ≤1.5s；无 UI 卡顿 | M0 |
| **M2 Pro 生产闭环** | 3–4 周 | `produce --jsonl`；任务队列、进度、取消、续跑；产物树；3D 视图 | 2048 Standard <10min；取消无残留；3D 60fps | M0 |
| **M3 时间切片 4D** | 3 周 | `EvolutionState` + 4 帧落盘 + 滑杆插值 + 帧导出 | 四帧可复现；切片连续；<200MB/帧 | M2 |
| **M4 守恒侵蚀** | 3 周 | Exner 泥沙收支 + 动态海平面 + 海岸/三角洲重调 | 预算误差 <1%；海平面曲线可解释 | M3 |
| **M5 均衡与气候** | 3–4 周 | Airy/flexure 轻量版；季节 EBM + 水汽；GAN 物理约束（小权重） | 山—盆关系合理；无温度漂移；接缝下降 >30% | M4 |
| **M6 验收与分发** | 2 周 | 基准固化、质检报告、打包签名、文档 | 所有指标过线；`.app` 可分发 | M5 |

与旧路线图相比：去掉 Web（原 8–10 周）与 2C2G 压测（原 2–3 周），省下的时间投入时间切片与守恒——因为它们才是差异化。

---

## 8 · 验收与质量协议

### 8.1 模型 / 科学

- 质量：泥沙收支闭合；海平面与冰量一致。
- 能量：EBM 全球平均温度稳定，无年际漂移。
- 形态：hypsometry、Hack 指数、Strahler 级序、海岸分形维数。
- 时间：切片间海岸/河流/群系连续，无跳变。

### 8.2 工程

- Pack v2 全链路 `verify` 通过；同参两次产物 hash 一致。
- Lite：见 §5.5；Pro：见 §4.8。
- UI：主线程单次阻塞 <50ms；10 万像素级画布 pan/zoom ≥50fps。
- 崩溃隔离：Pro 子进程异常退出，UI 保持可用并给出可读错误。

### 8.3 公平比较协议

同一 seed、同一区域、同一投影、同一尺寸下比较 Pro 与 Lite；同时报告 RMSE、SSIM、海岸线偏移、河流连续性，外加盲评 rubric（5 级）。禁止用不同区域截图对比。

---

## 9 · 风险与不做清单

| 风险 | 触发条件 | 缓解 |
|---|---|---|
| torch 打包体积过大 | PyInstaller 产物 >2GB | 权重按需下载；Draft/Standard 走轻量路径 |
| 长任务被打断 | 用户关闭窗口/休眠 | 子进程写 checkpoint；`--resume` 续跑；关闭前确认 |
| 内存峰值 | 4096 + 切片 + 3D 同时驻留 | 分阶段释放；切片 int16 + zlib；限制并发 1 |
| GAN 约束压平纹理 | λ 过大 | 小权重起步；盲评回归门禁 |
| 契约漂移 | 各模块自行加字段 | Pack v2 单一 schema；CI 跑 verify |
| macOS 签名/公证 | 分发阶段 | 提前准备证书；开发期用本地构建 |

**不做**：

- 不做浏览器版、不做 Web 服务、不做 8 秒在线生成、不做 2C2G / GPU worker 部署。
- 不做手机端 WebGPU Lite。
- 不做完整地图编辑器（区域/城市/道路编辑属于上层产品，例如 Matrees 式工具）。
- 不做实时协作 / CRDT / Wiki（属于 VVD 的领域）。
- 不在 UI 主线程跑任何模拟或 GAN。

---

## 10 · 与 Matrees / VVD 的边界

Terrain 只做"世界怎么长出来"，输出可复现的 Pack：

| 交付物 | 内容 | 消费方 |
|---|---|---|
| SeedPack v2 | 骨架、气候、水文、风格、切片 | Lite、Studio、第三方预览器 |
| MapPack v2 | 瓦片、3D LOD、印刷资产 | 地图产品（Matrees 式） |
| `provenance` | seed/模型版本/参数/时间 | 时间线/Wiki（VVD 式） |

一句话：**Matrees 让地图可读，VVD 让世界可写，Terrain 让世界可演化**；三者是上下游，不是竞品。

---

## 附录 A · 关键公式

**Lite 基场 + 残差**

$$z_L(x,t) = B_s(x,t) + \sum_{k=1}^{K} a_k N_k(x) + R_\varphi(x)$$

**Lite 海岸 SDF**

$$d_s(x)=\operatorname{signedDist}(x,\mathrm{coast}),\qquad w(x)=\mathrm{clip}\!\left(\frac{d_s(x)}{\varepsilon},0,1\right)$$

**时间插值**

$$z(t)=(1-\alpha)z_i+\alpha z_{i+1},\qquad \alpha=\frac{t-t_i}{t_{i+1}-t_i}$$

**近似误差**

$$e(x)=z_A(x)-z_P(x),\qquad E_{\mathrm{RMSE}}=\sqrt{\frac{1}{N}\sum_x e^2(x)}$$

**Exner 泥沙收支**

$$\frac{\partial \eta}{\partial t}=U-\nabla\cdot q_s,\qquad q_s=K Q^m S^n$$

**弹性板挠曲**

$$D\nabla^4 w+\rho_m g w=q,\qquad D=\frac{E T_e^3}{12(1-\nu^2)}$$

**季节能量平衡**

$$C\frac{\partial T}{\partial t}=S(1-\alpha)-(A+BT)+D\nabla^2 T$$

**GAN 物理约束损失**

$$L=L_{adv}+\lambda_{rec}L_{rec}+\lambda_s L_{slope}+\lambda_f L_{flow}+\lambda_t L_{tile}$$

---

## 附录 B · 源码索引

| 主题 | 入口 | 文件 |
|---|---|---|
| 板块演化 / 后向映射 / 边界 | `run_plate_sim`、`_classify` | `terrain/core/plate_sim.py` |
| 主生成链路 / GAN 瓦片 | `generate_world` | `terrain/core/generate_world.py` |
| 3D 导出 | `export_world` | `terrain/core/export_3d.py` |
| 气候 | `temperature_field`、`precipitation_field` | `terrain/modules/climate_module.py` |
| 水文 | `d8_flow_accumulation`、`droplet_erosion` | `terrain/modules/hydro_module.py` |
| Pack 契约 | `SeedPack`、`MapPack`、`verify` | `terrain/pack/io.py`、`terrain/pack/schema.py` |
| Pack 构建 | `build_seedpack_from_world` | `terrain/pack/build.py` |
| Pro 生产 | `produce` | `terrain/pro/produce.py` |
| Studio（待扩） | `_build_window`、`main` | `terrain/pro/studio.py` |
| 瓦片 / LOD | `build_tiles`、`build_lods` | `terrain/pro/tiles.py`、`terrain/pro/lod.py` |
| Lite 引擎 | `LiteEngine.render`、`render_png` | `terrain/lite/engine.py` |
| 基准 | `terrain.bench` | `terrain/bench/` |
| 应用入口 | — | `main.py`（v13 新增实现） |

---

## 附录 C · 术语

- **Pack**：可复现的世界数据包（SeedPack 轻、MapPack 全）。
- **切片（slice）**：某个 `time_myr` 的完整状态快照。
- **摊销近似（amortized approximation）**：Lite 复用 Pro 预计算骨架，只算局部残差。
- **Provenance**：能重建产物的最小元数据集合（seed/版本/参数/时间）。
- **Tier**：Draft / Standard / Ultra / Research 四档生产配置。
