# Terrain Lite · Web

零依赖的设备端渲染 demo（原生 ES module + Canvas，WebGPU 可用时自动启用）。

```bash
cd terrain/web
python3 -m http.server 8000
# 打开 http://localhost:8000/            → 默认 demo.seedpack
# 打开 http://localhost:8000/?pack=demo.seedpack
# 加 &gpu=0 可强制走 JS 回退（对比两端耗时）
```

- `seedpack.js` — 手写 ZIP（DecompressionStream）+ `.npy` 解析，返回 `{base, dem, temp, precip, biome, hydro, style, packHash}`
- `lite_engine.js` — `new LiteEngine(pack).render({size, cx, cy, zoom, quality})` → `{canvas, ms, backend}`
- `index.html` — 拖拽平移 / 滚轮缩放 / 512·1024·2048 / quality 切换 / HUD
- `demo.seedpack` — 复制自 `terrain/output/smoke.seedpack`

后端：`webgpu`（WGSL compute：synth → erode ×N → shade）失败或不可用时自动回退 `js`。
WGSL 编译错误、WebGPU validation error 都会抛回 `render()` 并回退，不会静默黑屏。
需通过 http(s) 打开（`file://` 下 ES module 与 fetch 会被浏览器拦截）。
