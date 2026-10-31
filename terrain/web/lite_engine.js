// lite_engine.js — Terrain Lite 的浏览器端实现（WebGPU 优先，JS 回退）
//
// 与 terrain/lite/engine.py 一一对应：SeedPack 基场（双线性）→ 细节 ridged fBm
// （坡度 × 海岸距离调制）→ 海岸 SDF（精确距离变换）→ 2–3 轮坡度限制侵蚀 →
// 河网叠加 → 9 群系着色 + hillshade。
//
//   const engine = new LiteEngine(pack);              // pack = await loadSeedPack(url)
//   const {canvas, ms, backend} = await engine.render({size: 1024, quality: 'balanced'});
//
// 后端：navigator.gpu 可用时走 WGSL 计算着色器（backend='webgpu'），
// 否则走 JS typed array（backend='js'）。两端共用同一套规则、同一套 hash 噪声，
// 只是执行位置不同；GPU 失败会自动回退。

export const QUALITY = {
  fast: { octaves: 3, erosion: 2, amp: 700 },
  balanced: { octaves: 5, erosion: 2, amp: 900 },
  high: { octaves: 6, erosion: 3, amp: 1100 },
};

export const SEA_LEVEL = 0.5;      // v01 海平面
const COAST_FADE_PX = 8.0;         // 海岸细节衰减带（px @512）
const SLOPE_REF = 2000.0;          // 坡度归一化参考（m/px @512）
const SDF_RADIUS = 24;             // SDF 有效半径（与 Python 的 EDT 等效）
const FREQ0 = 12.0;                // fBm 首层频率
const LACUNARITY = 2.03;
const GAIN = 0.55;

// 米制 <-> v01 拐点（与 terrain/pack/schema.py 同源）
const M_KNOTS = [-11000, -6000, -4000, -2500, -1000, -200, 0,
  100, 400, 1000, 2000, 3500, 5000, 8849];
const V_KNOTS = [0.00, 0.08, 0.22, 0.34, 0.42, 0.47, 0.50,
  0.53, 0.60, 0.70, 0.82, 0.92, 0.97, 1.00];

const DEFAULT_PALETTE = {
  ocean_deep: '#0b2d5b', ocean_shallow: '#2b6ea8', beach: '#d9d3a6',
  plain: '#5d8a4a', forest: '#2f5d33', mountain: '#8a7a66', snow: '#f4f6f8',
};
const DEFAULT_BIOME = ['#b8c7d9', '#4f7a4a', '#b9a35e', '#6b8f4e', '#2f5d33',
  '#cbb98a', '#9cbf5a', '#2e7d32', '#14532d'];
const DEFAULT_HILL = { azimuth: 315, altitude: 45, strength: 0.6 };

// ==================== 1. 基础工具 ====================
const clamp = (v, lo, hi) => (v < lo ? lo : (v > hi ? hi : v));
const clamp01 = (v) => clamp(v, 0, 1);
const smoothstep01 = (t) => {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
};

function interpKnots(v, xs, ys) {
  const n = xs.length;
  if (v <= xs[0]) return ys[0];
  if (v >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= v) lo = mid; else hi = mid;
  }
  const t = (v - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + (ys[hi] - ys[lo]) * t;
}

/** 归一化 v01 → 米制高程（与 Python v2m 同一组拐点）。 */
export function v2m(v) { return interpKnots(v, V_KNOTS, M_KNOTS); }
/** 米制高程 → 归一化 v01。 */
export function m2v(m) { return interpKnots(m, M_KNOTS, V_KNOTS); }

function hexRgb(hex) {
  const s = String(hex).replace('#', '');
  return [parseInt(s.slice(0, 2), 16) / 255,
    parseInt(s.slice(2, 4), 16) / 255, parseInt(s.slice(4, 6), 16) / 255];
}

function axis(S, zoom, center) {
  const out = new Float32Array(S);
  for (let i = 0; i < S; i++) out[i] = center + ((i + 0.5) / S - 0.5) / zoom;
  return out;
}

/** 归一化坐标下的单点双线性采样（行=y，列=x）。 */
function bilinearAt(field, n, wx, wy) {
  if (!field) return 0;
  const fx = clamp(wx * n - 0.5, 0, n - 1);
  const fy = clamp(wy * n - 0.5, 0, n - 1);
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, n - 1), y1 = Math.min(y0 + 1, n - 1);
  const tx = fx - x0, ty = fy - y0;
  const a = field[y0 * n + x0], b = field[y0 * n + x1];
  const c = field[y1 * n + x0], d = field[y1 * n + x1];
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

function nearestAt(field, n, wx, wy) {
  const ix = clamp(Math.round(wx * n - 0.5), 0, n - 1);
  const iy = clamp(Math.round(wy * n - 0.5), 0, n - 1);
  return field[iy * n + ix];
}

// ==================== 2. 确定性噪声（JS 与 WGSL 同一函数）====================
const U32 = 0x100000000;

/** 整数 hash → [0,1)。JS 与 WGSL 使用完全相同的位运算。 */
export function hash01(ix, iy, oct, seed) {
  let h = (Math.imul(ix | 0, 0x27d4eb2d) ^ Math.imul(iy | 0, 0x165667b1) ^
    Math.imul(oct | 0, 0x9e3779b1) ^ Math.imul(seed | 0, 0x85ebca6b)) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x2c1b3c6d) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0x297a2d39) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h / U32;
}

function valueNoise(S, freq, oct, seed, py, px) {
  const inv = freq / S;
  const ty = (py + 0.5) * inv, tx = (px + 0.5) * inv;
  const fy0 = Math.floor(ty), fx0 = Math.floor(tx);
  const fy = smoothstep01(ty - fy0), fx = smoothstep01(tx - fx0);
  const iy0 = clamp(fy0, 0, freq), ix0 = clamp(fx0, 0, freq);
  const iy1 = iy0 + 1, ix1 = ix0 + 1;
  const v00 = hash01(ix0, iy0, oct, seed), v01 = hash01(ix1, iy0, oct, seed);
  const v10 = hash01(ix0, iy1, oct, seed), v11 = hash01(ix1, iy1, oct, seed);
  return (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy;
}

/** 多尺度 ridged fBm → [0,1]（每层 1-|2n-1|）。 */
function ridgedFbm(S, octaves, seed, py, px) {
  let total = 0, amp = 1, norm = 0, freq = FREQ0;
  for (let k = 0; k < octaves; k++) {
    const f = Math.round(clamp(freq, 2, 256));
    const n = valueNoise(S, f, k + 1, seed, py, px);
    total += amp * (1 - Math.abs(2 * n - 1));
    norm += amp;
    amp *= GAIN;
    freq *= LACUNARITY;
  }
  return total / Math.max(norm, 1e-6);
}

// ==================== 3. 参数 ====================
/** 归一化渲染参数（等价于 Python LiteParams）。 */
export function makeParams(p = {}) {
  const size = Math.round(p.size ?? 512);
  if (!(size >= 64 && size <= 4096)) throw new Error(`size 应在 64..4096，实际 ${size}`);
  const zoom = Number(p.zoom ?? 1);
  if (!(zoom >= 0.05 && zoom <= 64)) throw new Error(`zoom 应在 0.05..64，实际 ${zoom}`);
  const quality = p.quality ?? 'balanced';
  if (!QUALITY[quality]) throw new Error(`quality 应为 fast/balanced/high，实际 ${quality}`);
  return {
    size, zoom, quality,
    cx: clamp01(Number(p.cx ?? 0.5)),
    cy: clamp01(Number(p.cy ?? 0.5)),
  };
}

/** 参数指纹（决定噪声种子）：与 Python 的 sha256 不同，但同样只依赖参数。 */
function paramDigest(p) {
  const s = JSON.stringify([p.size, p.zoom.toFixed(6), p.cx.toFixed(6),
    p.cy.toFixed(6), p.quality]);
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h | 0;
}

// ==================== 4. 样式 ====================
function resolveStyle(style) {
  const st = style || {};
  const pal = { ...DEFAULT_PALETTE, ...(st.palette || {}) };
  const names = (st.biome_colors && st.biome_colors.length ? st.biome_colors : DEFAULT_BIOME).slice(0, 9);
  while (names.length < 9) names.push(DEFAULT_BIOME[names.length]);
  const hill = { ...DEFAULT_HILL, ...(st.hillshade || {}) };
  const rgb = {};
  for (const k of Object.keys(pal)) rgb[k] = hexRgb(pal[k]);
  return {
    palette: rgb,
    river: hexRgb(pal.river || pal.ocean_shallow),
    biome: names.map(hexRgb),
    hill,
  };
}

// ==================== 5. 海岸 SDF（Felzenszwalb 精确距离变换）====================
function edt1d(f, n, d, v, z) {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const dq = q - v[k];
    d[q] = dq * dq + f[v[k]];
  }
}

/** 到最近 feature 像素的平方距离（feature 处为 0）。 */
function edtSq(mask, S) {
  const N = S * S;
  const g = new Float32Array(N);
  const f = new Float64Array(S), d = new Float64Array(S);
  const v = new Int32Array(S), z = new Float64Array(S + 1);
  const INF = 1e20;
  for (let i = 0; i < N; i++) g[i] = mask[i] ? 0 : INF;
  for (let x = 0; x < S; x++) {
    for (let y = 0; y < S; y++) f[y] = g[y * S + x];
    edt1d(f, S, d, v, z);
    for (let y = 0; y < S; y++) g[y * S + x] = d[y];
  }
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) f[x] = g[y * S + x];
    edt1d(f, S, d, v, z);
    for (let x = 0; x < S; x++) g[y * S + x] = d[x];
  }
  return g;
}

function coastSdf(land, S) {
  const water = new Uint8Array(S * S);
  for (let i = 0; i < land.length; i++) water[i] = land[i] ? 0 : 1;
  const dLand = edtSq(land, S), dWater = edtSq(water, S);
  const sdf = new Float32Array(S * S);
  for (let i = 0; i < sdf.length; i++) {
    sdf[i] = land[i]
      ? Math.sqrt(dLand[i]) - 0.5
      : -(Math.sqrt(dWater[i]) - 0.5);
  }
  return sdf;
}

// ==================== 6. 河网栅格化 ====================
function stampDisc(mask, S, py, px, radius) {
  const r = Math.max(0.5, radius);
  const y0 = Math.max(0, Math.floor(py - r)), y1 = Math.min(S - 1, Math.ceil(py + r));
  const x0 = Math.max(0, Math.floor(px - r)), x1 = Math.min(S - 1, Math.ceil(px + r));
  if (x1 < x0 || y1 < y0) return;
  const r2 = r <= 0.7 ? 0.25 : r * r;
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - py;
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - px;
      if (dy * dy + dx * dx <= r2) mask[y * S + x] = 1;
    }
  }
}

/** 从边界泛洪，填掉封闭空洞（等价 scipy binary_fill_holes）。 */
function fillHoles(mask, S) {
  const N = S * S;
  const outside = new Uint8Array(N);
  const stack = new Int32Array(N);
  let sp = 0;
  const push = (i) => {
    if (!mask[i] && !outside[i]) { outside[i] = 1; stack[sp++] = i; }
  };
  for (let x = 0; x < S; x++) { push(x); push((S - 1) * S + x); }
  for (let y = 0; y < S; y++) { push(y * S); push(y * S + S - 1); }
  while (sp > 0) {
    const i = stack[--sp];
    const y = (i / S) | 0, x = i - y * S;
    if (y > 0) push(i - S);
    if (y < S - 1) push(i + S);
    if (x > 0) push(i - 1);
    if (x < S - 1) push(i + 1);
  }
  for (let i = 0; i < N; i++) if (!outside[i]) mask[i] = 1;
  return mask;
}

/** pack 折线（[y,x]，pack 像素坐标）→ 输出分辨率河网掩膜。 */
export function rasterHydro(hydro, packSize, S, zoom, cx, cy) {
  const mask = new Uint8Array(S * S);
  if (!hydro) return mask;
  const k = (zoom * S) / packSize;
  const half = S / 2 - 0.5;
  const toOut = (pts) => {
    const ys = new Float32Array(pts.length), xs = new Float32Array(pts.length);
    for (let i = 0; i < pts.length; i++) {
      ys[i] = (pts[i][0] / packSize - cy) * zoom * S + half;
      xs[i] = (pts[i][1] / packSize - cx) * zoom * S + half;
    }
    return { ys, xs };
  };

  for (const river of hydro.rivers || []) {
    if (!river || river.length < 2) continue;
    const level = Math.min(3, Math.floor(river.length / 30));
    const radius = Math.max(0.5, (1 + 2 * level) * k * 0.5);
    const { ys, xs } = toOut(river);
    for (let i = 0; i < ys.length; i++) stampDisc(mask, S, ys[i], xs[i], radius);
  }

  const lakes = new Uint8Array(S * S);
  let any = false;
  for (const lake of hydro.lakes || []) {
    if (!lake || lake.length < 3) continue;
    const { ys, xs } = toOut(lake);
    any = true;
    for (let i = 0; i < ys.length; i++) stampDisc(lakes, S, ys[i], xs[i], Math.max(0.5, 1.2 * k));
  }
  if (any) {
    fillHoles(lakes, S);
    for (let i = 0; i < mask.length; i++) if (lakes[i]) mask[i] = 1;
  }
  return mask;
}

// ==================== 7. 群系 LUT ====================
function biomeAt(temp, precip) {
  if (temp < -5) return 0;
  if (temp < 8) return precip < 250 ? 0 : 1;
  if (temp < 20) {
    if (precip < 350) return 2;
    if (precip < 900) return 3;
    return 4;
  }
  if (precip < 300) return 5;
  if (precip < 900) return 6;
  if (precip < 1800) return 7;
  return 8;
}

// ==================== 8. 引擎 ====================
export class LiteEngine {
  /**
   * @param {object} pack loadSeedPack() 的返回值
   * @param {{canvas?: HTMLCanvasElement, webgpu?: boolean}} [opts]
   */
  constructor(pack, opts = {}) {
    if (!pack || !pack.base || !pack.base.data) throw new Error('LiteEngine 需要 seedpack');
    this.pack = pack;
    this.seed = (pack.seed | 0);
    this.packSize = pack.size | 0;
    this.style = pack.style || null;
    this.opts = opts;
    this.canvas = opts.canvas || null;
    this.preferGPU = opts.webgpu !== false;
    this._gpu = null;          // null=未尝试 / false=不可用 / object=已就绪
    this.backend = 'js';
  }

  _ensureCanvas(S) {
    if (!this.canvas) {
      if (typeof document !== 'undefined') {
        this.canvas = document.createElement('canvas');
      } else if (typeof OffscreenCanvas !== 'undefined') {
        this.canvas = new OffscreenCanvas(S, S);
      } else {
        throw new Error('没有可用的 canvas，请通过 opts.canvas 传入');
      }
    }
    if (this.canvas.width !== S || this.canvas.height !== S) {
      this.canvas.width = S;
      this.canvas.height = S;
    }
    return this.canvas;
  }

  // ---- 与分辨率无关的 CPU 前置：基场 / SDF / 河网 / 群系 ----
  _prologue(p) {
    const S = p.size, N = S * S, N0 = this.packSize;
    const wx = axis(S, p.zoom, p.cx);
    const wy = axis(S, p.zoom, p.cy);
    const baseSrc = this.pack.base.data;
    const demSrc = this.pack.dem ? this.pack.dem.data : null;
    const tempSrc = this.pack.temp ? this.pack.temp.data : null;
    const precipSrc = this.pack.precip ? this.pack.precip.data : null;
    const biomeSrc = this.pack.biome ? this.pack.biome.data : null;

    const base01 = new Float32Array(N);
    const height0 = new Float32Array(N);
    const land = new Uint8Array(N);
    const temp = new Float32Array(N);
    const precip = new Float32Array(N);
    const biome = new Uint8Array(N);

    for (let y = 0; y < S; y++) {
      const ny = wy[y];
      for (let x = 0; x < S; x++) {
        const i = y * S + x, nx = wx[x];
        const b = clamp01(bilinearAt(baseSrc, N0, nx, ny));
        const h = demSrc ? bilinearAt(demSrc, N0, nx, ny) : v2m(b);
        base01[i] = b;
        height0[i] = h;
        land[i] = b >= SEA_LEVEL ? 1 : 0;
        if (biomeSrc) {
          biome[i] = nearestAt(biomeSrc, N0, nx, ny);
        } else {
          const t = tempSrc ? bilinearAt(tempSrc, N0, nx, ny)
            : 30 - 55 * Math.abs(2 * ny - 1) - Math.max(h, 0) / 60;
          const pr = precipSrc ? bilinearAt(precipSrc, N0, nx, ny) : 900;
          temp[i] = t;
          precip[i] = pr;
          biome[i] = biomeAt(t, pr);
        }
      }
    }

    const sdf = coastSdf(land, S);
    const river = rasterHydro(this.pack.hydro, N0, S, p.zoom, p.cx, p.cy);
    const nseed = ((this.seed ^ paramDigest(p)) >>> 0) & 0x7fffffff;
    return { S, base01, height0, land, temp, precip, biome, sdf, river, nseed };
  }

  _stats(p, height, land, river, biome, ms, backend) {
    let hmin = Infinity, hmax = -Infinity, landN = 0, riverN = 0;
    const seen = new Uint8Array(9);
    for (let i = 0; i < height.length; i++) {
      const h = height[i];
      if (h < hmin) hmin = h;
      if (h > hmax) hmax = h;
      if (land[i]) landN++;
      if (river[i]) riverN++;
      seen[biome[i] & 15] = 1;
    }
    let bio = 0;
    for (let k = 0; k < 9; k++) if (seen[k]) bio++;
    return {
      size: p.size, quality: p.quality, zoom: p.zoom, cx: p.cx, cy: p.cy,
      octaves: QUALITY[p.quality].octaves,
      erosion_rounds: QUALITY[p.quality].erosion,
      seed: this.seed, pack_size: this.packSize,
      land_frac: +(landN / height.length).toFixed(4),
      river_pixels: riverN, biome_count: bio,
      height_min: +hmin.toFixed(1), height_max: +hmax.toFixed(1),
      backend, ms: +ms.toFixed(1),
    };
  }

  // ---- JS 后端：全部 typed array，纯 CPU ----
  _renderJS(p, pro) {
    const t0 = performance.now();
    const S = p.size, N = S * S, scale = S / 512;
    const q = QUALITY[p.quality];
    const { base01, height0, land, sdf, biome, river } = pro;
    let height = Float32Array.from(height0);

    // 1) 基场坡度（决定细节振幅）
    const snorm = new Float32Array(N);
    const dref = (SLOPE_REF * 512) / S;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const xl = x > 0 ? i - 1 : i, xr = x < S - 1 ? i + 1 : i;
        const yu = y > 0 ? i - S : i, yd = y < S - 1 ? i + S : i;
        const gx = (height0[xr] - height0[xl]) / (xr - xl || 1);
        const gy = (height0[yd] - height0[yu]) / (((yd - yu) / S) || 1);
        snorm[i] = clamp01(Math.hypot(gx, gy) / dref);
      }
    }

    // 2) 细节：ridged fBm × 坡度/海岸调制
    const coastPx = Math.min(SDF_RADIUS, Math.max(1, COAST_FADE_PX * scale));
    const nseed = pro.nseed;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const coastW = smoothstep01(Math.abs(sdf[i]) / coastPx);
        if (coastW <= 0) continue;
        const amp = q.amp * (0.25 + 0.75 * snorm[i]) * coastW;
        const detail = (ridgedFbm(S, q.octaves, nseed, y, x) - 0.5) * 2;
        height[i] += detail * amp * (land[i] ? 1 : 0.15);
      }
    }

    // 3) 轻侵蚀：坡度限制（与 Python np.roll 的环绕语义一致）
    const limit = (600 * Math.sqrt(512 / S)) / Math.max(p.zoom, 0.25);
    for (let r = 0; r < q.erosion; r++) {
      const next = new Float32Array(N);
      for (let y = 0; y < S; y++) {
        const yUp = ((y - 1 + S) % S) * S;
        const yDn = ((y + 1) % S) * S;
        for (let x = 0; x < S; x++) {
          const i = y * S + x;
          const up = height[yUp + x];
          const dn = height[yDn + x];
          const lf = height[y * S + ((x - 1 + S) % S)];
          const rt = height[y * S + ((x + 1) % S)];
          const h = height[i];
          const avg = 0.25 * (up + dn + lf + rt);
          const steep = Math.max(Math.abs(up - h), Math.abs(dn - h),
            Math.abs(lf - h), Math.abs(rt - h));
          const move = clamp((steep - limit) / limit, 0, 1);
          const m = (land[i] && h > 0 ? 1 : 0) * move * 0.35;
          next[i] = h + (avg - h) * m;
        }
      }
      height = next;
    }

    // 4) 河网下切
    const widthPx = Math.max(1, 1 + (2 * p.zoom * S) / this.packSize);
    const carve = 35 * widthPx;
    for (let i = 0; i < N; i++) {
      if (river[i] && land[i]) height[i] -= carve;
    }

    // 5) 着色
    const st = resolveStyle(p.style || this.style);
    const pal = st.palette;
    const az = (st.hill.azimuth * Math.PI) / 180;
    const alt = (st.hill.altitude * Math.PI) / 180;
    const strength = st.hill.strength;
    const ca = Math.cos(alt), sa = Math.sin(alt);
    const saz = Math.sin(az), caz = Math.cos(az);
    const beachPx = Math.max(1, 2 * scale);
    const nscale = 0.15;
    const out = new Uint8ClampedArray(N * 4);

    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const i = y * S + x;
        const h = height[i], s = sdf[i];
        let r, g, b;
        if (!land[i]) {
          const t = smoothstep01(clamp01(-h / 5000));
          r = pal.ocean_shallow[0] * (1 - t) + pal.ocean_deep[0] * t;
          g = pal.ocean_shallow[1] * (1 - t) + pal.ocean_deep[1] * t;
          b = pal.ocean_shallow[2] * (1 - t) + pal.ocean_deep[2] * t;
        } else {
          const c = st.biome[clamp(biome[i], 0, 8)];
          r = c[0]; g = c[1]; b = c[2];

          const bw = smoothstep01(1 - clamp01(s / beachPx)) * clamp01(1 - h / 260);
          if (bw > 0) {
            r = r * (1 - bw) + pal.beach[0] * bw;
            g = g * (1 - bw) + pal.beach[1] * bw;
            b = b * (1 - bw) + pal.beach[2] * bw;
          }

          const mount = smoothstep01(clamp01((h - 900) / 1400)) *
            smoothstep01(clamp01(snorm[i] * 1.6)) * 0.55;
          if (mount > 0) {
            r = r * (1 - mount) + pal.mountain[0] * mount;
            g = g * (1 - mount) + pal.mountain[1] * mount;
            b = b * (1 - mount) + pal.mountain[2] * mount;
          }

          const snow = smoothstep01(clamp01((h - 2600) / 1400));
          if (snow > 0) {
            r = r * (1 - snow) + pal.snow[0] * snow;
            g = g * (1 - snow) + pal.snow[1] * snow;
            b = b * (1 - snow) + pal.snow[2] * snow;
          }

          // hillshade（梯度按 km 归一 + 固定陡度）
          const xl = x > 0 ? i - 1 : i, xr = x < S - 1 ? i + 1 : i;
          const yu = y > 0 ? i - S : i, yd = y < S - 1 ? i + S : i;
          const gx = (height[xr] - height[xl]) / (xr - xl || 1) / 1000;
          const gy = (height[yd] - height[yu]) / (((yd - yu) / S) || 1) / 1000;
          const norm = Math.sqrt(gx * gx + gy * gy + nscale * nscale);
          const dot = (-gx * ca * saz - gy * ca * caz + nscale * sa) /
            Math.max(norm, 1e-6);
          const shade = clamp(0.75 + strength * 0.45 * dot, 0.55, 1.35);
          r *= shade; g *= shade; b *= shade;
        }

        if (river[i] && land[i]) { r = st.river[0]; g = st.river[1]; b = st.river[2]; }

        const o = i * 4;
        out[o] = r * 255; out[o + 1] = g * 255; out[o + 2] = b * 255; out[o + 3] = 255;
      }
    }

    const ms = performance.now() - t0;
    this.backend = 'js';
    return {
      rgba: out, height, land, river, biome, ms,
      stats: this._stats(p, height, land, river, biome, ms, 'js'),
    };
  }

  _paint(rgba, S) {
    const canvas = this._ensureCanvas(S);
    const ctx = canvas.getContext('2d');
    const data = rgba instanceof Uint8ClampedArray ? rgba : new Uint8ClampedArray(rgba);
    ctx.putImageData(new ImageData(data, S, S), 0, 0);
    return canvas;
  }

  /**
   * 渲染一帧。
   * @param {{size?:number, cx?:number, cy?:number, zoom?:number, quality?:string, style?:object}} params
   * @returns {Promise<{canvas, ms, backend, stats}>}
   */
  async render(params = {}) {
    const p = makeParams(params);
    const t0 = performance.now();
    const pro = this._prologue(p);

    let res = null;
    if (this.preferGPU && await this._initGPU()) {
      try {
        res = await this._renderGPU(p, pro);
      } catch (err) {
        console.warn('[lite] WebGPU 渲染失败，回退到 JS：', err);
        this._gpu = false;
        res = null;
      }
    }
    if (!res) res = this._renderJS(p, pro);

    const canvas = this._paint(res.rgba, p.size);
    return {
      canvas,
      ms: performance.now() - t0,
      backend: res.stats.backend,
      stats: res.stats,
    };
  }

  // ==================== 9. WebGPU 后端 ====================
  async _initGPU() {
    if (this._gpu === false) return false;
    if (this._gpu) return true;
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      this._gpu = false;
      return false;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) { this._gpu = false; return false; }
      const device = await adapter.requestDevice();
      if (device.addEventListener) {
        device.addEventListener('uncapturederror', (e) => {
          console.warn('[lite] WebGPU 未捕获错误：', e.error && e.error.message);
        });
      }
      this._gpu = { device, adapter };
      return true;
    } catch (err) {
      console.warn('[lite] 没有可用的 WebGPU 适配器：', err);
      this._gpu = false;
      return false;
    }
  }

  /** 生成 WGSL（样式颜色直接烘进源码，省掉额外 uniform 绑定）。 */
  _wgslSource(st) {
    const v3 = (c) => `vec3<f32>(${c.map((x) => x.toFixed(5)).join(', ')})`;
    let biomeCases = '';
    st.biome.forEach((c, i) => {
      biomeCases += `    case ${i}u: { return ${v3(c)}; }\n`;
    });
    return `
struct Par {
  size: f32, packSize: f32, noiseSeed: u32, octaves: f32,
  zoom: f32, cx: f32, cy: f32, amp: f32,
  erosionLimit: f32, gain: f32, lacunarity: f32, freq0: f32,
  slopeRef: f32, coastPx: f32, beachPx: f32, biomeValid: f32,
  nscale: f32, waterAmp: f32, riverCarve: f32, seaLevel: f32,
  pad0: f32, pad1: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<uniform> P: Par;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;   // base01, height0, temp, precip
@group(0) @binding(2) var<storage, read> aux: array<u32>;         // 群系索引
@group(0) @binding(3) var<storage, read> sdfB: array<f32>;
@group(0) @binding(4) var<storage, read> riverB: array<u32>;
@group(0) @binding(5) var<storage, read_write> hA: array<f32>;
@group(0) @binding(6) var<storage, read_write> hB: array<f32>;
@group(0) @binding(7) var<storage, read_write> outp: array<u32>;

const OCEAN_DEEP = ${v3(st.palette.ocean_deep)};
const OCEAN_SHALLOW = ${v3(st.palette.ocean_shallow)};
const BEACH = ${v3(st.palette.beach)};
const MOUNTAIN = ${v3(st.palette.mountain)};
const SNOW = ${v3(st.palette.snow)};
const RIVER = ${v3(st.river)};
const AZ = ${((st.hill.azimuth * Math.PI) / 180).toFixed(6)};
const ALT = ${((st.hill.altitude * Math.PI) / 180).toFixed(6)};
const STRENGTH = ${Number(st.hill.strength).toFixed(5)};

fn biomeColor(i: u32) -> vec3<f32> {
  switch (i) {
${biomeCases}    default: { return ${v3(st.biome[2])}; }
  }
}

fn biomeFromClimate(t: f32, p: f32) -> u32 {
  if (t < -5.0) { return 0u; }
  if (t < 8.0) { return select(0u, 1u, p >= 250.0); }
  if (t < 20.0) {
    if (p < 350.0) { return 2u; }
    return select(3u, 4u, p >= 900.0);
  }
  if (p < 300.0) { return 5u; }
  if (p < 900.0) { return 6u; }
  if (p < 1800.0) { return 7u; }
  return 8u;
}

fn smooth01(t: f32) -> f32 {
  let x = clamp(t, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

fn hash01(ix: i32, iy: i32, oct: i32, seed: u32) -> f32 {
  var h: u32 = (bitcast<u32>(ix) * 0x27d4eb2du)
             ^ (bitcast<u32>(iy) * 0x165667b1u)
             ^ (bitcast<u32>(oct) * 0x9e3779b1u)
             ^ (seed * 0x85ebca6bu);
  h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
  h = (h ^ (h >> 13u)) * 0x297a2d39u;
  h = h ^ (h >> 16u);
  return f32(h) * 2.3283064365386963e-10;
}

fn vnoise(px: f32, py: f32, freq: i32, oct: i32) -> f32 {
  let inv = f32(freq) / P.size;
  let tx = (px + 0.5) * inv;
  let ty = (py + 0.5) * inv;
  let fx0 = floor(tx);
  let fy0 = floor(ty);
  let fx = smooth01(tx - fx0);
  let fy = smooth01(ty - fy0);
  let ix0 = clamp(i32(fx0), 0, freq);
  let iy0 = clamp(i32(fy0), 0, freq);
  let ix1 = ix0 + 1;
  let iy1 = iy0 + 1;
  let v00 = hash01(ix0, iy0, oct, P.noiseSeed);
  let v01 = hash01(ix1, iy0, oct, P.noiseSeed);
  let v10 = hash01(ix0, iy1, oct, P.noiseSeed);
  let v11 = hash01(ix1, iy1, oct, P.noiseSeed);
  return mix(mix(v00, v01, fx), mix(v10, v11, fx), fy);
}

fn fbm(px: f32, py: f32, octaves: i32) -> f32 {
  var total = 0.0;
  var amp = 1.0;
  var norm = 0.0;
  var freq = P.freq0;
  for (var k = 0; k < octaves; k = k + 1) {
    let f = i32(clamp(round(freq), 2.0, 256.0));
    let n = vnoise(px, py, f, k + 1);
    total = total + amp * (1.0 - abs(2.0 * n - 1.0));
    norm = norm + amp;
    amp = amp * P.gain;
    freq = freq * P.lacunarity;
  }
  return total / max(norm, 1e-6);
}

fn snormAt(i: u32, x: i32, y: i32, n: i32) -> f32 {
  let S = u32(n);
  let xl = max(x - 1, 0);
  let xr = min(x + 1, n - 1);
  let yu = max(y - 1, 0);
  let yd = min(y + 1, n - 1);
  let hr = src[u32(y) * S + u32(xr)].y;
  let hl = src[u32(y) * S + u32(xl)].y;
  let hd = src[u32(yd) * S + u32(x)].y;
  let hu = src[u32(yu) * S + u32(x)].y;
  let gx = (hr - hl) / f32(xr - xl);
  let gy = (hd - hu) / f32(yd - yu);
  return clamp(length(vec2<f32>(gx, gy)) / P.slopeRef, 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn synth(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = i32(P.size);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= n || y >= n) { return; }
  let S = u32(n);
  let i = u32(y) * S + u32(x);
  let s = src[i];
  let sn = snormAt(i, x, y, n);
  let coastW = smooth01(abs(sdfB[i]) / P.coastPx);
  let ampv = P.amp * (0.25 + 0.75 * sn) * coastW;
  let detail = (fbm(f32(x), f32(y), i32(P.octaves)) - 0.5) * 2.0;
  let wamp = select(P.waterAmp, 1.0, s.x >= P.seaLevel);
  hA[i] = s.y + detail * ampv * wamp;
}

@compute @workgroup_size(8, 8)
fn erode(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = i32(P.size);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= n || y >= n) { return; }
  let S = u32(n);
  let i = u32(y) * S + u32(x);
  let xl = u32((x - 1 + n) % n);
  let xr = u32((x + 1) % n);
  let yu = u32((y - 1 + n) % n);
  let yd = u32((y + 1) % n);
  let h = hA[i];
  let up = hA[yu * S + u32(x)];
  let dn = hA[yd * S + u32(x)];
  let lf = hA[u32(y) * S + xl];
  let rt = hA[u32(y) * S + xr];
  let avg = 0.25 * (up + dn + lf + rt);
  let steep = max(max(abs(up - h), abs(dn - h)), max(abs(lf - h), abs(rt - h)));
  let mv = clamp((steep - P.erosionLimit) / P.erosionLimit, 0.0, 1.0);
  let land = select(0.0, 1.0, (src[i].x >= P.seaLevel) && (h > 0.0));
  let m = land * mv * 0.35;
  hB[i] = h + (avg - h) * m;
}

fn pack4(c: vec3<f32>) -> u32 {
  let r = u32(clamp(c.r, 0.0, 1.0) * 255.0 + 0.5);
  let g = u32(clamp(c.g, 0.0, 1.0) * 255.0 + 0.5);
  let b = u32(clamp(c.b, 0.0, 1.0) * 255.0 + 0.5);
  return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = i32(P.size);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= n || y >= n) { return; }
  let S = u32(n);
  let i = u32(y) * S + u32(x);
  let h = hA[i];
  let s = src[i];
  let isLand = s.x >= P.seaLevel;
  var col: vec3<f32>;
  if (!isLand) {
    let t = smooth01(clamp(-h / 5000.0, 0.0, 1.0));
    col = mix(OCEAN_SHALLOW, OCEAN_DEEP, t);
  } else {
    var b = aux[i];
    if (P.biomeValid < 0.5) { b = biomeFromClimate(s.z, s.w); }
    col = biomeColor(b);
    let bw = smooth01(1.0 - clamp(sdfB[i] / P.beachPx, 0.0, 1.0))
           * clamp(1.0 - h / 260.0, 0.0, 1.0);
    col = mix(col, BEACH, bw);
    let sn = snormAt(i, x, y, n);
    let mount = smooth01(clamp((h - 900.0) / 1400.0, 0.0, 1.0))
              * smooth01(clamp(sn * 1.6, 0.0, 1.0)) * 0.55;
    col = mix(col, MOUNTAIN, mount);
    let snow = smooth01(clamp((h - 2600.0) / 1400.0, 0.0, 1.0));
    col = mix(col, SNOW, snow);

    let xl = u32(max(x - 1, 0));
    let xr = u32(min(x + 1, n - 1));
    let yu = u32(max(y - 1, 0));
    let yd = u32(min(y + 1, n - 1));
    let gx = (hA[u32(y) * S + xr] - hA[u32(y) * S + xl])
           / f32(i32(xr) - i32(xl)) / 1000.0;
    let gy = (hA[yd * S + u32(x)] - hA[yu * S + u32(x)])
           / f32(i32(yd) - i32(yu)) / 1000.0;
    let nrm = sqrt(gx * gx + gy * gy + P.nscale * P.nscale);
    let ca = cos(ALT);
    let dotn = (-gx * ca * sin(AZ) - gy * ca * cos(AZ) + P.nscale * sin(ALT))
             / max(nrm, 1e-6);
    let sh = clamp(0.75 + STRENGTH * 0.45 * dotn, 0.55, 1.35);
    col = col * sh;
  }
  if (riverB[i] == 1u && isLand) { col = RIVER; }
  outp[i] = pack4(col);
}
`;
  }

  async _renderGPU(p, pro) {
    const t0 = performance.now();
    const { device } = this._gpu;
    device.pushErrorScope('validation');
    const S = p.size, N = S * S, q = QUALITY[p.quality];
    const st = resolveStyle(p.style || this.style);
    const { GPUBufferUsage, GPUShaderStage, GPUMapMode } = globalThis;

    // 输入打包：vec4(base01, height0, temp, precip)
    const srcData = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      const o = i * 4;
      srcData[o] = pro.base01[i];
      srcData[o + 1] = pro.height0[i];
      srcData[o + 2] = pro.temp[i];
      srcData[o + 3] = pro.precip[i];
    }
    const byteN = N * 4;
    const mk = (data, extra) => {
      const b = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra,
      });
      device.queue.writeBuffer(b, 0, data);
      return b;
    };
    const uArr = new ArrayBuffer(96);
    const f32 = new Float32Array(uArr);
    const u32v = new Uint32Array(uArr);
    const scale = S / 512;
    f32[0] = S;
    f32[1] = this.packSize;
    u32v[2] = pro.nseed >>> 0;
    f32[3] = q.octaves;
    f32[4] = p.zoom;
    f32[5] = p.cx;
    f32[6] = p.cy;
    f32[7] = q.amp;
    f32[8] = (600 * Math.sqrt(512 / S)) / Math.max(p.zoom, 0.25);
    f32[9] = GAIN;
    f32[10] = LACUNARITY;
    f32[11] = FREQ0;
    f32[12] = (SLOPE_REF * 512) / S;
    f32[13] = Math.min(SDF_RADIUS, Math.max(1, COAST_FADE_PX * scale));
    f32[14] = Math.max(1, 2 * scale);              // beachPx
    f32[15] = this.pack.biome ? 1 : 0;             // biomeValid
    f32[16] = 0.15;                                // nscale
    f32[17] = 0.15;                                // waterAmp
    f32[18] = 35 * Math.max(1, 1 + (2 * p.zoom * S) / this.packSize);
    f32[19] = SEA_LEVEL;

    const uBuf = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uBuf, 0, uArr);

    const srcBuf = mk(srcData, 0);
    const auxBuf = mk(new Uint32Array(pro.biome), 0);
    const sdfBuf = mk(pro.sdf, 0);
    const riverBuf = mk(new Uint32Array(pro.river), 0);
    const mkRW = () => device.createBuffer({
      size: byteN,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const hA = mkRW(), hB = mkRW(), outBuf = mkRW();
    device.queue.writeBuffer(hA, 0, pro.height0);

    const module = device.createShaderModule({ code: this._wgslSource(st) });
    if (module.getCompilationInfo) {
      const info = await module.getCompilationInfo();
      const errs = (info.messages || []).filter((m) => m.type === 'error');
      if (errs.length) {
        await device.popErrorScope();
        throw new Error('WGSL 编译失败：' + errs.map(
          (m) => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | '));
      }
    }
    const bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const pipe = (entryPoint) => device.createComputePipeline({
      layout, compute: { module, entryPoint },
    });
    const pSynth = pipe('synth'), pErode = pipe('erode'), pShade = pipe('shade');
    const bg = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: uBuf } },
        { binding: 1, resource: { buffer: srcBuf } },
        { binding: 2, resource: { buffer: auxBuf } },
        { binding: 3, resource: { buffer: sdfBuf } },
        { binding: 4, resource: { buffer: riverBuf } },
        { binding: 5, resource: { buffer: hA } },
        { binding: 6, resource: { buffer: hB } },
        { binding: 7, resource: { buffer: outBuf } },
      ],
    });

    const wg = Math.ceil(S / 8);
    const enc = device.createCommandEncoder();
    const run = (pipeObj) => {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeObj);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wg, wg);
      pass.end();
    };
    run(pSynth);
    for (let r = 0; r < q.erosion; r++) {
      run(pErode);
      enc.copyBufferToBuffer(hB, 0, hA, 0, byteN);
    }
    run(pShade);
    const staging = device.createBuffer({
      size: byteN,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    enc.copyBufferToBuffer(outBuf, 0, staging, 0, byteN);
    device.queue.submit([enc.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    // pack4 按 r|g<<8|b<<16|a<<24 打包，小端内存里正好是 RGBA 顺序
    const rgba = new Uint8ClampedArray(staging.getMappedRange().slice(0));
    staging.unmap();

    for (const b of [srcBuf, auxBuf, sdfBuf, riverBuf, hA, hB, outBuf, staging, uBuf]) b.destroy();
    const gpuErr = await device.popErrorScope();
    if (gpuErr) throw new Error('WebGPU 校验错误：' + gpuErr.message);
    const ms = performance.now() - t0;
    return {
      rgba,
      stats: this._stats(p, pro.height0, pro.land, pro.river, pro.biome, ms, 'webgpu'),
    };
  }
}
