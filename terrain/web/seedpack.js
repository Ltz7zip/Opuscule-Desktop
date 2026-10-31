// seedpack.js — 手写 ZIP + .npy 读取（零依赖，原生 ES module）
//
// 只用浏览器自带能力：
//   - ZIP：解析 EOCD / central directory / local header；method 8 用
//     DecompressionStream('deflate-raw') 解压，method 0 直接拷贝。
//   - .npy：支持 header v1/v2/v3、dtype <f4 / <f8 / |u1 / <u2 / <i4 等，C 序。
//   - 校验：manifest.pack_hash 必须存在；默认只校验条目长度，
//     verify:true 时用 crypto.subtle 逐条校验 sha256。
//
// 入口：loadSeedPack(urlOrArrayBuffer, {verify}) → 见文件末尾。

const SIG_EOCD = 0x06054b50;
const SIG_CEN = 0x02014b50;
const SIG_LOC = 0x04034b50;

const TD = new TextDecoder('utf-8');

function findEOCD(u8) {
  const stop = Math.max(0, u8.length - 22 - 65535);
  for (let i = u8.length - 22; i >= stop; i--) {
    if (u8[i] === 0x50 && u8[i + 1] === 0x4b && u8[i + 2] === 0x05 && u8[i + 3] === 0x06) {
      return i;
    }
  }
  return -1;
}

async function inflateRaw(buf) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('当前浏览器不支持 DecompressionStream，无法解压 seedpack');
  }
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  const out = await new Response(stream).arrayBuffer();
  return out;
}

/** 读取 zip，返回 Map<name, ArrayBuffer>（已解压）。 */
export async function readZip(buffer) {
  const u8 = new Uint8Array(buffer);
  const dv = new DataView(buffer);
  const eocd = findEOCD(u8);
  if (eocd < 0) throw new Error('不是合法 ZIP：找不到 EOCD');
  const count = dv.getUint16(eocd + 10, true);
  let off = dv.getUint32(eocd + 16, true);

  const meta = [];
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(off, true) !== SIG_CEN) throw new Error('中央目录损坏');
    const method = dv.getUint16(off + 10, true);
    const csize = dv.getUint32(off + 20, true);
    const usize = dv.getUint32(off + 24, true);
    const nlen = dv.getUint16(off + 28, true);
    const elen = dv.getUint16(off + 30, true);
    const clen = dv.getUint16(off + 32, true);
    const lho = dv.getUint32(off + 42, true);
    if (csize === 0xffffffff || lho === 0xffffffff) throw new Error('暂不支持 ZIP64');
    const name = TD.decode(u8.subarray(off + 46, off + 46 + nlen));
    meta.push({ name, method, csize, usize, lho });
    off += 46 + nlen + elen + clen;
  }

  const files = new Map();
  for (const m of meta) {
    if (dv.getUint32(m.lho, true) !== SIG_LOC) throw new Error(`本地头损坏：${m.name}`);
    const nlen = dv.getUint16(m.lho + 26, true);
    const elen = dv.getUint16(m.lho + 28, true);
    const start = m.lho + 30 + nlen + elen;
    const raw = buffer.slice(start, start + m.csize);
    let data = raw;
    if (m.method === 8) data = await inflateRaw(raw);
    else if (m.method !== 0) throw new Error(`不支持的压缩方式 ${m.method}：${m.name}`);
    if (m.usize && data.byteLength !== m.usize) {
      throw new Error(`${m.name} 解压后长度不符（${data.byteLength}/${m.usize}）`);
    }
    files.set(m.name, data);
  }
  return files;
}

const DTYPES = {
  '<f4': Float32Array, '<f8': Float64Array,
  '|u1': Uint8Array, '<u1': Uint8Array, '|i1': Int8Array,
  '<u2': Uint16Array, '<i2': Int16Array,
  '<u4': Uint32Array, '<i4': Int32Array,
};

/** 解析 .npy → {dtype, shape, data, fortranOrder}。 */
export function parseNpy(buffer) {
  const u8 = new Uint8Array(buffer);
  if (u8.length < 10 || u8[0] !== 0x93 || TD.decode(u8.subarray(1, 6)) !== 'NUMPY') {
    throw new Error('不是 .npy 文件');
  }
  const major = u8[6];
  let start, headerLen;
  const dv = new DataView(buffer);
  if (major === 1) {
    headerLen = dv.getUint16(8, true);
    start = 10;
  } else if (major === 2 || major === 3) {
    headerLen = dv.getUint32(8, true);
    start = 12;
  } else {
    throw new Error(`不支持的 npy 版本 ${major}`);
  }
  const header = TD.decode(u8.subarray(start, start + headerLen));
  const descr = (header.match(/'descr'\s*:\s*'([^']+)'/) || [])[1];
  const fortran = /'fortran_order'\s*:\s*True/.test(header);
  const shapeTxt = (header.match(/'shape'\s*:\s*\(([^)]*)\)/) || [])[1];
  if (!descr || shapeTxt === undefined) throw new Error('npy 头缺少 descr/shape');
  const shape = shapeTxt.split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const Ctor = DTYPES[descr];
  if (!Ctor) throw new Error(`不支持的 dtype：${descr}`);
  if (fortran) throw new Error('暂不支持 Fortran 序 .npy');
  const count = shape.reduce((a, b) => a * b, 1);
  const offset = start + headerLen;
  const data = new Ctor(buffer.slice(offset, offset + count * Ctor.BYTES_PER_ELEMENT));
  return { dtype: descr, shape, data, fortranOrder: false };
}

async function sha256Hex(buf) {
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function toArrayBuffer(x) {
  if (x instanceof ArrayBuffer) return x;
  if (ArrayBuffer.isView(x)) {
    return x.buffer.slice(x.byteOffset, x.byteOffset + x.byteLength);
  }
  throw new TypeError('loadSeedPack 需要 URL 字符串或 ArrayBuffer');
}

/**
 * 载入 .seedpack。
 * @param {string|ArrayBuffer} urlOrArrayBuffer
 * @param {{verify?: boolean}} [opts] verify=true 时逐条校验 sha256（较慢）
 */
export async function loadSeedPack(urlOrArrayBuffer, opts = {}) {
  const t0 = performance.now();
  let buffer;
  if (typeof urlOrArrayBuffer === 'string') {
    const resp = await fetch(urlOrArrayBuffer);
    if (!resp.ok) throw new Error(`加载失败 ${resp.status}：${urlOrArrayBuffer}`);
    buffer = await resp.arrayBuffer();
  } else {
    buffer = toArrayBuffer(urlOrArrayBuffer);
  }

  const files = await readZip(buffer);
  if (!files.has('manifest.json')) throw new Error('不是 seedpack：缺少 manifest.json');
  const manifest = JSON.parse(TD.decode(files.get('manifest.json')));
  if (!manifest.pack_hash) throw new Error('manifest 缺少 pack_hash');
  if (manifest.format !== 'seedpack') throw new Error(`format 不是 seedpack：${manifest.format}`);

  const listed = new Map((manifest.entries || []).map((e) => [e.path, e]));
  for (const [name, data] of files) {
    if (name === 'manifest.json') continue;
    const e = listed.get(name);
    if (!e) throw new Error(`清单未登记条目：${name}`);
    if (e.bytes !== data.byteLength) {
      throw new Error(`${name} 长度不符（${data.byteLength}/${e.bytes}）`);
    }
    if (opts.verify && e.sha256 && (await sha256Hex(data)) !== e.sha256) {
      throw new Error(`${name} sha256 校验失败`);
    }
  }

  const npy = (name) => (files.has(name) ? parseNpy(files.get(name)) : null);
  const json = (name) => (files.has(name) ? JSON.parse(TD.decode(files.get(name))) : null);

  const base = npy('global/base.npy');
  if (!base) throw new Error('seedpack 缺少 global/base.npy');

  return {
    manifest,
    base,
    dem: npy('global/dem_m.npy'),
    temp: npy('climate/temperature.npy'),
    precip: npy('climate/precipitation.npy'),
    biome: npy('hydro/biome.npy'),
    hydro: json('hydro/network.json'),
    style: json('style.json'),
    size: manifest.size,
    seed: manifest.seed,
    packHash: manifest.pack_hash,
    bytes: buffer.byteLength,
    loadMs: performance.now() - t0,
  };
}
