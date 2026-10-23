// Encoder worker: owns the wasm instance. Frames arrive as transferred
// VideoFrames, are copied straight into Go memory, and leave as AV1 temporal
// units (transferred ArrayBuffers).
//
// TinyGo builds (encoder-*.wasm) and the stock Go build (encoder-go.wasm) both
// export the av* functions with //go:wasmexport, but each needs its own
// wasm_exec.js and names its memory export differently ("memory" / "mem").

const FORMAT = { I420: 0, I420A: 0, NV12: 1, RGBA: 2, BGRA: 3, RGBX: 4, BGRX: 5 };

let ex = null;          // wasm exports
let memory = null;      // wasm linear memory
let cfg = { bitrate: 1_000_000, fps: 30, speed: -2 };
let encW = 0, encH = 0; // active encoder size; 0 = needs (re)init
let forceNextKey = true;

function mem() { return new Uint8Array(memory.buffer); }

function lastError() {
  const p = ex.avErrPtr(), n = ex.avErrLen();
  return n ? new TextDecoder().decode(mem().slice(p, p + n)) : 'unknown error';
}

async function load(url) {
  importScripts(url.includes('-go.') ? 'wasm_exec_go.js' : 'wasm_exec.js');
  const go = new Go();
  const { instance } = await WebAssembly.instantiateStreaming(fetch(url), go.importObject);
  go.run(instance);
  ex = instance.exports;
  memory = ex.memory || ex.mem;
  encW = encH = 0;
  postMessage({ type: 'ready', wasm: url });
}

function ensureEncoder(w, h) {
  if (w === encW && h === encH) return;
  if (ex.avInit(w, h, cfg.bitrate, cfg.fps, cfg.speed) !== 0) throw new Error('avInit: ' + lastError());
  encW = w; encH = h; forceNextKey = true;
}

// Copy the frame's visible rect into Go's buffer as tightly packed planes and
// return the format code avEncode expects.
async function copyIn(frame, w, h) {
  const rect = { x: frame.visibleRect.x, y: frame.visibleRect.y, width: w, height: h };
  const ptr = ex.avFrameBuf();
  const ySize = w * h, cSize = (w / 2) * (h / 2);
  let code = FORMAT[frame.format];
  let layout, size, opts;
  if (frame.format === 'I420' || frame.format === 'I420A') {
    layout = [{ offset: 0, stride: w }, { offset: ySize, stride: w / 2 }, { offset: ySize + cSize, stride: w / 2 }];
    if (frame.format === 'I420A') layout.push({ offset: ySize + 2 * cSize, stride: w });
    size = frame.format === 'I420A' ? 2 * ySize + 2 * cSize : ySize + 2 * cSize;
    opts = { rect, layout };
  } else if (frame.format === 'NV12') {
    layout = [{ offset: 0, stride: w }, { offset: ySize, stride: w }];
    size = ySize + 2 * cSize;
    opts = { rect, layout };
  } else if (code !== undefined) {
    size = ySize * 4;
    opts = { rect, layout: [{ offset: 0, stride: w * 4 }] };
  } else {
    // 4:2:2 / 4:4:4 / high bit depth: let the browser convert to RGBA.
    code = FORMAT.RGBA;
    size = ySize * 4;
    opts = { rect, layout: [{ offset: 0, stride: w * 4 }], format: 'RGBA' };
  }
  await frame.copyTo(new Uint8Array(memory.buffer, ptr, size), opts);
  return code;
}

async function encodeFrame(frame, forceKey) {
  try {
    const w = frame.visibleRect.width & ~1, h = frame.visibleRect.height & ~1;
    ensureEncoder(w, h);
    const timestamp = frame.timestamp;
    const format = frame.format;
    const code = await copyIn(frame, w, h);
    frame.close();
    const t0 = performance.now();
    const n = ex.avEncode(code, forceKey || forceNextKey ? 1 : 0);
    const encodeMs = performance.now() - t0;
    if (n < 0) throw new Error('avEncode: ' + lastError());
    forceNextKey = false;
    const p = ex.avOutPtr();
    const data = mem().slice(p, p + n).buffer;
    postMessage({
      type: 'chunk', data, key: ex.avIsKey() === 1, timestamp, encodeMs,
      qindex: ex.avQIndex(), width: w, height: h, format,
    }, [data]);
  } catch (err) {
    try { frame.close(); } catch (_) {}
    postMessage({ type: 'error', message: String(err && err.message || err) });
  }
}

// IVF container for the temporal units, laid out exactly like the native
// `go run ./verify` output so the two files can be compared with cmp.
function ivf(w, h, fps, chunks) {
  const out = new Uint8Array(32 + chunks.reduce((n, c) => n + 12 + c.length, 0));
  const dv = new DataView(out.buffer);
  out.set([0x44, 0x4b, 0x49, 0x46]);            // "DKIF"
  dv.setUint16(6, 32, true);
  out.set([0x41, 0x56, 0x30, 0x31], 8);         // "AV01"
  dv.setUint16(12, w, true);
  dv.setUint16(14, h, true);
  dv.setUint32(16, fps, true);
  dv.setUint32(20, 1, true);
  dv.setUint32(24, chunks.length, true);
  let o = 32;
  chunks.forEach((c, i) => {
    dv.setUint32(o, c.length, true);
    dv.setBigUint64(o + 4, BigInt(i), true);
    out.set(c, o + 12);
    o += 12 + c.length;
  });
  return out;
}

// Encode `frames` frames of the Go-generated synthetic scene through the same
// avEncode path the page uses and return the IVF with its SHA-256.
async function verify(m) {
  if (ex.avInit(m.w, m.h, m.bitrate, m.fps, m.speed) !== 0) throw new Error('avInit: ' + lastError());
  const chunks = [];
  for (let t = 0; t < m.frames; t++) {
    ex.avSynth(t);
    const n = ex.avEncode(FORMAT.I420, t === 0 ? 1 : 0);
    if (n < 0) throw new Error('avEncode: ' + lastError());
    const p = ex.avOutPtr();
    chunks.push(mem().slice(p, p + n));
  }
  encW = encH = 0; // the page's own encoder must be re-created
  const file = ivf(m.w, m.h, m.fps, chunks);
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', file));
  const sha = Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
  postMessage({ type: 'verify', sha, bytes: file.length, file: file.buffer }, [file.buffer]);
}

onmessage = async (e) => {
  const m = e.data;
  try {
    switch (m.type) {
      case 'load': await load(m.wasm); break;
      case 'config':
        cfg = { bitrate: m.bitrate, fps: m.fps, speed: m.speed };
        encW = encH = 0; // re-init (and keyframe) on the next frame
        break;
      case 'frame': await encodeFrame(m.frame, m.forceKey); break;
      case 'bench': {
        // Best of `passes` runs; every run starts a fresh encoder.
        let best = Infinity;
        for (let i = 0; i < (m.passes || 1); i++) {
          const ms = ex.avBench(m.frames);
          if (ms < 0) { best = -1; break; }
          best = Math.min(best, ms);
        }
        postMessage({ type: 'bench', msPerFrame: best });
        break;
      }
      case 'verify': await verify(m); break;
    }
  } catch (err) {
    // A Go panic surfaces as a wasm trap; report it instead of going silent.
    if (m.type === 'bench') postMessage({ type: 'bench', msPerFrame: -1 });
    postMessage({ type: 'error', message: `${m.type}: ${err && err.message || err}` });
  }
};
