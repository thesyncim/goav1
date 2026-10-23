// Main thread: capture, pacing, WebCodecs decode, and stats. The encoder runs
// in worker.js; at most one frame is in flight, newer frames are dropped
// while it is busy, so latency stays bounded on slow machines.
const $ = (id) => document.getElementById(id);
const ui = {
  source: $('source'), res: $('res'), bitrate: $('bitrate'), speed: $('speed'), wasm: $('wasm'),
  start: $('start'), stop: $('stop'), key: $('key'), bench: $('bench'),
  preview: $('preview'), synth: $('synth'), out: $('out'), srcInfo: $('srcInfo'), outInfo: $('outInfo'),
  sEnc: $('sEnc'), sFps: $('sFps'), sKbps: $('sKbps'), sQ: $('sQ'), sLat: $('sLat'), sDrop: $('sDrop'), log: $('log'),
};
const FPS = 30;

function log(msg, cls) {
  const line = document.createElement('div');
  line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  if (cls) line.className = cls;
  ui.log.prepend(line);
}

// ---------- worker ----------
let worker = null, workerReady = null, loadedWasm = '';
function ensureWorker() {
  if (worker && loadedWasm === ui.wasm.value) return workerReady;
  if (worker) worker.terminate();
  loadedWasm = ui.wasm.value;
  worker = new Worker('worker.js');
  workerReady = new Promise((resolve) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') { log(`loaded ${m.wasm}`); resolve(); }
      else if (m.type === 'chunk') onChunk(m);
      else if (m.type === 'bench') onBench(m);
      else if (m.type === 'error') { busy = false; log(m.message, 'bad'); }
    };
  });
  worker.onerror = (e) => { busy = false; ui.bench.disabled = false; log(`worker: ${e.message}`, 'bad'); };
  worker.postMessage({ type: 'load', wasm: loadedWasm });
  return workerReady;
}

function sendConfig() {
  worker.postMessage({ type: 'config', bitrate: +ui.bitrate.value, fps: FPS, speed: +ui.speed.value });
}

// ---------- capture ----------
let running = false, busy = false, wantKey = false;
let stream = null, reader = null, synthTimer = 0;
let scaler = null, scalerCtx = null;
const sentAt = new Map();
const stats = { frames: 0, bytes: 0, encMs: 0, drops: 0, lat: 0, latN: 0, q: 0 };

function targetSize() { const [w, h] = ui.res.value.split('x').map(Number); return { w, h }; }

// Hand a captured frame to the encoder (or drop it if the encoder is busy).
function submit(frame) {
  if (!running || busy) { stats.drops += running ? 1 : 0; frame.close(); return; }
  const { w, h } = targetSize();
  if (frame.displayWidth !== w || frame.displayHeight !== h) {
    if (!scaler || scaler.width !== w || scaler.height !== h) {
      scaler = new OffscreenCanvas(w, h); scalerCtx = scaler.getContext('2d');
    }
    scalerCtx.drawImage(frame, 0, 0, w, h);
    const scaled = new VideoFrame(scaler, { timestamp: frame.timestamp });
    frame.close();
    frame = scaled;
  }
  ui.srcInfo.textContent = `${frame.displayWidth}×${frame.displayHeight} ${frame.format ?? ''}`;
  busy = true;
  sentAt.set(frame.timestamp, performance.now());
  worker.postMessage({ type: 'frame', frame, forceKey: wantKey }, [frame]);
  wantKey = false;
}

async function startCamera() {
  const { w, h } = targetSize();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: FPS } }, audio: false,
  });
  ui.preview.hidden = false; ui.synth.hidden = true;
  ui.preview.srcObject = stream;
  await ui.preview.play();
  const track = stream.getVideoTracks()[0];
  if ('MediaStreamTrackProcessor' in window) {
    reader = new MediaStreamTrackProcessor({ track }).readable.getReader();
    (async () => {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        submit(value);
      }
    })();
  } else {
    // Firefox/Safari: sample the <video> element per presented frame.
    const tick = () => {
      if (!running) return;
      submit(new VideoFrame(ui.preview, { timestamp: Math.round(performance.now() * 1000) }));
      ui.preview.requestVideoFrameCallback(tick);
    };
    ui.preview.requestVideoFrameCallback(tick);
  }
}

function startSynthetic() {
  const { w, h } = targetSize();
  ui.synth.width = w; ui.synth.height = h;
  ui.preview.hidden = true; ui.synth.hidden = false;
  const ctx = ui.synth.getContext('2d');
  let t = 0;
  synthTimer = setInterval(() => {
    t++;
    const g = ctx.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, `hsl(${(t * 2) % 360} 70% 45%)`);
    g.addColorStop(1, `hsl(${(t * 2 + 160) % 360} 70% 30%)`);
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 6; i++) {
      const a = t / 30 + i;
      ctx.fillStyle = `hsl(${i * 60} 80% 60%)`;
      ctx.beginPath();
      ctx.arc(w / 2 + Math.cos(a) * w * 0.32, h / 2 + Math.sin(a * 1.3) * h * 0.3, h * 0.08, 0, 7);
      ctx.fill();
    }
    ctx.fillStyle = '#fff'; ctx.font = `bold ${Math.round(h / 9)}px system-ui`;
    ctx.fillText(`goav1 wasm  ${new Date().toLocaleTimeString()}  #${t}`, w * 0.04, h * 0.92);
    submit(new VideoFrame(ui.synth, { timestamp: Math.round(t * 1e6 / FPS) }));
  }, 1000 / FPS);
}

// ---------- decode + display ----------
let decoder = null, decW = 0, decH = 0;
const demo = { enc: 0, dec: 0, err: 0, encMs: 0 }; // cumulative counters for #demo
const outCtx = ui.out.getContext('2d');

async function makeDecoder(w, h) {
  if (decoder && decoder.state !== 'closed') decoder.close();
  const candidates = ['av01.0.04M.08', 'av01.0.05M.08', 'av01.0.08M.08', 'av01.0.09M.08'];
  let config = null;
  for (const codec of candidates) {
    const c = { codec, codedWidth: w, codedHeight: h, optimizeForLatency: true };
    if ((await VideoDecoder.isConfigSupported(c)).supported) { config = c; break; }
  }
  if (!config) throw new Error('this browser has no WebCodecs AV1 decoder');
  decoder = new VideoDecoder({
    output: (frame) => {
      // The encoder pads coded height/width to a multiple of 8 and signals the
      // true size in render_size, which Chrome does not apply; crop here.
      if (ui.out.width !== decW) ui.out.width = decW;
      if (ui.out.height !== decH) ui.out.height = decH;
      outCtx.drawImage(frame, 0, 0, decW, decH, 0, 0, decW, decH);
      demo.dec++;
      const t0 = sentAt.get(frame.timestamp);
      if (t0 !== undefined) { stats.lat += performance.now() - t0; stats.latN++; sentAt.delete(frame.timestamp); }
      frame.close();
    },
    error: (e) => { demo.err++; log(`decoder: ${e.message}`, "bad"); },
  });
  decoder.configure(config);
  decW = w; decH = h;
  log(`decoder ${config.codec} ${w}×${h}`);
}

let decoderPending = null;
async function onChunk(m) {
  busy = false;
  demo.enc++; demo.encMs += m.encodeMs;
  stats.frames++; stats.bytes += m.data.byteLength; stats.encMs += m.encodeMs; stats.q = m.qindex;
  ui.outInfo.textContent = `${m.width}×${m.height} ${m.key ? 'KEY ' : ''}${m.data.byteLength} B`;
  if (m.key && (m.width !== decW || m.height !== decH || !decoder || decoder.state === 'closed')) {
    decoderPending = makeDecoder(m.width, m.height).catch((e) => log(e.message, 'bad'));
  }
  if (decoderPending) { await decoderPending; }
  if (!decoder || decoder.state !== 'configured') return;
  decoder.decode(new EncodedVideoChunk({ type: m.key ? 'key' : 'delta', timestamp: m.timestamp, data: m.data }));
}

// ---------- stats ----------
setInterval(() => {
  if (!running) return;
  const f = stats.frames;
  ui.sEnc.textContent = f ? (stats.encMs / f).toFixed(1) : '–';
  ui.sFps.textContent = f.toString();
  ui.sKbps.textContent = Math.round(stats.bytes * 8 / 1000).toString();
  ui.sQ.textContent = f ? stats.q.toString() : '–';
  ui.sLat.textContent = stats.latN ? (stats.lat / stats.latN).toFixed(0) : '–';
  ui.sDrop.textContent = stats.drops.toString();
  stats.frames = stats.bytes = stats.encMs = stats.drops = stats.lat = stats.latN = 0;
  if (sentAt.size > 300) sentAt.clear();
}, 1000);

// ---------- controls ----------
async function start() {
  ui.start.disabled = true;
  try {
    await ensureWorker();
    sendConfig();
    running = true; busy = false; wantKey = true;
    if (ui.source.value === 'camera') await startCamera(); else startSynthetic();
    ui.stop.disabled = false; ui.key.disabled = false;
    log(`started ${ui.source.value} ${ui.res.value} @ ${ui.bitrate.value / 1000} kbps, effort ${ui.speed.value}`, 'good');
  } catch (e) {
    log(e.message || String(e), 'bad');
    stop();
  }
}

function stop() {
  running = false;
  clearInterval(synthTimer);
  if (reader) { reader.cancel().catch(() => {}); reader = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  ui.start.disabled = false; ui.stop.disabled = true; ui.key.disabled = true;
}

function restartIfRunning() { if (running) { stop(); start(); } }

ui.start.onclick = start;
ui.stop.onclick = stop;
ui.key.onclick = () => { wantKey = true; };
ui.source.onchange = restartIfRunning;
ui.res.onchange = restartIfRunning;
ui.wasm.onchange = restartIfRunning;
ui.bitrate.onchange = () => { if (running) sendConfig(); };
ui.speed.onchange = () => { if (running) sendConfig(); };

ui.bench.onclick = async () => {
  ui.bench.disabled = true;
  if (running) stop();
  await ensureWorker();
  log(`benchmark: 60 synthetic 640×360 frames with ${loadedWasm}…`);
  worker.postMessage({ type: 'bench', frames: 60 });
};
function onBench(m) {
  ui.bench.disabled = false;
  log(m.msPerFrame < 0 ? 'benchmark failed' :
    `benchmark ${loadedWasm}: ${m.msPerFrame.toFixed(2)} ms/frame (${(1000 / m.msPerFrame).toFixed(1)} fps) at 640×360`, 'good');
}

if (!('VideoDecoder' in window)) log('WebCodecs is unavailable in this browser; decoding will not work.', 'bad');

// http://localhost:8090/#bench benchmarks every wasm build on load (no clicks
// needed) and writes the summary into the page title: best of 3 runs of 60
// frames per build.
function workerRequest(msg, want) {
  return new Promise((resolve, reject) => {
    const prev = worker.onmessage;
    worker.onmessage = (e) => {
      if (e.data.type === want) { worker.onmessage = prev; resolve(e.data); }
      else if (e.data.type === 'error') { worker.onmessage = prev; reject(new Error(e.data.message)); }
      else prev(e);
    };
    worker.postMessage(msg);
  });
}

if (location.hash === '#bench') {
  (async () => {
    const results = [];
    for (const opt of ui.wasm.options) {
      ui.wasm.value = opt.value;
      try {
        await ensureWorker();
        const { msPerFrame } = await workerRequest({ type: 'bench', frames: 60, passes: 3 }, 'bench');
        results.push(`${opt.value.replace(/^encoder-|\.wasm$/g, '')}=${msPerFrame.toFixed(1)}`);
        log(`bench ${opt.value}: ${msPerFrame.toFixed(2)} ms/frame`, 'good');
      } catch (e) {
        results.push(`${opt.value.replace(/^encoder-|\.wasm$/g, '')}=ERR(${e.message})`);
      }
    }
    document.title = 'BENCH ' + results.join(' ');
  })();
}

// http://localhost:8090/#verify encodes 60 frames of the Go-generated synthetic
// scene with every build (640x360, CBR 1 Mbps, effort 0, one thread), uploads
// each IVF to the server's /dump/ endpoint (go run ./serve -dump DIR) and
// writes the SHA-256 of each into the page title. Compare the files with the
// native `go run ./verify` output.
if (location.hash === '#verify') {
  (async () => {
    const results = [];
    for (const opt of ui.wasm.options) {
      ui.wasm.value = opt.value;
      const name = opt.value.replace(/^encoder-|\.wasm$/g, '');
      try {
        await ensureWorker();
        const r = await workerRequest({ type: 'verify', w: 640, h: 360, bitrate: 1_000_000, fps: 30, speed: 0, frames: 60 }, 'verify');
        await fetch(`dump/encoder-${name}.ivf`, { method: 'POST', body: r.file });
        results.push(`${name}=${r.sha}`);
        log(`verify ${opt.value}: ${r.bytes} bytes ${r.sha}`, 'good');
      } catch (e) {
        results.push(`${name}=ERR(${e.message})`);
      }
    }
    document.title = 'VERIFY ' + results.join(' ');
  })();
}

// http://localhost:8090/#demo runs the whole page pipeline (synthetic canvas
// source -> VideoFrame -> worker -> wasm encoder -> WebCodecs AV1 decoder ->
// canvas) for 6 s with every build at 640x360 and writes
// "name=<encoded>/<decoded>/<decoder errors>/<mean encode ms>/<canvas spread>"
// into the title. A healthy run has decoded == encoded, no errors and a
// canvas spread well above zero (the decoded picture is not blank).
if (location.hash === '#demo') {
  (async () => {
    const results = [];
    ui.source.value = 'synthetic';
    ui.res.value = '640x360';
    for (const opt of ui.wasm.options) {
      ui.wasm.value = opt.value;
      Object.assign(demo, { enc: 0, dec: 0, err: 0, encMs: 0 });
      await start();
      await new Promise((r) => setTimeout(r, 6000));
      stop();
      await new Promise((r) => setTimeout(r, 500)); // let the decoder drain
      let lo = 255, hi = 0;
      try {
        const px = outCtx.getImageData(0, 0, decW, decH).data;
        for (let i = 0; i < px.length; i += 4 * 97) { lo = Math.min(lo, px[i + 1]); hi = Math.max(hi, px[i + 1]); }
      } catch (_) {}
      results.push(`${opt.value.replace(/^encoder-|\.wasm$/g, '')}=${demo.enc}/${demo.dec}/${demo.err}/${(demo.encMs / Math.max(demo.enc, 1)).toFixed(1)}/${hi - lo}`);
    }
    document.title = 'DEMO ' + results.join(' ');
  })();
}
