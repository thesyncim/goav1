// Decoder worker: loads one goav1 decoder build (TinyGo or stock Go; each
// needs its own wasm_exec.js) and serves open/bench/play requests.
const url = new URL(location.href).searchParams.get('wasm');
importScripts(url === 'decoder-go.wasm' ? 'wasm_exec_go.js' : 'wasm_exec.js');

let playing = false;

const ready = (async () => {
  const go = new Go();
  const { instance } = await WebAssembly.instantiateStreaming(fetch(url), go.importObject);
  go.run(instance);
  postMessage({ type: 'ready', wasm: url });
})();

function check(err) { if (err) throw new Error(err); }

async function play(fps) {
  playing = true;
  const interval = fps > 0 ? 1000 / fps : 0;
  let due = performance.now(), n = 0;
  while (playing) {
    const t0 = performance.now();
    const f = goav1dec.next();
    const decodeMs = performance.now() - t0;
    if (f instanceof Error) throw f;
    if (f === null) { postMessage({ type: 'end' }); break; }
    const frame = new VideoFrame(f.data, {
      format: 'I420', codedWidth: f.width, codedHeight: f.height, timestamp: n++ * 33333,
    });
    postMessage({ type: 'frame', frame, decodeMs }, [frame]);
    due += interval;
    const wait = due - performance.now();
    await new Promise((r) => setTimeout(r, Math.max(0, wait)));
    if (wait < -200) due = performance.now(); // fell behind; don't burst
  }
  playing = false;
}

onmessage = async (e) => {
  const m = e.data;
  try {
    await ready;
    switch (m.type) {
      case 'open': check(goav1dec.open(new Uint8Array(m.ivf))); postMessage({ type: 'opened' }); break;
      case 'bench': {
        const r = goav1dec.bench(m.passes);
        if (r instanceof Error) throw r;
        postMessage({ type: 'bench', ...r });
        break;
      }
      case 'play': check(goav1dec.reset()); play(m.fps).catch((err) => postMessage({ type: 'error', message: String(err.message || err) })); break;
      case 'stop': playing = false; break;
    }
  } catch (err) {
    postMessage({ type: 'error', message: `${m.type}: ${err && err.message || err}` });
  }
};
