// Decode page: fetch an IVF, decode it with goav1 compiled to wasm in a
// worker, and draw the frames; or benchmark every decoder build.
const $ = (id) => document.getElementById(id);
const ui = {
  clip: $('clip'), file: $('file'), wasm: $('wasm'), fps: $('fps'), play: $('play'), stop: $('stop'),
  bench: $('bench'), benchAll: $('benchAll'), out: $('out'), outInfo: $('outInfo'),
  sDec: $('sDec'), sFps: $('sFps'), sFrames: $('sFrames'), results: $('results'), log: $('log'),
};
const ctx = ui.out.getContext('2d');

function log(msg, cls) {
  const line = document.createElement('div');
  line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  if (cls) line.className = cls;
  ui.log.prepend(line);
}

// Corpus clips served by ./serve under /corpus/.
const CLIPS = ['p720_inter_q20', 'p720_inter_q32', 'p360_inter_q32', 'p360_intra_q32',
  'p360_inter_q32_2tiles', 'p360_inter_q32_10bit', 'p288_inter_q32', 'p144_inter_q32'];
for (const c of CLIPS) ui.clip.add(new Option(c, c));

let worker = null, workerWasm = '', listener = null;
function startWorker(wasm) {
  if (worker && workerWasm === wasm) return Promise.resolve();
  if (worker) worker.terminate();
  workerWasm = wasm;
  worker = new Worker(`decode-worker.js?wasm=${encodeURIComponent(wasm)}`);
  worker.onmessage = (e) => listener && listener(e.data);
  worker.onerror = (e) => log(`worker: ${e.message}`, 'bad');
  return request(null, 'ready');
}
// Send msg (if any) and resolve with the first reply of type `want`.
function request(msg, want) {
  return new Promise((resolve, reject) => {
    listener = (m) => {
      if (m.type === want) { listener = onPlayback; resolve(m); }
      else if (m.type === 'error') { listener = onPlayback; reject(new Error(m.message)); }
    };
    if (msg) worker.postMessage(msg);
  });
}

// A clip name is a benchcorpus clip, or "real/<stream>" for a stream from
// testdata/realcorpus/streams. An explicit name ignores the file picker.
async function loadClip(name) {
  if (!name && ui.file.files.length) return { name: ui.file.files[0].name, ivf: await ui.file.files[0].arrayBuffer(), md5: null };
  name = name || ui.clip.value;
  const url = name.startsWith('real/') ? `realcorpus/streams/${name.slice(5)}` : `corpus/${name}`;
  const [ivf, md5] = await Promise.all([
    fetch(`${url}.ivf`).then((r) => { if (!r.ok) throw new Error(`fetch ${name}: ${r.status}`); return r.arrayBuffer(); }),
    fetch(`${url}.md5`).then((r) => r.ok ? r.text() : null),
  ]);
  return { name, ivf, md5: md5 && md5.trim().split(/\s+/)[0] };
}

async function openIn(wasm, clip) {
  await startWorker(wasm);
  await request({ type: 'open', ivf: clip.ivf.slice(0) }, 'opened');
}

// ---------- playback ----------
const stats = { n: 0, ms: 0, shown: 0 };
function onPlayback(m) {
  if (m.type === 'frame') {
    const f = m.frame;
    if (ui.out.width !== f.codedWidth) ui.out.width = f.codedWidth;
    if (ui.out.height !== f.codedHeight) ui.out.height = f.codedHeight;
    ctx.drawImage(f, 0, 0);
    ui.outInfo.textContent = `${f.codedWidth}×${f.codedHeight}`;
    f.close();
    stats.n++; stats.ms += m.decodeMs; stats.shown++;
    ui.sFrames.textContent = stats.shown;
  } else if (m.type === 'end') { log('end of stream'); setPlaying(false); }
  else if (m.type === 'error') { log(m.message, 'bad'); setPlaying(false); }
}
listener = onPlayback;
setInterval(() => {
  if (!stats.n) return;
  ui.sDec.textContent = (stats.ms / stats.n).toFixed(1);
  ui.sFps.textContent = stats.n;
  stats.n = stats.ms = 0;
}, 1000);

function setPlaying(on) { ui.play.disabled = on; ui.stop.disabled = !on; }

ui.play.onclick = async () => {
  setPlaying(true);
  try {
    const clip = await loadClip();
    await openIn(ui.wasm.value, clip);
    stats.shown = 0;
    worker.postMessage({ type: 'play', fps: +ui.fps.value });
    log(`playing ${clip.name} with ${ui.wasm.value}`, 'good');
  } catch (e) { log(e.message, 'bad'); setPlaying(false); }
};
ui.stop.onclick = () => { worker && worker.postMessage({ type: 'stop' }); setPlaying(false); };

// ---------- benchmark ----------
async function benchOne(wasm, clip, passes) {
  await openIn(wasm, clip);
  const r = await request({ type: 'bench', passes }, 'bench');
  const ok = clip.md5 ? r.md5 === clip.md5 : null;
  const row = ui.results.insertRow();
  for (const v of [clip.name, wasm.replace('.wasm', ''), r.frames, r.msPerFrame.toFixed(2),
    (1000 / r.msPerFrame).toFixed(0), r.allocs.toFixed(1), (r.allocBytes / 1024).toFixed(1), ok === null ? r.md5.slice(0, 8) : ok ? 'match' : 'MISMATCH']) row.insertCell().textContent = v;
  if (ok === false) row.className = 'bad';
  return r;
}

const runs = []; // {key, bad} per run, for #verify
const short = (wasm) => wasm.replace(/^decoder-|\.wasm$/g, '');

// Decode `clips` with every build in `wasms`, `passes` timed passes each, and
// return one summary entry per run (a failing run is reported, not fatal).
async function runBench(wasms, clips, passes = 3) {
  ui.bench.disabled = ui.benchAll.disabled = true;
  ui.stop.onclick();
  const summary = [];
  for (const name of clips) {
    let clip;
    try { clip = await loadClip(name); } catch (e) { log(e.message, 'bad'); summary.push(`${name}=ERR(${e.message})`); runs.push({ key: name, bad: true }); continue; }
    for (const wasm of wasms) {
      const key = `${clip.name}/${short(wasm)}`;
      try {
        log(`bench ${clip.name} ${wasm}…`);
        const r = await benchOne(wasm, clip, passes);
        const mismatch = clip.md5 && r.md5 !== clip.md5;
        runs.push({ key, bad: !!mismatch || (!clip.md5 && location.hash === "#verify") });
        summary.push(`${key}=${r.msPerFrame.toFixed(1)}${mismatch ? '!' : ''}`);
        summary.push(`${key}/gcs=${r.gcs}`, `${key}/allocs=${r.allocs.toFixed(1)}`, `${key}/KiB=${(r.allocBytes / 1024).toFixed(1)}`);
      } catch (e) {
        log(`${key}: ${e.message}`, 'bad');
        summary.push(`${key}=ERR(${e.message})`);
        runs.push({ key, bad: true });
        if (worker) { worker.terminate(); worker = null; workerWasm = ''; } // a trapped instance is unusable
      }
    }
  }
  ui.bench.disabled = ui.benchAll.disabled = false;
  return summary;
}

ui.bench.onclick = () => runBench([ui.wasm.value], [ui.clip.value]);
ui.benchAll.onclick = () => runBench([...ui.wasm.options].map((o) => o.value), [ui.clip.value]);

// decode.html#bench benchmarks every build on the main clips and writes the
// summary into the page title (for headless runs).
if (location.hash === '#bench') {
  runBench([...ui.wasm.options].map((o) => o.value), ['p720_inter_q20', 'p360_inter_q32', 'p360_intra_q32'])
    .then((s) => { document.title = 'BENCH ' + s.join(' '); });
}

// decode.html?clips=a,b,real/c#verify decodes each clip once with every build
// and compares the output MD5 with the clip's .md5 file. Without ?clips= it
// runs every clip the server lists under /corpus/. The title reports
// "VERIFY <runs> <mismatches> <list of failures>".
if (location.hash === '#verify') {
  (async () => {
    const wanted = new URLSearchParams(location.search).get('clips');
    let clips;
    if (wanted) clips = wanted.split(',');
    else {
      const listing = await fetch('corpus/').then((r) => r.text());
      clips = [...listing.matchAll(/href="([^"]+)\.ivf"/g)].map((m) => decodeURIComponent(m[1]));
    }
    const wasms = [...ui.wasm.options].map((o) => o.value);
    await runBench(wasms, clips, 1);
    const bad = runs.filter((r) => r.bad).map((r) => r.key);
    document.title = `VERIFY ${runs.length} ${bad.length} ${bad.join(' ')}`;
  })();
}
