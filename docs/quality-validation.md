# Quality Validation Protocol

This project currently has strong correctness checks: encoded streams decode,
WebRTC dependency metadata is tested, and the realtime path is benchmarked for
latency and bitrate. That is not enough to claim competitive visual quality.
Single-scene PSNR is a smoke signal, not a codec-quality result.

Quality work should be judged with a reproducible low-delay rate-distortion
protocol:

- Test real clips, not only the deterministic synthetic scene. The corpus must
  include camera motion, talking heads, sports or fast motion, screen content,
  animation, noise or low light, and at least one hard texture/pan sequence.
  For claim-supporting runs, use `-require-corpus -min-clips N` so synthetic
  rows, missing input files, and undersized manifests fail before encoding.
- Compare against named encoder builds and settings. For realtime WebRTC work,
  baselines must use low-delay CBR settings with no random-access lookahead or
  hidden alt-ref advantage unless goav1 is given the same latency budget.
- Run at multiple bitrates per clip. Four points is the minimum for BD-rate;
  one bitrate is useful only as a local regression check. For
  claim-supporting runs, use `-summary-csv` with `-require-summary` so missing
  or invalid required BD-rate rows fail the run.
- Measure decoded output against the same source frames. Explicit raw I420 input
  files must match the declared frame count exactly; extra trailing frames or
  bytes are rejected instead of silently benchmarking a prefix. Report actual
  bitrate from compressed payload bytes, not just requested bitrate.
- Prefer perceptual metrics when available. VMAF should be reported when the
  local FFmpeg build has libvmaf; PSNR, SSIM, and XPSNR remain useful secondary
  metrics and regression guards. For claim-supporting runs, use
  `-require-metrics` so a missing metric fails explicitly instead of becoming
  an `NA` column.
- Make baselines mandatory. For claim-supporting runs, use
  `-require-encoders all` or list the required baselines explicitly so missing
  tools, failed encodes, and skipped encoder rows fail the run instead of
  quietly weakening the comparison.
- Report speed and latency with the quality result. A slower baseline is not a
  realtime peer unless its latency constraints match.

The local `qualitybench` command is the audit harness for this. It can run the
goav1 encoder plus installed external AV1 encoders over the same raw I420 clip,
decode their output, and emit one CSV row per encoder/bitrate.

Example:

```sh
go run ./cmd/qualitybench \
  -manifest corpus/clips.csv \
  -bitrates 3000000,6000000,9000000,12000000 \
  -encoders goav1,aomenc,svt-av1 \
  -anchor aomenc -fps 60 -layers 1 -tiles 0 -golden 0 -keyint 60 \
  -require-corpus -min-clips 6 \
  -require-encoders all \
  -require-metrics xpsnr,vmaf \
  -gomaxprocs 4 \
  -gogc off \
  -goav1-max-threads 4 \
  -goav1-effort 0 \
  -goav1-scene-cut=false \
  -timing-mode e2e \
  -run-order shuffle -shuffle-seed 1 \
  -runs 3 -warmup-runs 1 \
  -aom-cpu-used 8 \
  -svt-preset 13 \
  -svt-lp 0 \
  -csv quality.csv -summary-csv quality-summary.csv -require-summary \
  -stats-csv quality-encoder-stats.csv \
  -frame-metrics-csv quality-frame-metrics.csv \
  -metadata-json quality-metadata.json \
  -ffmpeg-bin /opt/homebrew/bin/ffmpeg \
  -ffmpeg-sha256 <sha256-of-ffmpeg> \
  -ffmpeg-av1-decoder libdav1d \
  -vmaf-model version=vmaf_v0.6.1 \
  -aomenc-bin /opt/homebrew/bin/aomenc \
  -aomenc-sha256 <sha256-of-aomenc> \
  -svt-bin /opt/homebrew/bin/SvtAv1EncApp \
  -svt-sha256 <sha256-of-SvtAv1EncApp> \
  -environment-notes "fixed power mode; idle machine; cool start" \
  -cpu-affinity "none" \
  -power-mode "plugged in; high power mode" \
  -thermal-state "cool start; no throttling observed" \
  -frequency-policy "macOS automatic" \
  -background-load "idle machine; no concurrent jobs" \
  -aom-threads 4 \
  -aom-row-mt 1 \
  -svt-asm neon \
  -publish \
  -workdir /tmp/goav1-quality
```

If the local FFmpeg build lacks `libvmaf`, a command that requires VMAF exits
before encoding. Use that failure as a toolchain setup signal; do not treat a
non-VMAF run as state-of-the-art visual validation.

The same strict path is available as `make qualitybench-publish`; set
`QUALITYBENCH_MANIFEST`, `QUALITYBENCH_ENVIRONMENT_NOTES`,
`QUALITYBENCH_CPU_AFFINITY`, `QUALITYBENCH_POWER_MODE`,
`QUALITYBENCH_THERMAL_STATE`, `QUALITYBENCH_FREQUENCY_POLICY`,
`QUALITYBENCH_BACKGROUND_LOAD`, `QUALITYBENCH_FFMPEG_BIN`,
`QUALITYBENCH_FFMPEG_SHA256`, and the matching `QUALITYBENCH_AOMENC_*` /
`QUALITYBENCH_SVT_*` variables for selected external encoders, then override the
other `QUALITYBENCH_*` variables when sweeping speed, assembly, bitrate settings,
the per-command timeout (`QUALITYBENCH_COMMAND_TIMEOUT`, default `30m`), or the
explicit FFmpeg AV1 decoder (`QUALITYBENCH_FFMPEG_AV1_DECODER`, default
`libdav1d`) and VMAF model (`QUALITYBENCH_VMAF_MODEL`, default
`version=vmaf_v0.6.1`). Publish runs that require VMAF must use either
`version=...` or `path=/absolute/model`; path-based models are SHA-256 hashed
into the metadata sidecar.

If `-input` is omitted, `qualitybench` uses the same deterministic synthetic
scene as `encbench`. That path is for smoke testing the harness, not for quality
claims.

Publishable benchmark rows must record structured machine-state controls, not
just prose notes: CPU affinity or the explicit value `none`, power mode, thermal
state, frequency policy/governor, and background-load policy. Keep
`-environment-notes` for extra context, but do not use it as the only place these
controls are documented. On platforms where the OS exposes process CPU
affinity, publish mode records the observed affinity/online CPU lists and rejects
claims such as `cpu-affinity=none` when the process is actually running under a
restricted CPU mask.

Internal Go microbenchmark rows used to justify SIMD or hot-path changes should
use `make bench-go-publish`, not the smoke-oriented `make bench` or
`make bench-all` targets. The publish runner requires a clean tracked worktree,
explicit structured machine-state controls, `-benchmem`, at least five measured
runs, fixed `GOMAXPROCS`, a single matching `go test -cpu` value, explicit
`GOGC`, distinct raw-output and metadata paths, no ambient `GOFLAGS`,
`GOMEMLIMIT`, `GODEBUG`, Go target/compiler/cache overrides such as `GOAMD64`,
`GOARM64`, `GOEXPERIMENT`, `CGO_ENABLED`, `CC`, `CXX`, `GOCACHE`,
`GOMODCACHE`, `GOPATH`, or `GOTMPDIR`, and explicit package/benchmark
selection. Publish mode rejects
defaulted package, benchmark, count, benchtime, CPU, GOMAXPROCS, GC, output,
and metadata settings; pass every control explicitly.
It writes the raw `go test` output and a metadata JSON sidecar containing the
git revision, Go runtime, `GOGC`, command line, output SHA-256, and run
controls. Example:

```sh
make bench-go-publish \
  GO_BENCH_PUBLISH_PKG=./internal/av1/tile \
  GO_BENCH_PUBLISH_BENCH='^BenchmarkCoeffCulLevel$$' \
  GO_BENCH_PUBLISH_OUT=/tmp/goav1-coeff-cul-level.txt \
  GO_BENCH_PUBLISH_METADATA_JSON=/tmp/goav1-coeff-cul-level.json \
  GO_BENCH_PUBLISH_ENVIRONMENT_NOTES="fixed power mode; idle machine; cool start" \
  GO_BENCH_PUBLISH_CPU_AFFINITY=none \
  GO_BENCH_PUBLISH_POWER_MODE="plugged in; high power mode" \
  GO_BENCH_PUBLISH_THERMAL_STATE="cool start; no throttling observed" \
  GO_BENCH_PUBLISH_FREQUENCY_POLICY="macOS automatic" \
  GO_BENCH_PUBLISH_BACKGROUND_LOAD="idle machine; no concurrent jobs" \
  GO_BENCH_PUBLISH_GOMAXPROCS=1 \
  GO_BENCH_PUBLISH_CPU=1 \
  GO_BENCH_PUBLISH_COUNT=7 \
  GO_BENCH_PUBLISH_BENCHTIME=500ms \
  GO_BENCH_PUBLISH_GOGC=off
```

Rows from plain `go test -bench`, `make bench`, or `make bench-all` remain
useful for local exploration, but do not use them as publishable performance
evidence unless the same command controls and metadata are reproduced.

Use `-publish` for rows that will be copied into performance or quality tables.
Publish mode requires a clean git worktree, explicit `-bitrates`,
`-encoders`, `-workdir`, `-csv`, `-metadata-json`, `-manifest`,
`-require-corpus`, `-min-clips`, `-require-encoders all`, `-require-metrics`,
`-summary-csv`, `-frame-metrics-csv`, `-require-summary`, `-gomaxprocs`,
`-fps`, `-layers`, `-tiles`, `-golden`, `-keyint`, `-anchor`, `-timing-mode e2e`,
`-run-order shuffle`, explicit `-shuffle-seed`, `-runs >= 3`,
`-warmup-runs >= 1`, and explicit `-vmaf-model` when VMAF is required, plus a
non-empty `-environment-notes` value and explicit non-empty `-cpu-affinity`,
`-power-mode`, `-thermal-state`, `-frequency-policy`, and `-background-load`
values. It also
requires explicit goav1 execution-lane, effort, and scene-cut settings, an
explicit `-gogc` value with ambient `GOFLAGS`, `GOMEMLIMIT`, and `GODEBUG`
unset, explicit absolute `-ffmpeg-bin`, `-aomenc-bin`, and `-svt-bin` paths
with matching SHA-256 pins for every selected external tool, an explicit
FFmpeg AV1 decoder when external baselines are selected, explicit
libaom concurrency settings and realtime speed setting when `aomenc` is
selected, explicit SVT preset, parallelism, and assembly settings when
`svt-av1` is selected, distinct CSV, metadata, summary, and diagnostic artifact
paths, an empty `-workdir` before timing starts, exact raw I420 input byte
counts for every manifest row,
and the manifest-declared
`fps`, `pix_fmt=i420`, `bit_depth=8`, `chroma=4:2:0`, `sha256`, `source_id`,
`source_url`, `source_license`, and `category` fields. Declared
input hashes are verified before timing starts. Publish mode rejects duplicate
encoder/bitrate entries, requires the
BD-rate anchor to be one of the selected encoders, and requires at least four
distinct bitrate points. When `aomenc` or `svt-av1` baselines are
selected, publish mode requires `-layers 1`; goav1 multi-temporal-layer/SVC
audits must stay goav1-only until equivalent external baseline settings are
implemented and recorded. When `aomenc` or `svt-av1` baselines are selected,
publish mode requires `-goav1-scene-cut=false` because the external low-delay
baseline command lines disable scene-cut-equivalent keyframe insertion.

`-timing-mode core` preserves the historical goav1 timer that accumulates only
per-frame `Encode` calls. Use it for local code-path profiling, not for fair
tables. `-timing-mode e2e` times goav1 setup, encode calls, encoded artifact
writes, and encoder shutdown, while external rows continue to time the encoder
command invocation. Metric YUV is decoded after timing for every encoder;
when an external baseline or explicit `-ffmpeg-av1-decoder` is selected, goav1
rows wrap the persisted low-overhead temporal units in an IVF sidecar and decode
that sidecar through the same FFmpeg AV1 decoder path used for AOM/SVT rows.
goav1-only local runs without an FFmpeg decoder still replay the persisted
low-overhead stream through the public decoder, never encoder reconstruction
buffers. The metadata JSON records command paths, binary SHA-256 hashes, and
version/help probes for the external tools used by the run. It also records
`manifest_sha256`, the Go runtime build settings exposed by the benchmark
binary, full `go env -json` metadata, VMAF model file SHA-256 when
`-vmaf-model path=...` is used, observed CPU affinity/frequency probe data when
available, the external command timeout, the effective `GOMAXPROCS`, CPU
count/model when available, hostname,
OS/kernel version when available, `PATH`, selected Go runtime environment
variables (`GOFLAGS`, `GOGC`, `GOMEMLIMIT`, `GODEBUG`), and structured
CPU-affinity, power-mode, thermal-state, frequency-policy, and background-load
fields plus free-form environment notes for extra context.
It also records `run_order` and `shuffle_seed`. Publish mode requires
`-run-order shuffle -shuffle-seed N` so claim-supporting rows use a
deterministic order without always running the same encoder first. For local
diagnostics, use
`-run-order bitrate-encoder` to preserve the historical loop order,
`-run-order encoder-bitrate` to keep one encoder warm across the bitrate sweep,
or `-run-order shuffle -shuffle-seed N` to rotate encoder/bitrate tuple order
deterministically.
For repeated runs, qualitybench writes one normal CSV row per encoder/bitrate
using the median wall-time measured sample after warmups. Warmups and measured
samples run in deterministic sample passes across the selected encoder/bitrate
tuples, so one tuple does not receive all of its samples in a single load or
thermal window. The metadata JSON stores every measured sample plus min,
median, max, and IQR wall time for the tuple, and records
`sample_order=interleaved-by-sample-pass`. Publish mode also requires the
measured samples for each encoder/bitrate tuple to produce identical compressed
byte counts, encoded artifact hashes, decoded byte counts, and decoded hashes;
hash drift fails the tuple instead of silently picking the fastest or median
sample.
For goav1 rows, `encoded_path` is a replayable `uint32_le length + low-overhead
temporal-unit payload` stream. `compressed_bytes` remains the sum of payload
bytes, while `encoded_bytes` and `encoded_sha256` describe that on-disk
length-prefixed artifact. When metric decode uses FFmpeg, the goav1 row settings
also record the generated IVF sidecar path, container, bytes, and SHA-256.
External encoder IVF outputs are parsed through the shared exact IVF reader, so
the codec signature, dimensions, frame count, non-empty payloads, and complete
frame payloads are validated before payload bytes can affect `actual_bps`.

For speed comparisons against SVT-AV1, do not treat numeric concurrency knobs as
equivalent. `GOMAXPROCS` is a Go scheduler processor cap; goav1
`-goav1-max-threads` is the encoder execution-lane cap passed to
`VideoEncoderConfig.MaxThreads`; SVT-AV1 `--lp` is an encoder parallelism level
in the range `0..6`, where `0` lets SVT choose from the machine. goav1
`-goav1-effort` maps to the WebRTC effort level where `0` is the default
quality/speed balance; `qualitybench -svt-preset` forwards to SVT `--preset`,
where higher presets are faster with a quality tradeoff. Publishable rows must
report both encoders' speed/effort knobs. Use `-gomaxprocs` to make the Go
scheduler cap explicit for a run. A fair report should include the chosen
`GOMAXPROCS`, the chosen `-goav1-max-threads`, the chosen `-goav1-effort`, the
chosen `-svt-preset`, the chosen `-svt-lp`, and the CSV/metadata timing columns:
`encode_wall_sec`, `cpu_user_sec`, `cpu_system_sec`, `cpu_total_sec`, and
`observed_parallelism`. Use wall time for user-visible speed, and CPU seconds
or `observed_parallelism=cpu_total_sec/encode_wall_sec` to check whether one
encoder consumed a larger CPU budget. Publish mode fails a measured tuple when
any successful sample lacks positive wall time or process CPU timing, so copied
tables cannot silently omit CPU-budget evidence. If sweeping SVT levels, report
each `--lp` as an SVT level, not as a target thread count. For a closest-budget SVT
row, sweep `-svt-lp 0..6` and select by measured `observed_parallelism`, not by
matching `GOMAXPROCS=N` to `--lp N`.

Also report SVT's assembly tier. SVT-AV1 `--asm` defaults to `max`, which may
use kernels above baseline NEON on Apple silicon, such as `neon_dotprod` or
`neon_i8mm`. `qualitybench -svt-asm` forwards this limiter and records it in
metadata. Use `-svt-asm neon` for a baseline-NEON row against goav1's current
arm64 SIMD coverage, and omit it or pass `-svt-asm max` for a best-SVT row.

Also report libaom's speed and concurrency settings.
`qualitybench -aom-cpu-used` forwards to `aomenc --cpu-used` in realtime mode,
`-aom-threads` forwards to `aomenc --threads`, and `-aom-row-mt` forwards to
`aomenc --row-mt`. All three are recorded in metadata, so single-thread rows
must use `-aom-threads 1`, row-mt experiments must state `-aom-row-mt 0` or
`-aom-row-mt 1`, and realtime-speed sweeps must state each `-aom-cpu-used`
value. The external baseline commands also pin and record profile-0, 8-bit,
I420 identity settings; `aomenc` is run with `--quiet` so progress logging is
not part of the timed encode path. External decoded YUV must match the exact
expected raw I420 byte count before metrics are accepted.

When VMAF is required, publish mode requires an explicit `-vmaf-model` value.
The value is forwarded to FFmpeg `libvmaf`'s `model` option and written to the
metadata JSON as `vmaf_model`; use this to pin rows to a named model such as
`version=vmaf_v0.6.1` instead of relying on the FFmpeg build's implicit
default.

The external baseline settings recorded in metadata are part of the benchmark
contract:

| Encoder | Low-delay/rate pins | Speed and parallelism pins | Stream and picture pins |
| --- | --- | --- | --- |
| `aomenc` | `--rt`, `--end-usage=cbr`, `--lag-in-frames=0`, `--auto-alt-ref=0`, `--enable-fwd-kf=0`, `--drop-frame=0`, `--buf-sz=1000`, `--buf-initial-sz=500`, `--buf-optimal-sz=600` | `--cpu-used`, `--threads`, and `--row-mt` from `-aom-cpu-used`, `-aom-threads`, and `-aom-row-mt`; `--quiet` is always set | profile 0, 8-bit I420, `--target-bitrate`, `--fps`, `--limit`, `--kf-min-dist`, `--kf-max-dist`, and optional `--tile-columns` |
| `SvtAv1EncApp` | `--rc 2`, `--buf-sz 1000`, `--buf-initial-sz 500`, `--buf-optimal-sz 600`, `--lookahead 0`, `--pred-struct 1`, `--rtc 1`, `--scd 0`, `--enable-tf 0`, `--irefresh-type 2` | `--preset`, `--lp`, and optional `--asm` from `-svt-preset`, `-svt-lp`, and `-svt-asm` | profile 0, level 0, 8-bit I420 (`--color-format 1`), `--tbr`, `--fps-num`, `--fps-denom`, `--frames`, `--keyint`, `--progress 0`, and optional `--tile-columns` |

When `-stats-csv` is set, goav1 rows also include encoder decision counters:
partition choices, block sizes, skip/coded block counts, references, inter
modes, transform types, and tile/frame counts. These counters are diagnostic
evidence for choosing the next parity slice; they are not substitutes for
decoded-output metrics or BD-rate.

Use `-frame-stats-csv` when investigating rate-control or mode-selection
regressions. It records goav1 per-frame bytes, temporal layer, keyframe flag,
qindex before and after encode, encode time, and per-frame decision-counter
deltas. Treat it as diagnostic evidence for explaining RD results, not as a
visual-quality metric.

Use `-frame-metrics-csv` to align decoded-output PSNR/SSIM traces with
`-frame-stats-csv`. This is useful for locating the frames where a rate-control
or mode-decision event becomes visible in decoded quality; it is still
diagnostic context, while clip-level BD-rate and required perceptual metrics
remain the claim-supporting result. Frame indexes in both diagnostic CSVs are
zero-based. Publish mode requires `-frame-metrics-csv` and rejects PSNR/SSIM
frame traces whose index set does not exactly match the configured frame count.
VMAF JSON is likewise accepted only when it contains exactly the configured
number of frame entries.

Use `-metadata-json` for claim-supporting runs. The sidecar records the
goav1 git revision and dirty state, Go runtime, selected configuration,
required corpus settings, metrics, encoders, and summary enforcement,
metric-filter availability, tool paths/version probes, per-clip source
geometry, declared raw format, expected raw byte counts, actual input byte
counts, declared and actual SHA-256 hashes, source/provenance fields,
per-encoder invocations or goav1 settings, compressed payload byte counts,
encoded output SHA-256 hashes, and decoded YUV SHA-256 hashes. It also records
the exact configured binary paths, expected SHA-256 pins, actual binary
SHA-256s, and whether each hash was verified.

For a corpus, use `-manifest` instead of `-input`. The manifest is CSV with a
header. Local exploratory runs may use the minimal geometry columns, but
publishable rows require the full raw-format and provenance columns:

```csv
clip,input,width,height,frames,fps,pix_fmt,bit_depth,chroma,sha256,source_id,source_url,source_license,category
talking_head,clips/talking_head_1920x1080_i420.yuv,1920,1080,120,60,i420,8,4:2:0,0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef,lab-head,https://example.invalid/head,CC-BY-4.0,talking-head
screen,clips/screen_1280x720_i420.yuv,1280,720,120,60,i420,8,4:2:0,fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210,lab-screen,https://example.invalid/screen,CC-BY-4.0,screen-content
```

Relative `input` paths resolve from the manifest's directory. `fps` is optional
for exploratory manifests and falls back to `-fps`; publish manifests must
declare `fps` on every row. If `sha256` is present, it is verified in every mode.
Each clip gets its own work subdirectory and its own raw/summary CSV rows.

When `-require-corpus` is set, `qualitybench` requires `-manifest`, requires
`-min-clips` to be at least 2, rejects manifest rows with an empty `input`, and
checks that each input path exists before encoding. This gate verifies only the
machine-checkable corpus contract; clip category coverage still has to be
curated and documented by the experiment owner.

When `-summary-csv` is set, `qualitybench` writes BD-rate rows for each
candidate encoder against `-anchor` (default: the first encoder in `-encoders`).
Positive `bd_rate_pct` means the candidate needed more bitrate than the anchor
over the common metric range; negative means it needed less. Rows with fewer
than four valid points, missing metrics, or no overlapping quality range are
reported as explicit errors instead of synthesized numbers. When
`-require-summary` is set, any missing or non-`ok` summary row for a
`-require-metrics` metric and selected non-anchor encoder makes the command
exit nonzero after writing the summary CSV.

References:

- [AOM Common Test Conditions v8.0](https://aomedia.org/docs/CWG-F384o_AV2_CTC_v8.pdf)
  defines codec experiments around objective metrics, test sequences,
  configurations, rate-distortion curves, and reporting.
- [Netflix VMAF](https://github.com/Netflix/vmaf) is a full-reference
  perceptual-quality metric; use it when the local toolchain exposes
  `libvmaf`.
