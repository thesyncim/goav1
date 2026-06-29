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
  -goav1-max-threads 4 \
  -goav1-effort 0 \
  -timing-mode e2e \
  -run-order shuffle -shuffle-seed 1 \
  -runs 3 -warmup-runs 1 \
  -aom-cpu-used 8 \
  -svt-preset 13 \
  -svt-lp 0 \
  -csv quality.csv -summary-csv quality-summary.csv -require-summary \
  -stats-csv quality-encoder-stats.csv \
  -metadata-json quality-metadata.json \
  -aom-threads 4 \
  -aom-row-mt 1 \
  -svt-asm neon \
  -publish \
  -workdir /tmp/goav1-quality
```

If the local FFmpeg build lacks `libvmaf`, a command that requires VMAF exits
before encoding. Use that failure as a toolchain setup signal; do not treat a
non-VMAF run as state-of-the-art visual validation.

If `-input` is omitted, `qualitybench` uses the same deterministic synthetic
scene as `encbench`. That path is for smoke testing the harness, not for quality
claims.

Use `-publish` for rows that will be copied into performance or quality tables.
Publish mode requires a clean git worktree, explicit `-bitrates`,
`-encoders`, `-workdir`, `-csv`, `-metadata-json`, `-manifest`,
`-require-corpus`, `-min-clips`, `-require-encoders all`, `-require-metrics`,
`-summary-csv`, `-require-summary`, `-gomaxprocs`, `-fps`, `-layers`,
`-tiles`, `-golden`, `-keyint`, `-anchor`, `-timing-mode e2e`,
`-run-order shuffle`, explicit `-shuffle-seed`, `-runs >= 3`, and
`-warmup-runs >= 1`. It also requires explicit goav1 execution-lane and effort
settings, explicit libaom concurrency settings and realtime speed setting when
`aomenc` is selected, explicit SVT preset, parallelism, and assembly settings
when `svt-av1` is selected, exact raw I420 input byte counts for every manifest
row, and manifest-declared
`pix_fmt=i420`, `bit_depth=8`, `chroma=4:2:0`, `sha256`, `source_id`,
`source_url`, `source_license`, and `category` fields. Declared
input hashes are verified before timing starts. Publish mode rejects duplicate
encoder/bitrate entries, requires the
BD-rate anchor to be one of the selected encoders, and requires at least four
distinct bitrate points. When `aomenc` or `svt-av1` baselines are
selected, publish mode requires `-layers 1`; goav1 multi-temporal-layer/SVC
audits must stay goav1-only until equivalent external baseline settings are
implemented and recorded.

`-timing-mode core` preserves the historical goav1 timer that accumulates only
per-frame `Encode` calls. Use it for local code-path profiling, not for fair
tables. `-timing-mode e2e` times goav1 setup, encode calls, encoded artifact
writes, and encoder shutdown, while external rows continue to time the encoder
command invocation. Metric YUV is decoded after timing for every encoder;
goav1 rows are decoded by replaying the persisted low-overhead payload stream
through the public decoder, not by scoring encoder reconstruction buffers. The
metadata JSON records command paths, binary SHA-256 hashes, and version/help
probes for the external tools used by the run. It also records
`manifest_sha256`, the effective `GOMAXPROCS`, CPU count/model when available,
and selected Go runtime environment variables (`GOFLAGS`, `GOGC`, `GOMEMLIMIT`,
`GODEBUG`).
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
`sample_order=interleaved-by-sample-pass`.
For goav1 rows, `encoded_path` is a replayable `uint32_le length + low-overhead
temporal-unit payload` stream. `compressed_bytes` remains the sum of payload
bytes, while `encoded_bytes` and `encoded_sha256` describe that on-disk
length-prefixed artifact.

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
encoder consumed a larger CPU budget. If sweeping SVT levels, report each
`--lp` as an SVT level, not as a target thread count. For a closest-budget SVT
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
I420 identity settings; external decoded YUV must match the exact expected raw
I420 byte count before metrics are accepted.

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
zero-based.

Use `-metadata-json` for claim-supporting runs. The sidecar records the
goav1 git revision and dirty state, Go runtime, selected configuration,
required corpus settings, metrics, encoders, and summary enforcement,
metric-filter availability, tool paths/version probes, per-clip source
geometry, declared raw format, expected raw byte counts, actual input byte
counts, declared and actual SHA-256 hashes, source/provenance fields,
per-encoder invocations or goav1 settings, compressed payload byte counts,
encoded output SHA-256 hashes, and decoded YUV SHA-256 hashes.

For a corpus, use `-manifest` instead of `-input`. The manifest is CSV with a
header. Local exploratory runs may use the minimal geometry columns, but
publishable rows require the full raw-format and provenance columns:

```csv
clip,input,width,height,frames,fps,pix_fmt,bit_depth,chroma,sha256,source_id,source_url,source_license,category
talking_head,clips/talking_head_1920x1080_i420.yuv,1920,1080,120,60,i420,8,4:2:0,0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef,lab-head,https://example.invalid/head,CC-BY-4.0,talking-head
screen,clips/screen_1280x720_i420.yuv,1280,720,120,60,i420,8,4:2:0,fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210,lab-screen,https://example.invalid/screen,CC-BY-4.0,screen-content
```

Relative `input` paths resolve from the manifest's directory. `fps` is optional
and falls back to `-fps`. If `sha256` is present, it is verified in every mode.
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
