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
- Measure decoded output against the same source frames. Report actual bitrate
  from compressed payload bytes, not just requested bitrate.
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
  -manifest corpus/clips.csv -fps 60 \
  -bitrates 3000000,6000000,9000000,12000000 \
  -encoders goav1,aomenc,svt-av1 \
  -anchor aomenc -layers 3 -keyint 60 \
  -require-corpus -min-clips 6 \
  -require-encoders all \
  -require-metrics xpsnr,vmaf \
  -gomaxprocs 4 \
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
Publish mode requires a clean tracked git worktree, explicit `-workdir`, `-csv`,
`-metadata-json`, `-manifest`, `-require-corpus`, `-min-clips`,
`-require-encoders all`, `-require-metrics`, `-summary-csv`,
`-require-summary`, and `-gomaxprocs`. It also requires explicit libaom
concurrency settings when `aomenc` is selected, explicit SVT parallelism and
assembly settings when `svt-av1` is selected, and exact raw I420 input byte
counts for every manifest row.

For speed comparisons against SVT-AV1, do not treat numeric concurrency knobs as
equivalent. `GOMAXPROCS` is a Go scheduler processor cap; SVT-AV1 `--lp` is an
encoder parallelism level in the range `0..6`, where `0` lets SVT choose from
the machine. Use `-gomaxprocs` to make the goav1 cap explicit for a run. A fair
report should include the chosen `GOMAXPROCS`, the chosen `-svt-lp`, and the
CSV/metadata timing columns: `encode_wall_sec`,
`cpu_user_sec`, `cpu_system_sec`, `cpu_total_sec`, and
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

Also report libaom's concurrency settings. `qualitybench -aom-threads` forwards
the value to `aomenc --threads`, and `-aom-row-mt` forwards the value to
`aomenc --row-mt`. Both are recorded in metadata, so single-thread rows must use
`-aom-threads 1`, row-mt experiments must state `-aom-row-mt 0` or
`-aom-row-mt 1`, and multi-thread rows must state both chosen values.

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
geometry, expected raw byte counts, actual input byte counts, SHA-256 hashes,
per-encoder invocations or goav1 settings, compressed payload byte counts,
encoded output SHA-256 hashes, and decoded YUV SHA-256 hashes.

For a corpus, use `-manifest` instead of `-input`. The manifest is CSV with a
header and these columns:

```csv
clip,input,width,height,frames,fps
talking_head,clips/talking_head_1920x1080_i420.yuv,1920,1080,120,60
screen,clips/screen_1280x720_i420.yuv,1280,720,120,60
```

Relative `input` paths resolve from the manifest's directory. `fps` is optional
and falls back to `-fps`. Each clip gets its own work subdirectory and its own
raw/summary CSV rows.

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
