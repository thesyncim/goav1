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
  -csv quality.csv -summary-csv quality-summary.csv -require-summary \
  -stats-csv quality-encoder-stats.csv \
  -metadata-json quality-metadata.json \
  -workdir /tmp/goav1-quality
```

If the local FFmpeg build lacks `libvmaf`, a command that requires VMAF exits
before encoding. Use that failure as a toolchain setup signal; do not treat a
non-VMAF run as state-of-the-art visual validation.

If `-input` is omitted, `qualitybench` uses the same deterministic synthetic
scene as `encbench`. That path is for smoke testing the harness, not for quality
claims.

When `-stats-csv` is set, goav1 rows also include encoder decision counters:
partition choices, block sizes, skip/coded block counts, references, inter
modes, transform types, and tile/frame counts. These counters are diagnostic
evidence for choosing the next parity slice; they are not substitutes for
decoded-output metrics or BD-rate.

Use `-metadata-json` for claim-supporting runs. The sidecar records the
goav1 git revision and dirty state, Go runtime, selected configuration,
required corpus settings, metrics, encoders, and summary enforcement,
metric-filter availability, tool paths/version probes, per-clip source
geometry, expected raw byte counts, actual input byte counts, SHA-256 hashes,
and per-encoder invocations or goav1 settings.

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
