# Benchmark Protocol

This document is the benchmark entry point. Codec quality, WebRTC, and
performance claims should use the stricter protocol in
[`quality-validation.md`](quality-validation.md); this page keeps the short
checklist easy to find.

Publishable rows must satisfy these controls:

- Use `make bench-go-publish`, `make bench-corpus-publish`, or
  `make qualitybench-publish`; ad hoc `go test -bench`, `make bench`, and local
  smoke runs are not claim-supporting rows.
- Pin every external tool by absolute path and SHA-256 before timing starts.
- Keep ambient Go target, compiler, cache, and runtime env overrides unset; use
  explicit runner flags for intentional controls.
- Record structured CPU affinity, power mode, thermal state, frequency policy,
  and background-load fields.
- Use corpus-backed real clips for quality claims and require manifest/source
  hashes, exact frame metric traces, and required metrics/encoders.
- Keep timeout, run order, shuffle seed, warmup, measured run count,
  `GOMAXPROCS`, GC, thread/parallelism, bitrate, FPS, scalability, and VMAF
  model settings in metadata.
- Derive any published comparison table from the saved CSV/JSON sidecars with
  stable sorting and fixed formatting; omit volatile fields such as generation
  timestamp, absolute temporary paths, and host name from byte-for-byte table
  artifacts.

For the full command examples and validation rules, see
[`quality-validation.md`](quality-validation.md).
