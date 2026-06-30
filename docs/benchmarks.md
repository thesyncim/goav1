# Benchmark Protocol

This document is the benchmark entry point. Codec quality, WebRTC, and
performance claims should use the stricter protocol in
[`quality-validation.md`](quality-validation.md); this page keeps the short
checklist easy to find.

Publishable rows must satisfy these controls:

- Use `make bench-go-publish`, `make bench-corpus-publish`, or
  `make qualitybench-publish`; ad hoc `go test -bench`, `make bench`, and local
  smoke runs are not claim-supporting rows.
- Pin the Go executable and every external tool by absolute path and SHA-256
  before timing starts; metadata must collect `go env` from that pinned Go
  executable, not from ambient `PATH`.
- For Go microbenchmark publish rows, use one concrete package and an exact
  `^BenchmarkName$` selector. Publish mode parses the raw `go test` output and
  rejects zero-row runs, unexpected benchmark rows, CPU-suffix drift, or missing
  repeated samples.
- Keep ambient Go target, compiler, cache, and runtime env overrides unset; use
  explicit runner flags for intentional controls.
- Record structured CPU affinity, power mode, thermal state, frequency policy,
  and background-load fields. When the OS exposes CPU affinity/frequency
  probes, metadata records the observed state and publish mode rejects a
  machine-checkable contradiction such as `cpu-affinity=none` under a restricted
  Linux CPU mask.
- Use corpus-backed real clips for quality claims and require manifest/source
  hashes, exact frame metric traces, and required metrics/encoders.
- For generated decoder corpus publish rows, require the v2 manifest produced
  from `GOAV1_BENCH_SOURCES_TSV`; it carries row-level source provenance and
  publish mode rejects fewer than two source clips or two content categories by
  default.
- Keep timeout, run order, shuffle seed, warmup, measured run count,
  `GOMAXPROCS`, GC, thread/parallelism, bitrate, FPS, scalability, and VMAF
  model settings in metadata.
- Derive any published comparison table from the saved CSV/JSON sidecars with
  stable sorting and fixed formatting; omit volatile fields such as generation
  timestamp, absolute temporary paths, and host name from byte-for-byte table
  artifacts.

For the full command examples and validation rules, see
[`quality-validation.md`](quality-validation.md).
