//go:build goav1_oracle

package testvector

// Cross-decoder throughput benchmark.
//
// PURPOSE
//
// TestCrossDecoderThroughput is a PERFORMANCE-TRACKING tool, not a conformance
// test. It times goav1's full software decode against the reference C decoders
// (libaom's aomdec, dav1d, and SVT-AV1's SvtAv1DecApp if present) on the
// bundled libaom conformance IVF vectors so we can watch goav1's relative
// throughput move over time. It deliberately never runs under the normal
// `go test ./...` invocation: it is gated behind GOAV1_CROSS_BENCH=1 AND the
// goav1_oracle build tag, and is invoked via `make bench-cross`, which also
// sets GOAV1_STRICT_MD5=1.
//
// WHAT IS MEASURED (and why it is a FAIR comparison)
//
//  1. SAME WORK. goav1 is timed through runLibaomFrameWorkDryRun — the very
//     same oracle harness the conformance suite uses. That harness runs the
//     FULL decode INCLUDING the post-filter chain (loop-filter / CDEF /
//     loop-restoration / super-res / film-grain) for every completed frame and
//     verifies every emitted frame's MD5 against the official libaom digests.
//     We keep this benchmark on the oracle harness rather than cmd/aom-go-dec
//     so timing rows remain coupled to conformance proof instead of CLI output
//     plumbing. Because the timed path is the MD5-verifying oracle path, a
//     timing run that completes without t.Fatal IS a proof that goav1 produced
//     the correct, fully post-filtered output.
//
//  2. UNIFORM METHODOLOGY. goav1, aomdec, and dav1d are run single-threaded
//     (goav1 worker pool = 1; aomdec --threads=1; dav1d --threads 1). SVT is
//     requested with SvtAv1DecApp --lp 1, which SVT documents as a parallelism
//     level rather than a processor/thread count. All decoders are decode-only
//     with output DISCARDED
//     (aomdec --rawvideo -o /dev/null; dav1d --muxer null -o /dev/null). Each
//     (decoder, vector) is warmed up once, then run best-of-N (crossBenchRuns,
//     N>=7); we keep the MIN wall-clock, which is the standard way to reject
//     scheduler/IO noise.
//
//  3. IN-PROCESS vs SUBPROCESS ASYMMETRY (read this — it dominates the numbers
//     on these tiny clips). goav1 is timed IN-PROCESS: no exec, no process
//     startup, no fork. aomdec/dav1d/SVT are SUBPROCESSES whose measured
//     wall-clock INCLUDES OS process startup, dynamic-linker, and decoder
//     init. The bundled vectors are tiny (e.g. 352x288, a handful of frames),
//     so that fixed per-process overhead is a LARGE fraction of each external
//     run and the raw comparison is heavily startup-biased AGAINST the C
//     decoders. To make this visible and adjustable we MEASURE each external
//     decoder's process-startup baseline (best-of-N of its --help/--version
//     fast path) and report BOTH:
//        - "raw"      : measured wall-clock as-is (includes startup), and
//        - "adjusted" : wall-clock minus the measured startup baseline,
//     clearly labelled. Neither number is "the truth"; the raw number is what
//     a shell pipeline sees, the adjusted number approximates steady-state
//     decode for a long-lived process. goav1 has no startup line because it is
//     in-process. We also aggregate across ALL bundled vectors (total frames /
//     total time) so one tiny clip cannot dominate the headline ratio.
//
// HOW TO ADD A LARGER CLIP LATER
//
// Drop a larger AV1 IVF and its libaom .md5 sidecar into
// internal/av1/testdata/libaom/, add a RemoteVector entry to
// LibaomRemoteManifest() (libaom_manifest.go) tagged with a suite level that
// crossBenchVectors() selects (SuiteLevelFast today), and it is picked up
// automatically. A larger clip amortizes the subprocess-startup overhead and
// makes the raw/adjusted numbers converge — which is exactly why a bigger clip
// is the right way to get a less startup-dominated comparison.
//
// SVT-AV1 AUTO-DETECTION
//
// SvtAv1DecApp is looked up on PATH via exec.LookPath. If present it is added
// to the comparison automatically; if absent (the common case here) it is
// skipped with a logged note. The same graceful skip applies to aomdec and
// dav1d — a missing decoder is never a hard failure.

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/thesyncim/goav1/internal/av1/ivf"
)

// crossBenchRuns is the best-of-N sample count (N>=7 per the methodology). We
// keep the MIN wall-clock across runs to reject scheduler/IO noise.
const crossBenchRuns = 9

// externalDecoder describes one reference C decoder we shell out to.
type externalDecoder struct {
	// name is the column label in the report.
	name string
	// lookups are candidate binary names/paths tried in order with
	// exec.LookPath (and as an absolute path); the first that resolves wins.
	lookups []string
	// decodeArgs builds the decode-only, output-discarded
	// argv (excluding the binary itself) for the given input IVF path.
	decodeArgs func(bin, input string) []string
	// startupArgs builds the argv used to measure fixed process-startup
	// overhead (a fast no-decode path such as --help/--version).
	startupArgs func(bin string) []string
	// versionArgs builds the argv used for report-only binary version capture.
	// It can differ from startupArgs because a low-overhead startup probe is
	// not always the best version probe.
	versionArgs func(bin string) []string
}

// crossBenchExternalDecoders is the registry of reference decoders. SVT-AV1 is
// listed here and auto-included only when SvtAv1DecApp resolves on PATH.
func crossBenchExternalDecoders() []externalDecoder {
	return []externalDecoder{
		{
			name:    "aomdec",
			lookups: []string{"aomdec", "/tmp/aom-build/aomdec", "/opt/homebrew/bin/aomdec"},
			decodeArgs: func(_ string, input string) []string {
				// --rawvideo applies the full post-filter chain and emits
				// displayable frames; -o /dev/null discards them.
				return []string{"--rawvideo", "--threads=1", "-o", os.DevNull, input}
			},
			startupArgs: func(_ string) []string { return []string{"--help"} },
			versionArgs: func(_ string) []string { return []string{"--version"} },
		},
		{
			name:    "dav1d",
			lookups: []string{"dav1d", "/opt/homebrew/bin/dav1d"},
			decodeArgs: func(_ string, input string) []string {
				// --muxer null discards output but still runs the full decode
				// + post-filter (film grain defaults on for the null muxer).
				return []string{"--quiet", "--muxer", "null", "-o", os.DevNull, "--threads", "1", "-i", input}
			},
			startupArgs: func(_ string) []string { return []string{"--version"} },
			versionArgs: func(_ string) []string { return []string{"--version"} },
		},
		{
			// SVT-AV1 decoder: auto-detected, absent on this machine.
			name:    "SvtAv1DecApp",
			lookups: []string{"SvtAv1DecApp"},
			decodeArgs: func(_ string, input string) []string {
				// Request SVT --lp 1, decode-only, discard output. SVT documents
				// --lp as a parallelism level rather than a thread-count knob.
				return []string{"-i", input, "-o", os.DevNull, "--lp", "1"}
			},
			startupArgs: func(_ string) []string { return []string{"--help"} },
			versionArgs: func(_ string) []string { return []string{"--version"} },
		},
	}
}

func (d externalDecoder) versionArgsFor(bin string) []string {
	if d.versionArgs != nil {
		return d.versionArgs(bin)
	}
	return d.startupArgs(bin)
}

// resolveBinary returns the first resolvable binary path among the candidates.
func (d externalDecoder) resolveBinary() (string, bool) {
	for _, cand := range d.lookups {
		if p, err := exec.LookPath(cand); err == nil {
			return p, true
		}
		// Allow an explicit absolute path that is executable but not on PATH.
		if filepath.IsAbs(cand) {
			if info, err := os.Stat(cand); err == nil && !info.IsDir() && info.Mode()&0o111 != 0 {
				return cand, true
			}
		}
	}
	return "", false
}

// crossBenchVector is one bundled vector resolved to an on-disk IVF path with
// its frame count.
type crossBenchVector struct {
	vector RemoteVector
	path   string
	frames int
}

// crossBenchVectors selects the fast-suite vectors and resolves each to its
// on-disk IVF path via EnsureRemoteFile — the exact resolver the oracle
// harness uses. The bundled, checksum-pinned vectors live in the libaom cache
// dir (internal/av1/testdata/libaom); EnsureRemoteFile returns the cached path
// when present and only downloads on a cache miss. Any vector that cannot be
// resolved (e.g. offline cache miss) is skipped with a logged note rather than
// failing the benchmark. We also count each clip's IVF frames for the fps
// denominator, shared by every decoder for an apples-to-apples frames/sec.
func crossBenchVectors(t *testing.T) []crossBenchVector {
	t.Helper()
	cacheDir := libaomCacheDir(t)
	manifest := LibaomRemoteManifest()
	selected := manifest.SelectRemote(SuiteLevelFast, 0, nil)
	out := make([]crossBenchVector, 0, len(selected))
	for _, v := range selected {
		path, err := EnsureRemoteFile(context.Background(), cacheDir, manifest.BaseURL, v.Stream)
		if err != nil {
			t.Logf("cross-bench: skipping %s (could not resolve bundled IVF: %v)", v.Name, err)
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Logf("cross-bench: skipping %s (read %s: %v)", v.Name, path, err)
			continue
		}
		frames, err := countIVFFrames(data)
		if err != nil {
			t.Logf("cross-bench: skipping %s (cannot count frames: %v)", v.Name, err)
			continue
		}
		out = append(out, crossBenchVector{vector: v, path: path, frames: frames})
	}
	return out
}

// countIVFFrames returns the number of IVF frames (temporal units) in the
// stream. This is the per-clip frame count fed to the external decoders and is
// used as the fps denominator across all decoders for an apples-to-apples
// frames/sec.
func countIVFFrames(data []byte) (int, error) {
	it, err := ivf.NewIterator(data)
	if err != nil {
		return 0, err
	}
	n := 0
	for {
		_, ok, err := it.Next()
		if err != nil {
			return 0, err
		}
		if !ok {
			break
		}
		n++
	}
	return n, nil
}

// minDuration runs fn warmup+N times and returns the minimum observed wall
// clock, or an error from the first failing run.
func minDuration(warmup int, runs int, fn func() error) (time.Duration, error) {
	for i := 0; i < warmup; i++ {
		if err := fn(); err != nil {
			return 0, err
		}
	}
	best := time.Duration(1<<63 - 1)
	for i := 0; i < runs; i++ {
		start := time.Now()
		if err := fn(); err != nil {
			return 0, err
		}
		if d := time.Since(start); d < best {
			best = d
		}
	}
	return best, nil
}

// runExternal executes one decoder invocation, discarding stdout/stderr on the
// timed success path. On failure it reruns once with captured output so normal
// benchmark timing never pays progress/log buffering overhead.
func runExternal(bin string, args []string) error {
	cmd := exec.Command(bin, args...)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Run(); err == nil {
		return nil
	}

	cmd = exec.Command(bin, args...)
	var sink bytes.Buffer
	cmd.Stdout = &sink
	cmd.Stderr = &sink
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(sink.String()))
	}
	return nil
}

// decoderResult holds aggregated timing for one decoder across all vectors.
type decoderResult struct {
	name string
	// inProcess marks goav1 (no subprocess startup; no startup baseline).
	inProcess bool
	// startup is the measured fixed process-startup baseline (0 for goav1).
	startup time.Duration
	// startupSamples records every measured startup probe for publishable corpus
	// reports. The tiny cross-bench report does not populate it.
	startupSamples corpusDurationSamples
	// totalRaw is the summed selected wall-clock across all vectors (raw,
	// includes per-invocation startup for external decoders). The tiny
	// cross-bench selects the minimum; the corpus publish benchmark selects the
	// median.
	totalRaw time.Duration
	// totalFrames is the summed IVF frame count across all decoded vectors.
	totalFrames int
	// perVector records the selected raw time per vector path, used for the
	// detail table.
	perVector map[string]time.Duration
	// perVectorSamples records every measured corpus run for the publish JSON
	// sidecar. It is intentionally optional for non-publish benchmark paths.
	perVectorSamples map[string]corpusDurationSamples
}

// TestCrossDecoderThroughput is the gated cross-decoder performance benchmark.
// It never runs under a plain `go test` invocation: it requires both the
// goav1_oracle build tag and GOAV1_CROSS_BENCH=1.
func TestCrossDecoderThroughput(t *testing.T) {
	if os.Getenv("GOAV1_CROSS_BENCH") != "1" {
		t.Skip("set GOAV1_CROSS_BENCH=1 to run the cross-decoder performance benchmark (perf tool, not a conformance gate)")
	}

	vectors := crossBenchVectors(t)
	if len(vectors) == 0 {
		t.Skip("cross-bench: no bundled fast-suite IVF vectors present on disk")
	}

	// ----- goav1 (in-process, full decode + post-filter, MD5-verified) -----
	goav1 := decoderResult{name: "goav1", inProcess: true, perVector: map[string]time.Duration{}}
	for _, cv := range vectors {
		cv := cv
		// runLibaomFrameWorkDryRun verifies every emitted frame's MD5 against
		// the official libaom digests; if it returns without t.Fatal, the
		// timed full decode is provably conformance-correct.
		best, err := minDuration(1, crossBenchRuns, func() error {
			runLibaomFrameWorkDryRun(t, cv.vector)
			return nil
		})
		if err != nil {
			t.Fatalf("goav1 decode %s: %v", cv.vector.Name, err)
		}
		goav1.perVector[cv.path] = best
		goav1.totalRaw += best
		goav1.totalFrames += cv.frames
	}

	results := []decoderResult{goav1}

	// ----- external reference decoders (subprocess, startup-inclusive) -----
	for _, dec := range crossBenchExternalDecoders() {
		bin, ok := dec.resolveBinary()
		if !ok {
			t.Logf("cross-bench: %s not found on PATH (exec.LookPath) — skipping", dec.name)
			continue
		}
		t.Logf("cross-bench: %s resolved to %s", dec.name, bin)

		// Measure fixed process-startup baseline via the fast no-decode path.
		startup, err := minDuration(1, crossBenchRuns, func() error {
			// --help/--version exit non-zero on some builds; ignore the exit
			// status, we only want the startup wall-clock.
			_ = runExternal(bin, dec.startupArgs(bin))
			return nil
		})
		if err != nil {
			t.Logf("cross-bench: %s startup probe failed (%v); treating baseline as 0", dec.name, err)
			startup = 0
		}

		res := decoderResult{name: dec.name, startup: startup, perVector: map[string]time.Duration{}}
		usable := true
		for _, cv := range vectors {
			cv := cv
			best, err := minDuration(1, crossBenchRuns, func() error {
				return runExternal(bin, dec.decodeArgs(bin, cv.path))
			})
			if err != nil {
				t.Logf("cross-bench: %s failed to decode %s (%v) — excluding decoder", dec.name, cv.vector.Name, err)
				usable = false
				break
			}
			res.perVector[cv.path] = best
			res.totalRaw += best
			res.totalFrames += cv.frames
		}
		if usable {
			results = append(results, res)
		}
	}

	printCrossBenchReport(t, vectors, results)
}

// printCrossBenchReport renders the comparison tables and the fairness caveats.
func printCrossBenchReport(t *testing.T, vectors []crossBenchVector, results []decoderResult) {
	t.Helper()

	var b strings.Builder
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "==================================================================================\n")
	fmt.Fprintf(&b, " goav1 cross-decoder throughput  (PERF TRACKING ONLY — NOT a conformance gate)\n")
	fmt.Fprintf(&b, "==================================================================================\n")
	fmt.Fprintf(&b, " vectors: %d bundled libaom IVF clips   best-of-%d (min wall-clock)   requested single-thread controls\n", len(vectors), crossBenchRuns)
	fmt.Fprintf(&b, " goav1: IN-PROCESS, full decode + post-filter (LF/CDEF/LR/super-res/film-grain),\n")
	fmt.Fprintf(&b, "        MD5-verified against official libaom digests (correctness proven by run).\n")
	fmt.Fprintf(&b, " others: SUBPROCESS, decode-only, output discarded; raw time INCLUDES process\n")
	fmt.Fprintf(&b, "         startup. These clips are tiny, so startup dominates the raw numbers.\n")
	fmt.Fprintf(&b, " SVT note: SvtAv1DecApp uses --lp 1 here; --lp is SVT's parallelism level, not a verified thread count.\n")
	fmt.Fprintf(&b, "==================================================================================\n\n")

	// Find goav1 for the speedup baseline.
	var base decoderResult
	for _, r := range results {
		if r.inProcess {
			base = r
		}
	}

	// ---- Aggregate table (total frames / total time across ALL vectors) ----
	fmt.Fprintf(&b, "AGGREGATE (all vectors combined; one tiny clip cannot dominate)\n")
	fmt.Fprintf(&b, "%-14s %7s %12s %10s %12s %10s %10s\n",
		"decoder", "frames", "raw_ms", "raw_fps", "adj_ms", "adj_fps", "vs_goav1")
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 82))

	for _, r := range results {
		rawMS := float64(r.totalRaw.Nanoseconds()) / 1e6
		rawFPS := fpsOf(r.totalFrames, r.totalRaw)

		// Adjusted: subtract one startup baseline per vector invocation
		// (each external decode is a fresh process).
		adj := r.totalRaw - r.startup*time.Duration(len(vectors))
		if adj < 0 {
			adj = 0
		}
		adjMS := float64(adj.Nanoseconds()) / 1e6
		adjFPS := fpsOf(r.totalFrames, adj)

		// Speedup relative to goav1 (raw vs goav1 raw; goav1 has no startup,
		// so its raw == adjusted).
		var vs string
		if r.inProcess {
			vs = "1.00x"
		} else if base.totalRaw > 0 && r.totalRaw > 0 {
			ratio := float64(base.totalRaw) / float64(r.totalRaw)
			vs = fmt.Sprintf("%.2fx", ratio)
		} else {
			vs = "n/a"
		}

		if r.inProcess {
			fmt.Fprintf(&b, "%-14s %7d %12.3f %10.1f %12s %10s %10s\n",
				r.name+"*", r.totalFrames, rawMS, rawFPS, "(in-proc)", "(in-proc)", vs)
		} else {
			fmt.Fprintf(&b, "%-14s %7d %12.3f %10.1f %12.3f %10.1f %10s\n",
				r.name, r.totalFrames, rawMS, rawFPS, adjMS, adjFPS, vs)
		}
	}
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 82))
	fmt.Fprintf(&b, "* goav1 is in-process (no subprocess startup); raw == adjusted for goav1.\n")
	fmt.Fprintf(&b, "  vs_goav1 = goav1_raw / decoder_raw  (>1 means the external RAW time is faster).\n")
	fmt.Fprintf(&b, "  adj_ms subtracts one measured process-startup baseline per vector invocation:\n")
	for _, r := range results {
		if r.inProcess {
			continue
		}
		fmt.Fprintf(&b, "    %-14s startup baseline = %.3f ms/process\n", r.name, float64(r.startup.Nanoseconds())/1e6)
	}
	fmt.Fprintf(&b, "\n")

	// ---- Per-vector detail (raw best wall-clock, ms) ----
	fmt.Fprintf(&b, "PER-VECTOR raw best wall-clock (ms)  [startup-inclusive for external decoders]\n")
	// Sort decoders for stable columns: goav1 first, then by name.
	cols := append([]decoderResult(nil), results...)
	sort.SliceStable(cols, func(i, j int) bool {
		if cols[i].inProcess != cols[j].inProcess {
			return cols[i].inProcess
		}
		return cols[i].name < cols[j].name
	})
	fmt.Fprintf(&b, "%-44s %6s", "vector", "frames")
	for _, c := range cols {
		fmt.Fprintf(&b, " %12s", c.name)
	}
	fmt.Fprintf(&b, " %10s", "dav1d_x")
	fmt.Fprintf(&b, "\n%s\n", strings.Repeat("-", 44+7+13*len(cols)+11))
	for _, cv := range vectors {
		fmt.Fprintf(&b, "%-44s %6d", truncate(cv.vector.Stream.Name, 44), cv.frames)
		var goTime, dav1dTime time.Duration
		for _, c := range cols {
			if d, ok := c.perVector[cv.path]; ok {
				if c.inProcess {
					goTime = d
				}
				if c.name == "dav1d" {
					dav1dTime = d
				}
				fmt.Fprintf(&b, " %12.3f", float64(d.Nanoseconds())/1e6)
			} else {
				fmt.Fprintf(&b, " %12s", "-")
			}
		}
		if goTime > 0 && dav1dTime > 0 {
			fmt.Fprintf(&b, " %9.2fx", float64(goTime)/float64(dav1dTime))
		} else {
			fmt.Fprintf(&b, " %10s", "-")
		}
		fmt.Fprintf(&b, "\n")
	}
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "dav1d_x = goav1 raw ms / dav1d raw ms for that vector; higher is a larger gap.\n")
	fmt.Fprintf(&b, "CAVEAT (do not hide this): on these tiny bundled clips the external decoders'\n")
	fmt.Fprintf(&b, "raw numbers are dominated by per-process startup, NOT steady-state decode. Use\n")
	fmt.Fprintf(&b, "adj_* for a closer steady-state estimate, or add a larger clip (see file header).\n")

	t.Log(b.String())
}

// fpsOf returns frames/second, guarding against zero duration.
func fpsOf(frames int, d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(frames) / d.Seconds()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 3 {
		return s[:n]
	}
	return s[:n-3] + "..."
}
