package goav1

import (
	"fmt"
	"maps"
	"os"
	"os/exec"
	"slices"
	"testing"
	"time"
)

// TestGCInvisibleCodecCPU bounds how much CPU time the codec's own thread
// loses while the application's GC runs. Decode (one worker) and encode (one
// thread) run on a locked OS thread, interleaved with a synthetic
// non-allocating memory kernel, idle and under churn10 (pointerful garbage at
// GOGC=10, collections back to back), for several rounds. The gated value is
// the codec's churn/idle thread-CPU ratio divided by the synthetic kernel's:
// what any busy neighbour costs a thread (shared cache, memory bandwidth,
// clock, core placement) cancels, and what is left is the codec's own
// exposure: write barriers, mark assists from allocation, stack growth.
// Calibration on a loaded M4 Max (load average 70-90): the codec measures
// 0.93-1.06, and 30000 small pointerful allocations per decode op (about
// 2 MiB, the size of one stray allocation per block) measure 1.38. The
// per-op allocation tests catch allocations directly; this gate catches GC
// exposure that allocates nothing. It reports only and never fails (see below).
//
// The check runs in a child process with GOMAXPROCS=4, so parallel tests in
// this package cannot disturb it.
func TestGCInvisibleCodecCPU(t *testing.T) {
	const isolatedEnv = "GOAV1_GC_CPU_ISOLATED"
	if _, ok := threadCPU(); !ok {
		t.Skip("no thread CPU clock on this platform")
	}
	if raceEnabled {
		t.Skip("race instrumentation distorts codec CPU and takes too long for this gate")
	}
	if os.Getenv(isolatedEnv) != "1" {
		if testing.Short() {
			t.Skip("spawns a child test process")
		}
		exe, err := os.Executable()
		if err != nil {
			t.Fatalf("os.Executable: %v", err)
		}
		cmd := exec.Command(exe, "-test.run=^TestGCInvisibleCodecCPU$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), isolatedEnv+"=1", "GOMAXPROCS=4", "GOGC=100", "GOMEMLIMIT=off")
		out, err := cmd.CombinedOutput()
		t.Logf("isolated run:\n%s", out)
		if err != nil {
			t.Fatalf("isolated codec CPU check failed: %v", err)
		}
		return
	}
	dec, decClose := gcPressureDecodeOp(t, 1)
	defer decClose()
	enc, encClose := gcPressureEncodeOp(t, 1)
	defer encClose()
	synth := gcSynthMem(make([]byte, 4<<20))
	works := []gcCPUWork{
		{"decode", dec, 4},
		{"encode", enc, 2},
		{"synth", synth, 4},
	}
	// A shared runner can be noisy for a whole measurement (the synthetic
	// kernel's own churn ratio has been seen at 1.31 and 1.39 on runs of the
	// same commit), so a work that is over the limit is measured once more
	// and fails only if it is over the limit both times. A real regression is
	// over on every attempt (the injected 30000 allocations per decode op
	// measured 1.38 on the loaded calibration machine).
	var over map[string]string
	for attempt := 1; attempt <= 2; attempt++ {
		rel, msgs := measureGCInvisibleCPU(t, works)
		next := map[string]string{}
		for name, r := range rel {
			t.Logf("attempt %d: %s", attempt, msgs[name])
			if r > gcCPUThreshold && (attempt == 1 || over[name] != "") {
				next[name] = msgs[name]
			}
		}
		over = next
		if len(over) == 0 {
			return
		}
	}
	// Report only. On shared CI runners this ratio has failed three times on
	// commits that change no GC behaviour: decode at 1.21-1.24 while main and
	// the change measured the same locally. On a quiet machine, 30000 injected
	// allocations per decode op measured only 1.02-1.12. It is neither stable
	// nor sensitive enough to gate. The deterministic gates guard GC
	// invisibility instead: zero codec allocation sites, scannable-heap
	// ceilings, pointer-free block visits and park-allocation counts.
	for _, name := range slices.Sorted(maps.Keys(over)) {
		t.Logf("over the limit on both attempts (report only); last: %s", over[name])
	}
}

// gcCPUThreshold is TestGCInvisibleCodecCPU's limit on a codec's churn10/idle
// thread-CPU ratio relative to the synthetic kernel's.
const gcCPUThreshold = 1.20

type gcCPUWork struct {
	name string
	op   func()
	reps int
}

// measureGCInvisibleCPU runs the idle/churn10 rounds once and returns, for
// every work but the last (the synthetic kernel), its churn10/idle
// thread-CPU ratio relative to the synthetic kernel's, with a log line each.
func measureGCInvisibleCPU(t *testing.T, works []gcCPUWork) (rel map[string]float64, msgs map[string]string) {
	t.Helper()
	const rounds = 7
	modes := []string{"idle", "churn10"}
	cpu := make([][][]float64, len(modes)) // [mode][work][round]
	for m := range cpu {
		cpu[m] = make([][]float64, len(works))
	}
	var churnCycles float64
	for range rounds {
		for m, mode := range modes {
			stop := startGCPressure(mode)
			time.Sleep(50 * time.Millisecond)
			for w, wk := range works {
				r := measureLocked(wk.reps, wk.op)
				cpu[m][w] = append(cpu[m][w], r.codecCPU)
				if mode == "churn10" {
					churnCycles += r.cycles * float64(wk.reps)
				}
			}
			stop()
		}
	}
	if churnCycles < float64(rounds) {
		t.Fatalf("churn10 ran %.0f collections over %d rounds: regime not engaged", churnCycles, rounds)
	}
	med := func(v []float64) float64 {
		v = slices.Clone(v)
		slices.Sort(v)
		return v[len(v)/2]
	}
	ratio := func(w int) float64 { return med(cpu[1][w]) / med(cpu[0][w]) }
	synthRatio := ratio(len(works) - 1)
	t.Logf("churn10: %.0f collections; synthetic kernel thread-CPU ratio %.3f", churnCycles, synthRatio)
	rel = map[string]float64{}
	msgs = map[string]string{}
	for w, wk := range works[:len(works)-1] {
		raw := ratio(w)
		rel[wk.name] = raw / synthRatio
		msgs[wk.name] = fmt.Sprintf("%s: thread-CPU churn10/idle %.3f, relative to the synthetic kernel %.3f (limit %.2f)", wk.name, raw, rel[wk.name], gcCPUThreshold)
	}
	return rel, msgs
}
