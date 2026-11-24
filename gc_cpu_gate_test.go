package goav1

import (
	"fmt"
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
// exposure: write barriers, mark assists from allocation, stack growth. The
// threshold is generous so loaded CI runners stay green; a hot-path
// allocation or pointer store per block shows up far above it.
//
// The check runs in a child process with GOMAXPROCS=4, so parallel tests in
// this package cannot disturb it.
func TestGCInvisibleCodecCPU(t *testing.T) {
	const isolatedEnv = "GOAV1_GC_CPU_ISOLATED"
	if _, ok := threadCPU(); !ok {
		t.Skip("no thread CPU clock on this platform")
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
	const (
		rounds    = 7
		threshold = 1.40
	)
	dec, decClose := gcPressureDecodeOp(t, 1)
	defer decClose()
	enc, encClose := gcPressureEncodeOp(t, 1)
	defer encClose()
	synth := gcSynthMem(make([]byte, 4<<20))
	works := []struct {
		name string
		op   func()
		reps int
	}{
		{"decode", dec, 4},
		{"encode", enc, 2},
		{"synth", synth, 4},
	}
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
	for w, wk := range works[:len(works)-1] {
		raw := ratio(w)
		rel := raw / synthRatio
		msg := fmt.Sprintf("%s: thread-CPU churn10/idle %.3f, relative to the synthetic kernel %.3f (limit %.2f)", wk.name, raw, rel, threshold)
		t.Log(msg)
		if rel > threshold {
			t.Error(msg)
		}
	}
}
