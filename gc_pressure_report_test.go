package goav1

import (
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// gcPressureControlRegimes are background loads that keep the machine as busy
// as a churning application but run no collection. They separate what any
// busy neighbour costs a thread (shared cache, memory bandwidth, cluster
// clock) from what the collector costs:
//
//   - spin: a goroutine in a register-only loop (no memory traffic);
//   - traffic: the churner's pointer writes over the same 64 MiB ring,
//     without allocating, so no collection is triggered.
var gcPressureControlRegimes = []string{"spin", "traffic"}

func startControlPressure(mode string) func() {
	var stop atomic.Bool
	var wg sync.WaitGroup
	spin := func() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			x := uint64(1)
			for !stop.Load() {
				for range 1 << 12 {
					x = x*6364136223846793005 + 1442695040888963407
				}
			}
			gcPressureSink.Store(x)
		}()
	}
	switch mode {
	case "spin":
		spin()
	case "traffic":
		const live = 64 << 20 / 64
		ring := make([]*gcPressureNode, live)
		for i := range ring {
			ring[i] = new(gcPressureNode)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			v := new(int)
			for i := 0; !stop.Load(); i++ {
				n := ring[i%live]
				n.next = ring[(i+1)%live]
				n.payload[0] = v
			}
		}()
	}
	return func() {
		stop.Store(true)
		wg.Wait()
	}
}

var gcPressureSink atomic.Uint64

// gcSynthALU is a non-allocating register/L1 loop: what a perfectly
// GC-invisible compute kernel loses to a busy neighbour.
func gcSynthALU(buf []byte) func() {
	return func() {
		var s uint32
		for range 3200 {
			for i := 0; i+8 <= len(buf); i += 8 {
				s = s*31 + uint32(buf[i])*uint32(buf[i+3]) + uint32(buf[i+7])
			}
		}
		gcPressureSink.Store(uint64(s))
	}
}

// gcSynthMem streams a codec-sized (4 MiB) working set with reads and writes.
func gcSynthMem(buf []byte) func() {
	return func() {
		for range 48 {
			for i := 64; i < len(buf); i += 16 {
				buf[i] += buf[i-64] >> 1
			}
		}
	}
}

// TestGCPressureReport prints the codec's per-op thread CPU and wall time
// under each GC regime and each control load, relative to idle, next to two
// synthetic non-allocating kernels. Regimes are interleaved round by round so
// a change in machine load hits all of them alike; each cell is the median of
// the rounds. It is a measurement tool: set GOAV1_GC_PRESSURE_REPORT=1 and
// pin GOMAXPROCS (for example GOMAXPROCS=4) on a machine with idle cores.
func TestGCPressureReport(t *testing.T) {
	if os.Getenv("GOAV1_GC_PRESSURE_REPORT") == "" {
		t.Skip("set GOAV1_GC_PRESSURE_REPORT=1")
	}
	if _, ok := threadCPU(); !ok {
		t.Skip("no thread CPU clock")
	}
	rounds := 10
	type workload struct {
		name string
		op   func()
		reps int
	}
	dec, decClose := gcPressureDecodeOp(t, 1)
	defer decClose()
	enc, encClose := gcPressureEncodeOp(t, 1)
	defer encClose()
	alu := make([]byte, 16<<10)
	mem := make([]byte, 4<<20)
	for i := range alu {
		alu[i] = byte(i * 7)
	}
	works := []workload{
		{"decode w1", dec, 6},
		{"encode t1", enc, 3},
		{"synth ALU", gcSynthALU(alu), 6},
		{"synth mem", gcSynthMem(mem), 6},
	}
	regimes := append(append([]string{"idle", "idle2"}, gcPressureRegimes[1:]...), gcPressureControlRegimes...)
	cpu := map[string][][]float64{}
	wall := map[string][][]float64{}
	gcCycles := map[string]float64{}
	for range rounds {
		for _, mode := range regimes {
			var stop func()
			if slices.Contains(gcPressureControlRegimes, mode) {
				stop = startControlPressure(mode)
			} else {
				stop = startGCPressure(mode)
			}
			time.Sleep(150 * time.Millisecond) // reach the regime's steady state
			for wi, w := range works {
				r := measureLocked(w.reps, w.op)
				if cpu[mode] == nil {
					cpu[mode] = make([][]float64, len(works))
					wall[mode] = make([][]float64, len(works))
				}
				cpu[mode][wi] = append(cpu[mode][wi], r.codecCPU)
				wall[mode][wi] = append(wall[mode][wi], r.wallNs)
				gcCycles[mode] += r.cycles * float64(w.reps)
			}
			stop()
		}
	}
	med := func(v []float64) float64 {
		v = slices.Clone(v)
		slices.Sort(v)
		return v[len(v)/2]
	}
	var b strings.Builder
	fmt.Fprintf(&b, "GOMAXPROCS=%d rounds=%d; cells: thread-CPU ratio / wall ratio vs idle (median of rounds)\n", runtime.GOMAXPROCS(0), rounds)
	fmt.Fprintf(&b, "%-10s %8s", "regime", "gc/round")
	for _, w := range works {
		fmt.Fprintf(&b, " %17s", w.name)
	}
	b.WriteString("\n")
	for _, mode := range regimes {
		fmt.Fprintf(&b, "%-10s %8.1f", mode, gcCycles[mode]/float64(rounds))
		for wi := range works {
			c := med(cpu[mode][wi]) / med(cpu["idle"][wi])
			wl := med(wall[mode][wi]) / med(wall["idle"][wi])
			fmt.Fprintf(&b, "     %5.3f / %5.3f", c, wl)
		}
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "idle per op: ")
	for wi, w := range works {
		fmt.Fprintf(&b, "%s %.2fms wall; ", w.name, med(wall["idle"][wi])/1e6)
	}
	t.Log("\n" + b.String())
}

// measureLocked runs op n times on a locked OS thread under whatever load is
// already running.
func measureLocked(n int, op func()) gcPressureResult {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	before := readGCPressure()
	cpu0, _ := threadCPU()
	t0 := time.Now()
	for range n {
		op()
	}
	wall := time.Since(t0)
	cpu1, _ := threadCPU()
	after := readGCPressure()
	return gcPressureResult{
		wallNs:   float64(wall) / float64(n),
		codecCPU: float64(cpu1-cpu0) / float64(n),
		cycles:   (after.cycles - before.cycles) / float64(n),
	}
}
