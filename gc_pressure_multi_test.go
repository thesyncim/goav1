//go:build darwin || linux

package goav1

import (
	"fmt"
	"os"
	"runtime"
	"runtime/metrics"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

// Multi-worker GC exposure. With several decode workers or encoder lanes the
// codec's CPU is spread over goroutines on any OS thread, so a single
// thread's clock no longer covers it. The multi-worker measurement takes the
// process CPU time (getrusage) and removes what is not the codec's:
//
//   - the churn goroutine, which owns its OS thread and publishes that
//     thread's CPU time (gcChurnCPU), including its own mark assists;
//   - the collector's background work: /cpu/classes/gc/total minus mark
//     assists (the churner's assists are already in its thread, the codec's
//     are zero), and the scavenger.
//
// What is left is the codec's workers, the scheduler work they cause
// (parking, waking, spinning Ms), their write barriers and stack scans. The
// runtime's GC CPU classes are estimates from wall time, refreshed once per
// cycle, so the result is cross-checked against a synthetic non-allocating
// kernel with the same shape (worker goroutines fed and joined through
// channels) and attributed with cycle-sampled traces of single-regime runs.

// procCPU returns the process's user+system CPU time in nanoseconds.
func procCPU() int64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return (int64(ru.Utime.Sec)+int64(ru.Stime.Sec))*1e9 + (int64(ru.Utime.Usec)+int64(ru.Stime.Usec))*1e3
}

// threadCPUNs converts threadCPU units to nanoseconds, calibrated once against
// the process clock while only the calling thread is busy.
var threadCPUNs = sync.OnceValue(func() float64 {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	best := 0.0
	for range 3 {
		p0 := procCPU()
		t0, _ := threadCPU()
		x := uint64(1)
		for d := time.Now(); time.Since(d) < 40*time.Millisecond; {
			for range 1 << 14 {
				x = x*6364136223846793005 + 1442695040888963407
			}
		}
		gcPressureSink.Store(x)
		t1, _ := threadCPU()
		p1 := procCPU()
		if r := float64(p1-p0) / float64(t1-t0); best == 0 || r < best {
			best = r
		}
	}
	return best
})

var gcBackgroundMetricNames = []string{
	"/cpu/classes/gc/total:cpu-seconds",
	"/cpu/classes/gc/mark/assist:cpu-seconds",
	"/cpu/classes/scavenge/total:cpu-seconds",
	"/gc/cycles/total:gc-cycles",
}

// gcBackground returns the collector's and scavenger's CPU time outside mark
// assists, in nanoseconds, and the GC cycle count.
func gcBackground() (int64, uint64) {
	s := make([]metrics.Sample, len(gcBackgroundMetricNames))
	for i, n := range gcBackgroundMetricNames {
		s[i].Name = n
	}
	metrics.Read(s)
	ns := (s[0].Value.Float64() - s[1].Value.Float64() + s[2].Value.Float64()) * 1e9
	return int64(ns), s[3].Value.Uint64()
}

type multiCPU struct {
	proc, churn, gc float64 // ns per op
	cycles          float64 // GC cycles per op
}

// codec is the process CPU attributable to the measured work.
func (m multiCPU) codec() float64 { return m.proc - m.churn - m.gc }

// measureMulti runs op n times under whatever load is running and splits the
// process CPU time it took.
func measureMulti(n int, op func()) multiCPU {
	conv := threadCPUNs()
	g0, c0 := gcBackground()
	ch0 := gcChurnCPU.Load()
	p0 := procCPU()
	for range n {
		op()
	}
	p1 := procCPU()
	ch1 := gcChurnCPU.Load()
	g1, c1 := gcBackground()
	fn := float64(n)
	return multiCPU{
		proc:   float64(p1-p0) / fn,
		churn:  float64(ch1-ch0) * conv / fn,
		gc:     float64(g1-g0) / fn,
		cycles: float64(c1-c0) / fn,
	}
}

// gcSynthPool is the multi-worker control: workers persistent goroutines that
// each stream their own share of a 4 MiB working set per op, fed and joined
// through channels like the codec's pools.
func gcSynthPool(workers int) (func(), func()) {
	work := make(chan int, workers)
	done := make(chan struct{}, workers)
	bufs := make([][]byte, workers)
	for i := range bufs {
		bufs[i] = make([]byte, (4<<20)/workers)
	}
	for range workers {
		go func() {
			for i := range work {
				gcSynthWork(bufs[i%workers])
				done <- struct{}{}
			}
		}()
	}
	return func() {
		for range 16 {
			for i := range workers {
				work <- i
			}
			for range workers {
				<-done
			}
		}
	}, func() { close(work) }
}

// gcSynthWork streams one worker's share of the synthetic working set.
func gcSynthWork(buf []byte) {
	for range 12 {
		for j := 64; j < len(buf); j += 16 {
			buf[j] += buf[j-64] >> 1
		}
	}
}

// gcSynthRespawn is gcSynthPool with workers that exit when their share is
// done and are spawned again for the next one, instead of parking on a
// channel. Each worker is a preallocated no-argument func value, so the go
// statement allocates no closure and reuses a dead g; the workers' result
// sends go to a buffered channel and never block. Only the caller's join
// parks.
func gcSynthRespawn(workers int) (func(), func()) {
	done := make(chan struct{}, workers)
	fns := make([]func(), workers)
	for i := range fns {
		buf := make([]byte, (4<<20)/workers)
		fns[i] = func() {
			gcSynthWork(buf)
			done <- struct{}{}
		}
	}
	return func() {
		for range 16 {
			for _, fn := range fns {
				go fn()
			}
			for range workers {
				<-done
			}
		}
	}, func() {}
}

// gcSynthStocked is gcSynthPool with a sudog stocker: after every
// collection (which empties the runtime's central sudog list) a herd of
// goroutines parks on one channel at once and is then released, so the
// sudogs the herd allocates (on its own goroutines, paying any sweep debt
// there) spill from the P caches back into the central list, where a worker
// whose P ran dry may find them. It is probabilistic: measured with forced
// collections it cuts the pool's sudog allocations about 8x, not to zero,
// and allocates about 136 sudogs per collection on the herd, most of which
// the next collection drops from the central list again.
func gcSynthStocked(workers int) (func(), func()) {
	op, closePool := gcSynthPool(workers)
	var stopped atomic.Bool
	var arm func()
	arm = func() {
		sentinel := new(struct {
			p   *int
			pad [32]byte
		})
		runtime.AddCleanup(sentinel, func(struct{}) {
			if stopped.Load() {
				return
			}
			gcSynthStockerHerd(256)
			arm()
		}, struct{}{})
	}
	arm()
	return op, func() {
		stopped.Store(true)
		closePool()
	}
}

// gcSynthStockerHerd parks n goroutines on one channel together, then
// releases them.
func gcSynthStockerHerd(n int) {
	gate := make(chan struct{})
	var parked, exited sync.WaitGroup
	parked.Add(n)
	exited.Add(n)
	for range n {
		go func() {
			parked.Done()
			<-gate
			exited.Done()
		}()
	}
	parked.Wait()
	for range 64 {
		runtime.Gosched() // let the herd reach the receive
	}
	close(gate)
	exited.Wait()
}

// gcSynthDepth grows the calling goroutine's stack by about kib KiB, the way a
// codec worker's call chain does before its first block.
//
//go:noinline
func gcSynthDepth(kib int) byte {
	var frame [1024]byte
	frame[kib&1023] = byte(kib)
	if kib > 1 {
		frame[0] += gcSynthDepth(kib - 1)
	}
	return frame[0] + frame[kib&1023]
}

// BenchmarkGCSynthRespawnCost is the scheduling cost of one dispatch-and-join
// of 8 workers whose call chain needs a codec-sized stack, with persistent
// workers parked on a channel against workers that exit and are respawned (a
// dead g keeps only a starting-size stack, so a respawned worker grows its
// stack again from 2 KiB). The work
// itself is the stack growth plus a token store, so cpu-ns/op is the
// scheduling and stack overhead. No collection runs.
func BenchmarkGCSynthRespawnCost(b *testing.B) {
	const workers = 8
	for _, kib := range []int{4, 16, 32, 64} {
		b.Run(fmt.Sprintf("chan/stack=%dKiB", kib), func(b *testing.B) {
			work := make(chan int, workers)
			done := make(chan struct{}, workers)
			for range workers {
				go func() {
					for k := range work {
						gcPressureSink.Add(uint64(gcSynthDepth(k)))
						done <- struct{}{}
					}
				}()
			}
			defer close(work)
			gcSynthRespawnBench(b, func() {
				for range workers {
					work <- kib
				}
				for range workers {
					<-done
				}
			})
		})
		b.Run(fmt.Sprintf("respawn/stack=%dKiB", kib), func(b *testing.B) {
			done := make(chan struct{}, workers)
			fn := func() {
				gcPressureSink.Add(uint64(gcSynthDepth(kib)))
				done <- struct{}{}
			}
			gcSynthRespawnBench(b, func() {
				for range workers {
					go fn()
				}
				for range workers {
					<-done
				}
			})
		})
	}
}

func gcSynthRespawnBench(b *testing.B, op func()) {
	op()
	runtime.GC()
	b.ResetTimer()
	p0 := procCPU()
	for b.Loop() {
		op()
	}
	b.ReportMetric(float64(procCPU()-p0)/float64(b.N), "cpu-ns/op")
}

// TestGCParkProbe counts, per park site, the sudog and g allocations of the
// synthetic pools and the codec workloads with a collection forced before
// every op. A measurement tool: set GOAV1_GC_PRESSURE_MULTI=1.
func TestGCParkProbe(t *testing.T) {
	if os.Getenv("GOAV1_GC_PRESSURE_MULTI") == "" {
		t.Skip("set GOAV1_GC_PRESSURE_MULTI=1")
	}
	synthSite := func(f runtime.Frame) bool {
		return strings.Contains(f.Function, "gcSynth") && !strings.Contains(f.Function, "gcSynthStockerHerd")
	}
	for _, s := range []struct {
		name string
		mk   func(int) (func(), func())
	}{{"synth chan", gcSynthPool}, {"synth respawn", gcSynthRespawn}, {"synth stocked", gcSynthStocked}} {
		op, closeFn := s.mk(8)
		op()
		measureParkAllocs(200, op, synthSite).check(t, s.name)
		closeFn()
	}
	dec, closeDec := gcPressureDecodeOp(t, 8)
	dec()
	measureParkAllocs(8, dec, isCodecFrame).check(t, "decode w8 (per clip)")
	closeDec()
	enc, closeEnc := gcPressureEncodeOp(t, 0)
	enc()
	measureParkAllocs(8, enc, isCodecFrame).check(t, "encode t0 (16 frames)")
	closeEnc()
}

// gcPressureMark spins a few milliseconds in a function named after the
// regime's index, so a cycle-sampled trace of an interleaved run can split its
// samples into measurement windows by regime.
func gcPressureMark(i int) {
	switch i {
	case 0:
		gcPressureMarkRegime0()
	case 1:
		gcPressureMarkRegime1()
	case 2:
		gcPressureMarkRegime2()
	default:
		gcPressureMarkRegime3()
	}
}

//go:noinline
func gcPressureMarkRegime0() { gcPressureSpin() }

//go:noinline
func gcPressureMarkRegime1() { gcPressureSpin() }

//go:noinline
func gcPressureMarkRegime2() { gcPressureSpin() }

//go:noinline
func gcPressureMarkRegime3() { gcPressureSpin() }

func gcPressureSpin() {
	x := uint64(1)
	for d := time.Now(); time.Since(d) < 5*time.Millisecond; {
		for range 1 << 10 {
			x = x*6364136223846793005 + 1442695040888963407
		}
	}
	gcPressureSink.Store(x)
}

// TestGCPressureMulti prints the multi-worker codec CPU (process CPU minus the
// churner and the collector's background work) per regime relative to idle,
// next to a synthetic worker-pool kernel. Regimes are interleaved round by
// round. It is a measurement tool: set GOAV1_GC_PRESSURE_MULTI=1 and pin
// GOMAXPROCS. GOAV1_GC_PRESSURE_WORK (decode, encode, synth, synthrespawn,
// synthstock; comma list)
// and GOAV1_GC_PRESSURE_REGIMES select a subset, which makes single-regime
// processes for cycle-sampled traces; GOAV1_GC_PRESSURE_ROUNDS sets the
// round count; GOAV1_GC_PRESSURE_IVF picks the decode clip.
func TestGCPressureMulti(t *testing.T) {
	if os.Getenv("GOAV1_GC_PRESSURE_MULTI") == "" {
		t.Skip("set GOAV1_GC_PRESSURE_MULTI=1")
	}
	if _, ok := threadCPU(); !ok {
		t.Skip("no thread CPU clock")
	}
	rounds := 8
	if s := os.Getenv("GOAV1_GC_PRESSURE_ROUNDS"); s != "" {
		rounds, _ = strconv.Atoi(s)
	}
	regimes := []string{"idle", "idle2", "churn10", "memlimit"}
	if s := os.Getenv("GOAV1_GC_PRESSURE_REGIMES"); s != "" {
		regimes = strings.Split(s, ",")
	}
	want := map[string]bool{"decode": true, "encode": true, "synth": true}
	if s := os.Getenv("GOAV1_GC_PRESSURE_WORK"); s != "" {
		want = map[string]bool{}
		for _, w := range strings.Split(s, ",") {
			want[w] = true
		}
	}
	type workload struct {
		name string
		op   func()
		reps int
	}
	var works []workload
	if want["decode"] {
		dec, closeFn := gcPressureDecodeOp(t, 8)
		defer closeFn()
		works = append(works, workload{"decode w8", dec, 3})
	}
	if want["encode"] {
		enc, closeFn := gcPressureEncodeOp(t, 0)
		defer closeFn()
		works = append(works, workload{"encode t0", enc, 2})
	}
	for _, sw := range []struct {
		key, name string
		mk        func(int) (func(), func())
	}{{"synth", "synth pool8", gcSynthPool}, {"synthrespawn", "synth respawn8", gcSynthRespawn}, {"synthstock", "synth stocked8", gcSynthStocked}} {
		if want[sw.key] {
			synth, closeFn := sw.mk(8)
			defer closeFn()
			works = append(works, workload{sw.name, synth, 3})
		}
	}
	for _, w := range works {
		w.op() // warm
	}
	cpu := map[string][][]multiCPU{}
	for range rounds {
		for _, mode := range regimes {
			stop := startGCPressure(strings.TrimSuffix(mode, "2"))
			time.Sleep(100 * time.Millisecond)
			if cpu[mode] == nil {
				cpu[mode] = make([][]multiCPU, len(works))
			}
			for wi, w := range works {
				gcPressureMark(slices.Index(regimes, mode))
				cpu[mode][wi] = append(cpu[mode][wi], measureMulti(w.reps, w.op))
			}
			stop()
		}
	}
	med := func(v []multiCPU, f func(multiCPU) float64) float64 {
		x := make([]float64, len(v))
		for i := range v {
			x[i] = f(v[i])
		}
		slices.Sort(x)
		return x[len(x)/2]
	}
	codec := func(m multiCPU) float64 { return m.codec() }
	proc := func(m multiCPU) float64 { return m.proc }
	var b strings.Builder
	fmt.Fprintf(&b, "GOMAXPROCS=%d rounds=%d; cells: codec CPU ratio vs idle (process CPU ratio) [gc cycles/op, churn ms/op, gc-bg ms/op]\n", runtime.GOMAXPROCS(0), rounds)
	for _, mode := range regimes {
		fmt.Fprintf(&b, "%-9s", mode)
		for wi, w := range works {
			v := cpu[mode][wi]
			base := cpu[regimes[0]][wi]
			fmt.Fprintf(&b, "  %s %.3f (%.3f) [%.1f, %.1f, %.1f]", w.name,
				med(v, codec)/med(base, codec), med(v, proc)/med(base, proc),
				med(v, func(m multiCPU) float64 { return m.cycles }),
				med(v, func(m multiCPU) float64 { return m.churn })/1e6,
				med(v, func(m multiCPU) float64 { return m.gc })/1e6)
		}
		b.WriteString("\n")
	}
	for wi, w := range works {
		fmt.Fprintf(&b, "%s idle codec CPU %.1f ms/op; ", w.name, med(cpu[regimes[0]][wi], codec)/1e6)
	}
	t.Log("\n" + b.String())
}
