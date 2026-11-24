package goav1

import (
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// The GC-pressure benchmarks measure what a garbage collection running
// concurrently in the host application costs the codec. Each runs the same
// codec work under every regime in gcPressureRegimes. codec-cpu/op is the CPU
// time of the codec's own thread (comparable across regimes of one run, in
// platform units); ns/op is wall time. gc-cycles/op confirms the regime and
// assist-ms/op, pause-ms/op and gc-cpu-ms/op show where a slowdown comes
// from. The decode benchmark uses GOAV1_GC_PRESSURE_IVF when set (a corpus
// clip) and a generated 640x360 stream otherwise; GOAV1_GC_PRESSURE_WORKERS
// sets its worker count. The encode benchmark encodes generated 640x360
// frames. Run with a pinned GOMAXPROCS (-cpu 4) on a machine with idle cores.

type gcPressureNode struct {
	next    *gcPressureNode
	payload [6]*int
}

// gcPressureRegimes are the background loads the benchmarks and the CPU-time
// gate run the codec under:
//
//   - idle: no collection is triggered (the codec allocates nothing);
//   - churn100, churn10: a goroutine allocates pointerful garbage over a
//     64 MiB live linked heap at GOGC=100 (a typical service) or GOGC=10
//     (collections back to back);
//   - memlimit: the same churn with GOGC=off and a GOMEMLIMIT 160 MiB above
//     the process's current footprint, so the limit paces collection;
//   - forcegc: a goroutine calls runtime.GC in a loop (pathological; the
//     stop-the-world phases dominate).
var gcPressureRegimes = []string{"idle", "churn100", "churn10", "memlimit", "forcegc"}

// gcChurnCPU accumulates the churn goroutine's thread CPU time, in threadCPU
// units, published every few thousand allocations.
var gcChurnCPU atomic.Uint64

// startGCPressure starts the regime's background load and returns its stop
// function.
func startGCPressure(mode string) func() {
	var stop atomic.Bool
	var wg sync.WaitGroup
	restore := func() {}
	churn := func() {
		// The ring is filled before churning starts, so the live heap is at
		// its full 64 MiB when the codec is measured.
		ready := make(chan struct{})
		wg.Add(1)
		go func() {
			defer wg.Done()
			// The churner owns its OS thread and publishes that thread's CPU
			// time, so multi-worker measurements can take it out of the
			// process total.
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			c0, _ := threadCPU()
			const live = 64 << 20 / 64
			ring := make([]*gcPressureNode, live)
			v := new(int)
			for i := range ring {
				ring[i] = &gcPressureNode{}
			}
			close(ready)
			publish := func() {
				c, _ := threadCPU()
				gcChurnCPU.Add(c - c0)
				c0 = c
			}
			for i := 0; !stop.Load(); i++ {
				n := &gcPressureNode{next: ring[(i+1)%live]}
				n.payload[0] = v
				ring[i%live] = n
				if i&(1<<12-1) == 0 {
					publish()
				}
			}
			publish()
		}()
		<-ready
	}
	switch mode {
	case "churn100", "churn10":
		pct := 100
		if mode == "churn10" {
			pct = 10
		}
		old := debug.SetGCPercent(pct)
		restore = func() { debug.SetGCPercent(old) }
		churn()
	case "memlimit":
		// The limit is set from the footprint with the ring full, 32 MiB
		// above it, so the limit (not GOGC) paces collection.
		oldPct := debug.SetGCPercent(-1)
		churn()
		runtime.GC()
		s := []metrics.Sample{{Name: "/memory/classes/total:bytes"}, {Name: "/memory/classes/heap/released:bytes"}}
		metrics.Read(s)
		used := int64(s[0].Value.Uint64() - s[1].Value.Uint64())
		oldLimit := debug.SetMemoryLimit(used + 32<<20)
		restore = func() {
			debug.SetMemoryLimit(oldLimit)
			debug.SetGCPercent(oldPct)
		}
	case "forcegc":
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stop.Load() {
				runtime.GC()
			}
		}()
	}
	return func() {
		stop.Store(true)
		wg.Wait()
		restore()
		runtime.GC()
	}
}

var gcPressureMetricNames = []string{
	"/gc/cycles/total:gc-cycles",
	"/cpu/classes/gc/mark/assist:cpu-seconds",
	"/sched/pauses/total/gc:seconds",
	"/cpu/classes/gc/total:cpu-seconds",
	"/gc/scan/stack:bytes",
}

type gcPressureSnapshot struct {
	cycles, assist, pause, gcCPU, stackScan float64
}

func readGCPressure() gcPressureSnapshot {
	s := make([]metrics.Sample, len(gcPressureMetricNames))
	for i, name := range gcPressureMetricNames {
		s[i].Name = name
	}
	metrics.Read(s)
	var snap gcPressureSnapshot
	snap.cycles = float64(s[0].Value.Uint64())
	snap.assist = s[1].Value.Float64()
	h := s[2].Value.Float64Histogram()
	for i, c := range h.Counts {
		lo, hi := h.Buckets[i], h.Buckets[i+1]
		if math.IsInf(hi, 1) {
			hi = lo
		}
		if math.IsInf(lo, -1) {
			lo = 0
		}
		snap.pause += float64(c) * (lo + hi) / 2
	}
	snap.gcCPU = s[3].Value.Float64()
	snap.stackScan = float64(s[4].Value.Uint64())
	return snap
}

// gcPressureResult is one regime's per-op cost.
type gcPressureResult struct {
	wallNs, codecCPU                   float64 // codecCPU in threadCPU units; 0 if unavailable
	cycles, assistMs, pauseMs, gcCPUMs float64
	stackScanKiB                       float64
}

// measureGCPressure runs op n times under mode with the calling goroutine
// locked to its OS thread, so codecCPU counts only that thread's CPU time:
// with one decode worker or one encoder thread that is all of the codec's
// work and none of the churner's or the mark workers'. wallNs also counts
// time the codec waited (stop-the-world, stack-scan preemption, a busy
// machine).
func measureGCPressure(mode string, n int, op func(), timer func(bool)) gcPressureResult {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	op() // warm
	stop := startGCPressure(mode)
	before := readGCPressure()
	if timer != nil {
		timer(true)
	}
	cpu0, cpuOK := threadCPU()
	t0 := time.Now()
	for range n {
		op()
	}
	t1 := time.Now()
	cpu1, _ := threadCPU()
	if timer != nil {
		timer(false)
	}
	after := readGCPressure()
	stop()
	fn := float64(n)
	r := gcPressureResult{
		wallNs:       float64(t1.Sub(t0)) / fn,
		cycles:       (after.cycles - before.cycles) / fn,
		assistMs:     (after.assist - before.assist) * 1e3 / fn,
		pauseMs:      (after.pause - before.pause) * 1e3 / fn,
		gcCPUMs:      (after.gcCPU - before.gcCPU) * 1e3 / fn,
		stackScanKiB: after.stackScan / 1024,
	}
	if cpuOK {
		r.codecCPU = float64(cpu1-cpu0) / fn
	}
	return r
}

func runGCPressure(b *testing.B, op func()) {
	for _, mode := range gcPressureRegimes {
		b.Run(mode, func(b *testing.B) {
			r := measureGCPressure(mode, b.N, op, func(start bool) {
				if start {
					b.ResetTimer()
				} else {
					b.StopTimer()
				}
			})
			if r.codecCPU > 0 {
				b.ReportMetric(r.codecCPU, "codec-cpu/op")
			}
			b.ReportMetric(r.cycles, "gc-cycles/op")
			b.ReportMetric(r.assistMs, "assist-ms/op")
			b.ReportMetric(r.pauseMs, "pause-ms/op")
			b.ReportMetric(r.gcCPUMs, "gc-cpu-ms/op")
			b.ReportMetric(r.stackScanKiB, "stack-scan-KiB")
		})
	}
}

func gcPressureDecodeOp(tb testing.TB, workers int) (func(), func()) {
	var dec *Decoder
	var err error
	if path := os.Getenv("GOAV1_GC_PRESSURE_IVF"); path != "" {
		ivf, rerr := os.ReadFile(path)
		if rerr != nil {
			tb.Fatal(rerr)
		}
		dec, err = NewDecoderFromIVF(ivf, WithWorkers(workers))
	} else {
		dec, err = NewDecoder(gcInvisibleStream(tb), WithWorkers(workers))
	}
	if err != nil {
		tb.Fatal(err)
	}
	return func() {
		if err := dec.Reset(); err != nil {
			tb.Fatal(err)
		}
		for {
			_, ok, err := dec.DecodeNext()
			if err != nil {
				tb.Fatal(err)
			}
			if !ok {
				return
			}
		}
	}, dec.Close
}

func gcPressureEncodeOp(tb testing.TB, threads int) (func(), func()) {
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: gcInvisibleW, Height: gcInvisibleH, TargetBitrate: 1_500_000, Framerate: 30, MaxThreads: threads,
	})
	if err != nil {
		tb.Fatal(err)
	}
	frames := make([]I420Frame, 8)
	for i := range frames {
		frames[i] = gcInvisibleFrame()
		fillMovingI420(frames[i], i)
	}
	n := 0
	return func() {
		// Every op is the same work: a keyframe, then 15 inter frames.
		for i := range 16 {
			if _, err := enc.Encode(frames[n%len(frames)], i == 0); err != nil {
				tb.Fatal(err)
			}
			n++
		}
	}, func() { _ = enc.Close() }
}

func BenchmarkGCPressureDecode(b *testing.B) {
	workers := 1
	if s := os.Getenv("GOAV1_GC_PRESSURE_WORKERS"); s != "" {
		var err error
		if workers, err = strconv.Atoi(s); err != nil || workers < 1 {
			b.Fatalf("GOAV1_GC_PRESSURE_WORKERS=%q", s)
		}
	}
	op, closeFn := gcPressureDecodeOp(b, workers)
	defer closeFn()
	runGCPressure(b, op)
}

func BenchmarkGCPressureEncode(b *testing.B) {
	threads := 1
	if os.Getenv("GOAV1_GC_PRESSURE_THREADS") == "0" {
		threads = 0
	}
	op, closeFn := gcPressureEncodeOp(b, threads)
	defer closeFn()
	runGCPressure(b, op)
}
