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
)

// The GC-pressure benchmarks measure what a garbage collection running
// concurrently in the host application costs the codec: write barriers in hot
// loops, mark assists and stop-the-world pauses. Each runs the same codec work
// under three regimes:
//
//   - idle: no collection is triggered (the codec allocates nothing);
//   - churn: a goroutine allocates pointerful garbage over a 64 MiB live
//     linked heap at GOGC=10, so collections run back to back and the
//     churner, not the codec, is charged the assists;
//   - forcegc: a goroutine calls runtime.GC in a loop.
//
// Compare ns/op across the regimes; gc-cycles/op confirms the regime and
// assist-ms/op and pause-ms/op show where a slowdown comes from. The decode
// benchmark uses GOAV1_GC_PRESSURE_IVF when set (a corpus clip) and a
// generated 640x360 stream otherwise; GOAV1_GC_PRESSURE_WORKERS sets its
// worker count. The encode benchmark encodes generated 640x360 frames.

type gcPressureNode struct {
	next    *gcPressureNode
	payload [6]*int
}

// startGCPressure starts the regime's background load and returns its stop
// function.
func startGCPressure(mode string) func() {
	var stop atomic.Bool
	var wg sync.WaitGroup
	restore := func() {}
	switch mode {
	case "churn":
		old := debug.SetGCPercent(10)
		restore = func() { debug.SetGCPercent(old) }
		wg.Add(1)
		go func() {
			defer wg.Done()
			const live = 64 << 20 / 64
			ring := make([]*gcPressureNode, live)
			v := new(int)
			for i := 0; !stop.Load(); i++ {
				n := &gcPressureNode{next: ring[(i+1)%live]}
				n.payload[0] = v
				ring[i%live] = n
			}
		}()
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
	}
}

var gcPressureMetricNames = []string{
	"/gc/cycles/total:gc-cycles",
	"/cpu/classes/gc/mark/assist:cpu-seconds",
	"/sched/pauses/total/gc:seconds",
}

type gcPressureSnapshot struct {
	cycles, assist, pause float64
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
	return snap
}

func runGCPressure(b *testing.B, op func()) {
	for _, mode := range []string{"idle", "churn", "forcegc"} {
		b.Run(mode, func(b *testing.B) {
			op() // warm
			stop := startGCPressure(mode)
			before := readGCPressure()
			b.ResetTimer()
			for range b.N {
				op()
			}
			b.StopTimer()
			after := readGCPressure()
			stop()
			n := float64(b.N)
			b.ReportMetric((after.cycles-before.cycles)/n, "gc-cycles/op")
			b.ReportMetric((after.assist-before.assist)*1e3/n, "assist-ms/op")
			b.ReportMetric((after.pause-before.pause)*1e3/n, "pause-ms/op")
		})
	}
}

func BenchmarkGCPressureDecode(b *testing.B) {
	var dec *Decoder
	var err error
	workers := 1
	if s := os.Getenv("GOAV1_GC_PRESSURE_WORKERS"); s != "" {
		if workers, err = strconv.Atoi(s); err != nil || workers < 1 {
			b.Fatalf("GOAV1_GC_PRESSURE_WORKERS=%q", s)
		}
	}
	if path := os.Getenv("GOAV1_GC_PRESSURE_IVF"); path != "" {
		ivf, rerr := os.ReadFile(path)
		if rerr != nil {
			b.Fatal(rerr)
		}
		dec, err = NewDecoderFromIVF(ivf, WithWorkers(workers))
	} else {
		dec, err = NewDecoder(gcInvisibleStream(b), WithWorkers(workers))
	}
	if err != nil {
		b.Fatal(err)
	}
	defer dec.Close()
	runGCPressure(b, func() {
		if err := dec.Reset(); err != nil {
			b.Fatal(err)
		}
		for {
			_, ok, err := dec.DecodeNext()
			if err != nil {
				b.Fatal(err)
			}
			if !ok {
				return
			}
		}
	})
}

func BenchmarkGCPressureEncode(b *testing.B) {
	threads := 1
	if os.Getenv("GOAV1_GC_PRESSURE_THREADS") == "0" {
		threads = 0
	}
	enc, err := NewVideoEncoder(VideoEncoderConfig{
		Width: gcInvisibleW, Height: gcInvisibleH, TargetBitrate: 1_500_000, Framerate: 30, MaxThreads: threads,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer enc.Close()
	frames := make([]I420Frame, 8)
	for i := range frames {
		frames[i] = gcInvisibleFrame()
		fillMovingI420(frames[i], i)
	}
	n := 0
	runGCPressure(b, func() {
		for range 16 {
			if _, err := enc.Encode(frames[n%len(frames)], n%64 == 0); err != nil {
				b.Fatal(err)
			}
			n++
		}
	})
}
