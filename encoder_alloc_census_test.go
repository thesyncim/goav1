package goav1

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"testing"
)

// TestEncoderAllocCensus reports heap allocations per encoder lifecycle
// event: construction, the first keyframe, the first inter frames, steady
// state, forced keyframes, scene cuts, threading changes and RTC
// reconfiguration. It is a measurement tool (GOAV1_ALLOC_CENSUS=1); input
// is the raw 1280x720 I420 file GOAV1_CENSUS_YUV, else a synthetic scene.
func TestEncoderAllocCensus(t *testing.T) {
	if os.Getenv("GOAV1_ALLOC_CENSUS") == "" {
		t.Skip("set GOAV1_ALLOC_CENSUS=1")
	}
	const w, h = 1280, 720
	frames := censusFrames(t, w, h, 120)
	scene := censusScene(w, h)

	var ms runtime.MemStats
	// GOAV1_CENSUS_MEMPROFILE=<prefix> (with -memprofilerate=1) writes the
	// cumulative allocs profile after every phase as <prefix>.NN; diff
	// consecutive snapshots (pprof -diff_base) to isolate one phase.
	profOut := os.Getenv("GOAV1_CENSUS_MEMPROFILE")
	phase := 0
	measure := func(name string, n int, f func()) {
		runtime.GC()
		defer func() {
			if profOut == "" {
				return
			}
			// The allocs profile publishes a cycle two GCs after it ends.
			runtime.GC()
			runtime.GC()
			runtime.GC()
			phase++
			if pf, err := os.Create(fmt.Sprintf("%s.%02d", profOut, phase)); err == nil {
				_ = pprof.Lookup("allocs").WriteTo(pf, 0)
				_ = pf.Close()
			}
			fmt.Printf("census phase %02d = %s\n", phase, name)
		}()
		runtime.ReadMemStats(&ms)
		m0, b0 := ms.Mallocs, ms.TotalAlloc
		f()
		runtime.ReadMemStats(&ms)
		dm, db := ms.Mallocs-m0, ms.TotalAlloc-b0
		fmt.Printf("census %-34s allocs=%8d bytes=%11d", name, dm, db)
		if n > 1 {
			fmt.Printf("  per-frame allocs=%.1f bytes=%.0f", float64(dm)/float64(n), float64(db)/float64(n))
		}
		fmt.Println()
	}

	for _, threads := range []int{1, 4} {
		p := fmt.Sprintf("video t%d ", threads)
		var enc *VideoEncoder
		measure(p+"NewVideoEncoder", 1, func() {
			var err error
			enc, err = NewVideoEncoder(VideoEncoderConfig{Width: w, Height: h, TargetBitrate: 1_000_000, Framerate: 30, MaxThreads: threads})
			if err != nil {
				t.Fatal(err)
			}
		})
		i := 0
		enc1 := func(force bool, f I420Frame) {
			if _, err := enc.Encode(f, force); err != nil {
				t.Fatal(err)
			}
			i++
		}
		measure(p+"first keyframe", 1, func() { enc1(false, frames[0]) })
		measure(p+"inter frames 1-15", 15, func() {
			for range 15 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"inter frames 16-63", 48, func() {
			for range 48 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"steady 64-111", 48, func() {
			for range 48 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"forced keyframe", 1, func() { enc1(true, frames[i%len(frames)]) })
		measure(p+"after forced key x16", 16, func() {
			for range 16 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"scene cut + 8", 9, func() {
			enc1(false, scene)
			for range 8 {
				enc1(false, frames[i%len(frames)])
			}
		})
		measure(p+"forced key on noise", 1, func() { enc1(true, scene) })
		measure(p+"second forced key on noise", 1, func() { enc1(true, scene) })
		measure(p+"second scene cut + 8", 9, func() {
			enc1(false, scene)
			for range 8 {
				enc1(false, frames[i%len(frames)])
			}
		})
		if threads == 1 {
			measure(p+"SetMaxThreads(4) + 16", 16, func() {
				enc.SetMaxThreads(4)
				for range 16 {
					enc1(false, frames[i%len(frames)])
				}
			})
		}
		measure(p+"Close", 1, func() { _ = enc.Close() })
	}

	var rtc *RTCEncoder
	measure("rtc NewRTCEncoder", 1, func() {
		var err error
		rtc, err = NewRTCEncoder(VideoEncoderConfig{Width: w, Height: h, TargetBitrate: 1_000_000, Framerate: 30, MaxThreads: 1})
		if err != nil {
			t.Fatal(err)
		}
	})
	j := 0
	rtc1 := func(force bool) {
		if _, err := rtc.Encode(frames[j%len(frames)], force); err != nil {
			t.Fatal(err)
		}
		j++
	}
	measure("rtc first 64 frames", 64, func() {
		for range 64 {
			rtc1(false)
		}
	})
	measure("rtc steady 48", 48, func() {
		for range 48 {
			rtc1(false)
		}
	})
	cfg := rtc.Config()
	measure("rtc bitrate change + 16", 16, func() {
		c := cfg
		c.TargetBitrateKbps /= 2
		c.MinBitrateKbps /= 2
		c.MaxBitrateKbps /= 2
		for k := range c.SpatialLayers {
			c.SpatialLayers[k].TargetBitrateKbps /= 2
			c.SpatialLayers[k].MinBitrateKbps /= 2
			c.SpatialLayers[k].MaxBitrateKbps /= 2
		}
		if err := rtc.SetConfig(c); err != nil {
			t.Fatalf("%v: %+v", err, c)
		}
		for range 16 {
			rtc1(false)
		}
	})
	measure("rtc forced key + 16", 17, func() {
		rtc1(true)
		for range 16 {
			rtc1(false)
		}
	})
	measure("rtc layers change (L1T2) + 16", 16, func() {
		c := rtc.Config()
		c.Scalability = EncoderScalabilityModeL1T2
		c.TemporalLayerCount = 2
		c.SpatialLayers[0].TemporalLayers = 2
		if err := rtc.SetConfig(c); err != nil {
			t.Logf("layers change: %v", err)
			return
		}
		for range 16 {
			rtc1(false)
		}
	})
	measure("rtc Close", 1, func() { _ = rtc.Close() })
}

func censusFrames(t *testing.T, w, h, n int) []I420Frame {
	size := w * h * 3 / 2
	var raw []byte
	if p := os.Getenv("GOAV1_CENSUS_YUV"); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		raw = b
	}
	out := make([]I420Frame, n)
	for i := range out {
		var buf []byte
		if len(raw) >= (i+1)*size {
			buf = raw[i*size : (i+1)*size]
		} else {
			buf = make([]byte, size)
			for y := range h {
				for x := range w {
					buf[y*w+x] = byte((x + 3*i + (y/8)*5) ^ (y >> 2))
				}
			}
			for k := w * h; k < size; k++ {
				buf[k] = 128
			}
		}
		out[i] = I420Frame{Y: buf[:w*h], U: buf[w*h : w*h+w*h/4], V: buf[w*h+w*h/4:], YStride: w, ChromaStride: w / 2, Width: w, Height: h}
	}
	return out
}

func censusScene(w, h int) I420Frame {
	size := w * h * 3 / 2
	buf := make([]byte, size)
	for i := range buf[:w*h] {
		buf[i] = byte(255 - (i*7)%251)
	}
	for k := w * h; k < size; k++ {
		buf[k] = 60
	}
	return I420Frame{Y: buf[:w*h], U: buf[w*h : w*h+w*h/4], V: buf[w*h+w*h/4:], YStride: w, ChromaStride: w / 2, Width: w, Height: h}
}
