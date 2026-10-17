package motion

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

func TestLumaSubpelProberMatchesRegularPredictor(t *testing.T) {
	const (
		width  = 96
		height = 96
		stride = 112
		ox     = 32
		oy     = 32
	)
	ref := make([]byte, stride*height)
	for i := range ref {
		ref[i] = byte((i*37 + i/stride*11) & 0xff)
	}
	refPlane := frame.Plane{Pix: ref, Stride: stride, Width: width, Height: height}
	deltas := []Vector{
		{Row: 0, Col: 0},
		{Row: 0, Col: 4},
		{Row: 4, Col: 0},
		{Row: 4, Col: 4},
		{Row: -6, Col: 2},
		{Row: 8, Col: -8},
		{Row: 10, Col: -12},
		{Row: -12, Col: 12},
	}

	for _, n := range []int{4, 8, 16, 32} {
		var prober LumaSubpelProber
		prober.Init(refPlane, ox, oy, n)
		for _, delta := range deltas {
			got := make([]byte, n*n)
			if ok := prober.Predict(got, delta); !ok {
				t.Fatalf("n=%d delta=%+v: prober rejected interior probe", n, delta)
			}
			want := make([]byte, n*n)
			posX := int64(ox)*16 + int64(delta.Col)*2
			posY := int64(oy)*16 + int64(delta.Row)*2
			refX := int(posX >> 4)
			refY := int(posY >> 4)
			subX := int(posX & 15)
			subY := int(posY & 15)
			if err := PredictInterPlaneBlockFromOriginWithFilter(frame.Plane{
				Pix: want, Stride: n, Width: n, Height: n,
			}, refPlane, 1, 0, 0, refX, refY, n, n, subX, subY, RegularFilters); err != nil {
				t.Fatalf("n=%d delta=%+v: reference predictor: %v", n, delta, err)
			}
			if string(got) != string(want) {
				t.Fatalf("n=%d delta=%+v: prober output diverged", n, delta)
			}
			if n >= 8 {
				gotFast := make([]byte, n*n)
				if ok := predictSubpelForSize(&prober, gotFast, n, delta); !ok {
					t.Fatalf("n=%d delta=%+v: sized prober rejected interior probe", n, delta)
				}
				if string(gotFast) != string(want) {
					t.Fatalf("n=%d delta=%+v: sized prober output diverged", n, delta)
				}
			}
		}
	}
}

func TestLumaSubpelProberClampedScratchMatchesScalarOracle(t *testing.T) {
	const (
		width  = 148
		height = 148
		stride = 160
		guard  = 32
	)
	refStorage := bytes.Repeat([]byte{0x49}, guard+stride*height+guard)
	ref := refStorage[guard : guard+stride*height]
	for y := 0; y < height; y++ {
		row := ref[y*stride : (y+1)*stride]
		for x := 0; x < width; x++ {
			row[x] = byte((x*37 + y*53 + x*y*3) & 0xff)
		}
		// Deliberately make row padding unlike the right-edge pixel. A SIMD
		// load that leaks past the visible plane must not affect clamped output.
		for x := width; x < stride; x++ {
			row[x] = byte(0xd0 + y%31)
		}
	}
	refPlane := frame.Plane{Pix: ref, Stride: stride, Width: width, Height: height}
	deltas := []Vector{
		{Row: 0, Col: 1},
		{Row: 0, Col: -1},
		{Row: 0, Col: 3},
		{Row: 0, Col: -3},
		{Row: 0, Col: 2},
		{Row: 0, Col: -2},
		{Row: 0, Col: 6},
		{Row: 0, Col: -6},
		{Row: 1, Col: 0},
		{Row: -1, Col: 0},
		{Row: 3, Col: 0},
		{Row: -3, Col: 0},
		{Row: 2, Col: 0},
		{Row: -2, Col: 0},
		{Row: 6, Col: 0},
		{Row: -6, Col: 0},
		{Row: 0, Col: 4},
		{Row: 0, Col: -12},
		{Row: 4, Col: 0},
		{Row: -12, Col: 0},
		{Row: 4, Col: 4},
		{Row: 2, Col: -2},
		{Row: 6, Col: -6},
		{Row: -12, Col: 12},
		{Row: 12, Col: -12},
		{Row: 0, Col: 0},
	}

	for _, n := range []int{3, 4, 5, 8, 12, 16, 24, 32, 64, 128} {
		midX, midY := (width-n)/2, (height-n)/2
		origins := [][2]int{
			{0, 0}, {width - n, 0}, {0, height - n}, {width - n, height - n},
			{0, midY}, {width - n, midY}, {midX, 0}, {midX, height - n},
			{-2, midY}, {width - n + 2, midY}, {midX, -2}, {midX, height - n + 2},
		}
		probeDeltas := deltas
		if n >= 64 {
			// Exercise every signed single-axis phase on the largest blocks
			// too, plus a diagonal path to keep mixed-axis coverage.
			origins = origins[:8] // corners and side centers; exact-window tests below add off-frame origins
			probeDeltas = []Vector{
				{Row: 0, Col: 1}, {Row: 0, Col: -1}, {Row: 0, Col: 3}, {Row: 0, Col: -3},
				{Row: 0, Col: 2}, {Row: 0, Col: -2}, {Row: 0, Col: 6}, {Row: 0, Col: -6},
				{Row: 1, Col: 0}, {Row: -1, Col: 0}, {Row: 3, Col: 0}, {Row: -3, Col: 0},
				{Row: 2, Col: 0}, {Row: -2, Col: 0}, {Row: 6, Col: 0}, {Row: -6, Col: 0},
				{Row: 0, Col: 12}, {Row: 0, Col: -12}, {Row: 12, Col: 0}, {Row: -12, Col: 0},
				{Row: -12, Col: 12},
			}
		}
		for _, origin := range origins {
			var prober LumaSubpelProber
			prober.Init(refPlane, origin[0], origin[1], n)
			if prober.valid {
				t.Fatalf("n=%d origin=%v: expected clamped edge probe", n, origin)
			}
			for _, delta := range probeDeltas {
				storage := bytes.Repeat([]byte{0xa7}, guard+n*n+guard)
				dst := storage[guard : guard+n*n+guard]
				if !predictSubpelForSize(&prober, dst, n, delta) {
					t.Fatalf("n=%d origin=%v delta=%+v: prober rejected edge probe", n, origin, delta)
				}

				want := make([]byte, n*n)
				predictSubpelScalarClampedForTest(t, want, refPlane, origin[0], origin[1], n, delta)
				if !bytes.Equal(dst[:n*n], want) {
					t.Fatalf("n=%d origin=%v delta=%+v: prober output diverged from scalar clamped oracle", n, origin, delta)
				}
				if !bytes.Equal(storage[:guard], bytes.Repeat([]byte{0xa7}, guard)) ||
					!bytes.Equal(dst[n*n:], bytes.Repeat([]byte{0xa7}, guard)) {
					t.Fatalf("n=%d origin=%v delta=%+v: output escaped its exact block window", n, origin, delta)
				}
			}
		}
	}
}

func TestLumaSubpelProberClampedScratchMinimalWindows(t *testing.T) {
	const (
		width  = 148
		height = 148
		stride = 160
	)
	// The source allocation ends at the final visible pixel, with no row
	// padding or guard bytes after the last row. The destination has exactly
	// n*n bytes so checkptr and ordinary bounds checks see the real contract.
	for _, n := range []int{8, 64, 128} {
		ref := make([]byte, (height-1)*stride+width)
		for y := 0; y < height; y++ {
			rowEnd := (y + 1) * stride
			if y == height-1 {
				rowEnd = len(ref)
			}
			row := ref[y*stride : rowEnd]
			for x := 0; x < width; x++ {
				row[x] = byte((x*19 + y*43 + x*y*5) & 0xff)
			}
			for x := width; x < len(row); x++ {
				row[x] = byte(0xe1 + y%17)
			}
		}
		refPlane := frame.Plane{Pix: ref, Stride: stride, Width: width, Height: height}
		midX, midY := (width-n)/2, (height-n)/2
		probes := []struct {
			origin [2]int
			delta  Vector
		}{
			{[2]int{0, 0}, Vector{Row: 0, Col: -12}},
			{[2]int{0, 0}, Vector{Row: -12, Col: 0}},
			{[2]int{0, 0}, Vector{Row: -12, Col: -12}},
			{[2]int{-2, midY}, Vector{Row: 0, Col: 2}},
			{[2]int{midX, -2}, Vector{Row: 2, Col: 0}},
			{[2]int{width - n, height - n}, Vector{Row: 0, Col: 12}},
			{[2]int{width - n, height - n}, Vector{Row: 12, Col: 0}},
			{[2]int{width - n, height - n}, Vector{Row: 12, Col: 12}},
		}
		for _, probe := range probes {
			var prober LumaSubpelProber
			prober.Init(refPlane, probe.origin[0], probe.origin[1], n)
			dst := make([]byte, n*n)
			if !predictSubpelForSize(&prober, dst, n, probe.delta) {
				t.Fatalf("n=%d origin=%v delta=%+v: prober rejected minimal edge window", n, probe.origin, probe.delta)
			}
			want := make([]byte, n*n)
			predictSubpelScalarClampedForTest(t, want, refPlane, probe.origin[0], probe.origin[1], n, probe.delta)
			if !bytes.Equal(dst, want) {
				t.Fatalf("n=%d origin=%v delta=%+v: output diverged on minimal source/destination windows", n, probe.origin, probe.delta)
			}
		}
	}
}

func TestLumaSubpelProberClampedScratchReuseAcrossAxes(t *testing.T) {
	const (
		width  = 96
		height = 96
		stride = 112
		n      = 64
		guard  = 9
	)
	refStorage := bytes.Repeat([]byte{0x27}, guard+stride*height+guard)
	ref := refStorage[guard : guard+stride*height]
	for y := 0; y < height; y++ {
		row := ref[y*stride : (y+1)*stride]
		for x := 0; x < width; x++ {
			row[x] = byte((x*29 + y*47 + x*y) & 0xff)
		}
		for x := width; x < stride; x++ {
			row[x] = byte(0x80 + y%63)
		}
	}
	refPlane := frame.Plane{Pix: ref, Stride: stride, Width: width, Height: height}
	var prober LumaSubpelProber
	prober.Init(refPlane, 0, 0, n)
	deltas := []Vector{
		{Row: 0, Col: -12}, // X-only, crossing the left edge
		{Row: -12, Col: 0}, // Y-only, crossing the top edge
		{Row: -12, Col: -12},
		{Row: 0, Col: 12},
		{Row: 12, Col: 0},
		{Row: 12, Col: 12},
	}
	for i, delta := range deltas {
		storage := bytes.Repeat([]byte{0x5c}, guard+n*n+guard)
		dst := storage[guard : guard+n*n+guard]
		if !prober.Predict64x64(dst, delta) {
			t.Fatalf("step %d delta=%+v: prober rejected edge probe", i, delta)
		}
		want := make([]byte, n*n)
		predictSubpelScalarClampedForTest(t, want, refPlane, 0, 0, n, delta)
		if !bytes.Equal(dst[:n*n], want) {
			t.Fatalf("step %d delta=%+v: reused scratch changed output", i, delta)
		}
		if !bytes.Equal(storage[:guard], bytes.Repeat([]byte{0x5c}, guard)) ||
			!bytes.Equal(dst[n*n:], bytes.Repeat([]byte{0x5c}, guard)) {
			t.Fatalf("step %d delta=%+v: output escaped its exact block window", i, delta)
		}
	}
}

// predictSubpelScalarClampedForTest is an independent oracle for prober edge
// probes. It calls the pure-Go clamped kernels directly, bypassing the shared
// architecture dispatch used by both the prober and general predictor.
func predictSubpelScalarClampedForTest(t *testing.T, dst []byte, ref frame.Plane, ox, oy, n int, delta Vector) {
	t.Helper()
	posX := int64(ox)*16 + int64(delta.Col)*2
	posY := int64(oy)*16 + int64(delta.Row)*2
	refX, refY := int(posX>>4), int(posY>>4)
	subX, subY := int(posX&15), int(posY&15)
	dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
	if subX != 0 && subY != 0 {
		convolve2D8ClampedPureGo(dstPlane, ref, 0, 0, refX, refY, n, n, regularSubpelKernel(n, subX), regularSubpelKernel(n, subY))
	} else if subX != 0 {
		convolveX8ClampedPureGo(dstPlane, ref, 0, 0, refX, refY, n, n, regularSubpelKernel(n, subX))
	} else if subY != 0 {
		convolveY8ClampedPureGo(dstPlane, ref, 0, 0, refX, refY, n, n, regularSubpelKernel(n, subY))
	} else {
		if err := copyPlaneBlockClamped(dstPlane, ref, 1, 0, 0, refX, refY, n, n); err != nil {
			t.Fatalf("scalar clamped copy oracle: %v", err)
		}
	}
}

func TestLumaSubpelProberMatchesRegularPredictorAtEdges(t *testing.T) {
	const (
		width  = 96
		height = 96
		stride = 112
	)
	ref := make([]byte, stride*height)
	for i := range ref {
		ref[i] = byte((i*37 + i/stride*11) & 0xff)
	}
	refPlane := frame.Plane{Pix: ref, Stride: stride, Width: width, Height: height}
	deltas := []Vector{
		{Row: 0, Col: 0},
		{Row: 0, Col: 4},
		{Row: 4, Col: 0},
		{Row: 4, Col: 4},
		{Row: -6, Col: 2},
		{Row: 8, Col: -8},
		{Row: 10, Col: -12},
		{Row: -12, Col: 12},
	}

	for _, n := range []int{4, 8, 16, 32} {
		origins := [][2]int{
			{1, 32},
			{width - n - 2, 32},
			{32, 1},
			{32, height - n - 2},
			{1, 1},
			{width - n - 2, height - n - 2},
		}
		for _, origin := range origins {
			var prober LumaSubpelProber
			ox, oy := origin[0], origin[1]
			prober.Init(refPlane, ox, oy, n)
			for _, delta := range deltas {
				got := make([]byte, n*n)
				if ok := prober.Predict(got, delta); !ok {
					t.Fatalf("n=%d origin=%v delta=%+v: prober rejected edge probe", n, origin, delta)
				}
				want := make([]byte, n*n)
				posX := int64(ox)*16 + int64(delta.Col)*2
				posY := int64(oy)*16 + int64(delta.Row)*2
				refX := int(posX >> 4)
				refY := int(posY >> 4)
				subX := int(posX & 15)
				subY := int(posY & 15)
				if err := PredictInterPlaneBlockFromOriginWithFilter(frame.Plane{
					Pix: want, Stride: n, Width: n, Height: n,
				}, refPlane, 1, 0, 0, refX, refY, n, n, subX, subY, RegularFilters); err != nil {
					t.Fatalf("n=%d origin=%v delta=%+v: reference predictor: %v", n, origin, delta, err)
				}
				if string(got) != string(want) {
					t.Fatalf("n=%d origin=%v delta=%+v: prober output diverged", n, origin, delta)
				}
				if n >= 8 {
					gotFast := make([]byte, n*n)
					if ok := predictSubpelForSize(&prober, gotFast, n, delta); !ok {
						t.Fatalf("n=%d origin=%v delta=%+v: sized prober rejected edge probe", n, origin, delta)
					}
					if string(gotFast) != string(want) {
						t.Fatalf("n=%d origin=%v delta=%+v: sized prober output diverged", n, origin, delta)
					}
				}
			}
		}
	}
}

func predictSubpelForSize(p *LumaSubpelProber, dst []byte, n int, delta Vector) bool {
	switch n {
	case 8:
		return p.Predict8x8(dst, delta)
	case 16:
		return p.Predict16x16(dst, delta)
	case 32:
		return p.Predict32x32(dst, delta)
	case 64:
		return p.Predict64x64(dst, delta)
	}
	return p.Predict(dst, delta)
}

var subpelProbeSink byte

func BenchmarkLumaSubpelProberPredict(b *testing.B) {
	const (
		width  = 192
		height = 192
		stride = 208
		ox     = 64
		oy     = 64
	)
	ref := make([]byte, stride*height)
	for i := range ref {
		ref[i] = byte((i*37 + i/stride*11) & 0xff)
	}
	refPlane := frame.Plane{Pix: ref, Stride: stride, Width: width, Height: height}
	deltas := [...]Vector{
		{Row: 0, Col: 4},
		{Row: 4, Col: 0},
		{Row: 4, Col: 4},
		{Row: -6, Col: 2},
	}

	for _, n := range []int{8, 16, 32} {
		name := "8x8"
		if n == 16 {
			name = "16x16"
		} else if n == 32 {
			name = "32x32"
		}
		b.Run(name, func(b *testing.B) {
			var prober LumaSubpelProber
			prober.Init(refPlane, ox, oy, n)
			dst := make([]byte, n*n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if !predictSubpelForSize(&prober, dst, n, deltas[i&3]) {
					b.Fatal("interior probe rejected")
				}
			}
			subpelProbeSink = dst[(b.N-1)&(len(dst)-1)]
		})
	}
}

func BenchmarkLumaSubpelProberPredictEdgeFallback(b *testing.B) {
	const (
		width  = 192
		height = 192
		stride = 208
		ox     = 1
		oy     = 64
	)
	ref := make([]byte, stride*height)
	for i := range ref {
		ref[i] = byte((i*37 + i/stride*11) & 0xff)
	}
	refPlane := frame.Plane{Pix: ref, Stride: stride, Width: width, Height: height}
	deltas := [...]Vector{
		{Row: 0, Col: 4},
		{Row: 4, Col: 0},
		{Row: 4, Col: 4},
		{Row: -6, Col: 2},
	}

	for _, n := range []int{8, 16, 32} {
		name := "8x8"
		if n == 16 {
			name = "16x16"
		} else if n == 32 {
			name = "32x32"
		}
		b.Run(name, func(b *testing.B) {
			var prober LumaSubpelProber
			prober.Init(refPlane, ox, oy, n)
			dst := make([]byte, n*n)
			dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				delta := deltas[i&3]
				if !predictSubpelForSize(&prober, dst, n, delta) {
					posX := int64(ox)*16 + int64(delta.Col)*2
					posY := int64(oy)*16 + int64(delta.Row)*2
					refX := int(posX >> 4)
					refY := int(posY >> 4)
					subX := int(posX & 15)
					subY := int(posY & 15)
					if err := PredictInterPlaneBlockFromOriginWithFilter(dstPlane, refPlane, 1, 0, 0, refX, refY, n, n, subX, subY, RegularFilters); err != nil {
						b.Fatal(err)
					}
				}
			}
			subpelProbeSink = dst[(b.N-1)&(len(dst)-1)]
		})
	}
}

func BenchmarkLumaSubpelProberPredictWideDeltaFallback(b *testing.B) {
	const (
		width  = 192
		height = 192
		stride = 208
		ox     = 64
		oy     = 64
	)
	ref := make([]byte, stride*height)
	for i := range ref {
		ref[i] = byte((i*37 + i/stride*11) & 0xff)
	}
	refPlane := frame.Plane{Pix: ref, Stride: stride, Width: width, Height: height}
	deltas := [...]Vector{
		{Row: 12, Col: 12},
		{Row: -12, Col: 10},
		{Row: 10, Col: -12},
		{Row: -10, Col: -10},
	}

	for _, n := range []int{8, 16, 32} {
		name := "8x8"
		if n == 16 {
			name = "16x16"
		} else if n == 32 {
			name = "32x32"
		}
		b.Run(name, func(b *testing.B) {
			var prober LumaSubpelProber
			prober.Init(refPlane, ox, oy, n)
			dst := make([]byte, n*n)
			dstPlane := frame.Plane{Pix: dst, Stride: n, Width: n, Height: n}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				delta := deltas[i&3]
				if !predictSubpelForSize(&prober, dst, n, delta) {
					posX := int64(ox)*16 + int64(delta.Col)*2
					posY := int64(oy)*16 + int64(delta.Row)*2
					refX := int(posX >> 4)
					refY := int(posY >> 4)
					subX := int(posX & 15)
					subY := int(posY & 15)
					if err := PredictInterPlaneBlockFromOriginWithFilter(dstPlane, refPlane, 1, 0, 0, refX, refY, n, n, subX, subY, RegularFilters); err != nil {
						b.Fatal(err)
					}
				}
			}
			subpelProbeSink = dst[(b.N-1)&(len(dst)-1)]
		})
	}
}
