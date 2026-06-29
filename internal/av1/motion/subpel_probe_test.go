package motion

import (
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
