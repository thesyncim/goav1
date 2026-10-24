//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"bytes"
	"math/rand"
	"testing"
)

// boxColsRef is ScaleFilterCols_C with the ARM BLENDER, tap for tap.
func boxColsRef(dst, s []uint8, x, dx int) {
	for i := range dst {
		xi := x >> 16
		a, b := int(s[xi]), int(s[min(xi+1, len(s)-1)])
		dst[i] = uint8(a + (((x&0xffff)*(b-a) + 0x8000) >> 16))
		x += dx
	}
}

func boxColsRandomRow(rng *rand.Rand, n int) []uint8 {
	s := make([]uint8, n)
	switch rng.Intn(3) {
	case 0:
		rng.Read(s)
	case 1: // extremes: the largest |b-a|
		for i := range s {
			s[i] = []uint8{0, 255}[rng.Intn(2)]
		}
	default:
		for i := range s {
			s[i] = uint8(i * 7)
		}
	}
	return s
}

// The table kernel gives boxFilterCols' samples for any width, phase and
// step in [1, 2], including a last chunk pulled back over the previous one
// and a right tap clamped to the row.
func TestBoxColsPlanMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	ran := 0
	var cache boxColsCache
	for iter := 0; iter < 4000; iter++ {
		sw := 32 + rng.Intn(300)
		if iter%7 == 0 {
			sw = 32 + rng.Intn(8)
		}
		dx := 1<<16 + rng.Intn(1<<16+1)
		switch iter % 11 {
		case 0:
			dx = 1 << 16
		case 1:
			dx = 2 << 16
		case 2:
			dx = 0x18000
		}
		// The output count that fits: the last tap stays inside the row.
		x := 0
		if iter%3 == 0 {
			x = dx>>1 - 32768
		} else {
			x = rng.Intn(1 << 16)
		}
		dwMax := (((sw-1)<<16)-x)/dx + 1
		if dwMax < 16 {
			continue
		}
		dw := 16 + rng.Intn(dwMax-15)
		if iter%5 == 0 {
			dw = dwMax
		}
		plan := cache.plan(sw, dw, x, dx, 1<<10)
		if plan == nil {
			t.Fatalf("sw=%d dw=%d x=%#x dx=%#x: no plan", sw, dw, x, dx)
		}
		s := boxColsRandomRow(rng, sw)
		got, want := make([]uint8, dw+3), make([]uint8, dw+3)
		got[dw], got[dw+1], got[dw+2] = 0xa5, 0xa5, 0xa5
		want[dw], want[dw+1], want[dw+2] = 0xa5, 0xa5, 0xa5
		ran++
		plan.filter(got[:dw], s)
		boxColsRef(want[:dw], s, x, dx)
		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("sw=%d dw=%d x=%#x dx=%#x: sample %d = %d, want %d", sw, dw, x, dx, i, got[i], want[i])
				}
			}
		}
	}
	if ran < 3000 {
		t.Fatalf("only %d of 4000 random rows ran", ran)
	}
}

// Every (a, b) pair at the weights where the byte split of the blend is
// most likely to slip.
func TestBoxColsBlendByteSplit(t *testing.T) {
	fs := []int{0, 1, 2, 127, 128, 129, 254, 255, 256, 257, 0x7fff, 0x8000, 0x8001, 0xfeff, 0xff00, 0xff01, 0xfffe, 0xffff}
	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 64; i++ {
		fs = append(fs, rng.Intn(1<<16))
	}
	var cache boxColsCache
	const sw = 64
	for _, f := range fs {
		// A step of exactly 1.0 from phase f: output i blends s[i], s[i+1] at f.
		dw := 40
		plan := cache.plan(sw, dw, f, 1<<16, 1<<10)
		if plan == nil {
			t.Fatalf("f=%#x: no plan", f)
		}
		for a := 0; a < 256; a++ {
			for b := 0; b < 256; b += 1 {
				s := make([]uint8, sw)
				for i := range s {
					s[i] = uint8(a)
					if i&1 == 1 {
						s[i] = uint8(b)
					}
				}
				got, want := make([]uint8, dw), make([]uint8, dw)
				plan.filter(got, s)
				boxColsRef(want, s, f, 1<<16)
				if !bytes.Equal(got, want) {
					t.Fatalf("f=%#x a=%d b=%d: %v want %v", f, a, b, got, want)
				}
			}
		}
	}
}

// A full plane through the bilinear-down path, SIMD columns against the
// scalar row and column reference, on the layer geometries and odd sizes.
func TestScalePlaneBilinearDownColsMatchesScalar(t *testing.T) {
	var scratch boxScaleScratch
	steps := append([]struct{ sw, sh, dw, dh int }{
		{1280, 720, 853, 480}, {853, 480, 568, 320}, {640, 360, 427, 240}, {427, 240, 284, 160},
		{135, 135, 68, 68}, {270, 135, 135, 68}, {101, 77, 67, 51}, {64, 64, 40, 41}, {1920, 1080, 1280, 720},
	}, boxTestSteps...)
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 40; i++ {
		sw, sh := 32+rng.Intn(200), 8+rng.Intn(60)
		steps = append(steps, struct{ sw, sh, dw, dh int }{sw, sh, max(sw*(2+rng.Intn(3))/4, 1), max(sh*(2+rng.Intn(3))/4, 1)})
	}
	for _, st := range steps {
		const pad = 5
		src := boxTestPlane(st.sw, st.sh, st.sw+pad)
		got, want := make([]uint8, st.dw*st.dh), make([]uint8, st.dw*st.dh)
		scalePlaneBox8(got, st.dw, st.dw, st.dh, src, st.sw+pad, st.sw, st.sh, &scratch)
		boxScalePlaneRef(want, st.dw, st.dh, src, st.sw+pad, st.sw, st.sh)
		if !bytes.Equal(got, want) {
			t.Fatalf("%+v: plane differs from the scalar scaler", st)
		}
		if st.sw >= 32 && st.dw >= 16 && st.sh >= 8 && st == steps[0] && scratch.cols.plans[0].dw != st.dw {
			t.Fatalf("%+v: the column table was not used", st)
		}
	}
}

// boxScalePlaneRef is scalePlaneBox8 with scalar columns only.
func boxScalePlaneRef(dst []uint8, dw, dh int, src []uint8, srcStride, sw, sh int) {
	filter := boxScaleFilter(sw, sh, dw, dh)
	x, y, dx, dy := boxScaleSlope(sw, sh, dw, dh, filter)
	switch {
	case dw == sw && dh == sh:
		for j := 0; j < dh; j++ {
			copy(dst[j*dw:j*dw+dw], src[j*srcStride:j*srcStride+sw])
		}
		return
	case 2*dw == sw && 2*dh == sh, filter == boxFilterBox && dh*2 < sh, filter == boxFilterNone:
		var scratch boxScaleScratch
		scalePlaneBox8(dst, dw, dw, dh, src, srcStride, sw, sh, &scratch)
		return
	}
	row := make([]uint8, sw)
	maxY := (sh - 1) << 16
	y = min(y, maxY)
	for j := 0; j < dh; j++ {
		yi := y >> 16
		s := src[yi*srcStride : yi*srcStride+sw]
		if filter == boxFilterBilinear {
			if yf := (y >> 8) & 255; yf != 0 {
				boxInterpolateRow(row, s, src[(yi+1)*srcStride:(yi+1)*srcStride+sw], yf, 128)
				s = row
			}
		}
		boxColsRef(dst[j*dw:j*dw+dw], s, x, dx)
		y = min(y+dy, maxY)
	}
}

func BenchmarkBoxColsPlan8(b *testing.B) {
	for _, tc := range []struct {
		name   string
		sw, dw int
		x, dx  int
	}{
		{"2/3 672", 1008, 672, 0x4000, 0x18000},
		{"640->427", 640, 427, 0, 0},
		{"853->568", 853, 568, 0, 0},
	} {
		x, dx := tc.x, tc.dx
		if dx == 0 {
			dx = boxFixedDiv(tc.sw, tc.dw)
			x = dx>>1 - 32768
		}
		s := boxTestPlane(tc.sw, 1, tc.sw)
		d := make([]byte, tc.dw)
		var cache boxColsCache
		plan := cache.plan(tc.sw, tc.dw, x, dx, 1<<10)
		b.Run(tc.name, func(b *testing.B) {
			b.SetBytes(int64(tc.dw))
			for b.Loop() {
				plan.filter(d, s)
			}
		})
	}
}
