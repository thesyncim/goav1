// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package restoration

import (
	"math/rand"
	"testing"

	"simd/archsimd"
)

func TestSGRU8XLookupMatchesTable(t *testing.T) {
	for base := 0; base < 256; base += 16 {
		var z [16]uint8
		for i := range z {
			z[i] = uint8(base + i)
		}
		var got [16]uint8
		sgrU8X(archsimd.LoadUint8x16Array(&z)).StoreArray(&got)
		for i, v := range got {
			if want := SGRProjSgr - int(xByXPlus1[base+i]); int(v) != want {
				t.Fatalf("x(%d) = %d, want %d", base+i, v, want)
			}
		}
	}
}

// sgrU8ReferencePasses is the int32 path ApplySelfguidedRestorationU8 runs
// without the vector passes.
func sgrU8ReferencePasses(src []uint8, srcStride, srcOrigin, width, height, paramsIndex int, flt0, flt1 []int32) {
	dgdStride := width + 2*SGRProjBorderHorz
	dgd := make([]int32, dgdStride*(height+2*SGRProjBorderVert))
	dgdOrigin := SGRProjBorderVert*dgdStride + SGRProjBorderHorz
	for row := -SGRProjBorderVert; row < height+SGRProjBorderVert; row++ {
		for col := -SGRProjBorderHorz; col < width+SGRProjBorderHorz; col++ {
			dgd[dgdOrigin+row*dgdStride+col] = int32(src[srcOrigin+row*srcStride+col])
		}
	}
	bufStride := sgrBufferStride(width)
	bufLen := bufStride * (height + 2*SGRProjBorderVert)
	aBuf := make([]int32, bufLen)
	bBuf := make([]int32, bufLen)
	params := SGRParameterTable[paramsIndex]
	if params.Radius[0] > 0 {
		selfguidedFast(dgd, dgdOrigin, width, height, dgdStride, flt0, width, 8, paramsIndex, 0, aBuf, bBuf, bufStride)
	}
	if params.Radius[1] > 0 {
		selfguided(dgd, dgdOrigin, width, height, dgdStride, flt1, width, 8, paramsIndex, 1, aBuf, bBuf, bufStride)
	}
}

func TestSGRU8PassesMatchReference(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	sizes := [][2]int{{64, 64}, {1, 1}, {2, 3}, {7, 5}, {8, 8}, {15, 17}, {16, 1}, {33, 31}, {63, 64}, {64, 63}, {48, 64}, {64, 2}}
	for i := 0; i < 40; i++ {
		sizes = append(sizes, [2]int{1 + rng.Intn(64), 1 + rng.Intn(64)})
	}
	fills := []string{"random", "zero", "max", "checker", "ramp"}
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		for _, fill := range fills {
			stride := w + 2*SGRProjBorderHorz + rng.Intn(5)
			rows := h + 2*SGRProjBorderVert
			src := make([]uint8, stride*rows)
			for i := range src {
				switch fill {
				case "random":
					src[i] = uint8(rng.Intn(256))
				case "max":
					src[i] = 255
				case "checker":
					if (i/stride+i%stride)&1 != 0 {
						src[i] = 255
					}
				case "ramp":
					src[i] = uint8(i * 37)
				}
			}
			origin := SGRProjBorderVert*stride + SGRProjBorderHorz
			for p := range SGRProjParams {
				want0 := make([]int32, w*h)
				want1 := make([]int32, w*h)
				sgrU8ReferencePasses(src, stride, origin, w, h, p, want0, want1)
				got0 := make([]int32, w*h)
				got1 := make([]int32, w*h)
				if !sgrU8Passes(src, stride, origin, w, h, SGRParameterTable[p], got0, got1) {
					t.Fatalf("%dx%d params %d: vector passes declined", w, h, p)
				}
				for i := range want0 {
					if SGRParameterTable[p].Radius[0] != 0 && got0[i] != want0[i] {
						t.Fatalf("%dx%d %s params %d: flt0[%d,%d] = %d, want %d", w, h, fill, p, i/w, i%w, got0[i], want0[i])
					}
					if SGRParameterTable[p].Radius[1] != 0 && got1[i] != want1[i] {
						t.Fatalf("%dx%d %s params %d: flt1[%d,%d] = %d, want %d", w, h, fill, p, i/w, i%w, got1[i], want1[i])
					}
				}
			}
		}
	}
}

func TestSGRU8PassesZeroAlloc(t *testing.T) {
	w, h := 64, 64
	stride := w + 2*SGRProjBorderHorz
	src := make([]uint8, stride*(h+2*SGRProjBorderVert))
	for i := range src {
		src[i] = uint8(i * 13)
	}
	flt0 := make([]int32, w*h)
	flt1 := make([]int32, w*h)
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	if n := testing.AllocsPerRun(10, func() {
		sgrU8Passes(src, stride, origin, w, h, SGRParameterTable[0], flt0, flt1)
	}); n != 0 {
		t.Fatalf("allocs = %v", n)
	}
}

func BenchmarkSGRU8Passes(b *testing.B) {
	w, h := 64, 64
	stride := w + 2*SGRProjBorderHorz
	rng := rand.New(rand.NewSource(1))
	src := make([]uint8, stride*(h+2*SGRProjBorderVert))
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	dst := make([]uint8, w*h)
	n, err := SelfguidedScratchLen(w, h)
	if err != nil {
		b.Fatal(err)
	}
	scratch := make([]int32, n)
	b.Run("vector", func(b *testing.B) {
		for b.Loop() {
			if err := ApplySelfguidedRestorationU8(src, stride, origin, dst, w, w, h, 0, [2]int8{-32, 31}, scratch); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("int32", func(b *testing.B) {
		flt0 := make([]int32, w*h)
		flt1 := make([]int32, w*h)
		for b.Loop() {
			sgrU8ReferencePassesNoAlloc(src, stride, origin, w, h, 0, flt0, flt1, scratch)
		}
	})
}

// sgrU8ReferencePassesNoAlloc is sgrU8ReferencePasses on caller scratch (the
// dispatched int32 kernels), for the benchmark baseline.
func sgrU8ReferencePassesNoAlloc(src []uint8, srcStride, srcOrigin, width, height, paramsIndex int, flt0, flt1 []int32, scratch []int32) {
	dgdStride := width + 2*SGRProjBorderHorz
	dgdLen := dgdStride * (height + 2*SGRProjBorderVert)
	dgd := scratch[:dgdLen]
	dgdOrigin := SGRProjBorderVert*dgdStride + SGRProjBorderHorz
	for row := -SGRProjBorderVert; row < height+SGRProjBorderVert; row++ {
		s := src[srcOrigin+row*srcStride-SGRProjBorderHorz:]
		d := dgd[dgdOrigin+row*dgdStride-SGRProjBorderHorz:]
		for col := 0; col < width+2*SGRProjBorderHorz; col++ {
			d[col] = int32(s[col])
		}
	}
	bufStride := sgrBufferStride(width)
	bufLen := bufStride * (height + 2*SGRProjBorderVert)
	ab := scratch[dgdLen+2*width*height:]
	selfguidedFastImpl(dgd, dgdOrigin, width, height, dgdStride, flt0, width, 8, paramsIndex, 0, ab[:bufLen], ab[bufLen:2*bufLen], bufStride)
	selfguidedImpl(dgd, dgdOrigin, width, height, dgdStride, flt1, width, 8, paramsIndex, 1, ab[:bufLen], ab[bufLen:2*bufLen], bufStride)
}
