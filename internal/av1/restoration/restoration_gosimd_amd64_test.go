//go:build goexperiment.simd && amd64 && !purego

package restoration

import (
	"simd/archsimd"
	"slices"
	"testing"
)

// TestSGRRollingIntermediateAVX2MatchesFullOracle checks the AVX2 A/B producer
// against the frozen full box-sum oracle for every parameter set, radius,
// pass and bit depth, on random and on all-maximum samples, and that it
// writes nothing past the coefficient buffers.
func TestSGRRollingIntermediateAVX2MatchesFullOracle(t *testing.T) {
	if !sgrIntermediateAVX2 {
		t.Skip("no AVX2")
	}
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xa52)
	for _, sz := range [][2]int{{1, 1}, {2, 3}, {6, 6}, {7, 9}, {8, 8}, {14, 5}, {16, 8}, {31, 33}, {32, 64}, {63, 64}, {64, 63}, {64, 64}} {
		w, h := sz[0], sz[1]
		stride := w + 9
		origin := 3*stride + 4
		bs := sgrBufferStride(w)
		bl := bs * (h + 6)
		dgd := make([]int32, stride*(h+6))
		for _, bd := range []int{8, 10, 12} {
			for fill := 0; fill < 2; fill++ {
				for i := range dgd {
					if fill == 0 {
						dgd[i] = int32(rnd.pseudoUniform(1 << bd))
					} else {
						dgd[i] = int32(1<<bd - 1)
					}
				}
				for pi, p := range SGRParameterTable {
					for ri, r := range p.Radius {
						if r == 0 {
							continue
						}
						for pass := 0; pass < 2; pass++ {
							wantA, wantB := make([]int32, bl), make([]int32, bl)
							gotA, gotB := make([]int32, bl+7), make([]int32, bl+7)
							for i := range gotA {
								gotA[i] = int32(uint32(i)*2654435761 + 0xfedc)
								gotB[i] = ^gotA[i]
							}
							tailA, tailB := slices.Clone(gotA[bl:]), slices.Clone(gotB[bl:])
							frozenSGRIntermediate(dgd, origin, w, h, stride, bd, pi, ri, pass, wantA, wantB, bs)
							calculateIntermediateRollingAVX2(dgd, origin, w, h, stride, bd, p, ri, pass, gotA[:bl], gotB[:bl], bs)
							step := 1
							if pass != 0 {
								step = 2
							}
							for y := -1; y < h+1; y += step {
								for x := -1; x < w+1; x++ {
									k := (y+3)*bs + x + 3
									if gotA[k] != wantA[k] || gotB[k] != wantB[k] {
										t.Fatalf("%dx%d bd=%d fill=%d params=%d radius=%d pass=%d x=%d y=%d A=%d/%d B=%d/%d", w, h, bd, fill, pi, ri, pass, x, y, gotA[k], wantA[k], gotB[k], wantB[k])
									}
								}
							}
							if !slices.Equal(gotA[bl:], tailA) || !slices.Equal(gotB[bl:], tailB) {
								t.Fatal("scratch tail written")
							}
						}
					}
				}
			}
		}
	}
}

// BenchmarkAMD64SGRIntermediate compares the scalar rolling A/B producer
// with the AVX2 one on a 64x64 processing unit.
func BenchmarkAMD64SGRIntermediate(b *testing.B) {
	const w, h = 64, 64
	stride := w + 9
	origin := 3*stride + 4
	bs := sgrBufferStride(w)
	bl := bs * (h + 6)
	dgd := make([]int32, stride*(h+6))
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xb11)
	for i := range dgd {
		dgd[i] = int32(rnd.pseudoUniform(256))
	}
	aBuf, bBuf := make([]int32, bl), make([]int32, bl)
	p := SGRParameterTable[0]
	for _, c := range []struct {
		name     string
		ri, pass int
	}{{"r2/pass1", 0, 1}, {"r1/pass0", 1, 0}} {
		b.Run("shape="+c.name+"/impl=generic", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				calculateIntermediateRolling(dgd, origin, w, h, stride, 8, p, c.ri, c.pass, aBuf, bBuf, bs)
			}
		})
		b.Run("shape="+c.name+"/impl=avx2", func(b *testing.B) {
			if !sgrIntermediateAVX2 {
				b.Skip("no AVX2")
			}
			for i := 0; i < b.N; i++ {
				calculateIntermediateRollingAVX2(dgd, origin, w, h, stride, 8, p, c.ri, c.pass, aBuf, bBuf, bs)
			}
		})
	}
}

// BenchmarkAMD64WienerU8 compares the 8-bit Wiener passes on a 64x64 unit:
// generic is the scalar reference, avx2 the bound amd64 kernels.
func BenchmarkAMD64WienerU8(b *testing.B) {
	_, _, _, scratch, info := benchWienerU8Inputs(b)
	// A frame-like stride, so the padded 16-byte loads stay resident and the
	// SIMD horizontal pass runs (benchWienerU8Inputs' tight stride makes it
	// fall back).
	const stride = 96
	origin := WienerHalfwin*stride + 8
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xabc)
	src := randomU8Plane(rnd, stride, 64+2*WienerHalfwin)
	round0, round1 := wienerRounds(8)
	dst := make([]uint8, 64*64)
	type pass struct {
		h func([]uint8, int, int, int, int, WienerFilter, int, []uint16)
		v func([]uint16, int, []uint8, int, int, int, WienerFilter, int)
	}
	for _, c := range []struct {
		name string
		p    pass
	}{{"generic", pass{wienerHorizontalU8, wienerVerticalU8}}, {"avx2_8wide", pass{wienerHorizontalU8Impl, wienerVerticalU8Impl}}, {"avx2", pass{wienerHorizontalU8Impl, wienerVerticalU8Impl}}} {
		wienerU8Wide16 = c.name != "avx2_8wide"
		b.Run("pass=horizontal/impl="+c.name, func(b *testing.B) {
			for b.Loop() {
				c.p.h(src, stride, origin, 64, 64, info.HFilter, round0, scratch)
			}
		})
		b.Run("pass=vertical/impl="+c.name, func(b *testing.B) {
			for b.Loop() {
				c.p.v(scratch, 64, dst, 64, 64, 64, info.VFilter, round1)
			}
		})
	}
	wienerU8Wide16 = true
}

// wienerSIMDDetected and sgrSIMDDetected report whether init bound the SIMD
// Wiener and self-guided kernels (AVX2 is required by the amd64 kernels).
func wienerSIMDDetected() bool { return archsimd.X86.AVX2() }

func sgrSIMDDetected() bool { return archsimd.X86.AVX2() }

// sgrSIMDBoxsumKernel is the box-sum kernel bound alongside the SIMD
// self-guided drivers.
var sgrSIMDBoxsumKernel = boxsumSIMD

var (
	wantWienerHorizontalU8Impl any = wienerHorizontalU8SIMDChecked
	wantWienerVerticalU8Impl   any = wienerVerticalU8SIMD
)
