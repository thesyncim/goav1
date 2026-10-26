//go:build goexperiment.simd && amd64 && !purego

package restoration

import "testing"

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
