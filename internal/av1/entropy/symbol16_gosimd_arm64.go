//go:build goexperiment.simd && arm64 && !purego

package entropy

import "simd/archsimd"

// symbol16SIMD gates the vector symbol search for large alphabets.
const symbol16SIMD = true

// symbol16Lanes holds, per last = n_symbols - 1, dav1d's msac coeffs row
// (EC_MIN_PROB * (n_symbols - 1 - i)) and a lane mask that keeps the bounds
// and CDF entries below last; lanes at and past last (the terminal zero and
// the adaptation count) stay zero in the bounds and untouched by the update.
var symbol16Lanes = func() (t [MaxSymbols][2][MaxSymbols]uint16) {
	for last := range MaxSymbols {
		for i := 0; i < last; i++ {
			t[last][0][i] = uint16(ecMinProb * (last - i))
			t[last][1][i] = 0xffff
		}
	}
	return t
}()

// decodeSymbol16SIMD is dav1d's arm64 msac_decode_symbol_adapt16
// (src/arm/64/msac.S) for alphabets of 9 to 16 symbols: every bound
// v[i] = ((rng >> 8) * (cdf[i] >> EC_PROB_SHIFT) >> 1) + EC_MIN_PROB * (last - i)
// is computed in 16-bit lanes, the symbol is the count of bounds above the
// coded value, and the CDF adapts under the same lane mask. NEON has no
// sqdmulh form in archsimd, so the product is split at bit 7 of the CDF
// entry to stay within 16 bits: (a * r) >> 1 == (a >> 1) * r + (a & 1) * (r >> 1)
// with a = cdf >> 6. It returns the symbol and its [lower, upper) interval.
func decodeSymbol16SIMD(values *[MaxSymbols + 1]uint16, last int, rng, coded uint32, update bool) (int, uint32, uint32) {
	last &= MaxSymbols - 1
	tab := &symbol16Lanes[last]
	c0 := archsimd.LoadUint16x8Array((*[8]uint16)(values[0:8]))
	c1 := archsimd.LoadUint16x8Array((*[8]uint16)(values[8:16]))
	m0 := archsimd.LoadUint16x8Array((*[8]uint16)(tab[0][0:8]))
	m1 := archsimd.LoadUint16x8Array((*[8]uint16)(tab[0][8:16]))
	k0 := archsimd.LoadUint16x8Array((*[8]uint16)(tab[1][0:8]))
	k1 := archsimd.LoadUint16x8Array((*[8]uint16)(tab[1][8:16]))
	// Scalars reach the vector unit through a store and a vector load (eight
	// copies per 128-bit row), which Apple cores forward more cheaply than
	// lane inserts; constants come from memory, since archsimd shifts by an
	// immediate lower to a GPR insert plus a register shift.
	const lanes = 0x0001000100010001
	splat := [3][2]uint64{
		{uint64(rng>>8) * lanes, uint64(rng>>8) * lanes},
		{uint64(rng>>9) * lanes, uint64(rng>>9) * lanes},
		{uint64(coded) * lanes, uint64(coded) * lanes},
	}
	r := archsimd.LoadUint64x2Array(&splat[0]).ReshapeToUint16s()
	rh := archsimd.LoadUint64x2Array(&splat[1]).ReshapeToUint16s()
	cd := archsimd.LoadUint64x2Array(&splat[2]).ReshapeToUint16s()
	k := &symbol16Consts
	one := archsimd.LoadUint16x8Array(&k.one)
	sh7 := archsimd.LoadInt16x8Array(&k.shr7)
	bit6 := archsimd.LoadUint16x8Array(&k.bit6)
	sh6 := archsimd.LoadInt16x8Array(&k.shr6)

	v0 := c0.Shift(sh7).Mul(r).Add(c0.And(bit6).Shift(sh6).Mul(rh)).Add(m0).And(k0)
	v1 := c1.Shift(sh7).Mul(r).Add(c1.And(bit6).Shift(sh6).Mul(rh)).Add(m1).And(k1)
	// 1 where coded < v (the bounds decrease, so these lead), else 0.
	s0 := v0.SubSaturated(cd).Min(one)
	s1 := v1.SubSaturated(cd).Min(one)
	n := int(s0.Add(s1).ReduceSum()) & (MaxSymbols - 1)

	// dav1d stores the bounds for indexed access: bound[n] is the upper end
	// (the whole range for symbol 0) and bound[n+1] the lower (0 past last).
	var bound [MaxSymbols + 2]uint16
	bound[0] = uint16(rng)
	v0.Store(bound[1:9])
	v1.Store(bound[9:17])
	upper := uint32(bound[n])
	lower := uint32(bound[n+1])

	if update {
		count := values[last+1]
		rate := archsimd.LoadInt16x8Array(&k.shrRate[(count>>4)&3])
		top := archsimd.LoadUint16x8Array(&k.top)
		values0 := adaptCDF8(c0, s0, k0, top, rate)
		values1 := adaptCDF8(c1, s1, k1, top, rate)
		values0.Store(values[0:8])
		values1.Store(values[8:16])
		if count < MaxCDFCount {
			values[last+1] = count + 1
		}
	}
	return n, lower, upper
}

// adaptCDF8 is update_cdf on eight entries: up (toward 32768) where s is 1,
// down otherwise, and unchanged where the lane mask k is clear.
func adaptCDF8(c, s, k, top archsimd.Uint16x8, rate archsimd.Int16x8) archsimd.Uint16x8 {
	down := c.Sub(c.Shift(rate))
	up := c.Add(top.Sub(c).Shift(rate))
	next := down.Add(up.Sub(down).Mul(s))
	return c.Add(next.Sub(c).And(k))
}

func splat8[T uint16 | int16](v T) (a [8]T) {
	for i := range a {
		a[i] = v
	}
	return a
}

// symbol16Consts are the kernel's vector constants; a negative USHL count is
// a right shift. shrRate is indexed by count >> 4 (rate 5, 6, 7).
var symbol16Consts = struct {
	one, bit6, top [8]uint16
	shr7, shr6     [8]int16
	shrRate        [4][8]int16
}{
	one:     splat8[uint16](1),
	bit6:    splat8[uint16](64),
	top:     splat8[uint16](CDFProbTop),
	shr7:    splat8[int16](-7),
	shr6:    splat8[int16](-6),
	shrRate: [4][8]int16{splat8[int16](-5), splat8[int16](-6), splat8[int16](-7), splat8[int16](-7)},
}
