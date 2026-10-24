//go:build goexperiment.simd && arm64 && !purego

package entropy

import (
	"math/rand/v2"
	"testing"
)

// symbol16Reference is the sequential msac search and update_cdf.
func symbol16Reference(values *[MaxSymbols + 1]uint16, symbols int, rng, coded uint32, update bool) (int, uint32, uint32) {
	rngHi := rng >> 8
	upper, lower, symbol, last := rng, uint32(0), 0, symbols-1
	minTerm := ecMinProb * uint32(last)
	for symbol < last {
		lower = cdfBound(rngHi, uint32(values[symbol]), minTerm)
		if coded >= lower {
			break
		}
		symbol++
		upper = lower
		minTerm -= ecMinProb
	}
	if symbol == last {
		lower = 0
	}
	if update {
		count := values[symbols]
		rate := uint(5 + ((count >> 4) & 3))
		for i := 0; i < last; i++ {
			values[i] = uint16(adaptCDFEntry(uint32(values[i]), rate, i < symbol))
		}
		if count < MaxCDFCount {
			values[symbols] = count + 1
		}
	}
	return symbol, lower, upper
}

func randomInverseCDF(rnd *rand.Rand, symbols int) [MaxSymbols + 1]uint16 {
	var v [MaxSymbols + 1]uint16
	// Inverse CDF: strictly decreasing from below 32768 to 0 at symbols-1.
	cut := make([]int, symbols-1)
	for i := range cut {
		cut[i] = 1 + rnd.IntN(32767)
	}
	for i := 0; i < len(cut); i++ {
		for j := i + 1; j < len(cut); j++ {
			if cut[j] > cut[i] {
				cut[i], cut[j] = cut[j], cut[i]
			}
		}
	}
	for i, c := range cut {
		v[i] = uint16(c)
	}
	v[symbols] = uint16(rnd.IntN(MaxCDFCount + 1))
	for i := symbols + 1; i <= MaxSymbols; i++ {
		v[i] = uint16(rnd.Uint32())
	}
	return v
}

func TestDecodeSymbol16SIMDMatchesScalar(t *testing.T) {
	rnd := rand.New(rand.NewPCG(1, 2))
	for iter := 0; iter < 400000; iter++ {
		symbols := 9 + rnd.IntN(8)
		v := randomInverseCDF(rnd, symbols)
		rng := uint32(0x8000 + rnd.IntN(0x8000))
		coded := uint32(rnd.IntN(int(rng)))
		update := rnd.IntN(4) != 0
		want := v
		ws, wl, wu := symbol16Reference(&want, symbols, rng, coded, update)
		got := v
		gs, gl, gu := decodeSymbol16SIMD(&got, symbols-1, rng, coded, update)
		if gs != ws || gl != wl || gu != wu || got != want {
			t.Fatalf("symbols=%d rng=%#x coded=%#x cdf=%v: got (%d,%d,%d) %v want (%d,%d,%d) %v", symbols, rng, coded, v, gs, gl, gu, got, ws, wl, wu, want)
		}
	}
}
