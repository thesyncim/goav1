//go:build !(goexperiment.simd && arm64 && !purego)

package entropy

const symbol16SIMD = false

func decodeSymbol16SIMD(values *[MaxSymbols + 1]uint16, last int, rng, coded uint32, update bool) (int, uint32, uint32) {
	panic("unreachable")
}
