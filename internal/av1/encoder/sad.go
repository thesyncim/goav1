package encoder

// sad.go hosts the motion-search SAD kernel dispatch. sad8x8Impl computes the
// sum of absolute differences of one 8x8 block; implementations may ignore
// limit (it is an early-exit hint, not a contract), so callers must compare
// the returned total against their own threshold. Every architecture variant
// must be bit-exact with sad8x8PureGo.
var sad8x8Impl = sad8x8PureGo

// sad8x8PureGo is the portable reference with a row-granular early exit.
func sad8x8PureGo(src, ref []byte, stride int, limit int) int {
	total := 0
	for r := range 8 {
		row := r * stride
		for c := range 8 {
			d := int(src[row+c]) - int(ref[row+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
		if total >= limit {
			return total
		}
	}
	return total
}
