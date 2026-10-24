package encoder

// lpd1SampledSADPureGo is lpd1SampledSAD's reference definition.
func lpd1SampledSADPureGo(src, ref []byte, srcOff, refOff, srcStride, refStride, w, rows, step int) int {
	total := 0
	for r := 0; r < rows; r++ {
		s := src[srcOff+r*step*srcStride:]
		d := ref[refOff+r*step*refStride:]
		for c := 0; c < w; c++ {
			diff := int(s[c]) - int(d[c])
			if diff < 0 {
				diff = -diff
			}
			total += diff
		}
	}
	return total
}
