//go:build !(goexperiment.simd && arm64 && !purego)

package encoder

func svtDownsample2DArch(dst []byte, dstStride int, src []byte, srcStride, w, h int) {
	svtDownsample2D(dst, dstStride, src, srcStride, w, h)
}

func sad16xNx4(src []byte, srcStride int, ref []byte, refStride, rows int) (int, int, int, int) {
	var s [4]int
	for k := range s {
		for r := 0; r < rows; r++ {
			a := src[r*srcStride : r*srcStride+16]
			b := ref[r*refStride+k : r*refStride+k+16]
			for c := range a {
				d := int(a[c]) - int(b[c])
				if d < 0 {
					d = -d
				}
				s[k] += d
			}
		}
	}
	return s[0], s[1], s[2], s[3]
}
