//go:build goexperiment.simd && arm64 && !purego

package encoder

import "simd/archsimd"

// lpd1SampledSAD is the row-subsampled SAD the chroma detector uses
// (svt_nxm_sad_kernel called with stride << shift and height >> shift), as
// svt_nxm_sad_kernel_helper_neon: 16-byte rows directly, 8-byte rows two to
// a vector. The detector samples at most 8 rows of at most 32 samples, so a
// uint16 lane gains at most 16*510.
func lpd1SampledSAD(src, ref []byte, srcOff, refOff, srcStride, refStride, w, rows, step int) int {
	if rows <= 0 || w <= 0 {
		return 0
	}
	ss, rs := step*srcStride, step*refStride
	if (w != 8 && w&15 != 0) || rows > 8 || w > 32 {
		return lpd1SampledSADPureGo(src, ref, srcOff, refOff, srcStride, refStride, w, rows, step)
	}
	if uint(ss) > rowStrideMax || uint(rs) > rowStrideMax {
		panic(errKernelArgs)
	}
	src, ref = src[srcOff:], ref[refOff:]
	var acc archsimd.Uint16x8
	if w == 8 {
		// One window per row pair: row 2i at w[:8], row 2i+1 at w[ss:].
		sn, rn := ss+8, rs+8
		for i := 0; i < rows/2; i++ {
			so, ro := 2*i*ss, 2*i*rs
			sw := src[so : so+sn : so+sn]
			rw := ref[ro : ro+rn : ro+rn]
			acc = absAcc16(acc, pack2((*[8]byte)(sw), (*[8]byte)(sw[ss:])), pack2((*[8]byte)(rw), (*[8]byte)(rw[rs:])))
		}
		total := sumU16(acc)
		if rows&1 != 0 {
			r := rows - 1
			s := src[r*ss : r*ss+8]
			d := ref[r*rs : r*rs+8]
			for c := range s {
				diff := int(s[c]) - int(d[c])
				total += max(diff, -diff)
			}
		}
		return total
	}
	for r := range rows {
		s, d := src[r*ss:r*ss+w], ref[r*rs:r*rs+w]
		for c := 0; c+16 <= len(s) && c+16 <= len(d); c += 16 {
			acc = absAcc16(acc, archsimd.LoadUint8x16Array((*[16]uint8)(s[c:c+16])), archsimd.LoadUint8x16Array((*[16]uint8)(d[c:c+16])))
		}
	}
	return sumU16(acc)
}
