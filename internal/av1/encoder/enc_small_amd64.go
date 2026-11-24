//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"

	"github.com/thesyncim/goav1/internal/av1/cdef"
)

// AVX2 forms of the encoder's small arm64-only kernels. VPSADBW sums the
// absolute differences of each 8-byte group into a 64-bit lane, so the SADs
// and byte sums below need no lane-overflow care; every kernel falls back to
// its scalar reference without AVX2.

// encAVX2 gates the kernels in this file.
var encAVX2 = gosimdKernelsSupported()

// lpd1SampledSAD is the row-subsampled SAD the chroma detector uses
// (svt_nxm_sad_kernel called with stride << shift and height >> shift), as
// svt_nxm_sad_kernel_helper_avx2: 16-byte rows directly, 8-byte rows two to a
// vector.
func lpd1SampledSAD(src, ref []byte, srcOff, refOff, srcStride, refStride, w, rows, step int) int {
	if rows <= 0 || w <= 0 {
		return 0
	}
	ss, rs := step*srcStride, step*refStride
	if !encAVX2 || (w != 8 && w&15 != 0) || rows > 8 || w > 32 {
		return lpd1SampledSADPureGo(src, ref, srcOff, refOff, srcStride, refStride, w, rows, step)
	}
	if uint(ss) > rowStrideMax || uint(rs) > rowStrideMax {
		panic(errKernelArgs)
	}
	src, ref = src[srcOff:], ref[refOff:]
	var acc archsimd.Uint64x2
	if w == 8 {
		// One window per row pair: row 2i at w[:8], row 2i+1 at w[ss:].
		sn, rn := ss+8, rs+8
		for i := 0; i < rows/2; i++ {
			so, ro := 2*i*ss, 2*i*rs
			sw := src[so : so+sn : so+sn]
			rw := ref[ro : ro+rn : ro+rn]
			acc = acc.Add(pack2((*[8]byte)(sw), (*[8]byte)(sw[ss:])).SumOf8AbsDiff(pack2((*[8]byte)(rw), (*[8]byte)(rw[rs:]))))
		}
		total := sumU64(acc)
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
			acc = acc.Add(archsimd.LoadUint8x16Array((*[16]uint8)(s[c : c+16])).SumOf8AbsDiff(archsimd.LoadUint8x16Array((*[16]uint8)(d[c : c+16]))))
		}
	}
	return sumU64(acc)
}

// memsetBytes sets every byte of dst to v: the memset of
// svt_aom_generate_padding's left and right borders and of the flat DC
// predictions. Go lowers a byte-store loop to one store per sample.
func memsetBytes(dst []byte, v byte) {
	i := 0
	if encAVX2 && len(dst) >= 16 {
		x := archsimd.BroadcastUint8x16(v)
		for ; i+16 <= len(dst); i += 16 {
			x.StoreArray((*[16]uint8)(dst[i : i+16]))
		}
	}
	for ; i < len(dst); i++ {
		dst[i] = v
	}
}

// cdefDist8 is SVT-AV1's svt_aom_compute_cdef_dist_8bit_avx2 for the 8x8
// luma blocks: rows i and i+subsampling pack into one 16-byte vector, widen to
// sixteen int16 lanes, and VPMADDWD squares and pair-sums the differences
// into int32 lanes. Other block shapes take the scalar reference.
func cdefDist8(src []byte, sstride int, filtered []byte, fstride int, dlist []cdef.BlockPosition, bwLog2, bhLog2 int, subsampling int) uint64 {
	if !encAVX2 || bwLog2 != 3 || bhLog2 != 3 || (subsampling != 1 && subsampling != 2 && subsampling != 4) {
		return computeCDEFDist8(src, sstride, filtered, fstride, dlist, bwLog2, bhLog2, subsampling)
	}
	if len(dlist) == 0 {
		return 0
	}
	if uint(sstride) > rowStrideMax || uint(fstride) > rowStrideMax {
		panic(errKernelArgs)
	}
	// Each row pair (i, i+subsampling) is one bounds-checked window: the
	// first row at w[:8], the second at w[sd:].
	sd, fd := subsampling*sstride, subsampling*fstride
	sn, fn := sd+8, fd+8
	pairs := 4 / subsampling
	var acc archsimd.Int32x8
	var total uint64
	for n, b := range dlist {
		so := int(b.BY)*8*sstride + int(b.BX)*8
		fo := int(b.BY)*8*fstride + int(b.BX)*8
		for k := 0; k < pairs; k++ {
			o, q := so+2*k*sd, fo+2*k*fd
			sw := src[o : o+sn : o+sn]
			fw := filtered[q : q+fn : q+fn]
			s := pack2((*[8]byte)(sw), (*[8]byte)(sw[sd:])).ExtendToUint16().AsInt16x16()
			f := pack2((*[8]byte)(fw), (*[8]byte)(fw[fd:])).ExtendToUint16().AsInt16x16()
			d := s.Sub(f)
			acc = acc.Add(d.DotProductPairs(d))
		}
		// An int32 lane gains at most 4*2*65025 a block: flushing every 128
		// blocks keeps the eight-lane total below 2^31.
		if n&127 == 127 {
			total += uint64(uint32(reduceInt32x8(acc)))
			acc = archsimd.Int32x8{}
		}
	}
	total += uint64(uint32(reduceInt32x8(acc)))
	archsimd.ClearAVXUpperBits()
	return total
}
