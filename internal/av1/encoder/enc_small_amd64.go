//go:build goexperiment.simd && amd64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"

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
	_ = src[srcOff+(rows-1)*ss+w-1]
	_ = ref[refOff+(rows-1)*rs+w-1]
	// Strided rows inside the extents checked above; same shape as the SAD rows (checked rows +9-86%).
	sp := unsafe.Pointer(&src[srcOff])
	rp := unsafe.Pointer(&ref[refOff])
	var acc archsimd.Uint64x2
	if w == 8 {
		r := 0
		for ; r+2 <= rows; r += 2 {
			acc = acc.Add(pack2Rows(unsafe.Add(sp, r*ss), ss).SumOf8AbsDiff(pack2Rows(unsafe.Add(rp, r*rs), rs)))
		}
		total := sumU64(acc)
		if r < rows {
			s := src[srcOff+r*ss : srcOff+r*ss+8]
			d := ref[refOff+r*rs : refOff+r*rs+8]
			for c := range s {
				diff := int(s[c]) - int(d[c])
				total += max(diff, -diff)
			}
		}
		return total
	}
	for r := range rows {
		s, d := unsafe.Add(sp, r*ss), unsafe.Add(rp, r*rs)
		for c := 0; c < w; c += 16 {
			acc = acc.Add(load16(unsafe.Add(s, c)).SumOf8AbsDiff(load16(unsafe.Add(d, c))))
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
	// Bounds: every block reads up to 7 rows and 7 columns past its origin.
	maxSO, maxFO := 0, 0
	for _, b := range dlist {
		maxSO = max(maxSO, (int(b.BY)*8+7)*sstride+int(b.BX)*8+7)
		maxFO = max(maxFO, (int(b.BY)*8+7)*fstride+int(b.BX)*8+7)
	}
	_ = src[maxSO]
	_ = filtered[maxFO]
	// Strided rows inside the extents checked above; same shape as the SAD rows (checked rows +9-86%).
	sp := unsafe.Pointer(unsafe.SliceData(src))
	fp := unsafe.Pointer(unsafe.SliceData(filtered))
	var acc archsimd.Int32x8
	pairStep := 2 * subsampling
	var total uint64
	for n, b := range dlist {
		so := int(b.BY)*8*sstride + int(b.BX)*8
		fo := int(b.BY)*8*fstride + int(b.BX)*8
		for i := 0; i < 8; i += pairStep {
			s := pack2Rows(unsafe.Add(sp, so+i*sstride), subsampling*sstride).ExtendToUint16().AsInt16x16()
			f := pack2Rows(unsafe.Add(fp, fo+i*fstride), subsampling*fstride).ExtendToUint16().AsInt16x16()
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
