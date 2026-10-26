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

func init() {
	if encAVX2 {
		realtimeAvg8x8QuadImpl = realtimeAvg8x8QuadAVX2
	}
}

// realtimeAvg8x8QuadAVX2 is realtimeAvg8x8QuadWide on 16-byte rows: VPSADBW
// against zero sums the left and right eight bytes of a row into the two
// lanes, so each eight-row band is two accumulators.
func realtimeAvg8x8QuadAVX2(src []byte, stride int) (int, int, int, int) {
	_ = src[15*stride+15]
	p := unsafe.Pointer(&src[0])
	var zero archsimd.Uint8x16
	var top, bot archsimd.Uint64x2
	for r := 0; r < 8; r++ {
		top = top.Add(load16(unsafe.Add(p, r*stride)).SumOf8AbsDiff(zero))
		bot = bot.Add(load16(unsafe.Add(p, (r+8)*stride)).SumOf8AbsDiff(zero))
	}
	tl, tr := int(top.GetElem(0)), int(top.GetElem(1))
	bl, br := int(bot.GetElem(0)), int(bot.GetElem(1))
	return (tl + 32) >> 6, (tr + 32) >> 6, (bl + 32) >> 6, (br + 32) >> 6
}

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

// sad8x4x4 computes four 8x4 SADs of one source block against four reference
// origins sharing stride: svt_aom_compute8x4_sad_kernel, the row-subsampled
// 8x8 SAD of SVT-AV1's SUB_SAD_SEARCH when called with a doubled stride.
func sad8x4x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	_ = src[3*stride+7]
	_ = ref0[3*stride+7]
	_ = ref1[3*stride+7]
	_ = ref2[3*stride+7]
	_ = ref3[3*stride+7]
	if !encAVX2 {
		return sad8x4PureGo(src, ref0, stride), sad8x4PureGo(src, ref1, stride),
			sad8x4PureGo(src, ref2, stride), sad8x4PureGo(src, ref3, stride)
	}
	sp := unsafe.Pointer(&src[0])
	p0, p1, p2, p3 := unsafe.Pointer(&ref0[0]), unsafe.Pointer(&ref1[0]), unsafe.Pointer(&ref2[0]), unsafe.Pointer(&ref3[0])
	o := 2 * stride
	s, t := pack2Rows(sp, stride), pack2Rows(step(sp, o), stride)
	c0 := s.SumOf8AbsDiff(pack2Rows(p0, stride)).Add(t.SumOf8AbsDiff(pack2Rows(step(p0, o), stride)))
	c1 := s.SumOf8AbsDiff(pack2Rows(p1, stride)).Add(t.SumOf8AbsDiff(pack2Rows(step(p1, o), stride)))
	c2 := s.SumOf8AbsDiff(pack2Rows(p2, stride)).Add(t.SumOf8AbsDiff(pack2Rows(step(p2, o), stride)))
	c3 := s.SumOf8AbsDiff(pack2Rows(p3, stride)).Add(t.SumOf8AbsDiff(pack2Rows(step(p3, o), stride)))
	return sumU64(c0), sumU64(c1), sumU64(c2), sumU64(c3)
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
