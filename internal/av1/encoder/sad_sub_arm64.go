//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"unsafe"
)

// sad8x4x4 computes four 8x4 SADs of one source block against four reference
// origins sharing stride: svt_aom_compute8x4_sad_kernel, the row-subsampled
// 8x8 SAD of SVT-AV1's SUB_SAD_SEARCH when called with a doubled stride.
func sad8x4x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	_ = src[3*stride+7]
	_ = ref0[3*stride+7]
	_ = ref1[3*stride+7]
	_ = ref2[3*stride+7]
	_ = ref3[3*stride+7]
	sp := unsafe.Pointer(&src[0])
	p0, p1, p2, p3 := unsafe.Pointer(&ref0[0]), unsafe.Pointer(&ref1[0]), unsafe.Pointer(&ref2[0]), unsafe.Pointer(&ref3[0])
	s := pack2Rows(sp, stride)
	c0 := widen16(absDiffU8x16(s, pack2Rows(p0, stride)))
	c1 := widen16(absDiffU8x16(s, pack2Rows(p1, stride)))
	c2 := widen16(absDiffU8x16(s, pack2Rows(p2, stride)))
	c3 := widen16(absDiffU8x16(s, pack2Rows(p3, stride)))
	o := 2 * stride
	s = pack2Rows(step(sp, o), stride)
	c0 = c0.Add(widen16(absDiffU8x16(s, pack2Rows(step(p0, o), stride))))
	c1 = c1.Add(widen16(absDiffU8x16(s, pack2Rows(step(p1, o), stride))))
	c2 = c2.Add(widen16(absDiffU8x16(s, pack2Rows(step(p2, o), stride))))
	c3 = c3.Add(widen16(absDiffU8x16(s, pack2Rows(step(p3, o), stride))))
	return int(c0.ReduceSum()), int(c1.ReduceSum()), int(c2.ReduceSum()), int(c3.ReduceSum())
}
