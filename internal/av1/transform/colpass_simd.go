// SIMD dispatch for the inverse-transform column pass.
//
// The separable inverse transform's second 1D pass runs down columns. In the
// scratch buffer a column's successive elements are spaced one row apart
// (stride == width int32), but two *adjacent* columns share each row's memory:
// scratch[row*width+col] and scratch[row*width+col+1] are contiguous. The
// batched kernels below exploit that by holding column col in lane 0 and
// column col+1 in lane 1 of a 64-bit-element NEON register and transforming
// both columns at once. The fixed-point products, rounding shifts and stage
// clamps are bit-for-bit identical to the scalar pure-Go reference (which is
// just inverse1D run on each column); on every non-NEON target the slot is the
// pure-Go reference.
//
// The dispatch slots are resolved exactly once at package init (see
// colpass_dispatch_*.go), mirroring the row-pass pattern: a single indirect
// call in steady state with no per-call feature-detection branch.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// inverseDCT8Col2 / inverseDCT16Col2 transform two adjacent columns of the
// scratch buffer in place. base points at scratch[col]; rowStride is the
// element distance between successive rows (== block width). The two columns
// (col, col+1) are independent; the result for each equals the corresponding
// single-column scalar kernel.
//
// These are dispatch slots bound once at init. They must not be mutated
// concurrently with live decoding.
var (
	inverseDCT8Col2Impl  = inverseDCT8Col2PureGo
	inverseDCT16Col2Impl = inverseDCT16Col2PureGo
	inverseDCT32Col2Impl = inverseDCT32Col2PureGo
	inverseDCT64Col2Impl = inverseDCT64Col2PureGo
)

// inverseDCT8Col4 through inverseDCT64Col4 transform four adjacent columns of the
// scratch buffer in place (dav1d's four-lane column shape, src/arm/64/itx16.S).
// The result for each column equals the corresponding single-column scalar
// kernel. Same binding rules as the two-column slots.
var (
	inverseDCT8Col4Impl  = inverseDCT8Col4ViaCol2
	inverseDCT16Col4Impl = inverseDCT16Col4ViaCol2
	inverseDCT32Col4Impl = inverseDCT32Col4PureGo
	inverseDCT64Col4Impl = inverseDCT64Col4PureGo
)

// inverseADSTCol4 slots transform four adjacent columns of the scratch buffer
// in place with the (flip-)ADST vertical transform (four
// columns per int32 lane group, dav1d's inv_adst_4s_x16_neon shape). The
// result for each column equals the corresponding single-column scalar kernel.
// Same binding rules as the DCT four-column slots.
var (
	inverseADST8Col4Impl      = inverseADST8Col4PureGo
	inverseADST8Col4FlipImpl  = inverseADST8Col4FlipPureGo
	inverseADST16Col4Impl     = inverseADST16Col4PureGo
	inverseADST16Col4FlipImpl = inverseADST16Col4FlipPureGo
	inverseIdentity4Col4Impl  = inverseIdentity4Col4PureGo
	inverseIdentity8Col4Impl  = inverseIdentity8Col4PureGo
	inverseIdentity16Col4Impl = inverseIdentity16Col4PureGo
	inverseIdentity32Col4Impl = inverseIdentity32Col4PureGo
)

// inverse1DCol2 applies the 1D inverse column transform to two adjacent
// columns (col, col+1) of buf, which is the row-major scratch buffer with the
// given rowStride (== width). For the DCT lengths that have a batched kernel it
// routes through the dispatch slot; otherwise it falls back to two scalar
// inverse1D calls. The result is always identical to running inverse1D on each
// column separately.
func inverse1DCol2(buf []int32, rowStride int, length int, typ tx1DType, min int32, max int32) {
	if typ == tx1DDCT {
		switch length {
		case dct8Size:
			inverseDCT8Col2Impl(buf, rowStride, min, max)
			return
		case dct16Size:
			inverseDCT16Col2Impl(buf, rowStride, min, max)
			return
		case dct32Size:
			inverseDCT32Col2Impl(buf, rowStride, min, max)
			return
		case dct64Size:
			inverseDCT64Col2Impl(buf, rowStride, min, max)
			return
		}
	}
	// No batched kernel for this length/type: run the scalar column kernel twice.
	inverse1D(buf, rowStride, length, typ, min, max)
	inverse1D(buf[1:], rowStride, length, typ, min, max)
}

// inverse1DCol4 applies the 1D inverse column transform to four adjacent
// columns (col..col+3) of buf. For the DCT lengths that have a four-column
// kernel it routes through the dispatch slot; otherwise it falls back to two
// batched column pairs. The result is always identical to running inverse1D
// on each column separately.
func inverse1DCol4(buf []int32, rowStride int, length int, typ tx1DType, min int32, max int32) {
	switch typ {
	case tx1DDCT:
		switch length {
		case dct8Size:
			inverseDCT8Col4Impl(buf, rowStride, min, max)
			return
		case dct16Size:
			inverseDCT16Col4Impl(buf, rowStride, min, max)
			return
		case dct32Size:
			inverseDCT32Col4Impl(buf, rowStride, min, max)
			return
		case dct64Size:
			inverseDCT64Col4Impl(buf, rowStride, min, max)
			return
		}
	case tx1DADST:
		if length == adst8Size {
			inverseADST8Col4Impl(buf, rowStride, min, max)
			return
		}
		if length == adst16Size {
			inverseADST16Col4Impl(buf, rowStride, min, max)
			return
		}
	case tx1DFlipADST:
		if length == adst8Size {
			inverseADST8Col4FlipImpl(buf, rowStride, min, max)
			return
		}
		if length == adst16Size {
			inverseADST16Col4FlipImpl(buf, rowStride, min, max)
			return
		}
	case tx1DIdentity:
		switch length {
		case 4:
			inverseIdentity4Col4Impl(buf, rowStride, min, max)
			return
		case 8:
			inverseIdentity8Col4Impl(buf, rowStride, min, max)
			return
		case 16:
			inverseIdentity16Col4Impl(buf, rowStride, min, max)
			return
		case 32:
			inverseIdentity32Col4Impl(buf, rowStride, min, max)
			return
		}
	}
	inverse1DCol2(buf, rowStride, length, typ, min, max)
	inverse1DCol2(buf[2:], rowStride, length, typ, min, max)
}

// --- Pure-Go batched references --------------------------------------------
//
// These are the canonical reference implementations every SIMD variant must
// match bit-for-bit. Each simply applies the established scalar column kernel
// to both adjacent columns.

func inverseDCT8Col2PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT8(buf, rowStride, min, max)
	inverseDCT8(buf[1:], rowStride, min, max)
}

func inverseDCT16Col2PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT16(buf, rowStride, min, max)
	inverseDCT16(buf[1:], rowStride, min, max)
}

func inverseDCT32Col2PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT32(buf, rowStride, min, max)
	inverseDCT32(buf[1:], rowStride, min, max)
}

func inverseDCT64Col2PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT64(buf, rowStride, min, max)
	inverseDCT64(buf[1:], rowStride, min, max)
}

func inverseDCT8Col4ViaCol2(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT8Col2Impl(buf, rowStride, min, max)
	inverseDCT8Col2Impl(buf[2:], rowStride, min, max)
}

func inverseDCT16Col4ViaCol2(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT16Col2Impl(buf, rowStride, min, max)
	inverseDCT16Col2Impl(buf[2:], rowStride, min, max)
}

func inverseDCT8Col4PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT8Col2PureGo(buf, rowStride, min, max)
	inverseDCT8Col2PureGo(buf[2:], rowStride, min, max)
}

func inverseDCT16Col4PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT16Col2PureGo(buf, rowStride, min, max)
	inverseDCT16Col2PureGo(buf[2:], rowStride, min, max)
}

func inverseDCT32Col4PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT32Col2PureGo(buf, rowStride, min, max)
	inverseDCT32Col2PureGo(buf[2:], rowStride, min, max)
}

func inverseDCT64Col4PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseDCT64Col2PureGo(buf, rowStride, min, max)
	inverseDCT64Col2PureGo(buf[2:], rowStride, min, max)
}

func inverseADST8Col4PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseADST8(buf, rowStride, min, max)
	inverseADST8(buf[1:], rowStride, min, max)
	inverseADST8(buf[2:], rowStride, min, max)
	inverseADST8(buf[3:], rowStride, min, max)
}

func inverseADST8Col4FlipPureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseFlipADST1D(buf, rowStride, adst8Size, min, max)
	inverseFlipADST1D(buf[1:], rowStride, adst8Size, min, max)
	inverseFlipADST1D(buf[2:], rowStride, adst8Size, min, max)
	inverseFlipADST1D(buf[3:], rowStride, adst8Size, min, max)
}

func inverseADST16Col4PureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseADST16(buf, rowStride, min, max)
	inverseADST16(buf[1:], rowStride, min, max)
	inverseADST16(buf[2:], rowStride, min, max)
	inverseADST16(buf[3:], rowStride, min, max)
}

func inverseADST16Col4FlipPureGo(buf []int32, rowStride int, min int32, max int32) {
	inverseFlipADST1D(buf, rowStride, adst16Size, min, max)
	inverseFlipADST1D(buf[1:], rowStride, adst16Size, min, max)
	inverseFlipADST1D(buf[2:], rowStride, adst16Size, min, max)
	inverseFlipADST1D(buf[3:], rowStride, adst16Size, min, max)
}

func inverseIdentity4Col4PureGo(buf []int32, rowStride int, _, _ int32) {
	inverseIdentityCol4PureGo(buf, rowStride, 4)
}

func inverseIdentity8Col4PureGo(buf []int32, rowStride int, _, _ int32) {
	inverseIdentityCol4PureGo(buf, rowStride, 8)
}

func inverseIdentity16Col4PureGo(buf []int32, rowStride int, _, _ int32) {
	inverseIdentityCol4PureGo(buf, rowStride, 16)
}

func inverseIdentity32Col4PureGo(buf []int32, rowStride int, _, _ int32) {
	inverseIdentityCol4PureGo(buf, rowStride, 32)
}

func inverseIdentityCol4PureGo(buf []int32, rowStride int, length int) {
	inverseIdentity1D(buf, rowStride, length)
	inverseIdentity1D(buf[1:], rowStride, length)
	inverseIdentity1D(buf[2:], rowStride, length)
	inverseIdentity1D(buf[3:], rowStride, length)
}
