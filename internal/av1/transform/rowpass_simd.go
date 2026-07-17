// SIMD dispatch for the inverse-transform row pass.
//
// The separable inverse transform's first 1D pass runs along contiguous rows
// (stride 1). Because the butterfly within a single row is a tight data
// dependency chain, the available parallelism is *across* rows: the same
// coefficient index in two adjacent rows is transformed by identical
// arithmetic. The batched kernels below exploit that by processing two rows
// at once. On arm64 with NEON they widen each int32 lane to an int64 lane
// (SMULL/SMLAL) so the fixed-point products, rounding shifts and stage clamps
// are bit-for-bit identical to the scalar pure-Go reference; on every other
// target they fall back to running the scalar kernel twice.
//
// The dispatch slots are resolved exactly once at package init (see
// rowpass_dispatch_*.go), mirroring the dsp-package minmax pattern: a single
// indirect call in steady state with no per-call feature-detection branch.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// inverseDCT4Row2 / inverseDCT8Row2 / inverseDCT16Row2 / inverseADST4Row2 /
// inverseADST8Row2 transform two adjacent stride-1 rows in place. r0 and r1
// each hold at least the transform length. The two rows are independent; the
// result for each row equals the corresponding single-row scalar kernel.
//
// These are dispatch slots bound once at init. They must not be mutated
// concurrently with live decoding.
var (
	inverseDCT4Row2Impl  = inverseDCT4Row2PureGo
	inverseDCT8Row2Impl  = inverseDCT8Row2PureGo
	inverseDCT16Row2Impl = inverseDCT16Row2PureGo
	inverseDCT32Row2Impl = inverseDCT32Row2PureGo
	inverseDCT64Row2Impl = inverseDCT64Row2PureGo
	inverseADST4Row2Impl = inverseADST4Row2PureGo
	inverseADST8Row2Impl = inverseADST8Row2PureGo
)

// inverseDCT8Row4 through inverseDCT64Row4 transform four adjacent stride-1
// rows in place (dav1d's transposed four-lane row shape,
// src/arm/64/itx16.S). The result for each row equals the corresponding
// single-row scalar kernel. Same binding rules as the two-row slots.
var (
	inverseDCT8Row4Impl  = inverseDCT8Row4ViaRow2
	inverseDCT16Row4Impl = inverseDCT16Row4ViaRow2
	inverseDCT32Row4Impl = inverseDCT32Row4PureGo
	inverseDCT64Row4Impl = inverseDCT64Row4PureGo
)

// inverseADST8/16Row4 and their flip variants transform four adjacent
// stride-1 rows in place with the (flip-)ADST horizontal transform (transposed
// four-lane row shape). The result for each row equals the corresponding
// single-row scalar kernel. Same binding rules as the DCT four-row slots.
var (
	inverseADST8Row4Impl      = inverseADST8Row4PureGo
	inverseADST8Row4FlipImpl  = inverseADST8Row4FlipPureGo
	inverseADST16Row4Impl     = inverseADST16Row4PureGo
	inverseADST16Row4FlipImpl = inverseADST16Row4FlipPureGo
)

// inverse1DRow2 applies the 1D inverse row transform to two adjacent
// contiguous rows r0 and r1 (each stride 1, length len). For the lengths and
// types that have a batched kernel it routes through the dispatch slot;
// otherwise it falls back to two scalar inverse1DRow calls. The result is
// always identical to running inverse1DRow on each row separately.
func inverse1DRow2(r0 []int32, r1 []int32, length int, typ tx1DType, min int32, max int32) {
	switch typ {
	case tx1DDCT:
		switch length {
		case dct4Size:
			inverseDCT4Row2Impl(r0, r1, min, max)
			return
		case dct8Size:
			inverseDCT8Row2Impl(r0, r1, min, max)
			return
		case dct16Size:
			inverseDCT16Row2Impl(r0, r1, min, max)
			return
		case dct32Size:
			inverseDCT32Row2Impl(r0, r1, min, max)
			return
		case dct64Size:
			inverseDCT64Row2Impl(r0, r1, min, max)
			return
		}
	case tx1DADST:
		switch length {
		case adst4Size:
			inverseADST4Row2Impl(r0, r1, min, max)
			return
		case adst8Size:
			inverseADST8Row2Impl(r0, r1, min, max)
			return
		}
	}
	// No batched kernel for this length/type: run the scalar row kernel twice.
	inverse1DRow(r0, length, typ, min, max)
	inverse1DRow(r1, length, typ, min, max)
}

// inverse1DRow4 applies the 1D inverse row transform to four adjacent
// contiguous rows. For the DCT lengths that have a four-row kernel it routes
// through the dispatch slot; otherwise it falls back to two batched row
// pairs. The result is always identical to running inverse1DRow on each row
// separately.
func inverse1DRow4(r0, r1, r2, r3 []int32, length int, typ tx1DType, min int32, max int32) {
	switch typ {
	case tx1DDCT:
		switch length {
		case dct8Size:
			inverseDCT8Row4Impl(r0, r1, r2, r3, min, max)
			return
		case dct16Size:
			inverseDCT16Row4Impl(r0, r1, r2, r3, min, max)
			return
		case dct32Size:
			inverseDCT32Row4Impl(r0, r1, r2, r3, min, max)
			return
		case dct64Size:
			inverseDCT64Row4Impl(r0, r1, r2, r3, min, max)
			return
		}
	case tx1DADST:
		switch length {
		case adst8Size:
			inverseADST8Row4Impl(r0, r1, r2, r3, min, max)
			return
		case adst16Size:
			inverseADST16Row4Impl(r0, r1, r2, r3, min, max)
			return
		}
	case tx1DFlipADST:
		switch length {
		case adst8Size:
			inverseADST8Row4FlipImpl(r0, r1, r2, r3, min, max)
			return
		case adst16Size:
			inverseADST16Row4FlipImpl(r0, r1, r2, r3, min, max)
			return
		}
	}
	inverse1DRow2(r0, r1, length, typ, min, max)
	inverse1DRow2(r2, r3, length, typ, min, max)
}

// --- Pure-Go batched references --------------------------------------------
//
// These are the canonical reference implementations every SIMD variant must
// match bit-for-bit. Each simply applies the established scalar row kernel to
// both rows; the scalar kernels are themselves the bit-exact references.

func inverseDCT4Row2PureGo(r0 []int32, r1 []int32, min int32, max int32) {
	inverseDCT1D(r0, 1, dct4Size, min, max)
	inverseDCT1D(r1, 1, dct4Size, min, max)
}

func inverseDCT8Row2PureGo(r0 []int32, r1 []int32, min int32, max int32) {
	inverseDCT8Row(r0, min, max)
	inverseDCT8Row(r1, min, max)
}

func inverseDCT16Row2PureGo(r0 []int32, r1 []int32, min int32, max int32) {
	inverseDCT16Row(r0, min, max)
	inverseDCT16Row(r1, min, max)
}

func inverseDCT32Row2PureGo(r0 []int32, r1 []int32, min int32, max int32) {
	inverseDCT32Row(r0, min, max)
	inverseDCT32Row(r1, min, max)
}

func inverseDCT64Row2PureGo(r0 []int32, r1 []int32, min int32, max int32) {
	inverseDCT64Row(r0, min, max)
	inverseDCT64Row(r1, min, max)
}

func inverseDCT32Row4PureGo(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseDCT32Row2PureGo(r0, r1, min, max)
	inverseDCT32Row2PureGo(r2, r3, min, max)
}

func inverseDCT8Row4PureGo(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseDCT8Row2PureGo(r0, r1, min, max)
	inverseDCT8Row2PureGo(r2, r3, min, max)
}

func inverseDCT16Row4PureGo(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseDCT16Row2PureGo(r0, r1, min, max)
	inverseDCT16Row2PureGo(r2, r3, min, max)
}

// The default DCT8/DCT16 four-row route preserves the best architecture's
// two-row dispatch (notably AVX2 on amd64). ARM64 replaces these slots with
// native four-lane kernels during package initialization.
func inverseDCT8Row4ViaRow2(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseDCT8Row2Impl(r0, r1, min, max)
	inverseDCT8Row2Impl(r2, r3, min, max)
}

func inverseDCT16Row4ViaRow2(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseDCT16Row2Impl(r0, r1, min, max)
	inverseDCT16Row2Impl(r2, r3, min, max)
}

func inverseDCT64Row4PureGo(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseDCT64Row2PureGo(r0, r1, min, max)
	inverseDCT64Row2PureGo(r2, r3, min, max)
}

func inverseADST16Row4PureGo(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseADST1D(r0, 1, adst16Size, min, max)
	inverseADST1D(r1, 1, adst16Size, min, max)
	inverseADST1D(r2, 1, adst16Size, min, max)
	inverseADST1D(r3, 1, adst16Size, min, max)
}

func inverseADST8Row4PureGo(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseADST8(r0, 1, min, max)
	inverseADST8(r1, 1, min, max)
	inverseADST8(r2, 1, min, max)
	inverseADST8(r3, 1, min, max)
}

func inverseADST8Row4FlipPureGo(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseFlipADST1D(r0, 1, adst8Size, min, max)
	inverseFlipADST1D(r1, 1, adst8Size, min, max)
	inverseFlipADST1D(r2, 1, adst8Size, min, max)
	inverseFlipADST1D(r3, 1, adst8Size, min, max)
}

func inverseADST16Row4FlipPureGo(r0, r1, r2, r3 []int32, min int32, max int32) {
	inverseFlipADST1D(r0, 1, adst16Size, min, max)
	inverseFlipADST1D(r1, 1, adst16Size, min, max)
	inverseFlipADST1D(r2, 1, adst16Size, min, max)
	inverseFlipADST1D(r3, 1, adst16Size, min, max)
}

func inverseADST4Row2PureGo(r0 []int32, r1 []int32, min int32, max int32) {
	inverseADST1D(r0, 1, adst4Size, min, max)
	inverseADST1D(r1, 1, adst4Size, min, max)
}

func inverseADST8Row2PureGo(r0 []int32, r1 []int32, min int32, max int32) {
	inverseADST8Row(r0, min, max)
	inverseADST8Row(r1, min, max)
}
