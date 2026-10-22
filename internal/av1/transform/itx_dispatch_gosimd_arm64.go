// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

// colClampBoundNEON is the stage-range envelope the four-column int32-lane
// kernels are proven overflow-free for: every supported bit depth's row and
// column stage bounds satisfy |bound| <= 1<<19 (stageRangeBounds caps at
// bitDepth 12: rowBits 20). Wider bounds fall back to pure Go.
const colClampBoundNEON = 1 << 19

// init binds the Go SIMD batched inverse kernels that replaced the arm64 NEON
// assembly. The column kernels run four (or two) columns in int32 lanes and the
// row kernels run rows through a 4x4 transpose; the adapters keep the same
// envelope gating the assembly adapters had and fall back to the pure-Go
// reference outside it, so output bytes do not depend on the path taken.
func init() {
	inverseDCT4Col4Impl = inverseDCT4Col4SIMDAdapter
	inverseDCT8Col2Impl = inverseDCT8Col2SIMDAdapter
	inverseDCT8Col4Impl = inverseDCT8Col4SIMDAdapter
	inverseDCT16Col2Impl = inverseDCT16Col2SIMDAdapter
	inverseDCT16Col4Impl = inverseDCT16Col4SIMDAdapter
	inverseADST16Col4Impl = inverseADST16Col4SIMDAdapter
	inverseADST16Col4FlipImpl = inverseADST16Col4FlipSIMDAdapter
	inverseDCT8Row2Impl = inverseDCT8Row2SIMDAdapter
	inverseDCT16Row2Impl = inverseDCT16Row2SIMDAdapter
	inverseADST16Row4Impl = inverseADST16Row4SIMDAdapter
	inverseADST16Row4FlipImpl = inverseADST16Row4FlipSIMDAdapter
	inverseDCT32Col2Impl = inverseDCT32Col2SIMDAdapter
	inverseDCT32Col4Impl = inverseDCT32Col4SIMDAdapter
	inverseDCT32Low8Col4Impl = inverseDCT32Low8Col4SIMDAdapter
	inverseDCT32Row2Impl = inverseDCT32Row2SIMDAdapter
	inverseDCT32Row4Impl = inverseDCT32Row4SIMDAdapter
	inverseDCT32Low16Col4Impl = inverseDCT32Low16Col4SIMDAdapter
	inverseDCT32Row4LowImpl = inverseDCT32Row4LowSIMDAdapter
	inverseDCT32Row2LowImpl = inverseDCT32Row2LowSIMDAdapter
	inverseDCT16Row2LowImpl = inverseDCT16Row2LowSIMDAdapter
	inverseDCT16LowCol4Impl = inverseDCT16LowCol4SIMDAdapter
}

// simdColEnvelope reports whether the [min, max] stage clamp lies inside the
// +/-2^19 envelope the int32-lane kernels are exact for. Column kernels rely on
// the staged pre-clamp exactly as the assembly kernels did; row kernels read
// rows that inverseSeparableBlockClampedRowsToScratch staged through
// stageTransposeClamp into the same [min, max] range, so checking the bounds
// covers every input without scanning the rows.
func simdColEnvelope(min, max int32) bool {
	return min >= -colClampBoundNEON && max < colClampBoundNEON
}

func inverseDCT4Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct4Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT4Col4PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT4Col4SIMD(buf, rowStride, min, max)
}

func inverseDCT8Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct8Size-1)*rowStride+2 || !simdColEnvelope(min, max) {
		inverseDCT8Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT8Col2SIMD(buf, rowStride, min, max)
}

func inverseDCT16Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct16Size-1)*rowStride+2 || !simdColEnvelope(min, max) {
		inverseDCT16Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT16Col2SIMD(buf, rowStride, min, max)
}

func inverseDCT8Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct8Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT8Col2SIMDAdapter(buf, rowStride, min, max)
		inverseDCT8Col2SIMDAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT8Col4SIMD(buf, rowStride, min, max)
}

func inverseDCT16Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT16Col2SIMDAdapter(buf, rowStride, min, max)
		inverseDCT16Col2SIMDAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT16Col4SIMD(buf, rowStride, min, max)
}

func inverseADST16Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (adst16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseADST16Col4PureGo(buf, rowStride, min, max)
		return
	}
	inverseADST16Col4SIMD(buf, rowStride, min, max)
}

func inverseADST16Col4FlipSIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (adst16Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseADST16Col4FlipPureGo(buf, rowStride, min, max)
		return
	}
	inverseADST16Col4FlipSIMD(buf, rowStride, min, max)
}

func inverseDCT8Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct8Size || len(r1) < dct8Size || !simdColEnvelope(min, max) {
		inverseDCT8Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT8Row2SIMD(r0[:dct8Size], r1[:dct8Size], min, max)
}

func inverseDCT16Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size || !simdColEnvelope(min, max) {
		inverseDCT16Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT16Row2SIMD(r0[:dct16Size], r1[:dct16Size], min, max)
}

func inverseADST16Row4SIMDAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst16Size || len(r1) < adst16Size || len(r2) < adst16Size || len(r3) < adst16Size ||
		!simdColEnvelope(min, max) {
		inverseADST16Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	inverseADST16Row4SIMD(r0[:adst16Size], r1[:adst16Size], r2[:adst16Size], r3[:adst16Size], min, max)
}

func inverseADST16Row4FlipSIMDAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < adst16Size || len(r1) < adst16Size || len(r2) < adst16Size || len(r3) < adst16Size ||
		!simdColEnvelope(min, max) {
		inverseADST16Row4FlipPureGo(r0, r1, r2, r3, min, max)
		return
	}
	inverseADST16Row4FlipSIMD(r0[:adst16Size], r1[:adst16Size], r2[:adst16Size], r3[:adst16Size], min, max)
}

func inverseDCT32Col2SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 2 || len(buf) < (dct32Size-1)*rowStride+2 || !simdColEnvelope(min, max) {
		inverseDCT32Col2PureGo(buf, rowStride, min, max)
		return
	}
	inverseDCT32Col2SIMD(buf, rowStride, min, max)
}

func inverseDCT32Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct32Size-1)*rowStride+4 || !simdColEnvelope(min, max) {
		inverseDCT32Col2SIMDAdapter(buf, rowStride, min, max)
		inverseDCT32Col2SIMDAdapter(buf[2:], rowStride, min, max)
		return
	}
	inverseDCT32Col4SIMD(buf, rowStride, min, max)
}

func inverseDCT32Low8Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	// Replacing omitted input rows with zero also requires the column clamp to
	// preserve zero; use the full kernel for intervals wholly above or below it.
	if rowStride < 4 || len(buf) < (dct32Size-1)*rowStride+4 || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT32Col4SIMDAdapter(buf, rowStride, min, max)
		return
	}
	inverseDCT32Col4Low8SIMD(buf, rowStride, min, max)
}

func inverseDCT32Row2SIMDAdapter(r0, r1 []int32, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || !simdColEnvelope(min, max) {
		inverseDCT32Row2PureGo(r0, r1, min, max)
		return
	}
	inverseDCT32Row2SIMD(r0[:dct32Size], r1[:dct32Size], min, max)
}

func inverseDCT32Row4SIMDAdapter(r0, r1, r2, r3 []int32, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || len(r2) < dct32Size || len(r3) < dct32Size ||
		!simdColEnvelope(min, max) {
		inverseDCT32Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	inverseDCT32Row4SIMD(r0[:dct32Size], r1[:dct32Size], r2[:dct32Size], r3[:dct32Size], min, max)
}

func inverseDCT32Low16Col4SIMDAdapter(buf []int32, rowStride int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct32Size-1)*rowStride+4 || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT32Col4SIMDAdapter(buf, rowStride, min, max)
		return
	}
	inverseDCT32Col4Low16SIMD(buf, rowStride, min, max)
}

func inverseDCT32Row4LowSIMDAdapter(r0, r1, r2, r3 []int32, lowN int, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || len(r2) < dct32Size || len(r3) < dct32Size ||
		!simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT32Row4PureGo(r0, r1, r2, r3, min, max)
		return
	}
	if lowN <= 8 {
		inverseDCT32Row4Low8SIMD(r0[:dct32Size], r1[:dct32Size], r2[:dct32Size], r3[:dct32Size], min, max)
		return
	}
	inverseDCT32Row4Low16SIMD(r0[:dct32Size], r1[:dct32Size], r2[:dct32Size], r3[:dct32Size], min, max)
}

func inverseDCT32Row2LowSIMDAdapter(r0, r1 []int32, lowN int, min, max int32) {
	if len(r0) < dct32Size || len(r1) < dct32Size || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT32Row2PureGo(r0, r1, min, max)
		return
	}
	if lowN <= 8 {
		inverseDCT32Row2Low8SIMD(r0[:dct32Size], r1[:dct32Size], min, max)
		return
	}
	inverseDCT32Row2Low16SIMD(r0[:dct32Size], r1[:dct32Size], min, max)
}

func inverseDCT16Row2LowSIMDAdapter(r0, r1 []int32, lowN int, min, max int32) {
	if len(r0) < dct16Size || len(r1) < dct16Size || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT16Row2SIMDAdapter(r0, r1, min, max)
		return
	}
	if lowN <= 4 {
		inverseDCT16Row2Low4SIMD(r0[:dct16Size], r1[:dct16Size], min, max)
		return
	}
	inverseDCT16Row2Low8SIMD(r0[:dct16Size], r1[:dct16Size], min, max)
}

func inverseDCT16LowCol4SIMDAdapter(buf []int32, rowStride int, lowN int, min, max int32) {
	if rowStride < 4 || len(buf) < (dct16Size-1)*rowStride+4 || !simdColEnvelope(min, max) || min > 0 || max < 0 {
		inverseDCT16Col4SIMDAdapter(buf, rowStride, min, max)
		return
	}
	if lowN <= 4 {
		inverseDCT16Col4Low4SIMD(buf, rowStride, min, max)
		return
	}
	inverseDCT16Col4Low8SIMD(buf, rowStride, min, max)
}
