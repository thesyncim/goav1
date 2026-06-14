package motion

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// TestCompoundRound0 verifies the round_0 selection ported from libaom's
// get_conv_params_no_round() (av1/common/convolve.c). round_0 = ROUND0_BITS
// (3) for bit depths <= 10. At bd == 12 the highbd intbufrange adjustment
// (intbufrange = bd + FILTER_BITS - round_0 + 2 > 16) raises round_0 to 5.
func TestCompoundRound0(t *testing.T) {
	cases := []struct {
		bitDepth uint8
		want     int
	}{
		// intbufrange = bd + 7 - 3 + 2 = bd + 6.
		{8, 3},  // intbufrange = 14, not > 16.
		{10, 3}, // intbufrange = 16, not > 16.
		{12, 5}, // intbufrange = 18 > 16 => round0 += 2.
	}
	for _, tc := range cases {
		if got := compoundRound0(tc.bitDepth); got != tc.want {
			t.Errorf("compoundRound0(%d)=%d want %d", tc.bitDepth, got, tc.want)
		}
	}
}

// TestRoundPowerOfTwoCompound verifies the rounding helper used by the d16
// mask subsampling path (matches libaom's ROUND_POWER_OF_TWO macro).
func TestRoundPowerOfTwoCompound(t *testing.T) {
	cases := []struct {
		value, bits, want int
	}{
		{0, 2, 0},
		{1, 2, 0},    // (1 + 2) >> 2 = 0
		{2, 2, 1},    // (2 + 2) >> 2 = 1
		{100, 2, 25}, // (100 + 2) >> 2 = 25
		{7, 1, 4},    // (7 + 1) >> 1 = 4
	}
	for _, tc := range cases {
		if got := roundPowerOfTwoCompound(tc.value, tc.bits); got != tc.want {
			t.Errorf("roundPowerOfTwoCompound(%d,%d)=%d want %d", tc.value, tc.bits, got, tc.want)
		}
	}
}

// TestCompoundMaskSample verifies the (possibly subsampled) A64 mask read that
// mirrors aom_*_blend_a64_d16_mask_c subw/subh handling.
func TestCompoundMaskSample(t *testing.T) {
	// 4x4 mask. Lay out so subsampling reductions are easy to verify.
	stride := 4
	mask := []byte{
		10, 21, 30, 40,
		20, 30, 50, 60,
		1, 2, 3, 4,
		5, 6, 7, 8,
	}

	// No subsampling: direct read.
	if v, ok := compoundMaskSample(mask, stride, 1, 2, false, false); !ok || v != 50 {
		t.Errorf("noSub mask[1][2]=%d ok=%v want 50", v, ok)
	}
	// subX only: (a + b + 1) >> 1 over horizontally adjacent pair at row*stride+2*col.
	// row=0 col=0 -> (10 + 21 + 1) >> 1 = 16.
	if v, ok := compoundMaskSample(mask, stride, 0, 0, true, false); !ok || v != 16 {
		t.Errorf("subX mask[0][0]=%d ok=%v want 16", v, ok)
	}
	// subY only: (top + bottom + 1) >> 1 over vertically adjacent rows.
	// row=0 col=0 -> (mask[0][0]=10 + mask[1][0]=20 + 1) >> 1 = 15.
	if v, ok := compoundMaskSample(mask, stride, 0, 0, false, true); !ok || v != 15 {
		t.Errorf("subY mask[0][0]=%d ok=%v want 15", v, ok)
	}
	// subX && subY: ROUND_POWER_OF_TWO(sum of 2x2, 2).
	// row=0 col=0 -> sum(10,21,20,30)=81 -> (81 + 2) >> 2 = 20.
	if v, ok := compoundMaskSample(mask, stride, 0, 0, true, true); !ok || v != 20 {
		t.Errorf("subXY mask[0][0]=%d ok=%v want 20", v, ok)
	}
	// Out-of-bounds index returns ok=false.
	if _, ok := compoundMaskSample(mask, stride, 3, 3, false, true); ok {
		t.Errorf("subY out-of-range expected ok=false")
	}
}

// flatCompoundBuf builds a CompoundConvBuf of the given size filled with a
// single CONV_BUF value.
func flatCompoundBuf(width, height int, value uint16) *CompoundConvBuf {
	buf := &CompoundConvBuf{Width: uint8(width), Height: uint8(height)}
	for i := 0; i < width*height; i++ {
		buf.Data[i] = value
	}
	return buf
}

// convValForPixel returns the CONV_BUF d16 value whose equal-weight blend
// (fwd==bck==8) inverts exactly to pixel P at the given bit depth. With
// fwd==bck==8 the BlendCompoundAvg arithmetic is:
//
//	tmp = v*8 + v*8 = v*16; tmp >>= 4 (=> v); tmp -= roundOffset
//	out = ROUND_POWER_OF_TWO(v - roundOffset, roundBits)
//
// so v = (P << roundBits) + roundOffset yields out == P exactly.
func convValForPixel(p int, bitDepth uint8) uint16 {
	round0 := compoundRound0(bitDepth)
	offsetBits := int(bitDepth) + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	roundBits := 2*filterBits - round0 - compoundRound1Bits
	return uint16((p << roundBits) + roundOffset)
}

func readPixel8(dst frame.Plane, x, y int) int { return int(dst.Pix[y*dst.Stride+x]) }
func readPixel16(dst frame.Plane, x, y int) int {
	off := y*dst.Stride + x*2
	return int(dst.Pix[off]) | int(dst.Pix[off+1])<<8
}

// TestBlendCompoundAvg verifies the distance-weighted compound average
// (av1_dist_wtd_convolve_* do_average blend). With equal forward/backward
// offsets and equal predictors, the output must invert exactly to the seeded
// pixel value at each bit depth.
func TestBlendCompoundAvg(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		bps      int
		pixel    int
	}{
		{"bd8", 8, 1, 100},
		{"bd10", 10, 2, 600},
		{"bd12", 12, 2, 2500},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 4, 4
			v := convValForPixel(tc.pixel, tc.bitDepth)
			buf0 := flatCompoundBuf(w, h, v)
			buf1 := flatCompoundBuf(w, h, v)
			dst := frame.Plane{Pix: make([]byte, w*h*tc.bps), Stride: w * tc.bps, Width: w, Height: h}
			// fwdOffset + bckOffset must sum to 16 (DIST_PRECISION); equal split.
			if err := BlendCompoundAvg(dst, buf0, buf1, tc.bps, tc.bitDepth, 0, 0, w, h, 8, 8); err != nil {
				t.Fatalf("BlendCompoundAvg: %v", err)
			}
			for y := range h {
				for x := range w {
					var got int
					if tc.bps == 1 {
						got = readPixel8(dst, x, y)
					} else {
						got = readPixel16(dst, x, y)
					}
					if got != tc.pixel {
						t.Fatalf("pixel(%d,%d)=%d want %d", x, y, got, tc.pixel)
					}
				}
			}
		})
	}
}

// TestBlendCompoundAvgErrors verifies the validation guards reject mismatched
// buffer dimensions and out-of-range destination regions.
func TestBlendCompoundAvgErrors(t *testing.T) {
	buf0 := flatCompoundBuf(4, 4, 1000)
	buf1 := flatCompoundBuf(4, 4, 1000)
	dst := frame.Plane{Pix: make([]byte, 16), Stride: 4, Width: 4, Height: 4}

	if err := BlendCompoundAvg(dst, nil, buf1, 1, 8, 0, 0, 4, 4, 8, 8); err != ErrInvalidMotion {
		t.Errorf("nil buf0 err=%v want ErrInvalidMotion", err)
	}
	mismatched := flatCompoundBuf(2, 2, 1000)
	if err := BlendCompoundAvg(dst, mismatched, buf1, 1, 8, 0, 0, 4, 4, 8, 8); err != ErrInvalidMotion {
		t.Errorf("dim mismatch err=%v want ErrInvalidMotion", err)
	}
	if err := BlendCompoundAvg(dst, buf0, buf1, 1, 8, 2, 2, 4, 4, 8, 8); err != ErrInvalidMotion {
		t.Errorf("region overflow err=%v want ErrInvalidMotion", err)
	}
}

// TestBlendCompoundMaskD16 verifies the soft-mask d16 blend
// (aom_blend_a64_d16_mask_c). A full mask (m=64) selects src0 entirely; an
// empty mask (m=0) selects src1 entirely. With the CONV_BUF value chosen to
// invert to a known pixel, the selected predictor's pixel must appear.
func TestBlendCompoundMaskD16(t *testing.T) {
	const w, h = 4, 4
	const bd = uint8(8)
	pixel := 100
	v := convValForPixel(pixel, bd)
	buf0 := flatCompoundBuf(w, h, v)
	buf1 := flatCompoundBuf(w, h, 0)

	// Full mask selects src0.
	maskFull := make([]byte, w*h)
	for i := range maskFull {
		maskFull[i] = compoundBlendA64MaxAlpha
	}
	dst := frame.Plane{Pix: make([]byte, w*h), Stride: w, Width: w, Height: h}
	if err := BlendCompoundMaskD16(dst, buf0, buf1, 1, bd, 0, 0, w, h, maskFull, w, false, false); err != nil {
		t.Fatalf("BlendCompoundMaskD16 full: %v", err)
	}
	if got := readPixel8(dst, 1, 1); got != pixel {
		t.Errorf("full-mask pixel=%d want %d", got, pixel)
	}

	// Empty mask selects src1; swap which buffer carries v.
	buf0b := flatCompoundBuf(w, h, 0)
	buf1b := flatCompoundBuf(w, h, v)
	maskEmpty := make([]byte, w*h) // all zero
	dst2 := frame.Plane{Pix: make([]byte, w*h), Stride: w, Width: w, Height: h}
	if err := BlendCompoundMaskD16(dst2, buf0b, buf1b, 1, bd, 0, 0, w, h, maskEmpty, w, false, false); err != nil {
		t.Fatalf("BlendCompoundMaskD16 empty: %v", err)
	}
	if got := readPixel8(dst2, 1, 1); got != pixel {
		t.Errorf("empty-mask pixel=%d want %d", got, pixel)
	}
}

// TestBuildDiffWtdMaskD16 verifies the difference-weighted mask construction
// (av1_build_compound_diffwtd_mask_d16_c). For bd=8 the round shift is
// 2*7 - 3 - 7 + 0 = 4, then m = 38 + ROUND_POWER_OF_TWO(|s0-s1|, 4) / 16,
// clamped to [0, 64].
func TestBuildDiffWtdMaskD16(t *testing.T) {
	const w, h = 2, 2
	const bd = uint8(8)
	maskStride := w

	cases := []struct {
		name   string
		s0, s1 uint16
		invert bool
		want   byte
	}{
		// diff=0 -> rpot(0,4)=0 -> m=38.
		{"equal", 1000, 1000, false, 38},
		// diff=160 -> rpot(160,4)=10 -> m=38+0=38 (10/16=0).
		{"smalldiff", 1160, 1000, false, 38},
		// diff=16000 -> rpot=1000 -> m=38+62=100 clamp 64.
		{"saturate", 17000, 1000, false, 64},
		// invert of equal: 64-38 = 26.
		{"equal-inverted", 1000, 1000, true, 26},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			buf0 := flatCompoundBuf(w, h, tc.s0)
			buf1 := flatCompoundBuf(w, h, tc.s1)
			mask := make([]byte, w*h)
			if err := BuildDiffWtdMaskD16(mask, maskStride, buf0, buf1, bd, w, h, tc.invert); err != nil {
				t.Fatalf("BuildDiffWtdMaskD16: %v", err)
			}
			for i, m := range mask {
				if m != tc.want {
					t.Fatalf("mask[%d]=%d want %d", i, m, tc.want)
				}
			}
		})
	}
}

// TestBuildDiffWtdMaskD16Errors verifies the validation guards reject
// undersized mask buffers and mismatched predictor dimensions.
func TestBuildDiffWtdMaskD16Errors(t *testing.T) {
	buf0 := flatCompoundBuf(4, 4, 1000)
	buf1 := flatCompoundBuf(4, 4, 2000)
	short := make([]byte, 4) // too small for 4x4
	if err := BuildDiffWtdMaskD16(short, 4, buf0, buf1, 8, 4, 4, false); err != ErrInvalidMotion {
		t.Errorf("short mask err=%v want ErrInvalidMotion", err)
	}
	mismatched := flatCompoundBuf(2, 2, 1000)
	mask := make([]byte, 16)
	if err := BuildDiffWtdMaskD16(mask, 4, mismatched, buf1, 8, 4, 4, false); err != ErrInvalidMotion {
		t.Errorf("dim mismatch err=%v want ErrInvalidMotion", err)
	}
}

// TestPredictInterCompoundRefToConvBufCopyRoundTrip exercises the integer
// (subX==0, subY==0) compound convolve path (av1_dist_wtd_convolve_2d_copy):
// each CONV_BUF sample is (src << bits) + roundOffset. Feeding two identical
// copy predictors through the equal-weight BlendCompoundAvg must reproduce the
// source pixels exactly, validating that the convolve-to-convbuf scaling and
// the blend-back rounding are mutually consistent across bit depths. This is
// the same precision contract libaom relies on: keep the predictor un-rounded
// at d16 precision, round once after combining.
func TestPredictInterCompoundRefToConvBufCopyRoundTrip(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		bps      int
	}{
		{"bd8", 8, 1},
		{"bd10", 10, 2},
		{"bd12", 12, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 8, 8
			// Build a ref plane with a per-pixel gradient (kept within the bit
			// depth range). loadSample*Clamped clamps to plane bounds, and the
			// integer-MV (subX=subY=0) copy reads exactly (refX+x, refY+y), so
			// place the block at (0,0) within an exactly-sized plane.
			ref := frame.Plane{Pix: make([]byte, w*h*tc.bps), Stride: w * tc.bps, Width: w, Height: h}
			want := make([]int, w*h)
			for y := range h {
				for x := range w {
					p := (x*7 + y*11) & ((1 << tc.bitDepth) - 1)
					want[y*w+x] = p
					if tc.bps == 1 {
						ref.Pix[y*ref.Stride+x] = byte(p)
					} else {
						off := y*ref.Stride + x*2
						ref.Pix[off] = byte(p)
						ref.Pix[off+1] = byte(p >> 8)
					}
				}
			}

			var buf0, buf1 CompoundConvBuf
			if err := PredictInterCompoundRefToConvBuf(&buf0, ref, tc.bps, tc.bitDepth, 0, 0, w, h, 0, 0, RegularFilters); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf buf0: %v", err)
			}
			if err := PredictInterCompoundRefToConvBuf(&buf1, ref, tc.bps, tc.bitDepth, 0, 0, w, h, 0, 0, RegularFilters); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf buf1: %v", err)
			}

			dst := frame.Plane{Pix: make([]byte, w*h*tc.bps), Stride: w * tc.bps, Width: w, Height: h}
			if err := BlendCompoundAvg(dst, &buf0, &buf1, tc.bps, tc.bitDepth, 0, 0, w, h, 8, 8); err != nil {
				t.Fatalf("BlendCompoundAvg: %v", err)
			}
			for y := range h {
				for x := range w {
					var got int
					if tc.bps == 1 {
						got = readPixel8(dst, x, y)
					} else {
						got = readPixel16(dst, x, y)
					}
					if got != want[y*w+x] {
						t.Fatalf("pixel(%d,%d)=%d want %d", x, y, got, want[y*w+x])
					}
				}
			}
		})
	}
}

func TestPredictInterCompoundRefToConvBuf8OptimizedMatchesReference(t *testing.T) {
	const (
		refW   = 23
		refH   = 19
		stride = 32
	)
	ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
	for y := range refH {
		for x := range refW {
			ref.Pix[y*stride+x] = byte((x*37 + y*53 + x*y*3 + 11) & 0xff)
		}
	}

	cases := []struct {
		name          string
		refX, refY    int
		width, height int
		subX, subY    int
		filters       InterpFilters
	}{
		{"copy interior", 3, 4, 16, 8, 0, 0, RegularFilters},
		{"copy width4 interior", 7, 6, 4, 8, 0, 0, RegularFilters},
		{"copy edge", -2, 16, 16, 8, 0, 0, RegularFilters},
		{"x regular interior", 2, 4, 16, 8, 5, 0, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"x width4 interior", 7, 6, 4, 8, 5, 0, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"x smooth right edge", 13, 5, 8, 8, 7, 0, InterpFilters{X: InterpEightTapSmooth, Y: InterpEightTapRegular}},
		{"y sharp top edge", 4, -3, 16, 8, 0, 11, InterpFilters{X: InterpEightTapRegular, Y: InterpMultiTapSharp}},
		{"y width4 interior", 7, 6, 4, 8, 0, 11, InterpFilters{X: InterpEightTapRegular, Y: InterpMultiTapSharp}},
		{"2d regular interior", 4, 4, 16, 16, 3, 9, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}},
		{"2d width4 interior", 7, 6, 4, 8, 3, 9, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}},
		{"2d fourtap left edge", -2, 3, 4, 16, 4, 12, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"2d bilinear bottom edge", 9, 14, 8, 4, 8, 8, InterpFilters{X: InterpBilinear, Y: InterpBilinear}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got, gotScratch CompoundConvBuf
			if err := PredictInterCompoundRefToConvBuf(&got, ref, 1, 8, tc.refX, tc.refY, tc.width, tc.height, tc.subX, tc.subY, tc.filters); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf: %v", err)
			}
			var scratch CompoundConvolveScratch
			if err := PredictInterCompoundRefToConvBufWithScratch(&gotScratch, ref, 1, 8, tc.refX, tc.refY, tc.width, tc.height, tc.subX, tc.subY, tc.filters, &scratch); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBufWithScratch: %v", err)
			}
			want, err := referenceCompoundConvBuf8(ref, tc.refX, tc.refY, tc.width, tc.height, tc.subX, tc.subY, tc.filters)
			if err != nil {
				t.Fatalf("reference: %v", err)
			}
			if int(got.Width) != tc.width || int(got.Height) != tc.height || int(gotScratch.Width) != tc.width || int(gotScratch.Height) != tc.height {
				t.Fatalf("dims default=%dx%d scratch=%dx%d want %dx%d", got.Width, got.Height, gotScratch.Width, gotScratch.Height, tc.width, tc.height)
			}
			for i := range tc.width * tc.height {
				if got.Data[i] != want[i] {
					t.Fatalf("convbuf default[%d]=%d want %d", i, got.Data[i], want[i])
				}
				if gotScratch.Data[i] != want[i] {
					t.Fatalf("convbuf scratch[%d]=%d want %d", i, gotScratch.Data[i], want[i])
				}
			}
		})
	}
}

func TestPredictInterCompoundRefToConvBufHighBDOptimizedMatchesClamped(t *testing.T) {
	const (
		refW   = 37
		refH   = 31
		stride = 96
	)
	cases := []struct {
		name          string
		refX, refY    int
		width, height int
		subX, subY    int
		filters       InterpFilters
	}{
		{"copy interior", 5, 6, 16, 8, 0, 0, RegularFilters},
		{"copy edge", -1, 27, 16, 4, 0, 0, RegularFilters},
		{"x regular interior", 5, 6, 16, 8, 5, 0, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
		{"x smooth edge", 28, 8, 8, 8, 7, 0, InterpFilters{X: InterpEightTapSmooth, Y: InterpEightTapRegular}},
		{"y sharp interior", 6, 6, 16, 8, 0, 11, InterpFilters{X: InterpEightTapRegular, Y: InterpMultiTapSharp}},
		{"y bilinear edge", 4, -2, 16, 8, 0, 8, InterpFilters{X: InterpEightTapRegular, Y: InterpBilinear}},
		{"2d regular interior", 6, 6, 16, 16, 3, 9, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapSmooth}},
		{"2d fourtap edge", -2, 3, 4, 16, 4, 12, InterpFilters{X: InterpEightTapRegular, Y: InterpEightTapRegular}},
	}
	for _, bitDepth := range []uint8{10, 12} {
		ref := frame.Plane{Pix: make([]byte, stride*refH), Stride: stride, Width: refW, Height: refH}
		mask := (1 << bitDepth) - 1
		for y := range refH {
			for x := range refW {
				p := (x*43 + y*59 + x*y*5 + 17) & mask
				off := y*stride + x*2
				ref.Pix[off] = byte(p)
				ref.Pix[off+1] = byte(p >> 8)
			}
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				var got, gotScratch, want CompoundConvBuf
				if err := PredictInterCompoundRefToConvBuf(&got, ref, 2, bitDepth, tc.refX, tc.refY, tc.width, tc.height, tc.subX, tc.subY, tc.filters); err != nil {
					t.Fatalf("PredictInterCompoundRefToConvBuf: %v", err)
				}
				var scratch CompoundConvolveScratch
				if err := PredictInterCompoundRefToConvBufWithScratch(&gotScratch, ref, 2, bitDepth, tc.refX, tc.refY, tc.width, tc.height, tc.subX, tc.subY, tc.filters, &scratch); err != nil {
					t.Fatalf("PredictInterCompoundRefToConvBufWithScratch: %v", err)
				}
				wantOut, ok := compoundConvBufView(&want, tc.width, tc.height)
				if !ok {
					t.Fatal("invalid reference convbuf dims")
				}
				xKernel, err := interpKernel(tc.filters.X, tc.width, tc.subX)
				if err != nil {
					t.Fatalf("x kernel: %v", err)
				}
				yKernel, err := interpKernel(tc.filters.Y, tc.height, tc.subY)
				if err != nil {
					t.Fatalf("y kernel: %v", err)
				}
				round0 := compoundRound0(bitDepth)
				offsetBits := int(bitDepth) + 2*filterBits - round0
				roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
				switch {
				case tc.subX != 0 && tc.subY != 0:
					var im compoundIM
					predictInterCompoundRefHighBDToConvBuf2DClamped(wantOut, ref, tc.refX, tc.refY, tc.width, tc.height, xKernel, yKernel, round0, offsetBits, int(bitDepth), &im)
				case tc.subX != 0:
					predictInterCompoundRefHighBDToConvBufXClamped(wantOut, ref, tc.refX, tc.refY, tc.width, tc.height, xKernel, round0, roundOffset)
				case tc.subY != 0:
					predictInterCompoundRefHighBDToConvBufYClamped(wantOut, ref, tc.refX, tc.refY, tc.width, tc.height, yKernel, round0, roundOffset)
				default:
					predictInterCompoundRefHighBDToConvBufCopyClamped(wantOut, ref, tc.refX, tc.refY, tc.width, tc.height, round0, roundOffset)
				}
				if int(got.Width) != tc.width || int(got.Height) != tc.height || int(gotScratch.Width) != tc.width || int(gotScratch.Height) != tc.height {
					t.Fatalf("dims default=%dx%d scratch=%dx%d want %dx%d", got.Width, got.Height, gotScratch.Width, gotScratch.Height, tc.width, tc.height)
				}
				for i := range tc.width * tc.height {
					if got.Data[i] != want.Data[i] {
						t.Fatalf("convbuf default[%d]=%d want %d", i, got.Data[i], want.Data[i])
					}
					if gotScratch.Data[i] != want.Data[i] {
						t.Fatalf("convbuf scratch[%d]=%d want %d", i, gotScratch.Data[i], want.Data[i])
					}
				}
			})
		}
	}
}

func referenceCompoundConvBuf8(ref frame.Plane, refX int, refY int, width int, height int, subX int, subY int, filters InterpFilters) ([]uint16, error) {
	xKernel, err := interpKernel(filters.X, width, subX)
	if err != nil {
		return nil, err
	}
	yKernel, err := interpKernel(filters.Y, height, subY)
	if err != nil {
		return nil, err
	}
	out := make([]uint16, width*height)
	foX := filterTaps/2 - 1
	foY := filterTaps/2 - 1
	round0 := compoundRound0(8)
	offsetBits := 8 + 2*filterBits - round0
	roundOffset := (1 << (offsetBits - compoundRound1Bits)) + (1 << (offsetBits - compoundRound1Bits - 1))
	load := func(x, y int) int {
		return int(loadSample8Clamped(ref, x, y))
	}
	switch {
	case subX != 0 && subY != 0:
		const imStride = maxBlockSize
		var im [((maxBlockSize + filterTaps - 1) * maxBlockSize)]int32
		imH := height + filterTaps - 1
		for y := range imH {
			for x := range width {
				sum := 1 << (8 + filterBits - 1)
				for k := range filterTaps {
					sum += int(xKernel[k]) * load(refX+x-foX+k, refY-foY+y)
				}
				im[y*imStride+x] = int32(roundPowerOfTwo(sum, round0))
			}
		}
		for y := range height {
			for x := range width {
				sum := 1 << offsetBits
				for k := range filterTaps {
					sum += int(yKernel[k]) * int(im[(y+k)*imStride+x])
				}
				out[y*width+x] = uint16(roundPowerOfTwo(sum, compoundRound1Bits))
			}
		}
	case subX != 0:
		bits := filterBits - compoundRound1Bits
		for y := range height {
			for x := range width {
				res := 0
				for k := range filterTaps {
					res += int(xKernel[k]) * load(refX+x-foX+k, refY+y)
				}
				res = (1 << bits) * roundPowerOfTwo(res, round0)
				res += roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	case subY != 0:
		bits := filterBits - round0
		for y := range height {
			for x := range width {
				res := 0
				for k := range filterTaps {
					res += int(yKernel[k]) * load(refX+x, refY+y-foY+k)
				}
				res *= 1 << bits
				res = roundPowerOfTwo(res, compoundRound1Bits) + roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	default:
		bits := 2*filterBits - compoundRound1Bits - round0
		for y := range height {
			for x := range width {
				res := load(refX+x, refY+y) << bits
				res += roundOffset
				out[y*width+x] = uint16(res)
			}
		}
	}
	return out, nil
}

func TestPredictScaledCompoundRefToConvBufIdentityMatchesUnscaled(t *testing.T) {
	cases := []struct {
		name     string
		bitDepth uint8
		bps      int
	}{
		{"bd8", 8, 1},
		{"bd10", 10, 2},
		{"bd12", 12, 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const w, h = 16, 16
			ref := frame.Plane{Pix: make([]byte, w*h*tc.bps), Stride: w * tc.bps, Width: w, Height: h}
			mask := (1 << tc.bitDepth) - 1
			for y := range h {
				for x := range w {
					p := (x*17 + y*29 + 11) & mask
					if tc.bps == 1 {
						ref.Pix[y*ref.Stride+x] = byte(p)
					} else {
						off := y*ref.Stride + x*2
						ref.Pix[off] = byte(p)
						ref.Pix[off+1] = byte(p >> 8)
					}
				}
			}

			sf, err := NewScaleFactors(w, h, w, h)
			if err != nil {
				t.Fatal(err)
			}
			startX, startY, xStep, yStep, err := sf.ScaledBlockOrigin(0, 0, Vector{}, false, false)
			if err != nil {
				t.Fatal(err)
			}
			xTable, err := SubpelKernelTableFor(InterpEightTapRegular, w)
			if err != nil {
				t.Fatal(err)
			}
			yTable, err := SubpelKernelTableFor(InterpEightTapRegular, h)
			if err != nil {
				t.Fatal(err)
			}

			var scaled, unscaled CompoundConvBuf
			if err := PredictScaledCompoundRefToConvBuf(&scaled, ref, tc.bps, tc.bitDepth, w, h, startX, xStep, startY, yStep, xTable, yTable); err != nil {
				t.Fatalf("PredictScaledCompoundRefToConvBuf: %v", err)
			}
			if err := PredictInterCompoundRefToConvBuf(&unscaled, ref, tc.bps, tc.bitDepth, 0, 0, w, h, 0, 0, RegularFilters); err != nil {
				t.Fatalf("PredictInterCompoundRefToConvBuf: %v", err)
			}
			if scaled.Width != unscaled.Width || scaled.Height != unscaled.Height {
				t.Fatalf("dims got=%dx%d want=%dx%d", scaled.Width, scaled.Height, unscaled.Width, unscaled.Height)
			}
			for i := range w * h {
				if scaled.Data[i] != unscaled.Data[i] {
					t.Fatalf("convbuf[%d]=%d want %d", i, scaled.Data[i], unscaled.Data[i])
				}
			}
		})
	}
}

// TestPredictInterCompoundRefToConvBufErrors verifies the parameter guards.
func TestPredictInterCompoundRefToConvBufErrors(t *testing.T) {
	ref := frame.Plane{Pix: make([]byte, 64), Stride: 8, Width: 8, Height: 8}
	var buf CompoundConvBuf
	// nil buffer.
	if err := PredictInterCompoundRefToConvBuf(nil, ref, 1, 8, 0, 0, 8, 8, 0, 0, RegularFilters); err != ErrInvalidMotion {
		t.Errorf("nil buf err=%v want ErrInvalidMotion", err)
	}
	// Out-of-range subpel position.
	if err := PredictInterCompoundRefToConvBuf(&buf, ref, 1, 8, 0, 0, 8, 8, -1, 0, RegularFilters); err != ErrInvalidMotion {
		t.Errorf("bad subX err=%v want ErrInvalidMotion", err)
	}
	// Bit-depth / sample-width mismatch (8-bit declared but 2 bytes/sample).
	if err := PredictInterCompoundRefToConvBuf(&buf, ref, 2, 8, 0, 0, 8, 8, 0, 0, RegularFilters); err != ErrInvalidMotion {
		t.Errorf("bps mismatch err=%v want ErrInvalidMotion", err)
	}
}
