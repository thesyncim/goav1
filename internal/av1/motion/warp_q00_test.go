package motion

import (
	"bytes"
	"compress/zlib"
	"io"
	"os"
	"testing"
)

// libaomQ00WarpRefBin returns the captured Y-plane of libaom's reconstruction
// of quantizer_00 frame 0, which serves as the reference frame for frame 1.
// The bytes are read from testdata/warp_q00_ref.bin.zlib so the snapshot
// stays out of the binary even though the warp regression below needs to
// drive a full 352x288 reference window through PredictWarpedPlaneBlockBitDepth.
func libaomQ00WarpRefBin(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/warp_q00_ref.bin.zlib")
	if err != nil {
		t.Fatalf("read testdata: %v", err)
	}
	zr, err := zlib.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("zlib reader: %v", err)
	}
	defer zr.Close()
	out, err := io.ReadAll(zr)
	if err != nil {
		t.Fatalf("zlib decompress: %v", err)
	}
	const expected = 352 * 288
	if len(out) != expected {
		t.Fatalf("snapshot size = %d want %d", len(out), expected)
	}
	return out
}

// TestPredictWarpedPlaneBlockQ00Frame1MI80x4 pins our warpAffine8 dispatch
// against av1_warp_affine_c() on the first WARPED_CAUSAL block of libaom's
// quantizer_00 frame 1 (mi=(80,4), 8x16 luma, MotionMode=WARPED_CAUSAL with
// MV (-1,-7) sampling frame 0). The warp inputs are derived from a real
// bitstream decode pass and the expected output bytes were produced by
// invoking av1_warp_affine_c() directly with the same inputs.
//
// This pins both the shear-parameter dispatch (alpha=-320, beta=-64,
// gamma=0, delta=0) and the filter table indexing for a block whose
// initial sub-block lands at the upper-left corner with no edge clamping,
// catching ±1 rounding drift in the horizontal/vertical reduce stages.
func TestPredictWarpedPlaneBlockQ00Frame1MI80x4(t *testing.T) {
	refBytes := libaomQ00WarpRefBin(t)
	refPlane, _ := testPlane(352, 288, 1, 352)
	copy(refPlane.Pix, refBytes)

	dst, _ := testPlane(352, 288, 1, 352)

	matrix := [6]int32{42805, -7571, 65230, -57, 0, 65509}
	alpha := int16(-320)
	beta := int16(-64)
	gamma := int16(0)
	delta := int16(0)

	if err := PredictWarpedPlaneBlockBitDepth(dst, refPlane, 1, 8, 320, 16, 8, 16,
		matrix, alpha, beta, gamma, delta, false, false); err != nil {
		t.Fatalf("PredictWarpedPlaneBlockBitDepth: %v", err)
	}

	// Expected bytes produced by av1_warp_affine_c() (libaom v3.13.1) with the
	// inputs above on the same reference plane.
	want := [16][8]byte{
		{98, 110, 99, 90, 93, 91, 99, 98},
		{89, 99, 90, 82, 82, 83, 92, 90},
		{87, 97, 91, 82, 84, 87, 92, 90},
		{86, 95, 87, 78, 79, 81, 90, 90},
		{90, 99, 93, 86, 88, 89, 101, 101},
		{102, 106, 99, 95, 95, 95, 104, 99},
		{94, 98, 89, 84, 86, 84, 94, 93},
		{92, 94, 87, 83, 84, 84, 93, 94},
		{97, 93, 88, 84, 84, 84, 92, 90},
		{88, 102, 93, 80, 95, 90, 93, 102},
		{100, 117, 106, 110, 103, 101, 112, 99},
		{115, 102, 129, 134, 83, 124, 124, 78},
		{100, 107, 120, 97, 107, 121, 100, 116},
		{106, 121, 89, 93, 119, 89, 109, 131},
		{117, 102, 103, 121, 88, 104, 125, 91},
		{111, 110, 121, 118, 114, 114, 104, 106},
	}

	for y := 0; y < 16; y++ {
		for x := 0; x < 8; x++ {
			got := dst.Pix[(16+y)*dst.Stride+(320+x)]
			if got != want[y][x] {
				t.Fatalf("dst[%d,%d] = %d want %d (libaom av1_warp_affine_c)",
					320+x, 16+y, got, want[y][x])
			}
		}
	}
}
