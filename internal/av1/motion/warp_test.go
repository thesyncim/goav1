package motion

import "testing"

func TestPredictWarpedPlaneBlockIdentityConstant8Bit(t *testing.T) {
	ref, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	for i := range ref.Pix {
		ref.Pix[i] = 91
	}
	matrix := [6]int32{0, 0, warpedModelOne, 0, 0, warpedModelOne}
	if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 1, 8, 8, 8, 16, 16, matrix, 0, 0, 0, 0, false, false); err != nil {
		t.Fatal(err)
	}
	for y := 8; y < 24; y++ {
		for x := 8; x < 24; x++ {
			if got := dst.Pix[y*dst.Stride+x]; got != 91 {
				t.Fatalf("sample(%d,%d)=%d want 91", x, y, got)
			}
		}
	}
}

func TestPredictWarpedPlaneBlockIdentityConstantHighBD(t *testing.T) {
	ref, _ := testPlane(32, 32, 2, 64)
	dst, _ := testPlane(32, 32, 2, 64)
	for y := 0; y < ref.Height; y++ {
		for x := 0; x < ref.Width; x++ {
			storeHighBDSample(ref, x, y, 777)
		}
	}
	matrix := [6]int32{0, 0, warpedModelOne, 0, 0, warpedModelOne}
	if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 2, 10, 8, 8, 16, 16, matrix, 0, 0, 0, 0, false, false); err != nil {
		t.Fatal(err)
	}
	for y := 8; y < 24; y++ {
		for x := 8; x < 24; x++ {
			if got := loadHighBDSample(dst, x, y); got != 777 {
				t.Fatalf("sample(%d,%d)=%d want 777", x, y, got)
			}
		}
	}
}

func TestPredictWarpedPlaneBlockAllocs(t *testing.T) {
	ref, _ := testPlane(32, 32, 1, 32)
	dst, _ := testPlane(32, 32, 1, 32)
	matrix := [6]int32{0, 0, warpedModelOne, 0, 0, warpedModelOne}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 1, 8, 8, 8, 16, 16, matrix, 0, 0, 0, 0, false, false); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("warped prediction allocated: %f", allocs)
	}
}
