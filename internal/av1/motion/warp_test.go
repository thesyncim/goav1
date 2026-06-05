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

func BenchmarkPredictWarpedPlaneBlock8Bit64(b *testing.B) {
	ref, _ := testPlane(96, 96, 1, 96)
	dst, _ := testPlane(96, 96, 1, 96)
	for y := 0; y < ref.Height; y++ {
		for x := 0; x < ref.Width; x++ {
			ref.Pix[y*ref.Stride+x] = byte((x*17 + y*29 + x*y) & 0xff)
		}
	}
	matrix := [6]int32{42805, -7571, 65230, -57, 0, 65509}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := PredictWarpedPlaneBlockBitDepth(dst, ref, 1, 8, 16, 16, 64, 64, matrix, -320, -64, 0, -64, false, false); err != nil {
			b.Fatal(err)
		}
	}
}

// FuzzPredictWarpedPlaneBlockBitDepth stresses the warped affine prediction
// dispatcher with random matrix coefficients, alpha/beta/gamma/delta scales,
// block placements, and subsampling. Out-of-range parameters are clipped to
// the contract validated by PredictWarpedPlaneBlockBitDepth so the function
// must return cleanly without panicking or writing past the destination
// region.
func FuzzPredictWarpedPlaneBlockBitDepth(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint8(0), uint8(0), uint8(0), int16(0), int16(0), int16(0), int16(0))
	f.Add(uint8(1), uint8(2), uint8(8), uint8(8), uint8(7), uint8(7), int16(64), int16(-32), int16(16), int16(-8))
	f.Add(uint8(2), uint8(5), uint8(2), uint8(3), uint8(15), uint8(7), int16(255), int16(-255), int16(127), int16(-127))
	f.Add(uint8(3), uint8(7), uint8(1), uint8(0), uint8(3), uint8(15), int16(-512), int16(256), int16(-128), int16(64))
	f.Add(uint8(4), uint8(0), uint8(0), uint8(0), uint8(0), uint8(0), int16(32767), int16(-32768), int16(32767), int16(-32768))

	f.Fuzz(func(t *testing.T, rawShape uint8, rawSubs uint8, rawDstX uint8, rawDstY uint8, rawW uint8, rawH uint8, alphaRaw int16, betaRaw int16, gammaRaw int16, deltaRaw int16) {
		bytesPerSample := 1
		bitDepth := uint8(8)
		switch rawShape & 3 {
		case 1:
			bytesPerSample = 2
			bitDepth = 10
		case 2:
			bytesPerSample = 2
			bitDepth = 12
		}

		planeW := 64
		planeH := 64
		blockW := 8 + 8*(int(rawShape>>2)&3)
		blockH := 8 + 8*(int(rawShape>>4)&3)
		if blockW > planeW {
			blockW = planeW
		}
		if blockH > planeH {
			blockH = planeH
		}
		blockW &^= 7
		blockH &^= 7
		if blockW == 0 {
			blockW = 8
		}
		if blockH == 0 {
			blockH = 8
		}

		dstX := int(rawDstX) % (planeW - blockW + 1)
		dstY := int(rawDstY) % (planeH - blockH + 1)
		dstX &^= 7
		dstY &^= 7

		stride := planeW * bytesPerSample
		ref, _ := testPlane(planeW, planeH, bytesPerSample, stride)
		dst, _ := testPlane(planeW, planeH, bytesPerSample, stride)

		maxSample := uint16((1 << bitDepth) - 1)
		for y := 0; y < ref.Height; y++ {
			for x := 0; x < ref.Width; x++ {
				v := uint16((x*7 + y*11) & 0xff)
				if bytesPerSample == 2 {
					v = uint16((x*13+y*17)&0x3ff) & maxSample
				}
				if bytesPerSample == 1 {
					ref.Pix[y*ref.Stride+x] = byte(v)
				} else {
					storeHighBDSample(ref, x, y, v)
				}
			}
		}

		const warpRange = warpedModelOne / 2
		matrix := [6]int32{
			int32((int(alphaRaw) % (planeW + 1)) << 4),
			int32((int(betaRaw) % (planeH + 1)) << 4),
			warpedModelOne + int32(alphaRaw)%warpRange,
			int32(gammaRaw) % warpRange,
			int32(deltaRaw) % warpRange,
			warpedModelOne + int32(betaRaw)%warpRange,
		}
		// libaom keeps alpha/beta/gamma/delta in [-512, 511] after parameter
		// reduction; clamp to mirror that bitstream contract.
		clamp := func(v int16) int16 {
			if v > 511 {
				return 511
			}
			if v < -512 {
				return -512
			}
			return v
		}
		alpha := clamp(alphaRaw)
		beta := clamp(betaRaw)
		gamma := clamp(gammaRaw)
		delta := clamp(deltaRaw)

		subsamplingX := rawSubs&1 != 0
		subsamplingY := rawSubs&2 != 0

		err := PredictWarpedPlaneBlockBitDepth(dst, ref, bytesPerSample, bitDepth, dstX, dstY, blockW, blockH, matrix, alpha, beta, gamma, delta, subsamplingX, subsamplingY)
		if err != nil {
			// Any rejection must come from contract validation, not a panic.
			return
		}

		for y := 0; y < dst.Height; y++ {
			for x := 0; x < dst.Width; x++ {
				if x >= dstX && x < dstX+blockW && y >= dstY && y < dstY+blockH {
					continue
				}
				if bytesPerSample == 1 {
					if got := dst.Pix[y*dst.Stride+x]; got != 0 {
						t.Fatalf("dst outside region modified at (%d,%d): %d", x, y, got)
					}
				} else {
					if got := loadHighBDSample(dst, x, y); got != 0 {
						t.Fatalf("dst outside region modified at (%d,%d): %d", x, y, got)
					}
				}
			}
		}
		for y := dstY; y < dstY+blockH; y++ {
			for x := dstX; x < dstX+blockW; x++ {
				var got uint16
				if bytesPerSample == 1 {
					got = uint16(dst.Pix[y*dst.Stride+x])
				} else {
					got = loadHighBDSample(dst, x, y)
				}
				if got > maxSample {
					t.Fatalf("dst sample(%d,%d)=%d exceeds max %d", x, y, got, maxSample)
				}
			}
		}
	})
}
