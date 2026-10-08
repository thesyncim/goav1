package transform

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestForwardBlock8BitResidualTrustedMatchesPureGo(t *testing.T) {
	types := make([]Type, TypeCount)
	for i := range types {
		types[i] = Type(i)
	}
	rng := rand.New(rand.NewSource(0x8b17))

	for _, size := range allTransformSizes {
		width, height := int(size.Width), int(size.Height)
		coeffSize := adjustedScanSize(size)
		coeffWidth, coeffHeight := int(coeffSize.Width), int(coeffSize.Height)
		residualStride := width + 3
		coeffStride := coeffHeight + 2
		residual := make([]int16, (height-1)*residualStride+width)
		coeffLen := (coeffWidth-1)*coeffStride + coeffHeight
		scratch := make([]int32, width*height)

		for _, typ := range types {
			if typ != TypeDCTDCT && !forwardGenericSupported(size, typ) {
				continue
			}
			t.Run(fmt.Sprintf("%s/%d", sizeName(size), typ), func(t *testing.T) {
				for pattern := 0; pattern < 2; pattern++ {
					// Padding is deliberately outside the 8-bit residual domain. The
					// contract covers addressed block samples, not row padding.
					for i := range residual {
						residual[i] = 30000
					}
					for row := 0; row < height; row++ {
						for col := 0; col < width; col++ {
							i := row*residualStride + col
							if pattern == 0 {
								switch (row*width + col) % 4 {
								case 0:
									residual[i] = -255
								case 1:
									residual[i] = 255
								case 2:
									residual[i] = -1
								default:
									residual[i] = 1
								}
							} else {
								src, pred := byte(rng.Intn(256)), byte(rng.Intn(256))
								residual[i] = int16(src) - int16(pred)
							}
						}
					}

					got := make([]int32, coeffLen)
					want := make([]int32, coeffLen)
					for i := range got {
						got[i], want[i] = 0x13579, 0x13579
					}
					gotScratch := make([]int32, len(scratch))
					wantScratch := make([]int32, len(scratch))
					for i := range gotScratch {
						gotScratch[i], wantScratch[i] = -0x2468, -0x2468
					}

					err := ForwardBlock8BitResidualTrusted(got, coeffStride, residual, residualStride, gotScratch, size, typ)
					if err != nil {
						t.Fatalf("trusted transform: %v", err)
					}
					if err := forwardBlock8BitTrustedPureGoReference(want, coeffStride, residual, residualStride, wantScratch, size, typ); err != nil {
						t.Fatalf("pure-Go reference: %v", err)
					}
					for i := range want {
						if got[i] != want[i] {
							t.Fatalf("coeff[%d]=%d want %d", i, got[i], want[i])
						}
					}
				}
			})
		}
	}
}

// forwardBlock8BitTrustedPureGoReference keeps the oracle independent for each
// path that the trusted entry can dispatch directly. Shapes outside those
// paths use the ordinary dispatcher because the trusted API delegates to it.
func forwardBlock8BitTrustedPureGoReference(coeff []int32, coeffStride int, residual []int16, residualStride int, scratch []int32, size Size, typ Type) error {
	if typ == TypeDCTDCT {
		switch size {
		case Size{Width: 4, Height: 4}:
			forwardDCT4x4PureGo(coeff, coeffStride, residual, residualStride)
			return nil
		case Size{Width: 8, Height: 8}:
			forwardDCT8x8PureGo(coeff, coeffStride, residual, residualStride)
			return nil
		case Size{Width: 16, Height: 16}:
			forwardDCT16x16PureGo(coeff, coeffStride, residual, residualStride)
			return nil
		case Size{Width: 32, Height: 32}:
			forwardDCT32x32PureGo(coeff, coeffStride, residual, residualStride)
			return nil
		default:
			return forwardDCTBySize(coeff, coeffStride, residual, residualStride, size)
		}
	}
	if size == (Size{Width: 8, Height: 8}) && forwardBlock8x8HybridSupported(typ) {
		if !forwardBlock8x8HybridPureGo(coeff, coeffStride, residual, residualStride, scratch, typ) {
			return ErrInvalidTransform
		}
		return nil
	}
	return forwardGenericBlock(coeff, coeffStride, residual, residualStride, scratch, size, typ)
}

func TestForwardBlock8BitResidualTrustedRetainsShapeChecks(t *testing.T) {
	residual := make([]int16, 64)
	coeff := make([]int32, 64)
	scratch := make([]int32, 64)

	if err := ForwardBlock8BitResidualTrusted(coeff[:63], 8, residual, 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT); err != ErrInvalidTransform {
		t.Fatalf("short coefficient buffer error=%v, want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock8BitResidualTrusted(coeff, 8, residual[:63], 8, scratch, Size{Width: 8, Height: 8}, TypeDCTDCT); err != ErrInvalidTransform {
		t.Fatalf("short residual buffer error=%v, want %v", err, ErrInvalidTransform)
	}
	if err := ForwardBlock8BitResidualTrusted(coeff, 8, residual, 8, scratch[:63], Size{Width: 8, Height: 8}, TypeADSTDCT); err != ErrInvalidTransform {
		t.Fatalf("short hybrid scratch error=%v, want %v", err, ErrInvalidTransform)
	}
}
