//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestAMD64FusedITXDispatch checks that the AVX2 fused kernels are live on
// this machine: every 8-bit shape accepts a realistic block.
func TestAMD64FusedITXDispatch(t *testing.T) {
	if !itxAVX2 {
		t.Skip("no AVX2")
	}
	for _, c := range amd64ITXBenchCases() {
		dst := make([]byte, c.w*c.h)
		if !c.fused(dst, c.w, c.coeff) {
			t.Errorf("%s: fused kernel declined a realistic block", c.name)
		}
	}
}

type amd64ITXCase struct {
	name       string
	w, h       int
	size       Size
	txType     Type
	rows, cols int
	coeff      []int32
	fused      func(dst []byte, stride int, coeff []int32) bool
}

func amd64ITXBenchCases() []amd64ITXCase {
	rng := rand.New(rand.NewSource(7))
	mk := func(name string, w, h int, txType Type, rows, cols int, fused func(c amd64ITXCase, dst []byte, stride int, coeff []int32) bool) amd64ITXCase {
		scanW, scanH := min(w, 32), min(h, 32)
		coeff := make([]int32, scanW*scanH)
		for c := 0; c < cols; c++ {
			for r := 0; r < rows; r++ {
				coeff[c*scanH+r] = int32(rng.Intn(513) - 256)
			}
		}
		cs := amd64ITXCase{name: name, w: w, h: h, size: Size{Width: uint8(w), Height: uint8(h)}, txType: txType, rows: rows, cols: cols, coeff: coeff}
		cs.fused = func(dst []byte, stride int, coeff []int32) bool { return fused(cs, dst, stride, coeff) }
		return cs
	}
	sq := func(c amd64ITXCase, dst []byte, stride int, coeff []int32) bool {
		scanH := min(c.h, 32)
		switch {
		case c.w == 4:
			return InverseAdd4x4(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows)
		case c.txType != TypeDCTDCT && c.w == 16:
			return InverseAdd16x16Hybrid(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.txType != TypeDCTDCT:
			return InverseAdd8x8Hybrid(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.w == 8:
			return InverseAddDCT8x8(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		case c.w == 16 && c.h == 16:
			return InverseAddDCT16x16(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		case c.w == 16:
			return InverseAddDCT16x32(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		case c.w == 32 && c.h == 16:
			return InverseAddDCT32x16(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		case c.w == 32 && c.rows <= 8 && c.cols <= 8:
			return InverseAddDCT32x32Low8(dst, stride, dst, stride, coeff, scanH)
		case c.w == 32 && c.rows <= 16 && c.cols <= 16:
			return InverseAddDCT32x32Low16(dst, stride, dst, stride, coeff, scanH, c.rows)
		case c.w == 32:
			return InverseAddDCT32x32Full(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		default:
			return InverseAddDCT64x64(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		}
	}
	return []amd64ITXCase{
		mk("4x4/dct", 4, 4, TypeDCTDCT, 4, 4, sq),
		mk("4x4/adst_adst", 4, 4, TypeADSTADST, 4, 4, sq),
		mk("4x4/flipadst_dct", 4, 4, TypeFlipADSTDCT, 4, 4, sq),
		mk("4x4/v_dct", 4, 4, TypeVDCT, 4, 4, sq),
		mk("8x8/dct/low4", 8, 8, TypeDCTDCT, 4, 4, sq),
		mk("8x8/dct/full", 8, 8, TypeDCTDCT, 8, 8, sq),
		mk("8x8/adst_adst", 8, 8, TypeADSTADST, 8, 8, sq),
		mk("16x16/dct/low4", 16, 16, TypeDCTDCT, 4, 4, sq),
		mk("16x16/dct/full", 16, 16, TypeDCTDCT, 16, 16, sq),
		mk("16x16/adst_dct", 16, 16, TypeADSTDCT, 8, 8, sq),
		mk("32x32/dct/low8", 32, 32, TypeDCTDCT, 8, 8, sq),
		mk("32x32/dct/low16", 32, 32, TypeDCTDCT, 16, 16, sq),
		mk("32x32/dct/full", 32, 32, TypeDCTDCT, 32, 32, sq),
		mk("64x64/dct/low8", 64, 64, TypeDCTDCT, 8, 8, sq),
		mk("64x64/dct/full", 64, 64, TypeDCTDCT, 32, 32, sq),
		mk("16x32/dct", 16, 32, TypeDCTDCT, 16, 16, sq),
		mk("32x16/dct", 32, 16, TypeDCTDCT, 16, 16, sq),
	}
}

// BenchmarkAMD64FusedITX compares the AVX2 fused transform + add with the
// staged path it replaces (generic: the staged inverse transform into int32
// scratch, then the clipped residual add).
func BenchmarkAMD64FusedITX(b *testing.B) {
	for _, c := range amd64ITXBenchCases() {
		dst := make([]byte, c.w*c.h)
		raw := make([]int32, c.w*c.h)
		b.Run("shape="+c.name+"/impl=generic", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, c.coeff, min(c.h, 32), c.size, c.txType, 8, c.rows, c.cols); err != nil {
					b.Fatal(err)
				}
				for j, v := range raw {
					dst[j] = byte(min(max(int32(dst[j])+((v+8)>>4), 0), 255))
				}
			}
		})
		b.Run("shape="+c.name+"/impl=avx2", func(b *testing.B) {
			if !itxAVX2 {
				b.Skip("no AVX2")
			}
			for i := 0; i < b.N; i++ {
				if !c.fused(dst, c.w, c.coeff) {
					b.Fatal("declined")
				}
			}
		})
	}
}
