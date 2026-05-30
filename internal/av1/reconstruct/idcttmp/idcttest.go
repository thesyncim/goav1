package main

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/reconstruct"
	"github.com/thesyncim/goav1/internal/av1/transform"
	"github.com/thesyncim/goav1/internal/av1/frame"
)

func main() {
	q := quantize.Quantizer{DC: 88, AC: 106}
	int32Scratch := make([]int32, 1024)
	residualScratch := make([]int16, 1024)
	cases := []struct {
		name string
		w, h int
		typ  transform.Type
	}{
		{"4x4 DCT", 4, 4, transform.TypeDCTDCT},
		{"8x4 DCT", 8, 4, transform.TypeDCTDCT},
		{"8x8 DCT", 8, 8, transform.TypeDCTDCT},
		{"16x16 DCT", 16, 16, transform.TypeDCTDCT},
	}
	for _, c := range cases {
		buf := make([]byte, c.w*c.h)
		plane := frame.Plane{Pix: buf, Stride: c.w, Width: c.w, Height: c.h}
		for i := range buf {
			buf[i] = 0
		}
		coeffs := make([]int16, c.w*c.h)
		coeffs[0] = 1
		cfg := reconstruct.Block{
			Size:      transform.Size{Width: c.w, Height: c.h},
			Transform: c.typ,
			Quantizer: q,
			EOB:       1,
		}
		if err := reconstruct.ReconstructPlaneBlock(plane, 1, 8, 0, 0, coeffs, c.h, int32Scratch, residualScratch, cfg); err != nil {
			panic(err)
		}
		fmt.Printf("%-12s -> pixel[0,0]=%d\n", c.name, buf[0])
	}
}
