package lfmask

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// BenchmarkBuilderCreateInter asserts the mask build carries zero allocations so
// it can run in the decode hot loop once wired in. The transform scratch lives on
// the reused Builder, so no per-block heap traffic is expected. The cases cover
// the dominant real-stream shapes: split and unsplit 16x16 and 32x32 inter
// blocks, with and without coded residual (skip).
func BenchmarkBuilderCreateInter(b *testing.B) {
	cases := []struct {
		name   string
		bs     tile.BlockSize
		ytx    tile.TransformSize
		split  [2]uint16
		uvtx   tile.TransformSize
		skip   int
		bx, by int
	}{
		{"16x16_split", tile.BlockSize16x16, tile.TransformSize16x16, [2]uint16{1, 0}, tile.TransformSize8x8, 0, 4, 8},
		{"16x16_skip", tile.BlockSize16x16, tile.TransformSize16x16, [2]uint16{}, tile.TransformSize8x8, 1, 4, 8},
		{"32x32_split", tile.BlockSize32x32, tile.TransformSize32x32, [2]uint16{1, 0x5}, tile.TransformSize16x16, 0, 8, 8},
		{"32x32_nosplit", tile.BlockSize32x32, tile.TransformSize32x32, [2]uint16{}, tile.TransformSize16x16, 0, 8, 8},
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			var builder Builder
			var m FilterMask
			stride := 64
			lc := LevelCache{Cells: make([][4]uint8, stride*64), Stride: stride}
			ay := make([]uint8, 64)
			ly := make([]uint8, 32)
			auv := make([]uint8, 32)
			luv := make([]uint8, 32)
			lv := Levels{YVert: 20, YHorz: 21, U: 18, V: 18}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				builder.CreateInter(&m, lc, lv, c.bx, c.by, stride, 64, c.skip, c.bs,
					c.ytx, c.split, c.uvtx, I420(),
					ay[c.bx:], ly[c.by&31:], auv[c.bx>>1:], luv[(c.by&31)>>1:])
			}
		})
	}
}
