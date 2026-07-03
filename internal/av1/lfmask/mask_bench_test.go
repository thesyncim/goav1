package lfmask

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// BenchmarkBuilderCreateInter asserts the mask build carries zero allocations so
// it can run in the decode hot loop once wired in. The transform scratch lives on
// the reused Builder, so no per-block heap traffic is expected.
func BenchmarkBuilderCreateInter(b *testing.B) {
	var builder Builder
	var m FilterMask
	stride := 32
	lc := LevelCache{Cells: make([][4]uint8, stride*32), Stride: stride}
	ay := make([]uint8, 32)
	ly := make([]uint8, 32)
	auv := make([]uint8, 32)
	luv := make([]uint8, 32)
	lv := Levels{YVert: 20, YHorz: 21, U: 18, V: 18}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Clear()
		for k := range ay {
			ay[k], ly[k] = initY, initY
			auv[k], luv[k] = initUV, initUV
		}
		builder.CreateInter(&m, lc, lv, 0, 0, stride, 32, 0, tile.BlockSize16x16,
			tile.TransformSize16x16, [2]uint16{1, 0}, tile.TransformSize8x8, I420(),
			ay, ly, auv, luv)
	}
}
