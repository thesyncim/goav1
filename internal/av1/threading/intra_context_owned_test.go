package threading

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkOwnedIntraContextsSelection(t *testing.T) {
	var scratch FrameWorkTileResidualScratch
	batch := FrameWorkBatch{}
	batch.FrameHeader.FrameType = parser.FrameTypeKey
	request := FrameWorkTileResidualRequest{}
	loop := tile.BlockLoopRequest{FrameType: parser.FrameTypeKey, DecodePredictionModes: true, ContextCarrier: &scratch.LoopContext}
	for _, ft := range []parser.FrameType{parser.FrameTypeKey, parser.FrameTypeIntraOnly} {
		b, q := batch, loop
		b.FrameHeader.FrameType = ft
		q.FrameType = ft
		if !frameWorkOwnedIntraContexts(&b, &scratch, request, q) {
			t.Fatal("eligible private intra route rejected", ft)
		}
	}
	for _, tc := range []struct {
		name  string
		alter func(*FrameWorkBatch, *FrameWorkTileResidualRequest, *tile.BlockLoopRequest)
	}{
		{"inter", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			b.FrameHeader.FrameType = parser.FrameTypeInter
			l.FrameType = parser.FrameTypeInter
		}},
		{"switch", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			b.FrameHeader.FrameType = parser.FrameTypeSwitch
			l.FrameType = parser.FrameTypeSwitch
		}},
		{"frame mismatch", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.FrameType = parser.FrameTypeIntraOnly
		}},
		{"frame IBC", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, _ *tile.BlockLoopRequest) {
			b.FrameSize.AllowIntrabc = true
		}},
		{"request IBC", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.AllowIntrabc = true
		}},
		{"skip-mode override", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.SkipMode.Enabled = true
		}},
		{"no prediction", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.DecodePredictionModes = false
		}},
		{"copied carrier", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			copied := scratch.LoopContext
			l.ContextCarrier = &copied
		}},
		{"caller neighborhood", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.Walk.UseNeighborBounds = true
		}},
		{"wavefront", func(b *FrameWorkBatch, _ *FrameWorkTileResidualRequest, _ *tile.BlockLoopRequest) {
			b.WavefrontWorkers = 2
		}},
		{"block hook", func(_ *FrameWorkBatch, r *FrameWorkTileResidualRequest, _ *tile.BlockLoopRequest) {
			r.AfterBlock = func(tile.BlockLoopVisit) error { return nil }
		}},
		{"SB hook", func(_ *FrameWorkBatch, _ *FrameWorkTileResidualRequest, l *tile.BlockLoopRequest) {
			l.BeforeSuperblock = func(tile.BlockLoopSuperblockVisit) error { return nil }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, r, l := batch, request, loop
			tc.alter(&b, &r, &l)
			if frameWorkOwnedIntraContexts(&b, &scratch, r, l) {
				t.Fatal("fallback incorrectly selected private contexts")
			}
		})
	}
}
