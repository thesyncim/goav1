package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestFrameWorkBatchSetupTemporalMotionField(t *testing.T) {
	ctx := FrameWorkBatch{
		FrameWorkFrameContext: FrameWorkFrameContext{
			Sequence: FrameWorkSequenceContextFromHeader(parser.SequenceHeader{
				EnableOrderHint: true,
				OrderHintBits:   5,
				ColorConfig:     parser.ColorConfig{BitDepth: 8},
			}),
			FrameHeader: parser.FrameHeaderPrefix{OrderHint: 8},
			FrameSize:   parser.FrameSize{CodedWidth: 64, Height: 64},
		},
	}
	_, _, length, err := ctx.ReferenceMVFrameShape()
	if err != nil {
		t.Fatal(err)
	}
	temporal, err := ctx.BindTemporalMotionField(make([]tile.TemporalMotionEntry, length))
	if err != nil {
		t.Fatal(err)
	}
	start, err := ctx.BindReferenceMVFrame(make([]tile.ReferenceMVEntry, length))
	if err != nil {
		t.Fatal(err)
	}
	start.Entries[2*start.Stride+2] = tile.ReferenceMVEntry{
		Ref:   tile.ReferenceFrameLast,
		MV:    motion.Vector{Row: 64, Col: 64},
		Valid: true,
	}
	ctx.TemporalMVs = &temporal
	var refOrderHints [parser.InterRefsPerFrame]uint8
	refOrderHints[tile.ReferenceFrameLast] = 0
	refOrderHints[tile.ReferenceFrameAltref] = 1
	ctx.ReferenceMVs[tile.ReferenceFrameLast] = tile.TemporalMotionReferenceFrame{
		Frame:         &start,
		OrderHint:     4,
		RefOrderHints: refOrderHints,
	}

	stats, err := ctx.SetupTemporalMotionField()
	if err != nil {
		t.Fatal(err)
	}
	if stats.Projections != 1 || stats.RefStamp != 1 {
		t.Fatalf("stats=%+v", stats)
	}
	if got := temporal.Entries[1*temporal.Stride+1]; got != (tile.TemporalMotionEntry{
		MV:             motion.Vector{Row: 64, Col: 64},
		RefFrameOffset: 4,
		Valid:          true,
	}) {
		t.Fatalf("projected entry=%+v", got)
	}
}

func TestFrameWorkBatchSetupTemporalMotionFieldRejectsMissingField(t *testing.T) {
	emptyBatch := FrameWorkBatch{}
	if _, err := emptyBatch.SetupTemporalMotionField(); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("err=%v want %v", err, ErrInvalidBatch)
	}
}
