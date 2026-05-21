package goav1

import (
	"errors"
	"testing"
)

func TestDecoderFinishFrameSurface(t *testing.T) {
	format := FrameFormat{Width: 16, Height: 16, BitDepth: 8, SubsamplingX: true, SubsamplingY: true, Align: 32}
	layout, err := FrameRequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size*2)
	var frames [2]Frame
	var free [2]int
	var used [2]bool
	pool, err := BindFramePool(backing, format, frames[:], free[:], used[:])
	if err != nil {
		t.Fatal(err)
	}
	sequence := SequenceHeader{ColorConfig: ColorConfig{
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
	}}
	size := FrameSize{CodedWidth: 16, Height: 16}

	index0, _, err := DecoderAcquireFrameSurface(&pool, sequence, size, 32)
	if err != nil {
		t.Fatal(err)
	}
	index1, _, err := DecoderAcquireFrameSurface(&pool, sequence, size, 32)
	if err != nil {
		t.Fatal(err)
	}

	var refs DecoderSurfaceReferences
	var releases [RefFrames]int
	if _, err := refs.Refresh(0xff, index0, releases[:]); err != nil {
		t.Fatal(err)
	}

	count, err := DecoderFinishFrameSurface(&refs, &pool, DecoderEvent{
		Kind:      DecoderEventTileGroup,
		FrameSize: FrameSize{RefreshFrameFlags: 0xff},
		TileGroup: TileGroup{Final: true},
	}, index1, releases[:])
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 || releases[0] != index0 {
		t.Fatalf("count=%d release=%d want 1,%d", count, releases[0], index0)
	}
	if _, err := pool.Frame(index0); !errors.Is(err, ErrFrameInvalidSlot) {
		t.Fatalf("released frame err=%v want %v", err, ErrFrameInvalidSlot)
	}
}
