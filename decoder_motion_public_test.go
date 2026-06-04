package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicDecoderReferenceMVAndTemporalMotionBinding(t *testing.T) {
	sequence := av1.SequenceHeader{ColorConfig: av1.ColorConfig{BitDepth: 8}}
	size := av1.FrameSize{CodedWidth: 130, UpscaledWidth: 130, Height: 65, SuperResDenominator: 8}
	cols, rows, length, err := av1.DecoderFrameWorkReferenceMVFrameShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	if cols != 17 || rows != 9 || length != 153 {
		t.Fatalf("shape=%d,%d,%d want 17,9,153", cols, rows, length)
	}

	mvEntries := make([]av1.TileReferenceMVEntry, length+1)
	for i := range mvEntries {
		mvEntries[i] = av1.TileReferenceMVEntry{
			Ref:   av1.TileReferenceFrameGolden,
			MV:    av1.MotionVector{Row: 8, Col: -4},
			Valid: true,
		}
	}
	mvFrame, err := av1.BindDecoderFrameWorkReferenceMVFrame(sequence, size, mvEntries)
	if err != nil {
		t.Fatal(err)
	}
	if int(mvFrame.Cols) != cols || int(mvFrame.Rows) != rows || int(mvFrame.Stride) != cols || len(mvFrame.Entries) != length {
		t.Fatalf("mv frame=%+v len=%d", mvFrame, len(mvFrame.Entries))
	}
	for i, entry := range mvFrame.Entries {
		if entry.Ref != av1.TileReferenceFrameNone || entry.Valid {
			t.Fatalf("mv entry %d=%+v want NONE invalid", i, entry)
		}
	}
	if !mvEntries[length].Valid || mvEntries[length].Ref != av1.TileReferenceFrameGolden {
		t.Fatalf("caller MV entry past grid was modified: %+v", mvEntries[length])
	}
	if _, err := av1.BindDecoderFrameWorkReferenceMVFrame(sequence, size, mvEntries[:length-1]); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("short MV frame err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}

	temporalEntries := make([]av1.TileTemporalMotionEntry, length+1)
	for i := range temporalEntries {
		temporalEntries[i] = av1.TileTemporalMotionEntry{
			MV:             av1.MotionVector{Row: -16, Col: 12},
			RefFrameOffset: 3,
			Valid:          true,
		}
	}
	temporal, err := av1.BindDecoderFrameWorkTemporalMotionField(sequence, size, temporalEntries)
	if err != nil {
		t.Fatal(err)
	}
	if int(temporal.Cols) != cols || int(temporal.Rows) != rows || int(temporal.Stride) != cols || len(temporal.Entries) != length {
		t.Fatalf("temporal field=%+v len=%d", temporal, len(temporal.Entries))
	}
	for i, entry := range temporal.Entries {
		if entry.Valid || entry.MV != (av1.MotionVector{}) || entry.RefFrameOffset != 0 {
			t.Fatalf("temporal entry %d=%+v want zero invalid", i, entry)
		}
	}
	if !temporalEntries[length].Valid || temporalEntries[length].RefFrameOffset != 3 {
		t.Fatalf("caller temporal entry past grid was modified: %+v", temporalEntries[length])
	}
	if _, err := av1.BindDecoderFrameWorkTemporalMotionField(sequence, size, temporalEntries[:length-1]); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("short temporal field err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderMotionBindingAllocs(t *testing.T) {
	sequence := av1.SequenceHeader{ColorConfig: av1.ColorConfig{BitDepth: 8}}
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	_, _, length, err := av1.DecoderFrameWorkReferenceMVFrameShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	mvEntries := make([]av1.TileReferenceMVEntry, length)
	temporalEntries := make([]av1.TileTemporalMotionEntry, length)

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, err = av1.DecoderFrameWorkReferenceMVFrameShape(sequence, size)
		if err != nil {
			return
		}
		_, err = av1.BindDecoderFrameWorkReferenceMVFrame(sequence, size, mvEntries)
		if err != nil {
			return
		}
		_, err = av1.BindDecoderFrameWorkTemporalMotionField(sequence, size, temporalEntries)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
