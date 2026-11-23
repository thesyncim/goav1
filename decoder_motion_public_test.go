package goav1

import (
	"errors"
	"testing"

	internalmotion "github.com/thesyncim/goav1/internal/av1/motion"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPublicDecoderReferenceMVAndTemporalMotionBinding(t *testing.T) {
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 8}}
	size := FrameSize{CodedWidth: 130, UpscaledWidth: 130, Height: 65, SuperResDenominator: 8}
	cols, rows, length, err := decoderFrameWorkReferenceMVFrameShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	if cols != 17 || rows != 9 || length != 153 {
		t.Fatalf("shape=%d,%d,%d want 17,9,153", cols, rows, length)
	}

	mvEntries := make([]internaltile.ReferenceMVEntry, length+1)
	for i := range mvEntries {
		mvEntries[i] = internaltile.ReferenceMVEntry{
			Ref:   internaltile.ReferenceFrameGolden,
			MV:    internalmotion.Vector{Row: 8, Col: -4},
			Valid: true,
		}
	}
	mvFrame, err := bindDecoderFrameWorkReferenceMVFrame(sequence, size, mvEntries)
	if err != nil {
		t.Fatal(err)
	}
	if int(mvFrame.Cols) != cols || int(mvFrame.Rows) != rows || int(mvFrame.Stride) != cols || len(mvFrame.Entries) != length {
		t.Fatalf("mv frame=%+v len=%d", mvFrame, len(mvFrame.Entries))
	}
	for i, entry := range mvFrame.Entries {
		if entry.Ref != internaltile.ReferenceFrameNone || entry.Valid {
			t.Fatalf("mv entry %d=%+v want NONE invalid", i, entry)
		}
	}
	if !mvEntries[length].Valid || mvEntries[length].Ref != internaltile.ReferenceFrameGolden {
		t.Fatalf("caller MV entry past grid was modified: %+v", mvEntries[length])
	}
	if _, err := bindDecoderFrameWorkReferenceMVFrame(sequence, size, mvEntries[:length-1]); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("short MV frame err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}

	temporalEntries := make([]internaltile.TemporalMotionEntry, length+1)
	for i := range temporalEntries {
		temporalEntries[i] = internaltile.TemporalMotionEntry{
			MV:             internalmotion.Vector{Row: -16, Col: 12},
			RefFrameOffset: 3,
			Valid:          true,
		}
	}
	temporal, err := bindDecoderFrameWorkTemporalMotionField(sequence, size, temporalEntries)
	if err != nil {
		t.Fatal(err)
	}
	if int(temporal.Cols) != cols || int(temporal.Rows) != rows || int(temporal.Stride) != cols || len(temporal.Entries) != length {
		t.Fatalf("temporal field=%+v len=%d", temporal, len(temporal.Entries))
	}
	for i, entry := range temporal.Entries {
		if entry.Valid || entry.MV != (internalmotion.Vector{}) || entry.RefFrameOffset != 0 {
			t.Fatalf("temporal entry %d=%+v want zero invalid", i, entry)
		}
	}
	if !temporalEntries[length].Valid || temporalEntries[length].RefFrameOffset != 3 {
		t.Fatalf("caller temporal entry past grid was modified: %+v", temporalEntries[length])
	}
	if _, err := bindDecoderFrameWorkTemporalMotionField(sequence, size, temporalEntries[:length-1]); !errors.Is(err, internalthreading.ErrInvalidBatch) {
		t.Fatalf("short temporal field err=%v want %v", err, internalthreading.ErrInvalidBatch)
	}
}

func TestPublicDecoderMotionBindingAllocs(t *testing.T) {
	sequence := SequenceHeader{ColorConfig: internalparser.ColorConfig{BitDepth: 8}}
	size := FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	_, _, length, err := decoderFrameWorkReferenceMVFrameShape(sequence, size)
	if err != nil {
		t.Fatal(err)
	}
	mvEntries := make([]internaltile.ReferenceMVEntry, length)
	temporalEntries := make([]internaltile.TemporalMotionEntry, length)

	allocs := testing.AllocsPerRun(1000, func() {
		_, _, _, err = decoderFrameWorkReferenceMVFrameShape(sequence, size)
		if err != nil {
			return
		}
		_, err = bindDecoderFrameWorkReferenceMVFrame(sequence, size, mvEntries)
		if err != nil {
			return
		}
		_, err = bindDecoderFrameWorkTemporalMotionField(sequence, size, temporalEntries)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
