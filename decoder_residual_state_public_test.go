package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicDecoderTileResidualCDFStorageLifecycle(t *testing.T) {
	var initial av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&initial, 64); err != nil {
		t.Fatal(err)
	}
	if err := initial.DeltaQ.Update(2); err != nil {
		t.Fatal(err)
	}
	initialCDFs := av1.DecoderFrameWorkTileResidualCDFsFromStorage(&initial)
	if initialCDFs.Loop.Partition == nil || initialCDFs.Coeff.Coeff == nil || initialCDFs.TransformType == nil {
		t.Fatalf("initial CDF view=%+v", initialCDFs)
	}

	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := av1.DecoderFrameWorkBatch{
		Payload: []byte{0x00, 0xff},
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Quantization: av1.QuantizationParams{BaseQIdx: 91},
		},
		InitialTileResidualCDFs:       &initial,
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []av1.TileJob{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}

	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, &storage); err != nil {
		t.Fatal(err)
	}
	if !publicCDFValuesEqual(storage.DeltaQ.Values(), initial.DeltaQ.Values()) {
		t.Fatalf("storage delta q=%v want %v", storage.DeltaQ.Values(), initial.DeltaQ.Values())
	}
	if err := storage.DeltaQ.Update(1); err != nil {
		t.Fatal(err)
	}

	var state av1.TileDecodeState
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, &state); err != nil {
		t.Fatal(err)
	}
	if state.RetainFrameContext || state.CurrentBaseQIdx != 91 {
		t.Fatalf("state=%+v", state)
	}
	if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 0, &state, &storage); err != nil {
		t.Fatal(err)
	}
	if retainedValid {
		t.Fatal("non-update tile retained frame context")
	}

	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 1, &state); err != nil {
		t.Fatal(err)
	}
	if !state.RetainFrameContext || state.CurrentBaseQIdx != 91 {
		t.Fatalf("update state=%+v", state)
	}
	if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 1, &state, &storage); err != nil {
		t.Fatal(err)
	}
	if !retainedValid {
		t.Fatal("update tile did not retain frame context")
	}
	if !publicCDFValuesEqual(retained.DeltaQ.Values(), storage.DeltaQ.Values()) {
		t.Fatalf("retained delta q=%v want %v", retained.DeltaQ.Values(), storage.DeltaQ.Values())
	}

	batch.InitialTileResidualCDFs = nil
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, &storage); err != nil {
		t.Fatal(err)
	}
	if publicCDFValuesEqual(storage.DeltaQ.Values(), initial.DeltaQ.Values()) {
		t.Fatal("default fallback unexpectedly reused initial frame context")
	}
}

func TestPublicDecoderTileResidualStateRejectsInvalidInputs(t *testing.T) {
	batch := av1.DecoderFrameWorkBatch{
		Payload: []byte{0xaa},
		Jobs:    []av1.TileJob{{Offset: 0, Size: 2}},
	}
	var state av1.TileDecodeState
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil cdf storage err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, -1, &state); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil state err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, &state); !errors.Is(err, av1.ErrTileInvalidPlan) {
		t.Fatalf("invalid range err=%v want %v", err, av1.ErrTileInvalidPlan)
	}
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 0, &state, nil); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil retain storage err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(nil, 0); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil default cdf storage err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
	if got := av1.DecoderFrameWorkTileResidualCDFsFromStorage(nil); got.Loop.Partition != nil || got.Coeff.Coeff != nil || got.TransformType != nil {
		t.Fatalf("nil CDF view=%+v", got)
	}
	if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 0, nil, &storage); !errors.Is(err, av1.ErrThreadingInvalidBatch) {
		t.Fatalf("nil retain state err=%v want %v", err, av1.ErrThreadingInvalidBatch)
	}
}

func TestPublicDecoderTileResidualStateAllocs(t *testing.T) {
	var initial av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&initial, 64); err != nil {
		t.Fatal(err)
	}
	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := av1.DecoderFrameWorkBatch{
		Payload: []byte{0x00, 0xff},
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Quantization: av1.QuantizationParams{BaseQIdx: 73},
		},
		InitialTileResidualCDFs:       &initial,
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []av1.TileJob{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	var state av1.TileDecodeState
	var err error

	allocs := testing.AllocsPerRun(1000, func() {
		err = av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, &storage)
		if err != nil {
			return
		}
		_ = av1.DecoderFrameWorkTileResidualCDFsFromStorage(&storage)
		err = av1.InitDecoderFrameWorkJobDecodeState(batch, 1, &state)
		if err != nil {
			return
		}
		err = av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 1, &state, &storage)
	})
	if err != nil {
		t.Fatal(err)
	}
	if allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}

func publicCDFValuesEqual(a []uint16, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
