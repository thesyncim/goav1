package goav1_test

import (
	"errors"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func FuzzPublicDecodeAndReconstructDecoderFrameWorkJobResiduals(f *testing.F) {
	f.Add([]byte{0x00, 0x00, 0x00, 0x00}, uint8(64), false)
	f.Add([]byte{0xff, 0x80, 0x00, 0x7f}, uint8(3), true)
	f.Add([]byte{0xaa, 0x55, 0x11, 0xee, 0x00}, uint8(127), false)

	f.Fuzz(func(t *testing.T, payload []byte, rawQ uint8, sideMaps bool) {
		if len(payload) == 0 || len(payload) > 128 {
			return
		}
		qIndex := rawQ | 1
		output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
		publicFillDecoderPostFilterPlane(output.Y)
		batch := av1.DecoderFrameWorkBatch{
			Output:  output,
			Payload: payload,
			FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
				Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
					ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true},
				}),
				FrameSize:    av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8},
				Quantization: av1.QuantizationParams{BaseQIdx: qIndex},
				TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
			},
			Jobs: []av1.TileJob{{SBCols: 1, SBRows: 1, Offset: 0, Size: len(payload)}},
		}
		if sideMaps {
			publicBindResidualFuzzSideMaps(t, &batch)
		}

		var state av1.TileDecodeState
		if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, &state); err != nil {
			t.Fatal(err)
		}
		var storage av1.DecoderFrameWorkTileResidualCDFStorage
		if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, qIndex); err != nil {
			t.Fatal(err)
		}
		loopReq, err := av1.DecoderFrameWorkJobBlockLoopRequest(batch, 0, nil, nil, 0, nil)
		if err != nil {
			t.Fatal(err)
		}
		int32Len, int16Len, err := av1.DecoderFrameWorkResidualMaxScratchLen(batch, qIndex, 0, av1.DecoderFrameWorkPlaneY)
		if err != nil {
			t.Fatal(err)
		}

		var scratch av1.DecoderFrameWorkTileResidualScratch
		req := av1.DecoderFrameWorkTileResidualRequest{
			Loop:          loopReq,
			TransformMode: batch.TransformRef.TransformMode,
			Transforms: func(visit av1.TileBlockLoopVisit) (av1.DecoderFrameWorkBlockTransforms, error) {
				return av1.ReadDecoderFrameWorkBlockTransforms(batch, &state, visit)
			},
			Int32Scratch:    make([]int32, int32Len),
			ResidualScratch: make([]int16, int16Len),
		}
		stats, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, &state, av1.DecoderFrameWorkTileResidualCDFsFromStorage(&storage), &scratch, req)
		if err != nil {
			return
		}
		if stats.TXBs != stats.NonZero+stats.AllZero {
			t.Fatalf("residual stats=%+v inconsistent txb counts", stats)
		}
		if stats.Loop.CoefficientTXBs != stats.TXBs {
			t.Fatalf("loop stats=%+v residual stats=%+v", stats.Loop, stats)
		}
	})
}

func FuzzPublicDecodeAndRetainDecoderFrameWorkBatchResiduals(f *testing.F) {
	f.Add([]byte{0x00, 0x00, 0x00, 0x00}, uint8(64), true, false)
	f.Add([]byte{0xff, 0x80, 0x00, 0x7f}, uint8(3), false, true)
	f.Add([]byte{0xaa, 0x55, 0x11, 0xee, 0x00, 0x42}, uint8(127), true, true)

	f.Fuzz(func(t *testing.T, payload []byte, rawQ uint8, updateContext bool, sideMaps bool) {
		if len(payload) < 2 || len(payload) > 128 {
			return
		}
		qIndex := rawQ | 1
		output := publicDecoderPostFilterFrame(t, av1.FrameFormat{Width: 128, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
		publicFillDecoderPostFilterPlane(output.Y)
		firstSize := len(payload) / 2
		batch := av1.DecoderFrameWorkBatch{
			Output:  output,
			Payload: payload,
			FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
				Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
					ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true},
				}),
				FrameSize:    av1.FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8},
				Quantization: av1.QuantizationParams{BaseQIdx: qIndex},
				TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
			},
			Jobs: []av1.TileJob{
				{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: firstSize},
				{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: firstSize, Size: len(payload) - firstSize, UpdatesFrameContext: updateContext},
			},
		}
		var retained av1.DecoderFrameWorkTileResidualCDFStorage
		retainedValid := false
		batch.RetainedTileResidualCDFs = &retained
		batch.RetainedTileResidualCDFsValid = &retainedValid
		if sideMaps {
			publicBindResidualFuzzSideMaps(t, &batch)
		}

		var storage av1.DecoderFrameWorkTileResidualCDFStorage
		if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, qIndex); err != nil {
			t.Fatal(err)
		}
		int32Len, int16Len, err := av1.DecoderFrameWorkResidualMaxScratchLen(batch, qIndex, 0, av1.DecoderFrameWorkPlaneY)
		if err != nil {
			t.Fatal(err)
		}
		loopContextLen, err := av1.DecoderFrameWorkBatchResidualLoopContextAboveLen(batch)
		if err != nil {
			t.Fatal(err)
		}

		var state av1.TileDecodeState
		var scratch av1.DecoderFrameWorkTileResidualScratch
		req := av1.DecoderFrameWorkBatchResidualRequest{
			Tile: av1.DecoderFrameWorkTileResidualRequest{
				TransformMode: batch.TransformRef.TransformMode,
				Transforms: func(visit av1.TileBlockLoopVisit) (av1.DecoderFrameWorkBlockTransforms, error) {
					return av1.ReadDecoderFrameWorkBlockTransforms(batch, &state, visit)
				},
				Int32Scratch:    make([]int32, int32Len),
				ResidualScratch: make([]int16, int16Len),
			},
			LoopContextAbove: make([]av1.TileBlockLoopRootAboveContext, loopContextLen),
		}
		stats, err := av1.DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, &state, &storage, &scratch, req)
		if err != nil {
			return
		}
		if stats.TXBs != stats.NonZero+stats.AllZero {
			t.Fatalf("batch residual stats=%+v inconsistent txb counts", stats)
		}
		if stats.Loop.CoefficientTXBs != stats.TXBs {
			t.Fatalf("loop stats=%+v residual stats=%+v", stats.Loop, stats)
		}
		if retainedValid != updateContext {
			t.Fatalf("retained=%v want %v", retainedValid, updateContext)
		}
	})
}

func FuzzPublicDecoderFrameWorkBatchResidualRunnerSideData(f *testing.F) {
	f.Add(uint8(1), uint8(0), false, false, false)
	f.Add(uint8(2), uint8(1), true, false, false)
	f.Add(uint8(4), uint8(2), true, true, false)
	f.Add(uint8(3), uint8(3), false, false, true)

	f.Fuzz(func(t *testing.T, rawWorkers uint8, rawCDEFBits uint8, activeRestoration bool, omitRestorationScratch bool, corruptLoopMap bool) {
		workers := int(rawWorkers%4) + 1
		sequence := av1.SequenceHeader{
			EnableCDEF:        true,
			EnableRestoration: true,
			ColorConfig: av1.ColorConfig{
				BitDepth:   8,
				MonoChrome: true,
			},
		}
		size := av1.FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8}
		cdefBits := rawCDEFBits & 3
		cdef := av1.CDEFParams{
			Bits:          cdefBits,
			StrengthCount: uint8(1) << cdefBits,
		}
		restoration := av1.RestorationParams{}
		if activeRestoration {
			restoration = av1.RestorationParams{
				Type:      [3]av1.RestorationType{av1.RestorationWiener},
				UnitSizeY: 64,
			}
		}

		batch := av1.DecoderFrameWorkBatch{
			FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
				Sequence:     av1.DecoderFrameWorkSequenceContextFromHeader(sequence),
				FrameSize:    size,
				CDEF:         cdef,
				Restoration:  restoration,
				Quantization: av1.QuantizationParams{BaseQIdx: 64},
				TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
			},
			Jobs: []av1.TileJob{
				{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 1},
				{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 1, Size: 1},
			},
		}
		runnerSize, err := av1.DecoderFrameWorkBatchResidualRunnerScratchLen(batch, workers)
		if err != nil {
			t.Fatal(err)
		}
		runnerScratch := publicDecoderBatchResidualRunnerScratch(runnerSize)
		if omitRestorationScratch {
			runnerScratch.RestorationRequests = nil
		}
		runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(runnerSize, runnerScratch)
		if err != nil {
			t.Fatal(err)
		}

		sideSize, err := av1.DecoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
		if err != nil {
			t.Fatal(err)
		}
		side, err := av1.BindDecoderFrameWorkSideData(sequence, size, cdef, restoration, av1.DecoderFrameWorkSideDataScratch{
			CDEFIndexMap:             make([]uint8, sideSize.CDEFIndexMap),
			CDEFReadMap:              make([]bool, sideSize.CDEFReadMap),
			LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, sideSize.LoopFilterMap),
			RestorationRecords:       make([]av1.TileRestorationUnitRecord, sideSize.RestorationRecords),
			RestorationBoundaryAbove: make([]uint16, sideSize.RestorationBoundaryAbove),
			RestorationBoundaryBelow: make([]uint16, sideSize.RestorationBoundaryBelow),
		})
		if err != nil {
			t.Fatal(err)
		}
		side.CDEFIndexMap.Read[0] = true
		side.LoopFilterMap.Records[0].Valid = true
		if len(side.RestorationFrameBuffers.Records[0]) != 0 {
			side.RestorationFrameBuffers.Records[0][0].Index = 99
		}
		if corruptLoopMap {
			side.LoopFilterMap = av1.DecoderFrameWorkLoopFilterMap{Stride: 1, Rows: 1}
		}

		err = av1.SetDecoderFrameWorkBatchResidualRunnerSideData(&runner, side)
		if activeRestoration && omitRestorationScratch {
			if !errors.Is(err, av1.ErrFrameShortBuffer) {
				t.Fatalf("missing restoration requests err=%v want %v", err, av1.ErrFrameShortBuffer)
			}
			return
		}
		if corruptLoopMap {
			if !errors.Is(err, av1.ErrThreadingInvalidBatch) {
				t.Fatalf("corrupt loop map err=%v want %v", err, av1.ErrThreadingInvalidBatch)
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if side.CDEFIndexMap.Read[0] || side.LoopFilterMap.Records[0].Valid {
			t.Fatalf("side data was not reset: cdef=%v loop=%+v", side.CDEFIndexMap.Read[0], side.LoopFilterMap.Records[0])
		}
		if runner.CDEFIndexMap.Stride != side.CDEFIndexMap.Stride ||
			runner.LoopFilterMap.Stride != side.LoopFilterMap.Stride {
			t.Fatalf("runner side maps cdef=%+v loop=%+v side=%+v", runner.CDEFIndexMap, runner.LoopFilterMap, side)
		}
		if activeRestoration {
			if len(runner.RestorationRequests) < workers ||
				!runner.RestorationRequests[0].Buffers.Plan.Active {
				t.Fatalf("runner restoration requests=%+v workers=%d", runner.RestorationRequests, workers)
			}
		}
	})
}

func publicBindResidualFuzzSideMaps(t *testing.T, batch *av1.DecoderFrameWorkBatch) {
	t.Helper()
	batch.CDEF = av1.CDEFParams{Bits: 0, StrengthCount: 1}
	sequence := av1.SequenceHeader{
		Use128x128Superblock: batch.Sequence.Use128x128Superblock,
		ColorConfig:          batch.Sequence.ColorConfig,
	}
	_, _, cdefLength, err := av1.DecoderFrameWorkCDEFIndexMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, batch.FrameSize, batch.CDEF, make([]uint8, cdefLength), make([]bool, cdefLength))
	if err != nil {
		t.Fatal(err)
	}
	_, _, loopLength, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	loopMap, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, batch.FrameSize, make([]av1.DecoderFrameWorkLoopFilterBlockRecord, loopLength))
	if err != nil {
		t.Fatal(err)
	}
	batch.CDEFIndexMap = &cdefMap
	batch.LoopFilterMap = &loopMap
}
