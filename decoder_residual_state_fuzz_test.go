package goav1

import (
	"errors"
	"testing"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
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
		output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
		publicFillDecoderPostFilterPlane(output.Y)
		batch := internaldecoder.FrameWorkBatch{
			Output:  output,
			Payload: payload,
			FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
				Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
					ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
				}),
				FrameSize:    FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8},
				Quantization: internalparser.QuantizationParams{BaseQIdx: qIndex},
				TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
			},
			Jobs: []internaltile.Job{{SBCols: 1, SBRows: 1, Offset: 0, Size: uint32(len(payload))}},
		}
		if sideMaps {
			publicBindResidualFuzzSideMaps(t, &batch)
		}

		var state internaltile.DecodeState
		if err := initDecoderFrameWorkJobDecodeState(batch, 0, &state); err != nil {
			t.Fatal(err)
		}
		var storage internalthreading.FrameWorkTileResidualCDFStorage
		if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, qIndex); err != nil {
			t.Fatal(err)
		}
		loopReq, err := decoderFrameWorkJobBlockLoopRequest(batch, 0, nil, nil, 0, nil)
		if err != nil {
			t.Fatal(err)
		}
		int32Len, int16Len, err := decoderFrameWorkResidualMaxScratchLen(batch, qIndex, 0, internaldecoder.FrameWorkPlaneY)
		if err != nil {
			t.Fatal(err)
		}

		var scratch internalthreading.FrameWorkTileResidualScratch
		req := internalthreading.FrameWorkTileResidualRequest{
			Loop:            loopReq,
			TransformMode:   batch.TransformRef.TransformMode,
			Int32Scratch:    make([]int32, int32Len),
			ResidualScratch: make([]int16, int16Len),
		}
		stats, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, &state, decoderFrameWorkTileResidualCDFsFromStorage(&storage), &scratch, req)
		if err != nil {
			return
		}
		if stats.TXBs != stats.NonZero+stats.AllZero {
			t.Fatalf("residual stats=%+v inconsistent txb counts", stats)
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
		output := publicDecoderPostFilterFrame(t, FrameFormat{Width: 128, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
		publicFillDecoderPostFilterPlane(output.Y)
		firstSize := len(payload) / 2
		batch := internaldecoder.FrameWorkBatch{
			Output:  output,
			Payload: payload,
			FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
				Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
					ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
				}),
				FrameSize:    FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8},
				Quantization: internalparser.QuantizationParams{BaseQIdx: qIndex},
				TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
			},
			Jobs: []internaltile.Job{
				{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: uint32(firstSize)},
				{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: uint32(firstSize), Size: uint32(len(payload) - firstSize), UpdatesFrameContext: updateContext},
			},
		}
		var retained internalthreading.FrameWorkTileResidualCDFStorage
		retainedValid := false
		batch.RetainedTileResidualCDFs = &retained
		batch.RetainedTileResidualCDFsValid = &retainedValid
		if sideMaps {
			publicBindResidualFuzzSideMaps(t, &batch)
		}

		var storage internalthreading.FrameWorkTileResidualCDFStorage
		if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, qIndex); err != nil {
			t.Fatal(err)
		}
		int32Len, int16Len, err := decoderFrameWorkResidualMaxScratchLen(batch, qIndex, 0, internaldecoder.FrameWorkPlaneY)
		if err != nil {
			t.Fatal(err)
		}
		loopContextLen, err := decoderFrameWorkBatchResidualLoopContextAboveLen(batch)
		if err != nil {
			t.Fatal(err)
		}

		var state internaltile.DecodeState
		var scratch internalthreading.FrameWorkTileResidualScratch
		req := decoderFrameWorkBatchResidualRequest{
			Tile: internalthreading.FrameWorkTileResidualRequest{
				TransformMode:   batch.TransformRef.TransformMode,
				Int32Scratch:    make([]int32, int32Len),
				ResidualScratch: make([]int16, int16Len),
			},
			LoopContextAbove: make([]internaltile.BlockLoopRootAboveContext, loopContextLen),
		}
		stats, err := decodeAndRetainDecoderFrameWorkBatchResiduals(batch, &state, &storage, &scratch, req)
		if err != nil {
			return
		}
		if stats.TXBs != stats.NonZero+stats.AllZero {
			t.Fatalf("batch residual stats=%+v inconsistent txb counts", stats)
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
		sequence := SequenceHeader{
			EnableCDEF:        true,
			EnableRestoration: true,
			ColorConfig: internalparser.ColorConfig{
				BitDepth:   8,
				MonoChrome: true,
			},
		}
		size := FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8}
		cdefBits := rawCDEFBits & 3
		cdef := internalparser.CDEFParams{
			Bits:          cdefBits,
			StrengthCount: uint8(1) << cdefBits,
		}
		restoration := internalparser.RestorationParams{}
		if activeRestoration {
			restoration = internalparser.RestorationParams{
				Type:      [3]internalparser.RestorationType{internalparser.RestorationWiener},
				UnitSizeY: 64,
			}
		}

		batch := internaldecoder.FrameWorkBatch{
			FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
				Sequence:     internalthreading.FrameWorkSequenceContextFromHeader(sequence),
				FrameSize:    size,
				CDEF:         cdef,
				Restoration:  restoration,
				Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
				TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
			},
			Jobs: []internaltile.Job{
				{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 1},
				{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 1, Size: 1},
			},
		}
		runnerSize, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, workers)
		if err != nil {
			t.Fatal(err)
		}
		runnerScratch := publicDecoderBatchResidualRunnerScratch(runnerSize)
		if omitRestorationScratch {
			runnerScratch.RestorationRequests = nil
		}
		runner, err := bindDecoderFrameWorkBatchResidualRunner(runnerSize, runnerScratch)
		if err != nil {
			t.Fatal(err)
		}

		sideSize, err := decoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
		if err != nil {
			t.Fatal(err)
		}
		side, err := bindDecoderFrameWorkSideData(sequence, size, cdef, restoration, decoderFrameWorkSideDataScratch{
			CDEFIndexMap:             make([]uint8, sideSize.CDEFIndexMap),
			CDEFReadMap:              make([]bool, sideSize.CDEFReadMap),
			LoopFilterMap:            make([]internalthreading.FrameWorkLoopFilterBlockRecord, sideSize.LoopFilterMap),
			RestorationRecords:       make([]internaltile.RestorationUnitRecord, sideSize.RestorationRecords),
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
			side.LoopFilterMap = internaldecoder.FrameWorkLoopFilterMap{Stride: 1, Rows: 1}
		}

		err = setDecoderFrameWorkBatchResidualRunnerSideData(&runner, side)
		if activeRestoration && omitRestorationScratch {
			if !errors.Is(err, internalframe.ErrShortBuffer) {
				t.Fatalf("missing restoration requests err=%v want %v", err, internalframe.ErrShortBuffer)
			}
			return
		}
		if corruptLoopMap {
			if !errors.Is(err, internalthreading.ErrInvalidBatch) {
				t.Fatalf("corrupt loop map err=%v want %v", err, internalthreading.ErrInvalidBatch)
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

func FuzzPublicRunDecoderFrameWorkEventWithResidualRunner(f *testing.F) {
	f.Add([]byte{0x00, 0x00}, uint8(64), uint8(0), true)
	f.Add([]byte{0xff, 0x80, 0x00, 0x7f}, uint8(3), uint8(1), false)
	f.Add([]byte{0xaa, 0x55, 0x11, 0xee, 0x00, 0x42}, uint8(127), uint8(2), true)

	f.Fuzz(func(t *testing.T, payload []byte, rawQ uint8, rawCDEFBits uint8, sideData bool) {
		if len(payload) < 2 || len(payload) > 256 {
			return
		}
		firstSize := len(payload) / 2
		if firstSize == 0 || firstSize > 256 {
			return
		}
		tilePayload := make([]byte, len(payload)+1)
		tilePayload[0] = byte(firstSize - 1)
		copy(tilePayload[1:], payload[:firstSize])
		copy(tilePayload[1+firstSize:], payload[firstSize:])

		sequence := SequenceHeader{
			EnableCDEF: true,
			ColorConfig: internalparser.ColorConfig{
				BitDepth:   8,
				MonoChrome: true,
			},
		}
		cdefBits := rawCDEFBits & 3
		cdef := internalparser.CDEFParams{
			Bits:          cdefBits,
			StrengthCount: uint8(1) << cdefBits,
		}
		event := publicDecoderResidualRunnerFrameEvent()
		event.Unit.Payload = tilePayload
		event.CDEF = cdef
		event.Quantization.BaseQIdx = rawQ | 1

		var scratchSpans [2]internalparser.TileSpan
		var scratchJobs [2]internaltile.Job
		var scratchBatches [1]internalthreading.Batch
		eventScratch, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
		if err != nil {
			t.Fatal(err)
		}
		runner, err := bindDecoderFrameWorkBatchResidualRunner(eventScratch.Runner, publicDecoderBatchResidualRunnerScratch(eventScratch.Runner))
		if err != nil {
			t.Fatal(err)
		}

		var side *decoderFrameWorkSideData
		var boundSide decoderFrameWorkSideData
		if sideData {
			boundSide, err = bindDecoderFrameWorkResidualEventSideData(sequence, &event, publicDecoderFrameWorkSideDataScratch(eventScratch.SideData))
			if err != nil {
				t.Fatal(err)
			}
			side = &boundSide
		}

		workerPool, err := internalthreading.NewPool(1)
		if err != nil {
			t.Fatal(err)
		}
		defer workerPool.Close()

		pool := publicDecoderPostFilterFramePool(t, FrameFormat{
			Width:      int(event.FrameSize.CodedWidth),
			Height:     int(event.FrameSize.Height),
			BitDepth:   8,
			MonoChrome: true,
			Align:      64,
		}, 1)
		var refs internaldecoder.SurfaceReferences
		var state internaldecoder.FrameWorkState
		var referenceSurfaces [internalparser.InterRefsPerFrame]int
		var referenceFrames [internalparser.InterRefsPerFrame]*Frame
		var spans [2]internalparser.TileSpan
		var jobs [2]internaltile.Job
		var batches [1]internalthreading.Batch
		var releases [internalparser.RefFrames]int

		result, err := runDecoderFrameWorkEventWithResidualRunner(decoderFrameWorkResidualEventRequest{
			State:             &state,
			Refs:              &refs,
			FramePool:         &pool,
			Sequence:          sequence,
			Event:             event,
			Align:             64,
			ReferenceSurfaces: referenceSurfaces[:],
			ReferenceFrames:   referenceFrames[:],
			Workers:           1,
			Spans:             spans[:],
			Jobs:              jobs[:],
			Batches:           batches[:],
			Releases:          releases[:],
			WorkerPool:        workerPool,
			Runner:            &runner,
			SideData:          side,
		})
		if err != nil {
			return
		}
		if result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) ||
			result.Output == nil ||
			state.Active() {
			t.Fatalf("event runner result=%+v output=%p active=%v", result, result.Output, state.Active())
		}
		if _, ok := refs.ReferenceSlot(0); !ok {
			t.Fatal("event runner did not publish decoded key frame")
		}
	})
}

func publicBindResidualFuzzSideMaps(t *testing.T, batch *internaldecoder.FrameWorkBatch) {
	t.Helper()
	batch.CDEF = internalparser.CDEFParams{Bits: 0, StrengthCount: 1}
	sequence := SequenceHeader{
		Use128x128Superblock: batch.Sequence.Use128x128Superblock,
		ColorConfig:          batch.Sequence.ColorConfig,
	}
	_, _, cdefLength, err := decoderFrameWorkCDEFIndexMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	cdefMap, err := bindDecoderFrameWorkCDEFIndexMap(sequence, batch.FrameSize, batch.CDEF, make([]uint8, cdefLength), make([]bool, cdefLength))
	if err != nil {
		t.Fatal(err)
	}
	_, _, loopLength, err := decoderFrameWorkLoopFilterMapShape(sequence, batch.FrameSize)
	if err != nil {
		t.Fatal(err)
	}
	loopMap, err := bindDecoderFrameWorkLoopFilterMap(sequence, batch.FrameSize, make([]internalthreading.FrameWorkLoopFilterBlockRecord, loopLength))
	if err != nil {
		t.Fatal(err)
	}
	batch.CDEFIndexMap = &cdefMap
	batch.LoopFilterMap = &loopMap
}
