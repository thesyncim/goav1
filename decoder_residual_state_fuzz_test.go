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
