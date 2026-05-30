package goav1_test

import (
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func FuzzPublicDecoderFrameWorkSideDataBinding(f *testing.F) {
	f.Add(uint16(64), uint16(64), uint8(8), false, false, uint8(0), uint8(0), false)
	f.Add(uint16(128), uint16(96), uint8(10), true, true, uint8(1), uint8(1), false)
	f.Add(uint16(320), uint16(180), uint8(12), true, false, uint8(2), uint8(2), true)

	f.Fuzz(func(t *testing.T, rawWidth uint16, rawHeight uint16, rawBitDepth uint8, ssX bool, ssY bool, rawCDEFBits uint8, rawRestoration uint8, shortScratch bool) {
		width := uint32(rawWidth%512) + 1
		height := uint32(rawHeight%512) + 1
		bitDepth := uint8(8)
		switch rawBitDepth % 3 {
		case 1:
			bitDepth = 10
		case 2:
			bitDepth = 12
		}
		restorationType := publicFuzzRestorationType(rawRestoration)
		sequence := av1.SequenceHeader{
			EnableCDEF:        true,
			EnableRestoration: true,
			ColorConfig: av1.ColorConfig{
				BitDepth:     bitDepth,
				SubsamplingX: ssX,
				SubsamplingY: ssY,
			},
		}
		size := av1.FrameSize{
			CodedWidth:          width,
			UpscaledWidth:       width,
			Height:              height,
			SuperResDenominator: 8,
		}
		cdefBits := rawCDEFBits & 3
		cdef := av1.CDEFParams{
			Bits:          cdefBits,
			StrengthCount: uint8(1) << cdefBits,
		}
		restoration := av1.RestorationParams{
			Type:       [3]av1.RestorationType{restorationType, restorationType, restorationType},
			UnitSizeY:  64,
			UnitSizeUV: 64,
		}
		if sequence.ColorConfig.MonoChrome {
			restoration.Type[1] = av1.RestorationNone
			restoration.Type[2] = av1.RestorationNone
		}

		scratchSize, err := av1.DecoderFrameWorkSideDataScratchLen(sequence, size, cdef, restoration)
		if err != nil {
			return
		}
		scratch := av1.DecoderFrameWorkSideDataScratch{
			CDEFIndexMap:             make([]uint8, scratchSize.CDEFIndexMap),
			CDEFReadMap:              make([]bool, scratchSize.CDEFReadMap),
			LoopFilterMap:            make([]av1.DecoderFrameWorkLoopFilterBlockRecord, scratchSize.LoopFilterMap),
			RestorationRecords:       make([]av1.TileRestorationUnitRecord, scratchSize.RestorationRecords),
			RestorationBoundaryAbove: make([]uint16, scratchSize.RestorationBoundaryAbove),
			RestorationBoundaryBelow: make([]uint16, scratchSize.RestorationBoundaryBelow),
		}
		if shortScratch && len(scratch.LoopFilterMap) != 0 {
			scratch.LoopFilterMap = scratch.LoopFilterMap[:len(scratch.LoopFilterMap)-1]
		}
		side, err := av1.BindDecoderFrameWorkSideData(sequence, size, cdef, restoration, scratch)
		if err != nil {
			return
		}
		if shortScratch {
			t.Fatal("short side-data scratch unexpectedly bound")
		}
		postSide := av1.DecoderFrameWorkPostFilterSideData(side)
		if len(postSide.CDEFIndexMap.Index) != scratchSize.CDEFIndexMap ||
			len(postSide.CDEFIndexMap.Read) != scratchSize.CDEFReadMap ||
			len(postSide.LoopFilterMap.Records) != scratchSize.LoopFilterMap ||
			side.RestorationFrameBuffers.Plan.UnitRecordLen() != scratchSize.RestorationRecords {
			t.Fatalf("side data=%+v post=%+v scratch=%+v", side, postSide, scratchSize)
		}
	})
}

func FuzzPublicDecoderFrameWorkPostFilterScratchContext(f *testing.F) {
	f.Add(uint16(8), uint16(32), uint16(13), uint8(0), uint8(0))
	f.Add(uint16(64), uint16(64), uint16(64), uint8(1), uint8(1))
	f.Add(uint16(128), uint16(72), uint16(160), uint8(2), uint8(7))

	f.Fuzz(func(t *testing.T, rawCodedWidth uint16, rawHeight uint16, rawUpscaledWidth uint16, rawAlign uint8, flags uint8) {
		codedWidth := uint32(rawCodedWidth%256) + 1
		height := uint32(rawHeight%256) + 1
		upscaledWidth := max(uint32(rawUpscaledWidth%320)+1, codedWidth)
		align := 1 << (rawAlign % 7)
		mono := flags&1 != 0
		superRes := flags&2 != 0
		filmGrain := flags&4 != 0
		bitDepth := uint8(8)
		switch (flags >> 3) & 3 {
		case 1:
			bitDepth = 10
		case 2:
			bitDepth = 12
		}

		sequence := av1.SequenceHeader{
			EnableSuperRes: superRes,
			ColorConfig: av1.ColorConfig{
				BitDepth:     bitDepth,
				MonoChrome:   mono,
				SubsamplingX: !mono,
				SubsamplingY: !mono,
			},
		}
		size := av1.FrameSize{
			CodedWidth:          codedWidth,
			UpscaledWidth:       codedWidth,
			Height:              height,
			SuperResDenominator: 8,
		}
		if superRes {
			size.SuperResEnabled = true
			size.UpscaledWidth = upscaledWidth
			size.SuperResDenominator = 9 + uint8(flags&7)
		}
		event := av1.DecoderEvent{
			Kind:           av1.DecoderEventTileGroup,
			SequenceHeader: sequence,
			FrameSize:      size,
			TileGroup:      av1.TileGroup{Final: true},
		}
		if filmGrain {
			event.FilmGrain = av1.FilmGrainParams{
				ParamsPresent: true,
				Apply:         true,
				Seed:          uint16(0x1000) | uint16(flags),
				BitDepth:      bitDepth,
				NumYPoints:    1,
				YPoints:       [av1.MaxFilmGrainYPoints][2]uint8{{0, 64}},
				ScalingShift:  8,
				Overlap:       true,
			}
		}

		var scratchOutput av1.Frame
		ctx, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, align, nil, &scratchOutput)
		if err != nil {
			return
		}
		if ctx.Output != &scratchOutput || len(scratchOutput.Y.Pix) != 0 || scratchOutput.Format.Width != int(codedWidth) {
			t.Fatalf("ctx=%+v output=%+v codedWidth=%d", ctx, scratchOutput, codedWidth)
		}

		var runner av1.DecoderFrameWorkCallerPostFilterScratchRunner
		first, err := runner.ScratchLen(ctx)
		if err != nil {
			return
		}
		runner.Scratch = publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(first))
		full, err := runner.ScratchLen(ctx)
		if err != nil {
			return
		}
		if superRes && full.SuperRes.OutputFrame == 0 {
			t.Fatalf("superres scratch missing: first=%+v full=%+v", first, full)
		}
		if filmGrain && full.FilmGrain.LumaSamples == 0 {
			t.Fatalf("film grain scratch missing: first=%+v full=%+v", first, full)
		}
	})
}

func publicFuzzRestorationType(raw uint8) av1.RestorationType {
	switch raw % 3 {
	case 1:
		return av1.RestorationWiener
	case 2:
		return av1.RestorationSGRProj
	default:
		return av1.RestorationNone
	}
}
