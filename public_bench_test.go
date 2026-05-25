package goav1_test

import (
	"testing"

	av1 "github.com/thesyncim/goav1"
)

var publicBenchmarkSink int

func BenchmarkPublicLowOverheadOBUIterator(b *testing.B) {
	stream := publicBenchmarkLowOverheadStream()
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		it := av1.NewLowOverheadIterator(stream)
		count := 0
		payloadBytes := 0
		for {
			unit, ok, err := it.Next()
			if err != nil {
				b.Fatal(err)
			}
			if !ok {
				break
			}
			count++
			payloadBytes += len(unit.Payload)
		}
		if count != 13 {
			b.Fatalf("count=%d want 13", count)
		}
		sum += payloadBytes
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicTemporalUnitIterator(b *testing.B) {
	stream := publicBenchmarkTemporalUnitStream()
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		it := av1.NewTemporalUnitIterator(stream)
		count := 0
		rawBytes := 0
		for {
			unit, ok, err := it.Next()
			if err != nil {
				b.Fatal(err)
			}
			if !ok {
				break
			}
			count++
			rawBytes += len(unit.Raw)
		}
		if count != 4 {
			b.Fatalf("count=%d want 4", count)
		}
		sum += rawBytes
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicAnnexBIterator(b *testing.B) {
	stream := publicBenchmarkAnnexBStream()
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		it := av1.NewAnnexBIterator(stream)
		count := 0
		rawBytes := 0
		for {
			unit, ok, err := it.Next()
			if err != nil {
				b.Fatal(err)
			}
			if !ok {
				break
			}
			count++
			rawBytes += len(unit.Raw)
		}
		if count != 9 {
			b.Fatalf("count=%d want 9", count)
		}
		sum += rawBytes
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicRTPPacketizeAndAssemble(b *testing.B) {
	frame := publicBenchmarkRTPFrame()
	limits := av1.RTPPayloadSizeLimits{MaxPayloadLen: 120}
	var packetizerOBUs [16]av1.RTPPacketizerOBU
	var packets [32]av1.RTPPacketPlan
	var work [32]av1.RTPPacketPlan
	var packetBytes [32][160]byte
	var payloads [32][]byte
	var assembled [1024]byte
	var frameOBUs [16]av1.RTPFrameOBU

	size, err := av1.RTPPacketizerScratchLen(frame, limits, packetizerOBUs[:])
	if err != nil {
		b.Fatal(err)
	}
	if size.OBUs > len(packetizerOBUs) || size.Packets > len(packets) || size.Work > len(work) {
		b.Fatalf("scratch size=%+v exceeds benchmark storage", size)
	}

	b.SetBytes(int64(len(frame)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		firstPass, err := av1.RTPPacketizerScratchLen(frame, limits, nil)
		if err != nil {
			b.Fatal(err)
		}
		if firstPass.OBUs != size.OBUs {
			b.Fatalf("first pass OBUs=%d want %d", firstPass.OBUs, size.OBUs)
		}
		size, err := av1.RTPPacketizerScratchLen(frame, limits, packetizerOBUs[:])
		if err != nil {
			b.Fatal(err)
		}
		packetizer, err := av1.NewRTPPacketizer(frame, limits, true, true, packetizerOBUs[:size.OBUs], packets[:size.Packets], work[:size.Work])
		if err != nil {
			b.Fatal(err)
		}
		count := 0
		for {
			if count >= len(packetBytes) {
				b.Fatal("too many RTP payloads")
			}
			n, _, ok, err := packetizer.NextPacket(packetBytes[count][:])
			if err != nil {
				b.Fatal(err)
			}
			if !ok {
				break
			}
			payloads[count] = packetBytes[count][:n]
			count++
		}
		assembledLen, obuCount, err := av1.AssembleRTPFrameSize(payloads[:count])
		if err != nil {
			b.Fatal(err)
		}
		wrote, assembledCount, err := av1.AssembleRTPFrame(assembled[:assembledLen], payloads[:count], frameOBUs[:obuCount])
		if err != nil {
			b.Fatal(err)
		}
		if wrote != len(frame) || assembledCount != 5 {
			b.Fatalf("assembled=%d,%d want %d,5", wrote, assembledCount, len(frame))
		}
		sum += count + wrote
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicFrameBindAndSampleRoundTrip(b *testing.B) {
	format := av1.FrameFormat{
		Width:        128,
		Height:       72,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}
	layout, err := av1.FrameRequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	frame, err := av1.BindFrame(backing, format)
	if err != nil {
		b.Fatal(err)
	}
	sampleLen, err := av1.FrameSamplePlaneLen(frame.Y, 1)
	if err != nil {
		b.Fatal(err)
	}
	samples := make([]uint16, sampleLen)

	b.SetBytes(int64(len(backing)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		frame, err := av1.BindFrame(backing, format)
		if err != nil {
			b.Fatal(err)
		}
		plane, err := av1.LoadFrameSamplePlane(samples, frame.Y, 1)
		if err != nil {
			b.Fatal(err)
		}
		if err := av1.StoreFrameSamplePlane(frame.Y, 1, plane); err != nil {
			b.Fatal(err)
		}
		sum += plane.Stride + frame.Layout.Size
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicPredictionReconstructionAndLoopFilter(b *testing.B) {
	plane := publicPredictionPlane(64, 64, 1, 64)
	ref := publicPredictionPlane(64, 64, 1, 64)
	fillPublicMotionPlane(ref, 1)
	edges := av1.PredictionIntraEdges{
		Above:              make([]uint16, 16),
		Left:               make([]uint16, 16),
		AboveLeft:          96,
		AboveAvailable:     true,
		LeftAvailable:      true,
		AboveLeftAvailable: true,
	}
	for i := range edges.Above {
		edges.Above[i] = uint16(90 + i)
		edges.Left[i] = uint16(70 + i)
	}
	quantized := make([]int16, 8*8)
	quantized[0] = 4
	quantized[1] = -2
	quantized[8] = 3
	reconstruct := av1.ReconstructBlock{
		Size:      av1.TransformSize{Width: 8, Height: 8},
		Transform: av1.TransformTypeDCTDCT,
		Quantizer: av1.Quantizer{DC: 4, AC: 8},
	}
	int32Len, int16Len, err := av1.ReconstructBlockScratchLen(reconstruct)
	if err != nil {
		b.Fatal(err)
	}
	int32Scratch := make([]int32, int32Len)
	residualScratch := make([]int16, int16Len)
	thresholds := av1.LoopFilterThresholds{Limit: 20, BlockLimit: 48, HighEdgeVariance: 1}

	b.SetBytes(64 * 64)
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := av1.FillPlaneBlock(plane, 1, 0, 0, plane.Width, plane.Height, 64); err != nil {
			b.Fatal(err)
		}
		if err := av1.PredictIntraPlaneBlock(plane, 1, 8, 8, 8, 16, 16, av1.PredictionIntraModePaeth, edges); err != nil {
			b.Fatal(err)
		}
		if err := av1.PredictInterPlaneBlockFromOrigin(plane, ref, 1, 24, 24, 20, 20, 16, 16, 0, 0); err != nil {
			b.Fatal(err)
		}
		if err := av1.ReconstructPlaneBlock(plane, 1, 8, 8, 8, quantized, 8, int32Scratch, residualScratch, reconstruct); err != nil {
			b.Fatal(err)
		}
		if err := av1.ApplyLoopFilter4Edge(plane, 1, 8, av1.LoopFilterEdgeHorizontal, 8, 16, 16, thresholds); err != nil {
			b.Fatal(err)
		}
		min, max, err := av1.MinMaxAbsDiff8x8(plane.Pix[8*plane.Stride+8:], plane.Stride, ref.Pix[8*ref.Stride+8:], ref.Stride, 1)
		if err != nil {
			b.Fatal(err)
		}
		sum += int(min) + int(max)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderPredictionBridge(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	visit := publicDecoderPredictionIntraVisit(av1.TileIntraModeDC)
	var scratch av1.DecoderFrameWorkIntraPredictionScratch
	publicSeedDecoderPredictionIntraEdges(output, 10, 50)

	b.SetBytes(16 * 16)
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := av1.PredictDecoderFrameWorkBlockLumaIntra(batch, 0, visit, &scratch); err != nil {
			b.Fatal(err)
		}
		sum += int(output.Y.Pix[16*output.Y.Stride+16])
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicTileCoefficientReplay(b *testing.B) {
	payload := make([]byte, 16)
	job := av1.TileJob{Offset: 0, Size: len(payload)}
	req := av1.TileLumaCoeffTreeRequest{
		TreeRequest: av1.TileTransformTreeRequest{Size: av1.TileBlockSize4x4, VisibleW4: 1, VisibleH4: 1},
		Tree:        av1.TileTransformTreeResult{Y: av1.TileTransformSize4x4},
		Class:       av1.TransformClass2D,
	}
	var cdfs av1.TileCoeffCDFs
	var state av1.TileDecodeState
	var ctx av1.TileCoeffEntropyContext
	var scratch av1.TileCoeffTreeScratch

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := av1.InitTileCoeffCDFsDefault(&cdfs, 0); err != nil {
			b.Fatal(err)
		}
		if err := av1.ResetTileDecodeState(&state, payload, job, av1.TileDecodeOptions{}); err != nil {
			b.Fatal(err)
		}
		ctx.Reset()
		stats, err := av1.DecodeTileLumaCoefficients(&state, &cdfs, &ctx, &scratch, req, func(block av1.TileLumaCoeffBlock) error {
			sum += block.Result.EOB
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
		sum += stats.TXBs
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicOutputFilters(b *testing.B) {
	superSrc := av1.FrameSamplePlane{Pix: make([]uint16, 128*32), Stride: 128, Width: 128, Height: 32}
	superDst := av1.FrameSamplePlane{Pix: make([]uint16, 192*32), Stride: 192, Width: 192, Height: 32}
	for i := range superSrc.Pix {
		superSrc.Pix[i] = uint16(i & 0xff)
	}
	lut := publicFilmGrainScaling(64)
	rowDst, rowSrc := publicFilmGrainRowBuffers(128, 4, 128, 100)
	lumaGrain := make([]int16, av1.FilmGrainLumaGrainSamples)
	publicSetFilmGrainLuma(lumaGrain, 0xd9, 0, 0, 0, 0, 64)
	lumaParams := av1.FilmGrainLumaRowParams{
		Seed:         0,
		Width:        128,
		Height:       4,
		Stride:       128,
		BitDepth:     8,
		ScalingShift: 8,
	}

	b.SetBytes(int64(len(superSrc.Pix)*2 + len(rowSrc)*2))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := av1.UpscaleSuperResPlane(superSrc, superDst, 8); err != nil {
			b.Fatal(err)
		}
		if err := av1.ApplyFilmGrainLumaRow(rowDst, rowSrc, lumaGrain, lut[:], lumaParams); err != nil {
			b.Fatal(err)
		}
		sum += int(superDst.Pix[i%len(superDst.Pix)]) + int(rowDst[i%len(rowDst)])
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderPostFilterBinding(b *testing.B) {
	sequence := publicDecoderPostFilterSequence()
	size := av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	_, _, loopFilterLength, err := av1.DecoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		b.Fatal(err)
	}
	loopFilterRecords := make([]av1.DecoderFrameWorkLoopFilterBlockRecord, loopFilterLength)
	loopFilterEdges := make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, 4)
	loopFilterSize := av1.DecoderFrameWorkLoopFilterPostFilterScratchSize{Edges: len(loopFilterEdges)}
	index := make([]uint8, 1)
	read := make([]bool, 1)
	cdefSize := av1.DecoderFrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{64, 16, 16},
		Dst:           [3]int{64, 16, 16},
		DirectionGrid: 1,
		VarianceGrid:  1,
		Input:         av1.CDEFInputBufferSize,
		UnitDst:       av1.CDEFInputBufferSize,
	}
	var sampleScratch [3][]uint16
	var dstScratch [3][]uint16
	for plane := 0; plane < 3; plane++ {
		sampleScratch[plane] = make([]uint16, cdefSize.Samples[plane])
		dstScratch[plane] = make([]uint16, cdefSize.Dst[plane])
	}
	directionGrid := make([]av1.CDEFDirectionGrid, cdefSize.DirectionGrid)
	varianceGrid := make([]av1.CDEFVarianceGrid, cdefSize.VarianceGrid)
	inputScratch := make([]uint16, cdefSize.Input)
	unitDstScratch := make([]uint16, cdefSize.UnitDst)

	b.SetBytes(int64(loopFilterLength + len(index) + cdefSize.Input + cdefSize.UnitDst))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		loopMap, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, size, loopFilterRecords)
		if err != nil {
			b.Fatal(err)
		}
		loopReq, err := av1.BindDecoderFrameWorkLoopFilterPostFilterRequest(loopFilterSize, loopMap, loopFilterEdges)
		if err != nil {
			b.Fatal(err)
		}
		cdefMap, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, size, av1.CDEFParams{Bits: 1, StrengthCount: 2}, index, read)
		if err != nil {
			b.Fatal(err)
		}
		cdefReq, err := av1.BindDecoderFrameWorkCDEFPostFilterRequest(cdefSize, cdefMap, sampleScratch, dstScratch, directionGrid, varianceGrid, inputScratch, unitDstScratch)
		if err != nil {
			b.Fatal(err)
		}
		sum += loopMap.Stride + loopReq.Map.Rows + len(loopReq.Edges) + cdefReq.IndexMap.Rows
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualState(b *testing.B) {
	var initial av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&initial, 64); err != nil {
		b.Fatal(err)
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
	blockLoopBatch := publicBenchmarkDecoderBlockLoopBatch()
	rootColumns, err := av1.DecoderFrameWorkJobBlockLoopContextRootColumns(blockLoopBatch, 0)
	if err != nil {
		b.Fatal(err)
	}
	above := make([]av1.TileBlockLoopRootAboveContext, rootColumns)
	segMap := make([]uint8, 96*96)

	b.SetBytes(int64(len(batch.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := av1.InitDecoderFrameWorkTileResidualCDFStorage(batch, &storage); err != nil {
			b.Fatal(err)
		}
		cdfs := av1.DecoderFrameWorkTileResidualCDFsFromStorage(&storage)
		if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 1, &state); err != nil {
			b.Fatal(err)
		}
		if err := av1.RetainDecoderFrameWorkTileResidualCDFStorage(batch, 1, &state, &storage); err != nil {
			b.Fatal(err)
		}
		carrier, err := av1.BindTileBlockLoopContextCarrier(rootColumns, above)
		if err != nil {
			b.Fatal(err)
		}
		req, err := av1.DecoderFrameWorkJobBlockLoopRequest(blockLoopBatch, 0, segMap, nil, 96, &carrier)
		if err != nil {
			b.Fatal(err)
		}
		if cdfs.TransformType == nil || !retainedValid || req.ContextCarrier != &carrier {
			b.Fatal("residual state not initialized")
		}
		sum += int(state.CurrentBaseQIdx) + int(req.SBSizeMIB)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualDecode(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := av1.DecoderFrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 256),
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
				ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    av1.FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8},
			Quantization: av1.QuantizationParams{BaseQIdx: 64},
			TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
		},
		Jobs: []av1.TileJob{{SBCols: 1, SBRows: 1, Offset: 0, Size: 256}},
	}
	var state av1.TileDecodeState
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		b.Fatal(err)
	}
	cdfs := av1.DecoderFrameWorkTileResidualCDFsFromStorage(&storage)
	loopReq, err := av1.DecoderFrameWorkJobBlockLoopRequest(batch, 0, nil, nil, 0, nil)
	if err != nil {
		b.Fatal(err)
	}
	int32Len, int16Len, err := av1.DecoderFrameWorkResidualScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlaneY, av1.TransformSize{Width: 64, Height: 64}, av1.TransformTypeDCTDCT)
	if err != nil {
		b.Fatal(err)
	}
	var scratch av1.DecoderFrameWorkTileResidualScratch
	req := av1.DecoderFrameWorkTileResidualRequest{
		Loop:          loopReq,
		TransformMode: batch.TransformRef.TransformMode,
		Transforms: func(visit av1.TileBlockLoopVisit) (av1.DecoderFrameWorkBlockTransforms, error) {
			return av1.ReadDecoderFrameWorkInterBlockTransforms(batch, &state, visit)
		},
		Int32Scratch:    make([]int32, int32Len),
		ResidualScratch: make([]int16, int16Len),
	}

	b.SetBytes(int64(len(batch.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := av1.InitDecoderFrameWorkJobDecodeState(batch, 0, &state); err != nil {
			b.Fatal(err)
		}
		stats, err := av1.DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, &state, cdfs, &scratch, req)
		if err != nil {
			b.Fatal(err)
		}
		sum += stats.Residuals + stats.TXBs
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderBlockCoeffReconstruction(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderBlockCoeffSimpleBatch(output)
	req := publicDecoderBlockCoeffSimpleRequest()
	req.Int32Scratch, req.ResidualScratch = publicDecoderBlockCoeffScratch(b, batch, req, av1.DecoderFrameWorkPlaneY)

	b.SetBytes(4 * 4 * 2)
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		fillPublicReconstructPlane(output.Y, output.Layout.BytesPerSample, 128)
		if err := av1.ReconstructDecoderFrameWorkBlockCoeff(batch, 0, req); err != nil {
			b.Fatal(err)
		}
		sum += int(output.Y.Pix[i%len(output.Y.Pix)])
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderCoeffReplayReconstructionAdapter(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderBlockCoeffSimpleBatch(output)
	req := publicDecoderBlockCoeffSimpleRequest()
	ctx := publicDecoderBlockCoeffReplayContext(b, batch, req, av1.DecoderFrameWorkPlaneY)
	block := av1.TileLumaCoeffBlock{
		Block:     req.Block.Block,
		Transform: req.Transform,
		Result:    req.Block.Result,
		Coeffs:    req.Block.Coeffs,
		Scan:      req.Block.Scan,
	}

	b.SetBytes(4 * 4 * 2)
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		fillPublicReconstructPlane(output.Y, output.Layout.BytesPerSample, 128)
		if err := av1.ReconstructDecoderFrameWorkLumaCoeffBlock(batch, 0, ctx, block); err != nil {
			b.Fatal(err)
		}
		sum += int(output.Y.Pix[i%len(output.Y.Pix)])
	}
	publicBenchmarkSink = sum
}

func publicBenchmarkDecoderFrame(b *testing.B, format av1.FrameFormat) *av1.Frame {
	b.Helper()
	layout, err := av1.FrameRequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	frame, err := av1.BindFrame(make([]byte, layout.Size), format)
	if err != nil {
		b.Fatal(err)
	}
	return &frame
}

func publicBenchmarkDecoderBlockLoopBatch() av1.DecoderFrameWorkBatch {
	return av1.DecoderFrameWorkBatch{
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
				Use128x128Superblock: true,
				EnableDualFilter:     true,
				EnableFilterIntra:    true,
				EnableOrderHint:      true,
				OrderHintBits:        5,
				ColorConfig:          av1.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameHeader:         av1.FrameHeaderPrefix{OrderHint: 9},
			FrameSize:           av1.FrameSize{CodedWidth: 300, UpscaledWidth: 300, Height: 260, SuperResDenominator: 8},
			TileInfo:            av1.TileInfo{InterpolationFilter: av1.InterpolationSwitchable, UseRefFrameMVS: true},
			ReferenceOrderHints: [av1.InterRefsPerFrame]uint32{1, 9, 10, 4, 5, 6, 7},
			SkipMode:            av1.SkipModeParams{Allowed: true, Enabled: true},
			CDEF:                av1.CDEFParams{Bits: 2, StrengthCount: 4},
			Delta:               av1.DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1},
		},
		Jobs: []av1.TileJob{{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2}},
	}
}

func publicBenchmarkLowOverheadStream() []byte {
	stream := make([]byte, 0, 512)
	stream = appendPublicLowOverheadOBU(stream, av1.OBUTemporalDelimiter, nil)
	stream = appendPublicLowOverheadOBU(stream, av1.OBUSequenceHeader, []byte{0x12, 0x34, 0x56})
	for i := 0; i < 10; i++ {
		payload := make([]byte, 24)
		for j := range payload {
			payload[j] = byte(i*17 + j)
		}
		stream = appendPublicLowOverheadOBU(stream, av1.OBUTileGroup, payload)
	}
	stream = appendPublicLowOverheadOBU(stream, av1.OBUFrame, []byte{0xaa, 0xbb, 0xcc})
	return stream
}

func publicBenchmarkTemporalUnitStream() []byte {
	stream := make([]byte, 0, 512)
	for i := 0; i < 4; i++ {
		stream = appendPublicLowOverheadOBU(stream, av1.OBUTemporalDelimiter, nil)
		stream = appendPublicLowOverheadOBU(stream, av1.OBUFrameHeader, []byte{byte(i)})
		for j := 0; j < 2; j++ {
			stream = appendPublicLowOverheadOBU(stream, av1.OBUTileGroup, []byte{byte(i), byte(j), byte(i + j)})
		}
	}
	return stream
}

func publicBenchmarkAnnexBStream() []byte {
	td := []byte{byte(av1.OBUTemporalDelimiter) << 3}
	seq := []byte{byte(av1.OBUSequenceHeader) << 3, 0x01, 0x02}
	fh0 := []byte{byte(av1.OBUFrameHeader) << 3, 0x10}
	fh1 := []byte{byte(av1.OBUFrameHeader) << 3, 0x20}
	tile0 := []byte{byte(av1.OBUTileGroup) << 3, 0x30, 0x31, 0x32}
	tile1 := []byte{byte(av1.OBUTileGroup) << 3, 0x40, 0x41, 0x42}
	frame0 := []byte{byte(av1.OBUFrame) << 3, 0x50}
	frame1 := []byte{byte(av1.OBUFrame) << 3, 0x60}
	return appendPublicAnnexBStream(nil,
		[][][]byte{{td, seq}, {fh0, tile0, tile1}},
		[][][]byte{{td, frame0}, {fh1, frame1}},
	)
}

func publicBenchmarkRTPFrame() []byte {
	frame := make([]byte, 0, 512)
	frame = appendPublicLowOverheadOBU(frame, av1.OBUSequenceHeader, []byte{0xaa, 0xbb, 0xcc})
	for i := 0; i < 3; i++ {
		payload := make([]byte, 96)
		for j := range payload {
			payload[j] = byte(i*29 + j)
		}
		frame = appendPublicLowOverheadOBU(frame, av1.OBUTileGroup, payload)
	}
	frame = appendPublicLowOverheadOBU(frame, av1.OBUFrame, []byte{0xdd, 0xee})
	return frame
}
