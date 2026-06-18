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
			next, ok := packetizer.NextPacketSize()
			if !ok {
				break
			}
			n, _, ok, err := packetizer.NextPacket(packetBytes[count][:next])
			if err != nil {
				b.Fatal(err)
			}
			if !ok || n != next {
				b.Fatalf("packet n=%d ok=%v want %d,true", n, ok, next)
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

func BenchmarkPublicDecodeRTPPayload(b *testing.B) {
	rtpPayloads, wantVisible := publicDecoderAllocRTPPayloads(b)
	dec, err := av1.NewDecoderFromRTPPayloads(rtpPayloads)
	if err != nil {
		b.Fatalf("NewDecoderFromRTPPayloads: %v", err)
	}
	defer dec.Close()

	bytesPerRun := 0
	for _, payload := range rtpPayloads {
		bytesPerRun += len(payload)
	}
	decodeAll := func() int {
		if err := dec.Reset(); err != nil {
			b.Fatalf("Reset: %v", err)
		}
		visible := 0
		for i, payload := range rtpPayloads {
			frames, err := dec.DecodeRTPPayload(payload)
			if err != nil {
				b.Fatalf("DecodeRTPPayload packet %d: %v", i, err)
			}
			visible += len(frames)
		}
		if visible != wantVisible {
			b.Fatalf("decoded %d visible RTP frames, want %d", visible, wantVisible)
		}
		return visible
	}

	publicBenchmarkSink = decodeAll()
	b.SetBytes(int64(bytesPerRun))
	b.ReportAllocs()
	b.ResetTimer()
	sum := 0
	for i := 0; i < b.N; i++ {
		sum += decodeAll()
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
	job := av1.TileJob{Offset: 0, Size: uint32(len(payload))}
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
			sum += int(block.Result.EOB)
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
		sum += int(stats.TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicTileBlockCoefficientDecode(b *testing.B) {
	payload := make([]byte, 32)
	job := av1.TileJob{Offset: 0, Size: uint32(len(payload))}
	req := av1.TileBlockCoeffRequest{
		Transform: av1.TileTransformTreeRequest{
			Size:          av1.TileBlockSize8x8,
			VisibleW4:     2,
			VisibleH4:     2,
			Color:         av1.ColorConfig{MonoChrome: true},
			Inter:         true,
			TransformMode: av1.TransformModeLargest,
		},
		LumaType: av1.TransformTypeDCTDCT,
	}
	var transformCDFs av1.TileTransformCDFs
	var coeffCDFs av1.TileCoeffCDFs
	var state av1.TileDecodeState
	var transformCtx av1.TileTransformContext
	var coeffCtx av1.TileCoeffEntropyContext
	var scratch av1.TileBlockCoeffScratch

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := av1.InitTileTransformCDFsDefault(&transformCDFs); err != nil {
			b.Fatal(err)
		}
		if err := av1.InitTileCoeffCDFsDefault(&coeffCDFs, 0); err != nil {
			b.Fatal(err)
		}
		if err := av1.ResetTileDecodeState(&state, payload, job, av1.TileDecodeOptions{}); err != nil {
			b.Fatal(err)
		}
		transformCtx.Reset()
		coeffCtx.Reset()
		result, err := av1.DecodeTileBlockCoefficients(&state, av1.TileBlockCoeffCDFs{Transform: &transformCDFs, Coeff: &coeffCDFs}, &transformCtx, &coeffCtx, &scratch, req, func(block av1.TileBlockCoeffBlock) error {
			sum += int(block.Result.EOB) + int(block.Plane)
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
		sum += int(result.TotalStats().TXBs)
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
	loopFilterSchedule := make([]uint32, len(loopFilterEdges))
	loopFilterSize := av1.DecoderFrameWorkLoopFilterPostFilterScratchSize{Edges: len(loopFilterEdges), Schedule: len(loopFilterSchedule)}
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
	for plane := range 3 {
		sampleScratch[plane] = make([]uint16, cdefSize.Samples[plane])
		dstScratch[plane] = make([]uint16, cdefSize.Dst[plane])
	}
	directionGrid := make([]av1.CDEFDirectionGrid, cdefSize.DirectionGrid)
	varianceGrid := make([]av1.CDEFVarianceGrid, cdefSize.VarianceGrid)
	inputScratch := make([]uint16, cdefSize.Input)
	unitDstScratch := make([]uint16, cdefSize.UnitDst)
	postFilterSize := av1.DecoderFrameWorkPostFilterScratchSize{
		LoopFilter: loopFilterSize,
		CDEF:       cdefSize,
	}
	postFilterBuffers := av1.DecoderFrameWorkPostFilterRequestBuffers{
		LoopFilterEdges:    loopFilterEdges,
		LoopFilterSchedule: loopFilterSchedule,

		CDEFSampleScratch:  sampleScratch,
		CDEFDstScratch:     dstScratch,
		CDEFDirectionGrid:  directionGrid,
		CDEFVarianceGrid:   varianceGrid,
		CDEFInputScratch:   inputScratch,
		CDEFUnitDstScratch: unitDstScratch,
	}
	postFilterArenaSize := av1.DecoderFrameWorkPostFilterRequestScratchLen(postFilterSize)
	postFilterSide := av1.DecoderFrameWorkPostFilterRequestSideData{}
	postFilterArena := av1.DecoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:    make([]av1.DecoderFrameWorkLoopFilterPostFilterEdge, postFilterArenaSize.LoopFilterEdges),
		LoopFilterSchedule: make([]uint32, postFilterArenaSize.LoopFilterSchedule),
		CDEFDirectionGrid:  make([]av1.CDEFDirectionGrid, postFilterArenaSize.CDEFDirectionGrid),
		CDEFVarianceGrid:   make([]av1.CDEFVarianceGrid, postFilterArenaSize.CDEFVarianceGrid),
		ByteScratch:        make([]byte, postFilterArenaSize.ByteScratch),
		Uint16Scratch:      make([]uint16, postFilterArenaSize.Uint16Scratch),
		Int16Scratch:       make([]int16, postFilterArenaSize.Int16Scratch),
		Int32Scratch:       make([]int32, postFilterArenaSize.Int32Scratch),
	}

	b.SetBytes(int64(loopFilterLength + len(index) + cdefSize.Input + cdefSize.UnitDst))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		loopMap, err := av1.BindDecoderFrameWorkLoopFilterMap(sequence, size, loopFilterRecords)
		if err != nil {
			b.Fatal(err)
		}
		postFilterBuffers.LoopFilterMap = loopMap
		postFilterSide.LoopFilterMap = loopMap
		loopReq, err := av1.BindDecoderFrameWorkLoopFilterPostFilterRequest(loopFilterSize, loopMap, loopFilterEdges, loopFilterSchedule)
		if err != nil {
			b.Fatal(err)
		}
		cdefMap, err := av1.BindDecoderFrameWorkCDEFIndexMap(sequence, size, av1.CDEFParams{Bits: 1, StrengthCount: 2}, index, read)
		if err != nil {
			b.Fatal(err)
		}
		postFilterBuffers.CDEFIndexMap = cdefMap
		postFilterSide.CDEFIndexMap = cdefMap
		cdefReq, err := av1.BindDecoderFrameWorkCDEFPostFilterRequest(cdefSize, cdefMap, sampleScratch, dstScratch, directionGrid, varianceGrid, inputScratch, unitDstScratch)
		if err != nil {
			b.Fatal(err)
		}
		postReq, err := av1.BindDecoderFrameWorkPostFilterRequest(postFilterSize, postFilterBuffers)
		if err != nil {
			b.Fatal(err)
		}
		arenaReq, err := av1.BindDecoderFrameWorkPostFilterRequestFromScratch(postFilterSize, postFilterSide, postFilterArena)
		if err != nil {
			b.Fatal(err)
		}
		sum += int(loopMap.Stride) + int(loopReq.Map.Rows) + len(loopReq.Edges) + int(cdefReq.IndexMap.Rows) + len(postReq.CDEF.InputScratch) + len(arenaReq.CDEF.InputScratch)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderPostFilterScratchContext(b *testing.B) {
	sequence := publicDecoderPostFilterSequence()
	sequence.ColorConfig.MonoChrome = true
	sequence.ColorConfig.SubsamplingX = false
	sequence.ColorConfig.SubsamplingY = false
	size := av1.FrameSize{
		CodedWidth:          8,
		UpscaledWidth:       13,
		Height:              32,
		SuperResEnabled:     true,
		SuperResDenominator: 13,
	}
	event := av1.DecoderEvent{
		Kind:           av1.DecoderEventTileGroup,
		SequenceHeader: sequence,
		FrameSize:      size,
		TileGroup:      av1.TileGroup{Final: true},
		FilmGrain: av1.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [av1.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	var output av1.Frame

	b.SetBytes(int64(size.CodedWidth * size.Height))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		ctx, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 32, nil, &output)
		if err != nil {
			b.Fatal(err)
		}
		sum += ctx.Output.Format.Width + output.Layout.Size
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderCallerPostFilterScratchRunner(b *testing.B) {
	sequence := publicDecoderPostFilterSequence()
	sequence.ColorConfig.MonoChrome = true
	sequence.ColorConfig.SubsamplingX = false
	sequence.ColorConfig.SubsamplingY = false
	size := av1.FrameSize{
		CodedWidth:          8,
		UpscaledWidth:       13,
		Height:              32,
		SuperResEnabled:     true,
		SuperResDenominator: 13,
	}
	event := av1.DecoderEvent{
		Kind:           av1.DecoderEventTileGroup,
		SequenceHeader: sequence,
		FrameSize:      size,
		TileGroup:      av1.TileGroup{Final: true},
		FilmGrain: av1.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [av1.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	var scratchOutput av1.Frame
	scratchCtx, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 32, nil, &scratchOutput)
	if err != nil {
		b.Fatal(err)
	}
	var runner av1.DecoderFrameWorkCallerPostFilterScratchRunner
	first, err := runner.ScratchLen(scratchCtx)
	if err != nil {
		b.Fatal(err)
	}
	runner.Scratch = publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(first))
	full, err := runner.ScratchLen(scratchCtx)
	if err != nil {
		b.Fatal(err)
	}
	runner.Scratch = publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(full))

	output := publicBenchmarkDecoderFrame(b, scratchOutput.Format)
	for i := range output.Y.Pix {
		output.Y.Pix[i] = 100
	}
	ctx := av1.DecoderFrameWorkPostFilterContext{Event: event, Output: output}

	b.SetBytes(int64(size.CodedWidth*size.Height + size.UpscaledWidth*size.Height))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := runner.Apply(ctx); err != nil {
			b.Fatal(err)
		}
		if runner.Result.Completed != av1.DecoderFrameWorkPostFilterSuperRes|av1.DecoderFrameWorkPostFilterFilmGrain ||
			!runner.Context.DetachedPostFilterOutput() ||
			runner.Context.Output.Format.Width != int(size.UpscaledWidth) {
			b.Fatalf("result=%+v context=%+v", runner.Result, runner.Context)
		}
		sum += int(runner.Context.Output.Y.Pix[i%len(runner.Context.Output.Y.Pix)])
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderReusableCallerPostFilterRunner(b *testing.B) {
	sequence := publicDecoderPostFilterSequence()
	sequence.ColorConfig.MonoChrome = true
	sequence.ColorConfig.SubsamplingX = false
	sequence.ColorConfig.SubsamplingY = false
	size := av1.FrameSize{
		CodedWidth:          8,
		UpscaledWidth:       13,
		Height:              32,
		SuperResEnabled:     true,
		SuperResDenominator: 13,
	}
	event := av1.DecoderEvent{
		Kind:           av1.DecoderEventTileGroup,
		SequenceHeader: sequence,
		FrameSize:      size,
		TileGroup:      av1.TileGroup{Final: true},
		FilmGrain: av1.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [av1.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	var scratchOutput av1.Frame
	_, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 32, nil, &scratchOutput)
	if err != nil {
		b.Fatal(err)
	}
	output := publicBenchmarkDecoderFrame(b, scratchOutput.Format)
	for i := range output.Y.Pix {
		output.Y.Pix[i] = 100
	}
	ctx := av1.DecoderFrameWorkPostFilterContext{Event: event, Output: output}
	var runner av1.DecoderFrameWorkReusableCallerPostFilterRunner
	if err := runner.Apply(ctx); err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(size.CodedWidth*size.Height + size.UpscaledWidth*size.Height))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := runner.Apply(ctx); err != nil {
			b.Fatal(err)
		}
		out, ok := runner.PostFilterOutput()
		if !ok || out.Format.Width != int(size.UpscaledWidth) {
			b.Fatalf("post output ok=%v output=%+v", ok, out)
		}
		sum += int(out.Y.Pix[i%len(out.Y.Pix)])
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
	int32Len, int16Len, err := av1.DecoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, av1.DecoderFrameWorkPlaneY)
	if err != nil {
		b.Fatal(err)
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
		sum += int(stats.Residuals + stats.TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualBatchDecode(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, av1.FrameFormat{Width: 128, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	publicFillDecoderPostFilterPlane(output.Y)
	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := av1.DecoderFrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 512),
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
				ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    av1.FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8},
			Quantization: av1.QuantizationParams{BaseQIdx: 64},
			TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
		},
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []av1.TileJob{
			{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 256},
			{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 256, Size: 256, UpdatesFrameContext: true},
		},
	}
	var storage av1.DecoderFrameWorkTileResidualCDFStorage
	if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		b.Fatal(err)
	}
	scratchSize, err := av1.DecoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		b.Fatal(err)
	}
	state := &av1.TileDecodeState{}
	var scratch av1.DecoderFrameWorkTileResidualScratch
	req, err := av1.BindDecoderFrameWorkBatchResidualRequestScratch(scratchSize, av1.DecoderFrameWorkBatchResidualScratch{
		Int32Scratch:     make([]int32, scratchSize.Int32Scratch),
		ResidualScratch:  make([]int16, scratchSize.ResidualScratch),
		LoopContextAbove: make([]av1.TileBlockLoopRootAboveContext, scratchSize.LoopContextAbove),
	})
	if err != nil {
		b.Fatal(err)
	}
	req.Tile.TransformMode = batch.TransformRef.TransformMode
	req.Tile.UseDefaultTransforms = true

	b.SetBytes(int64(len(batch.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		retainedValid = false
		if err := av1.InitDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
			b.Fatal(err)
		}
		stats, err := av1.DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &storage, &scratch, req)
		if err != nil {
			b.Fatal(err)
		}
		sum += int(stats.Residuals + stats.TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualBatchRunner(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, av1.FrameFormat{Width: 128, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	publicFillDecoderPostFilterPlane(output.Y)
	var retained av1.DecoderFrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := av1.DecoderFrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 512),
		FrameWorkFrameContext: av1.DecoderFrameWorkFrameContext{
			Sequence: av1.DecoderFrameWorkSequenceContextFromHeader(av1.SequenceHeader{
				ColorConfig: av1.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    av1.FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8},
			Quantization: av1.QuantizationParams{BaseQIdx: 64},
			TransformRef: av1.TransformReferenceParams{TransformMode: av1.TransformModeLargest},
		},
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Batch:                         av1.TileBatch{Worker: 0},
		Jobs: []av1.TileJob{
			{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 256},
			{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 256, Size: 256, UpdatesFrameContext: true},
		},
	}
	size, err := av1.DecoderFrameWorkBatchResidualRunnerScratchLen(batch, 1)
	if err != nil {
		b.Fatal(err)
	}
	runner, err := av1.BindDecoderFrameWorkBatchResidualRunner(size, av1.DecoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]av1.TileDecodeState, size.Workers),
		Storages:                make([]av1.DecoderFrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]av1.DecoderFrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]av1.DecoderFrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]av1.DecoderFrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]av1.DecoderFrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]av1.DecoderFrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            make([]int32, size.Int32Scratch),
		ResidualScratch:         make([]int16, size.ResidualScratch),
		LoopContextAboveScratch: make([]av1.TileBlockLoopRootAboveContext, size.LoopContextAbove),
	})
	if err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(len(batch.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		retainedValid = false
		if err := runner.Run(batch); err != nil {
			b.Fatal(err)
		}
		sum += int(runner.Stats[0].Residuals + runner.Stats[0].TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualEventRunner(b *testing.B) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	sequence := av1.SequenceHeader{
		EnableCDEF: true,
		ColorConfig: av1.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	event.CDEF = av1.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [av1.MaxCDEFStrengths]uint8{63},
	}
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	eventScratch, err := av1.DecoderFrameWorkResidualEventScratchLen(sequence, event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(b, av1.FrameFormat{
		Width:      int(event.FrameSize.CodedWidth),
		Height:     int(event.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	eventRunner, side, err := av1.BindDecoderFrameWorkResidualEventRunner(eventScratch, sequence, event, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		Stats:             &stats,
	}, publicDecoderResidualEventScratch(eventScratch), &batchRunner)
	if err != nil {
		b.Fatal(err)
	}
	var probeOutput av1.Frame
	probeCtx, err := av1.DecoderFrameWorkPostFilterScratchContext(sequence, event, 64, &side, &probeOutput)
	if err != nil {
		b.Fatal(err)
	}
	var probe av1.DecoderFrameWorkSupportedPostFilterScratchRunner
	postScratchSize, err := probe.ScratchLen(probeCtx)
	if err != nil {
		b.Fatal(err)
	}
	postRunner := av1.DecoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(av1.DecoderFrameWorkPostFilterRequestScratchLen(postScratchSize)),
	}

	b.SetBytes(int64(len(event.Unit.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := eventRunner.RunWithPostFilterRunner(sequence, event, &side, &postRunner)
		if err != nil {
			b.Fatal(err)
		}
		if result.Output == nil || result.Run != (av1.DecoderFrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			b.Fatalf("result=%+v", result)
		}
		if !postRunner.Result.Completed.Has(av1.DecoderFrameWorkPostFilterCDEF) || postRunner.Context.RemainingPostFilters() != 0 {
			b.Fatalf("postfilter result=%+v remaining=%b", postRunner.Result, postRunner.Context.RemainingPostFilters())
		}
		sum += int(stats.Residuals + stats.TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualEventListRunner(b *testing.B) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	sequence := av1.SequenceHeader{
		EnableCDEF: true,
		ColorConfig: av1.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	frame := publicDecoderResidualRunnerFrameEvent()
	events := [...]av1.DecoderEvent{
		{
			Kind:                  av1.DecoderEventSequenceHeader,
			SequenceHeader:        sequence,
			NewCodedVideoSequence: true,
		},
		frame,
	}
	var scratchSpans [2]av1.TileSpan
	var scratchJobs [2]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	eventScratch, err := av1.DecoderFrameWorkResidualEventsScratchLen(sequence, events[:], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(b, av1.FrameFormat{
		Width:      int(frame.FrameSize.CodedWidth),
		Height:     int(frame.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var eventSideData av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualEventScratch(eventScratch)
	eventRunner, _, err := av1.BindDecoderFrameWorkResidualEventRunner(eventScratch, sequence, frame, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &eventSideData,
		Stats:             &stats,
	}, scratch, &batchRunner)
	if err != nil {
		b.Fatal(err)
	}

	b.SetBytes(int64(len(frame.Unit.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := eventRunner.RunEvents(sequence, events[:], scratch.SideData, nil)
		if err != nil {
			b.Fatal(err)
		}
		if result.Count != len(events) || result.ExecutedTileWork != 1 || result.CompletedFrames != 1 {
			b.Fatalf("result=%+v", result)
		}
		sum += int(stats.Residuals + stats.TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualStreamRunner(b *testing.B) {
	workerPool, err := av1.NewTileWorkerPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	lowOverhead := publicDecoderResidualLowOverheadStream()
	lowOverheads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	lowOverheadsBytes := 0
	for i := range lowOverheads {
		lowOverheadsBytes += len(lowOverheads[i])
	}
	rtpPayload := publicDecoderResidualRTPPayload()
	rtpPayloads := publicDecoderResidualFragmentedRTPPayloads()
	rtpPayloadsBytes := 0
	for i := range rtpPayloads {
		rtpPayloadsBytes += len(rtpPayloads[i])
	}
	var probeStream av1.DecoderStream
	var probeEvents [4]av1.DecoderEvent
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch
	lowPlan, err := av1.DecoderFrameWorkResidualLowOverheadStreamPlan(probeStream, lowOverhead, 1, probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	lowOverheadsPlan, err := av1.DecoderFrameWorkResidualLowOverheadStreamsPlan(probeStream, lowOverheads[:], 1, probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	rtpPlan, err := av1.DecoderFrameWorkResidualRTPPayloadStreamPlan(probeStream, 0, rtpPayload, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	rtpPayloadsPlan, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, rtpPayloads, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	streamPlan := lowPlan.Max(lowOverheadsPlan).Max(rtpPlan).Max(rtpPayloadsPlan)
	if !streamPlan.HasEvent() {
		b.Fatal("stream plan missing bind event")
	}
	pool := publicDecoderPostFilterFramePool(b, av1.FrameFormat{
		Width:        int(streamPlan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(streamPlan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream av1.DecoderStream
	var refs av1.DecoderSurfaceReferences
	var state av1.DecoderFrameWorkState
	var referenceSurfaces [av1.InterRefsPerFrame]int
	var referenceFrames [av1.InterRefsPerFrame]*av1.Frame
	var releases [av1.RefFrames]int
	var stats av1.DecoderFrameWorkTileResidualStats
	var side av1.DecoderFrameWorkSideData
	var batchRunner av1.DecoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(streamPlan.Size)
	runner, _, err := av1.BindDecoderFrameWorkResidualStreamPlanRunner(streamPlan, &stream, av1.DecoderFrameWorkResidualEventRuntime{
		State:             &state,
		Refs:              &refs,
		FramePool:         &pool,
		Align:             64,
		ReferenceSurfaces: referenceSurfaces[:],
		ReferenceFrames:   referenceFrames[:],
		Releases:          releases[:],
		WorkerPool:        workerPool,
		SideData:          &side,
		Stats:             &stats,
	}, scratch, &batchRunner)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("low-overhead", func(b *testing.B) {
		b.SetBytes(int64(len(lowOverhead)))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			pool.Reset()
			refs.Reset()
			state.Reset()
			if err := runner.Reset(); err != nil {
				b.Fatal(err)
			}
			result, err := runner.RunLowOverhead(lowOverhead, nil)
			if err != nil {
				b.Fatal(err)
			}
			if result.Run.CompletedFrames != 1 {
				b.Fatalf("result=%+v", result)
			}
			sum += int(stats.Residuals + stats.TXBs)
		}
		publicBenchmarkSink = sum
	})
	b.Run("low-overheads", func(b *testing.B) {
		b.SetBytes(int64(lowOverheadsBytes))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			pool.Reset()
			refs.Reset()
			state.Reset()
			if err := runner.Reset(); err != nil {
				b.Fatal(err)
			}
			result, err := runner.RunLowOverheads(lowOverheads[:], nil)
			if err != nil {
				b.Fatal(err)
			}
			if result.Run.CompletedFrames != 2 {
				b.Fatalf("result=%+v", result)
			}
			sum += int(stats.Residuals + stats.TXBs)
		}
		publicBenchmarkSink = sum
	})
	b.Run("low-overhead-into", func(b *testing.B) {
		b.SetBytes(int64(lowOverheadsBytes))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			pool.Reset()
			refs.Reset()
			state.Reset()
			if err := runner.Reset(); err != nil {
				b.Fatal(err)
			}
			var result av1.DecoderFrameWorkResidualStreamResult
			if err := runner.RunLowOverheadsInto(&result, lowOverheads[:], nil); err != nil {
				b.Fatal(err)
			}
			if result.Run.CompletedFrames != 2 {
				b.Fatalf("result=%+v", result)
			}
			sum += int(stats.Residuals + stats.TXBs)
		}
		publicBenchmarkSink = sum
	})
	b.Run("rtp", func(b *testing.B) {
		b.SetBytes(int64(len(rtpPayload)))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			pool.Reset()
			refs.Reset()
			state.Reset()
			if err := runner.Reset(); err != nil {
				b.Fatal(err)
			}
			result, err := runner.RunRTPPayload(rtpPayload, nil)
			if err != nil {
				b.Fatal(err)
			}
			if result.Run.CompletedFrames != 1 || runner.RTPUsed != 0 {
				b.Fatalf("result=%+v retained=%d", result, runner.RTPUsed)
			}
			sum += int(stats.Residuals + stats.TXBs)
		}
		publicBenchmarkSink = sum
	})
	b.Run("rtp-payloads", func(b *testing.B) {
		b.SetBytes(int64(rtpPayloadsBytes))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			pool.Reset()
			refs.Reset()
			state.Reset()
			if err := runner.Reset(); err != nil {
				b.Fatal(err)
			}
			result, err := runner.RunRTPPayloads(rtpPayloads, nil)
			if err != nil {
				b.Fatal(err)
			}
			if result.Run.CompletedFrames != 1 || runner.RTPUsed != 0 {
				b.Fatalf("result=%+v retained=%d", result, runner.RTPUsed)
			}
			sum += int(stats.Residuals + stats.TXBs)
		}
		publicBenchmarkSink = sum
	})
	b.Run("rtp-payload-into", func(b *testing.B) {
		b.SetBytes(int64(rtpPayloadsBytes))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			pool.Reset()
			refs.Reset()
			state.Reset()
			if err := runner.Reset(); err != nil {
				b.Fatal(err)
			}
			var result av1.DecoderFrameWorkResidualStreamResult
			if err := runner.RunRTPPayloadsInto(&result, rtpPayloads, nil); err != nil {
				b.Fatal(err)
			}
			if result.Run.CompletedFrames != 1 || runner.RTPUsed != 0 {
				b.Fatalf("result=%+v retained=%d", result, runner.RTPUsed)
			}
			sum += int(stats.Residuals + stats.TXBs)
		}
		publicBenchmarkSink = sum
	})
	b.Run("rtp-after-loss-into", func(b *testing.B) {
		b.SetBytes(int64(rtpPayloadsBytes + len(rtpPayload)))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			pool.Reset()
			refs.Reset()
			state.Reset()
			if err := runner.Reset(); err != nil {
				b.Fatal(err)
			}
			var result av1.DecoderFrameWorkResidualStreamResult
			if err := runner.RunRTPPayloadsInto(&result, rtpPayloads[:2], nil); err != nil {
				b.Fatal(err)
			}
			if runner.RTPUsed == 0 {
				b.Fatalf("missing retained fragment result=%+v", result)
			}
			if err := runner.RunRTPPayloadAfterLossInto(&result, rtpPayload, nil); err != nil {
				b.Fatal(err)
			}
			if result.Run.CompletedFrames != 1 || runner.RTPUsed != 0 {
				b.Fatalf("result=%+v retained=%d", result, runner.RTPUsed)
			}
			sum += int(stats.Residuals + stats.TXBs)
		}
		publicBenchmarkSink = sum
	})
}

func BenchmarkPublicDecoderResidualStreamScratchLen(b *testing.B) {
	lowOverhead := publicDecoderResidualLowOverheadStream()
	lowOverheads := [...][]byte{
		publicDecoderResidualLowOverheadStream(),
		publicDecoderResidualLowOverheadFrameStream(),
	}
	lowOverheadsBytes := 0
	for i := range lowOverheads {
		lowOverheadsBytes += len(lowOverheads[i])
	}
	rtpPayload := publicDecoderResidualRTPPayload()
	rtpPayloads := publicDecoderResidualFragmentedRTPPayloads()
	rtpPayloadsBytes := 0
	for i := range rtpPayloads {
		rtpPayloadsBytes += len(rtpPayloads[i])
	}
	var events [4]av1.DecoderEvent
	var rtpBuffer [128]byte
	var rtpSpans [4]av1.RTPObuSpan
	var scratchSpans [1]av1.TileSpan
	var scratchJobs [1]av1.TileJob
	var scratchBatches [1]av1.TileBatch

	b.Run("low-overhead", func(b *testing.B) {
		b.SetBytes(int64(len(lowOverhead)))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			var stream av1.DecoderStream
			size, err := av1.DecoderFrameWorkResidualLowOverheadStreamScratchLen(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
			if err != nil {
				b.Fatal(err)
			}
			sum += size.Events + size.Event.Plan.JobCount
		}
		publicBenchmarkSink = sum
	})
	b.Run("low-overheads", func(b *testing.B) {
		b.SetBytes(int64(lowOverheadsBytes))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			var stream av1.DecoderStream
			size, err := av1.DecoderFrameWorkResidualLowOverheadStreamsScratchLen(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
			if err != nil {
				b.Fatal(err)
			}
			sum += size.Events + size.Event.Plan.JobCount
		}
		publicBenchmarkSink = sum
	})
	b.Run("rtp", func(b *testing.B) {
		b.SetBytes(int64(len(rtpPayload)))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			var stream av1.DecoderStream
			size, err := av1.DecoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, rtpPayload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
			if err != nil {
				b.Fatal(err)
			}
			sum += size.Events + size.RTPBuffer + size.RTPSpans + size.Event.Plan.JobCount
		}
		publicBenchmarkSink = sum
	})
	b.Run("rtp-payloads", func(b *testing.B) {
		b.SetBytes(int64(rtpPayloadsBytes))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			var stream av1.DecoderStream
			size, err := av1.DecoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, rtpPayloads, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
			if err != nil {
				b.Fatal(err)
			}
			sum += size.Events + size.RTPBuffer + size.RTPSpans + size.Event.Plan.JobCount
		}
		publicBenchmarkSink = sum
	})
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

func BenchmarkPublicDecoderBlockCoeffDecodeReconstruction(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, av1.FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderBlockCoeffSimpleBatch(output)
	req := publicDecoderBlockCoeffDecodeRequest(b, batch, false)
	payload := make([]byte, 32)
	job := av1.TileJob{Offset: 0, Size: uint32(len(payload))}
	var state av1.TileDecodeState
	var transformCDFs av1.TileTransformCDFs
	var coeffCDFs av1.TileCoeffCDFs
	cdfs := av1.TileBlockCoeffCDFs{Transform: &transformCDFs, Coeff: &coeffCDFs}
	var transformCtx av1.TileTransformContext
	var coeffCtx av1.TileCoeffEntropyContext
	var scratch av1.TileBlockCoeffScratch

	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := av1.InitTileTransformCDFsDefault(&transformCDFs); err != nil {
			b.Fatal(err)
		}
		if err := av1.InitTileCoeffCDFsDefault(&coeffCDFs, 0); err != nil {
			b.Fatal(err)
		}
		if err := av1.ResetTileDecodeState(&state, payload, job, av1.TileDecodeOptions{}); err != nil {
			b.Fatal(err)
		}
		transformCtx.Reset()
		coeffCtx.Reset()
		fillPublicReconstructPlane(output.Y, output.Layout.BytesPerSample, 128)
		result, err := av1.DecodeAndReconstructDecoderFrameWorkBlockCoefficients(batch, 0, &state, cdfs, &transformCtx, &coeffCtx, &scratch, req, func(block av1.TileBlockCoeffBlock) error {
			sum += int(block.Result.EOB) + int(block.Plane)
			return nil
		})
		if err != nil {
			b.Fatal(err)
		}
		sum += int(result.TotalStats().TXBs) + int(output.Y.Pix[i%len(output.Y.Pix)])
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
			ReferenceOrderHints: [av1.InterRefsPerFrame]uint8{1, 9, 10, 4, 5, 6, 7},
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
	for i := range 10 {
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
	for i := range 4 {
		stream = appendPublicLowOverheadOBU(stream, av1.OBUTemporalDelimiter, nil)
		stream = appendPublicLowOverheadOBU(stream, av1.OBUFrameHeader, []byte{byte(i)})
		for j := range 2 {
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
	for i := range 3 {
		payload := make([]byte, 96)
		for j := range payload {
			payload[j] = byte(i*29 + j)
		}
		frame = appendPublicLowOverheadOBU(frame, av1.OBUTileGroup, payload)
	}
	frame = appendPublicLowOverheadOBU(frame, av1.OBUFrame, []byte{0xdd, 0xee})
	return frame
}
