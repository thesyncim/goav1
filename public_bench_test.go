package goav1

import (
	"testing"

	internalcdef "github.com/thesyncim/goav1/internal/av1/cdef"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
	internalfilmgrain "github.com/thesyncim/goav1/internal/av1/filmgrain"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalsuperres "github.com/thesyncim/goav1/internal/av1/superres"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

var publicBenchmarkSink int

func BenchmarkPublicKeyframeEncoderEncode1080p(b *testing.B) {
	const width, height = 1920, 1080
	enc, err := NewKeyframeEncoder(width, height, 80)
	if err != nil {
		b.Fatalf("NewKeyframeEncoder: %v", err)
	}
	b.Cleanup(func() {
		b.StopTimer()
		_ = enc.Close()
	})
	frames := [...]I420Frame{
		publicRTCMatrixFrame(width, height, 0),
		publicRTCMatrixFrame(width, height, 1),
		publicRTCMatrixFrame(width, height, 2),
		publicRTCMatrixFrame(width, height, 3),
	}
	if _, err := enc.Encode(frames[0]); err != nil {
		b.Fatalf("warm Encode: %v", err)
	}

	b.SetBytes(int64(width * height * 3 / 2))
	b.ReportAllocs()
	b.ResetTimer()
	sum := 0
	for i := 0; i < b.N; i++ {
		out, err := enc.Encode(frames[i&3])
		if err != nil {
			b.Fatalf("Encode: %v", err)
		}
		sum += len(out.Data) + len(enc.Reconstruction().Y)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicRTCEncoderEncodePicture1080p(b *testing.B) {
	for _, tc := range []struct {
		name string
		mode EncoderScalabilityMode
	}{
		{name: "L1T3", mode: EncoderScalabilityModeL1T3},
		{name: "S3T3", mode: internalencoder.ScalabilityModeS3T3},
	} {
		b.Run(tc.name, func(b *testing.B) {
			const width, height = 1920, 1080
			cfg := publicRTCMatrixConfig(width, height, tc.mode)
			cfg.TargetBitrateKbps = 1800
			cfg.MaxBitrateKbps = 2400
			enc, err := NewRTCEncoderWithConfig(cfg)
			if err != nil {
				b.Fatalf("NewRTCEncoderWithConfig: %v", err)
			}
			b.Cleanup(func() {
				b.StopTimer()
				_ = enc.Close()
			})

			frames := [...]I420Frame{
				publicRTCMatrixFrame(width, height, 0),
				publicRTCMatrixFrame(width, height, 1),
				publicRTCMatrixFrame(width, height, 2),
				publicRTCMatrixFrame(width, height, 3),
			}
			warmPublicRTCEncoderEncodePictureHotPath(b, enc, frames[:])

			b.SetBytes(publicRTCBenchmarkPictureBytes(enc.Config()))
			b.ReportAllocs()
			b.ResetTimer()

			sum := 0
			for i := 0; i < b.N; i++ {
				picture, err := enc.EncodePicture(frames[i&3], false)
				if err != nil {
					b.Fatalf("EncodePicture: %v", err)
				}
				for j := 0; j < picture.FrameNum; j++ {
					sum += len(picture.Frames[j].Data) + len(picture.Frames[j].DependencyDescriptor)
				}
			}
			publicBenchmarkSink = sum
		})
	}
}

func publicRTCBenchmarkPictureBytes(config EncoderConfig) int64 {
	var bytes int64
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		resolution := config.SpatialLayers[i].Resolution
		bytes += int64(resolution.Width) * int64(resolution.Height) * 3 / 2
	}
	return bytes
}

func BenchmarkPublicLowOverheadOBUIterator(b *testing.B) {
	stream := publicBenchmarkLowOverheadStream()
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		it := internalobu.NewLowOverheadIterator(stream)
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
		it := internalobu.NewTemporalUnitIterator(stream)
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
		it := internalobu.NewAnnexBIterator(stream)
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
	limits := RTPPayloadSizeLimits{MaxPayloadLen: 120}
	var packetizerOBUs [16]RTPPacketizerOBU
	var packets [32]RTPPacketPlan
	var work [32]RTPPacketPlan
	var packetBytes [32][160]byte
	var payloads [32][]byte
	var assembled [1024]byte
	var frameOBUs [16]RTPFrameOBU

	size, err := RTPPacketizerScratchLen(frame, limits, packetizerOBUs[:])
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
		firstPass, err := RTPPacketizerScratchLen(frame, limits, nil)
		if err != nil {
			b.Fatal(err)
		}
		if firstPass.OBUs != size.OBUs {
			b.Fatalf("first pass OBUs=%d want %d", firstPass.OBUs, size.OBUs)
		}
		size, err := RTPPacketizerScratchLen(frame, limits, packetizerOBUs[:])
		if err != nil {
			b.Fatal(err)
		}
		packetizer, err := NewRTPPacketizer(frame, limits, true, true, packetizerOBUs[:size.OBUs], packets[:size.Packets], work[:size.Work])
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
		assembledLen, obuCount, err := AssembleRTPFrameSize(payloads[:count])
		if err != nil {
			b.Fatal(err)
		}
		wrote, assembledCount, err := AssembleRTPFrame(assembled[:assembledLen], payloads[:count], frameOBUs[:obuCount])
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
	dec, err := NewDecoderFromRTPPayloads(rtpPayloads)
	if err != nil {
		b.Fatalf("NewDecoderFromRTPPayloads: %v", err)
	}
	b.Cleanup(func() {
		b.StopTimer()
		dec.Close()
	})

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
	for b.Loop() {
		sum += decodeAll()
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecodeRTPPayload1080p(b *testing.B) {
	rtpPayloads, wantVisible := publicDecoder1080pRTPPayloads(b, EncoderScalabilityModeL1T3)
	dec, err := NewDecoderFromRTPPayloads(rtpPayloads)
	if err != nil {
		b.Fatalf("NewDecoderFromRTPPayloads: %v", err)
	}
	b.Cleanup(func() {
		b.StopTimer()
		dec.Close()
	})

	bytesPerRun := publicRTPPayloadBytes(rtpPayloads)
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
			b.Fatalf("decoded %d visible 1080p RTP frames, want %d", visible, wantVisible)
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

func BenchmarkPublicLayeredDecodeRTPPayload1080p(b *testing.B) {
	rtpPayloads, _ := publicDecoder1080pRTPPayloads(b, internalencoder.ScalabilityModeL3T3)
	wantVisible := publicLayeredDecoderVisibleRTPPayloadFrames(b, rtpPayloads)
	dec, err := NewLayeredDecoderFromRTPPayloads(rtpPayloads)
	if err != nil {
		b.Fatalf("NewLayeredDecoderFromRTPPayloads: %v", err)
	}
	b.Cleanup(func() {
		b.StopTimer()
		dec.Close()
	})

	bytesPerRun := publicRTPPayloadBytes(rtpPayloads)
	decodeAll := func() int {
		if err := dec.Reset(); err != nil {
			b.Fatalf("Reset: %v", err)
		}
		visible := 0
		for i, payload := range rtpPayloads {
			frames, err := dec.DecodeRTPPayload(payload)
			if err != nil {
				b.Fatalf("LayeredDecoder DecodeRTPPayload packet %d: %v", i, err)
			}
			visible += len(frames)
		}
		if visible != wantVisible {
			b.Fatalf("decoded %d visible 1080p layered RTP frames, want %d", visible, wantVisible)
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

func publicRTPPayloadBytes(payloads [][]byte) int {
	total := 0
	for _, payload := range payloads {
		total += len(payload)
	}
	return total
}

func BenchmarkPublicFrameBindAndSampleRoundTrip(b *testing.B) {
	format := FrameFormat{
		Width:        128,
		Height:       72,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        32,
	}
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	frame, err := internalframe.Bind(backing, format)
	if err != nil {
		b.Fatal(err)
	}
	sampleLen, err := internalframe.SamplePlaneLen(frame.Y, 1)
	if err != nil {
		b.Fatal(err)
	}
	samples := make([]uint16, sampleLen)

	b.SetBytes(int64(len(backing)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		frame, err := internalframe.Bind(backing, format)
		if err != nil {
			b.Fatal(err)
		}
		plane, err := internalframe.LoadSamplePlane(samples, frame.Y, 1)
		if err != nil {
			b.Fatal(err)
		}
		if err := internalframe.StoreSamplePlane(frame.Y, 1, plane); err != nil {
			b.Fatal(err)
		}
		sum += plane.Stride + frame.Layout.Size
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderPredictionBridge(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := publicDecoderPredictionBatch(output)
	visit := publicDecoderPredictionIntraVisit(internaltile.IntraModeDC)
	var scratch internalthreading.FrameWorkIntraPredictionScratch
	publicSeedDecoderPredictionIntraEdges(output, 10, 50)

	b.SetBytes(16 * 16)
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := predictDecoderFrameWorkBlockLumaIntra(batch, 0, visit, &scratch); err != nil {
			b.Fatal(err)
		}
		sum += int(output.Y.Pix[16*output.Y.Stride+16])
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicOutputFilters(b *testing.B) {
	superSrc := internalframe.SamplePlane{Pix: make([]uint16, 128*32), Stride: 128, Width: 128, Height: 32}
	superDst := internalframe.SamplePlane{Pix: make([]uint16, 192*32), Stride: 192, Width: 192, Height: 32}
	for i := range superSrc.Pix {
		superSrc.Pix[i] = uint16(i & 0xff)
	}
	lut := publicFilmGrainScaling(64)
	rowDst, rowSrc := publicFilmGrainRowBuffers(128, 4, 128, 100)
	lumaGrain := make([]int16, internalfilmgrain.LumaGrainSamples)
	publicSetFilmGrainLuma(lumaGrain, 0xd9, 0, 0, 0, 0, 64)
	lumaParams := internalfilmgrain.LumaRowParams{
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
		if err := internalsuperres.UpscalePlane(superSrc, superDst, 8); err != nil {
			b.Fatal(err)
		}
		if err := internalfilmgrain.ApplyLumaRow(rowDst, rowSrc, lumaGrain, lut[:], lumaParams); err != nil {
			b.Fatal(err)
		}
		sum += int(superDst.Pix[i%len(superDst.Pix)]) + int(rowDst[i%len(rowDst)])
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderPostFilterBinding(b *testing.B) {
	sequence := publicDecoderPostFilterSequence()
	size := FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8}
	_, _, loopFilterLength, err := decoderFrameWorkLoopFilterMapShape(sequence, size)
	if err != nil {
		b.Fatal(err)
	}
	loopFilterRecords := make([]internalthreading.FrameWorkLoopFilterBlockRecord, loopFilterLength)
	loopFilterEdges := make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, 4)
	loopFilterSchedule := make([]uint32, len(loopFilterEdges))
	loopFilterSize := internaldecoder.FrameWorkLoopFilterPostFilterScratchSize{Edges: len(loopFilterEdges), Schedule: len(loopFilterSchedule)}
	index := make([]uint8, 1)
	read := make([]bool, 1)
	cdefSize := internaldecoder.FrameWorkCDEFPostFilterScratchSize{
		Samples:       [3]int{64, 16, 16},
		Dst:           [3]int{64, 16, 16},
		DirectionGrid: 1,
		VarianceGrid:  1,
		Input:         internalcdef.InputBufferSize,
		UnitDst:       internalcdef.InputBufferSize,
	}
	var sampleScratch [3][]uint16
	var dstScratch [3][]uint16
	for plane := range 3 {
		sampleScratch[plane] = make([]uint16, cdefSize.Samples[plane])
		dstScratch[plane] = make([]uint16, cdefSize.Dst[plane])
	}
	directionGrid := make([]internalcdef.DirectionGrid, cdefSize.DirectionGrid)
	varianceGrid := make([]internalcdef.VarianceGrid, cdefSize.VarianceGrid)
	inputScratch := make([]uint16, cdefSize.Input)
	unitDstScratch := make([]uint16, cdefSize.UnitDst)
	postFilterSize := internaldecoder.FrameWorkPostFilterScratchSize{
		LoopFilter: loopFilterSize,
		CDEF:       cdefSize,
	}
	postFilterBuffers := decoderFrameWorkPostFilterRequestBuffers{
		LoopFilterEdges:    loopFilterEdges,
		LoopFilterSchedule: loopFilterSchedule,

		CDEFSampleScratch:  sampleScratch,
		CDEFDstScratch:     dstScratch,
		CDEFDirectionGrid:  directionGrid,
		CDEFVarianceGrid:   varianceGrid,
		CDEFInputScratch:   inputScratch,
		CDEFUnitDstScratch: unitDstScratch,
	}
	postFilterArenaSize := decoderFrameWorkPostFilterRequestScratchLen(postFilterSize)
	postFilterSide := decoderFrameWorkPostFilterRequestSideData{}
	postFilterArena := decoderFrameWorkPostFilterRequestScratch{
		LoopFilterEdges:    make([]internaldecoder.FrameWorkLoopFilterPostFilterEdge, postFilterArenaSize.LoopFilterEdges),
		LoopFilterSchedule: make([]uint32, postFilterArenaSize.LoopFilterSchedule),
		CDEFDirectionGrid:  make([]internalcdef.DirectionGrid, postFilterArenaSize.CDEFDirectionGrid),
		CDEFVarianceGrid:   make([]internalcdef.VarianceGrid, postFilterArenaSize.CDEFVarianceGrid),
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
		loopMap, err := bindDecoderFrameWorkLoopFilterMap(sequence, size, loopFilterRecords)
		if err != nil {
			b.Fatal(err)
		}
		postFilterBuffers.LoopFilterMap = loopMap
		postFilterSide.LoopFilterMap = loopMap
		loopReq, err := bindDecoderFrameWorkLoopFilterPostFilterRequest(loopFilterSize, loopMap, loopFilterEdges, loopFilterSchedule)
		if err != nil {
			b.Fatal(err)
		}
		cdefMap, err := bindDecoderFrameWorkCDEFIndexMap(sequence, size, internalparser.CDEFParams{Bits: 1, StrengthCount: 2}, index, read)
		if err != nil {
			b.Fatal(err)
		}
		postFilterBuffers.CDEFIndexMap = cdefMap
		postFilterSide.CDEFIndexMap = cdefMap
		cdefReq, err := bindDecoderFrameWorkCDEFPostFilterRequest(cdefSize, cdefMap, sampleScratch, dstScratch, directionGrid, varianceGrid, inputScratch, unitDstScratch)
		if err != nil {
			b.Fatal(err)
		}
		postReq, err := bindDecoderFrameWorkPostFilterRequest(postFilterSize, postFilterBuffers)
		if err != nil {
			b.Fatal(err)
		}
		arenaReq, err := bindDecoderFrameWorkPostFilterRequestFromScratch(postFilterSize, postFilterSide, postFilterArena)
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
	size := FrameSize{
		CodedWidth:          8,
		UpscaledWidth:       13,
		Height:              32,
		SuperResEnabled:     true,
		SuperResDenominator: 13,
	}
	event := internaldecoder.Event{
		Kind:           internaldecoder.EventTileGroup,
		SequenceHeader: sequence,
		FrameSize:      size,
		TileGroup:      internalparser.TileGroup{Final: true},
		FilmGrain: internalparser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [internalparser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	var output Frame

	b.SetBytes(int64(size.CodedWidth * size.Height))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		ctx, err := decoderFrameWorkPostFilterScratchContext(sequence, &event, 32, nil, &output)
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
	size := FrameSize{
		CodedWidth:          8,
		UpscaledWidth:       13,
		Height:              32,
		SuperResEnabled:     true,
		SuperResDenominator: 13,
	}
	event := internaldecoder.Event{
		Kind:           internaldecoder.EventTileGroup,
		SequenceHeader: sequence,
		FrameSize:      size,
		TileGroup:      internalparser.TileGroup{Final: true},
		FilmGrain: internalparser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [internalparser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	var scratchOutput Frame
	scratchCtx, err := decoderFrameWorkPostFilterScratchContext(sequence, &event, 32, nil, &scratchOutput)
	if err != nil {
		b.Fatal(err)
	}
	var runner decoderFrameWorkCallerPostFilterScratchRunner
	first, err := runner.ScratchLen(scratchCtx)
	if err != nil {
		b.Fatal(err)
	}
	runner.Scratch = publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(first))
	full, err := runner.ScratchLen(scratchCtx)
	if err != nil {
		b.Fatal(err)
	}
	runner.Scratch = publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(full))

	output := publicBenchmarkDecoderFrame(b, scratchOutput.Format)
	for i := range output.Y.Pix {
		output.Y.Pix[i] = 100
	}
	ctx := internaldecoder.FrameWorkPostFilterContext{Event: event, Output: output}

	b.SetBytes(int64(size.CodedWidth*size.Height + size.UpscaledWidth*size.Height))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := runner.Apply(ctx); err != nil {
			b.Fatal(err)
		}
		if runner.Result.Completed != internaldecoder.FrameWorkPostFilterSuperRes|internaldecoder.FrameWorkPostFilterFilmGrain ||
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
	size := FrameSize{
		CodedWidth:          8,
		UpscaledWidth:       13,
		Height:              32,
		SuperResEnabled:     true,
		SuperResDenominator: 13,
	}
	event := internaldecoder.Event{
		Kind:           internaldecoder.EventTileGroup,
		SequenceHeader: sequence,
		FrameSize:      size,
		TileGroup:      internalparser.TileGroup{Final: true},
		FilmGrain: internalparser.FilmGrainParams{
			ParamsPresent: true,
			Apply:         true,
			Seed:          0x1234,
			BitDepth:      8,
			NumYPoints:    1,
			YPoints:       [internalparser.MaxFilmGrainYPoints][2]uint8{{0, 64}},
			ScalingShift:  8,
			Overlap:       true,
		},
	}
	var scratchOutput Frame
	_, err := decoderFrameWorkPostFilterScratchContext(sequence, &event, 32, nil, &scratchOutput)
	if err != nil {
		b.Fatal(err)
	}
	output := publicBenchmarkDecoderFrame(b, scratchOutput.Format)
	for i := range output.Y.Pix {
		output.Y.Pix[i] = 100
	}
	ctx := internaldecoder.FrameWorkPostFilterContext{Event: event, Output: output}
	var runner decoderFrameWorkReusableCallerPostFilterRunner
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
	var initial internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&initial, 64); err != nil {
		b.Fatal(err)
	}
	var retained internalthreading.FrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := internaldecoder.FrameWorkBatch{
		Payload: []byte{0x00, 0xff},
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Quantization: internalparser.QuantizationParams{BaseQIdx: 73},
		},
		InitialTileResidualCDFs:       &initial,
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []internaltile.Job{
			{Tile: 0, Offset: 0, Size: 1},
			{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true},
		},
	}
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	var state internaltile.DecodeState
	blockLoopBatch := publicBenchmarkDecoderBlockLoopBatch()
	rootColumns, err := decoderFrameWorkJobBlockLoopContextRootColumns(blockLoopBatch, 0)
	if err != nil {
		b.Fatal(err)
	}
	above := make([]internaltile.BlockLoopRootAboveContext, rootColumns)
	segMap := make([]uint8, 96*96)

	b.SetBytes(int64(len(batch.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := initDecoderFrameWorkTileResidualCDFStorage(batch, &storage); err != nil {
			b.Fatal(err)
		}
		cdfs := decoderFrameWorkTileResidualCDFsFromStorage(&storage)
		if err := initDecoderFrameWorkJobDecodeState(batch, 1, &state); err != nil {
			b.Fatal(err)
		}
		if err := retainDecoderFrameWorkTileResidualCDFStorage(batch, 1, &state, &storage); err != nil {
			b.Fatal(err)
		}
		carrier, err := bindTileBlockLoopContextCarrier(rootColumns, above)
		if err != nil {
			b.Fatal(err)
		}
		req, err := decoderFrameWorkJobBlockLoopRequest(blockLoopBatch, 0, segMap, nil, 96, &carrier)
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
	output := publicBenchmarkDecoderFrame(b, FrameFormat{Width: 64, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	batch := internaldecoder.FrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 256),
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
				ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    FrameSize{CodedWidth: 64, UpscaledWidth: 64, Height: 64, SuperResDenominator: 8},
			Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
			TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
		},
		Jobs: []internaltile.Job{{SBCols: 1, SBRows: 1, Offset: 0, Size: 256}},
	}
	var state internaltile.DecodeState
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		b.Fatal(err)
	}
	cdfs := decoderFrameWorkTileResidualCDFsFromStorage(&storage)
	loopReq, err := decoderFrameWorkJobBlockLoopRequest(batch, 0, nil, nil, 0, nil)
	if err != nil {
		b.Fatal(err)
	}
	int32Len, int16Len, err := decoderFrameWorkResidualMaxScratchLen(batch, batch.Quantization.BaseQIdx, 0, internaldecoder.FrameWorkPlaneY)
	if err != nil {
		b.Fatal(err)
	}
	var scratch internalthreading.FrameWorkTileResidualScratch
	req := internalthreading.FrameWorkTileResidualRequest{
		Loop:            loopReq,
		TransformMode:   batch.TransformRef.TransformMode,
		Int32Scratch:    make([]int32, int32Len),
		ResidualScratch: make([]int16, int16Len),
	}

	b.SetBytes(int64(len(batch.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		if err := initDecoderFrameWorkJobDecodeState(batch, 0, &state); err != nil {
			b.Fatal(err)
		}
		stats, err := decodeAndReconstructDecoderFrameWorkJobResiduals(batch, 0, &state, cdfs, &scratch, req)
		if err != nil {
			b.Fatal(err)
		}
		sum += int(stats.Residuals + stats.TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualBatchDecode(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, FrameFormat{Width: 128, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	publicFillDecoderPostFilterPlane(output.Y)
	var retained internalthreading.FrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := internaldecoder.FrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 512),
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
				ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8},
			Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
			TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
		},
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Jobs: []internaltile.Job{
			{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 256},
			{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 256, Size: 256, UpdatesFrameContext: true},
		},
	}
	var storage internalthreading.FrameWorkTileResidualCDFStorage
	if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
		b.Fatal(err)
	}
	scratchSize, err := decoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		b.Fatal(err)
	}
	state := &internaltile.DecodeState{}
	var scratch internalthreading.FrameWorkTileResidualScratch
	req, err := bindDecoderFrameWorkBatchResidualRequestScratch(scratchSize, decoderFrameWorkBatchResidualScratch{
		Int32Scratch:     make([]int32, scratchSize.Int32Scratch),
		ResidualScratch:  make([]int16, scratchSize.ResidualScratch),
		LoopContextAbove: make([]internaltile.BlockLoopRootAboveContext, scratchSize.LoopContextAbove),
	})
	if err != nil {
		b.Fatal(err)
	}
	req.Tile.TransformMode = batch.TransformRef.TransformMode

	b.SetBytes(int64(len(batch.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		retainedValid = false
		if err := initDecoderFrameWorkTileResidualCDFStorageDefault(&storage, batch.Quantization.BaseQIdx); err != nil {
			b.Fatal(err)
		}
		stats, err := decodeAndRetainDecoderFrameWorkBatchResiduals(batch, state, &storage, &scratch, req)
		if err != nil {
			b.Fatal(err)
		}
		sum += int(stats.Residuals + stats.TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualBatchRunner(b *testing.B) {
	output := publicBenchmarkDecoderFrame(b, FrameFormat{Width: 128, Height: 64, BitDepth: 8, MonoChrome: true, Align: 64})
	publicFillDecoderPostFilterPlane(output.Y)
	var retained internalthreading.FrameWorkTileResidualCDFStorage
	retainedValid := false
	batch := internaldecoder.FrameWorkBatch{
		Output:  output,
		Payload: make([]byte, 512),
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
				ColorConfig: internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameSize:    FrameSize{CodedWidth: 128, UpscaledWidth: 128, Height: 64, SuperResDenominator: 8},
			Quantization: internalparser.QuantizationParams{BaseQIdx: 64},
			TransformRef: internalparser.TransformReferenceParams{TransformMode: internalparser.TransformModeLargest},
		},
		RetainedTileResidualCDFs:      &retained,
		RetainedTileResidualCDFsValid: &retainedValid,
		Batch:                         internalthreading.Batch{Worker: 0},
		Jobs: []internaltile.Job{
			{SBX: 0, SBY: 0, SBCols: 1, SBRows: 1, Offset: 0, Size: 256},
			{SBX: 1, SBY: 0, SBCols: 1, SBRows: 1, Offset: 256, Size: 256, UpdatesFrameContext: true},
		},
	}
	size, err := decoderFrameWorkBatchResidualRunnerScratchLen(batch, 1)
	if err != nil {
		b.Fatal(err)
	}
	runner, err := bindDecoderFrameWorkBatchResidualRunner(size, decoderFrameWorkBatchResidualRunnerScratch{
		States:                  make([]internaltile.DecodeState, size.Workers),
		Storages:                make([]internalthreading.FrameWorkTileResidualCDFStorage, size.Workers),
		TileScratch:             make([]internalthreading.FrameWorkTileResidualScratch, size.Workers),
		RestorationRequests:     make([]internalthreading.FrameWorkTileRestorationRequest, size.RestorationRequests),
		PredictionScratch:       make([]internalthreading.FrameWorkPredictionScratch, size.Workers),
		InterPredictionScratch:  make([]internalthreading.FrameWorkInterPredictionScratch, size.Workers),
		Stats:                   make([]internalthreading.FrameWorkTileResidualStats, size.Workers),
		Int32Scratch:            make([]int32, size.Int32Scratch),
		ResidualScratch:         make([]int16, size.ResidualScratch),
		LoopContextAboveScratch: make([]internaltile.BlockLoopRootAboveContext, size.LoopContextAbove),
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
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	event := publicDecoderResidualRunnerFrameEvent()
	event.CDEF = internalparser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [internalparser.MaxCDEFStrengths]uint8{63},
	}
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	eventScratch, err := decoderFrameWorkResidualEventScratchLen(sequence, &event, 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}

	pool := publicDecoderPostFilterFramePool(b, FrameFormat{
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
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var batchRunner decoderFrameWorkBatchResidualRunner
	eventRunner, side, err := bindDecoderFrameWorkResidualEventRunner(eventScratch, sequence, &event, decoderFrameWorkResidualEventRuntime{
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
	var probeOutput Frame
	probeCtx, err := decoderFrameWorkPostFilterScratchContext(sequence, &event, 64, &side, &probeOutput)
	if err != nil {
		b.Fatal(err)
	}
	var probe decoderFrameWorkSupportedPostFilterScratchRunner
	postScratchSize, err := probe.ScratchLen(probeCtx)
	if err != nil {
		b.Fatal(err)
	}
	postRunner := decoderFrameWorkSupportedPostFilterScratchRunner{
		Scratch: publicDecoderPostFilterRequestScratch(decoderFrameWorkPostFilterRequestScratchLen(postScratchSize)),
	}

	b.SetBytes(int64(len(event.Unit.Payload)))
	b.ReportAllocs()
	b.ResetTimer()

	sum := 0
	for i := 0; i < b.N; i++ {
		pool.Reset()
		refs.Reset()
		state.Reset()
		result, err := eventRunner.RunWithPostFilterRunner(sequence, &event, &side, &postRunner)
		if err != nil {
			b.Fatal(err)
		}
		if result.Output == nil || result.Run != (internaldecoder.FrameWorkStepResult{ExecutedTileWork: true, CompletedFrame: true}) {
			b.Fatalf("result=%+v", result)
		}
		if !postRunner.Result.Completed.Has(internaldecoder.FrameWorkPostFilterCDEF) || postRunner.Context.RemainingPostFilters() != 0 {
			b.Fatalf("postfilter result=%+v remaining=%b", postRunner.Result, postRunner.Context.RemainingPostFilters())
		}
		sum += int(stats.Residuals + stats.TXBs)
	}
	publicBenchmarkSink = sum
}

func BenchmarkPublicDecoderResidualEventListRunner(b *testing.B) {
	workerPool, err := internalthreading.NewPool(1)
	if err != nil {
		b.Fatal(err)
	}
	defer workerPool.Close()

	sequence := SequenceHeader{
		EnableCDEF: true,
		ColorConfig: internalparser.ColorConfig{
			BitDepth:   8,
			MonoChrome: true,
		},
	}
	frame := publicDecoderResidualRunnerFrameEvent()
	events := [...]internaldecoder.Event{
		{
			Kind:                  internaldecoder.EventSequenceHeader,
			SequenceHeader:        sequence,
			NewCodedVideoSequence: true,
		},
		frame,
	}
	var scratchSpans [2]internalparser.TileSpan
	var scratchJobs [2]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	eventScratch, err := decoderFrameWorkResidualEventsScratchLen(sequence, events[:], 1, scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	pool := publicDecoderPostFilterFramePool(b, FrameFormat{
		Width:      int(frame.FrameSize.CodedWidth),
		Height:     int(frame.FrameSize.Height),
		BitDepth:   8,
		MonoChrome: true,
		Align:      64,
	}, 1)
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var eventSideData decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualEventScratch(eventScratch)
	eventRunner, _, err := bindDecoderFrameWorkResidualEventRunner(eventScratch, sequence, &frame, decoderFrameWorkResidualEventRuntime{
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
	workerPool, err := internalthreading.NewPool(1)
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
	var probeStream internaldecoder.Stream
	var probeEvents [4]internaldecoder.Event
	var probeRTPBuffer [256]byte
	var probeRTPSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch
	lowPlan, err := decoderFrameWorkResidualLowOverheadStreamPlan(probeStream, lowOverhead, 1, probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	lowOverheadsPlan, err := decoderFrameWorkResidualLowOverheadStreamsPlan(probeStream, lowOverheads[:], 1, probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	rtpPlan, err := decoderFrameWorkResidualRTPPayloadStreamPlan(probeStream, 0, rtpPayload, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	rtpPayloadsPlan, err := decoderFrameWorkResidualRTPPayloadsStreamPlan(probeStream, 0, rtpPayloads, 1, probeRTPBuffer[:], probeRTPSpans[:], probeEvents[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
	if err != nil {
		b.Fatal(err)
	}
	streamPlan := lowPlan.Max(lowOverheadsPlan).Max(rtpPlan).Max(rtpPayloadsPlan)
	if !streamPlan.HasEvent() {
		b.Fatal("stream plan missing bind event")
	}
	pool := publicDecoderPostFilterFramePool(b, FrameFormat{
		Width:        int(streamPlan.Bind.Event.FrameSize.CodedWidth),
		Height:       int(streamPlan.Bind.Event.FrameSize.Height),
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        64,
	}, 2)
	var stream internaldecoder.Stream
	var refs internaldecoder.SurfaceReferences
	var state internaldecoder.FrameWorkState
	var referenceSurfaces [internalparser.InterRefsPerFrame]int
	var referenceFrames [internalparser.InterRefsPerFrame]*Frame
	var releases [internalparser.RefFrames]int
	var stats internalthreading.FrameWorkTileResidualStats
	var side decoderFrameWorkSideData
	var batchRunner decoderFrameWorkBatchResidualRunner
	scratch := publicDecoderResidualStreamScratch(streamPlan.Size)
	runner, _, err := bindDecoderFrameWorkResidualStreamPlanRunner(streamPlan, &stream, decoderFrameWorkResidualEventRuntime{
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
			var result decoderFrameWorkResidualStreamResult
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
			var result decoderFrameWorkResidualStreamResult
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
			var result decoderFrameWorkResidualStreamResult
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
	var events [4]internaldecoder.Event
	var rtpBuffer [128]byte
	var rtpSpans [4]RTPObuSpan
	var scratchSpans [1]internalparser.TileSpan
	var scratchJobs [1]internaltile.Job
	var scratchBatches [1]internalthreading.Batch

	b.Run("low-overhead", func(b *testing.B) {
		b.SetBytes(int64(len(lowOverhead)))
		b.ReportAllocs()
		sum := 0
		for i := 0; i < b.N; i++ {
			var stream internaldecoder.Stream
			size, err := decoderFrameWorkResidualLowOverheadStreamScratchLen(stream, lowOverhead, 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
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
			var stream internaldecoder.Stream
			size, err := decoderFrameWorkResidualLowOverheadStreamsScratchLen(stream, lowOverheads[:], 1, events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
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
			var stream internaldecoder.Stream
			size, err := decoderFrameWorkResidualRTPPayloadStreamScratchLen(stream, 0, rtpPayload, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
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
			var stream internaldecoder.Stream
			size, err := decoderFrameWorkResidualRTPPayloadsStreamScratchLen(stream, 0, rtpPayloads, 1, rtpBuffer[:], rtpSpans[:], events[:], scratchSpans[:], scratchJobs[:], scratchBatches[:])
			if err != nil {
				b.Fatal(err)
			}
			sum += size.Events + size.RTPBuffer + size.RTPSpans + size.Event.Plan.JobCount
		}
		publicBenchmarkSink = sum
	})
}

func publicBenchmarkDecoderFrame(b *testing.B, format FrameFormat) *Frame {
	b.Helper()
	layout, err := internalframe.RequiredSize(format)
	if err != nil {
		b.Fatal(err)
	}
	frame, err := internalframe.Bind(make([]byte, layout.Size), format)
	if err != nil {
		b.Fatal(err)
	}
	return &frame
}

func publicBenchmarkDecoderBlockLoopBatch() internaldecoder.FrameWorkBatch {
	return internaldecoder.FrameWorkBatch{
		FrameWorkFrameContext: internaldecoder.FrameWorkFrameContext{
			Sequence: internalthreading.FrameWorkSequenceContextFromHeader(SequenceHeader{
				Use128x128Superblock: true,
				EnableDualFilter:     true,
				EnableFilterIntra:    true,
				EnableOrderHint:      true,
				OrderHintBits:        5,
				ColorConfig:          internalparser.ColorConfig{BitDepth: 8, MonoChrome: true},
			}),
			FrameHeader:         internalparser.FrameHeaderPrefix{OrderHint: 9},
			FrameSize:           FrameSize{CodedWidth: 300, UpscaledWidth: 300, Height: 260, SuperResDenominator: 8},
			TileInfo:            internalparser.TileInfo{InterpolationFilter: internalparser.InterpolationSwitchable, UseRefFrameMVS: true},
			ReferenceOrderHints: [internalparser.InterRefsPerFrame]uint8{1, 9, 10, 4, 5, 6, 7},
			SkipMode:            internalparser.SkipModeParams{Allowed: true, Enabled: true},
			CDEF:                internalparser.CDEFParams{Bits: 2, StrengthCount: 4},
			Delta:               internalparser.DeltaParams{DeltaQPresent: true, DeltaQResLog2: 1},
		},
		Jobs: []internaltile.Job{{Tile: 3, Row: 1, Col: 1, SBX: 1, SBY: 1, SBCols: 2, SBRows: 2}},
	}
}

func publicBenchmarkLowOverheadStream() []byte {
	stream := make([]byte, 0, 512)
	stream = appendOBUForTest(stream, internalobu.TypeTemporalDelimiter, nil)
	stream = appendOBUForTest(stream, internalobu.TypeSequenceHeader, []byte{0x12, 0x34, 0x56})
	for i := range 10 {
		payload := make([]byte, 24)
		for j := range payload {
			payload[j] = byte(i*17 + j)
		}
		stream = appendOBUForTest(stream, internalobu.TypeTileGroup, payload)
	}
	stream = appendOBUForTest(stream, internalobu.TypeFrame, []byte{0xaa, 0xbb, 0xcc})
	return stream
}

func publicBenchmarkTemporalUnitStream() []byte {
	stream := make([]byte, 0, 512)
	for i := range 4 {
		stream = appendOBUForTest(stream, internalobu.TypeTemporalDelimiter, nil)
		stream = appendOBUForTest(stream, internalobu.TypeFrameHeader, []byte{byte(i)})
		for j := range 2 {
			stream = appendOBUForTest(stream, internalobu.TypeTileGroup, []byte{byte(i), byte(j), byte(i + j)})
		}
	}
	return stream
}

func publicBenchmarkAnnexBStream() []byte {
	td := []byte{byte(internalobu.TypeTemporalDelimiter) << 3}
	seq := []byte{byte(internalobu.TypeSequenceHeader) << 3, 0x01, 0x02}
	fh0 := []byte{byte(internalobu.TypeFrameHeader) << 3, 0x10}
	fh1 := []byte{byte(internalobu.TypeFrameHeader) << 3, 0x20}
	tile0 := []byte{byte(internalobu.TypeTileGroup) << 3, 0x30, 0x31, 0x32}
	tile1 := []byte{byte(internalobu.TypeTileGroup) << 3, 0x40, 0x41, 0x42}
	frame0 := []byte{byte(internalobu.TypeFrame) << 3, 0x50}
	frame1 := []byte{byte(internalobu.TypeFrame) << 3, 0x60}
	return appendPublicAnnexBStream(nil,
		[][][]byte{{td, seq}, {fh0, tile0, tile1}},
		[][][]byte{{td, frame0}, {fh1, frame1}},
	)
}

func publicBenchmarkRTPFrame() []byte {
	frame := make([]byte, 0, 512)
	frame = appendOBUForTest(frame, internalobu.TypeSequenceHeader, []byte{0xaa, 0xbb, 0xcc})
	for i := range 3 {
		payload := make([]byte, 96)
		for j := range payload {
			payload[j] = byte(i*29 + j)
		}
		frame = appendOBUForTest(frame, internalobu.TypeTileGroup, payload)
	}
	frame = appendOBUForTest(frame, internalobu.TypeFrame, []byte{0xdd, 0xee})
	return frame
}
