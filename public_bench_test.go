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
