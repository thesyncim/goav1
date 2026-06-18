package encoder_test

import (
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

func TestEncodeScaledReferencePFrameLayerPoolDecodeMatchesRecon(t *testing.T) {
	const (
		baseW, baseH = 64, 48
		topW, topH   = 128, 96
		keyQ         = 64
		pQ           = 72
	)

	baseSrc := scaledReferenceDecodeBaseFrame(baseW, baseH)
	keyTU, baseRecon, err := encoder.EncodeKeyframeWithSequenceMax(baseSrc, keyQ, topW, topH)
	if err != nil {
		t.Fatalf("EncodeKeyframeWithSequenceMax: %v", err)
	}

	topSrc := scaleSource420(baseRecon, topW, topH)
	scaledTU, topRecon, err := encoder.EncodeScaledReferencePFrame(topSrc, baseRecon, pQ)
	if err != nil {
		t.Fatalf("EncodeScaledReferencePFrame: %v", err)
	}

	frames := decodeLayerPoolLowOverheads(t, keyTU, scaledTU)
	if len(frames) != 2 {
		t.Fatalf("decoded %d frames, want 2", len(frames))
	}
	comparePlane(t, "base Y", frames[0].Y, baseRecon.Y, baseW, baseH, baseW)
	comparePlane(t, "base U", frames[0].U, baseRecon.U, baseW/2, baseH/2, baseW/2)
	comparePlane(t, "base V", frames[0].V, baseRecon.V, baseW/2, baseH/2, baseW/2)
	comparePlane(t, "top Y", frames[1].Y, topRecon.Y, topW, topH, topW)
	comparePlane(t, "top U", frames[1].U, topRecon.U, topW/2, topH/2, topW/2)
	comparePlane(t, "top V", frames[1].V, topRecon.V, topW/2, topH/2, topW/2)
}

func decodeLayerPoolLowOverheads(t *testing.T, payloads ...[]byte) []*goav1.Frame {
	t.Helper()

	const workers = 1
	workerPool, err := goav1.NewTileWorkerPool(workers)
	if err != nil {
		t.Fatalf("NewTileWorkerPool: %v", err)
	}
	defer workerPool.Close()

	layerPool := newScaledReferenceDecodeLayerPool(t, 4, goav1.RefFrames+1)
	adapter := goav1.NewDecoderFrameLayerPool(&layerPool)

	var (
		stream        goav1.DecoderStream
		refs          goav1.DecoderSurfaceReferences
		state         goav1.DecoderFrameWorkState
		frameContexts goav1.DecoderSharedFrameContextStore
		stats         goav1.DecoderFrameWorkTileResidualStats
		postFilter    goav1.DecoderFrameWorkReusableSupportedPostFilterRunner
		sequence      goav1.SequenceHeader
		haveSequence  bool
	)
	referenceSurfaces := make([]int, goav1.InterRefsPerFrame)
	referenceFrames := make([]*goav1.Frame, goav1.InterRefsPerFrame)
	releases := make([]int, goav1.RefFrames)
	events := make([]goav1.DecoderEvent, 16)
	planSpans := make([]goav1.TileSpan, goav1.MaxTiles)
	planJobs := make([]goav1.TileJob, goav1.MaxTiles)
	planBatches := make([]goav1.TileBatch, goav1.MaxTiles)
	outputs := make([]*goav1.Frame, 0, len(payloads))

	for payloadIndex, payload := range payloads {
		count, err := stream.PushLowOverhead(payload, events)
		if err != nil {
			t.Fatalf("payload %d PushLowOverhead: %v", payloadIndex, err)
		}
		for eventIndex := 0; eventIndex < count; eventIndex++ {
			event := events[eventIndex]
			if event.Kind == goav1.DecoderEventSequenceHeader {
				sequence = event.SequenceHeader
				haveSequence = true
			} else if !haveSequence {
				if seq, ok := stream.SequenceHeader(); ok {
					sequence = seq
					haveSequence = true
				}
			}

			framePool := scaledReferenceDecodeFramePool(t, &layerPool, sequence, event)
			size, err := goav1.DecoderFrameWorkResidualEventScratchLen(sequence, event, workers, planSpans, planJobs, planBatches)
			if err != nil {
				t.Fatalf("payload %d event %d scratch len: %v", payloadIndex, eventIndex, err)
			}
			scratch := scaledReferenceDecodeEventScratch(size)

			var batchRunner goav1.DecoderFrameWorkBatchResidualRunner
			var batchRunnerPtr *goav1.DecoderFrameWorkBatchResidualRunner
			if size.Runner.Workers != 0 {
				batchRunner, err = goav1.BindDecoderFrameWorkBatchResidualRunner(size.Runner, scratch.Runner)
				if err != nil {
					t.Fatalf("payload %d event %d bind residual runner: %v", payloadIndex, eventIndex, err)
				}
				batchRunnerPtr = &batchRunner
			}

			var sideData goav1.DecoderFrameWorkSideData
			var sideDataPtr *goav1.DecoderFrameWorkSideData
			if scaledReferenceDecodeEventNeedsSideData(event) {
				sideData, err = goav1.BindDecoderFrameWorkSideData(sequence, event.FrameSize, event.CDEF, event.Restoration, scratch.SideData)
				if err != nil {
					t.Fatalf("payload %d event %d bind side data: %v", payloadIndex, eventIndex, err)
				}
				sideDataPtr = &sideData
			}

			globalSurface := func(local int) int {
				if framePool == nil {
					return -1
				}
				return goav1.DecoderLayerPoolGlobalSurfaceID(&layerPool, framePool, local)
			}
			result, err := goav1.RunDecoderFrameWorkEventWithResidualRunner(goav1.DecoderFrameWorkResidualEventRequest{
				State:             &state,
				Refs:              &refs,
				FramePool:         framePool,
				Sequence:          sequence,
				Event:             event,
				Align:             64,
				ReferenceSurfaces: referenceSurfaces,
				ReferenceFrames:   referenceFrames,
				Workers:           workers,
				Spans:             scratch.Spans,
				Jobs:              scratch.Jobs,
				Batches:           scratch.Batches,
				Releases:          releases,
				WorkerPool:        workerPool,
				Runner:            batchRunnerPtr,
				SideData:          sideDataPtr,
				PostRunner:        &postFilter,
				Stats:             &stats,
				External: goav1.DecoderFrameWorkExternalReferenceRuntime{
					Provider:      adapter,
					GlobalSurface: globalSurface,
					Releaser:      adapter,
					FrameContexts: &frameContexts,
				},
			})
			if err != nil {
				t.Fatalf("payload %d event %d run: %v", payloadIndex, eventIndex, err)
			}
			if goav1.DecoderEventOutputsFrame(event) {
				if result.Output == nil {
					t.Fatalf("payload %d event %d output frame is nil", payloadIndex, eventIndex)
				}
				outputs = append(outputs, result.Output)
			}
		}
	}
	return outputs
}

func scaledReferenceDecodeFramePool(t *testing.T, pool *goav1.FrameLayerPool, sequence goav1.SequenceHeader, event goav1.DecoderEvent) *goav1.FramePool {
	t.Helper()
	switch event.Kind {
	case goav1.DecoderEventFrameHeader, goav1.DecoderEventFrame, goav1.DecoderEventTileGroup:
		format, err := goav1.FrameCodedFormatFromHeaders(sequence, event.FrameSize, 64)
		if err != nil {
			t.Fatalf("FrameCodedFormatFromHeaders: %v", err)
		}
		framePool, err := pool.SubPool(format)
		if err != nil {
			t.Fatalf("layer SubPool: %v", err)
		}
		return framePool
	default:
		return nil
	}
}

func scaledReferenceDecodeEventNeedsSideData(event goav1.DecoderEvent) bool {
	switch event.Kind {
	case goav1.DecoderEventFrameHeader, goav1.DecoderEventFrame, goav1.DecoderEventTileGroup:
		return true
	default:
		return false
	}
}

func scaledReferenceDecodeEventScratch(size goav1.DecoderFrameWorkResidualEventScratchSize) goav1.DecoderFrameWorkResidualEventScratch {
	return goav1.DecoderFrameWorkResidualEventScratch{
		Runner: goav1.DecoderFrameWorkBatchResidualRunnerScratch{
			States:                  make([]goav1.TileDecodeState, size.Runner.Workers),
			Storages:                make([]goav1.DecoderFrameWorkTileResidualCDFStorage, size.Runner.Workers),
			TileScratch:             make([]goav1.DecoderFrameWorkTileResidualScratch, size.Runner.Workers),
			RestorationRequests:     make([]goav1.DecoderFrameWorkTileRestorationRequest, size.Runner.RestorationRequests),
			PredictionScratch:       make([]goav1.DecoderFrameWorkPredictionScratch, size.Runner.Workers),
			InterPredictionScratch:  make([]goav1.DecoderFrameWorkInterPredictionScratch, size.Runner.Workers),
			Stats:                   make([]goav1.DecoderFrameWorkTileResidualStats, size.Runner.Workers),
			Int32Scratch:            make([]int32, size.Runner.Int32Scratch),
			ResidualScratch:         make([]int16, size.Runner.ResidualScratch),
			LoopContextAboveScratch: make([]goav1.TileBlockLoopRootAboveContext, size.Runner.LoopContextAbove),
		},
		SideData: goav1.DecoderFrameWorkSideDataScratch{
			CDEFIndexMap:             make([]uint8, size.SideData.CDEFIndexMap),
			CDEFReadMap:              make([]bool, size.SideData.CDEFReadMap),
			LoopFilterMap:            make([]goav1.DecoderFrameWorkLoopFilterBlockRecord, size.SideData.LoopFilterMap),
			RestorationRecords:       make([]goav1.TileRestorationUnitRecord, size.SideData.RestorationRecords),
			RestorationBoundaryAbove: make([]uint16, size.SideData.RestorationBoundaryAbove),
			RestorationBoundaryBelow: make([]uint16, size.SideData.RestorationBoundaryBelow),
		},
		Spans:   make([]goav1.TileSpan, size.Plan.SpanCount),
		Jobs:    make([]goav1.TileJob, size.Plan.JobCount),
		Batches: make([]goav1.TileBatch, size.Plan.BatchCount),
	}
}

func newScaledReferenceDecodeLayerPool(t *testing.T, layers int, surfacesPerLayer int) goav1.FrameLayerPool {
	t.Helper()
	factory := goav1.FrameLayerFactoryFunc(func(format goav1.FrameFormat) (goav1.FramePool, error) {
		_, backingSize, err := goav1.FramePoolRequiredSize(format, surfacesPerLayer)
		if err != nil {
			return goav1.FramePool{}, err
		}
		return goav1.BindFramePool(make([]byte, backingSize), format,
			make([]goav1.Frame, surfacesPerLayer),
			make([]int, surfacesPerLayer),
			make([]bool, surfacesPerLayer))
	})
	layerPool, err := goav1.BindFrameLayerPool(
		make([]goav1.FramePool, layers),
		make([]goav1.FrameFormat, layers),
		make([]bool, layers),
		256,
		factory,
	)
	if err != nil {
		t.Fatalf("BindFrameLayerPool: %v", err)
	}
	return layerPool
}

func scaledReferenceDecodeBaseFrame(width, height int) encoder.SourceFrame420 {
	cw, ch := width/2, height/2
	f := encoder.SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	for y := range height {
		for x := range width {
			f.Y[y*width+x] = byte(47 + 3*x + 5*y + x*y)
		}
	}
	for y := range ch {
		for x := range cw {
			f.U[y*cw+x] = byte(96 + 3*x + 7*y)
			f.V[y*cw+x] = byte(151 + 5*x + 2*y)
		}
	}
	return f
}

func scaleSource420(src encoder.SourceFrame420, width, height int) encoder.SourceFrame420 {
	cw, ch := width/2, height/2
	dst := encoder.SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	scalePlane(dst.Y, dst.YStride, width, height, src.Y, src.YStride, src.Width, src.Height)
	scalePlane(dst.U, dst.ChromaStride, cw, ch, src.U, src.ChromaStride, src.Width/2, src.Height/2)
	scalePlane(dst.V, dst.ChromaStride, cw, ch, src.V, src.ChromaStride, src.Width/2, src.Height/2)
	return dst
}

func scalePlane(dst []byte, dstStride, dstWidth, dstHeight int, src []byte, srcStride, srcWidth, srcHeight int) {
	for y := range dstHeight {
		sy := y * srcHeight / dstHeight
		for x := range dstWidth {
			sx := x * srcWidth / dstWidth
			dst[y*dstStride+x] = src[sy*srcStride+sx]
		}
	}
}
