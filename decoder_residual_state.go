package goav1

func DecoderFrameWorkResidualScratchLen(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8, plane DecoderFrameWorkPlane, size TransformSize, typ TransformType) (int32Len int, int16Len int, err error) {
	q, lossless, err := batch.BlockQuantizer(currentQIndex, segmentID, plane)
	if err != nil {
		return 0, 0, err
	}
	return ReconstructBlockScratchLen(ReconstructBlock{
		Size:      size,
		Transform: typ,
		Quantizer: q,
		Lossless:  lossless,
	})
}

func DecoderFrameWorkResidualMaxScratchLen(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8, plane DecoderFrameWorkPlane) (int32Len int, int16Len int, err error) {
	_, lossless, err := batch.BlockQuantizer(currentQIndex, segmentID, plane)
	if err != nil {
		return 0, 0, err
	}
	return ReconstructBlockMaxScratchLen(lossless)
}

func DecoderFrameWorkBlockQuantizer(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8, plane DecoderFrameWorkPlane) (Quantizer, bool, error) {
	return batch.BlockQuantizer(currentQIndex, segmentID, plane)
}

func DecoderFrameWorkBlockQIndex(batch DecoderFrameWorkBatch, currentQIndex uint8, segmentID uint8) (uint8, bool, error) {
	return batch.BlockQIndex(currentQIndex, segmentID)
}

func DecoderFrameWorkBlockCoeffPlanePosition(batch DecoderFrameWorkBatch, index int, visit TileBlockVisit, block TileBlockCoeffBlock) (DecoderFrameWorkPlane, int, int, error) {
	return batch.BlockCoeffPlanePosition(index, visit, block)
}

func ReconstructDecoderFrameWorkBlockCoeff(batch DecoderFrameWorkBatch, index int, req DecoderFrameWorkBlockCoeffReconstruction) error {
	return batch.ReconstructBlockCoeff(index, req)
}

func InitDecoderFrameWorkTileResidualCDFStorageDefault(storage *DecoderFrameWorkTileResidualCDFStorage, baseQIndex uint8) error {
	return storage.InitDefault(baseQIndex)
}

func DecoderFrameWorkTileResidualCDFsFromStorage(storage *DecoderFrameWorkTileResidualCDFStorage) DecoderFrameWorkTileResidualCDFs {
	return storage.CDFs()
}

func InitDecoderFrameWorkTileResidualCDFStorage(batch DecoderFrameWorkBatch, storage *DecoderFrameWorkTileResidualCDFStorage) error {
	return batch.InitTileResidualCDFStorage(storage)
}

func InitDecoderFrameWorkJobDecodeState(batch DecoderFrameWorkBatch, index int, state *TileDecodeState) error {
	return batch.JobDecodeState(index, state)
}

func RetainDecoderFrameWorkTileResidualCDFStorage(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, storage *DecoderFrameWorkTileResidualCDFStorage) error {
	return batch.RetainTileResidualCDFStorage(index, state, storage)
}

func InitDecoderFrameWorkTileRestorationRequestReferences(req *DecoderFrameWorkTileRestorationRequest) error {
	return req.InitReferences()
}

func DecoderFrameWorkJobBlockLoopContextRootColumns(batch DecoderFrameWorkBatch, index int) (int, error) {
	return batch.JobBlockLoopContextRootColumns(index)
}

func BindTileBlockLoopContextCarrier(rootColumns int, above []TileBlockLoopRootAboveContext) (TileBlockLoopContextCarrier, error) {
	if rootColumns < 0 {
		return TileBlockLoopContextCarrier{}, ErrTileInvalidDecodeState
	}
	if len(above) < rootColumns {
		return TileBlockLoopContextCarrier{}, ErrFrameShortBuffer
	}
	clear(above[:rootColumns])
	return TileBlockLoopContextCarrier{
		Above: above[:rootColumns],
	}, nil
}

func DecoderFrameWorkJobBlockLoopRequest(batch DecoderFrameWorkBatch, index int, currentSegmentMap []uint8, previousSegmentMap []uint8, segmentMapStride int, carrier *TileBlockLoopContextCarrier) (TileBlockLoopRequest, error) {
	req, err := batch.JobBlockLoopRequest(index, currentSegmentMap, previousSegmentMap, segmentMapStride)
	if err != nil {
		return TileBlockLoopRequest{}, err
	}
	req.ContextCarrier = carrier
	return req, nil
}

func ReadDecoderFrameWorkIntraBlockTransforms(batch DecoderFrameWorkBatch, state *TileDecodeState, visit TileBlockLoopVisit) (DecoderFrameWorkBlockTransforms, error) {
	return batch.ReadIntraBlockTransforms(state, visit)
}

func ReadDecoderFrameWorkInterBlockTransforms(batch DecoderFrameWorkBatch, state *TileDecodeState, visit TileBlockLoopVisit) (DecoderFrameWorkBlockTransforms, error) {
	return batch.ReadInterBlockTransforms(state, visit)
}

func ReadDecoderFrameWorkBlockTransforms(batch DecoderFrameWorkBatch, state *TileDecodeState, visit TileBlockLoopVisit) (DecoderFrameWorkBlockTransforms, error) {
	return batch.ReadBlockTransforms(state, visit)
}

func DecodeAndReconstructDecoderFrameWorkJobResiduals(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, cdfs DecoderFrameWorkTileResidualCDFs, scratch *DecoderFrameWorkTileResidualScratch, req DecoderFrameWorkTileResidualRequest) (DecoderFrameWorkTileResidualStats, error) {
	return batch.DecodeAndReconstructJobResiduals(index, state, cdfs, scratch, req)
}

func DecodeAndRetainDecoderFrameWorkJobResiduals(batch DecoderFrameWorkBatch, index int, state *TileDecodeState, storage *DecoderFrameWorkTileResidualCDFStorage, scratch *DecoderFrameWorkTileResidualScratch, req DecoderFrameWorkTileResidualRequest) (DecoderFrameWorkTileResidualStats, error) {
	if storage == nil {
		return DecoderFrameWorkTileResidualStats{}, ErrThreadingInvalidBatch
	}
	if err := InitDecoderFrameWorkJobDecodeState(batch, index, state); err != nil {
		return DecoderFrameWorkTileResidualStats{}, err
	}
	stats, err := DecodeAndReconstructDecoderFrameWorkJobResiduals(batch, index, state, DecoderFrameWorkTileResidualCDFsFromStorage(storage), scratch, req)
	if err != nil {
		return stats, err
	}
	if err := RetainDecoderFrameWorkTileResidualCDFStorage(batch, index, state, storage); err != nil {
		return stats, err
	}
	return stats, nil
}

// DecoderFrameWorkBatchResidualRequest carries caller-owned state for decoding
// every job assigned to one frame-work worker batch.
type DecoderFrameWorkBatchResidualRequest struct {
	Tile DecoderFrameWorkTileResidualRequest

	CurrentSegmentMap  []uint8
	PreviousSegmentMap []uint8
	SegmentMapStride   int

	// LoopContextAbove optionally backs one job's root-column context at a time.
	// When nil, the helper leaves Tile.Loop.ContextCarrier unchanged.
	LoopContextAbove []TileBlockLoopRootAboveContext
}

// DecoderFrameWorkBatchResidualScratchSize reports the caller-owned scratch
// needed to decode all jobs in one frame-work worker batch.
type DecoderFrameWorkBatchResidualScratchSize struct {
	Int32Scratch     int
	ResidualScratch  int
	LoopContextAbove int
}

// DecoderFrameWorkBatchResidualScratch carries typed caller-owned arenas for
// BindDecoderFrameWorkBatchResidualRequestScratch.
type DecoderFrameWorkBatchResidualScratch struct {
	Int32Scratch     []int32
	ResidualScratch  []int16
	LoopContextAbove []TileBlockLoopRootAboveContext
}

// DecoderFrameWorkBatchResidualRunnerScratchSize reports the caller-owned
// per-worker state and flat scratch needed by DecoderFrameWorkBatchResidualRunner.
type DecoderFrameWorkBatchResidualRunnerScratchSize struct {
	Workers int
	Batch   DecoderFrameWorkBatchResidualScratchSize

	RestorationRequests int
	Int32Scratch        int
	ResidualScratch     int
	LoopContextAbove    int
}

// DecoderFrameWorkBatchResidualRunnerScratch carries caller-owned storage for
// BindDecoderFrameWorkBatchResidualRunner.
type DecoderFrameWorkBatchResidualRunnerScratch struct {
	States                  []TileDecodeState
	Storages                []DecoderFrameWorkTileResidualCDFStorage
	TileScratch             []DecoderFrameWorkTileResidualScratch
	RestorationRequests     []DecoderFrameWorkTileRestorationRequest
	PredictionScratch       []DecoderFrameWorkPredictionScratch
	InterPredictionScratch  []DecoderFrameWorkInterPredictionScratch
	Stats                   []DecoderFrameWorkTileResidualStats
	Int32Scratch            []int32
	ResidualScratch         []int16
	LoopContextAboveScratch []TileBlockLoopRootAboveContext
}

// DecoderFrameWorkBatchResidualRunner adapts the public residual batch decoder
// to DecoderFrameWorkBatchFunc with caller-owned per-worker state.
type DecoderFrameWorkBatchResidualRunner struct {
	Request DecoderFrameWorkBatchResidualRequest
	// UseDefaultPrediction wires each worker's PredictionScratch into Request
	// when Request.Tile.PredictionScratch is nil.
	UseDefaultPrediction bool

	States                 []TileDecodeState
	Storages               []DecoderFrameWorkTileResidualCDFStorage
	TileScratch            []DecoderFrameWorkTileResidualScratch
	RestorationRequests    []DecoderFrameWorkTileRestorationRequest
	PredictionScratch      []DecoderFrameWorkPredictionScratch
	InterPredictionScratch []DecoderFrameWorkInterPredictionScratch
	Stats                  []DecoderFrameWorkTileResidualStats

	BatchScratchSize DecoderFrameWorkBatchResidualScratchSize
	Int32Scratch     []int32
	ResidualScratch  []int16
	LoopContextAbove []TileBlockLoopRootAboveContext

	CDEFIndexMap              DecoderFrameWorkCDEFIndexMap
	LoopFilterMap             DecoderFrameWorkLoopFilterMap
	sideDataValid             bool
	sideDataRestorationActive bool
}

func DecoderFrameWorkResidualMaxFrameScratchLen(batch DecoderFrameWorkBatch, currentQIndex uint8) (int32Len int, int16Len int, err error) {
	planes := [3]DecoderFrameWorkPlane{DecoderFrameWorkPlaneY}
	planeCount := 1
	if !batch.Sequence.ColorConfig.MonoChrome {
		planes[1] = DecoderFrameWorkPlaneU
		planes[2] = DecoderFrameWorkPlaneV
		planeCount = 3
	}
	segmentCount := 1
	if batch.Segmentation.Enabled {
		segmentCount = MaxSegments
	}
	for segmentID := 0; segmentID < segmentCount; segmentID++ {
		for plane := 0; plane < planeCount; plane++ {
			next32, next16, err := DecoderFrameWorkResidualMaxScratchLen(batch, currentQIndex, uint8(segmentID), planes[plane])
			if err != nil {
				return 0, 0, err
			}
			if next32 > int32Len {
				int32Len = next32
			}
			if next16 > int16Len {
				int16Len = next16
			}
		}
	}
	return int32Len, int16Len, nil
}

func DecoderFrameWorkBatchResidualScratchLen(batch DecoderFrameWorkBatch, currentQIndex uint8) (DecoderFrameWorkBatchResidualScratchSize, error) {
	int32Len, residualLen, err := DecoderFrameWorkResidualMaxFrameScratchLen(batch, currentQIndex)
	if err != nil {
		return DecoderFrameWorkBatchResidualScratchSize{}, err
	}
	loopContextLen, err := DecoderFrameWorkBatchResidualLoopContextAboveLen(batch)
	if err != nil {
		return DecoderFrameWorkBatchResidualScratchSize{}, err
	}
	return DecoderFrameWorkBatchResidualScratchSize{
		Int32Scratch:     int32Len,
		ResidualScratch:  residualLen,
		LoopContextAbove: loopContextLen,
	}, nil
}

func DecoderFrameWorkBatchResidualRunnerScratchLen(batch DecoderFrameWorkBatch, workers int) (DecoderFrameWorkBatchResidualRunnerScratchSize, error) {
	if workers <= 0 {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, ErrThreadingInvalidWorkerCount
	}
	batchSize, err := DecoderFrameWorkBatchResidualScratchLen(batch, batch.Quantization.BaseQIdx)
	if err != nil {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, err
	}
	int32Len, ok := decoderFrameWorkResidualCheckedMul(batchSize.Int32Scratch, workers)
	if !ok {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, ErrFrameShortBuffer
	}
	residualLen, ok := decoderFrameWorkResidualCheckedMul(batchSize.ResidualScratch, workers)
	if !ok {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, ErrFrameShortBuffer
	}
	loopContextLen, ok := decoderFrameWorkResidualCheckedMul(batchSize.LoopContextAbove, workers)
	if !ok {
		return DecoderFrameWorkBatchResidualRunnerScratchSize{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkBatchResidualRunnerScratchSize{
		Workers:             workers,
		Batch:               batchSize,
		RestorationRequests: workers,
		Int32Scratch:        int32Len,
		ResidualScratch:     residualLen,
		LoopContextAbove:    loopContextLen,
	}, nil
}

func BindDecoderFrameWorkBatchResidualRequestScratch(size DecoderFrameWorkBatchResidualScratchSize, scratch DecoderFrameWorkBatchResidualScratch) (DecoderFrameWorkBatchResidualRequest, error) {
	if decoderFrameWorkResidualScratchTooShort(scratch.Int32Scratch, size.Int32Scratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.ResidualScratch, size.ResidualScratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.LoopContextAbove, size.LoopContextAbove) {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkBatchResidualRequest{
		Tile: DecoderFrameWorkTileResidualRequest{
			Int32Scratch:    scratch.Int32Scratch[:size.Int32Scratch],
			ResidualScratch: scratch.ResidualScratch[:size.ResidualScratch],
		},
		LoopContextAbove: scratch.LoopContextAbove[:size.LoopContextAbove],
	}, nil
}

func BindDecoderFrameWorkBatchResidualRunner(size DecoderFrameWorkBatchResidualRunnerScratchSize, scratch DecoderFrameWorkBatchResidualRunnerScratch) (DecoderFrameWorkBatchResidualRunner, error) {
	if size.Workers <= 0 {
		return DecoderFrameWorkBatchResidualRunner{}, ErrThreadingInvalidWorkerCount
	}
	if size.RestorationRequests < 0 || size.Batch.Int32Scratch < 0 || size.Batch.ResidualScratch < 0 || size.Batch.LoopContextAbove < 0 {
		return DecoderFrameWorkBatchResidualRunner{}, ErrFrameShortBuffer
	}
	if decoderFrameWorkResidualScratchTooShort(scratch.States, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Storages, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.TileScratch, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.PredictionScratch, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.InterPredictionScratch, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Stats, size.Workers) ||
		decoderFrameWorkResidualScratchTooShort(scratch.Int32Scratch, size.Int32Scratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.ResidualScratch, size.ResidualScratch) ||
		decoderFrameWorkResidualScratchTooShort(scratch.LoopContextAboveScratch, size.LoopContextAbove) {
		return DecoderFrameWorkBatchResidualRunner{}, ErrFrameShortBuffer
	}
	var restorationRequests []DecoderFrameWorkTileRestorationRequest
	if len(scratch.RestorationRequests) != 0 {
		if decoderFrameWorkResidualScratchTooShort(scratch.RestorationRequests, size.RestorationRequests) {
			return DecoderFrameWorkBatchResidualRunner{}, ErrFrameShortBuffer
		}
		restorationRequests = scratch.RestorationRequests[:size.RestorationRequests]
	}
	predictionScratch := scratch.PredictionScratch[:size.Workers]
	interScratch := scratch.InterPredictionScratch[:size.Workers]
	for i := range predictionScratch {
		predictionScratch[i].Inter = &interScratch[i]
	}
	return DecoderFrameWorkBatchResidualRunner{
		States:                 scratch.States[:size.Workers],
		Storages:               scratch.Storages[:size.Workers],
		TileScratch:            scratch.TileScratch[:size.Workers],
		RestorationRequests:    restorationRequests,
		PredictionScratch:      predictionScratch,
		InterPredictionScratch: interScratch,
		Stats:                  scratch.Stats[:size.Workers],
		BatchScratchSize:       size.Batch,
		Int32Scratch:           scratch.Int32Scratch[:size.Int32Scratch],
		ResidualScratch:        scratch.ResidualScratch[:size.ResidualScratch],
		LoopContextAbove:       scratch.LoopContextAboveScratch[:size.LoopContextAbove],
	}, nil
}

func (r *DecoderFrameWorkBatchResidualRunner) Run(batch DecoderFrameWorkBatch) error {
	worker, err := r.workerIndex(batch)
	if err != nil {
		return err
	}
	req, err := r.workerRequest(worker)
	if err != nil {
		return err
	}
	req.Tile.TransformMode = batch.TransformRef.TransformMode
	if req.Tile.Transforms == nil {
		req.Tile.UseDefaultTransforms = true
	}
	if err := InitDecoderFrameWorkTileResidualCDFStorage(batch, &r.Storages[worker]); err != nil {
		return err
	}
	stats, err := DecodeAndRetainDecoderFrameWorkBatchResiduals(batch, &r.States[worker], &r.Storages[worker], &r.TileScratch[worker], req)
	r.Stats[worker] = stats
	return err
}

func (r *DecoderFrameWorkBatchResidualRunner) SetSideData(side DecoderFrameWorkSideData) error {
	if r == nil {
		return ErrThreadingInvalidBatch
	}
	if side.RestorationFrameBuffers.Plan.Active && len(r.RestorationRequests) < len(r.States) {
		return ErrFrameShortBuffer
	}
	if err := side.CDEFIndexMap.Reset(); err != nil {
		return err
	}
	if err := side.LoopFilterMap.Reset(); err != nil {
		return err
	}
	if err := side.RestorationFrameBuffers.ResetRecords(); err != nil {
		return err
	}
	if side.RestorationFrameBuffers.Plan.Active {
		for i := range r.States {
			req := &r.RestorationRequests[i]
			req.Buffers = side.RestorationFrameBuffers
			if err := InitDecoderFrameWorkTileRestorationRequestReferences(req); err != nil {
				return err
			}
		}
	}
	r.CDEFIndexMap = side.CDEFIndexMap
	r.LoopFilterMap = side.LoopFilterMap
	r.sideDataValid = true
	r.sideDataRestorationActive = side.RestorationFrameBuffers.Plan.Active
	return nil
}

func SetDecoderFrameWorkBatchResidualRunnerSideData(runner *DecoderFrameWorkBatchResidualRunner, side DecoderFrameWorkSideData) error {
	return runner.SetSideData(side)
}

func (r *DecoderFrameWorkBatchResidualRunner) ResetStats() error {
	if r == nil {
		return ErrThreadingInvalidBatch
	}
	clear(r.Stats)
	return nil
}

func (r *DecoderFrameWorkBatchResidualRunner) TotalStats() (DecoderFrameWorkTileResidualStats, error) {
	if r == nil {
		return DecoderFrameWorkTileResidualStats{}, ErrThreadingInvalidBatch
	}
	var total DecoderFrameWorkTileResidualStats
	for _, stats := range r.Stats {
		decoderFrameWorkAccumulateResidualStats(&total, stats)
	}
	return total, nil
}

func DecoderFrameWorkBatchResidualLoopContextAboveLen(batch DecoderFrameWorkBatch) (int, error) {
	maxRootColumns := 0
	for i := range batch.Jobs {
		rootColumns, err := DecoderFrameWorkJobBlockLoopContextRootColumns(batch, i)
		if err != nil {
			return 0, err
		}
		if rootColumns > maxRootColumns {
			maxRootColumns = rootColumns
		}
	}
	return maxRootColumns, nil
}

func DecodeAndRetainDecoderFrameWorkBatchResiduals(batch DecoderFrameWorkBatch, state *TileDecodeState, storage *DecoderFrameWorkTileResidualCDFStorage, scratch *DecoderFrameWorkTileResidualScratch, req DecoderFrameWorkBatchResidualRequest) (DecoderFrameWorkTileResidualStats, error) {
	if state == nil || storage == nil || scratch == nil {
		return DecoderFrameWorkTileResidualStats{}, ErrThreadingInvalidBatch
	}
	var total DecoderFrameWorkTileResidualStats
	for i := range batch.Jobs {
		tileReq := req.Tile
		loopReq, err := DecoderFrameWorkJobBlockLoopRequest(batch, i, req.CurrentSegmentMap, req.PreviousSegmentMap, req.SegmentMapStride, req.Tile.Loop.ContextCarrier)
		if err != nil {
			return total, err
		}
		if req.LoopContextAbove != nil {
			rootColumns, err := DecoderFrameWorkJobBlockLoopContextRootColumns(batch, i)
			if err != nil {
				return total, err
			}
			carrier, err := BindTileBlockLoopContextCarrier(rootColumns, req.LoopContextAbove)
			if err != nil {
				return total, err
			}
			scratch.LoopContext = carrier
			loopReq.ContextCarrier = &scratch.LoopContext
		}
		decoderFrameWorkApplyBatchResidualLoopOverrides(&loopReq, req.Tile.Loop)
		tileReq.Loop = loopReq
		stats, err := DecodeAndRetainDecoderFrameWorkJobResiduals(batch, i, state, storage, scratch, tileReq)
		decoderFrameWorkAccumulateResidualStats(&total, stats)
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

func (r *DecoderFrameWorkBatchResidualRunner) workerIndex(batch DecoderFrameWorkBatch) (int, error) {
	if r == nil {
		return 0, ErrThreadingInvalidBatch
	}
	worker := int(batch.Batch.Worker)
	if worker < 0 || worker >= len(r.States) ||
		worker >= len(r.Storages) ||
		worker >= len(r.TileScratch) ||
		worker >= len(r.PredictionScratch) ||
		worker >= len(r.InterPredictionScratch) ||
		worker >= len(r.Stats) {
		return 0, ErrThreadingInvalidBatch
	}
	return worker, nil
}

func (r *DecoderFrameWorkBatchResidualRunner) workerRequest(worker int) (DecoderFrameWorkBatchResidualRequest, error) {
	req := r.Request
	int32Offset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.Int32Scratch)
	if !ok {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	residualOffset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.ResidualScratch)
	if !ok {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	loopOffset, ok := decoderFrameWorkResidualCheckedMul(worker, r.BatchScratchSize.LoopContextAbove)
	if !ok {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	if int32Offset > len(r.Int32Scratch) ||
		residualOffset > len(r.ResidualScratch) ||
		loopOffset > len(r.LoopContextAbove) {
		return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
	}
	scratch := DecoderFrameWorkBatchResidualScratch{
		Int32Scratch:     r.Int32Scratch[int32Offset:],
		ResidualScratch:  r.ResidualScratch[residualOffset:],
		LoopContextAbove: r.LoopContextAbove[loopOffset:],
	}
	bound, err := BindDecoderFrameWorkBatchResidualRequestScratch(r.BatchScratchSize, scratch)
	if err != nil {
		return DecoderFrameWorkBatchResidualRequest{}, err
	}
	req.Tile.Int32Scratch = bound.Tile.Int32Scratch
	req.Tile.ResidualScratch = bound.Tile.ResidualScratch
	req.LoopContextAbove = bound.LoopContextAbove
	if r.sideDataValid {
		req.Tile.CDEFIndexMap = &r.CDEFIndexMap
		req.Tile.LoopFilterMap = &r.LoopFilterMap
		if r.sideDataRestorationActive {
			if worker >= len(r.RestorationRequests) {
				return DecoderFrameWorkBatchResidualRequest{}, ErrFrameShortBuffer
			}
			req.Tile.Restoration = &r.RestorationRequests[worker]
		}
	}
	if req.Tile.PredictionScratch == nil && r.UseDefaultPrediction {
		req.Tile.PredictionScratch = &r.PredictionScratch[worker]
	}
	return req, nil
}

func decoderFrameWorkResidualCheckedMul(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > int(^uint(0)>>1)/a {
		return 0, false
	}
	return a * b, true
}

func decoderFrameWorkResidualScratchTooShort[T any](scratch []T, need int) bool {
	return need < 0 || len(scratch) < need
}

func decoderFrameWorkApplyBatchResidualLoopOverrides(dst *TileBlockLoopRequest, overrides TileBlockLoopRequest) {
	if overrides.ContextCarrier != nil {
		dst.ContextCarrier = overrides.ContextCarrier
	}
	dst.BeforeSuperblock = overrides.BeforeSuperblock
	dst.BeforeCoefficients = overrides.BeforeCoefficients
	dst.CoeffVisitor = overrides.CoeffVisitor
}

func decoderFrameWorkAccumulateResidualStats(total *DecoderFrameWorkTileResidualStats, next DecoderFrameWorkTileResidualStats) {
	total.Loop.PartitionReads += next.Loop.PartitionReads
	total.Loop.Blocks += next.Loop.Blocks
	total.Loop.SegmentPredictions += next.Loop.SegmentPredictions
	total.Loop.SegmentIDs += next.Loop.SegmentIDs
	total.Loop.Prefixes += next.Loop.Prefixes
	total.Loop.PredictionModes += next.Loop.PredictionModes
	total.Loop.IntraModes += next.Loop.IntraModes
	total.Loop.InterEntries += next.Loop.InterEntries
	total.Loop.InterReferences += next.Loop.InterReferences
	total.Loop.InterModes += next.Loop.InterModes
	total.Loop.RefMVStacks += next.Loop.RefMVStacks
	total.Loop.DRLIndices += next.Loop.DRLIndices
	total.Loop.InterMVReferences += next.Loop.InterMVReferences
	total.Loop.MotionVectors += next.Loop.MotionVectors
	total.Loop.MVResiduals += next.Loop.MVResiduals
	total.Loop.InterpFilters += next.Loop.InterpFilters
	total.Loop.InterIntras += next.Loop.InterIntras
	total.Loop.MotionModes += next.Loop.MotionModes
	total.Loop.CompoundBlends += next.Loop.CompoundBlends
	total.Loop.CoefficientBlocks += next.Loop.CoefficientBlocks
	total.Loop.CoefficientTXBs += next.Loop.CoefficientTXBs
	total.Loop.CoefficientNonZero += next.Loop.CoefficientNonZero
	total.Loop.CoefficientAllZero += next.Loop.CoefficientAllZero
	total.Loop.CoefficientEOBTotal += next.Loop.CoefficientEOBTotal
	total.Loop.DeltaReads += next.Loop.DeltaReads
	total.CoefficientBlocks += next.CoefficientBlocks
	total.SkippedBlocks += next.SkippedBlocks
	total.TXBs += next.TXBs
	total.NonZero += next.NonZero
	total.AllZero += next.AllZero
	total.EOBTotal += next.EOBTotal
	total.Residuals += next.Residuals
	total.Predictions += next.Predictions
	total.RestorationUnits += next.RestorationUnits
}
