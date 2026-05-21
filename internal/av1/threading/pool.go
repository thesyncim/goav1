package threading

import (
	"sync"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	decodework "github.com/thesyncim/goav1/internal/av1/work"
)

// BatchFunc processes one deterministic batch. The jobs slice is the exact
// contiguous range described by the Batch.
type BatchFunc func(batch Batch, jobs []tile.Job) error

// FrameWorkSequenceContext is the compact sequence-level context needed by
// frame-work callbacks. It avoids copying the full sequence header, including
// operating-point arrays, into every worker task.
type FrameWorkSequenceContext struct {
	Profile                    uint8
	Use128x128Superblock       bool
	SBSizeLog2                 uint8
	SBSizeMIB                  uint8
	EnableFilterIntra          bool
	EnableIntraEdgeFilter      bool
	EnableInterIntraCompound   bool
	EnableMaskedCompound       bool
	EnableWarpedMotion         bool
	EnableDualFilter           bool
	EnableOrderHint            bool
	EnableJNTComp              bool
	EnableRefFrameMVS          bool
	SeqForceScreenContentTools uint8
	SeqForceIntegerMV          uint8
	OrderHintBits              uint8
	EnableSuperRes             bool
	EnableCDEF                 bool
	EnableRestoration          bool
	ColorConfig                parser.ColorConfig
	FilmGrainParamsPresent     bool
}

// FrameWorkSequenceContextFromHeader derives worker-facing sequence context
// from a parsed sequence header without allocating.
func FrameWorkSequenceContextFromHeader(seq parser.SequenceHeader) FrameWorkSequenceContext {
	sbSizeLog2 := uint8(6)
	sbSizeMIB := uint8(16)
	if seq.Use128x128Superblock {
		sbSizeLog2 = 7
		sbSizeMIB = 32
	}
	return FrameWorkSequenceContext{
		Profile:                    seq.SeqProfile,
		Use128x128Superblock:       seq.Use128x128Superblock,
		SBSizeLog2:                 sbSizeLog2,
		SBSizeMIB:                  sbSizeMIB,
		EnableFilterIntra:          seq.EnableFilterIntra,
		EnableIntraEdgeFilter:      seq.EnableIntraEdgeFilter,
		EnableInterIntraCompound:   seq.EnableInterIntraCompound,
		EnableMaskedCompound:       seq.EnableMaskedCompound,
		EnableWarpedMotion:         seq.EnableWarpedMotion,
		EnableDualFilter:           seq.EnableDualFilter,
		EnableOrderHint:            seq.EnableOrderHint,
		EnableJNTComp:              seq.EnableJNTComp,
		EnableRefFrameMVS:          seq.EnableRefFrameMVS,
		SeqForceScreenContentTools: seq.SeqForceScreenContentTools,
		SeqForceIntegerMV:          seq.SeqForceIntegerMV,
		OrderHintBits:              seq.OrderHintBits,
		EnableSuperRes:             seq.EnableSuperRes,
		EnableCDEF:                 seq.EnableCDEF,
		EnableRestoration:          seq.EnableRestoration,
		ColorConfig:                seq.ColorConfig,
		FilmGrainParamsPresent:     seq.FilmGrainParamsPresent,
	}
}

// Valid reports whether the context was derived from a parsed sequence header.
func (c FrameWorkSequenceContext) Valid() bool {
	return c.ColorConfig.BitDepth != 0 && c.SBSizeLog2 != 0 && c.SBSizeMIB != 0
}

// FrameWorkFrameContext is the parsed frame context supplied to frame-work
// tile batches. It is copied from the current decoder event so reconstruction
// callbacks can map tile jobs to frame geometry and frame-level syntax without
// reparsing payload headers.
type FrameWorkFrameContext struct {
	Sequence     FrameWorkSequenceContext
	FrameHeader  parser.FrameHeaderPrefix
	FrameSize    parser.FrameSize
	TileInfo     parser.TileInfo
	Quantization parser.QuantizationParams
	Segmentation parser.SegmentationParams
	Delta        parser.DeltaParams
	LoopFilter   parser.LoopFilterParams
	CDEF         parser.CDEFParams
	Restoration  parser.RestorationParams
	TransformRef parser.TransformReferenceParams
	SkipMode     parser.SkipModeParams
	FrameMode    parser.FrameModeParams
	GlobalMotion parser.GlobalMotionParams
	FilmGrain    parser.FilmGrainParams
}

// FrameWorkBatch is the decoder context supplied to one frame-work tile batch.
// Payload, References, and Jobs alias caller-owned storage and are valid for
// the callback invocation.
type FrameWorkBatch struct {
	Step       decodework.FrameStep
	Output     *frame.Frame
	Payload    []byte
	References []*frame.Frame
	FrameWorkFrameContext
	DisableCDFUpdate bool
	Batch            Batch
	Jobs             []tile.Job
}

// JobPayload returns the exact tile payload bytes for Jobs[index]. The
// returned slice aliases Payload and is valid only for the callback invocation.
func (b FrameWorkBatch) JobPayload(index int) ([]byte, error) {
	if index < 0 || index >= len(b.Jobs) {
		return nil, ErrInvalidBatch
	}
	return b.Jobs[index].Payload(b.Payload)
}

// JobEntropyReader returns an entropy reader over Jobs[index]'s exact tile
// payload. The reader inherits the frame-level CDF update mode carried by b.
func (b FrameWorkBatch) JobEntropyReader(index int) (entropy.Reader, error) {
	if index < 0 || index >= len(b.Jobs) {
		return entropy.Reader{}, ErrInvalidBatch
	}
	return tile.NewEntropyReader(b.Payload, b.Jobs[index], tile.DecodeOptions{
		DisableCDFUpdate: b.DisableCDFUpdate,
	})
}

// JobDecodeState initializes state for Jobs[index]'s exact tile payload. The
// state records whether this job's adapted entropy context should be retained.
func (b FrameWorkBatch) JobDecodeState(index int, state *tile.DecodeState) error {
	if index < 0 || index >= len(b.Jobs) || state == nil {
		return ErrInvalidBatch
	}
	return state.Reset(b.Payload, b.Jobs[index], tile.DecodeOptions{
		DisableCDFUpdate: b.DisableCDFUpdate,
		BaseQIdx:         b.Quantization.BaseQIdx,
	})
}

// JobUpdatesFrameContext reports whether Jobs[index] is the designated tile
// whose adapted entropy state should refresh the frame context.
func (b FrameWorkBatch) JobUpdatesFrameContext(index int) (bool, error) {
	if index < 0 || index >= len(b.Jobs) {
		return false, ErrInvalidBatch
	}
	return b.Jobs[index].UpdatesFrameContext, nil
}

// ValidatePayloads checks that every job in the batch names a byte range inside
// Payload.
func (b FrameWorkBatch) ValidatePayloads() error {
	return tile.ValidatePayloads(b.Payload, b.Jobs)
}

// FrameWorkBatchFunc processes one deterministic frame-work tile batch.
type FrameWorkBatchFunc func(FrameWorkBatch) error

// Pool is a reusable bounded worker pool for frame/tile work.
type Pool struct {
	mu      sync.Mutex
	workers []poolWorker
	done    chan workerResult
	closed  bool
}

type poolWorker struct {
	tasks chan poolTask
}

type poolTask struct {
	fn         BatchFunc
	frameFn    FrameWorkBatchFunc
	frameBatch FrameWorkBatch
	batch      Batch
	jobs       []tile.Job
}

type workerResult struct {
	err error
}

// NewPool starts a reusable bounded worker pool. Pool creation is a cold-path
// operation; Execute is the reusable hot path.
func NewPool(workers int) (*Pool, error) {
	if workers <= 0 {
		return nil, ErrInvalidWorkerCount
	}
	p := &Pool{
		workers: make([]poolWorker, workers),
		done:    make(chan workerResult, workers),
	}
	for i := 0; i < workers; i++ {
		ch := make(chan poolTask)
		p.workers[i].tasks = ch
		go poolWorkerLoop(ch, p.done)
	}
	return p, nil
}

func (p *Pool) WorkerCount() int {
	if p == nil {
		return 0
	}
	return len(p.workers)
}

// Execute dispatches batches to their assigned worker lanes and waits for all
// lanes to finish. It does not allocate after the pool has been created.
func (p *Pool) Execute(batches []Batch, jobs []tile.Job, fn BatchFunc) error {
	if p == nil || len(p.workers) == 0 {
		return ErrInvalidWorkerCount
	}
	if fn == nil {
		return ErrInvalidCallback
	}
	if len(batches) == 0 {
		return nil
	}
	if len(batches) > len(p.workers) {
		return ErrInvalidBatch
	}
	if err := validateBatches(batches, jobs, len(p.workers)); err != nil {
		return err
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}

	for i := 0; i < len(batches); i++ {
		batch := batches[i]
		p.workers[batch.Worker].tasks <- poolTask{
			fn:    fn,
			batch: batch,
			jobs:  jobs[batch.FirstJob : batch.FirstJob+batch.Count],
		}
	}

	var firstErr error
	for i := 0; i < len(batches); i++ {
		result := <-p.done
		if firstErr == nil && result.err != nil {
			firstErr = result.err
		}
	}
	p.mu.Unlock()
	return firstErr
}

// ExecuteFrameWork dispatches frame-work batches with decoder context. It
// follows Execute's validation and synchronization rules while copying base
// context into each worker task.
func (p *Pool) ExecuteFrameWork(batches []Batch, jobs []tile.Job, base FrameWorkBatch, fn FrameWorkBatchFunc) error {
	if p == nil || len(p.workers) == 0 {
		return ErrInvalidWorkerCount
	}
	if fn == nil {
		return ErrInvalidCallback
	}
	if len(batches) == 0 {
		return nil
	}
	if len(batches) > len(p.workers) {
		return ErrInvalidBatch
	}
	if err := validateBatches(batches, jobs, len(p.workers)); err != nil {
		return err
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return ErrPoolClosed
	}

	for i := 0; i < len(batches); i++ {
		batch := batches[i]
		p.workers[batch.Worker].tasks <- poolTask{
			frameFn:    fn,
			frameBatch: base,
			batch:      batch,
			jobs:       jobs[batch.FirstJob : batch.FirstJob+batch.Count],
		}
	}

	var firstErr error
	for i := 0; i < len(batches); i++ {
		result := <-p.done
		if firstErr == nil && result.err != nil {
			firstErr = result.err
		}
	}
	p.mu.Unlock()
	return firstErr
}

func (p *Pool) Close() {
	if p == nil {
		return
	}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	p.closed = true
	for i := 0; i < len(p.workers); i++ {
		close(p.workers[i].tasks)
	}
	p.mu.Unlock()
}

func validateBatches(batches []Batch, jobs []tile.Job, workers int) error {
	for i := 0; i < len(batches); i++ {
		batch := batches[i]
		if int(batch.Worker) >= workers ||
			batch.FirstJob < 0 ||
			batch.Count <= 0 ||
			batch.FirstJob+batch.Count > len(jobs) {
			return ErrInvalidBatch
		}
		if batch.FirstTile != jobs[batch.FirstJob].Tile ||
			batch.LastTile != jobs[batch.FirstJob+batch.Count-1].Tile {
			return ErrInvalidBatch
		}
	}
	return nil
}

func poolWorkerLoop(tasks <-chan poolTask, done chan<- workerResult) {
	for task := range tasks {
		if task.frameFn != nil {
			ctx := task.frameBatch
			ctx.Batch = task.batch
			ctx.Jobs = task.jobs
			done <- workerResult{err: task.frameFn(ctx)}
			continue
		}
		done <- workerResult{err: task.fn(task.batch, task.jobs)}
	}
}
