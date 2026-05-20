package threading

import (
	"sync"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// BatchFunc processes one deterministic batch. The jobs slice is the exact
// contiguous range described by the Batch.
type BatchFunc func(batch Batch, jobs []tile.Job) error

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
	fn    BatchFunc
	batch Batch
	jobs  []tile.Job
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
		done <- workerResult{err: task.fn(task.batch, task.jobs)}
	}
}
