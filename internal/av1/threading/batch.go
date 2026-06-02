package threading

import "github.com/thesyncim/goav1/internal/av1/tile"

// Batch is a deterministic contiguous range of tile jobs assigned to one
// worker lane. It is a plan, not a goroutine; execution policy stays separate.
type Batch struct {
	FirstJob uint32
	Count    uint32
	Units    uint32

	Worker    uint16
	FirstTile uint16
	LastTile  uint16
}

// BuildBatches partitions jobs into deterministic contiguous worker batches.
// dst is caller-owned and may be reused across frames.
func BuildBatches(dst []Batch, jobs []tile.Job, workers int) (int, error) {
	if workers <= 0 {
		return 0, ErrInvalidWorkerCount
	}
	jobCount := len(jobs)
	if jobCount == 0 {
		return 0, nil
	}
	if uint64(jobCount) > uint64(^uint32(0)) {
		return 0, ErrInvalidJobs
	}

	batchCount := min(workers, jobCount)
	if len(dst) < batchCount {
		return 0, ErrBatchBufferTooSmall
	}
	if batchCount > int(^uint16(0))+1 {
		return 0, ErrInvalidWorkerCount
	}

	totalUnits := uint32(0)
	for i := range jobCount {
		units := jobUnits(jobs[i])
		if units == 0 {
			return 0, ErrInvalidJobs
		}
		totalUnits += units
	}

	first := 0
	remainingUnits := totalUnits
	for batch := 0; batch < batchCount; batch++ {
		remainingBatches := batchCount - batch
		count := jobCount - first
		units := uint32(0)
		if remainingBatches != 1 {
			target := ceilDiv32(remainingUnits, uint32(remainingBatches))
			count = chooseBatchCount(jobs[first:], target, remainingBatches)
			for i := 0; i < count; i++ {
				units += jobUnits(jobs[first+i])
			}
		} else {
			for i := first; i < jobCount; i++ {
				units += jobUnits(jobs[i])
			}
		}

		dst[batch] = Batch{
			FirstJob:  uint32(first),
			Count:     uint32(count),
			Units:     units,
			Worker:    uint16(batch),
			FirstTile: jobs[first].Tile,
			LastTile:  jobs[first+count-1].Tile,
		}
		first += count
		remainingUnits -= units
	}
	return batchCount, nil
}

func (b Batch) jobRange(jobCount int) (int, int, error) {
	start64 := uint64(b.FirstJob)
	end64 := start64 + uint64(b.Count)
	if b.Count == 0 || end64 > uint64(jobCount) || end64 > uint64(^uint(0)>>1) {
		return 0, 0, ErrInvalidBatch
	}
	return int(start64), int(end64), nil
}

func (b Batch) jobSlice(jobs []tile.Job) ([]tile.Job, error) {
	start, end, err := b.jobRange(len(jobs))
	if err != nil {
		return nil, err
	}
	return jobs[start:end], nil
}

func chooseBatchCount(jobs []tile.Job, target uint32, remainingBatches int) int {
	units := uint32(0)
	count := 0
	for count < len(jobs) {
		leftAfterTake := len(jobs) - (count + 1)
		if count != 0 && leftAfterTake < remainingBatches-1 {
			break
		}
		next := jobUnits(jobs[count])
		if count != 0 && absDiff32(target, units) <= absDiff32(target, units+next) {
			break
		}
		units += next
		count++
		if units >= target {
			break
		}
	}
	if count == 0 {
		return 1
	}
	return count
}

func jobUnits(job tile.Job) uint32 {
	return uint32(job.SBCols) * uint32(job.SBRows)
}

func ceilDiv32(a uint32, b uint32) uint32 {
	return (a + b - 1) / b
}

func absDiff32(a uint32, b uint32) uint32 {
	if a > b {
		return a - b
	}
	return b - a
}
