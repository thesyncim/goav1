package threading

import "github.com/thesyncim/goav1/internal/av1/tile"

// Batch is a deterministic contiguous range of tile jobs assigned to one
// worker lane. It is a plan, not a goroutine; execution policy stays separate.
type Batch struct {
	Worker uint16

	FirstJob int
	Count    int

	FirstTile uint16
	LastTile  uint16
	Units     uint32
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

	batchCount := min(workers, jobCount)
	if len(dst) < batchCount {
		return 0, ErrBatchBufferTooSmall
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
			Worker:    uint16(batch),
			FirstJob:  first,
			Count:     count,
			FirstTile: jobs[first].Tile,
			LastTile:  jobs[first+count-1].Tile,
			Units:     units,
		}
		first += count
		remainingUnits -= units
	}
	return batchCount, nil
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
