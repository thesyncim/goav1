package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestBuildBatchesBalancesContiguousJobs(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch

	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("n=%d", n)
	}
	if batches[0] != (Batch{Worker: 0, FirstJob: 0, Count: 2, FirstTile: 0, LastTile: 1, Units: 10}) {
		t.Fatalf("batch[0]=%+v", batches[0])
	}
	if batches[1] != (Batch{Worker: 1, FirstJob: 2, Count: 2, FirstTile: 2, LastTile: 3, Units: 15}) {
		t.Fatalf("batch[1]=%+v", batches[1])
	}
}

func TestBuildBatchesCapsWorkersToJobs(t *testing.T) {
	jobs := testJobs()
	var batches [8]Batch

	n, err := BuildBatches(batches[:], jobs[:], 8)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(jobs) {
		t.Fatalf("n=%d want %d", n, len(jobs))
	}
	for i := 0; i < n; i++ {
		if batches[i].Worker != uint16(i) || batches[i].Count != 1 || batches[i].FirstJob != i {
			t.Fatalf("batch[%d]=%+v", i, batches[i])
		}
	}
}

func TestBuildBatchesRejectsInvalidWorkerCount(t *testing.T) {
	var batches [1]Batch
	jobs := testJobs()
	_, err := BuildBatches(batches[:], jobs[:], 0)
	if !errors.Is(err, ErrInvalidWorkerCount) {
		t.Fatalf("BuildBatches err=%v want %v", err, ErrInvalidWorkerCount)
	}
}

func TestBuildBatchesRejectsShortBuffer(t *testing.T) {
	var batches [1]Batch
	jobs := testJobs()
	_, err := BuildBatches(batches[:], jobs[:], 2)
	if !errors.Is(err, ErrBatchBufferTooSmall) {
		t.Fatalf("BuildBatches err=%v want %v", err, ErrBatchBufferTooSmall)
	}
}

func TestBuildBatchesRejectsZeroAreaJob(t *testing.T) {
	jobs := [1]tile.Job{{Tile: 0, SBCols: 0, SBRows: 1}}
	var batches [1]Batch
	_, err := BuildBatches(batches[:], jobs[:], 1)
	if !errors.Is(err, ErrInvalidJobs) {
		t.Fatalf("BuildBatches err=%v want %v", err, ErrInvalidJobs)
	}
}

func TestFrameWorkBatchJobPayload(t *testing.T) {
	payload := []byte{0xaa, 0xbb, 0xcc}
	ctx := FrameWorkBatch{
		Payload: payload,
		Jobs: []tile.Job{
			{Offset: 1, Size: 2},
			{Offset: 0, Size: 1},
		},
	}

	if err := ctx.ValidatePayloads(); err != nil {
		t.Fatal(err)
	}
	data, err := ctx.JobPayload(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 2 || data[0] != 0xbb || data[1] != 0xcc {
		t.Fatalf("payload=%v", data)
	}
	if len(data) != 0 {
		data[0] = 0xdd
	}
	if payload[1] != 0xdd {
		t.Fatalf("payload did not alias: %v", payload)
	}
}

func TestFrameWorkBatchJobPayloadRejectsInvalidInputs(t *testing.T) {
	ctx := FrameWorkBatch{
		Payload: []byte{0xaa},
		Jobs:    []tile.Job{{Offset: 0, Size: 2}},
	}
	if _, err := ctx.JobPayload(-1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("negative index err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.JobPayload(1); !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("large index err=%v want %v", err, ErrInvalidBatch)
	}
	if _, err := ctx.JobPayload(0); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("invalid range err=%v want %v", err, tile.ErrInvalidPlan)
	}
	if err := ctx.ValidatePayloads(); !errors.Is(err, tile.ErrInvalidPlan) {
		t.Fatalf("ValidatePayloads err=%v want %v", err, tile.ErrInvalidPlan)
	}
}

func TestBuildBatchesAllocs(t *testing.T) {
	jobs := testJobs()
	var batches [4]Batch

	allocs := testing.AllocsPerRun(1000, func() {
		_, err := BuildBatches(batches[:], jobs[:], 3)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BuildBatches allocated: %f", allocs)
	}
}

func TestFrameWorkBatchJobPayloadAllocs(t *testing.T) {
	payload := []byte{0xaa, 0xbb, 0xcc}
	ctx := FrameWorkBatch{
		Payload: payload,
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 2},
		},
	}
	allocs := testing.AllocsPerRun(1000, func() {
		data, err := ctx.JobPayload(1)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != 2 {
			t.Fatalf("payload=%v", data)
		}
		if err := ctx.ValidatePayloads(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FrameWorkBatch.JobPayload allocated: %f", allocs)
	}
}

func FuzzBuildBatches(f *testing.F) {
	f.Add([]byte{2, 3, 2, 2, 2, 3, 3, 2, 3})
	f.Add([]byte{8, 1, 1})

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 3 {
			return
		}
		workers := int(data[0] & 7)
		count := int(data[1]&7) + 1
		if count > 8 {
			count = 8
		}
		if len(data) < 2+count {
			return
		}
		var jobs [8]tile.Job
		for i := 0; i < count; i++ {
			sb := data[2+i]
			jobs[i] = tile.Job{
				Tile:   uint16(i),
				SBCols: uint16(sb&7) + 1,
				SBRows: uint16((sb>>3)&7) + 1,
			}
		}
		var batches [8]Batch
		n, err := BuildBatches(batches[:], jobs[:count], workers)
		if err != nil {
			return
		}
		if n == 0 || n > workers || n > count {
			t.Fatalf("n=%d workers=%d count=%d", n, workers, count)
		}
		next := 0
		for i := 0; i < n; i++ {
			batch := batches[i]
			if batch.FirstJob != next || batch.Count <= 0 || batch.FirstJob+batch.Count > count || batch.Units == 0 {
				t.Fatalf("batch[%d]=%+v next=%d count=%d", i, batch, next, count)
			}
			next += batch.Count
		}
		if next != count {
			t.Fatalf("covered=%d count=%d", next, count)
		}
	})
}

func FuzzFrameWorkBatchJobPayload(f *testing.F) {
	f.Add(uint8(3), int16(0), int16(1))
	f.Add(uint8(3), int16(1), int16(2))
	f.Add(uint8(1), int16(0), int16(2))

	f.Fuzz(func(t *testing.T, payloadLen uint8, offset int16, size int16) {
		payload := make([]byte, int(payloadLen%64))
		job := tile.Job{Offset: int(offset), Size: int(size)}
		ctx := FrameWorkBatch{Payload: payload, Jobs: []tile.Job{job}}

		data, err := ctx.JobPayload(0)
		if err != nil {
			if _, _, rangeErr := job.PayloadRange(len(payload)); rangeErr == nil {
				t.Fatalf("JobPayload err=%v payloadLen=%d job=%+v", err, len(payload), job)
			}
			return
		}

		start, end, err := job.PayloadRange(len(payload))
		if err != nil {
			t.Fatalf("PayloadRange err=%v after JobPayload success", err)
		}
		if len(data) != end-start {
			t.Fatalf("len=%d want %d", len(data), end-start)
		}
		if len(data) != 0 && &data[0] != &payload[start] {
			t.Fatalf("payload does not alias")
		}
		if err := ctx.ValidatePayloads(); err != nil {
			t.Fatalf("ValidatePayloads err=%v", err)
		}
	})
}

func BenchmarkBuildBatches(b *testing.B) {
	jobs := testJobs()
	var batches [4]Batch

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = BuildBatches(batches[:], jobs[:], 3)
	}
}

func BenchmarkFrameWorkBatchJobPayload(b *testing.B) {
	payload := []byte{0xaa, 0xbb, 0xcc}
	ctx := FrameWorkBatch{
		Payload: payload,
		Jobs: []tile.Job{
			{Offset: 0, Size: 1},
			{Offset: 1, Size: 2},
		},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ctx.JobPayload(1)
	}
}

func testJobs() [4]tile.Job {
	return [4]tile.Job{
		{Tile: 0, SBCols: 3, SBRows: 2},
		{Tile: 1, SBCols: 2, SBRows: 2},
		{Tile: 2, SBCols: 3, SBRows: 3},
		{Tile: 3, SBCols: 2, SBRows: 3},
	}
}
