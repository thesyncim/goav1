package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestPoolExecute(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}

	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	var seen [4]uint16
	err = pool.Execute(batches[:n], jobs[:], func(batch Batch, batchJobs []tile.Job) error {
		if len(batchJobs) != batch.Count {
			t.Fatalf("jobs len=%d count=%d", len(batchJobs), batch.Count)
		}
		for i := 0; i < len(batchJobs); i++ {
			seen[batch.FirstJob+i] = batchJobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(jobs); i++ {
		if seen[i] != jobs[i].Tile+1 {
			t.Fatalf("seen[%d]=%d job=%+v", i, seen[i], jobs[i])
		}
	}
}

func TestPoolPropagatesFirstError(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	want := errors.New("decode batch")

	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	err = pool.Execute(batches[:n], jobs[:], func(batch Batch, batchJobs []tile.Job) error {
		if batch.Worker == 1 {
			return want
		}
		return nil
	})
	if !errors.Is(err, want) {
		t.Fatalf("Execute err=%v want %v", err, want)
	}
}

func TestPoolRejectsInvalidInputs(t *testing.T) {
	jobs := testJobs()
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	err = pool.Execute([]Batch{{Worker: 2, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}}, jobs[:], func(Batch, []tile.Job) error {
		return nil
	})
	if !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("Execute invalid batch err=%v want %v", err, ErrInvalidBatch)
	}

	err = pool.Execute([]Batch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}}, jobs[:], nil)
	if !errors.Is(err, ErrInvalidCallback) {
		t.Fatalf("Execute nil callback err=%v want %v", err, ErrInvalidCallback)
	}
}

func TestPoolRejectsAfterClose(t *testing.T) {
	jobs := testJobs()
	batch := Batch{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}
	pool, err := NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	pool.Close()
	pool.Close()

	err = pool.Execute([]Batch{batch}, jobs[:], func(Batch, []tile.Job) error {
		return nil
	})
	if !errors.Is(err, ErrPoolClosed) {
		t.Fatalf("Execute err=%v want %v", err, ErrPoolClosed)
	}
}

func TestPoolExecuteAllocs(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	allocs := testing.AllocsPerRun(1000, func() {
		err := pool.Execute(batches[:n], jobs[:], func(Batch, []tile.Job) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Pool.Execute allocated: %f", allocs)
	}
}

func BenchmarkPoolExecute(b *testing.B) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		b.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = pool.Execute(batches[:n], jobs[:], func(Batch, []tile.Job) error {
			return nil
		})
	}
}
