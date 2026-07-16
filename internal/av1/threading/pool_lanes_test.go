package threading

import (
	"errors"
	"sync/atomic"
	"testing"
)

var errPoolLaneTest = errors.New("lane test error")

type poolLaneTestRunner struct {
	hits    [4]atomic.Int64
	errLane int
}

func (r *poolLaneTestRunner) RunLane(lane int) error {
	r.hits[lane].Add(1)
	if lane == r.errLane {
		return errPoolLaneTest
	}
	return nil
}

func TestPoolExecuteLanes(t *testing.T) {
	pool, err := NewPool(4)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	runner := poolLaneTestRunner{errLane: -1}
	if err := pool.ExecuteLanes(4, &runner); err != nil {
		t.Fatal(err)
	}
	for lane := range 4 {
		if got := runner.hits[lane].Load(); got != 1 {
			t.Fatalf("lane %d ran %d times, want 1", lane, got)
		}
	}
}

func TestPoolExecuteLanesReturnsError(t *testing.T) {
	pool, err := NewPool(4)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	runner := poolLaneTestRunner{errLane: 2}
	if err := pool.ExecuteLanes(4, &runner); !errors.Is(err, errPoolLaneTest) {
		t.Fatalf("ExecuteLanes error = %v, want %v", err, errPoolLaneTest)
	}
}

func TestPoolExecuteLanesSteadyStateAllocs(t *testing.T) {
	pool, err := NewPool(4)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	runner := poolLaneTestRunner{errLane: -1}
	if err := pool.ExecuteLanes(4, &runner); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(100, func() {
		if err := pool.ExecuteLanes(4, &runner); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ExecuteLanes allocated %.2f objects per run", allocs)
	}
}
