package main

import (
	"bytes"
	"strings"
	"testing"
	"testing/synctest"
	"time"
)

func TestCollectTemporalUnitsWithBudgetProgressResetsInactivityDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		decoded := make(chan receivedTemporalUnit)
		go func() {
			for _, data := range [][]byte{{1}, {2}, {3}} {
				time.Sleep(20 * time.Millisecond)
				decoded <- receivedTemporalUnit{data: data}
			}
		}()

		got, err := collectTemporalUnitsWithBudget(decoded, 3, 200*time.Millisecond, 50*time.Millisecond, false)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 3 {
			t.Fatalf("collected %d temporal units, want 3", len(got))
		}
		for i, want := range [][]byte{{1}, {2}, {3}} {
			if !bytes.Equal(got[i], want) {
				t.Errorf("temporal unit %d = %v, want %v", i, got[i], want)
			}
		}
	})
}

func TestCollectTemporalUnitsWithBudgetStopsOnInactivityDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		_, err := collectTemporalUnitsWithBudget(
			make(chan receivedTemporalUnit), 1, 200*time.Millisecond, 20*time.Millisecond, false)
		if err == nil || !strings.Contains(err.Error(), "no temporal-unit progress") {
			t.Fatalf("error = %v, want no-progress timeout", err)
		}
	})
}

func TestCollectTemporalUnitsWithBudgetStopsOnOverallDeadlineDuringProgress(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		decoded := make(chan receivedTemporalUnit)
		stop := make(chan struct{})
		go func() {
			for i := byte(0); i < 10; i++ {
				select {
				case <-time.After(4 * time.Millisecond):
				case <-stop:
					return
				}
				select {
				case decoded <- receivedTemporalUnit{data: []byte{i}}:
				case <-stop:
					return
				}
			}
		}()

		_, err := collectTemporalUnitsWithBudget(decoded, 10, 25*time.Millisecond, 10*time.Millisecond, false)
		close(stop)
		synctest.Wait()
		if err == nil || !strings.Contains(err.Error(), "overall temporal-unit collection timeout") {
			t.Fatalf("error = %v, want overall timeout", err)
		}
	})
}

func TestCollectTemporalUnitsWithBudgetReportsClosedStream(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		decoded := make(chan receivedTemporalUnit)
		close(decoded)
		_, err := collectTemporalUnitsWithBudget(decoded, 1, time.Second, time.Second, false)
		if err == nil || !strings.Contains(err.Error(), "decoded stream closed after 0/1 temporal units") {
			t.Fatalf("error = %v, want closed-stream error", err)
		}
	})
}

func TestCollectTemporalUnitsWithBudgetRestartsAfterTransportLoss(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		decoded := make(chan receivedTemporalUnit)
		go func() {
			for _, unit := range []receivedTemporalUnit{
				{data: []byte{1}},
				{data: []byte{2}},
				{data: []byte{3}, dropped: 2},
				{data: []byte{4}},
				{data: []byte{5}},
			} {
				decoded <- unit
			}
		}()

		got, err := collectTemporalUnitsWithBudget(decoded, 3, time.Second, time.Second, true)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 3 {
			t.Fatalf("collected %v, want the three units from the loss on", got)
		}
		for i, want := range [][]byte{{3}, {4}, {5}} {
			if !bytes.Equal(got[i], want) {
				t.Fatalf("collected %v, want [[3] [4] [5]]", got)
			}
		}
	})
}
