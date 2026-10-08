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

		got, err := collectTemporalUnitsWithBudget(decoded, 3, 200*time.Millisecond, 50*time.Millisecond)
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
			make(chan receivedTemporalUnit), 1, 200*time.Millisecond, 20*time.Millisecond)
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

		_, err := collectTemporalUnitsWithBudget(decoded, 10, 25*time.Millisecond, 10*time.Millisecond)
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
		_, err := collectTemporalUnitsWithBudget(decoded, 1, time.Second, time.Second)
		if err == nil || !strings.Contains(err.Error(), "decoded stream closed after 0/1 temporal units") {
			t.Fatalf("error = %v, want closed-stream error", err)
		}
	})
}
