package tile

import (
	"errors"
	"testing"
)

func TestDecodeStateReset(t *testing.T) {
	payload := []byte{0x00, 0xff, 0x00}
	job := Job{Tile: 1, Offset: 1, Size: 1, UpdatesFrameContext: true}
	var state DecodeState

	if err := state.Reset(payload, job, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if state.Job != job {
		t.Fatalf("job=%+v want %+v", state.Job, job)
	}
	if !state.Reader.AllowCDFUpdate() {
		t.Fatal("CDF update disabled")
	}
	if !state.RetainFrameContext {
		t.Fatal("frame context not retained")
	}
	bit, err := state.Reader.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("bit=%d want 1", bit)
	}
}

func TestDecodeStateResetDisablesFrameContextRetention(t *testing.T) {
	payload := []byte{0xff}
	job := Job{Offset: 0, Size: 1, UpdatesFrameContext: true}
	var state DecodeState

	if err := state.Reset(payload, job, DecodeOptions{DisableCDFUpdate: true}); err != nil {
		t.Fatal(err)
	}
	if state.Reader.AllowCDFUpdate() {
		t.Fatal("CDF update enabled")
	}
	if state.RetainFrameContext {
		t.Fatal("frame context retained")
	}

	job.UpdatesFrameContext = false
	if err := state.Reset(payload, job, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if !state.Reader.AllowCDFUpdate() {
		t.Fatal("CDF update disabled")
	}
	if state.RetainFrameContext {
		t.Fatal("frame context retained")
	}
}

func TestDecodeStateResetRejectsInvalidInputs(t *testing.T) {
	var state DecodeState
	err := state.Reset([]byte{0xaa}, Job{Offset: 0, Size: 2}, DecodeOptions{})
	if !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("Reset err=%v want %v", err, ErrInvalidPlan)
	}

	var nilState *DecodeState
	err = nilState.Reset(nil, Job{}, DecodeOptions{})
	if !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil Reset err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestDecodeStateResetAllocs(t *testing.T) {
	payload := []byte{0x00, 0xff, 0x00}
	job := Job{Offset: 1, Size: 1, UpdatesFrameContext: true}
	var state DecodeState

	allocs := testing.AllocsPerRun(1000, func() {
		if err := state.Reset(payload, job, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		if !state.RetainFrameContext {
			t.Fatal("frame context not retained")
		}
		bit, err := state.Reader.ReadBit()
		if err != nil {
			t.Fatal(err)
		}
		if bit != 1 {
			t.Fatalf("bit=%d want 1", bit)
		}
	})
	if allocs != 0 {
		t.Fatalf("DecodeState.Reset allocated: %f", allocs)
	}
}

func FuzzDecodeStateReset(f *testing.F) {
	f.Add([]byte{0xff}, int16(0), int16(1), false, true)
	f.Add([]byte{0x00, 0xff, 0x00}, int16(1), int16(1), true, true)
	f.Add([]byte{0xaa}, int16(0), int16(2), false, false)

	f.Fuzz(func(t *testing.T, payload []byte, offset int16, size int16, disableCDFUpdate bool, updatesFrameContext bool) {
		if len(payload) > 64 {
			return
		}
		job := Job{Offset: int(offset), Size: int(size), UpdatesFrameContext: updatesFrameContext}
		var state DecodeState
		err := state.Reset(payload, job, DecodeOptions{DisableCDFUpdate: disableCDFUpdate})
		if err != nil {
			if _, _, rangeErr := job.PayloadRange(len(payload)); rangeErr == nil {
				t.Fatalf("Reset err=%v payloadLen=%d job=%+v", err, len(payload), job)
			}
			return
		}
		if state.Reader.AllowCDFUpdate() == disableCDFUpdate {
			t.Fatalf("AllowCDFUpdate=%v disableCDFUpdate=%v", state.Reader.AllowCDFUpdate(), disableCDFUpdate)
		}
		wantRetain := updatesFrameContext && !disableCDFUpdate
		if state.RetainFrameContext != wantRetain {
			t.Fatalf("RetainFrameContext=%v want %v", state.RetainFrameContext, wantRetain)
		}
		if _, err := state.Reader.ReadBit(); err != nil {
			t.Fatalf("ReadBit err=%v", err)
		}
	})
}

func BenchmarkDecodeStateReset(b *testing.B) {
	payload := []byte{0x00, 0xff, 0x00}
	job := Job{Offset: 1, Size: 1, UpdatesFrameContext: true}
	var state DecodeState

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = state.Reset(payload, job, DecodeOptions{})
	}
}
