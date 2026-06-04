package tile

import (
	"errors"
	"testing"
)

func TestJobPayload(t *testing.T) {
	payload := []byte{0, 1, 2, 3, 4, 5}
	job := Job{Offset: 2, Size: 3}

	start, end, err := job.PayloadRange(len(payload))
	if err != nil {
		t.Fatal(err)
	}
	if start != 2 || end != 5 {
		t.Fatalf("range=%d:%d", start, end)
	}
	data, err := job.Payload(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 3 || data[0] != 2 || data[2] != 4 {
		t.Fatalf("payload=%v", data)
	}
}

func TestJobPayloadAllowsEmptyTile(t *testing.T) {
	payload := []byte{0, 1, 2}
	data, err := (Job{Offset: 1}).Payload(payload)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 0 {
		t.Fatalf("payload=%v", data)
	}
}

func TestJobPayloadRejectsInvalidRanges(t *testing.T) {
	tests := []Job{
		{Offset: 3, Size: 1},
		{Offset: ^uint32(0), Size: 1},
		{Offset: 2, Size: ^uint32(0)},
	}
	for _, job := range tests {
		_, err := job.Payload([]byte{0, 1, 2})
		if !errors.Is(err, ErrInvalidPlan) {
			t.Fatalf("job=%+v err=%v want %v", job, err, ErrInvalidPlan)
		}
	}
}

func TestValidatePayloads(t *testing.T) {
	payload := []byte{0, 1, 2, 3}
	jobs := []Job{
		{Offset: 0, Size: 2},
		{Offset: 2, Size: 2},
	}
	if err := ValidatePayloads(payload, jobs); err != nil {
		t.Fatal(err)
	}

	jobs[1].Size = 3
	if err := ValidatePayloads(payload, jobs); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("ValidatePayloads err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestNewEntropyReaderUsesJobPayload(t *testing.T) {
	payload := []byte{0x00, 0xff, 0x00}
	job := Job{Offset: 1, Size: 1}

	r, err := NewEntropyReader(payload, job, DecodeOptions{DisableCDFUpdate: true})
	if err != nil {
		t.Fatal(err)
	}
	if r.AllowCDFUpdate() {
		t.Fatal("CDF update enabled")
	}
	bit, err := r.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("bit=%d want 1", bit)
	}

	r, err = NewEntropyReader(payload, job, DecodeOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if !r.AllowCDFUpdate() {
		t.Fatal("CDF update disabled")
	}
}

func TestNewEntropyReaderRejectsInvalidRanges(t *testing.T) {
	_, err := NewEntropyReader([]byte{0xaa}, Job{Offset: 0, Size: 2}, DecodeOptions{})
	if !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("NewEntropyReader err=%v want %v", err, ErrInvalidPlan)
	}
}

func TestJobPayloadAllocs(t *testing.T) {
	payload := []byte{0, 1, 2, 3, 4, 5}
	job := Job{Offset: 1, Size: 4}
	jobs := []Job{job}

	allocs := testing.AllocsPerRun(1000, func() {
		data, err := job.Payload(payload)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != 4 {
			t.Fatalf("payload=%v", data)
		}
		if err := ValidatePayloads(payload, jobs); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Job.Payload allocated: %f", allocs)
	}
}

func TestNewEntropyReaderAllocs(t *testing.T) {
	payload := []byte{0x00, 0xff, 0x00}
	job := Job{Offset: 1, Size: 1}

	allocs := testing.AllocsPerRun(1000, func() {
		r, err := NewEntropyReader(payload, job, DecodeOptions{DisableCDFUpdate: true})
		if err != nil {
			t.Fatal(err)
		}
		if r.AllowCDFUpdate() {
			t.Fatal("CDF update enabled")
		}
		bit, err := r.ReadBit()
		if err != nil {
			t.Fatal(err)
		}
		if bit != 1 {
			t.Fatalf("bit=%d want 1", bit)
		}
	})
	if allocs != 0 {
		t.Fatalf("NewEntropyReader allocated: %f", allocs)
	}
}

func FuzzJobPayload(f *testing.F) {
	f.Add(0, 0, 0)
	f.Add(1, 2, 4)
	f.Add(-1, 1, 4)
	f.Add(4, 1, 4)

	f.Fuzz(func(t *testing.T, offset int, size int, payloadLen int) {
		if payloadLen < 0 || payloadLen > 64 {
			return
		}
		payload := make([]byte, payloadLen)
		job := Job{Offset: uint32(offset), Size: uint32(size)}
		if offset < 0 || size < 0 {
			job = Job{Offset: ^uint32(0), Size: 1}
		}
		data, err := job.Payload(payload)
		if err != nil {
			if data != nil {
				t.Fatalf("data=%v err=%v", data, err)
			}
			return
		}
		start, end, err := job.PayloadRange(len(payload))
		if err != nil {
			t.Fatal(err)
		}
		if start != offset || end != offset+size || len(data) != size {
			t.Fatalf("range=%d:%d data=%d offset=%d size=%d", start, end, len(data), offset, size)
		}
	})
}

func FuzzNewEntropyReader(f *testing.F) {
	f.Add([]byte{0xff}, int16(0), int16(1), false)
	f.Add([]byte{0x00, 0xff, 0x00}, int16(1), int16(1), true)
	f.Add([]byte{0xaa}, int16(0), int16(2), false)

	f.Fuzz(func(t *testing.T, payload []byte, offset int16, size int16, disableCDFUpdate bool) {
		if len(payload) > 64 {
			return
		}
		job := Job{Offset: uint32(offset), Size: uint32(size)}
		if offset < 0 || size < 0 {
			job = Job{Offset: ^uint32(0), Size: 1}
		}
		r, err := NewEntropyReader(payload, job, DecodeOptions{DisableCDFUpdate: disableCDFUpdate})
		if err != nil {
			if _, _, rangeErr := job.PayloadRange(len(payload)); rangeErr == nil {
				t.Fatalf("NewEntropyReader err=%v payloadLen=%d job=%+v", err, len(payload), job)
			}
			return
		}
		if r.AllowCDFUpdate() == disableCDFUpdate {
			t.Fatalf("AllowCDFUpdate=%v disableCDFUpdate=%v", r.AllowCDFUpdate(), disableCDFUpdate)
		}
		if _, err := r.ReadBit(); err != nil {
			t.Fatalf("ReadBit err=%v", err)
		}
	})
}

func BenchmarkJobPayload(b *testing.B) {
	payload := []byte{0, 1, 2, 3, 4, 5}
	job := Job{Offset: 1, Size: 4}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = job.Payload(payload)
	}
}

func BenchmarkValidatePayloads(b *testing.B) {
	payload := []byte{0, 1, 2, 3, 4, 5}
	jobs := []Job{
		{Offset: 0, Size: 2},
		{Offset: 2, Size: 2},
		{Offset: 4, Size: 2},
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ValidatePayloads(payload, jobs)
	}
}

func BenchmarkNewEntropyReader(b *testing.B) {
	payload := []byte{0x00, 0xff, 0x00}
	job := Job{Offset: 1, Size: 1}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = NewEntropyReader(payload, job, DecodeOptions{DisableCDFUpdate: true})
	}
}
