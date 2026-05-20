package bitstream

import (
	"errors"
	"math"
	"testing"
)

func TestLEB128RoundTrip(t *testing.T) {
	values := []uint32{
		0,
		1,
		2,
		0x7f,
		0x80,
		0xff,
		0x4000,
		0x1fffff,
		0x0fffffff,
		math.MaxUint32,
	}

	var buf [MaxLEB128Bytes]byte
	for _, value := range values {
		n, err := PutLEB128(buf[:], value)
		if err != nil {
			t.Fatalf("PutLEB128(%d): %v", value, err)
		}

		got, consumed, err := ReadLEB128(buf[:n])
		if err != nil {
			t.Fatalf("ReadLEB128(%d): %v", value, err)
		}
		if got != value || consumed != n {
			t.Fatalf("round trip %d got value=%d consumed=%d want consumed=%d", value, got, consumed, n)
		}
		if LEB128Len(value) != n {
			t.Fatalf("LEB128Len(%d)=%d want %d", value, LEB128Len(value), n)
		}
	}
}

func TestReadLEB128RejectsMalformed(t *testing.T) {
	tests := []struct {
		name string
		in   []byte
		err  error
	}{
		{name: "empty", in: nil, err: ErrShortLEB128},
		{name: "truncated", in: []byte{0x80}, err: ErrShortLEB128},
		{name: "too many bytes", in: []byte{0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80}, err: ErrLEB128Overflow},
		{name: "value overflow", in: []byte{0x80, 0x80, 0x80, 0x80, 0x10}, err: ErrLEB128Overflow},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := ReadLEB128(tt.in)
			if !errors.Is(err, tt.err) {
				t.Fatalf("ReadLEB128() err=%v want %v", err, tt.err)
			}
		})
	}
}

func TestPutLEB128ShortBuffer(t *testing.T) {
	var buf [1]byte
	_, err := PutLEB128(buf[:], 0x80)
	if !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("PutLEB128 err=%v want %v", err, ErrShortBuffer)
	}
}

func TestLEB128Allocs(t *testing.T) {
	var buf [MaxLEB128Bytes]byte
	allocs := testing.AllocsPerRun(1000, func() {
		n, err := PutLEB128(buf[:], math.MaxUint32)
		if err != nil {
			t.Fatal(err)
		}
		_, _, err = ReadLEB128(buf[:n])
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("LEB128 hot path allocated: %f", allocs)
	}
}

func BenchmarkReadLEB128(b *testing.B) {
	src := []byte{0xff, 0xff, 0xff, 0xff, 0x0f}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _, _ = ReadLEB128(src)
	}
}

func BenchmarkPutLEB128(b *testing.B) {
	var dst [MaxLEB128Bytes]byte
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = PutLEB128(dst[:], math.MaxUint32)
	}
}
