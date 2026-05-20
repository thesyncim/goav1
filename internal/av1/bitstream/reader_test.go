package bitstream

import (
	"errors"
	"testing"
)

func TestReaderReadBits(t *testing.T) {
	r := NewReader([]byte{0b10110010, 0b01100000})

	got, err := r.ReadBits(3)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0b101 {
		t.Fatalf("first bits=%b", got)
	}

	bit, err := r.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("bit=%d want 1", bit)
	}

	got, err = r.ReadBits(5)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0b00100 {
		t.Fatalf("cross-byte bits=%b", got)
	}
}

func TestReaderUVLC(t *testing.T) {
	tests := []struct {
		in   byte
		want uint32
	}{
		{in: 0b10000000, want: 0},
		{in: 0b01000000, want: 1},
		{in: 0b01100000, want: 2},
		{in: 0b00100000, want: 3},
	}

	for _, tt := range tests {
		r := NewReader([]byte{tt.in})
		got, err := r.ReadUVLC()
		if err != nil {
			t.Fatal(err)
		}
		if got != tt.want {
			t.Fatalf("ReadUVLC(%08b)=%d want %d", tt.in, got, tt.want)
		}
	}
}

func TestReaderTrailingBits(t *testing.T) {
	r := NewReader([]byte{0b10100000, 0x00})
	if err := r.SkipBits(2); err != nil {
		t.Fatal(err)
	}
	if err := r.ReadTrailingBits(); err != nil {
		t.Fatal(err)
	}
}

func TestReaderRejectsBadTrailingBits(t *testing.T) {
	r := NewReader([]byte{0b11000000})
	if err := r.ReadTrailingBits(); !errors.Is(err, ErrInvalidTrailingBits) {
		t.Fatalf("ReadTrailingBits err=%v want %v", err, ErrInvalidTrailingBits)
	}
}

func TestReaderAllocs(t *testing.T) {
	src := []byte{0xff, 0x00, 0x80}
	allocs := testing.AllocsPerRun(1000, func() {
		r := NewReader(src)
		_, err := r.ReadBits(12)
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.ReadUVLC()
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Reader allocated: %f", allocs)
	}
}

func BenchmarkReaderReadBits(b *testing.B) {
	src := []byte{0xff, 0x00, 0xaa, 0x55}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := NewReader(src)
		_, _ = r.ReadBits(32)
	}
}
