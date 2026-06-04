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

func TestReaderReadBitsFastPathBoundaries(t *testing.T) {
	r := NewReader([]byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x80})
	got, err := r.ReadBits(64)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0x0123456789abcdef {
		t.Fatalf("aligned 64 bits=%016x", got)
	}
	got, err = r.ReadBits(1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 1 {
		t.Fatalf("tail bit=%d want 1", got)
	}

	r = NewReader([]byte{0x80, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0})
	if _, err := r.ReadBit(); err != nil {
		t.Fatal(err)
	}
	got, err = r.ReadBits(56)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0x002468acf13579 {
		t.Fatalf("misaligned 56 bits=%014x", got)
	}
}

func TestReaderReadBitsMatchesBitByBit(t *testing.T) {
	src := []byte{
		0x01, 0x23, 0x45, 0x67,
		0x89, 0xab, 0xcd, 0xef,
		0x55, 0xaa, 0x00, 0xff,
	}
	for offset := 0; offset < 8; offset++ {
		for n := 0; n <= 64; n++ {
			ref := NewReader(src)
			gotReader := NewReader(src)
			if err := ref.SkipBits(offset); err != nil {
				t.Fatal(err)
			}
			if err := gotReader.SkipBits(offset); err != nil {
				t.Fatal(err)
			}
			var want uint64
			for i := 0; i < n; i++ {
				bit, err := ref.ReadBit()
				if err != nil {
					t.Fatalf("ref offset=%d n=%d i=%d err=%v", offset, n, i, err)
				}
				want = (want << 1) | uint64(bit)
			}
			got, err := gotReader.ReadBits(uint8(n))
			if err != nil {
				t.Fatalf("ReadBits offset=%d n=%d err=%v", offset, n, err)
			}
			if got != want || gotReader.BitsRead() != offset+n {
				t.Fatalf("ReadBits offset=%d n=%d got=%x bits=%d want=%x bits=%d", offset, n, got, gotReader.BitsRead(), want, offset+n)
			}
		}
	}
}

func TestReaderErrorsDoNotAdvance(t *testing.T) {
	r := NewReader([]byte{0x80})
	if _, err := r.ReadBits(65); !errors.Is(err, ErrInvalidBitCount) {
		t.Fatalf("ReadBits invalid err=%v want %v", err, ErrInvalidBitCount)
	}
	if r.BitsRead() != 0 {
		t.Fatalf("invalid bit count advanced to %d", r.BitsRead())
	}
	if _, err := r.ReadBits(9); !errors.Is(err, ErrNotEnoughBits) {
		t.Fatalf("ReadBits short err=%v want %v", err, ErrNotEnoughBits)
	}
	if r.BitsRead() != 0 {
		t.Fatalf("short read advanced to %d", r.BitsRead())
	}
	if err := r.SkipBits(8); err != nil {
		t.Fatal(err)
	}
	if _, err := r.ReadBit(); !errors.Is(err, ErrNotEnoughBits) {
		t.Fatalf("ReadBit EOF err=%v want %v", err, ErrNotEnoughBits)
	}
	if r.BitsRead() != 8 {
		t.Fatalf("EOF bit read advanced to %d", r.BitsRead())
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

func BenchmarkReaderReadBitsMisaligned64(b *testing.B) {
	src := []byte{0x80, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := NewReader(src)
		_, _ = r.ReadBit()
		_, _ = r.ReadBits(64)
	}
}
