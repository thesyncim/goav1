package obu

import (
	"errors"
	"testing"
)

func TestNormalizeLowOverheadAddsSizeField(t *testing.T) {
	raw := []byte{byte(TypeFrameHeader) << 3, 0xaa, 0xbb}
	var dst [16]byte
	n, err := NormalizeLowOverhead(dst[:], raw)
	if err != nil {
		t.Fatal(err)
	}

	want := []byte{byte(TypeFrameHeader)<<3 | 0x02, 0x02, 0xaa, 0xbb}
	if string(dst[:n]) != string(want) {
		t.Fatalf("normalized=%x want=%x", dst[:n], want)
	}
}

func TestNormalizeLowOverheadPreservesExtension(t *testing.T) {
	raw := []byte{byte(TypeTileGroup)<<3 | 0x04, 0x28, 0x80}
	var dst [16]byte
	n, err := NormalizeLowOverhead(dst[:], raw)
	if err != nil {
		t.Fatal(err)
	}

	want := []byte{byte(TypeTileGroup)<<3 | 0x06, 0x28, 0x01, 0x80}
	if string(dst[:n]) != string(want) {
		t.Fatalf("normalized=%x want=%x", dst[:n], want)
	}
}

func TestNormalizeLowOverheadRejectsMismatchedExistingSize(t *testing.T) {
	raw := []byte{byte(TypeFrameHeader)<<3 | 0x02, 0x01, 0xaa, 0xbb}
	var dst [16]byte
	_, err := NormalizeLowOverhead(dst[:], raw)
	if !errors.Is(err, ErrSizeMismatch) {
		t.Fatalf("NormalizeLowOverhead err=%v want %v", err, ErrSizeMismatch)
	}
}

func TestNormalizeLowOverheadAllocs(t *testing.T) {
	raw := []byte{byte(TypeFrameHeader) << 3, 0xaa, 0xbb}
	var dst [16]byte
	allocs := testing.AllocsPerRun(1000, func() {
		_, err := NormalizeLowOverhead(dst[:], raw)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("NormalizeLowOverhead allocated: %f", allocs)
	}
}

func BenchmarkNormalizeLowOverhead(b *testing.B) {
	raw := []byte{byte(TypeFrameHeader) << 3, 0xaa, 0xbb}
	var dst [16]byte
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = NormalizeLowOverhead(dst[:], raw)
	}
}
