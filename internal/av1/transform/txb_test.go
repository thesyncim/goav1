package transform

import (
	"errors"
	"testing"
)

func TestMaxEOBMatchesLibaomBlockd(t *testing.T) {
	want := [...]int{
		16, 64, 256, 1024, 1024, 32, 32, 128, 128, 512,
		512, 1024, 1024, 64, 64, 256, 256, 512, 512,
	}
	for i, size := range libaomScanSizes {
		got, err := MaxEOB(size)
		if err != nil {
			t.Fatalf("MaxEOB(%+v): %v", size, err)
		}
		if got != want[i] {
			t.Fatalf("MaxEOB(%+v)=%d want %d", size, got, want[i])
		}
	}
}

func TestLowerLevelsCtxEOBMatchesLibaomTXBCommon(t *testing.T) {
	tests := []struct {
		size      Size
		scanIndex int
		want      int
	}{
		{size: Size{Width: 4, Height: 4}, scanIndex: 0, want: 0},
		{size: Size{Width: 4, Height: 4}, scanIndex: 1, want: 1},
		{size: Size{Width: 4, Height: 4}, scanIndex: 2, want: 1},
		{size: Size{Width: 4, Height: 4}, scanIndex: 3, want: 2},
		{size: Size{Width: 4, Height: 4}, scanIndex: 4, want: 2},
		{size: Size{Width: 4, Height: 4}, scanIndex: 5, want: 3},
		{size: Size{Width: 64, Height: 64}, scanIndex: 128, want: 1},
		{size: Size{Width: 64, Height: 64}, scanIndex: 256, want: 2},
		{size: Size{Width: 64, Height: 64}, scanIndex: 257, want: 3},
	}
	for _, tt := range tests {
		got, err := LowerLevelsCtxEOB(tt.size, tt.scanIndex)
		if err != nil {
			t.Fatalf("LowerLevelsCtxEOB(%+v,%d): %v", tt.size, tt.scanIndex, err)
		}
		if got != tt.want {
			t.Fatalf("LowerLevelsCtxEOB(%+v,%d)=%d want %d", tt.size, tt.scanIndex, got, tt.want)
		}
	}
}

func TestTXBHelpersRejectInvalidInputs(t *testing.T) {
	if _, err := MaxEOB(Size{Width: 4, Height: 12}); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid MaxEOB err=%v want %v", err, ErrInvalidTransform)
	}
	if _, err := LowerLevelsCtxEOB(Size{Width: 4, Height: 4}, 16); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("large eob ctx err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestTXBHelpersAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := MaxEOB(Size{Width: 64, Height: 64}); err != nil {
			t.Fatal(err)
		}
		if _, err := LowerLevelsCtxEOB(Size{Width: 64, Height: 64}, 257); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("txb helpers allocated: %f", allocs)
	}
}

func FuzzTXBHelpers(f *testing.F) {
	f.Add(uint8(0), uint16(0))
	f.Add(uint8(4), uint16(1023))
	f.Add(uint8(18), uint16(511))

	f.Fuzz(func(t *testing.T, rawSize uint8, rawIndex uint16) {
		size := libaomScanSizes[int(rawSize)%len(libaomScanSizes)]
		maxEOB, err := MaxEOB(size)
		if err != nil {
			t.Fatalf("MaxEOB(%+v): %v", size, err)
		}
		index := int(rawIndex) % maxEOB
		ctx, err := LowerLevelsCtxEOB(size, index)
		if err != nil {
			t.Fatalf("LowerLevelsCtxEOB(%+v,%d): %v", size, index, err)
		}
		if ctx < 0 || ctx > 3 {
			t.Fatalf("ctx=%d outside [0,3]", ctx)
		}
	})
}
