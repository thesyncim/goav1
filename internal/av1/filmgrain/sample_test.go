package filmgrain

import (
	"errors"
	"testing"
)

func TestLumaGrainSample(t *testing.T) {
	grain := testIndexedLumaGrain()
	tests := []struct {
		name     string
		offset   uint8
		blockCol int
		blockRow int
		x        int
		y        int
		wantCol  int
		wantRow  int
	}{
		{name: "current", offset: 0x00, x: 0, y: 0, wantCol: 9, wantRow: 9},
		{name: "left", offset: 0x00, blockCol: 1, x: 0, y: 0, wantCol: 41, wantRow: 9},
		{name: "top", offset: 0x00, blockRow: 1, x: 0, y: 0, wantCol: 9, wantRow: 41},
		{name: "top-left", offset: 0x00, blockCol: 1, blockRow: 1, x: 0, y: 0, wantCol: 41, wantRow: 41},
		{name: "max-current", offset: 0xff, x: 31, y: 31, wantCol: 70, wantRow: 70},
		{name: "max-overlap", offset: 0xff, blockCol: 1, blockRow: 1, x: 1, y: 1, wantCol: 72, wantRow: 72},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := LumaGrainSample(grain, tt.offset, tt.blockCol, tt.blockRow, tt.x, tt.y)
			if err != nil {
				t.Fatal(err)
			}
			want := int16(tt.wantRow*LumaGrainWidth + tt.wantCol)
			if got != want {
				t.Fatalf("sample=%d want %d", got, want)
			}
		})
	}
}

func TestLumaGrainSampleRejectsInvalidInputs(t *testing.T) {
	grain := testIndexedLumaGrain()
	tests := []struct {
		name     string
		grain    []int16
		blockCol int
		blockRow int
		x        int
		y        int
	}{
		{name: "short", grain: grain[:LumaGrainSamples-1]},
		{name: "bad-block-col", grain: grain, blockCol: 2},
		{name: "bad-block-row", grain: grain, blockRow: 2},
		{name: "negative-x", grain: grain, x: -1},
		{name: "negative-y", grain: grain, y: -1},
		{name: "wide-x", grain: grain, blockCol: 1, x: 31},
		{name: "wide-y", grain: grain, blockRow: 1, y: 31},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := LumaGrainSample(tt.grain, 0xff, tt.blockCol, tt.blockRow, tt.x, tt.y); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("LumaGrainSample err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func TestChromaGrainSample(t *testing.T) {
	grain := testIndexedChromaGrain()
	tests := []struct {
		name         string
		offset       uint8
		subsamplingX bool
		subsamplingY bool
		blockCol     int
		blockRow     int
		x            int
		y            int
		wantCol      int
		wantRow      int
	}{
		{name: "420-current", offset: 0x00, subsamplingX: true, subsamplingY: true, x: 0, y: 0, wantCol: 6, wantRow: 6},
		{name: "420-left", offset: 0x00, subsamplingX: true, subsamplingY: true, blockCol: 1, x: 0, y: 0, wantCol: 22, wantRow: 6},
		{name: "420-top", offset: 0x00, subsamplingX: true, subsamplingY: true, blockRow: 1, x: 0, y: 0, wantCol: 6, wantRow: 22},
		{name: "420-top-left", offset: 0x00, subsamplingX: true, subsamplingY: true, blockCol: 1, blockRow: 1, x: 0, y: 0, wantCol: 22, wantRow: 22},
		{name: "420-max-current", offset: 0xff, subsamplingX: true, subsamplingY: true, x: 15, y: 15, wantCol: 36, wantRow: 36},
		{name: "420-max-overlap", offset: 0xff, subsamplingX: true, subsamplingY: true, blockCol: 1, blockRow: 1, x: 0, y: 0, wantCol: 37, wantRow: 37},
		{name: "422-current", offset: 0x00, subsamplingX: true, x: 0, y: 0, wantCol: 6, wantRow: 9},
		{name: "422-top", offset: 0x00, subsamplingX: true, blockRow: 1, x: 0, y: 0, wantCol: 6, wantRow: 41},
		{name: "422-max-current", offset: 0xff, subsamplingX: true, x: 15, y: 31, wantCol: 36, wantRow: 70},
		{name: "422-max-overlap", offset: 0xff, subsamplingX: true, blockCol: 1, blockRow: 1, x: 0, y: 1, wantCol: 37, wantRow: 72},
		{name: "444-current", offset: 0x00, x: 0, y: 0, wantCol: 9, wantRow: 9},
		{name: "444-overlap", offset: 0xff, blockCol: 1, blockRow: 1, x: 1, y: 1, wantCol: 72, wantRow: 72},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ChromaGrainSample(grain, tt.offset, tt.subsamplingX, tt.subsamplingY, tt.blockCol, tt.blockRow, tt.x, tt.y)
			if err != nil {
				t.Fatal(err)
			}
			want := int16(tt.wantRow*ChromaGrainWidth + tt.wantCol)
			if got != want {
				t.Fatalf("sample=%d want %d", got, want)
			}
		})
	}
}

func TestChromaGrainSampleRejectsInvalidInputs(t *testing.T) {
	grain := testIndexedChromaGrain()
	tests := []struct {
		name         string
		grain        []int16
		subsamplingX bool
		subsamplingY bool
		blockCol     int
		blockRow     int
		x            int
		y            int
	}{
		{name: "short", grain: grain[:ChromaGrainSamples-1], subsamplingX: true, subsamplingY: true},
		{name: "bad-block-col", grain: grain, blockCol: 2},
		{name: "bad-block-row", grain: grain, blockRow: 2},
		{name: "negative-x", grain: grain, x: -1},
		{name: "negative-y", grain: grain, y: -1},
		{name: "wide-420-x", grain: grain, subsamplingX: true, subsamplingY: true, x: 16},
		{name: "wide-420-y", grain: grain, subsamplingX: true, subsamplingY: true, y: 16},
		{name: "wide-444-x", grain: grain, x: 32},
		{name: "wide-444-y", grain: grain, y: 32},
		{name: "bound-420-x", grain: grain, subsamplingX: true, subsamplingY: true, blockCol: 1, x: 15},
		{name: "bound-420-y", grain: grain, subsamplingX: true, subsamplingY: true, blockRow: 1, y: 15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ChromaGrainSample(tt.grain, 0xff, tt.subsamplingX, tt.subsamplingY, tt.blockCol, tt.blockRow, tt.x, tt.y); !errors.Is(err, ErrInvalidParams) {
				t.Fatalf("ChromaGrainSample err=%v want %v", err, ErrInvalidParams)
			}
		})
	}
}

func TestLumaGrainSampleAllocs(t *testing.T) {
	grain := testIndexedLumaGrain()
	chroma := testIndexedChromaGrain()
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := LumaGrainSample(grain, 0x21, 0, 0, 4, 5); err != nil {
			t.Fatal(err)
		}
		if _, err := LumaGrainSample(grain, 0x21, 1, 1, 1, 1); err != nil {
			t.Fatal(err)
		}
		if _, err := ChromaGrainSample(chroma, 0x21, true, true, 0, 0, 4, 5); err != nil {
			t.Fatal(err)
		}
		if _, err := ChromaGrainSample(chroma, 0x21, false, false, 1, 1, 1, 1); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("LumaGrainSample allocated: %f", allocs)
	}
}

func BenchmarkLumaGrainSample(b *testing.B) {
	grain := testIndexedLumaGrain()
	var sample int16
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var err error
		sample, err = LumaGrainSample(grain, uint8(i), 0, 0, i&31, (i>>5)&31)
		if err != nil {
			b.Fatal(err)
		}
	}
	if sample == 0 {
		b.Fatal("unexpected zero sample")
	}
}

func BenchmarkChromaGrainSample(b *testing.B) {
	grain := testIndexedChromaGrain()
	var sample int16
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var err error
		sample, err = ChromaGrainSample(grain, uint8(i), true, true, 0, 0, i&15, (i>>4)&15)
		if err != nil {
			b.Fatal(err)
		}
	}
	if sample == 0 {
		b.Fatal("unexpected zero sample")
	}
}

func testIndexedLumaGrain() []int16 {
	grain := make([]int16, LumaGrainSamples)
	for i := range grain {
		grain[i] = int16(i)
	}
	return grain
}

func testIndexedChromaGrain() []int16 {
	grain := make([]int16, ChromaGrainSamples)
	for i := range grain {
		grain[i] = int16(i)
	}
	return grain
}
