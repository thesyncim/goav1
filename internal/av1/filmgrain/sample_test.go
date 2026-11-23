package filmgrain

import (
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
			got := lumaGrainSample(grain, tt.offset, tt.blockCol, tt.blockRow, tt.x, tt.y)
			want := int16(tt.wantRow*LumaGrainWidth + tt.wantCol)
			if got != want {
				t.Fatalf("sample=%d want %d", got, want)
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
			got := chromaGrainSample(grain, tt.offset, chromaSubsamplingShift(tt.subsamplingX), chromaSubsamplingShift(tt.subsamplingY), tt.blockCol, tt.blockRow, tt.x, tt.y)
			want := int16(tt.wantRow*ChromaGrainWidth + tt.wantCol)
			if got != want {
				t.Fatalf("sample=%d want %d", got, want)
			}
		})
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
