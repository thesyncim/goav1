package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestEOBMultiContextForClassMatchesLibaom(t *testing.T) {
	tests := []struct {
		name  string
		class transform.Class
		want  uint8
	}{
		{name: "2d", class: transform.Class2D, want: 0},
		{name: "horizontal", class: transform.ClassHoriz, want: 1},
		{name: "vertical", class: transform.ClassVert, want: 1},
	}
	for _, tt := range tests {
		if got := eobMultiContextForClass(tt.class); got != tt.want {
			t.Fatalf("%s context=%d want %d", tt.name, got, tt.want)
		}
	}
}

func TestLumaCoeffScratchMatchesAdjustedLibaomScanBounds(t *testing.T) {
	var scratch LumaCoeffTreeScratch
	coeffs, scan, levels, err := scratch.coeffBuffers(TransformSize64x64, transform.Class2D)
	if err != nil {
		t.Fatal(err)
	}
	if len(coeffs) != 1024 || len(scan) != 1024 {
		t.Fatalf("scan buffers coeffs=%d scan=%d want adjusted 32x32", len(coeffs), len(scan))
	}
	if len(levels) != (32+txPadHorizontal)*(32+txPadHorizontal) {
		t.Fatalf("levels len=%d want adjusted padded 32x32", len(levels))
	}
}

func BenchmarkLumaCoeffTreeScratchClearCoeffDirty(b *testing.B) {
	var scratch LumaCoeffTreeScratch
	for i := range scratch.Coeffs {
		scratch.Coeffs[i] = int16(i + 1)
	}
	const dirtyLen = 128
	for i := 0; i < dirtyLen; i++ {
		scratch.InverseScan[i] = int16((i * 17) & (maxCoeffScanLen - 1))
	}
	b.ReportAllocs()
	for b.Loop() {
		scratch.coeffDirtyLen = dirtyLen
		scratch.clearCoeffDirty()
	}
}
