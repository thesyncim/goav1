package quantize

import (
	"testing"
)

func TestQuantizeVectorImplsAllocs(t *testing.T) {
	if quantizeFPBlockImpl == nil {
		t.Skip("missing vector kernel on this architecture")
	}
	q := Quantizer{DC: 107, AC: 130}
	coeff := make([]int32, 16*16)
	for i := range coeff {
		coeff[i] = int32(i*37 - 4096)
	}
	qcoeff16 := make([]int16, 16*16)
	allocs := testing.AllocsPerRun(1000, func() {
		if !quantizeFPBlockImpl(qcoeff16, coeff, 16, q, 1) {
			t.Fatal("quantizeFPBlockImpl refused")
		}
	})
	if allocs != 0 {
		t.Fatalf("block vector quantizers allocated %f objects/run, want 0", allocs)
	}
}
