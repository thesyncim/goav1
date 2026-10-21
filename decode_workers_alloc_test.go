package goav1_test

import (
	"fmt"
	"os"
	"testing"

	av1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// TestDecoderWorkersSteadyStateZeroAlloc pins the zero-allocation design on
// the multi-worker paths: SB-row wavefront lanes, the pooled CDEF and loop
// filter bands, multi-tile dispatch and the external super-res output. The
// wavefront threshold is lowered so these small clips engage it.
func TestDecoderWorkersSteadyStateZeroAlloc(t *testing.T) {
	defer threading.OverrideWavefrontMinSBRowsPerWorkerForTest(1)()
	clips := []string{
		"profile1-444-8bit-cdef-restoration-160x128.ivf",
		"profile1-444-8bit-edgemv-130x130.ivf",
		"profile1-444-8bit-multitile-2x1-256x256.ivf",
		"profile1-444-8bit-superres-inter-160x128.ivf",
		"profile1-444-8bit-filmgrain-96x96.ivf",
		"profile1-444-10bit-cdef-restoration-160x128.ivf",
		"superblock128-420-8bit-128x128.ivf",
		"superres-restoration-420-8bit-160x128.ivf",
	}
	for _, workers := range []int{2, 4} {
		for _, file := range clips {
			t.Run(fmt.Sprintf("%s/workers=%d", file, workers), func(t *testing.T) {
				ivf, err := os.ReadFile(profileClipPath(file))
				if err != nil {
					t.Fatalf("read clip: %v", err)
				}
				dec, err := av1.NewDecoderFromIVF(ivf, av1.WithWorkers(workers))
				if err != nil {
					t.Fatalf("NewDecoderFromIVF: %v", err)
				}
				defer dec.Close()
				decodeAll := func() {
					if err := dec.Reset(); err != nil {
						t.Fatalf("Reset: %v", err)
					}
					for {
						_, ok, err := dec.DecodeNext()
						if err != nil {
							t.Fatalf("DecodeNext: %v", err)
						}
						if !ok {
							return
						}
					}
				}
				for range 3 {
					decodeAll()
				}
				if allocs := testing.AllocsPerRun(10, decodeAll); allocs != 0 {
					t.Fatalf("DecodeNext allocs/run=%v want 0", allocs)
				}
			})
		}
	}
}
