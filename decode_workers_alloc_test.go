package goav1

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestDecoderWorkersSteadyStateZeroAlloc pins the zero-allocation design on
// the multi-worker paths: SB-row wavefront lanes, the pooled CDEF and loop
// filter bands, multi-tile dispatch and the external super-res output.
// wavefrontTestVector has enough SB rows to engage wavefront lanes at the
// production threshold.
func TestDecoderWorkersSteadyStateZeroAlloc(t *testing.T) {
	clips := []string{
		profileClipPath("profile1-444-8bit-cdef-restoration-160x128.ivf"),
		profileClipPath("profile1-444-8bit-edgemv-130x130.ivf"),
		profileClipPath("profile1-444-8bit-multitile-2x1-256x256.ivf"),
		profileClipPath("profile1-444-8bit-superres-inter-160x128.ivf"),
		profileClipPath("profile1-444-8bit-filmgrain-96x96.ivf"),
		profileClipPath("profile1-444-10bit-cdef-restoration-160x128.ivf"),
		profileClipPath("superblock128-420-8bit-128x128.ivf"),
		profileClipPath("superres-restoration-420-8bit-160x128.ivf"),
		wavefrontTestVector,
	}
	for _, workers := range []int{2, 4} {
		for _, path := range clips {
			t.Run(fmt.Sprintf("%s/workers=%d", filepath.Base(path), workers), func(t *testing.T) {
				ivf, err := os.ReadFile(path)
				if err != nil {
					t.Fatalf("read clip: %v", err)
				}
				dec, err := NewDecoderFromIVF(ivf, WithWorkers(workers))
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
