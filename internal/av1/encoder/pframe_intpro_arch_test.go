package encoder

import (
	"math/rand"
	"testing"
)

// The integer projection dispatch (Go SIMD under GOEXPERIMENT=simd on arm64,
// scalar otherwise) must match the scalar reference across the shapes, guard
// boundaries and lane-overflow edges the kernels accept and reject.
func TestRealtimeIntProArchMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1A7B))
	widths := []int{1, 15, 16, 17, 32, 48, 64, 128, 256, 1024, 2040, 2048, 2056, 2064}
	heights := []int{1, 3, 4, 8, 64, 128, 255, 256, 257, 258, 260, 264}
	for _, w := range widths {
		for _, h := range heights {
			for trial := range 3 {
				stride := w + rng.Intn(9)
				ref := make([]byte, (h-1)*stride+w)
				switch trial {
				case 0:
					for i := range ref {
						ref[i] = 255
					}
				default:
					fillPFrameTestBytes(rng, ref)
				}
				gotRow := make([]int16, w)
				wantRow := make([]int16, w)
				realtimeIntProRowInBoundsArch(gotRow, ref, stride, w, h, 5)
				realtimeIntProRowInBoundsPureGo(wantRow, ref, stride, w, h, 5)
				for i := range wantRow {
					if gotRow[i] != wantRow[i] {
						t.Fatalf("row w=%d h=%d trial %d [%d]: got %d want %d", w, h, trial, i, gotRow[i], wantRow[i])
					}
				}
				gotCol := make([]int16, h)
				wantCol := make([]int16, h)
				realtimeIntProColInBoundsArch(gotCol, ref, stride, w, h, 5)
				realtimeIntProColInBoundsPureGo(wantCol, ref, stride, w, h, 5)
				for i := range wantCol {
					if gotCol[i] != wantCol[i] {
						t.Fatalf("col w=%d h=%d trial %d [%d]: got %d want %d", w, h, trial, i, gotCol[i], wantCol[i])
					}
				}
			}
		}
	}
}
