package encoder

import (
	"math/rand"
	"testing"
)

// TestBuildQuarterPlaneExactWindows runs the quarter-plane downsample on source
// and destination buffers sized to exactly the bytes the kernel may read, with
// random and extreme (0/255) samples. The Go-native NEON body reads 64 source
// bytes per row group, so exact windows make any over-read fail under checkptr.
func TestBuildQuarterPlaneExactWindows(t *testing.T) {
	for _, tc := range []struct {
		name      string
		qw, qh    int
		srcStride int
	}{
		{name: "one vector column", qw: 16, qh: 3, srcStride: 64},
		{name: "two vector columns", qw: 32, qh: 2, srcStride: 130},
		{name: "vector plus tail", qw: 23, qh: 4, srcStride: 101},
		{name: "tail only", qw: 5, qh: 3, srcStride: 29},
		{name: "one row", qw: 33, qh: 1, srcStride: 140},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(tc.qw*131 + tc.qh)))
			for iter := range 64 {
				// The last source row ends at 4*qw bytes of the final row, so the
				// window is (4*qh-1)*stride + 4*qw bytes, not a full stride.
				srcLen := (tc.qh*4-1)*tc.srcStride + tc.qw*4
				src := make([]byte, srcLen)
				mode := iter % 4
				for i := range src {
					switch {
					case mode == 1:
						src[i] = 255
					case mode == 2:
						src[i] = 0
					case mode == 3:
						src[i] = uint8(rng.Intn(2)) * 255
					default:
						src[i] = uint8(rng.Intn(256))
					}
				}
				got := make([]byte, tc.qw*tc.qh)
				want := make([]byte, tc.qw*tc.qh)
				buildQuarterPlane(got, src, tc.srcStride, tc.qw, tc.qh)
				buildQuarterPlanePureGo(want, src, tc.srcStride, tc.qw, tc.qh)
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("iter %d quarter[%d]=%d want %d", iter, i, got[i], want[i])
					}
				}
			}
		})
	}
}
