package encoder

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestSelectIntraModeNMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x51ec7))
	const stride = 96
	src := make([]byte, stride*stride)
	recon := make([]byte, stride*stride)
	for i := range src {
		src[i] = byte(rng.Intn(256))
		recon[i] = byte(rng.Intn(256))
	}
	for _, n := range []int{4, 8, 16, 32} {
		for _, haveTop := range []bool{false, true} {
			for _, haveLeft := range []bool{false, true} {
				t.Run(fmt.Sprintf("n%d/top%v/left%v", n, haveTop, haveLeft), func(t *testing.T) {
					for i := 0; i < 64; i++ {
						px := 1 + rng.Intn(stride-n-1)
						py := 1 + rng.Intn(stride-n-1)
						gotPred := make([]byte, n*n)
						wantPred := make([]byte, n*n)
						got := selectIntraModeN(src, recon, stride, px, py, n, haveTop, haveLeft, gotPred)
						want := selectIntraModeNReference(src, recon, stride, px, py, n, haveTop, haveLeft, wantPred)
						if got != want {
							t.Fatalf("mode=%v want %v", got, want)
						}
						if string(gotPred) != string(wantPred) {
							t.Fatalf("prediction mismatch for mode %v", got)
						}
					}
				})
			}
		}
	}
}

func selectIntraModeNReference(srcPlane, reconPlane []byte, stride, px, py, n int, haveTop, haveLeft bool, pred []byte) tile.IntraMode {
	dc := dcPredictN(reconPlane, stride, px, py, n, haveTop, haveLeft)
	sadDC, sadV, sadH := 0, 1<<30, 1<<30
	for r := range n {
		row := (py+r)*stride + px
		for c := range n {
			d := int(srcPlane[row+c]) - int(dc)
			if d < 0 {
				d = -d
			}
			sadDC += d
		}
	}
	above := (py-1)*stride + px
	if haveTop {
		sadV = 0
		for r := range n {
			row := (py+r)*stride + px
			for c := range n {
				d := int(srcPlane[row+c]) - int(reconPlane[above+c])
				if d < 0 {
					d = -d
				}
				sadV += d
			}
		}
	}
	if haveLeft {
		sadH = 0
		for r := range n {
			row := (py+r)*stride + px
			left := int(reconPlane[row-1])
			for c := range n {
				d := int(srcPlane[row+c]) - left
				if d < 0 {
					d = -d
				}
				sadH += d
			}
		}
	}
	switch {
	case sadV+16 < sadDC && sadV <= sadH:
		for r := range n {
			copy(pred[r*n:r*n+n], reconPlane[above:above+n])
		}
		return tile.IntraModeVertical
	case sadH+16 < sadDC:
		for r := range n {
			v := reconPlane[(py+r)*stride+px-1]
			for c := range n {
				pred[r*n+c] = v
			}
		}
		return tile.IntraModeHorizontal
	default:
		for i := range n * n {
			pred[i] = dc
		}
		return tile.IntraModeDC
	}
}

func BenchmarkSelectIntraModeN(b *testing.B) {
	const stride = 128
	src := make([]byte, stride*stride)
	recon := make([]byte, stride*stride)
	for i := range src {
		src[i] = byte((i*37 + i>>3) & 0xff)
		recon[i] = byte((i*29 + i>>5 + 17) & 0xff)
	}
	var pred [32 * 32]byte
	for _, n := range []int{8, 16, 32} {
		b.Run(fmt.Sprintf("fused/n%d", n), func(b *testing.B) {
			mode := tile.IntraModeDC
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				px := 1 + (i*7)%(stride-n-1)
				py := 1 + (i*11)%(stride-n-1)
				mode = selectIntraModeN(src, recon, stride, px, py, n, true, true, pred[:n*n])
			}
			benchmarkIntraModeSink = mode
			benchmarkIntraPredSink = pred[0]
		})
		b.Run(fmt.Sprintf("reference/n%d", n), func(b *testing.B) {
			mode := tile.IntraModeDC
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				px := 1 + (i*7)%(stride-n-1)
				py := 1 + (i*11)%(stride-n-1)
				mode = selectIntraModeNReference(src, recon, stride, px, py, n, true, true, pred[:n*n])
			}
			benchmarkIntraModeSink = mode
			benchmarkIntraPredSink = pred[0]
		})
	}
}

var (
	benchmarkIntraModeSink tile.IntraMode
	benchmarkIntraPredSink byte
)
