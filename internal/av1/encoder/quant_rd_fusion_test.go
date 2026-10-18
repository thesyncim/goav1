package encoder

import (
	"math/rand"
	"reflect"
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// Frozen c07ac861 prepare path. The old quantizer and following dispatched
// statistics remain unchanged production functions, independent of the new
// fused helper. Compare all touched scratch, levels, state and panic behavior.
func (st *lossyEncodeState) prepareInterTXBDecideOriginalQuantRD(srcPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16) bool {
	var geo *txBlockGeometry
	switch {
	case w == 8 && h == 8:
		geo = &txBlockGeometry8x8
	case w == 16 && h == 16:
		geo = &txBlockGeometry16x16
	case w == 32 && h == 32:
		geo = &txBlockGeometry32x32
	}
	if geo == nil || len(qcoeff) < geo.coeffCount {
		return st.prepareInterTXB(srcPlane, pred, predStride, stride, px, py, w, h, q, qcoeff)
	}
	residualBlockImpl(st.resScratch[:geo.sampleCount], srcPlane, py*stride+px, stride, pred, predStride, w, h)
	dcZero, acZero := quantize.ZeroBinB(q, geo.txScale)
	thr0, thr := min(dcZero, acZero), acZero
	var mask uint8
	switch w {
	case 8:
		mask = transform.ForwardDCT8x8RowSkip8BitTrusted((*[64]int32)(st.tranScratch[:64]), (*[64]int16)(st.resScratch[:64]), thr0, thr)
	case 16:
		mask = transform.ForwardDCT16x16RowSkip8BitTrusted((*[256]int32)(st.tranScratch[:256]), (*[256]int16)(st.resScratch[:256]), thr0, thr)
	default:
		mask = transform.ForwardDCT32x32RowSkip8BitTrusted((*[1024]int32)(st.tranScratch[:1024]), (*[1024]int16)(st.resScratch[:1024]), thr0, thr)
	}
	if mask == 0 {
		clear(qcoeff[:geo.coeffCount])
		return true
	}
	// Quantize only the live rows; when every level is zero the block adds
	// nothing to the skip comparison.
	if !quantize.QuantizeBRowGroupsTrusted(qcoeff[:geo.coeffCount], st.tranScratch[:geo.coeffCount], w, mask, q, geo.txScale) {
		return true
	}
	return st.accumulateInterTXBStatsDiff(geo, q, qcoeff)
}

func TestPrepareInterTXBQuantRDOriginalState(t *testing.T) {
	rng := rand.New(rand.NewSource(8145))
	for _, n := range []int{8, 16, 32} {
		for trial := 0; trial < 40; trial++ {
			q := []quantize.Quantizer{{4, 4}, {120, 140}, {1828, 4}, {4, 1828}, {147, 148}, {148, 147}, {65536, 65536}}[trial%7]
			got, want := &lossyEncodeState{}, &lossyEncodeState{}
			for i := range got.resScratch {
				got.resScratch[i] = int16(i*101 - 32768)
			}
			for i := range got.tranScratch {
				got.tranScratch[i] = int32(i*733 - 39000)
			}
			for i := range got.lumaQ {
				got.lumaQ[i] = int16(i*11 + 0x1200)
			}
			got.rdDskip, got.rdDcode, got.rdRcode = -317, 117, 913
			*want = *got
			stride, predStride := n+7, n+3
			src, pred := make([]byte, (n+3)*stride), make([]byte, n*predStride)
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
			for i := range pred {
				pred[i] = byte(rng.Intn(256))
			}
			if trial%5 == 0 {
				clear(src)
				clear(pred)
			}
			if trial%5 == 1 {
				for i := range src {
					src[i] = 255
				}
				clear(pred)
			}
			gq, wq := got.lumaQ[:n*n], want.lumaQ[:n*n]
			if trial%10 == 2 {
				// Legal overlapping private scratch keeps the original ordered path.
				gq = unsafe.Slice((*int16)(unsafe.Pointer(&got.tranScratch[4])), n*n)
				wq = unsafe.Slice((*int16)(unsafe.Pointer(&want.tranScratch[4])), n*n)
			}
			if trial%10 == 3 {
				gq, wq = gq[:len(gq)-1], wq[:len(wq)-1]
			}
			if trial%10 == 4 {
				src = src[:7]
			}
			if trial%10 == 9 {
				src = src[:6*stride]
			}
			var gv, wv bool
			var gp, wp any
			func() {
				defer func() { gp = recover() }()
				gv = got.prepareInterTXBDecide(src, pred, predStride, stride, 3, 2, n, n, q, gq)
			}()
			func() {
				defer func() { wp = recover() }()
				wv = want.prepareInterTXBDecideOriginalQuantRD(src, pred, predStride, stride, 3, 2, n, n, q, wq)
			}()
			if gv != wv || reflect.TypeOf(gp) != reflect.TypeOf(wp) ||
				got.rdDskip != want.rdDskip || got.rdDcode != want.rdDcode || got.rdRcode != want.rdRcode ||
				got.resScratch != want.resScratch || got.tranScratch != want.tranScratch ||
				got.dqScratch != want.dqScratch || got.lumaQ != want.lumaQ {
				t.Fatalf("n=%d trial=%d: allzero=%v/%v panic=%v/%v rd=(%d,%d,%d)/(%d,%d,%d)", n, trial, gv, wv, gp, wp,
					got.rdDskip, got.rdDcode, got.rdRcode, want.rdDskip, want.rdDcode, want.rdRcode)
			}
		}
	}
}
