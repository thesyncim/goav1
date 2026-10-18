package encoder

import (
	"bytes"
	"fmt"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/dsp"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// rawPredReconOriginal retains the original complete copy followed by an
// independent scalar expression of the DSP numeric contract. The copy kernel
// itself is unchanged; its architecture-specific partial-overlap ordering is
// part of the legacy behavior that the fused path must preserve.
func rawPredReconOriginal(dst []byte, stride int, pred []byte, predStride int, raw []int32, w, h int) {
	copyPredToReconBlock(dst, pred, predStride, stride, 0, 0, w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r := (int64(raw[y*w+x]) + 8) >> 4
			if r < -32768 {
				r = -32768
			}
			if r > 32767 {
				r = 32767
			}
			v := int64(dst[y*stride+x]) + r
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			dst[y*stride+x] = byte(v)
		}
	}
}

var rawPredReconBoundaries = []int32{
	-1 << 31, -1<<31 + 7, -524305, -524304, -524297, -524296,
	-4097, -4096, -4089, -4088, -25, -24, -17, -16, -9, -8, -7, -1,
	0, 1, 7, 8, 9, 15, 16, 23, 24, 4071, 4072, 4087, 4088,
	524279, 524280, 524287, 524288, 524295, 524296, 1<<31 - 9, 1<<31 - 8, 1<<31 - 1,
}

func TestRawPredReconNumericBoundaries(t *testing.T) {
	const w = 64
	h := 256 * len(rawPredReconBoundaries) / w
	for _, kernel := range []struct {
		name string
		fn   func([]byte, int, []byte, int, []int32, int, int)
	}{{"scalar", addRawResidualToPredReconBlockPureGo}, {"dispatch", addRawResidualToPredReconBlock}} {
		t.Run(kernel.name, func(t *testing.T) {
			pred := make([]byte, w*h)
			raw := make([]int32, w*h)
			for i := range raw {
				pred[i] = byte(i)
				raw[i] = rawPredReconBoundaries[i/256]
			}
			want, got := make([]byte, w*h), make([]byte, w*h)
			rawPredReconOriginal(want, w, pred, w, raw, w, h)
			kernel.fn(got, w, pred, w, raw, w, h)
			if !bytes.Equal(got, want) {
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("pred=%d raw=%d: got %d want %d", pred[i], raw[i], got[i], want[i])
					}
				}
			}
		})
	}
}

func TestRawPredReconExactWindowsAndAliasing(t *testing.T) {
	for _, kernel := range []struct {
		name string
		fn   func([]byte, int, []byte, int, []int32, int, int)
	}{{"scalar", addRawResidualToPredReconBlockPureGo}, {"dispatch", addRawResidualToPredReconBlock}} {
		for _, w := range []int{4, 5, 8, 12, 16, 24, 32, 64} {
			for _, h := range []int{1, 2, 3, 8} {
				stride := w + 7
				span := (h-1)*stride + w
				for _, layout := range []struct {
					name         string
					d, p, ds, ps int
				}{
					{"disjoint", 16, 32 + span, stride, stride},
					{"inplace", 16, 16, stride, stride},
					{"forward", 17, 16, stride, stride},
					{"backward", 16, 19, stride, stride},
					{"crossrow", 16 + stride - 2, 16, stride, stride},
					{"differentstride", 16, 16, w + 9, w + 4},
				} {
					t.Run(fmt.Sprintf("%s/%dx%d/%s", kernel.name, w, h, layout.name), func(t *testing.T) {
						dlen := (h-1)*layout.ds + w
						plen := (h-1)*layout.ps + w
						n := max(layout.d+dlen, layout.p+plen) + 19
						got := make([]byte, n)
						for i := range got {
							got[i] = byte(i*73 + 17)
						}
						want := slices.Clone(got)
						raw := make([]int32, w*h)
						for i := range raw {
							raw[i] = rawPredReconBoundaries[(i*13+w+h)%len(rawPredReconBoundaries)]
						}
						beforeRaw := slices.Clone(raw)
						rawPredReconOriginal(want[layout.d:layout.d+dlen:layout.d+dlen], layout.ds, want[layout.p:layout.p+plen:layout.p+plen], layout.ps, raw, w, h)
						kernel.fn(got[layout.d:layout.d+dlen:layout.d+dlen], layout.ds, got[layout.p:layout.p+plen:layout.p+plen], layout.ps, raw, w, h)
						if !bytes.Equal(got, want) {
							t.Fatal("output, padding, or overlapping prediction differs from original")
						}
						if !slices.Equal(raw, beforeRaw) {
							t.Fatal("raw input changed")
						}
					})
				}
			}
		}
	}
}

func TestReconInterTXBResultOriginalState(t *testing.T) {
	for _, dims := range [][2]int{{4, 4}, {4, 16}, {16, 4}, {8, 8}, {8, 16}, {16, 8}, {16, 16}, {16, 32}, {32, 16}, {32, 32}} {
		w, h := dims[0], dims[1]
		geo, err := txBlockGeometryForDimensions(w, h)
		if err != nil {
			t.Fatal(err)
		}
		scan, err := transform.DefaultScan(geo.size, transform.Class2D)
		if err != nil {
			t.Fatal(err)
		}
		for _, txType := range []transform.Type{transform.TypeDCTDCT, transform.TypeADSTDCT, transform.TypeDCTADST, transform.TypeIDTX, transform.Type(255)} {
			for _, eob := range []int{0, 1, 2, 3, geo.coeffCount / 2, geo.coeffCount} {
				for _, allZero := range []bool{false, true} {
					t.Run(fmt.Sprintf("%dx%d/type%d/eob%d/zero%t", w, h, txType, eob, allZero), func(t *testing.T) {
						var got, want lossyEncodeState
						got.invScratch = make([]int32, 1024)
						for i := range got.invScratch {
							got.invScratch[i] = int32(i*911 + 17)
							got.dqScratch[i] = int32(i*131 - 7777)
							got.invResidual[i] = int16(i*31 - 12345)
						}
						want.dqScratch, want.invResidual = got.dqScratch, got.invResidual
						want.invScratch = slices.Clone(got.invScratch)
						qcoeff := make([]int16, geo.coeffCount)
						for i := range qcoeff {
							qcoeff[i] = int16((i*23+1)%97 - 48)
						}
						qBefore, scanBefore := slices.Clone(qcoeff), slices.Clone(scan)
						stride, predStride, px, py := w+13, w+3, 5, 2
						recon := make([]byte, (py+h+2)*stride)
						for i := range recon {
							recon[i] = byte(i*53 + 121)
						}
						reference := slices.Clone(recon)
						pred := make([]byte, (h-1)*predStride+w)
						for i := range pred {
							pred[i] = byte(i*71 + 29)
						}
						predBefore := slices.Clone(pred)
						q := quantize.Quantizer{DC: 1217, AC: 2039}
						result := tile.TXBDecodeResult{AllZero: allZero, EOB: uint16(eob)}
						wantErr := want.reconInterTXBResultOriginal(reference, pred, predStride, stride, px, py, w, h, q, qcoeff, geo, scan, txType, result)
						gotErr := got.reconInterTXBResult(recon, pred, predStride, stride, px, py, w, h, q, qcoeff, geo, scan, txType, result)
						if gotErr != wantErr {
							t.Fatalf("error %v want %v", gotErr, wantErr)
						}
						if !bytes.Equal(recon, reference) {
							t.Fatal("reconstruction or guard bytes differ")
						}
						if got.dqScratch != want.dqScratch || got.invResidual != want.invResidual || !slices.Equal(got.invScratch, want.invScratch) {
							t.Fatal("scratch state differs")
						}
						if !slices.Equal(qcoeff, qBefore) || !slices.Equal(scan, scanBefore) || !bytes.Equal(pred, predBefore) {
							t.Fatal("input changed")
						}
					})
				}
			}
		}
	}
}

// Frozen from b9e260f470c33a5369b5357d74a54b302044ea74. Keep the two-pass
// output and separate scan traversal independent of optimized reconstruction.
func (st *lossyEncodeState) reconInterTXBResultOriginal(reconPlane, pred []byte, predStride, stride, px, py, w, h int, q quantize.Quantizer, qcoeff []int16,
	geo txBlockGeometry, scan []int16, txType transform.Type, txbResult tile.TXBDecodeResult) error {
	n := geo.sampleCount
	if txbResult.AllZero {
		copyPredToReconBlock(reconPlane, pred, predStride, stride, px, py, w, h)
		return nil
	}
	if txbResult.EOB == 1 && txType == transform.TypeDCTDCT {
		dc := quantize.DequantizeDCCoeffBitDepthTrusted(qcoeff[0], q, geo.txScale, nil, 8)
		sample, err := transform.InverseDCTDCOnlySampleBitDepth(dc, st.invScratch[:n], geo.size, 8)
		if err != nil {
			return err
		}
		addConstantResidualToPredReconBlock(reconPlane, pred, predStride, stride, px, py, w, h, sample)
		return nil
	}

	dq := &st.dqScratch
	eob := int(txbResult.EOB)
	quantize.DequantizeBlockScaledBitDepthEOBTrusted(dq[:geo.coeffCount], geo.coeffHeight, qcoeff, geo.coeffHeight, scan, eob, geo.coeffWidth, geo.coeffHeight, q, geo.txScale, 8)
	if txType != transform.TypeIDTX {
		raw := st.invScratch[:n]
		// Every coefficient past the EOB is zero, so the row pass only needs
		// the transform rows the scan prefix reaches — the decoder's own
		// activeRows bound (reconstruct.activeTransformRowsFromScan).
		if err := transform.InverseBlockBitDepthRaw(raw, dq[:geo.coeffCount], geo.coeffHeight, geo.size, txType, 8, scanActiveRowsOriginal(scan, eob, geo.coeffHeight)); err != nil {
			return err
		}
		copyPredToReconBlock(reconPlane, pred, predStride, stride, px, py, w, h)
		dst := reconPlane[py*stride+px:]
		dsp.AddRawTransformPlaneBlockTrusted(dst, stride, 1, 0xff, w, h, raw, w)
		return nil
	}

	res := &st.invResidual
	if err := transform.InverseBlock(res[:n], w, dq[:geo.coeffCount], geo.coeffHeight, st.invScratch[:n], geo.size, txType); err != nil {
		return err
	}
	for r := range h {
		row := (py+r)*stride + px
		for c := range w {
			v := int(pred[r*predStride+c]) + int(res[r*w+c])
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			reconPlane[row+c] = uint8(v)
		}
	}
	return nil

}
func scanActiveRowsOriginal(scan []int16, eob, coeffHeight int) int {
	if eob > len(scan) {
		return 0
	}
	mask := coeffHeight - 1
	rows := 0
	for _, p := range scan[:eob] {
		if r := int(p) & mask; r >= rows {
			rows = r + 1
			if rows == coeffHeight {
				break
			}
		}
	}
	return rows
}
