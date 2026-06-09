package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// TestWriteIntraTransformTypeRoundTrip is the oracle gate for the intra
// tx_type writer: random allowed tx types across the codable transform sizes,
// intra modes, and both tx-set flavors, coded with adapting CDFs, must decode
// back exactly through ReadIntraTransformType with the decoder CDFs adapting
// in lockstep. No-symbol cases (lossless/qindex-0/single-type sets) are
// interleaved to prove they stay symbol-free on both sides.
func TestWriteIntraTransformTypeRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	sizes := []TransformSize{
		TransformSize4x4, TransformSize8x8, TransformSize16x16, TransformSize32x32,
		TransformSize4x8, TransformSize8x4, TransformSize16x8, TransformSize8x16,
	}

	type rec struct {
		req IntraTransformTypeRequest
		typ transform.Type
	}
	const n = 4000
	recs := make([]rec, 0, n)

	var encCDFs TransformTypeCDFs
	if err := encCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<15))

	for range n {
		size := sizes[rng.Intn(len(sizes))]
		mode := IntraMode(rng.Intn(int(intraModeCount)))
		reduced := rng.Intn(2) == 0
		req := IntraTransformTypeRequest{
			Size:         size,
			Mode:         mode,
			ReducedTXSet: reduced,
			QIndexKnown:  true,
			QIndex:       uint8(1 + rng.Intn(254)),
		}
		var typ transform.Type
		switch rng.Intn(8) {
		case 0: // lossless: no symbol
			req.Lossless = true
			typ = transform.TypeDCTDCT
		case 1: // qindex 0: no symbol
			req.QIndex = 0
			typ = transform.TypeDCTDCT
		default:
			set, err := ExtTXSetTypeFor(size, false, reduced)
			if err != nil {
				t.Fatal(err)
			}
			count, err := ExtTXTypeCount(set)
			if err != nil {
				t.Fatal(err)
			}
			symbol := rng.Intn(count)
			typ, err = ExtTXTypeFromSymbol(set, symbol)
			if err != nil {
				t.Fatal(err)
			}
		}
		if err := WriteIntraTransformType(&w, &encCDFs, req, typ); err != nil {
			t.Fatalf("write size=%v mode=%d reduced=%v typ=%v: %v", size, mode, reduced, typ, err)
		}
		recs = append(recs, rec{req: req, typ: typ})
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decCDFs TransformTypeCDFs
	if err := decCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	for i, r := range recs {
		got, err := state.ReadIntraTransformType(&decCDFs, r.req)
		if err != nil {
			t.Fatalf("rec %d read: %v", i, err)
		}
		if got != r.typ {
			t.Fatalf("rec %d (size=%v mode=%d reduced=%v): got %v want %v",
				i, r.req.Size, r.req.Mode, r.req.ReducedTXSet, got, r.typ)
		}
	}
}
