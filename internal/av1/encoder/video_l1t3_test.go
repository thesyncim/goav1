package encoder_test

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestVideoEncoderL1T3LayerDrops proves the three-layer temporal structure:
// the T0/T2/T1/T2 group pattern, slot 2 carrying the middle layer for its
// trailing T2, and - the WebRTC property the layers exist for - that
// dropping the T2 leaves, or both T1 and T2, still decodes bit-exact
// against the encoder reconstructions of the kept frames.
func TestVideoEncoderL1T3LayerDrops(t *testing.T) {
	const w, h = 192, 128
	rng := rand.New(rand.NewSource(31))
	bg := make([]byte, w*h)
	for y := range h {
		for x := range w {
			bg[y*w+x] = uint8(60 + (x/4+y/4)%64 + rng.Intn(50))
		}
	}
	makeFrame := func(idx int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: append([]byte(nil), bg...), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		// A small mover so every frame codes real residual.
		sx, sy := 8+idx*6, 16+idx*4
		for y := sy; y < sy+16 && y < h; y++ {
			for x := sx; x < sx+16 && x < w; x++ {
				f.Y[y*w+x] = 230
			}
		}
		for i := range f.U {
			f.U[i], f.V[i] = 116, 132
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 110)
	if err != nil {
		t.Fatal(err)
	}
	if err := enc.SetTemporalLayers(3); err != nil {
		t.Fatal(err)
	}
	const frames = 9
	var tus [][]byte
	var recons []encoder.SourceFrame420
	var tids []uint8
	for i := range frames {
		tid := enc.TemporalID()
		tu, key, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if key {
			tid = 0
		}
		tids = append(tids, tid)
		tus = append(tus, append([]byte(nil), tu...))
		rc := enc.Recon()
		recons = append(recons, encoder.SourceFrame420{
			Y: append([]byte(nil), rc.Y...), U: append([]byte(nil), rc.U...), V: append([]byte(nil), rc.V...),
			YStride: rc.YStride, ChromaStride: rc.ChromaStride, Width: w, Height: h,
		})
	}
	wantTIDs := []uint8{0, 2, 1, 2, 0, 2, 1, 2, 0}
	for i, tid := range tids {
		if tid != wantTIDs[i] {
			t.Fatalf("frame %d temporal id %d, want %d (got %v)", i, tid, wantTIDs[i], tids)
		}
	}

	decodeSubset := func(name string, keep func(tid uint8) bool) {
		var subTUs [][]byte
		var subRecons []encoder.SourceFrame420
		for i := range tus {
			if keep(tids[i]) {
				subTUs = append(subTUs, tus[i])
				subRecons = append(subRecons, recons[i])
			}
		}
		dec, err := goav1.NewDecoder(subTUs)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		defer dec.Close()
		i := 0
		for {
			batch, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("%s decode: %v", name, err)
			}
			if !ok {
				break
			}
			for _, f := range batch {
				for y := range h {
					for x := range w {
						if f.Y.Pix[y*f.Y.Stride+x] != subRecons[i].Y[y*w+x] {
							t.Fatalf("%s frame %d differs at (%d,%d)", name, i, x, y)
						}
					}
				}
				i++
			}
		}
		if i != len(subTUs) {
			t.Fatalf("%s decoded %d frames, want %d", name, i, len(subTUs))
		}
	}
	decodeSubset("all-layers", func(uint8) bool { return true })
	decodeSubset("drop-T2", func(tid uint8) bool { return tid < 2 })
	decodeSubset("T0-only", func(tid uint8) bool { return tid == 0 })
}
