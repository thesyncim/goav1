package encoder_test

import (
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestEncodePFrameVariableBlocks is the variable-block-size gate: global
// motion plus textured content makes the partition decider merge most areas
// into 16x16 blocks while the moving square edge splits to 8x8, and the
// decode must match the encoder reconstruction bit for bit (16x16 luma TXBs,
// 8x8 chroma TXBs, merged-MV ref stacks, cross-SB diagonal context).
func TestEncodePFrameVariableBlocks(t *testing.T) {
	const w, h = 128, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(9))
	src1 := encoder.SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	for i := range src1.Y {
		src1.Y[i] = uint8(80 + rng.Intn(40))
	}
	for i := range src1.U {
		src1.U[i] = 120
		src1.V[i] = 130
	}
	// Frame 2: global +2px motion with low-amplitude noise (16x16 merges with
	// real residual) plus a moving bright square (8x8 splits at its edges).
	src2 := src1
	src2.Y = make([]byte, w*h)
	for y := range h {
		for x := range w {
			sx, sy := max(0, x-2), max(0, y-2)
			src2.Y[y*w+x] = uint8(int(src1.Y[sy*w+sx]) + rng.Intn(13) - 6)
		}
	}
	for y := 60; y < 76; y++ {
		for x := 60; x < 76; x++ {
			src2.Y[y*w+x] = 230
		}
	}
	const qIndex = 60
	keyTU, keyRecon, err := encoder.EncodeKeyframe(src1, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	pTU, pRecon, err := encoder.EncodePFrame(src2, keyRecon, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	f := frames[1]
	comparePlane(t, "Y", f.Y, pRecon.Y, w, h, w)
	comparePlane(t, "U", f.U, pRecon.U, cw, ch, cw)
	comparePlane(t, "V", f.V, pRecon.V, cw, ch, cw)
	psnr := planePSNR(src2.Y, pRecon.Y)
	t.Logf("variable-block P: %d bytes, luma PSNR %.2f dB", len(pTU), psnr)
	if psnr < 30 {
		t.Fatalf("PSNR %.2f below floor", psnr)
	}

	// Second scenario: a clean global (-4,-4) full-pel shift of the keyframe
	// reconstruction itself, so merged blocks predict exactly. Every 16x16
	// child finds the same vector, the 32x32 tier merges, and the P frame
	// collapses to a handful of merged skip blocks - far below the keyframe.
	src3 := src1
	src3.Y = make([]byte, w*h)
	for y := range h {
		for x := range w {
			src3.Y[y*w+x] = keyRecon.Y[max(0, y-4)*w+max(0, x-4)]
		}
	}
	src3.U = make([]byte, cw*ch)
	src3.V = make([]byte, cw*ch)
	for y := range ch {
		for x := range cw {
			src3.U[y*cw+x] = keyRecon.U[max(0, y-2)*cw+max(0, x-2)]
			src3.V[y*cw+x] = keyRecon.V[max(0, y-2)*cw+max(0, x-2)]
		}
	}
	gTU, gRecon, err := encoder.EncodePFrame(src3, keyRecon, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("global-shift P: %d bytes vs key %d", len(gTU), len(keyTU))
	// The variance-partitioned inter frame still collapses the clean global
	// shift well below the keyframe while preserving the clamped edge stripes.
	if len(gTU)*4 >= len(keyTU) {
		t.Fatalf("global-shift P %d bytes not far below key %d", len(gTU), len(keyTU))
	}
	dec2, err := goav1.NewDecoder([][]byte{keyTU, gTU})
	if err != nil {
		t.Fatal(err)
	}
	defer dec2.Close()
	gframes, err := dec2.DecodeAll()
	if err != nil {
		t.Fatalf("decode global-shift: %v", err)
	}
	comparePlane(t, "gY", gframes[1].Y, gRecon.Y, w, h, w)
	comparePlane(t, "gU", gframes[1].U, gRecon.U, cw, ch, cw)
	comparePlane(t, "gV", gframes[1].V, gRecon.V, cw, ch, cw)
}
