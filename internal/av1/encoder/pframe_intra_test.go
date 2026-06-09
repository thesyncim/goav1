package encoder_test

import (
	"math/rand"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// TestEncodePFrameSceneChange is the intra-in-P gate: frame 2 cuts to an
// unrelated smooth scene, so motion search fails everywhere and the encoder
// must fall back to DC intra blocks inside the inter frame. Frame 3 then pans
// the new scene, exercising the ref-MV and mode contexts that intra
// neighbors leave behind. Both must decode bit-exact against the encoder
// reconstruction.
func TestEncodePFrameSceneChange(t *testing.T) {
	const w, h = 128, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(11))
	mk := func() encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		return f
	}
	// Scene A: dense noise.
	src1 := mk()
	for i := range src1.Y {
		src1.Y[i] = uint8(60 + rng.Intn(160))
	}
	// Scene B: smooth diagonal gradient - uncorrelated with A, trivially DC
	// predictable, so the intra fallback wins decisively.
	src2 := mk()
	for y := range h {
		for x := range w {
			src2.Y[y*w+x] = uint8(40 + (x+y)/2)
		}
	}
	// Frame 3: scene B panned by a full-pel (-4,-4).
	src3 := mk()
	for y := range h {
		for x := range w {
			src3.Y[y*w+x] = src2.Y[max(0, y-4)*w+max(0, x-4)]
		}
	}

	const qIndex = 60
	keyTU, keyRecon, err := encoder.EncodeKeyframe(src1, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	cutTU, cutRecon, err := encoder.EncodePFrame(src2, keyRecon, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	panTU, panRecon, err := encoder.EncodePFrame(src3, cutRecon, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("key %dB, scene-cut P %dB, pan P %dB", len(keyTU), len(cutTU), len(panTU))
	psnr := planePSNR(src2.Y, cutRecon.Y)
	t.Logf("scene-cut luma PSNR %.2f dB", psnr)
	if psnr < 32 {
		t.Fatalf("scene-cut PSNR %.2f below floor: intra fallback not effective", psnr)
	}
	// The smooth scene is far cheaper as intra than as motion residual from
	// uncorrelated noise; well under half the keyframe cost of scene A.
	if len(cutTU)*2 >= len(keyTU) {
		t.Fatalf("scene-cut P %dB not well below key %dB: intra fallback not engaged", len(cutTU), len(keyTU))
	}

	dec, err := goav1.NewDecoder([][]byte{keyTU, cutTU, panTU})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	for i, recon := range []encoder.SourceFrame420{cutRecon, panRecon} {
		f := frames[i+1]
		comparePlane(t, "Y", f.Y, recon.Y, w, h, w)
		comparePlane(t, "U", f.U, recon.U, cw, ch, cw)
		comparePlane(t, "V", f.V, recon.V, cw, ch, cw)
	}
}
