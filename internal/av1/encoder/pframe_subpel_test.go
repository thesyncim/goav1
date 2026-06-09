package encoder_test

import (
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	avframe "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// TestEncodePFrameSubpelMotion is the subpel gate: the second frame is the
// first resampled at a half-pixel offset, so only subpel motion vectors can
// predict it well. The decode must match the encoder reconstruction
// bit-for-bit (the encoder predicts through the decoder's own convolve), the
// P-frame must compress far below the keyframe (the refinement found the
// half-pel motion), and aomdec must agree when present.
func TestEncodePFrameSubpelMotion(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(31))
	src1 := encoder.SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for y := range h {
		for x := range w {
			src1.Y[y*w+x] = uint8(50 + rng.Intn(150)) // aperiodic texture
		}
	}
	for i := range src1.U {
		src1.U[i] = 120
		src1.V[i] = 130
	}
	// Frame 2: frame 1 shifted by half a pixel horizontally USING THE CODEC'S
	// OWN 8-TAP HALF-PEL FILTER, so a half-pel motion vector can predict it
	// exactly — full-pel-only search cannot.
	src2 := src1
	src2.Y = make([]byte, w*h)
	dstP := avframe.Plane{Pix: src2.Y, Stride: w, Width: w, Height: h}
	refP := avframe.Plane{Pix: src1.Y, Stride: w, Width: w, Height: h}
	halfPel := motion.Vector{Col: 4}
	for py := 0; py < h; py += 8 {
		for px := 0; px < w; px += 8 {
			refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, halfPel, false, false)
			if err != nil {
				t.Fatal(err)
			}
			if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstP, refP, 1, 8, px, py, refX, refY, 8, 8, subX, subY, motion.InterpFilters{}); err != nil {
				t.Fatal(err)
			}
		}
	}
	src2.U = append([]byte(nil), src1.U...)
	src2.V = append([]byte(nil), src1.V...)

	const qIndex = 60
	keyTU, keyRecon, err := encoder.EncodeKeyframe(src1, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	pTU, pRecon, err := encoder.EncodePFrame(src2, keyRecon, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("key %d bytes, half-pel P %d bytes", len(keyTU), len(pTU))
	if len(pTU)*6 >= len(keyTU) {
		t.Fatalf("P %d bytes not well below key %d: subpel motion not engaged?", len(pTU), len(keyTU))
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
	t.Logf("half-pel P luma PSNR %.2f dB", psnr)
	if psnr < 33 {
		t.Fatalf("PSNR %.2f below floor", psnr)
	}

	if aomdec, err := exec.LookPath("aomdec"); err == nil {
		stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 2)
		stream = ivf.AppendFrame(stream, keyTU, 0)
		stream = ivf.AppendFrame(stream, pTU, 1)
		dir := t.TempDir()
		p := filepath.Join(dir, "s.ivf")
		os.WriteFile(p, stream, 0o644)
		outPath := filepath.Join(dir, "o.yuv")
		if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
			t.Fatalf("aomdec: %v\n%s", err, out)
		}
		got, _ := os.ReadFile(outPath)
		want := append(append(append([]byte(nil), keyRecon.Y...), keyRecon.U...), keyRecon.V...)
		want = append(append(append(want, pRecon.Y...), pRecon.U...), pRecon.V...)
		if string(got) != string(want) {
			t.Fatal("aomdec output differs from reconstruction")
		}
		t.Log("aomdec: half-pel stream bit-exact")
	}
}
