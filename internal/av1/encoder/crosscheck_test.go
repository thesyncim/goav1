package encoder_test

import (
	"bytes"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/ivf"
)

// TestEncodedStreamDecodesInReferenceDecoders is the independent oracle: the
// encoder's output must decode in libaom's aomdec (and dav1d when present) to
// exactly the encoder reconstruction. This closes the residual risk of a
// blind spot shared between the goav1 encoder and decoder.
func TestEncodedStreamDecodesInReferenceDecoders(t *testing.T) {
	aomdec, err := exec.LookPath("aomdec")
	if err != nil {
		t.Skip("aomdec not on PATH")
	}
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(11))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(64 + rng.Intn(56))
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            append([]byte(nil), bg...),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		for i := range f.U {
			f.U[i] = 117
			f.V[i] = 135
		}
		sx, sy := 10+t*4, 14+t*2
		for y := sy; y < sy+20 && y < h; y++ {
			for x := sx; x < sx+20 && x < w; x++ {
				f.Y[y*w+x] = 215
			}
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 70)
	if err != nil {
		t.Fatal(err)
	}
	const frames = 6
	stream := ivf.AppendFileHeader(nil, w, h, 30, 1, frames)
	wantYUV := make([]byte, 0, frames*(w*h+2*cw*ch))
	for i := range frames {
		tu, _, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		stream = ivf.AppendFrame(stream, tu, uint64(i))
		r := enc.Recon()
		wantYUV = append(wantYUV, r.Y...)
		wantYUV = append(wantYUV, r.U...)
		wantYUV = append(wantYUV, r.V...)
	}

	dir := t.TempDir()
	ivfPath := filepath.Join(dir, "stream.ivf")
	if err := os.WriteFile(ivfPath, stream, 0o644); err != nil {
		t.Fatal(err)
	}

	check := func(name, bin string, args ...string) {
		outPath := filepath.Join(dir, name+".yuv")
		cmd := exec.Command(bin, append(args, ivfPath)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s failed: %v\n%s", name, err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("read %s output: %v", name, err)
		}
		if len(got) != len(wantYUV) {
			t.Fatalf("%s output %d bytes, want %d", name, len(got), len(wantYUV))
		}
		if !bytes.Equal(got, wantYUV) {
			for i := range got {
				if got[i] != wantYUV[i] {
					t.Fatalf("%s output differs first at byte %d: got %d want %d", name, i, got[i], wantYUV[i])
				}
			}
		}
		t.Logf("%s: %d frames bit-exact", name, frames)
	}

	check("aomdec", aomdec, "--rawvideo", "-o", filepath.Join(dir, "aomdec.yuv"))
	if dav1d, err := exec.LookPath("dav1d"); err == nil {
		check("dav1d", dav1d, "--muxer", "yuv", "-o", filepath.Join(dir, "dav1d.yuv"), "-i")
	}
}
