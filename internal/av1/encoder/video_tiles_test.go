package encoder_test

import (
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/ivf"
)

// TestVideoEncoderTileColumns is the tile-parallel gate: a width that selects
// four tile columns must produce streams whose every frame decodes bit-exact
// against the encoder reconstruction (per-tile entropy coders, CDF resets,
// tile-relative neighbor availability), cross-checked with aomdec when
// available.
func TestVideoEncoderTileColumns(t *testing.T) {
	const w, h = 1280, 192
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(21))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(60 + rng.Intn(80))
	}
	makeFrame := func(n int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: append([]byte(nil), bg...), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		sx, sy := 40+n*8, 30+n*6
		for y := sy; y < sy+40 && y < h; y++ {
			for x := sx; x < sx+40 && x < w; x++ {
				f.Y[y*w+x] = 225
			}
		}
		return f
	}
	enc, err := encoder.NewVideoEncoder(w, h, 70)
	if err != nil {
		t.Fatal(err)
	}
	const frames = 5
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	for i := range frames {
		tu, _, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		tus = append(tus, tu)
		recons = append(recons, cloneFrame(enc.Recon()))
	}
	dec, err := goav1.NewDecoder(tus)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	i := 0
	for {
		batch, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("decode frame %d: %v", i, err)
		}
		if !ok {
			break
		}
		for _, f := range batch {
			comparePlane(t, "Y", f.Y, recons[i].Y, w, h, w)
			comparePlane(t, "U", f.U, recons[i].U, cw, ch, cw)
			comparePlane(t, "V", f.V, recons[i].V, cw, ch, cw)
			i++
		}
	}
	if i != frames {
		t.Fatalf("decoded %d frames, want %d", i, frames)
	}

	if aomdec, err := exec.LookPath("aomdec"); err == nil {
		stream := ivf.AppendFileHeader(nil, w, h, 30, 1, uint32(frames))
		for n, tu := range tus {
			stream = ivf.AppendFrame(stream, tu, uint64(n))
		}
		dir := t.TempDir()
		p := filepath.Join(dir, "s.ivf")
		os.WriteFile(p, stream, 0o644)
		outPath := filepath.Join(dir, "o.yuv")
		if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
			t.Fatalf("aomdec: %v\n%s", err, out)
		}
		got, _ := os.ReadFile(outPath)
		var want []byte
		for _, r := range recons {
			want = append(want, r.Y...)
			want = append(want, r.U...)
			want = append(want, r.V...)
		}
		if string(got) != string(want) {
			t.Fatal("aomdec output differs from reconstruction")
		}
		t.Log("aomdec: 4-tile stream bit-exact")
	}
}
