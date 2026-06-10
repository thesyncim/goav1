package encoder_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/ivf"
)

// TestKeyframeDirectionalModes is the V/H intra gate: striped content
// predicts almost exactly from the neighboring edge once the first
// row/column of blocks is coded, so the directional modes must compress far
// below the DC-only cost (6.2 KB for the vertical scene before they
// existed), decode bit-exact, and pass aomdec.
func TestKeyframeDirectionalModes(t *testing.T) {
	const w, h = 256, 256
	cw, ch := w/2, h/2
	mk := func(vertical bool) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
			YStride: w, ChromaStride: cw, Width: w, Height: h,
		}
		for y := range h {
			for x := range w {
				p := x
				if !vertical {
					p = y
				}
				v := uint8(60)
				if (p/4)%2 == 1 {
					v = 200
				}
				f.Y[y*w+x] = v
			}
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		return f
	}
	for _, tc := range []struct {
		name     string
		vertical bool
	}{{"vertical-stripes", true}, {"horizontal-stripes", false}} {
		t.Run(tc.name, func(t *testing.T) {
			src := mk(tc.vertical)
			tu, recon, err := encoder.EncodeKeyframe(src, 60)
			if err != nil {
				t.Fatal(err)
			}
			psnr := planePSNR(src.Y, recon.Y)
			t.Logf("%s key: %d bytes, PSNR %.2f", tc.name, len(tu), psnr)
			if len(tu) > 3500 {
				t.Fatalf("key %d bytes: directional intra not engaged (DC-only cost ~6.2KB)", len(tu))
			}
			if psnr < 48 {
				t.Fatalf("PSNR %.2f below floor", psnr)
			}
			dec, err := goav1.NewDecoder([][]byte{tu})
			if err != nil {
				t.Fatal(err)
			}
			defer dec.Close()
			frames, err := dec.DecodeAll()
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			comparePlane(t, "Y", frames[0].Y, recon.Y, w, h, w)
			comparePlane(t, "U", frames[0].U, recon.U, cw, ch, cw)
			comparePlane(t, "V", frames[0].V, recon.V, cw, ch, cw)

			if aomdec, err := exec.LookPath("aomdec"); err == nil {
				stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 1)
				stream = ivf.AppendFrame(stream, tu, 0)
				dir := t.TempDir()
				p := filepath.Join(dir, "s.ivf")
				os.WriteFile(p, stream, 0o644)
				outPath := filepath.Join(dir, "o.yuv")
				if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
					t.Fatalf("aomdec: %v\n%s", err, out)
				}
				got, _ := os.ReadFile(outPath)
				want := append(append(append([]byte(nil), recon.Y...), recon.U...), recon.V...)
				if string(got) != string(want) {
					t.Fatal("aomdec output differs from reconstruction")
				}
			}
		})
	}
}
