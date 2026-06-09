package encoder_test

import (
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/ivf"
)

// TestVideoEncoderRealResolutions exercises the stream encoder at production
// resolutions that are multiples of 8 but not of 64 (partial superblocks on
// both axes): every frame must decode bit-identical to its reconstruction in
// goav1, and the muxed stream must decode bit-identical in aomdec when
// available.
func TestVideoEncoderRealResolutions(t *testing.T) {
	sizes := []struct{ w, h int }{
		{640, 360}, // nHD: 360 = 5*64 + 40
		{200, 120}, // tiny with partial SBs both axes
	}
	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.w, sz.h), func(t *testing.T) {
			w, h := sz.w, sz.h
			cw, ch := w/2, h/2
			rng := rand.New(rand.NewSource(int64(w)))
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
					f.U[i] = 120
					f.V[i] = 128
				}
				sx, sy := 16+t*6, 24+t*4
				for y := sy; y < sy+28 && y < h; y++ {
					for x := sx; x < sx+28 && x < w; x++ {
						f.Y[y*w+x] = 216
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
					t.Fatalf("decode: %v", err)
				}
				if !ok {
					break
				}
				for _, f := range batch {
					comparePlane(t, fmt.Sprintf("frame%d Y", i), f.Y, recons[i].Y, w, h, w)
					comparePlane(t, fmt.Sprintf("frame%d U", i), f.U, recons[i].U, cw, ch, cw)
					comparePlane(t, fmt.Sprintf("frame%d V", i), f.V, recons[i].V, cw, ch, cw)
					i++
				}
			}
			if i != frames {
				t.Fatalf("decoded %d frames, want %d", i, frames)
			}

			if aomdec, err := exec.LookPath("aomdec"); err == nil {
				stream := ivf.AppendFileHeader(nil, uint16(w), uint16(h), 30, 1, frames)
				var wantYUV []byte
				for j, tu := range tus {
					stream = ivf.AppendFrame(stream, tu, uint64(j))
					wantYUV = append(wantYUV, recons[j].Y...)
					wantYUV = append(wantYUV, recons[j].U...)
					wantYUV = append(wantYUV, recons[j].V...)
				}
				dir := t.TempDir()
				p := filepath.Join(dir, "s.ivf")
				os.WriteFile(p, stream, 0o644)
				outPath := filepath.Join(dir, "o.yuv")
				if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
					t.Fatalf("aomdec: %v\n%s", err, out)
				}
				got, _ := os.ReadFile(outPath)
				if string(got) != string(wantYUV) {
					t.Fatalf("aomdec output differs from reconstruction")
				}
				t.Logf("aomdec: %d frames bit-exact", frames)
			}
		})
	}
}
