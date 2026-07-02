package encoder_test

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	"github.com/thesyncim/goav1/internal/av1/ivf"
)

// TestVideoEncoderSwitchableInterpFilters is the round-trip gate for the
// base-frame interpolation filter search: a textured pan at a fractional-pel
// velocity forces subpel vectors, the filter search must actually code
// non-REGULAR filters on some blocks, and the decode (goav1, plus aomdec when
// installed) must match the encoder reconstruction bit for bit.
func TestVideoEncoderSwitchableInterpFilters(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2

	// A wide textured band mixing smooth gradients with hard edges, panned
	// at 0.5 px/frame horizontally and 0.25 px/frame vertically via
	// supersampled resampling so every P frame needs subpel motion.
	const ss = 4
	tw, th := (w+32)*ss, (h+32)*ss
	texture := make([]byte, tw*th)
	rng := rand.New(rand.NewSource(41))
	for y := 0; y < th; y++ {
		for x := 0; x < tw; x++ {
			v := 100 + 60*math.Sin(float64(x)/(17*ss))*math.Cos(float64(y)/(11*ss))
			if (x/(8*ss)+y/(8*ss))%7 == 0 {
				v += 70 // hard-edged tiles for the sharp side of the search
			}
			v += float64(rng.Intn(7))
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			texture[y*tw+x] = uint8(v)
		}
	}
	makeFrame := func(i int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		offX := 16*ss + i*ss/2 // 0.5 px per frame
		offY := 16*ss + i*ss/4 // 0.25 px per frame
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				f.Y[y*w+x] = texture[(y*ss+offY)*tw+x*ss+offX]
			}
		}
		for i := range f.U {
			f.U[i] = 118
			f.V[i] = 132
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 60)
	if err != nil {
		t.Fatal(err)
	}
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}
	enc.SetDecisionStatsEnabled(true)

	const frames = 10
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	for i := range frames {
		tu, _, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		tus = append(tus, append([]byte(nil), tu...))
		recons = append(recons, cloneFrame(enc.Recon()))
	}

	stats := enc.DecisionStats()
	total := stats.InterpFilters[0] + stats.InterpFilters[1] + stats.InterpFilters[2]
	if total == 0 {
		t.Fatal("no switchable interpolation filters coded; base frames should signal SWITCHABLE")
	}
	if stats.InterpFilters[1]+stats.InterpFilters[2] == 0 {
		t.Fatalf("filter search never chose a non-REGULAR filter (counts %v); the round trip must cover SMOOTH/SHARP", stats.InterpFilters)
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
		dir := t.TempDir()
		ivfData := ivf.AppendFileHeader(nil, w, h, 30, 1, uint32(len(tus)))
		var wantYUV []byte
		for i, tu := range tus {
			ivfData = ivf.AppendFrame(ivfData, tu, uint64(i))
			wantYUV = append(wantYUV, recons[i].Y...)
			wantYUV = append(wantYUV, recons[i].U...)
			wantYUV = append(wantYUV, recons[i].V...)
		}
		p := filepath.Join(dir, "interp.ivf")
		if err := os.WriteFile(p, ivfData, 0o644); err != nil {
			t.Fatal(err)
		}
		outPath := filepath.Join(dir, "interp.yuv")
		if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
			t.Fatalf("aomdec: %v\n%s", err, out)
		}
		got, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(wantYUV) {
			t.Fatalf("aomdec output %d bytes, want %d", len(got), len(wantYUV))
		}
		for i := range got {
			if got[i] != wantYUV[i] {
				t.Fatalf("aomdec output differs from encoder recon at byte %d", i)
			}
		}
	}
}
