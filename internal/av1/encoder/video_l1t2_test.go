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

// TestVideoEncoderL1T2DroppableLayer is the temporal-scalability gate: under
// the L1T2 pattern the full stream must decode with every frame bit-identical
// to its reconstruction, AND the layer-0 substream (all odd frames dropped)
// must decode to exactly the layer-0 reconstructions — proving the T1 frames
// are truly droppable (they refresh no reference state). When aomdec is on
// PATH both streams are verified by the reference decoder too.
func TestVideoEncoderL1T2DroppableLayer(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(17))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(70 + rng.Intn(50))
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
			f.U[i] = 121
			f.V[i] = 129
		}
		sx, sy := 8+t*4, 12+t*2
		for y := sy; y < sy+22 && y < h; y++ {
			for x := sx; x < sx+22 && x < w; x++ {
				f.Y[y*w+x] = 218
			}
		}
		return f
	}

	enc, err := encoder.NewVideoEncoder(w, h, 70)
	if err != nil {
		t.Fatal(err)
	}
	if err := enc.SetTemporalLayers(2); err != nil {
		t.Fatal(err)
	}

	const frames = 9
	tus := make([][]byte, 0, frames)
	recons := make([]encoder.SourceFrame420, 0, frames)
	tids := make([]uint8, 0, frames)
	for i := range frames {
		tid := enc.TemporalID()
		tu, isKey, err := enc.Encode(makeFrame(i), false)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		if (i == 0) != isKey {
			t.Fatalf("frame %d key=%v", i, isKey)
		}
		wantTID := uint8(0)
		if i%2 == 1 {
			wantTID = 1
		}
		if tid != wantTID {
			t.Fatalf("frame %d tid=%d want %d", i, tid, wantTID)
		}
		tus = append(tus, tu)
		recons = append(recons, cloneFrame(enc.Recon()))
		tids = append(tids, tid)
	}

	decodeAndCompare := func(name string, stream [][]byte, want []encoder.SourceFrame420) {
		dec, err := goav1.NewDecoder(stream)
		if err != nil {
			t.Fatal(err)
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
				comparePlane(t, fmt.Sprintf("%s frame%d Y", name, i), f.Y, want[i].Y, w, h, w)
				comparePlane(t, fmt.Sprintf("%s frame%d U", name, i), f.U, want[i].U, cw, ch, cw)
				comparePlane(t, fmt.Sprintf("%s frame%d V", name, i), f.V, want[i].V, cw, ch, cw)
				i++
			}
		}
		if i != len(want) {
			t.Fatalf("%s decoded %d frames, want %d", name, i, len(want))
		}
	}

	decodeAndCompare("full", tus, recons)

	// Layer-0 substream: drop every T1 frame.
	var t0TUs [][]byte
	var t0Recons []encoder.SourceFrame420
	for i := range frames {
		if tids[i] == 0 {
			t0TUs = append(t0TUs, tus[i])
			t0Recons = append(t0Recons, recons[i])
		}
	}
	decodeAndCompare("layer0", t0TUs, t0Recons)

	if aomdec, err := exec.LookPath("aomdec"); err == nil {
		dir := t.TempDir()
		runRef := func(name string, stream [][]byte, want []encoder.SourceFrame420) {
			ivfData := ivf.AppendFileHeader(nil, w, h, 30, 1, uint32(len(stream)))
			var wantYUV []byte
			for i, tu := range stream {
				ivfData = ivf.AppendFrame(ivfData, tu, uint64(i))
				wantYUV = append(wantYUV, want[i].Y...)
				wantYUV = append(wantYUV, want[i].U...)
				wantYUV = append(wantYUV, want[i].V...)
			}
			p := filepath.Join(dir, name+".ivf")
			os.WriteFile(p, ivfData, 0o644)
			outPath := filepath.Join(dir, name+".yuv")
			if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
				t.Fatalf("aomdec %s: %v\n%s", name, err, out)
			}
			got, err := os.ReadFile(outPath)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(wantYUV) {
				t.Fatalf("aomdec %s output differs from reconstruction", name)
			}
			t.Logf("aomdec %s: %d frames bit-exact", name, len(stream))
		}
		runRef("full", tus, recons)
		runRef("layer0", t0TUs, t0Recons)
	}
}
