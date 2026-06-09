package encoder_test

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// BenchmarkVideoEncoderPFrame measures steady-state inter-frame encoding of a
// moving 384x256 scene — the realtime hot path (motion search, skip decisions,
// transforms, entropy coding) plus any per-frame setup cost.
func BenchmarkVideoEncoderPFrame(b *testing.B) {
	const w, h = 640, 360
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(3))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(60 + rng.Intn(60))
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
			f.V[i] = 130
		}
		sx, sy := (t*4)%(w-32), (t*2)%(h-32)
		for y := sy; y < sy+32; y++ {
			for x := sx; x < sx+32; x++ {
				f.Y[y*w+x] = 220
			}
		}
		return f
	}
	frames := make([]encoder.SourceFrame420, 8)
	for i := range frames {
		frames[i] = makeFrame(i)
	}
	enc, err := encoder.NewVideoEncoder(w, h, 60)
	if err != nil {
		b.Fatal(err)
	}
	if _, _, err := enc.Encode(frames[0], false); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		if _, _, err := enc.Encode(frames[1+i%7], false); err != nil {
			b.Fatal(err)
		}
	}
}
