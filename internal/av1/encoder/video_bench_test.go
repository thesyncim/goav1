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

// BenchmarkVideoEncoderPFrame1080p is the SVT-AV1 comparison point: full-HD
// steady-state inter encoding with textured content, global drift, and two
// moving objects, exercising motion search, merge decisions, subpel, and
// four parallel tile columns.
func BenchmarkVideoEncoderPFrame1080p(b *testing.B) {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(9))
	bg := make([]byte, w*h)
	for i := range bg {
		bg[i] = uint8(50 + rng.Intn(90))
	}
	makeFrame := func(t int) encoder.SourceFrame420 {
		f := encoder.SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, cw*ch),
			V:            make([]byte, cw*ch),
			YStride:      w,
			ChromaStride: cw,
			Width:        w,
			Height:       h,
		}
		// Global pan of the textured background.
		dx := (t * 2) % 16
		for y := range h {
			copy(f.Y[y*w:y*w+w-dx], bg[y*w+dx:y*w+w])
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		for _, obj := range [2][3]int{{200 + t*12, 300, 96}, {1300 - t*9, 700, 64}} {
			ox, oy, n := obj[0], obj[1], obj[2]
			for y := oy; y < oy+n && y < h; y++ {
				for x := ox; x < ox+n && x < w; x++ {
					if x >= 0 {
						f.Y[y*w+x] = 215
					}
				}
			}
		}
		return f
	}
	frames := make([]encoder.SourceFrame420, 8)
	for i := range frames {
		frames[i] = makeFrame(i)
	}
	enc, err := encoder.NewVideoEncoder(w, h, 80)
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

// BenchmarkEncodeKeyframe1080p measures full-HD keyframe latency - the worst
// per-frame spike a realtime stream pays - through the tiled intra path.
func BenchmarkEncodeKeyframe1080p(b *testing.B) {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(2))
	f := encoder.SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	for i := range f.Y {
		f.Y[i] = uint8(50 + rng.Intn(120))
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 130
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := encoder.EncodeKeyframe(f, 80); err != nil {
			b.Fatal(err)
		}
	}
}
