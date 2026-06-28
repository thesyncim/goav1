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
	if err := enc.Prewarm(); err != nil {
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
	b.Cleanup(func() {
		_ = enc.Close()
	})
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	if _, _, err := enc.Encode(frames[0], false); err != nil {
		b.Fatal(err)
	}
	if err := enc.Flush(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := enc.Encode(frames[1+i%7], false); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncodeKeyframe1080p measures full-HD keyframe latency on a reusable
// stream encoder - the worst per-frame spike a realtime stream pays - through
// the tiled intra path.
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
	enc, err := encoder.NewVideoEncoder(w, h, 80)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		_ = enc.Close()
	})
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	if _, key, err := enc.Encode(f, true); err != nil {
		b.Fatal(err)
	} else if !key {
		b.Fatal("prewarm keyframe was not coded as keyframe")
	}
	if err := enc.Flush(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, key, err := enc.Encode(f, true); err != nil {
			b.Fatal(err)
		} else if !key {
			b.Fatal("forced frame was not coded as keyframe")
		}
	}
}

// BenchmarkEncodeKeyframeCold measures the full-HD one-shot convenience helper.
// It intentionally includes returned temporal-unit/reconstruction allocation
// and is not a realtime/WebRTC hot-path benchmark.
func BenchmarkEncodeKeyframeCold(b *testing.B) {
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

// BenchmarkEncodeKeyframeReusable1080p measures repeated full-HD keyframes
// through the reusable keyframe-only state. This is the zero-allocation
// alternative to the owned-output one-shot helper above.
func BenchmarkEncodeKeyframeReusable1080p(b *testing.B) {
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
	enc, err := encoder.NewKeyframeEncoder(w, h, 80)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		_ = enc.Close()
	})
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	if tu, recon, err := enc.Encode(f); err != nil {
		b.Fatal(err)
	} else if len(tu) == 0 || len(recon.Y) == 0 {
		b.Fatal("empty reusable keyframe output")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if tu, recon, err := enc.Encode(f); err != nil {
			b.Fatal(err)
		} else if len(tu) == 0 || len(recon.Y) == 0 {
			b.Fatal("empty reusable keyframe output")
		}
	}
}

// BenchmarkVideoEncoderPFramePan1080p measures the steady P-frame cost on
// camera-like content (box-blurred texture under a continuous pan with
// movers, the cmd/encbench scene shape) - the realtime budget meter. The
// noise benchmark above stays as the worst-case bound.
func BenchmarkVideoEncoderPFramePan1080p(b *testing.B) {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(15))
	wide := make([]byte, (w+512)*h)
	for y := range h {
		for x := 0; x < w+512; x++ {
			wide[y*(w+512)+x] = uint8(60 + (x/7+y/9)%70 + rng.Intn(25))
		}
	}
	// Cheap separable box blur to take the per-pixel noise down to camera
	// texture levels.
	for y := range h {
		row := wide[y*(w+512) : (y+1)*(w+512)]
		for x := 1; x < len(row)-1; x++ {
			row[x] = uint8((int(row[x-1]) + 2*int(row[x]) + int(row[x+1])) >> 2)
		}
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
		off := (t * 4) % 512
		for y := range h {
			copy(f.Y[y*w:(y+1)*w], wide[y*(w+512)+off:])
		}
		for i := range f.U {
			f.U[i] = 120
			f.V[i] = 130
		}
		return f
	}
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 8_000_000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		_ = enc.Close()
	})
	frames := make([]encoder.SourceFrame420, 32)
	for i := range frames {
		frames[i] = makeFrame(i)
	}
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	if _, _, err := enc.Encode(frames[0], true); err != nil {
		b.Fatal(err)
	}
	if err := enc.Flush(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := enc.Encode(frames[1+i%31], false); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStreamingKeyframe1080p measures a forced keyframe inside a live
// stream - the scene-cut path - where the coder pool and reconstruction
// buffer reuse keep the per-key allocation near zero.
func BenchmarkStreamingKeyframe1080p(b *testing.B) {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(15))
	src := encoder.SourceFrame420{Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch), YStride: w, ChromaStride: cw, Width: w, Height: h}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.Y[y*w+x] = uint8(60 + (x/7+y/9)%70 + rng.Intn(25))
		}
	}
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{TargetBitsPerSecond: 8_000_000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		_ = enc.Close()
	})
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := enc.Encode(src, true); err != nil {
			b.Fatal(err)
		}
	}
}
