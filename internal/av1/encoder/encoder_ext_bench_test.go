package encoder_test

import (
	"github.com/thesyncim/goav1/internal/av1/encoder"
	"math/rand"
	"os"
	"testing"
)

// benchCorpus measures steady-state back-to-back encoding of one local corpus
// segment. It is a goav1 hot-path profiler, not a publishable external
// comparison; use cmd/qualitybench -publish for SVT-AV1/libaom tables.
func benchCorpus(b *testing.B, path string, fps, bitrate int) {
	raw, err := os.ReadFile(path)
	if err != nil {
		b.Skip(err)
	}
	const w, h = 1920, 1080
	fl := w * h * 3 / 2
	if len(raw)%fl != 0 {
		b.Fatalf("%s size=%d is not an exact number of 1080p I420 frames", path, len(raw))
	}
	nf := len(raw) / fl
	if nf < 2 {
		b.Fatalf("%s has %d frame(s), want at least 2", path, nf)
	}
	frames := make([]encoder.SourceFrame420, nf)
	for i := 0; i < nf; i++ {
		base := i * fl
		frames[i] = encoder.SourceFrame420{
			Y: raw[base : base+w*h], U: raw[base+w*h : base+w*h+w*h/4], V: raw[base+w*h+w*h/4 : base+fl],
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
	}
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: bitrate, FramesPerSecond: fps, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		b.StopTimer()
		_ = enc.Close()
	})
	if err := enc.SetTemporalLayers(2); err != nil {
		b.Fatal(err)
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
		if _, _, err := enc.Encode(frames[1+i%(nf-1)], false); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkVideoEncoderRealC1080p is the complex-motion segment where search
// and residual coding dominate.
func BenchmarkVideoEncoderRealC1080p(b *testing.B) {
	benchCorpus(b, "/tmp/corpus/realC.yuv", 30, 5_000_000)
}

// BenchmarkVideoEncoderRealA1080p is the moderate camera segment.
func BenchmarkVideoEncoderRealA1080p(b *testing.B) {
	benchCorpus(b, "/tmp/corpus/realA.yuv", 30, 5_000_000)
}

// BenchmarkVideoEncoderScreen1080p is the screen-content segment.
func BenchmarkVideoEncoderScreen1080p(b *testing.B) {
	benchCorpus(b, "/tmp/corpus/screen.yuv", 60, 1_330_000)
}

// BenchmarkVideoEncoderPFrame measures steady-state inter-frame encoding of a
// moving 640x360 scene — the realtime hot path (motion search, skip decisions,
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
	b.Cleanup(func() {
		b.StopTimer()
		_ = enc.Close()
	})
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

// BenchmarkVideoEncoderPFrame1080p is a goav1 hot-path profiling point:
// full-HD steady-state inter encoding with textured content, global drift, and
// two moving objects, exercising motion search, merge decisions, subpel, and
// four parallel tile columns. Use cmd/qualitybench publish mode for external
// SVT-AV1/libaom comparison tables.
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
		b.StopTimer()
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
		b.StopTimer()
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

// BenchmarkEncodeKeyframeCold measures a full-HD keyframe on a freshly built
// encoder each iteration, including its scratch allocation. It is not a
// realtime/WebRTC hot-path benchmark.
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
		e, err := encoder.NewKeyframeEncoder(w, h, 80)
		if err != nil {
			b.Fatal(err)
		}
		if _, _, err := e.Encode(f); err != nil {
			b.Fatal(err)
		}
		if err := e.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncodeKeyframeReusable1080p measures repeated full-HD keyframes
// through the reusable keyframe-only state. This is the zero-allocation
// alternative to the cold encoder above.
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
		b.StopTimer()
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
	benchmarkVideoEncoderPFramePan1080p(b, 0, 0)
}

// BenchmarkVideoEncoderPFramePan1080pSingleThread measures the same camera-like
// 1080p steady-state path with tile, filter, and HME worker fan-out disabled
// for one-lane WebRTC deployments and profiler runs.
func BenchmarkVideoEncoderPFramePan1080pSingleThread(b *testing.B) {
	benchmarkVideoEncoderPFramePan1080p(b, 1, 0)
}

// BenchmarkVideoEncoderPFramePan1080pSingleThreadMinEffort measures the
// fastest valid single-thread camera path exposed through the WebRTC Speed
// control.
func BenchmarkVideoEncoderPFramePan1080pSingleThreadMinEffort(b *testing.B) {
	benchmarkVideoEncoderPFramePan1080p(b, 1, encoder.WebRTCMinEffortLevel)
}

// BenchmarkVideoEncoderPFrameStatic1080pSingleThread measures static-camera
// realtime P-frame cost where SVT-style static 64x64 ME bypass should stay hot.
func BenchmarkVideoEncoderPFrameStatic1080pSingleThread(b *testing.B) {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	f := encoder.SourceFrame420{
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
			f.Y[y*w+x] = uint8(48 + (x/5+y/7)%120)
		}
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 130
	}
	enc, err := encoder.NewVideoEncoderCBR(w, h, encoder.RateControlConfig{
		TargetBitsPerSecond: 8_000_000, FramesPerSecond: 60, MinQIndex: 20, MaxQIndex: 200,
	})
	if err != nil {
		b.Fatal(err)
	}
	enc.SetMaxThreads(1)
	b.Cleanup(func() {
		b.StopTimer()
		_ = enc.Close()
	})
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	if _, _, err := enc.Encode(f, true); err != nil {
		b.Fatal(err)
	}
	if err := enc.Flush(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := enc.Encode(f, false); err != nil {
			b.Fatal(err)
		}
	}
}

func benchmarkVideoEncoderPFramePan1080p(b *testing.B, maxThreads int, effort int8) {
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
	if maxThreads > 0 {
		enc.SetMaxThreads(maxThreads)
	}
	if effort != 0 {
		if err := enc.SetEffortLevel(effort); err != nil {
			b.Fatal(err)
		}
	}
	b.Cleanup(func() {
		b.StopTimer()
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

// BenchmarkVideoEncoderPipelinePan1080p is the throughput-pipelining
// allocation canary: the opt-in EncodeThroughput/Drain surface must stay at
// zero steady-state heap traffic just like the serial Encode path. It codes
// the camera-like pan under L1T2 with pipelining enabled.
func BenchmarkVideoEncoderPipelinePan1080p(b *testing.B) {
	const w, h = 1920, 1080
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(15))
	wide := make([]byte, (w+512)*h)
	for y := range h {
		for x := 0; x < w+512; x++ {
			wide[y*(w+512)+x] = uint8(60 + (x/7+y/9)%70 + rng.Intn(25))
		}
	}
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
	if err := enc.SetTemporalLayers(2); err != nil {
		b.Fatal(err)
	}
	// Overlap-eligible config: golden off, single-tile inter.
	enc.SetGoldenInterval(0)
	enc.SetMaxThreads(4)
	if err := enc.SetThroughputPipelining(true); err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() {
		b.StopTimer()
		_, _, _, _ = enc.Drain()
		_ = enc.Close()
	})
	frames := make([]encoder.SourceFrame420, 32)
	for i := range frames {
		frames[i] = makeFrame(i)
	}
	if err := enc.Prewarm(); err != nil {
		b.Fatal(err)
	}
	// Fill and settle the pipeline so the leaf coder/HME/filter/wavefront state
	// and pipeline buffers are all sized (through several overlaps) before
	// timing.
	if _, _, _, err := enc.EncodeThroughput(frames[0], true); err != nil {
		b.Fatal(err)
	}
	for i := 1; i < 9; i++ {
		if _, _, _, err := enc.EncodeThroughput(frames[i], false); err != nil {
			b.Fatal(err)
		}
	}
	if err := enc.Flush(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := enc.EncodeThroughput(frames[1+i%31], false); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStreamingKeyframe1080p measures a forced keyframe inside a live
// stream, where the coder pool and reconstruction
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
		b.StopTimer()
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

var webRTCStreamBenchmarkSink int

func BenchmarkWebRTCStreamEncodePictureModes(b *testing.B) {
	modes := []encoder.ScalabilityMode{
		encoder.ScalabilityModeL1T3,
		encoder.ScalabilityModeL2T3,
		encoder.ScalabilityModeL2T3_KEY,
		encoder.ScalabilityModeL2T3_KEY_SHIFT,
		encoder.ScalabilityModeS2T3,
		encoder.ScalabilityModeL3T3,
		encoder.ScalabilityModeL3T3_KEY,
		encoder.ScalabilityModeL3T3_KEY_SHIFT,
		encoder.ScalabilityModeS3T3h,
	}
	for _, mode := range modes {
		b.Run(mode.String(), func(b *testing.B) {
			cfg := webRTCDecodeMatrixConfig(mode)
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				b.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			b.Cleanup(func() {
				b.StopTimer()
				_ = stream.Close()
			})
			frames := make([]encoder.SourceFrame420, 8)
			for i := range frames {
				frames[i] = webRTCDecodeMatrixFrame(int(cfg.Resolution.Width), int(cfg.Resolution.Height), i)
			}
			warmWebRTCStreamBenchmark(b, stream, frames, mode)

			b.SetBytes(webRTCBenchmarkPictureBytes(stream.Config()))
			b.ReportAllocs()
			b.ResetTimer()
			sum := 0
			for i := 0; b.Loop(); i++ {
				picture, err := stream.EncodePicture(frames[1+i%7], false)
				if err != nil {
					b.Fatalf("EncodePicture(%s): %v", mode, err)
				}
				for frame := uint8(0); frame < picture.FrameNum; frame++ {
					sum += len(picture.Frames[frame].TU)
				}
			}
			webRTCStreamBenchmarkSink += sum
		})
	}
}

func BenchmarkWebRTCStreamEncodePicture1080p(b *testing.B) {
	for _, mode := range []encoder.ScalabilityMode{
		encoder.ScalabilityModeL1T3,
		encoder.ScalabilityModeS3T3,
	} {
		b.Run(mode.String(), func(b *testing.B) {
			cfg := webRTC1080pAllocConfig(mode)
			stream, err := encoder.NewWebRTCStreamConfig(cfg)
			if err != nil {
				b.Fatalf("NewWebRTCStreamConfig(%s): %v", mode, err)
			}
			b.Cleanup(func() {
				b.StopTimer()
				_ = stream.Close()
			})
			frames := []encoder.SourceFrame420{
				makeEncoder1080pFrame(0),
				makeEncoder1080pFrame(1),
				makeEncoder1080pFrame(2),
				makeEncoder1080pFrame(3),
			}
			warmWebRTCStreamBenchmark(b, stream, frames, mode)

			b.SetBytes(webRTCBenchmarkPictureBytes(stream.Config()))
			b.ReportAllocs()
			b.ResetTimer()
			sum := 0
			for i := 0; i < b.N; i++ {
				picture, err := stream.EncodePicture(frames[i&3], false)
				if err != nil {
					b.Fatalf("EncodePicture(%s): %v", mode, err)
				}
				for frame := uint8(0); frame < picture.FrameNum; frame++ {
					sum += len(picture.Frames[frame].TU)
				}
			}
			webRTCStreamBenchmarkSink += sum
		})
	}
}

func warmWebRTCStreamBenchmark(b *testing.B, stream *encoder.WebRTCStream, frames []encoder.SourceFrame420, mode encoder.ScalabilityMode) {
	b.Helper()
	if err := stream.Prewarm(); err != nil {
		b.Fatalf("Prewarm(%s): %v", mode, err)
	}
	for i := 0; i < len(frames)*2; i++ {
		forceKey := i == 0 || i == len(frames)
		picture, err := stream.EncodePicture(frames[i%len(frames)], forceKey)
		if err != nil {
			b.Fatalf("warm EncodePicture(%s, %d): %v", mode, i, err)
		}
		if picture.FrameNum == 0 {
			b.Fatalf("warm EncodePicture(%s, %d) emitted no frames", mode, i)
		}
	}
}

func webRTCBenchmarkPictureBytes(config encoder.Config) int64 {
	var bytes int64
	for i := uint8(0); i < config.SpatialLayerCount; i++ {
		resolution := config.SpatialLayers[i].Resolution
		bytes += int64(resolution.Width) * int64(resolution.Height) * 3 / 2
	}
	return bytes
}
