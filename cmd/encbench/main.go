// Command encbench measures goav1's realtime encoder on a deterministic
// synthetic 1080p scene (textured global pan plus two movers) and can dump
// the same scene as raw I420 so external encoders run on identical input.
//
//	encbench -dump seq.yuv          write the scene as raw I420
//	encbench -bitrate 6000000       encode with goav1 CBR and report
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	goav1 "github.com/thesyncim/goav1"
)

const (
	width  = 1920
	height = 1080
	frames = 120
	fps    = 60
)

func makeFrame(bg []byte, n int) goav1.I420Frame {
	cw, ch := width/2, height/2
	f := goav1.I420Frame{
		Y: make([]byte, width*height), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: width, ChromaStride: cw, Width: width, Height: height,
	}
	dx := (n * 2) % 16
	for y := range height {
		copy(f.Y[y*width:y*width+width-dx], bg[y*width+dx:y*width+width])
	}
	for i := range f.U {
		f.U[i] = 120
		f.V[i] = 130
	}
	for _, obj := range [2][3]int{{200 + n*12, 300, 96}, {1300 - n*9, 700, 64}} {
		ox, oy, sz := obj[0], obj[1], obj[2]
		for y := oy; y < oy+sz && y < height; y++ {
			for x := ox; x < ox+sz && x < width; x++ {
				if x >= 0 {
					f.Y[y*width+x] = 215
				}
			}
		}
	}
	return f
}

func psnr(a, b []byte) float64 {
	var sse float64
	for i := range a {
		d := float64(int(a[i]) - int(b[i]))
		sse += d * d
	}
	if sse == 0 {
		return 99
	}
	mse := sse / float64(len(a))
	return 10 * math.Log10(255*255/mse)
}

func main() {
	dump := flag.String("dump", "", "write the scene as raw I420 to this path and exit")
	bitrate := flag.Int("bitrate", 6_000_000, "CBR target in bits per second")
	layers := flag.Int("layers", 1, "temporal layers (1 flat, 2 or 3 layered)")
	tiles := flag.Int("tiles", 0, "tile columns override (0 = default)")
	input := flag.String("input", "", "encode this raw 1080p I420 file instead of the synthetic scene")
	frameStats := flag.Bool("framestats", false, "print per-frame size and PSNR")
	nframes := flag.Int("frames", frames, "frames to encode with -input")
	infps := flag.Int("fps", fps, "frame rate for rate control with -input")
	flag.Parse()

	rng := rand.New(rand.NewSource(9))
	bg := make([]byte, width*height)
	for i := range bg {
		bg[i] = uint8(50 + rng.Intn(90))
	}
	// One box-blur pass gives the texture the spatial correlation of natural
	// content; raw sample noise is incompressible for any codec and pins
	// every rate controller at its quantizer ceiling.
	blurred := make([]byte, width*height)
	for y := 1; y < height-1; y++ {
		for x := 1; x < width-1; x++ {
			sum := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					sum += int(bg[(y+dy)*width+x+dx])
				}
			}
			blurred[y*width+x] = uint8(sum / 9)
		}
	}
	copy(bg, blurred)

	if *dump != "" {
		out, err := os.Create(*dump)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer out.Close()
		for n := range frames {
			f := makeFrame(bg, n)
			out.Write(f.Y)
			out.Write(f.U)
			out.Write(f.V)
		}
		fmt.Printf("wrote %d frames of %dx%d I420\n", frames, width, height)
		return
	}

	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: width, Height: height,
		TargetBitrate: *bitrate, Framerate: *infps,
		TemporalLayers: *layers,
		TileColumns:    *tiles,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	nFrames := frames
	if *input != "" {
		nFrames = *nframes
	}
	srcs := make([]goav1.I420Frame, nFrames)
	if *input != "" {
		raw, err := os.ReadFile(*input)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		cw, ch := width/2, height/2
		frameLen := width*height + 2*cw*ch
		if len(raw) < nFrames*frameLen {
			fmt.Fprintf(os.Stderr, "input holds %d frames, need %d\n", len(raw)/frameLen, nFrames)
			os.Exit(1)
		}
		for n := range nFrames {
			base := n * frameLen
			srcs[n] = goav1.I420Frame{
				Y:       raw[base : base+width*height],
				U:       raw[base+width*height : base+width*height+cw*ch],
				V:       raw[base+width*height+cw*ch : base+frameLen],
				YStride: width, ChromaStride: cw, Width: width, Height: height,
			}
		}
	} else {
		for n := range nFrames {
			srcs[n] = makeFrame(bg, n)
		}
	}
	const warmup = 20
	totalBytes, steadyBytes := 0, 0
	var sumPSNR, steadyPSNR float64
	minPSNR := 1e9
	var encodeTime time.Duration
	for n := range nFrames {
		frameStart := time.Now()
		out, err := enc.Encode(srcs[n], false)
		encodeTime += time.Since(frameStart)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		totalBytes += len(out.Data)
		r := enc.Reconstruction()
		p := psnr(srcs[n].Y, r.Y)
		if *frameStats {
			fmt.Printf("frame %3d: %7d bytes  %6.2f dB  q=%d\n", n, len(out.Data), p, enc.QIndex())
		}
		sumPSNR += p
		if n >= warmup && p < minPSNR {
			minPSNR = p
		}
		if n >= warmup {
			steadyBytes += len(out.Data)
			steadyPSNR += p
		}
	}
	// Only the encode calls count: the harness's own PSNR pass costs
	// milliseconds per frame and external encoders do not pay it.
	elapsed := encodeTime
	perFrame := elapsed / time.Duration(nFrames)
	fmt.Printf("goav1: %d frames in %v (%.2f ms/frame, %.1f fps)\n", nFrames, elapsed.Round(time.Millisecond), float64(perFrame.Microseconds())/1000, float64(nFrames)/elapsed.Seconds())
	fmt.Printf("goav1: %.2f Mbps overall / %.2f Mbps steady-state (target %.2f), luma PSNR %.2f dB (steady %.2f, min %.2f), final qindex %d\n",
		float64(totalBytes*8**infps)/float64(nFrames)/1e6,
		float64(steadyBytes*8**infps)/float64(nFrames-warmup)/1e6,
		float64(*bitrate)/1e6, sumPSNR/float64(nFrames), steadyPSNR/float64(nFrames-warmup), minPSNR, enc.QIndex())
}
