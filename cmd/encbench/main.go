// Command encbench measures goav1's realtime encoder on a deterministic
// synthetic scene (textured global pan plus two movers) and can dump the same
// scene as raw I420 so external encoders run on identical input.
//
//	encbench -dump seq.yuv          write the scene as raw I420
//	encbench -bitrate 6000000       encode with goav1 CBR and report
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	goav1 "github.com/thesyncim/goav1"
)

const (
	defaultWidth  = 1920
	defaultHeight = 1080
	frames        = 120
	fps           = 60
)

func makeFrame(bg []byte, n, width, height int) goav1.I420Frame {
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
	scaleX := func(v int) int { return v * width / defaultWidth }
	scaleY := func(v int) int { return v * height / defaultHeight }
	max1 := func(v int) int {
		if v < 1 {
			return 1
		}
		return v
	}
	for _, obj := range [2][3]int{
		{scaleX(200) + n*max1(width/160), scaleY(300), max1(scaleX(96))},
		{scaleX(1300) - n*max1(width/213), scaleY(700), max1(scaleX(64))},
	} {
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

func goldenLabel(interval int) string {
	switch {
	case interval < 0:
		return "disabled"
	case interval == 0:
		return "default"
	default:
		return fmt.Sprintf("every %d base frames", interval)
	}
}

func main() {
	dump := flag.String("dump", "", "write the scene as raw I420 to this path and exit")
	bitrate := flag.Int("bitrate", 6_000_000, "CBR target in bits per second")
	layers := flag.Int("layers", 1, "temporal layers (1 flat, 2 or 3 layered)")
	tiles := flag.Int("tiles", 0, "tile columns override (0 = default)")
	golden := flag.Int("golden", 0, "golden refresh interval in base-layer frames (0 = encoder default, negative = disable)")
	width := flag.Int("width", defaultWidth, "frame width in pixels")
	height := flag.Int("height", defaultHeight, "frame height in pixels")
	input := flag.String("input", "", "encode this raw I420 file instead of the synthetic scene")
	frameStats := flag.Bool("framestats", false, "print per-frame size and PSNR")
	statsCSV := flag.String("csv", "", "write per-frame CSV stats to this path")
	nframes := flag.Int("frames", frames, "frames to encode or dump")
	warmupFrames := flag.Int("warmup", 20, "frames to exclude from steady-state summary")
	keyInterval := flag.Int("keyint", 0, "force a keyframe every N frames after frame 0 (0 = only first frame and scene cuts)")
	infps := flag.Int("fps", fps, "frame rate for rate control and bitrate reporting")
	flag.Parse()
	nFrames := *nframes
	if nFrames <= 0 {
		fmt.Fprintf(os.Stderr, "invalid frame count %d\n", nFrames)
		os.Exit(1)
	}
	if *warmupFrames < 0 {
		fmt.Fprintf(os.Stderr, "invalid warmup frame count %d\n", *warmupFrames)
		os.Exit(1)
	}
	if *keyInterval < 0 {
		fmt.Fprintf(os.Stderr, "invalid keyframe interval %d\n", *keyInterval)
		os.Exit(1)
	}
	if *width < 16 || *height < 16 || *width%2 != 0 || *height%2 != 0 {
		fmt.Fprintf(os.Stderr, "invalid frame size %dx%d: encbench requires even dimensions >= 16\n", *width, *height)
		os.Exit(1)
	}
	w, h := *width, *height
	lumaPixels := w * h

	rng := rand.New(rand.NewSource(9))
	bg := make([]byte, lumaPixels)
	for i := range bg {
		bg[i] = uint8(50 + rng.Intn(90))
	}
	// One box-blur pass gives the texture the spatial correlation of natural
	// content; raw sample noise is incompressible for any codec and pins
	// every rate controller at its quantizer ceiling.
	blurred := make([]byte, lumaPixels)
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			sum := 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					sum += int(bg[(y+dy)*w+x+dx])
				}
			}
			blurred[y*w+x] = uint8(sum / 9)
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
		for n := range nFrames {
			f := makeFrame(bg, n, w, h)
			out.Write(f.Y)
			out.Write(f.U)
			out.Write(f.V)
		}
		fmt.Printf("wrote %d frames of %dx%d I420\n", nFrames, w, h)
		return
	}

	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: *bitrate, Framerate: *infps,
		TemporalLayers: *layers,
		TileColumns:    *tiles,
		GoldenInterval: *golden,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	srcs := make([]goav1.I420Frame, nFrames)
	if *input != "" {
		raw, err := os.ReadFile(*input)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		cw, ch := w/2, h/2
		frameLen := lumaPixels + 2*cw*ch
		if len(raw) < nFrames*frameLen {
			fmt.Fprintf(os.Stderr, "input holds %d frames, need %d\n", len(raw)/frameLen, nFrames)
			os.Exit(1)
		}
		for n := range nFrames {
			base := n * frameLen
			srcs[n] = goav1.I420Frame{
				Y:       raw[base : base+lumaPixels],
				U:       raw[base+lumaPixels : base+lumaPixels+cw*ch],
				V:       raw[base+lumaPixels+cw*ch : base+frameLen],
				YStride: w, ChromaStride: cw, Width: w, Height: h,
			}
		}
	} else {
		for n := range nFrames {
			srcs[n] = makeFrame(bg, n, w, h)
		}
	}
	var csvFile *os.File
	var csvWriter *csv.Writer
	if *statsCSV != "" {
		csvFile, err = os.Create(*statsCSV)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer csvFile.Close()
		csvWriter = csv.NewWriter(csvFile)
		if err := csvWriter.Write([]string{"frame", "keyframe", "temporal_id", "bytes", "psnr_y", "qindex", "encode_ms"}); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	steadyStart := *warmupFrames
	if steadyStart >= nFrames {
		steadyStart = 0
	}
	totalBytes, steadyBytes := 0, 0
	var sumPSNR, steadyPSNR float64
	minPSNR := 1e9
	var layerFrames, layerBytes [8]int
	var layerPSNR [8]float64
	layerMinPSNR := [8]float64{1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9}
	var encodeTime time.Duration
	for n := range nFrames {
		frameStart := time.Now()
		forceKey := *keyInterval > 0 && n > 0 && n%*keyInterval == 0
		out, err := enc.Encode(srcs[n], forceKey)
		frameElapsed := time.Since(frameStart)
		encodeTime += frameElapsed
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		totalBytes += len(out.Data)
		r := enc.Reconstruction()
		p := psnr(srcs[n].Y, r.Y)
		if *frameStats {
			fmt.Printf("frame %3d: T%d key=%t  %7d bytes  %6.2f dB  q=%d\n", n, out.TemporalID, out.Keyframe, len(out.Data), p, enc.QIndex())
		}
		if csvWriter != nil {
			if err := csvWriter.Write([]string{
				strconv.Itoa(n),
				strconv.FormatBool(out.Keyframe),
				strconv.Itoa(int(out.TemporalID)),
				strconv.Itoa(len(out.Data)),
				strconv.FormatFloat(p, 'f', 4, 64),
				strconv.Itoa(int(enc.QIndex())),
				strconv.FormatFloat(float64(frameElapsed.Microseconds())/1000, 'f', 3, 64),
			}); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
		}
		sumPSNR += p
		if n >= steadyStart && p < minPSNR {
			minPSNR = p
		}
		if n >= steadyStart {
			steadyBytes += len(out.Data)
			steadyPSNR += p
			tid := int(out.TemporalID)
			if tid >= 0 && tid < len(layerFrames) {
				layerFrames[tid]++
				layerBytes[tid] += len(out.Data)
				layerPSNR[tid] += p
				if p < layerMinPSNR[tid] {
					layerMinPSNR[tid] = p
				}
			}
		}
	}
	if csvWriter != nil {
		csvWriter.Flush()
		if err := csvWriter.Error(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	// Only the encode calls count: the harness's own PSNR pass costs
	// milliseconds per frame and external encoders do not pay it.
	elapsed := encodeTime
	perFrame := elapsed / time.Duration(nFrames)
	fmt.Printf("goav1: %d frames in %v (%.2f ms/frame, %.1f fps)\n", nFrames, elapsed.Round(time.Millisecond), float64(perFrame.Microseconds())/1000, float64(nFrames)/elapsed.Seconds())
	steadyFrames := nFrames - steadyStart
	fmt.Printf("goav1: %.2f Mbps overall / %.2f Mbps steady-state (target %.2f), luma PSNR %.2f dB (steady %.2f, min %.2f), final qindex %d, golden %s\n",
		float64(totalBytes*8**infps)/float64(nFrames)/1e6,
		float64(steadyBytes*8**infps)/float64(steadyFrames)/1e6,
		float64(*bitrate)/1e6, sumPSNR/float64(nFrames), steadyPSNR/float64(steadyFrames), minPSNR, enc.QIndex(), goldenLabel(*golden))
	if *layers > 1 {
		for tid, count := range layerFrames {
			if count == 0 {
				continue
			}
			fmt.Printf("goav1: steady T%d: %d frames, %.2f Mbps, %.2f dB avg, %.2f dB min\n",
				tid, count,
				float64(layerBytes[tid]*8**infps)/float64(count)/1e6,
				layerPSNR[tid]/float64(count),
				layerMinPSNR[tid])
		}
	}
}
