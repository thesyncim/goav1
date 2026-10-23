// Command verify encodes the synthetic scene natively with the configuration
// the wasm encoder page uses, and writes the stream as IVF in the same layout
// the page's #verify mode produces, so the two files can be compared with cmp
// and decoded with any AV1 decoder.
//
//	go run ./verify -o native.ivf -recon native.yuv
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"time"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/examples/browser-wasm/pattern"
)

func main() {
	w := flag.Int("w", 640, "frame width")
	h := flag.Int("h", 360, "frame height")
	bitrate := flag.Int("bitrate", 1_000_000, "CBR target in bits per second")
	fps := flag.Int("fps", 30, "frame rate")
	speed := flag.Int("speed", 0, "encoder effort level")
	frames := flag.Int("frames", 60, "frames to encode")
	content := flag.String("content", "camera", "screen-content mode: auto, camera or screen")
	out := flag.String("o", "", "write the stream as IVF to this path")
	recon := flag.String("recon", "", "write the encoder reconstruction as raw I420 to this path")
	flag.Parse()

	var hint goav1.EncoderContentHint
	switch *content {
	case "auto":
		hint = goav1.EncoderContentAuto
	case "camera":
		hint = goav1.EncoderContentCamera
	case "screen":
		hint = goav1.EncoderContentScreen
	default:
		fatal(fmt.Errorf("unknown -content %q", *content))
	}
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: *w, Height: *h, TargetBitrate: *bitrate, Framerate: *fps,
		MaxThreads: 1, Speed: int8(*speed), Content: hint,
	})
	if err != nil {
		fatal(err)
	}
	defer enc.Close()

	var ivf []byte
	hdr := make([]byte, 32)
	copy(hdr[0:4], "DKIF")
	binary.LittleEndian.PutUint16(hdr[6:], 32)
	copy(hdr[8:12], "AV01")
	binary.LittleEndian.PutUint16(hdr[12:], uint16(*w))
	binary.LittleEndian.PutUint16(hdr[14:], uint16(*h))
	binary.LittleEndian.PutUint32(hdr[16:], uint32(*fps))
	binary.LittleEndian.PutUint32(hdr[20:], 1)
	binary.LittleEndian.PutUint32(hdr[24:], uint32(*frames))
	ivf = append(ivf, hdr...)

	var encodeTime time.Duration
	var reconFile *os.File
	if *recon != "" {
		if reconFile, err = os.Create(*recon); err != nil {
			fatal(err)
		}
		defer reconFile.Close()
	}
	f := goav1.I420Frame{
		Y: make([]byte, *w**h), U: make([]byte, *w**h/4), V: make([]byte, *w**h/4),
		YStride: *w, ChromaStride: *w / 2, Width: *w, Height: *h,
	}
	for t := 0; t < *frames; t++ {
		pattern.Fill(f, t)
		start := time.Now()
		res, err := enc.Encode(f, t == 0)
		encodeTime += time.Since(start)
		if err != nil {
			fatal(err)
		}
		fh := make([]byte, 12)
		binary.LittleEndian.PutUint32(fh[0:], uint32(len(res.Data)))
		binary.LittleEndian.PutUint64(fh[4:], uint64(t))
		ivf = append(append(ivf, fh...), res.Data...)
		if reconFile != nil {
			r := enc.Reconstruction()
			for _, plane := range [][]byte{r.Y, r.U, r.V} {
				if _, err := reconFile.Write(plane); err != nil {
					fatal(err)
				}
			}
		}
	}
	if *out != "" {
		if err := os.WriteFile(*out, ivf, 0o644); err != nil {
			fatal(err)
		}
	}
	fmt.Printf("%d bytes sha256 %x\n", len(ivf), sha256.Sum256(ivf))
	fmt.Printf("encode %.2f ms/frame\n", float64(encodeTime.Microseconds())/1000/float64(*frames))
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "verify:", err)
	os.Exit(1)
}
