// Command encoder is the goav1 realtime AV1 encoder compiled to WebAssembly
// with TinyGo. It exposes a small pointer-based ABI to the page's worker:
// the worker copies each camera frame straight into a Go-owned buffer
// (avFrameBuf), calls avEncode, and copies the temporal unit back out of
// wasm memory (avOutPtr/avOutLen).
//
//	tinygo build -target=wasm -opt=2 -no-debug -panic=trap \
//	    -llvm-features=+simd128 -o ../web/encoder-simd.wasm .
package main

import (
	"time"
	"unsafe"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/examples/browser-wasm/pattern"
)

// Pixel formats the worker can hand over; they match VideoFrame.format.
const (
	formatI420 = 0
	formatNV12 = 1
	formatRGBA = 2
	formatBGRA = 3
	formatRGBX = 4
	formatBGRX = 5
)

var (
	enc       *goav1.VideoEncoder
	width     int
	height    int
	frameBuf  []byte // source pixels written by JS
	yuvBuf    []byte // I420 scratch for RGB sources
	out       []byte // last temporal unit (copied: Encode reuses its buffer)
	lastKey   bool
	lastError string
)

// main stays blocked so the //go:wasmexport entry points remain callable;
// asyncify unwinds back to JS while it waits.
func main() { select {} }

//go:wasmexport avInit
func avInit(w, h, bitrate, fps, speed int32) int32 {
	if enc != nil {
		enc.Close()
		enc = nil
	}
	e, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width:         int(w),
		Height:        int(h),
		TargetBitrate: int(bitrate),
		Framerate:     int(fps),
		MaxThreads:    1,
		Speed:         int8(speed),
		Content:       goav1.EncoderContentCamera,
	})
	if err != nil {
		lastError = err.Error()
		return -1
	}
	enc, width, height = e, int(w), int(h)
	frameBuf = make([]byte, width*height*4)
	yuvBuf = make([]byte, width*height*3/2)
	return 0
}

//go:wasmexport avFrameBuf
func avFrameBuf() uint32 { return uint32(uintptr(unsafe.Pointer(&frameBuf[0]))) }

//go:wasmexport avOutPtr
func avOutPtr() uint32 {
	if len(out) == 0 {
		return 0
	}
	return uint32(uintptr(unsafe.Pointer(&out[0])))
}

//go:wasmexport avOutLen
func avOutLen() int32 { return int32(len(out)) }

//go:wasmexport avIsKey
func avIsKey() int32 {
	if lastKey {
		return 1
	}
	return 0
}

//go:wasmexport avQIndex
func avQIndex() int32 { return int32(enc.QIndex()) }

//go:wasmexport avErrPtr
func avErrPtr() uint32 {
	if lastError == "" {
		return 0
	}
	return uint32(uintptr(unsafe.Pointer(unsafe.StringData(lastError))))
}

//go:wasmexport avErrLen
func avErrLen() int32 { return int32(len(lastError)) }

// avEncode encodes the frame the worker wrote into frameBuf as tightly packed
// planes (Y then U,V for I420; Y then UV for NV12; one RGBA-style plane
// otherwise). It returns the temporal-unit size or -1 on error.
//
//go:wasmexport avEncode
func avEncode(format, forceKey int32) int32 {
	var (
		res goav1.EncodedFrame
		err error
	)
	ySize, cSize := width*height, (width/2)*(height/2)
	switch format {
	case formatI420:
		res, err = enc.Encode(goav1.I420Frame{
			Y: frameBuf[:ySize], U: frameBuf[ySize : ySize+cSize], V: frameBuf[ySize+cSize : ySize+2*cSize],
			YStride: width, ChromaStride: width / 2, Width: width, Height: height,
		}, forceKey != 0)
	case formatNV12:
		res, err = enc.EncodeNV12(goav1.NV12Frame{
			Y: frameBuf[:ySize], UV: frameBuf[ySize : ySize+2*cSize],
			YStride: width, UVStride: width, Width: width, Height: height,
		}, forceKey != 0)
	case formatRGBA, formatRGBX, formatBGRA, formatBGRX:
		bgr := format == formatBGRA || format == formatBGRX
		rgbToI420(frameBuf, yuvBuf, width, height, bgr)
		res, err = enc.Encode(goav1.I420Frame{
			Y: yuvBuf[:ySize], U: yuvBuf[ySize : ySize+cSize], V: yuvBuf[ySize+cSize:],
			YStride: width, ChromaStride: width / 2, Width: width, Height: height,
		}, forceKey != 0)
	default:
		lastError = "unsupported pixel format"
		return -1
	}
	if err != nil {
		lastError = err.Error()
		return -1
	}
	out = append(out[:0], res.Data...)
	lastKey = res.Keyframe
	return int32(len(out))
}

// rgbToI420 converts packed 32-bit RGB(A) to BT.601 limited-range I420,
// averaging each 2x2 block for chroma.
func rgbToI420(src, dst []byte, w, h int, bgr bool) {
	ri, bi := 0, 2
	if bgr {
		ri, bi = 2, 0
	}
	yp := dst[:w*h]
	up := dst[w*h : w*h+(w/2)*(h/2)]
	vp := dst[w*h+(w/2)*(h/2):]
	for y := 0; y < h; y += 2 {
		for x := 0; x < w; x += 2 {
			var rs, gs, bs int
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					p := ((y+dy)*w + x + dx) * 4
					r, g, b := int(src[p+ri]), int(src[p+1]), int(src[p+bi])
					yp[(y+dy)*w+x+dx] = byte((66*r+129*g+25*b+128)>>8 + 16)
					rs, gs, bs = rs+r, gs+g, bs+b
				}
			}
			r, g, b := (rs+2)>>2, (gs+2)>>2, (bs+2)>>2
			c := (y/2)*(w/2) + x/2
			up[c] = byte((-38*r-74*g+112*b+128)>>8 + 128)
			vp[c] = byte((112*r-94*g-18*b+128)>>8 + 128)
		}
	}
}

// avSynth writes frame t of the synthetic scene into the frame buffer as
// tightly packed I420 at the encoder's size (call avEncode with format 0
// next). The page's #verify mode encodes the scene this way and compares the
// stream with the native `go run ./verify` output.
//
//go:wasmexport avSynth
func avSynth(t int32) {
	ySize, cSize := width*height, (width/2)*(height/2)
	pattern.Fill(goav1.I420Frame{
		Y: frameBuf[:ySize], U: frameBuf[ySize : ySize+cSize], V: frameBuf[ySize+cSize : ySize+2*cSize],
		YStride: width, ChromaStride: width / 2, Width: width, Height: height,
	}, int(t))
}

// avBench encodes `frames` synthetic 640x360 frames (the same moving pattern
// as the native probe) on a private encoder and returns mean ms per frame.
//
//go:wasmexport avBench
func avBench(frames int32) float64 {
	const w, h = 640, 360
	e, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h, TargetBitrate: 1_000_000, Framerate: 30, MaxThreads: 1,
		Content: goav1.EncoderContentCamera,
	})
	if err != nil {
		lastError = err.Error()
		return -1
	}
	defer e.Close()
	f := goav1.I420Frame{
		Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4),
		YStride: w, ChromaStride: w / 2, Width: w, Height: h,
	}
	var total time.Duration
	for t := 0; t < int(frames); t++ {
		pattern.Fill(f, t)
		start := time.Now()
		if _, err := e.Encode(f, t == 0); err != nil {
			lastError = err.Error()
			return -1
		}
		total += time.Since(start)
	}
	return float64(total.Microseconds()) / 1000 / float64(frames)
}
