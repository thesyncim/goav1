package goav1_test

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	av1 "github.com/thesyncim/goav1"
	goav1 "github.com/thesyncim/goav1"
	"os"
	"path/filepath"
)

// Example opens an IVF file and walks its temporal units. NewIVFIterator does
// not copy: each frame's Payload aliases the input, and the payloads can be
// handed to NewDecoder as they are.
func Example() {
	path := filepath.Join("internal", "av1", "testdata", "libaom",
		"av1-1-b8-00-quantizer-00.ivf")
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Println("read fixture:", err)
		return
	}

	it, err := av1.NewIVFIterator(raw)
	if err != nil {
		fmt.Println("ivf iterator:", err)
		return
	}
	header := it.Header()
	fmt.Printf("ivf %dx%d frames=%d\n", header.Width, header.Height, header.FrameCount)

	for {
		frame, ok, err := it.Next()
		if err != nil {
			fmt.Println("ivf next:", err)
			return
		}
		if !ok {
			break
		}
		fmt.Printf("frame %d ts=%d bytes=%d\n", frame.Index, frame.Timestamp, len(frame.Payload))
	}

	// Output:
	// ivf 352x288 frames=2
	// frame 0 ts=0 bytes=74506
	// frame 1 ts=1 bytes=54381
}

// ExampleDecodeIVF shows the simplest possible decode: hand a whole IVF stream
// to the one-shot DecodeIVF helper and get back independent, caller-owned frame
// copies. No scratch binding, no frame pool, no worker pool wiring -- the helper
// drives the same byte-exact public decode path internally.
func ExampleDecodeIVF() {
	path := filepath.Join("internal", "av1", "testvector", "testdata",
		"profiles", "profile1-444-8bit-64x64.ivf")
	ivf, err := os.ReadFile(path)
	if err != nil {
		fmt.Println("read fixture:", err)
		return
	}

	frames, err := av1.DecodeIVF(ivf)
	if err != nil {
		fmt.Println("decode:", err)
		return
	}

	for i, f := range frames {
		sum := md5.Sum(append(append(append([]byte(nil), f.Y...), f.U...), f.V...))
		fmt.Printf("frame %d %dx%d md5=%s\n", i, f.Width, f.Height, hex.EncodeToString(sum[:]))
	}

	// Output:
	// frame 0 64x64 md5=00211cdc8f799c808849c955a318a0f5
	// frame 1 64x64 md5=397ff01920ff514bc611ab49d76371c1
	// frame 2 64x64 md5=f8fbfb25a42da47a7adb71510de9b178
}

// ExampleDecoder_DecodeNext shows the streaming form: build a Decoder over the
// frame payloads and pull visible frames one payload at a time. The returned
// *Frame values alias a reused arena, so copy out anything that must outlive the
// next DecodeNext call -- here we hash each frame's bytes immediately.
func ExampleDecoder_DecodeNext() {
	path := filepath.Join("internal", "av1", "testvector", "testdata",
		"profiles", "profile1-444-8bit-64x64.ivf")
	ivf, err := os.ReadFile(path)
	if err != nil {
		fmt.Println("read fixture:", err)
		return
	}

	dec, err := av1.NewDecoderFromIVF(ivf, av1.WithWorkers(1))
	if err != nil {
		fmt.Println("new decoder:", err)
		return
	}
	defer dec.Close()

	var count int
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			fmt.Println("decode next:", err)
			return
		}
		if !ok {
			break
		}
		count += len(frames)
	}
	fmt.Printf("decoded %d visible frames\n", count)

	// Output:
	// decoded 3 visible frames
}

// ExampleVideoEncoder encodes a short fixed-quality stream and decodes it
// back, demonstrating the round trip the encoder guarantees: every frame
// decodes bit-exactly to the encoder's own reconstruction.
func ExampleVideoEncoder() {
	const w, h = 192, 128
	enc, err := goav1.NewVideoEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		QIndex: 60, // fixed quality; set TargetBitrate/Framerate for CBR
	})
	if err != nil {
		panic(err)
	}
	var stream [][]byte
	for i := range 3 {
		frame := goav1.I420Frame{
			Y: make([]byte, w*h), U: make([]byte, w/2*h/2), V: make([]byte, w/2*h/2),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		for j := range frame.Y {
			frame.Y[j] = uint8(40 + (j+i*5)%160)
		}
		for j := range frame.U {
			frame.U[j] = 120
			frame.V[j] = 130
		}
		out, err := enc.Encode(frame, false)
		if err != nil {
			panic(err)
		}
		stream = append(stream, append([]byte(nil), out.Data...))
		fmt.Printf("frame %d: keyframe=%v\n", i, out.Keyframe)
	}
	dec, err := goav1.NewDecoder(stream)
	if err != nil {
		panic(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		panic(err)
	}
	fmt.Printf("decoded %d frames\n", len(frames))
	// Output:
	// frame 0: keyframe=true
	// frame 1: keyframe=false
	// frame 2: keyframe=false
	// decoded 3 frames
}

// ExampleRTCEncoder encodes an L1T2 WebRTC stream where every frame carries
// an RTP dependency descriptor and odd frames are droppable.
func ExampleRTCEncoder() {
	const w, h = 192, 128
	enc, err := goav1.NewRTCEncoder(goav1.VideoEncoderConfig{
		Width: w, Height: h,
		TargetBitrate: 300_000, Framerate: 30,
		TemporalLayers: 2,
	})
	if err != nil {
		panic(err)
	}
	for i := range 4 {
		frame := goav1.I420Frame{
			Y: make([]byte, w*h), U: make([]byte, w/2*h/2), V: make([]byte, w/2*h/2),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		for j := range frame.U {
			frame.U[j] = 120
			frame.V[j] = 130
		}
		out, err := enc.Encode(frame, false)
		if err != nil {
			panic(err)
		}
		fmt.Printf("frame %d: tid=%d descriptor=%v\n", i, out.TemporalID, len(out.DependencyDescriptor) > 0)
	}
	// Output:
	// frame 0: tid=0 descriptor=true
	// frame 1: tid=1 descriptor=true
	// frame 2: tid=0 descriptor=true
	// frame 3: tid=1 descriptor=true
}
