package goav1_test

import (
	"fmt"

	goav1 "github.com/thesyncim/goav1"
)

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
		stream = append(stream, out.Data)
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
