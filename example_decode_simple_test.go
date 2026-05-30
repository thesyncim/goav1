package goav1_test

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"

	av1 "github.com/thesyncim/goav1"
)

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
