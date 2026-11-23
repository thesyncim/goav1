package goav1_test

import (
	"fmt"
	"os"
	"path/filepath"

	av1 "github.com/thesyncim/goav1"
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
