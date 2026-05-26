package goav1_test

import (
	"fmt"
	"os"
	"path/filepath"

	av1 "github.com/thesyncim/goav1"
)

// Example demonstrates the basic public API surface: opening a libaom IVF
// test vector, iterating its AV1 frames, and walking each frame's OBU units
// with the low-overhead bitstream parser. This is the minimum useful smoke
// path through the read-side public API and serves as executable documentation
// for callers wiring goav1 into their own decoder.
//
// All buffers in this example are caller-owned; goav1 returns slices that
// alias the input bytes, so no copying is required and no allocations occur
// in the hot iterator path.
func Example() {
	// One of the libaom conformance vectors that goav1's fast dry-run suite
	// currently passes. The test data lives inside internal/av1/testdata and
	// is shipped with the repository so this example is self-contained.
	path := filepath.Join("internal", "av1", "testdata", "libaom",
		"av1-1-b8-00-quantizer-00.ivf")

	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Println("read fixture:", err)
		return
	}

	// Parse the DKIF/AV01 IVF header and walk frames. NewIVFIterator does
	// not copy the input; Frame.Payload aliases the slice we just read.
	it, err := av1.NewIVFIterator(raw)
	if err != nil {
		fmt.Println("ivf iterator:", err)
		return
	}
	header := it.Header()
	fmt.Printf("ivf %dx%d frames=%d\n",
		header.Width, header.Height, header.FrameCount)

	const maxFrames = 3
	var frames int
	for frames < maxFrames {
		frame, ok, err := it.Next()
		if err != nil {
			fmt.Println("ivf next:", err)
			return
		}
		if !ok {
			break
		}
		// Walk the OBUs inside the AV1 temporal unit payload using the
		// zero-allocation low-overhead iterator.
		obus, err := countOBUs(frame.Payload)
		if err != nil {
			fmt.Println("obu walk:", err)
			return
		}
		fmt.Printf("frame %d ts=%d bytes=%d obus=%d\n",
			frame.Index, frame.Timestamp, len(frame.Payload), obus)
		frames++
	}

	// Output:
	// ivf 352x288 frames=2
	// frame 0 ts=0 bytes=74506 obus=3
	// frame 1 ts=1 bytes=54381 obus=2
}

// countOBUs walks one IVF frame payload as a low-overhead AV1 OBU stream and
// returns the number of OBU units it contains. ParseLowOverheadOBU does not
// allocate; the returned Unit values reference the input slice directly.
func countOBUs(payload []byte) (int, error) {
	var count int
	for len(payload) > 0 {
		_, n, err := av1.ParseLowOverheadOBU(payload)
		if err != nil {
			return 0, err
		}
		count++
		payload = payload[n:]
	}
	return count, nil
}
