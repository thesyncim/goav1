package threading

import (
	"testing"
	"unsafe"
)

// TestCopyCDFStorageCoversEveryField fills every byte of a source storage
// and checks that copyCDFStorage reproduces it, so a field added to the
// storage (or to one of its CDF tables) without a copy fails here.
func TestCopyCDFStorageCoversEveryField(t *testing.T) {
	src := new(FrameWorkTileResidualCDFStorage)
	b := unsafe.Slice((*byte)(unsafe.Pointer(src)), unsafe.Sizeof(*src))
	for i := range b {
		b[i] = byte(i*7 + 1)
	}
	dst := new(FrameWorkTileResidualCDFStorage)
	copyCDFStorage(dst, src)
	if *dst != *src {
		t.Fatal("copyCDFStorage left a field uncopied")
	}
}

// TestCDFStorageInitDefaultMatchesBuild checks the cached defaults against a
// fresh build for every coefficient q-context.
func TestCDFStorageInitDefaultMatchesBuild(t *testing.T) {
	for _, q := range []uint8{0, 20, 21, 60, 61, 120, 121, 255} {
		var want, got FrameWorkTileResidualCDFStorage
		if err := want.buildDefault(q); err != nil {
			t.Fatal(err)
		}
		other := uint8(255)
		if q > 120 {
			other = 0
		}
		if err := got.InitDefault(other); err != nil { // start from another context
			t.Fatal(err)
		}
		if err := got.InitDefault(q); err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("baseQIndex %d: cached defaults differ from a fresh build", q)
		}
	}
}
