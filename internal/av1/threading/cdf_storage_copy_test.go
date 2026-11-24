package threading

import (
	"reflect"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/poison"
)

// TestCopyCDFStorageCoversEveryField fills every byte of a source storage
// and checks that copyCDFStorage reproduces it, so a field added to the
// storage (or to one of its CDF tables) without a copy fails here.
func TestCopyCDFStorageCoversEveryField(t *testing.T) {
	src := new(FrameWorkTileResidualCDFStorage)
	// entropy.CDF keeps its fields unexported, so each CDF is filled through
	// its API: all MaxSymbols+1 lanes (the whole values array) get a value
	// unique to that CDF.
	cdfType := reflect.TypeFor[entropy.CDF]()
	lane := uint16(0)
	fillCDF := func(v reflect.Value) bool {
		if v.Type() != cdfType {
			return false
		}
		c := v.Addr().Interface().(*entropy.CDF)
		if err := c.InitUniform(entropy.MaxSymbols); err != nil {
			t.Fatal(err)
		}
		for i := range c.Values() {
			lane++
			c.Values()[i] = lane
		}
		return true
	}
	n := 0
	if ro := poison.LeavesWith(src, func(i int) byte { return byte(i*7 + 1) }, func() bool { n++; return n&1 == 1 }, fillCDF); len(ro) != 0 {
		t.Fatalf("storage leaves the poison cannot set: %q", ro)
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
