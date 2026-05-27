// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

import (
	"errors"
	"testing"
)

// TestLoadSamplePlaneFullLoadsPastVisibleColumns covers the new variant
// introduced for postfilter passes (CDEF) whose superblock-edge reads extend
// past the visible plane width into the row-stride padding columns written by
// the reconstruction stage. The default LoadSamplePlane leaves those columns at
// whatever value dst already held; LoadSamplePlaneFull must instead populate
// every column up to Stride/bytesPerSample with the actual byte values.
func TestLoadSamplePlaneFullLoadsPastVisibleColumns8Bit(t *testing.T) {
	// Width=6 visible, Stride=8 bytes → 2 past-visible columns per row.
	plane := Plane{Pix: make([]byte, 8*3), Stride: 8, Width: 6, Height: 3}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Stride; x++ {
			// Encode (row, col) so we can verify past-visible columns weren't
			// confused with visible ones.
			plane.Pix[y*plane.Stride+x] = byte(10*y + x)
		}
	}

	need, err := SamplePlaneLen(plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, need)
	for i := range scratch {
		scratch[i] = 0xefef
	}
	samples, loadedWidth, err := LoadSamplePlaneFull(scratch, plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if loadedWidth != 8 {
		t.Fatalf("loadedWidth=%d want 8", loadedWidth)
	}
	if samples.Stride != 8 || samples.Width != 6 || samples.Height != 3 {
		t.Fatalf("samples=%+v", samples)
	}
	for y := 0; y < samples.Height; y++ {
		for x := 0; x < loadedWidth; x++ {
			if got, want := samples.Pix[y*samples.Stride+x], uint16(10*y+x); got != want {
				t.Fatalf("row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
}

// TestLoadSamplePlaneFullLoadsPastVisibleColumns16Bit verifies the same
// extension for little-endian 16-bit byte planes (10-bit / 12-bit content).
func TestLoadSamplePlaneFullLoadsPastVisibleColumns16Bit(t *testing.T) {
	// Width=3 visible samples, Stride=12 bytes → strideSamples=6, 3 past-visible
	// samples per row.
	plane := Plane{Pix: make([]byte, 12*2), Stride: 12, Width: 3, Height: 2}
	for y := 0; y < plane.Height; y++ {
		for s := 0; s < plane.Stride/2; s++ {
			sample := uint16(1000 + y*10 + s)
			plane.Pix[y*plane.Stride+s*2] = byte(sample)
			plane.Pix[y*plane.Stride+s*2+1] = byte(sample >> 8)
		}
	}

	need, err := SamplePlaneLen(plane, 2)
	if err != nil {
		t.Fatal(err)
	}
	samples, loadedWidth, err := LoadSamplePlaneFull(make([]uint16, need), plane, 2)
	if err != nil {
		t.Fatal(err)
	}
	if loadedWidth != 6 {
		t.Fatalf("loadedWidth=%d want 6", loadedWidth)
	}
	if samples.Stride != 6 || samples.Width != 3 || samples.Height != 2 {
		t.Fatalf("samples=%+v", samples)
	}
	for y := 0; y < samples.Height; y++ {
		for s := 0; s < loadedWidth; s++ {
			if got, want := samples.Pix[y*samples.Stride+s], uint16(1000+y*10+s); got != want {
				t.Fatalf("row=%d sample=%d got=%d want=%d", y, s, got, want)
			}
		}
	}
}

// TestLoadSamplePlaneFullPreservesWidth pins the contract that Width remains
// the visible plane width even though the loaded data extends past it.
// Callers (such as the CDEF postfilter) rely on Width reporting the visible
// extent so the filtered output never writes past it.
func TestLoadSamplePlaneFullPreservesWidth(t *testing.T) {
	plane := Plane{Pix: make([]byte, 16*2), Stride: 16, Width: 10, Height: 2}
	for i := range plane.Pix {
		plane.Pix[i] = byte(i)
	}
	samples, loadedWidth, err := LoadSamplePlaneFull(make([]uint16, 16*2), plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if samples.Width != 10 {
		t.Fatalf("Width=%d want 10 (visible)", samples.Width)
	}
	if loadedWidth != 16 {
		t.Fatalf("loadedWidth=%d want 16 (stride samples)", loadedWidth)
	}
}

// TestLoadSamplePlaneFullStrideEqualsWidth covers the degenerate layout where
// the plane has no row-stride padding columns: loadedWidth must equal Width and
// every visible sample must still be loaded correctly.
func TestLoadSamplePlaneFullStrideEqualsWidth(t *testing.T) {
	plane := Plane{Pix: make([]byte, 5*3), Stride: 5, Width: 5, Height: 3}
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			plane.Pix[y*plane.Stride+x] = byte(7*y + x)
		}
	}
	samples, loadedWidth, err := LoadSamplePlaneFull(make([]uint16, 15), plane, 1)
	if err != nil {
		t.Fatal(err)
	}
	if loadedWidth != 5 || samples.Stride != 5 || samples.Width != 5 {
		t.Fatalf("loadedWidth=%d samples=%+v", loadedWidth, samples)
	}
	for y := 0; y < samples.Height; y++ {
		for x := 0; x < samples.Width; x++ {
			if got, want := samples.Pix[y*samples.Stride+x], uint16(7*y+x); got != want {
				t.Fatalf("row=%d col=%d got=%d want=%d", y, x, got, want)
			}
		}
	}
}

// TestLoadSamplePlaneFullEmptyPlane keeps the empty-plane contract aligned
// with LoadSamplePlane: zero-sized planes yield an empty samples view and
// loadedWidth=0 without scanning Pix.
func TestLoadSamplePlaneFullEmptyPlane(t *testing.T) {
	samples, loadedWidth, err := LoadSamplePlaneFull(nil, Plane{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if loadedWidth != 0 || samples.Width != 0 || samples.Height != 0 || samples.Stride != 0 {
		t.Fatalf("loadedWidth=%d samples=%+v", loadedWidth, samples)
	}
}

// TestLoadSamplePlaneFullShortBuffer rejects scratch shorter than the
// caller-owned layout reported by SamplePlaneLen.
func TestLoadSamplePlaneFullShortBuffer(t *testing.T) {
	plane := Plane{Pix: make([]byte, 8*2), Stride: 8, Width: 4, Height: 2}
	if _, _, err := LoadSamplePlaneFull(make([]uint16, 3), plane, 1); !errors.Is(err, ErrShortBuffer) {
		t.Fatalf("err=%v want ErrShortBuffer", err)
	}
}

// TestLoadSamplePlaneFullShortPixForFinalRow rejects planes whose Pix slice is
// shorter than Height*Stride, which would otherwise let the past-visible read
// of the final row index out of bounds.
func TestLoadSamplePlaneFullShortPixForFinalRow(t *testing.T) {
	// Width=4, Stride=8, Height=2 needs 16 bytes; the truncated buffer holds
	// only the bytes through the last visible sample (1*8 + 4 = 12), which is
	// what samplePlaneLayout allows for LoadSamplePlane but is too short for
	// LoadSamplePlaneFull's past-visible read on row 1.
	plane := Plane{Pix: make([]byte, 12), Stride: 8, Width: 4, Height: 2}
	if _, _, err := LoadSamplePlaneFull(make([]uint16, 16), plane, 1); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("err=%v want ErrInvalidPlane", err)
	}
}

// TestLoadSamplePlaneFullRejectsInvalidBytesPerSample mirrors LoadSamplePlane's
// validation so callers can substitute the variants without changing error
// semantics.
func TestLoadSamplePlaneFullRejectsInvalidBytesPerSample(t *testing.T) {
	plane := Plane{Pix: make([]byte, 8), Stride: 8, Width: 4, Height: 1}
	if _, _, err := LoadSamplePlaneFull(make([]uint16, 8), plane, 3); !errors.Is(err, ErrInvalidPlane) {
		t.Fatalf("err=%v want ErrInvalidPlane", err)
	}
}
