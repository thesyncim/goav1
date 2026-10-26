package frame

import "testing"

// TestLoadSamplePlaneRowsMatchesFull checks that loading a plane in bands
// fills the same snapshot as LoadSamplePlaneFull.
func TestLoadSamplePlaneRowsMatchesFull(t *testing.T) {
	for _, bps := range []int{1, 2} {
		const w, h = 37, 23
		stride := (w + 5) * bps
		src := Plane{Pix: make([]byte, stride*h), Stride: stride, Width: w, Height: h}
		for i := range src.Pix {
			src.Pix[i] = byte(i*7 + i>>3)
		}
		if bps == 2 {
			for i := 1; i < len(src.Pix); i += 2 {
				src.Pix[i] &= 3
			}
		}
		n, err := SamplePlaneLen(src, bps)
		if err != nil {
			t.Fatal(err)
		}
		full := make([]uint16, n)
		if _, _, err := LoadSamplePlaneFull(full, src, bps); err != nil {
			t.Fatal(err)
		}
		banded := make([]uint16, n)
		dst, err := BindSamplePlane(banded, src, bps)
		if err != nil {
			t.Fatal(err)
		}
		for _, y := range [][2]int{{0, 5}, {5, 6}, {6, 17}, {17, h}} {
			if err := LoadSamplePlaneRows(dst, src, bps, y[0], y[1]); err != nil {
				t.Fatal(err)
			}
		}
		for i := range full {
			if full[i] != banded[i] {
				t.Fatalf("bps=%d: sample %d = %d, want %d", bps, i, banded[i], full[i])
			}
		}
	}
}
