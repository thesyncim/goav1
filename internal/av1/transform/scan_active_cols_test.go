package transform

import "testing"

// TestDefaultScanActiveCols checks the column bound against a direct walk of
// the default scan prefix for every valid size and class.
func TestDefaultScanActiveCols(t *testing.T) {
	for _, w := range []uint8{4, 8, 16, 32, 64} {
		for _, h := range []uint8{4, 8, 16, 32, 64} {
			size := Size{Width: w, Height: h}
			if !size.Valid() {
				continue
			}
			scanSize, err := ScanSize(size)
			if err != nil {
				t.Fatal(err)
			}
			height := int(scanSize.Height)
			for class := Class(0); class.Valid(); class++ {
				scan, err := DefaultScan(size, class)
				if err != nil {
					t.Fatal(err)
				}
				maxCol := -1
				for eob := 0; eob <= len(scan); eob++ {
					gotCols, ok := DefaultScanActiveCols(size, class, eob)
					if !ok {
						t.Fatalf("%dx%d class %d eob %d: not ok", w, h, class, eob)
					}
					if gotCols != maxCol+1 {
						t.Fatalf("%dx%d class %d eob %d: cols %d, want %d", w, h, class, eob, gotCols, maxCol+1)
					}
					if eob < len(scan) {
						maxCol = max(maxCol, int(scan[eob])/height)
					}
				}
				if _, ok := DefaultScanActiveCols(size, class, len(scan)+1); ok {
					t.Fatalf("%dx%d class %d: eob past the scan accepted", w, h, class)
				}
			}
		}
	}
	if _, ok := DefaultScanActiveCols(Size{Width: 64, Height: 64}, Class2D, -1); ok {
		t.Fatal("negative eob accepted")
	}
}
