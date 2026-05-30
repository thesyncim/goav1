package transform

import "testing"

// allTransformSizes enumerates every supported AV1 transform shape so the
// precompute equality checks cover the full domain.
var allTransformSizes = []Size{
	{Width: 4, Height: 4},
	{Width: 4, Height: 8},
	{Width: 8, Height: 4},
	{Width: 4, Height: 16},
	{Width: 8, Height: 8},
	{Width: 8, Height: 16},
	{Width: 16, Height: 4},
	{Width: 16, Height: 8},
	{Width: 16, Height: 32},
	{Width: 32, Height: 16},
	{Width: 32, Height: 64},
	{Width: 64, Height: 32},
	{Width: 8, Height: 32},
	{Width: 16, Height: 16},
	{Width: 16, Height: 64},
	{Width: 32, Height: 8},
	{Width: 32, Height: 32},
	{Width: 64, Height: 16},
	{Width: 64, Height: 64},
}

// referenceShift reproduces the historical Size.shift switch so the precomputed
// table can be proven byte-identical.
func referenceShift(s Size) (int, bool) {
	switch s {
	case Size{Width: 4, Height: 4},
		Size{Width: 4, Height: 8},
		Size{Width: 8, Height: 4}:
		return 0, true
	case Size{Width: 4, Height: 16},
		Size{Width: 8, Height: 8},
		Size{Width: 8, Height: 16},
		Size{Width: 16, Height: 4},
		Size{Width: 16, Height: 8},
		Size{Width: 16, Height: 32},
		Size{Width: 32, Height: 16},
		Size{Width: 32, Height: 64},
		Size{Width: 64, Height: 32}:
		return 1, true
	case Size{Width: 8, Height: 32},
		Size{Width: 16, Height: 16},
		Size{Width: 16, Height: 64},
		Size{Width: 32, Height: 8},
		Size{Width: 32, Height: 32},
		Size{Width: 64, Height: 16},
		Size{Width: 64, Height: 64}:
		return 2, true
	default:
		return 0, false
	}
}

// referenceAdjustedScanSize reproduces the historical adjustedScanSize switch.
func referenceAdjustedScanSize(size Size) Size {
	switch size {
	case Size{Width: 64, Height: 64},
		Size{Width: 64, Height: 32},
		Size{Width: 32, Height: 64}:
		return Size{Width: 32, Height: 32}
	case Size{Width: 64, Height: 16}:
		return Size{Width: 32, Height: 16}
	case Size{Width: 16, Height: 64}:
		return Size{Width: 16, Height: 32}
	default:
		return size
	}
}

func TestPrecomputedShiftMatchesReference(t *testing.T) {
	// Sweep the full (width,height) cross product of AV1 dimensions plus a few
	// invalid extents to confirm validity and shift agree with the reference.
	dims := []int{4, 8, 16, 32, 64, 12, 0}
	for _, w := range dims {
		for _, h := range dims {
			s := Size{Width: w, Height: h}
			wantShift, wantOK := referenceShift(s)
			gotShift, gotOK := s.shift()
			if gotOK != wantOK {
				t.Fatalf("shift valid %+v=%v want %v", s, gotOK, wantOK)
			}
			if wantOK && gotShift != wantShift {
				t.Fatalf("shift %+v=%d want %d", s, gotShift, wantShift)
			}
			if got := s.Valid(); got != wantOK {
				t.Fatalf("Valid %+v=%v want %v", s, got, wantOK)
			}
		}
	}
}

func TestPrecomputedAdjustedScanSizeMatchesReference(t *testing.T) {
	for _, s := range allTransformSizes {
		want := referenceAdjustedScanSize(s)
		if got := adjustedScanSize(s); got != want {
			t.Fatalf("adjustedScanSize %+v=%+v want %+v", s, got, want)
		}
	}
}

func TestPrecomputedScanOrdersMatchGenerator(t *testing.T) {
	for _, s := range allTransformSizes {
		scanSize := adjustedScanSize(s)
		total := scanSize.Width * scanSize.Height
		for mode := range numScanModes {
			wantScan := make([]int16, total)
			wantInverse := make([]int16, total)
			if err := fillScanOrderCompute(wantScan, wantInverse, scanSize, ScanMode(mode)); err != nil {
				t.Fatalf("generator %+v mode %d: %v", scanSize, mode, err)
			}

			gotScan := make([]int16, total)
			gotInverse := make([]int16, total)
			if err := FillScanOrder(gotScan, gotInverse, scanSize, ScanMode(mode)); err != nil {
				t.Fatalf("FillScanOrder %+v mode %d: %v", scanSize, mode, err)
			}

			for i := range wantScan {
				if gotScan[i] != wantScan[i] {
					t.Fatalf("scan %+v mode %d [%d]=%d want %d", scanSize, mode, i, gotScan[i], wantScan[i])
				}
				if gotInverse[i] != wantInverse[i] {
					t.Fatalf("inverse %+v mode %d [%d]=%d want %d", scanSize, mode, i, gotInverse[i], wantInverse[i])
				}
			}
		}
	}
}

// TestPrecomputedTablesAreSharedReadOnly confirms FillScanOrder copies out of
// the immutable table rather than returning the table slice itself, so callers
// cannot corrupt shared state.
func TestPrecomputedTablesAreSharedReadOnly(t *testing.T) {
	size := Size{Width: 4, Height: 4}
	idx := sizeIndex(size)
	orig := append([]int16(nil), scanTables[idx][ScanModeZigZag].scan...)

	scan := make([]int16, 16)
	inverse := make([]int16, 16)
	if err := FillScanOrder(scan, inverse, size, ScanModeZigZag); err != nil {
		t.Fatal(err)
	}
	for i := range scan {
		scan[i] = -1
	}
	for i := range orig {
		if scanTables[idx][ScanModeZigZag].scan[i] != orig[i] {
			t.Fatalf("table mutated at %d: %d want %d", i, scanTables[idx][ScanModeZigZag].scan[i], orig[i])
		}
	}
}
