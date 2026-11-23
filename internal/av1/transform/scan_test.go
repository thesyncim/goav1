package transform

import (
	"errors"
	"testing"
)

func TestScanOrderDependencyMatchesLibaom(t *testing.T) {
	for txSize, size := range libaomScanSizes {
		scanSize, err := ScanSize(size)
		if err != nil {
			t.Fatalf("ScanSize(%+v): %v", size, err)
		}
		for txType := range int(TypeCount) {
			if !libaomTxSizeTypeValid(txSize, txType) {
				continue
			}
			class, err := Type(txType).Class()
			if err != nil {
				t.Fatalf("Class(%d): %v", txType, err)
			}
			mode, err := DefaultScanMode(size, class)
			if err != nil {
				t.Fatalf("DefaultScanMode(%+v,%d): %v", size, class, err)
			}
			scanWidth := int(scanSize.Width)
			scanHeight := int(scanSize.Height)
			scan := make([]int16, scanWidth*scanHeight)
			inverse := make([]int16, len(scan))
			if err := FillDefaultScan(scan, inverse, size, class); err != nil {
				t.Fatalf("FillDefaultScan(%+v,%d): %v", size, class, err)
			}
			assertLibaomScanOrder(t, scan, inverse, scanWidth, scanHeight, mode)
		}
	}
}

func TestFillScanOrderKnownTables(t *testing.T) {
	tests := []struct {
		name     string
		size     Size
		mode     ScanMode
		wantScan []int16
	}{
		{
			name:     "default 4x4",
			size:     Size{Width: 4, Height: 4},
			mode:     ScanModeZigZag,
			wantScan: []int16{0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15},
		},
		{
			name:     "mcol 4x4",
			size:     Size{Width: 4, Height: 4},
			mode:     ScanModeCol1D,
			wantScan: []int16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
		{
			name:     "mrow 4x4",
			size:     Size{Width: 4, Height: 4},
			mode:     ScanModeRow1D,
			wantScan: []int16{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15},
		},
	}
	for _, tt := range tests {
		scan := make([]int16, len(tt.wantScan))
		inverse := make([]int16, len(tt.wantScan))
		if err := FillScanOrder(scan, inverse, tt.size, tt.mode); err != nil {
			t.Fatalf("%s FillScanOrder: %v", tt.name, err)
		}
		for i, want := range tt.wantScan {
			if scan[i] != want {
				t.Fatalf("%s scan[%d]=%d want %d", tt.name, i, scan[i], want)
			}
			if inverse[scan[i]] != int16(i) {
				t.Fatalf("%s inverse[%d]=%d want %d", tt.name, scan[i], inverse[scan[i]], i)
			}
		}
	}
}

func TestScanSizeAdjustsLibaom64CoefficientScans(t *testing.T) {
	tests := []struct {
		size Size
		want Size
	}{
		{size: Size{Width: 64, Height: 64}, want: Size{Width: 32, Height: 32}},
		{size: Size{Width: 64, Height: 32}, want: Size{Width: 32, Height: 32}},
		{size: Size{Width: 32, Height: 64}, want: Size{Width: 32, Height: 32}},
		{size: Size{Width: 64, Height: 16}, want: Size{Width: 32, Height: 16}},
		{size: Size{Width: 16, Height: 64}, want: Size{Width: 16, Height: 32}},
	}
	for _, tt := range tests {
		got, err := ScanSize(tt.size)
		if err != nil {
			t.Fatalf("ScanSize(%+v): %v", tt.size, err)
		}
		if got != tt.want {
			t.Fatalf("ScanSize(%+v)=%+v want %+v", tt.size, got, tt.want)
		}
	}
}

func TestFillScanOrderRejectsInvalidInputs(t *testing.T) {
	scan := make([]int16, 16)
	inverse := make([]int16, 16)
	if err := FillScanOrder(scan[:15], inverse, Size{Width: 4, Height: 4}, ScanModeZigZag); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("short scan err=%v want %v", err, ErrInvalidTransform)
	}
	if err := FillScanOrder(scan, inverse[:15], Size{Width: 4, Height: 4}, ScanModeZigZag); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("short inverse err=%v want %v", err, ErrInvalidTransform)
	}
	if err := FillScanOrder(scan, inverse, Size{Width: 4, Height: 12}, ScanModeZigZag); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid size err=%v want %v", err, ErrInvalidTransform)
	}
	if err := FillScanOrder(scan, inverse, Size{Width: 4, Height: 4}, ScanMode(99)); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid mode err=%v want %v", err, ErrInvalidTransform)
	}
	if _, err := DefaultScanMode(Size{Width: 4, Height: 4}, Class(99)); !errors.Is(err, ErrInvalidTransform) {
		t.Fatalf("invalid class err=%v want %v", err, ErrInvalidTransform)
	}
}

func TestFillScanOrderAllocs(t *testing.T) {
	var scan [32 * 32]int16
	var inverse [32 * 32]int16
	allocs := testing.AllocsPerRun(1000, func() {
		if err := FillDefaultScan(scan[:], inverse[:], Size{Width: 64, Height: 64}, Class2D); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("FillDefaultScan allocated: %f", allocs)
	}
}

func FuzzFillDefaultScan(f *testing.F) {
	f.Add(uint8(0), uint8(0))
	f.Add(uint8(4), uint8(2))
	f.Add(uint8(18), uint8(1))

	f.Fuzz(func(t *testing.T, rawSize uint8, rawClass uint8) {
		size := libaomScanSizes[int(rawSize)%len(libaomScanSizes)]
		class := Class(rawClass % 3)
		scanSize, err := ScanSize(size)
		if err != nil {
			t.Fatalf("ScanSize(%+v): %v", size, err)
		}
		scanWidth := int(scanSize.Width)
		scanHeight := int(scanSize.Height)
		scan := make([]int16, scanWidth*scanHeight)
		inverse := make([]int16, len(scan))
		mode, err := DefaultScanMode(size, class)
		if err != nil {
			t.Fatalf("DefaultScanMode(%+v,%d): %v", size, class, err)
		}
		if err := FillDefaultScan(scan, inverse, size, class); err != nil {
			t.Fatalf("FillDefaultScan(%+v,%d): %v", size, class, err)
		}
		assertLibaomScanOrder(t, scan, inverse, scanWidth, scanHeight, mode)
	})
}

func assertLibaomScanOrder(t *testing.T, scan []int16, inverse []int16, width int, height int, mode ScanMode) {
	t.Helper()
	dim := width + height - 1
	si := 0
	check := func(row int, col int) {
		t.Helper()
		coeff := col*height + row
		if inverse[coeff] != int16(si) || scan[si] != int16(coeff) {
			t.Fatalf("scan mismatch row=%d col=%d si=%d inverse=%d scan=%d wantCoeff=%d",
				row, col, si, inverse[coeff], scan[si], coeff)
		}
		si++
	}
	switch mode {
	case ScanModeZigZag:
		for i := range dim {
			if i%2 == 0 {
				for col := range width {
					row := i - col
					if row >= 0 && row < height {
						check(row, col)
					}
				}
				continue
			}
			for row := range height {
				col := i - row
				if col >= 0 && col < width {
					check(row, col)
				}
			}
		}
	case ScanModeColDiag:
		for i := range dim {
			for col := range width {
				row := i - col
				if row >= 0 && row < height {
					check(row, col)
				}
			}
		}
	case ScanModeRowDiag:
		for i := range dim {
			for row := range height {
				col := i - row
				if col >= 0 && col < width {
					check(row, col)
				}
			}
		}
	case ScanModeRow1D:
		for row := range height {
			for col := range width {
				check(row, col)
			}
		}
	case ScanModeCol1D:
		for col := range width {
			for row := range height {
				check(row, col)
			}
		}
	default:
		t.Fatalf("invalid scan mode %d", mode)
	}
	if si != len(scan) {
		t.Fatalf("visited %d coefficients want %d", si, len(scan))
	}
}

func libaomTxSizeTypeValid(txSize int, txType int) bool {
	txSizeSqrUp := libaomTxSizeSqrUp[txSize]
	txSetType := 5
	if txSizeSqrUp > 3 {
		txSetType = 0
	} else if txSizeSqrUp == 3 {
		txSetType = 1
	}
	return libaomExtTxUsed[txSetType][txType] != 0
}

var libaomScanSizes = [...]Size{
	{Width: 4, Height: 4},
	{Width: 8, Height: 8},
	{Width: 16, Height: 16},
	{Width: 32, Height: 32},
	{Width: 64, Height: 64},
	{Width: 4, Height: 8},
	{Width: 8, Height: 4},
	{Width: 8, Height: 16},
	{Width: 16, Height: 8},
	{Width: 16, Height: 32},
	{Width: 32, Height: 16},
	{Width: 32, Height: 64},
	{Width: 64, Height: 32},
	{Width: 4, Height: 16},
	{Width: 16, Height: 4},
	{Width: 8, Height: 32},
	{Width: 32, Height: 8},
	{Width: 16, Height: 64},
	{Width: 64, Height: 16},
}

var libaomTxSizeSqrUp = [...]int{
	0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4,
}

var libaomExtTxUsed = [...][16]int{
	{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	{1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	{1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
	{1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
}
