// Ported from libaom: av1/common/scan.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package transform

// Class identifies the directional class of an AV1 transform type.
type Class uint8

const (
	Class2D Class = iota
	ClassHoriz
	ClassVert
)

// Valid reports whether c names an AV1 transform class.
func (c Class) Valid() bool {
	switch c {
	case Class2D, ClassHoriz, ClassVert:
		return true
	default:
		return false
	}
}

// ScanMode identifies an AV1 coefficient scan traversal.
type ScanMode uint8

const (
	ScanModeZigZag ScanMode = iota
	ScanModeColDiag
	ScanModeRowDiag
	ScanModeCol1D
	ScanModeRow1D
)

// Valid reports whether m names an AV1 scan mode.
func (m ScanMode) Valid() bool {
	switch m {
	case ScanModeZigZag, ScanModeColDiag, ScanModeRowDiag, ScanModeCol1D, ScanModeRow1D:
		return true
	default:
		return false
	}
}

// ScanSize returns the coefficient scan dimensions for size. AV1 reuses the
// adjusted transform-block size for 64-wide or 64-high coefficient scans.
func ScanSize(size Size) (Size, error) {
	if !size.Valid() {
		return Size{}, ErrInvalidTransform
	}
	return adjustedScanSize(size), nil
}

// DefaultScanMode returns libaom's default scan mode for the transform class
// and coefficient scan dimensions.
func DefaultScanMode(size Size, class Class) (ScanMode, error) {
	scanSize, err := ScanSize(size)
	if err != nil || !class.Valid() {
		return 0, ErrInvalidTransform
	}
	switch class {
	case Class2D:
		if scanSize.Height == scanSize.Width {
			return ScanModeZigZag, nil
		}
		if scanSize.Height > scanSize.Width {
			return ScanModeRowDiag, nil
		}
		return ScanModeColDiag, nil
	case ClassVert:
		return ScanModeRow1D, nil
	case ClassHoriz:
		return ScanModeCol1D, nil
	default:
		return 0, ErrInvalidTransform
	}
}

// FillDefaultScan writes libaom's default scan and inverse-scan order for size
// and class into caller-provided buffers.
func FillDefaultScan(scan []int16, inverse []int16, size Size, class Class) error {
	scanSize, err := ScanSize(size)
	if err != nil {
		return err
	}
	mode, err := DefaultScanMode(size, class)
	if err != nil {
		return err
	}
	return FillScanOrder(scan, inverse, scanSize, mode)
}

// FillScanOrder writes the scan and inverse-scan order for size and mode into
// caller-provided buffers. Indices use AV1's coefficient raster order.
//
// The scan orders depend only on (size x mode) and are precomputed once at
// package init, so this copies from the immutable table instead of regenerating
// the order. Behaviour is byte-identical to the historical generator.
func FillScanOrder(scan []int16, inverse []int16, size Size, mode ScanMode) error {
	if !size.Valid() || !mode.Valid() {
		return ErrInvalidTransform
	}
	width := int(size.Width)
	height := int(size.Height)
	total := width * height
	if len(scan) < total || len(inverse) < total {
		return ErrInvalidTransform
	}
	if pair := scanTables[sizeIndex(size)][mode]; pair.scan != nil {
		copy(scan[:total], pair.scan)
		copy(inverse[:total], pair.inverse)
		return nil
	}
	// Fallback for any size reachable here but not pretabulated (none in the
	// supported set); keeps behaviour identical to the original generator.
	return fillScanOrderCompute(scan, inverse, size, mode)
}

// fillScanOrderCompute is the canonical scan-order generator. It is invoked at
// init to seed the precomputed tables and serves as a defensive fallback.
func fillScanOrderCompute(scan []int16, inverse []int16, size Size, mode ScanMode) error {
	width := int(size.Width)
	height := int(size.Height)
	si := 0
	put := func(row int, col int) {
		coeff := col*height + row
		scan[si] = int16(coeff)
		inverse[coeff] = int16(si)
		si++
	}

	dim := width + height - 1
	switch mode {
	case ScanModeZigZag:
		for i := range dim {
			if i%2 == 0 {
				for col := 0; col < width; col++ {
					row := i - col
					if row >= 0 && row < height {
						put(row, col)
					}
				}
				continue
			}
			for row := 0; row < height; row++ {
				col := i - row
				if col >= 0 && col < width {
					put(row, col)
				}
			}
		}
	case ScanModeColDiag:
		for i := range dim {
			for col := 0; col < width; col++ {
				row := i - col
				if row >= 0 && row < height {
					put(row, col)
				}
			}
		}
	case ScanModeRowDiag:
		for i := range dim {
			for row := 0; row < height; row++ {
				col := i - row
				if col >= 0 && col < width {
					put(row, col)
				}
			}
		}
	case ScanModeRow1D:
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				put(row, col)
			}
		}
	case ScanModeCol1D:
		for col := 0; col < width; col++ {
			for row := 0; row < height; row++ {
				put(row, col)
			}
		}
	default:
		return ErrInvalidTransform
	}
	return nil
}

// adjustedScanSize returns the coefficient scan dimensions for a valid size via
// the precomputed table. Callers guard with size.Valid() before reaching here;
// for any non-pretabulated input it falls back to the canonical computation.
func adjustedScanSize(size Size) Size {
	idx := sizeIndex(size)
	if idx >= 0 && sizeValidTable[idx] {
		return adjustedScanSizeTable[idx]
	}
	return computeAdjustedScanSize(size)
}
