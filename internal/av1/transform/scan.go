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
func FillScanOrder(scan []int16, inverse []int16, size Size, mode ScanMode) error {
	if !size.Valid() || !mode.Valid() {
		return ErrInvalidTransform
	}
	total := size.Width * size.Height
	if len(scan) < total || len(inverse) < total {
		return ErrInvalidTransform
	}

	si := 0
	put := func(row int, col int) {
		coeff := col*size.Height + row
		scan[si] = int16(coeff)
		inverse[coeff] = int16(si)
		si++
	}

	dim := size.Width + size.Height - 1
	switch mode {
	case ScanModeZigZag:
		for i := range dim {
			if i%2 == 0 {
				for col := 0; col < size.Width; col++ {
					row := i - col
					if row >= 0 && row < size.Height {
						put(row, col)
					}
				}
				continue
			}
			for row := 0; row < size.Height; row++ {
				col := i - row
				if col >= 0 && col < size.Width {
					put(row, col)
				}
			}
		}
	case ScanModeColDiag:
		for i := range dim {
			for col := 0; col < size.Width; col++ {
				row := i - col
				if row >= 0 && row < size.Height {
					put(row, col)
				}
			}
		}
	case ScanModeRowDiag:
		for i := range dim {
			for row := 0; row < size.Height; row++ {
				col := i - row
				if col >= 0 && col < size.Width {
					put(row, col)
				}
			}
		}
	case ScanModeRow1D:
		for row := 0; row < size.Height; row++ {
			for col := 0; col < size.Width; col++ {
				put(row, col)
			}
		}
	case ScanModeCol1D:
		for col := 0; col < size.Width; col++ {
			for row := 0; row < size.Height; row++ {
				put(row, col)
			}
		}
	default:
		return ErrInvalidTransform
	}
	return nil
}

func adjustedScanSize(size Size) Size {
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
