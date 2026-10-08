package decoder

// FrameWorkCDEFPostFilterHBD16BandBoundary holds immutable pre-CDEF HBD sample
// rows for an in-place unit-row band. Top and Bottom each contain the two
// samples rows immediately above rowStart and below rowEnd, respectively.
// Edge bands leave the corresponding plane slices nil.
type FrameWorkCDEFPostFilterHBD16BandBoundary struct {
	Top    [3][]uint16
	Bottom [3][]uint16
}
