package entropy

// CoeffScanContext is immutable scan metadata of the coefficient writers. Keep the eight-byte layout stable: position, padded
// position, lower/BR offsets, then the two EOB contexts.
type CoeffScanContext struct {
	Pos           uint16
	Padded        uint16
	Lower2DOffset int8
	BR2DOffset    int8
	LowerEOBCtx   uint8
	BREOBCtx      uint8
}
