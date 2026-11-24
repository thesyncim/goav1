package restoration

// The strided SIMD kernels take each row as a bounds-checked window of the
// caller's slice. Their footprint checks bound the strides by rowStrideMax
// and the row counts by rowsMax, so r*stride + width cannot overflow and the
// compiler proves a window s[r*stride : r*stride+w] with one compare; loads
// at constant offsets inside a window cost nothing more.
const (
	rowStrideMax = 1 << 30
	rowsMax      = 1 << 12
)
