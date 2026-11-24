package entropy

// CoefLevelsLen sizes dav1d's t->scratch.levels token buffer. A 32x32 2D block
// needs stride 32 times 32+2 columns and the 1D classes stride 16 times 16+2
// rows; the extra slack lets the token loops address rc + 2*stride with rc
// and stride masked to their maxima, so the loads need no bounds checks.
const CoefLevelsLen = 1024 + 2*64
