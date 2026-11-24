package encoder

import "errors"

// The strided SIMD kernels take each row as a bounds-checked window of the
// caller's slice. Their entry checks bound the stride by rowStrideMax and
// the row count by a small constant, so r*stride + width cannot overflow and
// the compiler proves a window s[r*stride : r*stride+w] with one compare;
// loads at constant offsets inside a window cost nothing more.
const rowStrideMax = 1 << 30

// rowsMax bounds the row counts (and block widths) the strided kernels
// accept, far above any block size, for the same reason.
const rowsMax = 1 << 10

// errKernelArgs is the panic value of a strided kernel called outside its
// contract (a stride above rowStrideMax or a row count above rowsMax).
var errKernelArgs = errors.New("encoder: strided kernel arguments out of range")
