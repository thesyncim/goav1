//go:build !goav1_poison

package encoder

// poisonSkippedClears is set by the goav1_poison build tag: state whose
// per-frame clear is skipped because nothing reads it is filled with values
// that change the output if read, so a gate run under the tag proves the
// skip exact.
const poisonSkippedClears = false
