//go:build !goav1_poison

package tile

// poisonSkippedClears is set by the goav1_poison build tag: zeroing that is
// skipped because nothing reads the region fills it with 0xA5 instead, so a
// gate run under the tag proves the skip is exact (any stale read changes the
// output).
const poisonSkippedClears = false
