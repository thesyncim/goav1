// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

// SlicesOverlap reports whether a and b share an element, using only
// pointer equality. Two slices of one backing array that both extend to its
// end (every two-index reslice does, as Bind's planes and the decoder's
// scratch arenas are cut) end at the same element; their starts then lie
// cap(a) and cap(b) elements before it, which places both ranges exactly.
// Slices of different arrays never overlap. The one case it cannot see is two
// overlapping three-index slices whose capacities were cut short of the
// array's end: those report false.
func SlicesOverlap[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	ea, eb := a[:cap(a)], b[:cap(b)]
	if &ea[len(ea)-1] != &eb[len(eb)-1] {
		return false
	}
	// a covers [end-cap(a), end-cap(a)+len(a)) and b likewise.
	return cap(b)-len(b) < cap(a) && cap(a)-len(a) < cap(b)
}
