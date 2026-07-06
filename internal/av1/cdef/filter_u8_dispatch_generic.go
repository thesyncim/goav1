// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (!amd64 && !arm64) || (arm64 && purego) || (amd64 && purego)

package cdef

// init binds the pure-Go 8-bit-dst CDEF block filter on architectures the
// dispatcher does not special-case, and on the arm64/amd64 purego builds.
// Kept only so the dispatch wiring stays symmetric across builds.
func init() {
	filterBlockU8Impl = filterBlockU8PureGo
}
