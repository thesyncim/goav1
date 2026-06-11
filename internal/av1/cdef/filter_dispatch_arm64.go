// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

// init binds the NEON CDEF block filter; it is bit-exact with the pure-Go
// reference (TestFilterBlockNEONMatchesPureGo) and routes narrow shapes
// back to it internally.
func init() {
	filterBlockImpl = filterBlockNEON
}
