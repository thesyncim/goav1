// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package cdef

// init binds the NEON 8-bit-dst CDEF block filter; it is bit-exact with the
// pure-Go reference (TestFilterBlockU8NEONMatchesPureGo) and routes narrow
// shapes back to it internally. The unit-level loop binds statically in
// filter_u8_neon_arm64.go.
func init() {
	filterBlockU8Impl = filterBlockU8NEON
}
