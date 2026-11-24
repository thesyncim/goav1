// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package motion

import "errors"

// The AVX2 MC kernels take each row as a bounds-checked window of the
// caller's slice. Their entry checks bound the strides by rowStrideMax and
// the row and column counts by rowsMax, so r*stride + c + width cannot
// overflow and the compiler proves a window s[o : o+n] with one compare;
// loads at constant offsets inside a window cost nothing more.
const rowStrideMax = 1 << 30

// rowsMax bounds the row counts and block widths the strided kernels
// accept, far above any block size, for the same reason.
const rowsMax = 1 << 10

// errKernelArgs is the panic value of a strided kernel called outside its
// contract (a stride above rowStrideMax or a count above rowsMax).
var errKernelArgs = errors.New("motion: strided kernel arguments out of range")

// mc8Get32 and mc8Get64 read a little-endian word as byte expressions the
// compiler merges into one load (encoding/binary is not inlined into the
// larger kernels, and a call spills every vector register).
func mc8Get32(s *[4]byte) uint32 {
	return uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 | uint32(s[3])<<24
}

func mc8Get64(s *[8]byte) uint64 {
	return uint64(s[0]) | uint64(s[1])<<8 | uint64(s[2])<<16 | uint64(s[3])<<24 |
		uint64(s[4])<<32 | uint64(s[5])<<40 | uint64(s[6])<<48 | uint64(s[7])<<56
}

// mc8Put32 and mc8Put64 are the merged little-endian stores.
func mc8Put32(d *[4]byte, x uint32) {
	d[0], d[1], d[2], d[3] = byte(x), byte(x>>8), byte(x>>16), byte(x>>24)
}

func mc8Put64(d *[8]byte, x uint64) {
	d[0], d[1], d[2], d[3] = byte(x), byte(x>>8), byte(x>>16), byte(x>>24)
	d[4], d[5], d[6], d[7] = byte(x>>32), byte(x>>40), byte(x>>48), byte(x>>56)
}
