// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package unsafeutil

import "unsafe"

// Block32 is Block with rows of thirty-two bytes: the AVX2 loop-filter
// kernels' 256-bit rows (thirty-two 8-bit samples, or sixteen 16-bit ones).
// It is its own type, made only by Block32At, which checks the thirty-two
// byte footprint; a 16-byte Block never yields a 32-byte row.
//
// Only Block32At makes a usable Block32. The zero Block32 is not one: its
// accessor would address memory from a nil base, so it must never be used.
type Block32[R BlockRows] struct {
	p      unsafe.Pointer
	stride uint
}

// Block32At returns the Block32 of len(R) rows of thirty-two bytes of v from
// row r, column c. It panics unless those rows and columns lie inside v.
//
// Bounds proof: r and c are below 2^31 (so non-negative, and the sums below
// cannot wrap), r+len(R) <= v's height keeps rows r..r+len(R)-1 inside v,
// and c+32 <= Width() keeps their columns c..c+31 inside it; every byte
// Row32 reaches is one of those rows' thirty-two bytes from column c, a byte
// of the footprint NewRows checked. The new base is v's row r, column c,
// itself one of those bytes (len(R) >= 2).
func Block32At[R BlockRows](v Rows, r, c int) Block32[R] {
	var n R
	if uint(r)|uint(c) >= rowsDimMax || uint(r)+uint(len(n)) > v.height || uint(c)+32 > v.width {
		panic(errRows)
	}
	return Block32[R]{p: unsafe.Add(v.p, uint(r)*v.stride+uint(c)), stride: v.stride}
}

// Row32 returns row k's thirty-two bytes. It panics unless k < len(R).
//
// Bounds proof: row k is one of the rows Block32At checked, and its
// thirty-two bytes from the block's column are the ones it checked.
func (b Block32[R]) Row32(k int) *[32]byte {
	var n R
	if uint(k) >= uint(len(n)) {
		panic(errRows)
	}
	return (*[32]byte)(unsafe.Add(b.p, uint(k)*b.stride))
}
