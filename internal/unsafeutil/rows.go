// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package unsafeutil

import (
	"errors"
	"unsafe"
)

// Rows is a footprint-validated view of a strided block of a byte slice:
// Height() rows of Width() bytes, row r starting r*stride bytes after row 0.
// It is what the strided kernels (motion compensation, the loop filter) index
// instead of the plane slice: the whole block is checked once, at NewRows, so
// a kernel whose row and column indices are constants, or loop variables
// bounded by Height() and Width(), pays no check per row.
//
// The zero Rows has no rows; every accessor on it panics.
type Rows struct {
	p      unsafe.Pointer // first byte of row 0 (nil for the zero Rows)
	stride uint
	height uint
	width  uint
}

// rowsDimMax bounds every NewRows dimension, so the footprint arithmetic
// below stays exact in 64 bits on every platform.
const rowsDimMax = 1 << 31

var errRows = errors.New("unsafeutil: Rows access out of range")

// RowsFit reports whether NewRows accepts a view of height rows of width
// bytes from off at stride over a slice of n bytes: 0 <= off, 0 <= stride,
// 1 <= width, 1 <= height, each below 1<<31, and the last row ends at or
// before n. Callers test it and take their fallback before NewRows; NewRows
// repeats the identical test, which the compiler then drops.
func RowsFit(n, off, stride, width, height int) bool {
	if uint(off) >= rowsDimMax || uint(stride) >= rowsDimMax || uint(width-1) >= rowsDimMax-1 || uint(height-1) >= rowsDimMax-1 {
		return false
	}
	return uint64(off)+uint64(height-1)*uint64(stride)+uint64(width) <= uint64(n)
}

// NewRows returns the view of height rows of width bytes of s whose row r is
// s[off+r*stride : off+r*stride+width]. It panics unless RowsFit(len(s), off,
// stride, width, height). Rows may overlap (stride < width). Panicking
// rather than reporting keeps the view's dimensions the caller's values on
// the path that uses them (constants fold into the accessors' checks).
//
// Bounds proof: with every dimension below 2^31, end = off +
// (height-1)*stride + width is below 2^63 and computed exactly in uint64; the
// check end <= len(s) puts every byte of every row, the byte at
// off + r*stride + c for r < height and c < width, at an index below len(s),
// hence inside s's allocation (and below MaxInt, so the uintptr offsets the
// accessors form cannot wrap). p is the address of s[off], an element of s
// (off < end <= len(s) because width >= 1), so it points into s's allocation
// and keeps it alive.
func NewRows(s []byte, off, stride, width, height int) Rows {
	if !RowsFit(len(s), off, stride, width, height) {
		panic(errRows)
	}
	return Rows{p: unsafe.Add(unsafe.Pointer(unsafe.SliceData(s)), off), stride: uint(stride), height: uint(height), width: uint(width)}
}

// Height is the number of rows.
func (v Rows) Height() int { return int(v.height) }

// Width is the number of bytes per row.
func (v Rows) Width() int { return int(v.width) }

// Stride is the distance in bytes between the starts of consecutive rows.
func (v Rows) Stride() int { return int(v.stride) }

// Sub returns the view of height rows of width bytes starting at row r,
// column c of v (rows r..r+height-1, columns c..c+width-1, at v's stride). It
// panics unless that block lies inside v and width and height are at least 1
// (a caller whose view covers its blocks never trips it; panicking rather
// than reporting keeps the result's dimensions the caller's constants).
//
// Bounds proof: r+height <= Height() and c+width <= Width() (each tested
// without overflow on non-negative operands), so every byte of the result's
// footprint is a byte of v's, which NewRows checked; the new first byte is
// v's row r, column c, inside v's footprint because height, width >= 1.
func (v Rows) Sub(r, c, width, height int) Rows {
	if uint(r) >= v.height || uint(height-1) >= v.height-uint(r) || uint(c) >= v.width || uint(width-1) >= v.width-uint(c) {
		panic(errRows)
	}
	return Rows{p: v.at(r, c), stride: v.stride, height: uint(height), width: uint(width)}
}

// BlockRows is the row count of a Block, as the length of an array type
// (the kernels' compile-time tap-row counts).
type BlockRows interface {
	~[2]struct{} | ~[4]struct{} | ~[6]struct{} | ~[8]struct{} | ~[14]struct{} | ~[16]struct{}
}

// Block is a validated view of len(R) rows of sixteen bytes at a stride: the
// two words a kernel needs (the first byte and the stride), with the row
// count in the type. Its accessors check constant indices against constants,
// so a kernel that names its tap rows by constant row numbers pays nothing
// per access.
type Block[R BlockRows] struct {
	p      unsafe.Pointer
	stride uint
}

// BlockAt returns the Block of len(R) rows of sixteen bytes of v from row r,
// column c. It panics unless those rows and columns lie inside v.
//
// Bounds proof: r and c are below 2^31 (so non-negative, and the sums below
// cannot wrap), r+len(R) <= Height() keeps rows r..r+len(R)-1 inside v, and
// c+16 <= Width() keeps their columns c..c+15 inside it; every byte the
// Block's accessors reach is one of those rows' sixteen bytes from column c,
// a byte of v's footprint.
func BlockAt[R BlockRows](v Rows, r, c int) Block[R] {
	var n R
	if uint(r)|uint(c) >= rowsDimMax || uint(r)+uint(len(n)) > v.height || uint(c)+16 > v.width {
		panic(errRows)
	}
	return Block[R]{p: v.at(r, c), stride: v.stride}
}

// Row16 returns row k's sixteen bytes. It panics unless k < len(R).
func (b Block[R]) Row16(k int) *[16]byte {
	var n R
	if uint(k) >= uint(len(n)) {
		panic(errRows)
	}
	return (*[16]byte)(unsafe.Add(b.p, uint(k)*b.stride))
}

// Row8 returns bytes c..c+7 of row k. It panics unless k < len(R) and
// c <= 8.
func (b Block[R]) Row8(k, c int) *[8]byte {
	var n R
	if uint(k) >= uint(len(n)) || uint(c) > 8 {
		panic(errRows)
	}
	return (*[8]byte)(unsafe.Add(b.p, uint(k)*b.stride+uint(c)))
}

// Row4 returns bytes c..c+3 of row k. It panics unless k < len(R) and
// c <= 12.
func (b Block[R]) Row4(k, c int) *[4]byte {
	var n R
	if uint(k) >= uint(len(n)) || uint(c) > 12 {
		panic(errRows)
	}
	return (*[4]byte)(unsafe.Add(b.p, uint(k)*b.stride+uint(c)))
}

// PutF32 stores the four bytes of v (its IEEE 754 bits in host byte order)
// at bytes c..c+3 of row k: a vector lane store (STR S) of four samples. It
// panics unless k < len(R) and c <= 12.
//
// Bounds proof: as Row4; float32 holds no pointers and the store needs no
// alignment on the platforms this runs on (unaligned scalar stores are
// architecturally allowed on arm64 and amd64, and checkptr checks alignment
// only for pointer-holding types).
func (b Block[R]) PutF32(k, c int, v float32) {
	var n R
	if uint(k) >= uint(len(n)) || uint(c) > 12 {
		panic(errRows)
	}
	*(*float32)(unsafe.Add(unsafe.Add(b.p, uint(c)), uint(k)*b.stride)) = v
}

// PutF64 stores the eight bytes of v at bytes c..c+7 of row k (STR D). It
// panics unless k < len(R) and c <= 8. Bounds proof: as Row8 and PutF32.
func (b Block[R]) PutF64(k, c int, v float64) {
	var n R
	if uint(k) >= uint(len(n)) || uint(c) > 8 {
		panic(errRows)
	}
	*(*float64)(unsafe.Add(unsafe.Add(b.p, uint(c)), uint(k)*b.stride)) = v
}

// in reports whether row r, columns c..c+n-1 lie inside the view.
// (n is at most 32 and c below 2^31 when it passes, so c+n cannot wrap.)
func (v Rows) in(r, c int, n uint) bool {
	return uint(r) < v.height && uint(c) < rowsDimMax && uint(c)+n <= v.width
}

// at is the address of row r, column c; in(r, c, n) must hold.
//
// Bounds proof: r < height and c + n <= width, so the n bytes from the
// result are bytes of row r of the footprint NewRows checked.
func (v Rows) at(r, c int) unsafe.Pointer {
	return unsafe.Add(v.p, uint(r)*v.stride+uint(c))
}

// Row4 returns the 4 bytes at column c of row r. It panics unless r <
// Height() and c+4 <= Width().
func (v Rows) Row4(r, c int) *[4]byte {
	if !v.in(r, c, 4) {
		panic(errRows)
	}
	return (*[4]byte)(v.at(r, c))
}

// Row8 is Row4 for 8 bytes.
func (v Rows) Row8(r, c int) *[8]byte {
	if !v.in(r, c, 8) {
		panic(errRows)
	}
	return (*[8]byte)(v.at(r, c))
}

// Row16 is Row4 for 16 bytes.
func (v Rows) Row16(r, c int) *[16]byte {
	if !v.in(r, c, 16) {
		panic(errRows)
	}
	return (*[16]byte)(v.at(r, c))
}

// Row24 is Row4 for 24 bytes.
func (v Rows) Row24(r, c int) *[24]byte {
	if !v.in(r, c, 24) {
		panic(errRows)
	}
	return (*[24]byte)(v.at(r, c))
}

// Row32 is Row4 for 32 bytes.
func (v Rows) Row32(r, c int) *[32]byte {
	if !v.in(r, c, 32) {
		panic(errRows)
	}
	return (*[32]byte)(v.at(r, c))
}
