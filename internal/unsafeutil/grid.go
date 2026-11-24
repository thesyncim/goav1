// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package unsafeutil

import "unsafe"

// GridElem is the element type of a Grid: pointer-free integers, so no
// value written through a Grid can become a pointer the garbage collector
// would follow.
type GridElem interface {
	uint8 | uint16 | int32
}

// GridRowArray is the fixed-size run of elements a Grid accessor returns:
// the motion-compensation kernels' row windows.
type GridRowArray[E GridElem] interface {
	[4]E | [8]E | [16]E | [24]E | [26]E | [30]E | [32]E | [64]E
}

// Grid is a footprint-validated view of a strided block of a slice of E:
// height rows of width elements, row r starting r*stride elements after row
// 0. The motion-compensation kernels index it instead of the plane, scratch
// and CONV_BUF slices: the block is checked once, at NewGrid, and a kernel
// whose row index is a loop variable bounded by the view's height, and whose
// window width is a constant, pays no check per row.
//
// Only NewGrid and Sub make a usable Grid. The zero Grid has no rows; every
// accessor on it panics.
type Grid[E GridElem] struct {
	p      unsafe.Pointer // first element of row 0 (nil for the zero Grid)
	stride uint           // in bytes (stride elements of E)
	height uint
	width  uint // in elements
}

// GridFit reports whether NewGrid accepts a view of height rows of width
// elements from element off at stride over a slice of n elements: off,
// stride, width-1 and height-1 all below 2^31 (so 1 <= width, height <=
// 2^31) and the last row ending at or before element n. Callers test it and
// take their fallback before NewGrid; NewGrid repeats the identical test,
// which the compiler then drops. The four range tests are one: the OR of the
// operands is below 2^31 exactly when each is.
func GridFit(n, off, stride, width, height int) bool {
	return uint(width-1)|uint(height-1)|uint(off)|uint(stride) < rowsDimMax &&
		uint64(off)+uint64(height-1)*uint64(stride)+uint64(width) <= uint64(n)
}

// NewGrid returns the view of height rows of width elements of s whose row
// r is s[off+r*stride : off+r*stride+width]. It panics unless GridFit(len(s),
// off, stride, width, height).
//
// Bounds proof: with off, stride and height-1 below 2^31 and width at most
// 2^31, the footprint end off + (height-1)*stride + width is below 2^63,
// exact in uint64, and at most len(s), so the element at off + r*stride + c
// for r < height and c < width is an element of s. The view keeps the stride
// in bytes, stride*Sizeof(E), and the accessors address that element as
// p + r*(stride*Sizeof(E)) + c*Sizeof(E): its byte offset from &s[0],
// (off + r*stride + c)*Sizeof(E) with Sizeof(E) <= 4, is below
// len(s)*Sizeof(E), the size of s's elements (at most MaxInt, so the uintptr
// offsets cannot wrap). Every element of s is aligned for E, so is every
// element the view addresses. p is &s[off] (off < end <= len(s) because
// width >= 1), inside s's allocation, and keeps it alive.
func NewGrid[E GridElem](s []E, off, stride, width, height int) Grid[E] {
	if !(uint(width-1)|uint(height-1)|uint(off)|uint(stride) < rowsDimMax &&
		uint64(off)+uint64(height-1)*uint64(stride)+uint64(width) <= uint64(len(s))) {
		panic(errRows)
	}
	return Grid[E]{p: unsafe.Add(unsafe.Pointer(unsafe.SliceData(s)), uintptr(off)*unsafe.Sizeof(s[0])), stride: uint(stride) * uint(unsafe.Sizeof(s[0])), height: uint(height), width: uint(width)}
}

// Height is the number of rows.
func (g Grid[E]) Height() int { return int(g.height) }

// Width is the number of elements per row.
func (g Grid[E]) Width() int { return int(g.width) }

// Fits reports whether Sub(r, c, width, height) is a block of g: width and
// height at least 1, rows r..r+height-1 and columns c..c+width-1 inside g.
// A kernel that tests it and returns early (or a caller that takes its
// fallback) before Sub pays for the test once: Sub repeats it, and the
// compiler drops the repeat and with it Sub's panic. The test is written so
// constant operands fold: r, c, width-1 and height-1 below 2^31 (one OR'd
// test), then r+height <= height of g and c+width <= width of g, sums that
// cannot wrap.
func (g Grid[E]) Fits(r, c, width, height int) bool {
	return uint(r)|uint(c)|uint(width-1)|uint(height-1) < rowsDimMax && uint(r)+uint(height) <= g.height && uint(c)+uint(width) <= g.width
}

// Sub returns the view of height rows of width elements starting at row r,
// column c of g. It panics unless that block lies inside g and width and
// height are at least 1.
//
// Bounds proof: r+height <= g's height and c+width <= g's width (each sum of
// operands below 2^31, so exact), so every element of the
// result's footprint is one of g's; the new first element is g's row r,
// column c, inside g's footprint because height, width >= 1. The stride is
// g's, so the result's footprint arithmetic is g's restricted.
func (g Grid[E]) Sub(r, c, width, height int) Grid[E] {
	if !g.Fits(r, c, width, height) {
		panic(errRows)
	}
	var e E
	return Grid[E]{p: unsafe.Add(g.p, uint(r)*g.stride+uint(c)*uint(unsafe.Sizeof(e))), stride: g.stride, height: uint(height), width: uint(width)}
}

// Row returns the len(A) elements at column c of row r of g. It panics
// unless r is below g's height and c+len(A) <= g's width.
//
// Bounds proof: r < height and c + len(A) <= width (c below 2^31 and len(A)
// at most 64, so the sum cannot wrap) put the len(A) elements inside row r of
// the footprint NewGrid checked.
func Row[A GridRowArray[E], E GridElem](g Grid[E], r, c int) *A {
	var a A
	if uint(r) >= g.height || uint(c) >= rowsDimMax || uint(c)+uint(len(a)) > g.width {
		panic(errRows)
	}
	// The address of row r, column c, written out rather than called so the
	// accessor leaves one inline mark at its call site.
	var e E
	return (*A)(unsafe.Add(g.p, uint(r)*g.stride+uint(c)*uint(unsafe.Sizeof(e))))
}

// CopyBlock copies the dst.Height() rows of dst.Width() bytes from the same
// rows and columns of src onto dst, as whole-array row copies (vector loads
// and stores), and reports true: the motion compensation put copy. It copies
// nothing and reports false unless src has at least as many rows and bytes
// per row as dst, and dst.Width() is 4, 8, 16, 32, 64, 128 or 256 (callers
// copy other widths themselves). It is a leaf: an odd first row alone, then
// two rows per iteration through a pointer pair per view, each pointer
// stepped only while rows remain.
//
// Bounds proof: the test puts rows 0..h-1, bytes 0..n-1 (h = dst.Height(),
// n = dst.Width()) inside both views' footprints. Every Grid NewGrid or Sub
// makes has height and width at least 1 and the zero Grid has width 0, so a
// dst that reaches a copying case has h >= 1. The loops form only the
// addresses p + k*stride + c, for rows k from 0 to the last row copied and
// columns c a multiple of the copy size below n (0, or the 64-byte quarters
// of a 128- or 256-byte row): the addresses the accessors would form, inside
// each footprint, each reaching bytes c..c+size-1 <= n-1 of its row. Views
// that overlap copy each row piece as a load then a store.
func CopyBlock(dst, src Grid[uint8]) bool {
	h, n := dst.height, dst.width
	if h > src.height || n > src.width {
		return false
	}
	d, s, ds, ss := dst.p, src.p, dst.stride, src.stride
	switch n {
	case 4:
		if h&1 != 0 {
			*(*[4]byte)(d) = *(*[4]byte)(s)
			if h == 1 {
				break
			}
			d, s, h = unsafe.Add(d, ds), unsafe.Add(s, ss), h-1
		}
		d1, s1 := unsafe.Add(d, ds), unsafe.Add(s, ss)
		for {
			*(*[4]byte)(d) = *(*[4]byte)(s)
			*(*[4]byte)(d1) = *(*[4]byte)(s1)
			if h -= 2; h == 0 {
				break
			}
			d, s, d1, s1 = unsafe.Add(d, 2*ds), unsafe.Add(s, 2*ss), unsafe.Add(d1, 2*ds), unsafe.Add(s1, 2*ss)
		}
	case 8:
		if h&1 != 0 {
			*(*[8]byte)(d) = *(*[8]byte)(s)
			if h == 1 {
				break
			}
			d, s, h = unsafe.Add(d, ds), unsafe.Add(s, ss), h-1
		}
		d1, s1 := unsafe.Add(d, ds), unsafe.Add(s, ss)
		for {
			*(*[8]byte)(d) = *(*[8]byte)(s)
			*(*[8]byte)(d1) = *(*[8]byte)(s1)
			if h -= 2; h == 0 {
				break
			}
			d, s, d1, s1 = unsafe.Add(d, 2*ds), unsafe.Add(s, 2*ss), unsafe.Add(d1, 2*ds), unsafe.Add(s1, 2*ss)
		}
	case 16:
		if h&1 != 0 {
			*(*[16]byte)(d) = *(*[16]byte)(s)
			if h == 1 {
				break
			}
			d, s, h = unsafe.Add(d, ds), unsafe.Add(s, ss), h-1
		}
		d1, s1 := unsafe.Add(d, ds), unsafe.Add(s, ss)
		for {
			*(*[16]byte)(d) = *(*[16]byte)(s)
			*(*[16]byte)(d1) = *(*[16]byte)(s1)
			if h -= 2; h == 0 {
				break
			}
			d, s, d1, s1 = unsafe.Add(d, 2*ds), unsafe.Add(s, 2*ss), unsafe.Add(d1, 2*ds), unsafe.Add(s1, 2*ss)
		}
	case 32:
		if h&1 != 0 {
			*(*[32]byte)(d) = *(*[32]byte)(s)
			if h == 1 {
				break
			}
			d, s, h = unsafe.Add(d, ds), unsafe.Add(s, ss), h-1
		}
		d1, s1 := unsafe.Add(d, ds), unsafe.Add(s, ss)
		for {
			*(*[32]byte)(d) = *(*[32]byte)(s)
			*(*[32]byte)(d1) = *(*[32]byte)(s1)
			if h -= 2; h == 0 {
				break
			}
			d, s, d1, s1 = unsafe.Add(d, 2*ds), unsafe.Add(s, 2*ss), unsafe.Add(d1, 2*ds), unsafe.Add(s1, 2*ss)
		}
	case 64:
		if h&1 != 0 {
			*(*[64]byte)(d) = *(*[64]byte)(s)
			if h == 1 {
				break
			}
			d, s, h = unsafe.Add(d, ds), unsafe.Add(s, ss), h-1
		}
		d1, s1 := unsafe.Add(d, ds), unsafe.Add(s, ss)
		for {
			*(*[64]byte)(d) = *(*[64]byte)(s)
			*(*[64]byte)(d1) = *(*[64]byte)(s1)
			if h -= 2; h == 0 {
				break
			}
			d, s, d1, s1 = unsafe.Add(d, 2*ds), unsafe.Add(s, 2*ss), unsafe.Add(d1, 2*ds), unsafe.Add(s1, 2*ss)
		}
	case 128, 256:
		// 64-byte column quarters: one larger array copy would call memmove.
		// (rows of these widths are rare; one row per iteration).
		for {
			for c := uint(0); c < n; c += 64 {
				*(*[64]byte)(unsafe.Add(d, c)) = *(*[64]byte)(unsafe.Add(s, c))
			}
			if h--; h == 0 {
				break
			}
			d, s = unsafe.Add(d, ds), unsafe.Add(s, ss)
		}
	default:
		return false
	}
	return true
}
