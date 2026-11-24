// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package unsafeutil_test

import (
	"math"
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// gridShapes are footprints (stride, width, height in elements) for the
// guard-page tests: exact-width rows, padded strides, one row, and widths
// that are and are not multiples of the row-array sizes.
var gridShapes = []struct{ stride, width, height int }{
	{4, 4, 1}, {64, 64, 3}, {80, 64, 5}, {37, 30, 4}, {300, 256, 3}, {130, 128, 2}, {70, 33, 7},
}

// gridRowsCheck reads and writes the first and last element of every
// in-range len(A)-element window of every row of g, through Row on g and on
// one-row sub-views: a footprint one element too long or misplaced would
// fault.
func gridRowsCheck[A unsafeutil.GridRowArray[E], E unsafeutil.GridElem](g unsafeutil.Grid[E]) {
	var a A
	n := len(a)
	for r := 0; r < g.Height(); r++ {
		for c := 0; c+n <= g.Width(); c++ {
			p := unsafeutil.Row[A](g, r, c)
			(*p)[0]++
			(*p)[n-1]++
			q := unsafeutil.Row[A](g.Sub(r, c, n, 1), 0, 0)
			(*q)[n-1]--
		}
	}
}

// gridAllRows runs gridRowsCheck for every row-array size.
func gridAllRows[E unsafeutil.GridElem](g unsafeutil.Grid[E]) {
	gridRowsCheck[[4]E](g)
	gridRowsCheck[[8]E](g)
	gridRowsCheck[[16]E](g)
	gridRowsCheck[[24]E](g)
	gridRowsCheck[[26]E](g)
	gridRowsCheck[[30]E](g)
	gridRowsCheck[[32]E](g)
	gridRowsCheck[[64]E](g)
}

func testGridGuardPage[E unsafeutil.GridElem](t *testing.T) {
	for _, d := range gridShapes {
		n := (d.height-1)*d.stride + d.width
		for side, s := range map[string][]E{"after": guardpage.After[E](t, n), "before": guardpage.Before[E](t, n)} {
			if !unsafeutil.GridFit(len(s), 0, d.stride, d.width, d.height) || unsafeutil.GridFit(len(s)-1, 0, d.stride, d.width, d.height) {
				t.Fatalf("%s %+v: GridFit at the exact footprint is wrong", side, d)
			}
			g := unsafeutil.NewGrid(s, 0, d.stride, d.width, d.height)
			if g.Width() != d.width || g.Height() != d.height {
				t.Fatalf("%s %+v: shape %dx%d", side, d, g.Width(), g.Height())
			}
			guardpage.Check(t, side, func() {
				gridAllRows(g)
				// The last row and column through Sub and Fits.
				if !g.Fits(d.height-1, d.width-4, 4, 1) || g.Fits(d.height-1, d.width-3, 4, 1) || g.Fits(d.height, 0, 1, 1) {
					t.Fatalf("%s %+v: Fits at the footprint edge is wrong", side, d)
				}
				sink ^= byte(unsafeutil.Row[[4]E](g.Sub(d.height-1, d.width-4, 4, 1), 0, 0)[3])
			})
		}
	}
}

// TestGridGuardPage builds views whose footprint ends at, or starts at, an
// inaccessible page and reaches every element of every row through Row, Sub
// and Fits, for each element type and row-array size.
func TestGridGuardPage(t *testing.T) {
	t.Run("uint8", testGridGuardPage[uint8])
	t.Run("uint16", testGridGuardPage[uint16])
	t.Run("int32", testGridGuardPage[int32])
}

// TestGridOffset checks a view that starts past the slice's first element:
// its row 0 is element off, and its last row ends at the guard page.
func TestGridOffset(t *testing.T) {
	const off, stride, width, height = 5, 40, 32, 6
	n := off + (height-1)*stride + width
	s := guardpage.After[uint16](t, n)
	g := unsafeutil.NewGrid(s, off, stride, width, height)
	if p := unsafeutil.Row[[8]uint16](g, 1, 3); &p[0] != &s[off+stride+3] {
		t.Fatal("row 1 column 3 is not element off+stride+3")
	}
	guardpage.Check(t, "after", func() { gridAllRows(g) })
}

// TestCopyRectGuardPage copies blocks of every supported width (and odd and
// even heights) between slices that end at, or start at, inaccessible pages,
// and checks every byte: the copy reaches exactly the height rows of width
// bytes.
func TestCopyRectGuardPage(t *testing.T) {
	for _, w := range []int{4, 8, 16, 32, 64, 128, 256} {
		for _, h := range []int{1, 2, 3, 8, 9} {
			for _, pad := range []int{0, 3} {
				stride := w + pad
				n := (h-1)*stride + w
				for side, alloc := range map[string]func(testing.TB, int) []byte{"after": guardpage.After[byte], "before": guardpage.Before[byte]} {
					src, dst := alloc(t, n), alloc(t, n)
					for i := range src {
						src[i] = byte(i*7 + 1)
						dst[i] = 0xee
					}
					guardpage.Check(t, side, func() {
						if !unsafeutil.CopyRect(dst, 0, stride, src, 0, stride, w, h) {
							t.Fatalf("%s w=%d h=%d: CopyRect declined", side, w, h)
						}
					})
					for i := range dst {
						want := byte(0xee)
						if i%stride < w {
							want = src[i]
						}
						if dst[i] != want {
							t.Fatalf("%s w=%d h=%d pad=%d: byte %d = %#x, want %#x", side, w, h, pad, i, dst[i], want)
						}
					}
				}
			}
		}
	}
}

func TestGridRejects(t *testing.T) {
	s := make([]uint16, 100)
	for _, c := range []struct{ off, stride, width, height int }{
		{-1, 10, 10, 1}, {0, -1, 10, 1}, {0, 10, 0, 1}, {0, 10, 10, 0}, {0, 10, 11, 10}, {91, 10, 10, 1},
		{0, 1 << 31, 1, 2}, {1 << 31, 0, 1, 1}, {0, 0, 1<<31 + 1, 1}, {0, 0, 1, 1<<31 + 1},
		{0, math.MaxInt / 2, 1, 3},
	} {
		if unsafeutil.GridFit(len(s), c.off, c.stride, c.width, c.height) {
			t.Errorf("GridFit(%d, %+v) = true", len(s), c)
		}
		mustPanic(t, func() { unsafeutil.NewGrid(s, c.off, c.stride, c.width, c.height) })
	}
	g := unsafeutil.NewGrid(s, 4, 10, 16, 8) // rows at 4, 14, ..., 74, 16 elements each
	mustPanic(t, func() { unsafeutil.Row[[8]uint16](g, 8, 0) })
	mustPanic(t, func() { unsafeutil.Row[[8]uint16](g, -1, 0) })
	mustPanic(t, func() { unsafeutil.Row[[8]uint16](g, 0, 9) })
	mustPanic(t, func() { unsafeutil.Row[[8]uint16](g, 0, -1) })
	mustPanic(t, func() { unsafeutil.Row[[24]uint16](g, 0, 0) })
	mustPanic(t, func() { g.Sub(7, 0, 16, 2) })
	mustPanic(t, func() { g.Sub(0, 1, 16, 1) })
	mustPanic(t, func() { g.Sub(0, 0, 0, 1) })
	mustPanic(t, func() { g.Sub(0, 0, 1, 0) })
	mustPanic(t, func() { g.Sub(-1, 0, 1, 1) })
	mustPanic(t, func() { g.Sub(0, -1, 1, 1) })
	for _, c := range []struct{ r, c, w, h int }{{7, 0, 16, 2}, {0, 1, 16, 1}, {0, 0, 0, 1}, {0, 0, 1, 0}, {-1, 0, 1, 1}, {0, -1, 1, 1}, {8, 0, 1, 1}} {
		if g.Fits(c.r, c.c, c.w, c.h) {
			t.Errorf("Fits(%+v) = true", c)
		}
	}
	if !g.Fits(7, 15, 1, 1) || !g.Fits(0, 0, 16, 8) {
		t.Error("Fits rejects an in-range block")
	}
	if p := unsafeutil.Row[[8]uint16](g.Sub(2, 3, 8, 2), 1, 0); &p[0] != &s[4+3*10+3] {
		t.Fatal("Sub row 1 is not element 4+3*10+3")
	}
	var zero unsafeutil.Grid[uint16]
	mustPanic(t, func() { unsafeutil.Row[[4]uint16](zero, 0, 0) })
	mustPanic(t, func() { zero.Sub(0, 0, 1, 1) })
	if zero.Fits(0, 0, 1, 1) {
		t.Error("the zero Grid fits a block")
	}
	b := make([]byte, 64)
	for _, c := range []struct{ dOff, dStride, sOff, sStride, w, h int }{
		{32, 16, 0, 16, 16, 3},    // destination past the end
		{0, 16, 49, 16, 16, 1},    // source past the end
		{0, 16, 32, 16, 12, 2},    // unsupported width
		{0, 16, 32, 16, 0, 2},     // no columns
		{0, 16, 32, 16, 16, 0},    // no rows
		{-1, 16, 32, 16, 4, 1},    // negative offset
		{0, -16, 32, 16, 4, 1},    // negative stride
		{0, 16, 0, 1 << 31, 4, 1}, // stride out of range
		{0, 16, 0, math.MaxInt / 2, 4, 3},
	} {
		if unsafeutil.CopyRect(b, c.dOff, c.dStride, b, c.sOff, c.sStride, c.w, c.h) {
			t.Errorf("CopyRect(%+v) copied", c)
		}
	}
	if !unsafeutil.CopyRect(b, 32, 16, b, 0, 16, 16, 2) || !unsafeutil.CopyRect(b, 0, 16, b, 48, 16, 16, 1) {
		t.Error("CopyRect declined an in-range block")
	}
}
