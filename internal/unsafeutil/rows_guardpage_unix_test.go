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

// TestRowsGuardPage builds views whose footprint ends at, or starts at, an
// inaccessible page and reads and writes every byte of every row through
// Row8 (on the view and on sub-views): a view one byte too long or misplaced
// would fault.
func TestRowsGuardPage(t *testing.T) {
	for _, d := range []struct{ stride, width, height int }{{8, 8, 1}, {37, 16, 5}, {64, 32, 9}, {40, 33, 16}, {16, 16, 14}} {
		n := (d.height-1)*d.stride + d.width
		for side, s := range map[string][]byte{"after": guardpage.After[byte](t, n), "before": guardpage.Before[byte](t, n)} {
			if !unsafeutil.RowsFit(len(s), 0, d.stride, d.width, d.height) || unsafeutil.RowsFit(len(s)-1, 0, d.stride, d.width, d.height) {
				t.Fatalf("%s %+v: RowsFit at the exact footprint is wrong", side, d)
			}
			v := unsafeutil.NewRows(s, 0, d.stride, d.width, d.height)
			if v.Width() != d.width {
				t.Fatalf("%s %+v: width %d", side, d, v.Width())
			}
			guardpage.Check(t, side, func() {
				for r := 0; r < d.height; r++ {
					for c := 0; c+8 <= d.width; c++ {
						p := v.Row8(r, c)
						p[0], p[7] = byte(r), byte(c)
						sink ^= v.Sub(r, c, 8, 1).Row8(0, 0)[7]
					}
				}
				last := v.Sub(d.height-1, d.width-8, 8, 1)
				sink ^= last.Row8(0, 0)[7]
			})
		}
	}
}

// TestBlockGuardPage walks every row of 16-byte Blocks that end at, or start
// at, an inaccessible page through Row16, PutF32 and PutF64.
func TestBlockGuardPage(t *testing.T) {
	const stride = 24
	n := 13*stride + 16
	for side, s := range map[string][]byte{"after": guardpage.After[byte](t, n), "before": guardpage.Before[byte](t, n)} {
		v := unsafeutil.NewRows(s, 0, stride, 16, 14)
		guardpage.Check(t, side, func() {
			b := unsafeutil.BlockAt[[14]struct{}](v, 0, 0)
			for k := 0; k < 14; k++ {
				b.Row16(k)[15] = byte(k)
				b.PutF32(k, 12, math.Float32frombits(0x04030201))
				b.PutF64(k, 8, math.Float64frombits(0x0807060504030201))
			}
			for k := 0; k < 14; k++ {
				r := b.Row16(k)
				if r[8] != 1 || r[15] != 8 {
					t.Fatalf("%s: row %d = %v", side, k, r)
				}
			}
			tail := unsafeutil.BlockAt[[2]struct{}](v, 12, 0)
			sink ^= tail.Row16(1)[15]
		})
	}
}

func TestRowsRejects(t *testing.T) {
	s := make([]byte, 100)
	for _, c := range []struct{ off, stride, width, height int }{
		{-1, 10, 10, 1}, {0, -1, 10, 1}, {0, 10, 0, 1}, {0, 10, 10, 0}, {0, 10, 11, 10}, {91, 10, 10, 1},
		{0, 1 << 31, 1, 2}, {1 << 31, 0, 1, 1}, {0, 0, 1 << 31, 1}, {0, 0, 1, 1 << 31},
		{0, math.MaxInt / 2, 1, 3},
	} {
		if unsafeutil.RowsFit(len(s), c.off, c.stride, c.width, c.height) {
			t.Errorf("RowsFit(%d, %+v) = true", len(s), c)
		}
		mustPanic(t, func() { unsafeutil.NewRows(s, c.off, c.stride, c.width, c.height) })
	}
	v := unsafeutil.NewRows(s, 4, 10, 20, 8) // rows at 4, 14, ..., 74, 20 bytes each
	mustPanic(t, func() { v.Row8(8, 0) })
	mustPanic(t, func() { v.Row8(-1, 0) })
	mustPanic(t, func() { v.Row8(0, 13) })
	mustPanic(t, func() { v.Row8(0, -1) })
	mustPanic(t, func() { v.Sub(7, 0, 20, 2) })
	mustPanic(t, func() { v.Sub(0, 1, 20, 1) })
	mustPanic(t, func() { v.Sub(0, 0, 0, 1) })
	mustPanic(t, func() { v.Sub(0, 0, 1, 0) })
	mustPanic(t, func() { v.Sub(-1, 0, 1, 1) })
	mustPanic(t, func() { v.Sub(0, -1, 1, 1) })
	mustPanic(t, func() { unsafeutil.BlockAt[[8]struct{}](v, 1, 0) })
	mustPanic(t, func() { unsafeutil.BlockAt[[2]struct{}](v, 0, 5) })
	mustPanic(t, func() { unsafeutil.BlockAt[[2]struct{}](v, 0, -1) })
	mustPanic(t, func() { unsafeutil.BlockAt[[2]struct{}](v, -1, 0) })
	b := unsafeutil.BlockAt[[2]struct{}](v, 6, 4)
	mustPanic(t, func() { b.Row16(2) })
	mustPanic(t, func() { b.Row16(-1) })
	mustPanic(t, func() { b.PutF32(0, 13, 0) })
	mustPanic(t, func() { b.PutF32(2, 0, 0) })
	mustPanic(t, func() { b.PutF64(0, 9, 0) })
	mustPanic(t, func() { b.PutF64(2, 0, 0) })
	if p := b.Row16(1); &p[0] != &s[4+7*10+4] {
		t.Fatalf("Block row 1 is not s[%d]", 4+7*10+4)
	}
	if p := v.Sub(2, 3, 8, 2).Row8(1, 0); &p[0] != &s[4+3*10+3] {
		t.Fatalf("Sub row 1 is not s[%d]", 4+3*10+3)
	}
}

func mustPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("no panic")
		}
	}()
	f()
}
