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
// inaccessible page and reads and writes every byte through every accessor:
// a view one byte too long or misplaced would fault.
func TestRowsGuardPage(t *testing.T) {
	for _, d := range []struct{ stride, width, height int }{{16, 16, 1}, {37, 16, 5}, {64, 32, 9}, {40, 33, 16}, {16, 16, 14}} {
		n := (d.height-1)*d.stride + d.width
		for side, s := range map[string][]byte{"after": guardpage.After[byte](t, n), "before": guardpage.Before[byte](t, n)} {
			if !unsafeutil.RowsFit(len(s), 0, d.stride, d.width, d.height) || unsafeutil.RowsFit(len(s)-1, 0, d.stride, d.width, d.height) {
				t.Fatalf("%s %+v: RowsFit at the exact footprint is wrong", side, d)
			}
			v := unsafeutil.NewRows(s, 0, d.stride, d.width, d.height)
			guardpage.Check(t, side, func() {
				for r := 0; r < v.Height(); r++ {
					for c := 0; c+4 <= v.Width(); c++ {
						p := v.Row4(r, c)
						p[0], p[3] = byte(r), byte(c)
					}
					for c := 0; c+8 <= v.Width(); c++ {
						sink ^= v.Row8(r, c)[7]
					}
					for c := 0; c+16 <= v.Width(); c++ {
						sink ^= v.Row16(r, c)[15]
					}
					for c := 0; c+24 <= v.Width(); c++ {
						sink ^= v.Row24(r, c)[23]
					}
					for c := 0; c+32 <= v.Width(); c++ {
						sink ^= v.Row32(r, c)[31]
					}
				}
				last := v.Sub(v.Height()-1, v.Width()-4, 4, 1)
				sink ^= last.Row4(0, 0)[3]
			})
		}
	}
}

// TestBlockGuardPage walks every row of 16-byte Blocks that end at, or start
// at, an inaccessible page through Row16, Row8, Row4, PutF32 and PutF64.
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
				sink ^= b.Row8(k, 8)[7] ^ b.Row4(k, 12)[3]
			}
			for k := 0; k < 14; k++ {
				r := b.Row16(k)
				if r[8] != 1 || r[15] != 8 {
					t.Fatalf("%s: row %d = %v", side, k, r)
				}
			}
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
	v := unsafeutil.NewRows(s, 4, 10, 20, 8) // rows 4..23, 14..33, ..., 74..93
	mustPanic(t, func() { v.Row4(8, 0) })
	mustPanic(t, func() { v.Row4(-1, 0) })
	mustPanic(t, func() { v.Row4(0, 17) })
	mustPanic(t, func() { v.Row4(0, -1) })
	mustPanic(t, func() { v.Row32(0, 0) })
	mustPanic(t, func() { v.Sub(7, 0, 20, 2) })
	mustPanic(t, func() { v.Sub(0, 1, 20, 1) })
	mustPanic(t, func() { v.Sub(0, 0, 0, 1) })
	mustPanic(t, func() { v.Sub(-1, 0, 1, 1) })
	mustPanic(t, func() { unsafeutil.BlockAt[[8]struct{}](v, 1, 0) })
	mustPanic(t, func() { unsafeutil.BlockAt[[2]struct{}](v, 0, 5) })
	mustPanic(t, func() { unsafeutil.BlockAt[[2]struct{}](v, 0, -1) })
	b := unsafeutil.BlockAt[[2]struct{}](v, 6, 4)
	mustPanic(t, func() { b.Row16(2) })
	mustPanic(t, func() { b.Row8(0, 9) })
	mustPanic(t, func() { b.Row4(1, 13) })
	mustPanic(t, func() { b.PutF32(0, 13, 0) })
	mustPanic(t, func() { b.PutF64(2, 0, 0) })
	if p := b.Row16(1); &p[0] != &s[4+7*10+4] {
		t.Fatalf("Block row 1 is not s[%d]", 4+7*10+4)
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
