// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package unsafeutil_test

import (
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// TestBlock32GuardPage walks every byte of every row of 32-byte Blocks whose
// footprint ends at, or starts at, an inaccessible page: a row one byte too
// long or misplaced would fault. The views are exactly as wide as the block
// (and wider, with the block at the last column that fits).
func TestBlock32GuardPage(t *testing.T) {
	for _, d := range []struct{ stride, width, height int }{{32, 32, 14}, {40, 32, 16}, {48, 37, 4}, {32, 64, 2}} {
		n := (d.height-1)*d.stride + d.width
		for side, s := range map[string][]byte{"after": guardpage.After[byte](t, n), "before": guardpage.Before[byte](t, n)} {
			v := unsafeutil.NewRows(s, 0, d.stride, d.width, d.height)
			guardpage.Check(t, side, func() {
				for c := 0; c+32 <= d.width; c++ {
					switch d.height {
					case 14:
						b := unsafeutil.Block32At[[14]struct{}](v, 0, c)
						for k := 0; k < 14; k++ {
							walk32(b.Row32(k), k)
						}
					case 16:
						b := unsafeutil.Block32At[[16]struct{}](v, 0, c)
						for k := 0; k < 16; k++ {
							walk32(b.Row32(k), k)
						}
					case 4:
						b := unsafeutil.Block32At[[4]struct{}](v, 0, c)
						for k := 0; k < 4; k++ {
							walk32(b.Row32(k), k)
						}
					}
					// The last two rows as their own block.
					b := unsafeutil.Block32At[[2]struct{}](v, d.height-2, c)
					walk32(b.Row32(0), 0)
					walk32(b.Row32(1), 1)
				}
			})
			// Every row reads back what the walk wrote at the last column.
			c := d.width - 32
			for r := 0; r < d.height; r++ {
				if got := s[r*d.stride+c+31]; got != ^byte(31+r) && got != ^byte(31+r-d.height+2) {
					t.Fatalf("%s %+v: row %d byte 31 = %#x", side, d, r, got)
				}
			}
		}
	}
}

// walk32 reads and writes every byte of p.
func walk32(p *[32]byte, k int) {
	for i := range p {
		sink ^= p[i]
		p[i] = ^byte(i + k)
	}
}

func TestBlock32Rejects(t *testing.T) {
	s := make([]byte, 400)
	v := unsafeutil.NewRows(s, 4, 40, 36, 8) // rows at 4, 44, ..., 284, 36 bytes each
	mustPanic(t, func() { unsafeutil.Block32At[[8]struct{}](v, 1, 0) })
	mustPanic(t, func() { unsafeutil.Block32At[[2]struct{}](v, 0, 5) })
	mustPanic(t, func() { unsafeutil.Block32At[[2]struct{}](v, 0, -1) })
	mustPanic(t, func() { unsafeutil.Block32At[[2]struct{}](v, -1, 0) })
	mustPanic(t, func() { unsafeutil.Block32At[[2]struct{}](v, 7, 0) })
	// A view narrower than 32 bytes has no 32-byte block at all.
	mustPanic(t, func() { unsafeutil.Block32At[[2]struct{}](unsafeutil.NewRows(s, 0, 40, 31, 2), 0, 0) })
	b := unsafeutil.Block32At[[2]struct{}](v, 6, 4)
	mustPanic(t, func() { b.Row32(2) })
	mustPanic(t, func() { b.Row32(-1) })
	if p := b.Row32(1); &p[0] != &s[4+7*40+4] {
		t.Fatalf("Block32 row 1 is not s[%d]", 4+7*40+4)
	}
	if p := unsafeutil.Block32At[[8]struct{}](v, 0, 4).Row32(7); &p[31] != &s[4+7*40+4+31] {
		t.Fatalf("Block32 row 7 byte 31 is not s[%d]", 4+7*40+4+31)
	}
}
