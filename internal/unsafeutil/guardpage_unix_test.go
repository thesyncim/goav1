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

var sink byte

// TestBytesOfUint16GuardPage walks every byte of the view of a uint16 slice
// that ends, or starts, at an inaccessible page: a view one byte too long or
// shifted would fault.
func TestBytesOfUint16GuardPage(t *testing.T) {
	for _, n := range []int{1, 7, 2048, 2049} {
		for side, s := range map[string][]uint16{"after": guardpage.After[uint16](t, n), "before": guardpage.Before[uint16](t, n)} {
			for i := range s {
				s[i] = uint16(i)
			}
			b := unsafeutil.BytesOfUint16(s)
			if len(b) != 2*n {
				t.Fatalf("%s n=%d: len %d", side, n, len(b))
			}
			guardpage.Check(t, side, func() {
				for i := range b {
					sink ^= b[i]
					b[i] = ^b[i]
				}
			})
			for i := range s {
				if s[i] != ^uint16(i) {
					t.Fatalf("%s n=%d: sample %d = %#x", side, n, i, s[i])
				}
			}
		}
	}
}

// TestViewGuardPage reads and writes every element of a typed view whose
// bytes end, or start, at an inaccessible page.
func TestViewGuardPage(t *testing.T) {
	for _, n := range []int{1, 3, 1024, 1025} {
		for side, b := range map[string][]byte{"after": guardpage.After[byte](t, 4*n), "before": guardpage.Before[byte](t, 4*n)} {
			v := unsafeutil.View[int32](b)
			if len(v) != n {
				t.Fatalf("%s n=%d: len %d", side, n, len(v))
			}
			guardpage.Check(t, side, func() {
				for i := range v {
					v[i] = int32(i) - 7
				}
				for i := range v {
					if v[i] != int32(i)-7 {
						t.Fatalf("%s n=%d: element %d = %d", side, n, i, v[i])
					}
				}
			})
		}
	}
}
