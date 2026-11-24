// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

import "testing"

func TestSlicesOverlap(t *testing.T) {
	buf := make([]uint16, 64)
	other := make([]uint16, 64)
	for _, tc := range []struct {
		name string
		a, b []uint16
		want bool
	}{
		{"same", buf, buf, true},
		{"disjoint-adjacent", buf[:10], buf[10:], false},
		{"touching", buf[:11], buf[10:20], true},
		{"inside", buf[5:50], buf[20:30], true},
		{"inside-rev", buf[20:30], buf[5:50], true},
		{"gap", buf[0:10], buf[30:40], false},
		{"other-array", buf, other, false},
		{"empty", buf[:0], buf, false},
		{"end", buf[63:], buf[60:], true},
	} {
		if got := SlicesOverlap(tc.a, tc.b); got != tc.want {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
	}
}
