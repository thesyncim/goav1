// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package unsafeutil

import (
	"encoding/binary"
	"testing"
)

func TestBytesOfUint16(t *testing.T) {
	if got := BytesOfUint16(nil); got != nil {
		t.Fatalf("nil input gave %v", got)
	}
	s := make([]uint16, 5, 9)
	for i := range s {
		s[i] = uint16(0x0102*i + 0x0a0b)
	}
	b := BytesOfUint16(s)
	if len(b) != 2*len(s) || cap(b) != 2*len(s) {
		t.Fatalf("len %d cap %d, want both %d", len(b), cap(b), 2*len(s))
	}
	for i := range s {
		if got := binary.NativeEndian.Uint16(b[2*i:]); got != s[i] {
			t.Fatalf("sample %d reads %#x through the view, want %#x", i, got, s[i])
		}
	}
	b[3] ^= 0xff
	if want := uint16(0x0102 + 0x0a0b); s[1] == want {
		t.Fatal("a write through the view did not reach the uint16 slice")
	}
}

func TestView(t *testing.T) {
	if got := View[int32](nil); got != nil {
		t.Fatalf("nil input gave %v", got)
	}
	raw := BytesOfUint16(make([]uint16, 16)) // uint16 backing: 2-byte aligned at least; heap blocks of 32 bytes are 8-aligned
	copy(raw, []byte{1, 0, 0, 0, 2, 0, 0, 0})
	v := View[int32](raw)
	if len(v) != 8 || cap(v) != 8 {
		t.Fatalf("len %d cap %d, want 8", len(v), cap(v))
	}
	if got := binary.NativeEndian.Uint32(raw[4:]); uint32(v[1]) != got {
		t.Fatalf("element 1 = %d, bytes say %d", v[1], got)
	}
	v[7] = -1
	if raw[31] != 0xff {
		t.Fatal("a write through the view did not reach the bytes")
	}
	type pair struct{ A, B int16 }
	if p := View[pair](raw[:8]); len(p) != 2 {
		t.Fatalf("struct view len %d want 2", len(p))
	}
}

func TestViewRejects(t *testing.T) {
	raw := make([]byte, 16)
	for name, f := range map[string]func(){
		"pointer element":  func() { View[*int](raw) },
		"slice element":    func() { View[[]byte](raw) },
		"struct w/ string": func() { View[struct{ S string }](raw) },
		"zero size":        func() { View[struct{}](raw) },
		"ragged length":    func() { View[int32](raw[:6]) },
		"misaligned":       func() { View[int32](raw[1:5]) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: View did not panic", name)
				}
			}()
			f()
		}()
	}
}
