// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package poison

import (
	"encoding/binary"
	"testing"
)

type inner struct {
	A uint16
	B bool
	c int8
}

type outer struct {
	X  uint32
	In [2]inner
	Y  int64
	z  bool
}

func TestLeaves(t *testing.T) {
	var v outer
	n := 0
	ro := Leaves(&v, func(i int) byte { return byte(i + 1) }, func() bool { n++; return n&1 == 1 })
	if len(ro) != 3 || ro[0] != ".In[].c" || ro[1] != ".In[].c" || ro[2] != ".z" {
		t.Fatalf("read-only leaves %q", ro)
	}
	var b [4]byte
	for i := range b {
		b[i] = byte(i + 1)
	}
	if v.X != binary.NativeEndian.Uint32(b[:]) {
		t.Fatalf("X = %#x", v.X)
	}
	if !v.In[0].B || v.In[1].B {
		t.Fatalf("bools %v %v, want alternating from true", v.In[0].B, v.In[1].B)
	}
	if v.Y == 0 || v.In[1].A == 0 || v.In[0].A == v.In[1].A {
		t.Fatalf("integers not filled from their offsets: %+v", v)
	}
	if v.In[0].c != 0 || v.z {
		t.Fatal("unexported leaves were written")
	}
}

func TestLeavesRejectsUncoveredKinds(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("a float leaf did not panic")
		}
	}()
	var v struct{ F float32 }
	Leaves(&v, func(int) byte { return 1 }, func() bool { return true })
}
