// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package tile

import (
	"reflect"
	"testing"
	"unsafe"
)

// TestBlockModeContextResetLayout pins the invariants resetForSuperblock relies
// on: the struct is pointer-free (so a byte clear is a valid zeroing) and the
// payload grids it skips are exactly the trailing fields.
func TestBlockModeContextResetLayout(t *testing.T) {
	rt := reflect.TypeOf(BlockModeContext{})
	var hasPointers func(reflect.Type) bool
	hasPointers = func(t reflect.Type) bool {
		switch t.Kind() {
		case reflect.Pointer, reflect.Slice, reflect.Map, reflect.Chan, reflect.Func, reflect.Interface, reflect.String, reflect.UnsafePointer:
			return true
		case reflect.Array:
			return hasPointers(t.Elem())
		case reflect.Struct:
			for i := 0; i < t.NumField(); i++ {
				if hasPointers(t.Field(i).Type) {
					return true
				}
			}
		}
		return false
	}
	if hasPointers(rt) {
		t.Fatal("BlockModeContext must stay pointer-free for resetForSuperblock")
	}
	n := rt.NumField()
	if rt.Field(n-2).Name != "GridInterMotion" || rt.Field(n-1).Name != "GridInterp" {
		t.Fatalf("payload grids must be the last two fields, got %s, %s", rt.Field(n-2).Name, rt.Field(n-1).Name)
	}

	var c BlockModeContext
	b := unsafe.Slice((*byte)(unsafe.Pointer(&c)), unsafe.Sizeof(c))
	for i := range b {
		b[i] = 0xa5
	}
	c.resetForSuperblock()
	want := BlockModeContext{GridInterMotion: c.GridInterMotion, GridInterp: c.GridInterp}
	if c != want {
		t.Fatal("resetForSuperblock left a non-payload field set")
	}
}
