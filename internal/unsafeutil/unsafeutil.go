// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package unsafeutil is the only non-test package in goav1 that imports
// unsafe (scripts/check_unsafe.sh enforces it). Each function takes and
// returns ordinary slices, checks its preconditions at entry, and keeps the
// pointer arithmetic inside; its doc comment carries the bounds proof. Add a
// function here only with that proof and a test that exercises it against a
// guard page.
package unsafeutil

import (
	"fmt"
	"reflect"
	"unsafe"
)

// BytesOfUint16 returns the bytes of s's backing array, 2*len(s) of them, in
// memory order (so the byte order of each sample is the host's).
//
// Bounds proof: the result starts at &s[0] and covers exactly the
// len(s)*2 bytes that s's elements occupy, so every byte it can address lies
// inside s's allocation. byte has alignment 1, so any address is valid, and
// neither type holds pointers, so the garbage collector sees nothing new.
// The capacity is 2*len(s), not 2*cap(s): a reslice of the result cannot
// reach past s's length. An empty s yields nil.
func BytesOfUint16(s []uint16) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 2*len(s))
}

// View returns b reinterpreted as len(b)/size(T) elements of T. It panics
// unless T is pointer-free, has a non-zero size, len(b) is a multiple of
// T's size and b's first byte is aligned for T.
//
// Bounds proof: the result starts at &b[0] and its len(b)/size(T) elements
// cover exactly len(b) bytes, all inside b; the capacity equals the length,
// so a reslice cannot reach past b. The alignment check makes every element
// address valid for T. T holding no pointers means no byte pattern written
// through b can become a pointer the garbage collector would follow, and the
// memory b came from (heap, stack or an mmap'd region) needs no pointer
// metadata. An empty b yields nil.
func View[T any](b []byte) []T {
	t := reflect.TypeFor[T]()
	size := int(t.Size())
	if size == 0 || hasPointers(t) {
		panic(fmt.Sprintf("unsafeutil.View: %v is zero-sized or holds pointers", t))
	}
	if len(b)%size != 0 {
		panic(fmt.Sprintf("unsafeutil.View: %d bytes is not a whole number of %v", len(b), t))
	}
	if len(b) == 0 {
		return nil
	}
	p := unsafe.Pointer(unsafe.SliceData(b))
	if uintptr(p)%uintptr(t.Align()) != 0 {
		panic(fmt.Sprintf("unsafeutil.View: %p is not aligned for %v", p, t))
	}
	return unsafe.Slice((*T)(p), len(b)/size)
}

// hasPointers reports whether a value of type t contains any pointer the
// garbage collector traces.
func hasPointers(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.Complex64, reflect.Complex128:
		return false
	case reflect.Array:
		return t.Len() > 0 && hasPointers(t.Elem())
	case reflect.Struct:
		for i := range t.NumField() {
			if hasPointers(t.Field(i).Type) {
				return true
			}
		}
		return false
	}
	return true
}
