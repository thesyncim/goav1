// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package poison fills test values with recognisable non-zero patterns
// through reflection, the safe replacement for writing a byte pattern over a
// struct's memory: every leaf gets a valid value (bools stay 0 or 1) and a
// field added later is covered automatically.
package poison

import (
	"encoding/binary"
	"fmt"
	"reflect"
)

// Leaves sets every leaf of *p in field order. An integer leaf takes the
// bytes byteAt(i) for each byte offset i it occupies within *p, assembled in
// host byte order, so the result matches writing byteAt over the memory of
// the integer fields. A bool leaf takes boolAt(). Leaves reflection cannot
// set (unexported fields) are left alone and their paths, such as
// ".inner[3].flag" with array indices elided to "[]", are returned so the
// caller can set them directly or fail. Leaves of other kinds (pointers,
// floats, strings, ...) panic: the pattern does not cover them.
func Leaves[T any](p *T, byteAt func(i int) byte, boolAt func() bool) (readOnly []string) {
	return LeavesWith(p, byteAt, boolAt, nil)
}

// LeavesWith is Leaves with a hook: custom is offered every value first
// (structs and arrays included) and, when it returns true, the value counts
// as poisoned and is not descended into. It lets a caller fill a type whose
// fields are unexported through that type's own exported API.
func LeavesWith[T any](p *T, byteAt func(i int) byte, boolAt func() bool, custom func(v reflect.Value) bool) (readOnly []string) {
	w := walker{byteAt: byteAt, boolAt: boolAt, custom: custom}
	w.walk(reflect.ValueOf(p).Elem(), 0, "")
	return w.readOnly
}

type walker struct {
	byteAt   func(int) byte
	boolAt   func() bool
	custom   func(reflect.Value) bool
	readOnly []string
}

func (w *walker) walk(v reflect.Value, off uintptr, path string) {
	if w.custom != nil && w.custom(v) {
		return
	}
	switch v.Kind() {
	case reflect.Array:
		size := v.Type().Elem().Size()
		for i := range v.Len() {
			w.walk(v.Index(i), off+uintptr(i)*size, path+"[]")
		}
		return
	case reflect.Struct:
		for i := range v.NumField() {
			f := v.Type().Field(i)
			w.walk(v.Field(i), off+f.Offset, path+"."+f.Name)
		}
		return
	}
	if !v.CanSet() {
		w.readOnly = append(w.readOnly, path)
		return
	}
	var raw [8]byte
	size := int(v.Type().Size())
	for k := range min(size, len(raw)) {
		raw[k] = w.byteAt(int(off) + k)
	}
	var bits uint64
	switch size {
	case 1:
		bits = uint64(raw[0])
	case 2:
		bits = uint64(binary.NativeEndian.Uint16(raw[:]))
	case 4:
		bits = uint64(binary.NativeEndian.Uint32(raw[:]))
	case 8:
		bits = binary.NativeEndian.Uint64(raw[:])
	}
	switch v.Kind() {
	case reflect.Bool:
		v.SetBool(w.boolAt())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(int64(bits))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		v.SetUint(bits)
	default:
		panic(fmt.Sprintf("poison: %s has kind %v, which the pattern does not cover", path, v.Kind()))
	}
}
