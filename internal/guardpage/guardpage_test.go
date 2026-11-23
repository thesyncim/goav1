// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package guardpage

import (
	"runtime/debug"
	"testing"
	"unsafe"
)

var sink uint16

func faults(f func()) (faulted bool) {
	defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
	defer func() { faulted = recover() != nil }()
	f()
	return false
}

func TestGuards(t *testing.T) {
	for _, n := range []int{1, 7, 4096, 5000} {
		a := After[uint16](t, n)
		b := Before[uint16](t, n)
		for i := range a {
			a[i], b[i] = uint16(i), uint16(i)
		}
		if !faults(func() { sink = *(*uint16)(unsafe.Add(unsafe.Pointer(&a[n-1]), 2)) }) {
			t.Fatalf("n=%d: read past After did not fault", n)
		}
		if !faults(func() { sink = *(*uint16)(unsafe.Add(unsafe.Pointer(&b[0]), -2)) }) {
			t.Fatalf("n=%d: read before Before did not fault", n)
		}
	}
}
