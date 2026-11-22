// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

// Package guardpage allocates test buffers that sit directly against
// inaccessible pages, so a kernel that reads or writes outside the region it
// was handed faults instead of silently touching neighbouring memory.
package guardpage

import (
	"runtime/debug"
	"syscall"
	"testing"
	"unsafe"
)

// After returns n elements whose last byte is followed by a PROT_NONE page.
func After[T any](tb testing.TB, n int) []T {
	b, _ := mapGuarded(tb, n*int(unsafe.Sizeof(*new(T))))
	return cast[T](b, n)
}

// Before returns n elements whose first byte follows a PROT_NONE page.
func Before[T any](tb testing.TB, n int) []T {
	_, b := mapGuarded(tb, n*int(unsafe.Sizeof(*new(T))))
	return cast[T](b, n)
}

// Check runs f and fails tb with the fault address if f touches a guard page.
func Check(tb testing.TB, name string, f func()) {
	tb.Helper()
	defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(interface{ Addr() uintptr }); ok {
				tb.Fatalf("%s: access outside the buffer at %#x", name, e.Addr())
			}
			panic(r)
		}
	}()
	f()
}

// mapGuarded maps size usable bytes between two PROT_NONE pages and returns
// the window ending at the trailing guard and the one starting at the
// leading guard.
func mapGuarded(tb testing.TB, size int) (tail, head []byte) {
	tb.Helper()
	page := syscall.Getpagesize()
	data := (size + page - 1) / page * page
	if data == 0 {
		data = page
	}
	mem, err := syscall.Mmap(-1, 0, data+2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		tb.Skipf("mmap: %v", err)
	}
	tb.Cleanup(func() {
		if err := syscall.Munmap(mem); err != nil {
			tb.Errorf("munmap: %v", err)
		}
	})
	if err := syscall.Mprotect(mem[:page], syscall.PROT_NONE); err != nil {
		tb.Skipf("mprotect: %v", err)
	}
	if err := syscall.Mprotect(mem[page+data:], syscall.PROT_NONE); err != nil {
		tb.Skipf("mprotect: %v", err)
	}
	return mem[page+data-size : page+data : page+data], mem[page : page+size : page+size]
}

func cast[T any](b []byte, n int) []T {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(unsafe.SliceData(b))), n)
}
