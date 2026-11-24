// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

// Package guardpage allocates test buffers that sit directly against
// inaccessible pages, so a kernel that reads or writes outside the region it
// was handed faults instead of silently touching neighbouring memory.
package guardpage

import (
	"reflect"
	"runtime/debug"
	"syscall"
	"testing"

	"github.com/thesyncim/goav1/internal/unsafeutil"
)

// After returns n elements whose last byte is followed by a PROT_NONE page.
func After[T any](tb testing.TB, n int) []T {
	b, _, _ := mapGuarded(tb, n*sizeOf[T]())
	return unsafeutil.View[T](b)
}

// Before returns n elements whose first byte follows a PROT_NONE page.
func Before[T any](tb testing.TB, n int) []T {
	_, b, _ := mapGuarded(tb, n*sizeOf[T]())
	return unsafeutil.View[T](b)
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

func sizeOf[T any]() int { return int(reflect.TypeFor[T]().Size()) }

// mapGuarded maps size usable bytes between two PROT_NONE pages and returns
// the window ending at the trailing guard, the one starting at the leading
// guard, and the whole mapping including both guard pages. The windows
// start aligned for any element type whose size divides size: tail ends on
// a page boundary and head starts on one.
func mapGuarded(tb testing.TB, size int) (tail, head, mem []byte) {
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
	return mem[page+data-size : page+data : page+data], mem[page : page+size : page+size], mem
}
