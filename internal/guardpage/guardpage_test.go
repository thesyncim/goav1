// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package guardpage

import (
	"runtime/debug"
	"syscall"
	"testing"
)

var sink byte

func faults(f func()) (faulted bool) {
	defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
	defer func() { faulted = recover() != nil }()
	f()
	return false
}

// TestGuards checks that the byte just past the After window and the byte
// just before the Before window sit on PROT_NONE pages, reading them through
// the whole mapping, and that both windows themselves are accessible.
func TestGuards(t *testing.T) {
	page := syscall.Getpagesize()
	for _, n := range []int{1, 7, 4096, 5000} {
		size := 2 * n
		tail, head, mem := mapGuarded(t, size)
		for i := range tail {
			tail[i], head[i] = byte(i), byte(i)
		}
		data := len(mem) - 2*page
		if !faults(func() { sink = mem[page+data] }) {
			t.Fatalf("n=%d: read past the tail window did not fault", n)
		}
		if !faults(func() { sink = mem[page-1] }) {
			t.Fatalf("n=%d: read before the head window did not fault", n)
		}
		if &tail[len(tail)-1] != &mem[page+data-1] || &head[0] != &mem[page] {
			t.Fatalf("n=%d: windows are not against the guard pages", n)
		}
		a := After[uint16](t, n)
		b := Before[uint16](t, n)
		if len(a) != n || len(b) != n {
			t.Fatalf("n=%d: After/Before lengths %d/%d", n, len(a), len(b))
		}
		for i := range a {
			a[i], b[i] = uint16(i), uint16(i)
		}
	}
}
