// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego && (darwin || linux)

package transform

import (
	"syscall"
	"testing"
	"unsafe"
)

func TestForwardDCT4x4SIMDDoesNotReadPastResidual(t *testing.T) {
	pageSize := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 2*pageSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Skipf("mmap unavailable: %v", err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(mem); err != nil {
			t.Errorf("munmap: %v", err)
		}
	})
	if err := syscall.Mprotect(mem[pageSize:], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect unavailable: %v", err)
	}

	// Place exactly 16 int16 samples at the end of the readable page. A full
	// 8-lane load from the last row crosses into the inaccessible page.
	input := mem[pageSize-16*2 : pageSize]
	residual := unsafe.Slice((*int16)(unsafe.Pointer(&input[0])), 16)
	for i := range residual {
		residual[i] = int16(i*17 - 128)
	}
	var want, got [16]int32
	forwardDCT4x4PureGo(want[:], 4, residual, 4)
	forwardDCT4x4SIMD(got[:], 4, residual, 4)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("coeff[%d]=%d want %d", i, got[i], want[i])
		}
	}
}
