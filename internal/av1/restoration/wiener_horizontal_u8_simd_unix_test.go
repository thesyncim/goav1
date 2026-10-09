// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build (darwin || linux) && goexperiment.simd && (arm64 || amd64) && !purego

package restoration

import (
	"os"
	"slices"
	"syscall"
	"testing"
)

func TestWienerHorizontalU8SIMDShortBorderDoesNotReadPastAllocation(t *testing.T) {
	const width, height = 16, 5
	stride := width + 2*WienerHalfwin
	srcLen := stride * (height + 2*WienerHalfwin)
	pageSize := os.Getpagesize()
	if srcLen >= pageSize {
		t.Fatalf("fixture length %d exceeds one page (%d)", srcLen, pageSize)
	}
	mapped, err := syscall.Mmap(-1, 0, 2*pageSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Skipf("mmap unavailable: %v", err)
	}
	defer syscall.Munmap(mapped)
	if err := syscall.Mprotect(mapped[pageSize:], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect unavailable: %v", err)
	}
	src := mapped[pageSize-srcLen : pageSize]
	origin := WienerHalfwin*stride + WienerHalfwin
	for i := range src {
		src[i] = uint8((i*53 + i/stride*7) & 0xff)
	}
	if wienerHorizontalU8SIMDCanRun(len(src), stride, origin, width, height) {
		t.Fatal("SIMD vector guard accepted a row with only the scalar border")
	}
	round0, _ := wienerRounds(8)
	filter := DefaultWienerInfo().HFilter
	want := make([]uint16, width*(height+2*WienerHalfwin))
	got := make([]uint16, len(want))
	wienerHorizontalU8(src, stride, origin, width, height, filter, round0, want)
	wienerHorizontalU8SIMDChecked(src, stride, origin, width, height, filter, round0, got)
	if !slices.Equal(got, want) {
		t.Fatal("SIMD wrapper differs from scalar output for the minimal-border allocation")
	}
}
