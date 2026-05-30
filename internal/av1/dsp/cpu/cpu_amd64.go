// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package cpu

// cpuid and xgetbv are thin CPUID / XGETBV wrappers implemented in
// cpuid_amd64.s. They let us detect the SIMD feature level at init without
// depending on golang.org/x/sys/cpu.
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax, edx uint32)

// init populates Detected on amd64 builds via CPUID.
//
// The Go toolchain guarantees a working SSE2 unit on every amd64 target
// (GOAMD64=v1 baseline), so SSE2 is reported unconditionally. AVX/AVX2 are
// gated on both the CPUID feature bits and OS support for the YMM register
// state (OSXSAVE + XCR0[2:1]), matching the standard runtime check. On hosts
// that do not advertise AVX2 — notably Rosetta 2, which translates AVX2 but
// reports it absent in CPUID — AVX2 stays false and the transform dispatch
// keeps its pure-Go (bit-exact) kernels.
func init() {
	Detected.SSE2 = true

	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 1 {
		return
	}
	_, _, ecx1, _ := cpuid(1, 0)
	const (
		bitOSXSAVE = 1 << 27
		bitAVX     = 1 << 28
	)
	hasOSXSAVE := ecx1&bitOSXSAVE != 0
	hasAVX := ecx1&bitAVX != 0
	if !hasOSXSAVE || !hasAVX {
		return
	}
	// XCR0 bits 1 (SSE/XMM) and 2 (AVX/YMM) must both be enabled by the OS for
	// AVX state to be usable.
	xcr0, _ := xgetbv()
	if xcr0&0x6 != 0x6 {
		return
	}
	if maxLeaf < 7 {
		return
	}
	_, ebx7, _, _ := cpuid(7, 0)
	const (
		bitAVX2    = 1 << 5
		bitAVX512F = 1 << 16
	)
	Detected.AVX2 = ebx7&bitAVX2 != 0
	// AVX-512 additionally needs the OS to enable the opmask/ZMM state
	// (XCR0 bits 5,6,7); guard on that before reporting it.
	if ebx7&bitAVX512F != 0 && xcr0&0xe0 == 0xe0 {
		Detected.AVX512 = true
	}
}
