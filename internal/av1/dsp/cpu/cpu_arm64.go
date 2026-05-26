// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64

package cpu

// init populates Detected on arm64 builds.
//
// The ARMv8-A baseline (which is what Go's arm64 port targets) makes the
// "Advanced SIMD" extension — i.e. NEON — mandatory. Every arm64 chip Go
// runs on therefore supports NEON, so we can report it unconditionally
// without consulting the OS. SVE is optional and currently absent from
// Apple silicon and most server cores Go ships to; detecting it requires
// reading HWCAP2 on Linux / sysctl on Darwin, which is out of scope for
// this scaffold and would pull in OS-specific code.
//
// TODO: add SVE detection (Linux: /proc/self/auxv HWCAP2_SVE bit; Darwin:
// hw.optional.arm.FEAT_SVE via sysctlbyname). Once added, kernel variants
// can opt in by checking Detected.SVE.
func init() {
	Detected.NEON = true
}
