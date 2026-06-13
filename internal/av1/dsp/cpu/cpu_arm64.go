// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64

package cpu

// init populates Detected on arm64 builds.
//
// The ARMv8-A baseline (which is what Go's arm64 port targets) makes the
// "Advanced SIMD" extension, i.e. NEON, mandatory. Optional extensions are
// filled by OS-specific helpers when the platform exposes stable feature bits.
func init() {
	Detected.NEON = true
	detectOptionalARM64(&Detected)
}
