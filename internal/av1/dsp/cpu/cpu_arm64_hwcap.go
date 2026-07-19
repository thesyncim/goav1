// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64

package cpu

// Auxiliary-vector (AT_HWCAP / AT_HWCAP2) decoding for arm64 optional-feature
// detection. Linux and Android expose the CPU feature bits this way; the pure
// decoding below is shared so it can be unit-tested on any arm64 host, while
// the /proc/self/auxv read lives in the linux-only detectOptionalARM64. Bit
// assignments follow arch/arm64/include/uapi/asm/hwcap.h.
const (
	atNull   = 0
	atHWCap  = 16
	atHWCap2 = 26

	// AT_HWCAP bits.
	hwcapASIMDDP = 1 << 20 // FEAT_DotProd (asimddp)
	hwcapSVE     = 1 << 22 // FEAT_SVE

	// AT_HWCAP2 bits.
	hwcap2SVE2 = 1 << 1  // FEAT_SVE2
	hwcap2I8MM = 1 << 13 // FEAT_I8MM
)

// parseAuxvHWCap scans a little-endian arm64 auxv image — pairs of uint64
// (tag, value) terminated by an AT_NULL tag — and returns the AT_HWCAP and
// AT_HWCAP2 values (0 when absent).
func parseAuxvHWCap(data []byte) (hwcap, hwcap2 uint64) {
	for i := 0; i+16 <= len(data); i += 16 {
		tag := le64(data[i:])
		val := le64(data[i+8:])
		switch tag {
		case atNull:
			return
		case atHWCap:
			hwcap = val
		case atHWCap2:
			hwcap2 = val
		}
	}
	return
}

// applyARM64HWCap sets the optional-feature fields from decoded hwcap words.
// NEON is mandatory on the ARMv8-A baseline and is set by init separately.
func applyARM64HWCap(f *Features, hwcap, hwcap2 uint64) {
	f.DOTPROD = hwcap&hwcapASIMDDP != 0
	f.SVE = hwcap&hwcapSVE != 0
	f.I8MM = hwcap2&hwcap2I8MM != 0
	f.SVE2 = hwcap2&hwcap2SVE2 != 0
}

func le64(b []byte) uint64 {
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}
