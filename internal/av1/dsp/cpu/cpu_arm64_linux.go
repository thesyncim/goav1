// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && linux

package cpu

import "os"

// detectOptionalARM64 fills DOTPROD/I8MM/SVE/SVE2 from the auxiliary vector.
// Linux (and Android, which shares the kernel ABI) publishes the arm64 feature
// bits as AT_HWCAP / AT_HWCAP2 entries in /proc/self/auxv. Reading the file
// directly keeps the package free of cgo and third-party dependencies — the
// same fallback golang.org/x/sys/cpu uses when a vDSO getauxval is absent.
func detectOptionalARM64(f *Features) {
	data, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		// No auxv (unusual sandbox): keep the mandatory-NEON-only path.
		return
	}
	hwcap, hwcap2 := parseAuxvHWCap(data)
	applyARM64HWCap(f, hwcap, hwcap2)
}
