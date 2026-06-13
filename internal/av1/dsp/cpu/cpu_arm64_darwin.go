// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && darwin

package cpu

import "syscall"

func detectOptionalARM64(f *Features) {
	f.DOTPROD = sysctlOptionalARMFeature("hw.optional.arm.FEAT_DotProd")
	f.I8MM = sysctlOptionalARMFeature("hw.optional.arm.FEAT_I8MM")
	f.SVE = sysctlOptionalARMFeature("hw.optional.arm.FEAT_SVE")
	f.SVE2 = sysctlOptionalARMFeature("hw.optional.arm.FEAT_SVE2")
}

func sysctlOptionalARMFeature(name string) bool {
	v, err := syscall.SysctlUint32(name)
	return err == nil && v != 0
}
