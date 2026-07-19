// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !darwin && !linux

package cpu

// Platforms without a supported optional-feature probe (e.g. the BSDs) keep
// the ARMv8-A mandatory NEON baseline only; Darwin (sysctl) and Linux/Android
// (/proc/self/auxv) fill in DOTPROD/I8MM/SVE via their own files.
func detectOptionalARM64(*Features) {}
