// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && purego

package quantize

// init binds the pure-Go dequant column variant on amd64 purego builds, where
// the AVX2 assembly kernel is excluded. The non-purego amd64 build binds the
// AVX2 variant from dequant_avx2_amd64.go when CPUID reports OS-enabled AVX2.
// This file keeps the dispatch wiring symmetric across architectures.
func init() {
	dequantColumnImpl = dequantColumnPureGo
}
