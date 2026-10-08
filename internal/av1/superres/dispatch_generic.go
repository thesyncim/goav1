// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build purego || !goexperiment.simd || (!amd64 && !arm64)

package superres

// init binds the pure-Go super-res row kernel on architectures the dispatcher
// does not special-case, on builds without GOEXPERIMENT=simd, and on purego
// builds, where the Go SIMD kernel is excluded. This file's only purpose is to
// keep the dispatch wiring symmetric.
func init() {
	upscaleRowImpl = upscaleRowPureGo
}
