// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && (amd64 || arm64) && !purego)

package restoration

// init binds the pure-Go Wiener passes on every build without the Go-native SIMD
// kernels: architectures the dispatcher does not special-case, the purego
// builds, and the default (non-goexperiment.simd) amd64/arm64 builds.
func init() {
	wienerHorizontalImpl = wienerHorizontal
	wienerHorizontalTrustedImpl = wienerHorizontalTrusted
	wienerVerticalImpl = wienerVertical
	wienerHorizontalU8Impl = wienerHorizontalU8
	wienerVerticalU8Impl = wienerVerticalU8
}
