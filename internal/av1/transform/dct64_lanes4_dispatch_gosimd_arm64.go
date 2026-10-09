// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the Go-native SIMD four-lane DCT64 column and row kernels. Every
// arm64 chip has NEON; the gate lets tests force the pure-Go path.
func init() {
	if cpu.Detected.NEON {
		inverseDCT64Col4Impl = inverseDCT64Col4SIMD
		inverseDCT64Row4Impl = inverseDCT64Row4SIMD
	}
}
