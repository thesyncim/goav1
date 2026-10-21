// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || !(amd64 || arm64) || purego

package motion

import "github.com/thesyncim/goav1/internal/av1/frame"

const scaledStep2Half = ScaleSubpelScale / 2

func convolveScale2D8Step2GoSIMD(frame.Plane, frame.Plane, int, int, int, int, int64, int64, *SubpelKernelTable, *SubpelKernelTable, int, *scaledIM) bool {
	return false
}
