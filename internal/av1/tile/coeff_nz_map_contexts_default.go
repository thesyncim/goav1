// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build (!arm64 && !amd64) || purego

package tile

import "github.com/thesyncim/goav1/internal/av1/transform"

func coeffNZMapContextsArch(levels []uint8, size TransformSize, class transform.Class, scan []int16, eob int, contexts []int8, maxEOB int) bool {
	return false
}

func coeffNZMapContexts2DFullArch(levels []uint8, size TransformSize, contexts []int8) bool {
	return false
}
