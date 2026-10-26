// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !(goexperiment.simd && amd64 && !purego)

package prediction

func filterIntraEdgeArch(edge []uint16, strength uint8) bool { return false }

func upsampleIntraEdgeArch(edge []uint16, origin, size int, max uint16) bool { return false }

func predictDirectionalZ1Arch(block planeBlock, above []uint16, upsampleAbove int, dx int, maxBaseX int) bool {
	return false
}

func predictDirectionalZ3Arch(block planeBlock, left []uint16, upsampleLeft int, dy int, maxBaseY int) bool {
	return false
}
