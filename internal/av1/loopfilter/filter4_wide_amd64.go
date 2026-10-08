// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

func lfFilter4U8Wide(pix []byte, q0Base, step, length int, params filter4Params, changed *uint) {
	m := lfFilter4Core[uint8](pix, q0Base, step, length, 1, params)
	if changed != nil {
		*changed |= m
	}
}
