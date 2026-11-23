// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goexperiment.simd || (!arm64 && !amd64) || purego

package decoder

func copyBytes68(dst, src []byte) { copy(dst[:68], src[:68]) }
func copyBytes66(dst, src []byte) { copy(dst[:66], src[:66]) }
func copyBytes36(dst, src []byte) { copy(dst[:36], src[:36]) }
func copyBytes34(dst, src []byte) { copy(dst[:34], src[:34]) }
