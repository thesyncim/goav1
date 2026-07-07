// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build !arm64 || purego || goav1_trace_rng

package entropy

import "unsafe"

// HasMVResidual reports that the arm64 motion-vector residual kernel is built
// in. Callers must keep the pure-Go chain as the dispatch fallback.
const HasMVResidual = false

// MVResidual is the no-kernel stub; it must never be called (dispatch is
// gated on HasMVResidual).
func (c *Cursor) MVResidual(mvCDFs unsafe.Pointer, precision int64) (joint, comp0, comp1 int64) {
	return 0, 0, 0
}
