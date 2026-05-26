// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !goav1_scaled_pred

package threading

// frameWorkScaledRefBuildEnabled is the compile-time half of the scaled-
// reference dispatcher's gate. The default build leaves it off so callers
// must set GOAV1_SCALED_PRED=1 to route SVC enhancement-layer prediction
// through the scaled convolver. The 7-vector fast suite has no SVC inputs,
// so its same-size translational fast path is unaffected.
const frameWorkScaledRefBuildEnabled = false
