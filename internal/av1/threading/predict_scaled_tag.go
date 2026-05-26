// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goav1_scaled_pred

package threading

// frameWorkScaledRefBuildEnabled is the compile-time half of the scaled-
// reference dispatcher's gate. Set when the goav1_scaled_pred build tag is on,
// pinning the scaled-prediction path live for the entire process.
const frameWorkScaledRefBuildEnabled = true
