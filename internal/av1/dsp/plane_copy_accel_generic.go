// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !arm64 || purego

package dsp

// HasAcceleratedDisjointPlaneCopy reports at compile time whether the target
// has a whole-rectangle strided-copy kernel.
const HasAcceleratedDisjointPlaneCopy = false
