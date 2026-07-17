// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package dsp

// HasAcceleratedDisjointPlaneCopy reports at compile time that the target has
// a whole-rectangle strided-copy kernel. Callers use it to avoid replacing the
// checked portable path with an indirect portable call on other targets.
const HasAcceleratedDisjointPlaneCopy = true
