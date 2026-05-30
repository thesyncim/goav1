// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

// filterBlockImpl is the dispatch slot for the CDEF per-block filter kernel.
// It is resolved exactly once, at package init (see filter_dispatch_*.go), so
// the per-call cost is a single indirect call with no feature-detection
// branches. Tests and benchmarks must not mutate this variable concurrently
// with live decoding.
//
// filterBlockPureGo in filter.go is the canonical bit-exact reference. Every
// tuned variant MUST match it sample for sample, including the constrain()
// clamp, the rounding term, and the CDEF_VERY_LARGE border handling.
var filterBlockImpl = filterBlockPureGo
