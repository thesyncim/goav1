// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

// filterBlockU8Impl is the dispatch slot for the 8-bit-dst CDEF per-block
// filter kernel, resolved once at package init exactly like filterBlockImpl
// (see filter_dispatch.go). filterBlockU8PureGo in filter_u8.go is the
// canonical bit-exact reference; every tuned variant MUST narrow to the same
// bytes, including the constrain() clamp, the rounding term, and the 0x4000
// VeryLarge border handling.
var filterBlockU8Impl = filterBlockU8PureGo

// The per-unit 8-bit block loop (filterUnitBlocksU8) dispatches by build tag
// (filter_u8_unit_generic.go / filter_u8_neon_arm64.go) for the same
// escape-analysis reason as filterUnitBlocks: a func-variable call would leak
// the decoder's stack-resident CDEF scratch to the heap.
