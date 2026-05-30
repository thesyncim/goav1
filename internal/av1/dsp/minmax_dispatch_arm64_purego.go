// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && purego

package dsp

// init binds the pure-Go MinMaxAbsDiff8x8 variant on arm64 builds compiled
// with the purego tag, which excludes all hand-written assembly. This keeps
// the dispatch wiring symmetric: the NEON variant lives in
// minmax_dispatch_arm64.go under the matching arm64 && !purego constraint.
func init() {
	minMaxAbsDiff8x8Impl = minMaxAbsDiff8x8PureGo
}
