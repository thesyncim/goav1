// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package cpu reports which SIMD instruction sets are usable on the current
// process. It is intended for goav1's DSP dispatch layer: hot kernels resolve
// their implementation variant at package init, the dispatcher reads the flags
// exported here, and the per-call code path becomes a single indirect call
// with zero feature-detection overhead.
//
// Detection uses only public, official queries and the package contains no
// assembly. On amd64 builds with GOEXPERIMENT=simd the flags come from
// simd/archsimd.X86, which already checks OS register-state support. Other
// amd64 builds, and purego builds, report only the SSE2 baseline and leave the
// higher flags false. arm64 uses the mandatory NEON baseline plus platform
// probes. The feature flags exposed by this package are the union of what the
// dispatch layer currently cares about — adding a new flag is a one-line
// addition once the underlying detection is wired in.
//
// Forwards compatibility: callers must treat these flags as advisory. Any
// SIMD variant a dispatcher selects MUST produce bit-exact output relative
// to the pure-Go reference; see ARCHITECTURE.md ("Generic Go reference
// first, SIMD parity second").
package cpu
