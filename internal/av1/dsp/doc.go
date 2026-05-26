// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package dsp defines stable pure-Go DSP reference entry points and the SIMD
// dispatch boundary.
//
// Every exported entry point that has a per-architecture variant follows the
// same pattern: a package-scope function variable holds the chosen
// implementation, an init() in a build-tagged dispatch file resolves it
// based on internal/av1/dsp/cpu, and the exported wrapper calls through the
// variable. Today only the pure-Go reference is wired in; see docs/dsp.md
// for the recipe to add an arch-specific variant.
package dsp
