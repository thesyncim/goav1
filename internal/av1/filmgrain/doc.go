// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package filmgrain implements AV1 film grain synthesis (spec §7.18.3).
//
// Film grain is applied as a normative post-processing step on the decoded,
// upscaled output frame using the per-frame film_grain_params parsed elsewhere.
// The pipeline mirrors libaom's av1/decoder/grain_synthesis.c: it generates a
// deterministic Gaussian noise template from the AR coefficients, derives the
// chroma grain from the luma template, applies a piecewise-linear scaling
// lookup keyed on the underlying sample value, and blends the scaled grain back
// into luma and chroma with an overlap region between grain blocks.
//
// The package is split by stage (random, gaussian, scaling, luma, chroma,
// blend, sample, apply) so each step can be tested against reference output in
// isolation. All routines operate on caller-provided buffers and allocate no
// per-pixel state.
package filmgrain
