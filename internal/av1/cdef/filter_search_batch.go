// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package cdef

// filter_search_batch.go evaluates every strength trial of the encoder's
// CDEF search for a filter block in one pass. SVT-AV1's cdef_seg_search
// (cdef_process.c) filters the block once per trial into a scratch and then
// measures that scratch against the source; the filtered rows of different
// trials share the center sample, the twelve tap samples, their differences
// and the clamp bounds, so one kernel entry per block that loops the trials
// over register-resident taps and accumulates each trial's distortion in
// place produces the same per-trial sums without the scratch copies, the
// per-row kernel entries and the distortion reloads. The per-trial results
// are exactly svt_cdef_filter_fb followed by compute_cdef_dist_8bit for that
// trial (FilterBlocksU8SubsampledTrusted plus a row-stepped SSE).

// MaxSearchTrials bounds the strength pairs one kernel call holds in its
// per-trial state; SearchBlocksU8Trusted evaluates longer lists in chunks.
const MaxSearchTrials = 16

// SearchTrial is one (primary, secondary) strength pair of the search. The
// primary strength is the frame-level value before the luma variance
// adjustment, which the kernel applies per block as svt_cdef_filter_fb does.
type SearchTrial struct {
	Primary   uint8
	Secondary uint8
}

// SearchBlocksU8Trusted computes, for each trial, the sum over the 8x8 luma
// blocks of the squared error between the source and the block filtered with
// that trial, on every rowStep-th row of each block (SVT's subsampling
// factor). A trial whose adjusted primary strength and secondary strength are
// both zero for a block measures that block unfiltered, as the search's
// untouched scratch rows do.
//
// src is the 8-bit source plane at the filter block's origin (stride
// srcStride); input is the 16-bit CDEF input buffer whose block origin is
// inputOrigin, holding the same reconstruction samples the trials filter.
// directions and variances are the block's search grids (dirinit), damping
// the luma damping. sse receives one sum per trial. Callers own the trusted
// geometry contract of FilterFrameBlocksU8Trusted: every block of blocks lies
// inside the 64x64 filter block, src holds its rows and input its halo.
//
// It returns false, leaving sse unspecified, when this build has no batched
// kernel or the geometry is outside its contract; the caller then runs the
// per-trial filter-then-measure loop.
func SearchBlocksU8Trusted(src []byte, srcStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, damping int, trials []SearchTrial, rowStep int, sse []uint64) bool {
	if rowStep < 1 || len(trials) == 0 || len(sse) < len(trials) {
		return false
	}
	for len(trials) > 0 {
		n := min(len(trials), MaxSearchTrials)
		if !searchBlocksU8(src, srcStride, input, inputOrigin, blocks, directions, variances, damping, trials[:n], rowStep, sse[:n]) {
			return false
		}
		trials, sse = trials[n:], sse[n:]
	}
	return true
}
