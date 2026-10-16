package encoder

import "github.com/thesyncim/goav1/internal/av1/motion"

// pred_stats.go lets the inter block's luma prediction statistics travel
// between the stages that compute them. The MDS0 candidate loop, the
// interpolation filter search and the transform decision each take
// realtimeInterResidualSSEVariance of a luma prediction of the same block;
// for a given vector and filter pair those statistics are fixed, so a later
// stage can reuse an earlier stage's result instead of rescanning the block.

// lumaPredStats is the most recent (vector, filters) -> (sse, variance)
// result for the current block.
type lumaPredStats struct {
	valid    bool
	mv       motion.Vector
	filters  motion.InterpFilters
	sse      uint32
	variance uint32
}

func (s *lumaPredStats) set(mv motion.Vector, filters motion.InterpFilters, sse, variance uint32) {
	*s = lumaPredStats{valid: true, mv: mv, filters: filters, sse: sse, variance: variance}
}

// lookup returns the statistics of the prediction for (mv, filters); a
// full-pel vector's prediction does not depend on the filters.
func (s *lumaPredStats) lookup(mv motion.Vector, filters motion.InterpFilters) (sse, variance uint32, ok bool) {
	if !s.valid || s.mv != mv {
		return 0, 0, false
	}
	if s.filters != filters && (mv.Row&7 != 0 || mv.Col&7 != 0) {
		return 0, 0, false
	}
	return s.sse, s.variance, true
}

// lumaPredKey names the prediction predY holds when the transform decision
// runs (valid only for an unscaled single-reference block).
type lumaPredKey struct {
	valid   bool
	mv      motion.Vector
	filters motion.InterpFilters
}
