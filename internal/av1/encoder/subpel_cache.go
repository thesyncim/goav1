package encoder

// subpelProbeCache is the lane-owned motion-search scratch: the fast luma
// predictor's 2-D intermediate (pred8.go) and the SVT light-PD1 sub-pel
// search's bilinear buffers (subpel_svt.go). Wavefront lanes keep their own
// copy (pframe_wavefront.go adopt).
type subpelProbeCache struct {
	scratch [pred8IMLen]int16
	svt     svtSubpelScratch
}

func (st *lossyEncodeState) ensureSubpelProbeCache() {
	if st.subpelCache == nil {
		st.subpelCache = new(subpelProbeCache)
	}
}
