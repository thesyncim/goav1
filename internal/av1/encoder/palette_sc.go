package encoder

import (
	"slices"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// palette_sc.go is SVT-AV1's luma palette search for the screen-content
// keyframes of preset 12 rtc (svt_aom_sig_deriv_multi_processes_rtc:
// palette_level = is_islice ? 3 : 0 for sc_class1), ported from
// Source/Lib/Codec/palette.c search_palette_luma and palette_rd_y with
// k_means_template.h:
//
//   - set_palette_level 3: dominant_color_step ~0 (no dominant-color
//     candidates), kmean_color_step 1, centroid_refinement 0.
//   - Blocks with 2..64 luma colors get one k-means candidate per palette
//     size n = min(colors, 8) down to 2: centroids seeded evenly between the
//     block's min and max (the two extremes when the block has exactly two
//     colors), up to 50 k-means iterations, then palette_rd_y: centroids
//     within (6 + (qindex >> 6)) of a cached neighbour color snap to it
//     (optimize_palette_colors), duplicates are removed, the indices are
//     recomputed against the final colors and the map extended past the
//     on-screen part.
//
// The keyframe coder prices each candidate like its intra modes (trial luma
// transform cost plus the palette syntax rate) and keeps the cheapest.

const paletteMaxCands = tile.PaletteMaxSize - tile.PaletteMinSize + 1

// paletteCand is one luma palette candidate: its colors and the bw x bh
// color index map (stride bw).
type paletteCand struct {
	res tile.PaletteModeResult
	m   [tile.MaxPalettePixels]uint8
}

// paletteSearch is the search scratch. vals / cnt are the block's distinct
// luma values (ascending) and their counts: k-means runs on this histogram,
// which gives every sample of a value the same nearest centroid, so the
// centroid updates, distances and stopping rule equal SVT's per-sample loop.
type paletteSearch struct {
	data    [tile.MaxPalettePixels]int
	cands   [paletteMaxCands]paletteCand
	counted [256]int
	vals    [64]int
	cnt     [64]int
	idx     [64]uint8
	pre     [64]uint8
}

// paletteNearest is svt_av1_calc_indices_dim1 for one value: the first
// centroid at the smallest squared distance.
func paletteNearest(v int, centroids []int) uint8 {
	d := v - centroids[0]
	minDist, best := d*d, uint8(0)
	for j := 1; j < len(centroids); j++ {
		d := v - centroids[j]
		if d*d < minDist {
			minDist, best = d*d, uint8(j)
		}
	}
	return best
}

func (s *paletteSearch) calcIndices(nv int, centroids []int) {
	for i, v := range s.vals[:nv] {
		s.idx[i] = paletteNearest(v, centroids)
	}
}

func (s *paletteSearch) totalDist(nv int, centroids []int) int64 {
	var dist int64
	for i, v := range s.vals[:nv] {
		d := int64(v - centroids[s.idx[i]])
		dist += d * d * int64(s.cnt[i])
	}
	return dist
}

// paletteLCGRand16 is lcg_rand16 (random.h).
func paletteLCGRand16(state *uint32) uint32 {
	*state = *state*1103515245 + 12345
	return *state / 65536 % 32768
}

// calcCentroids is calc_centroids_dim1: each centroid the rounded mean of
// its samples, an empty cluster reseeded with a pseudo-random sample.
func (s *paletteSearch) calcCentroids(data []int, nv int, centroids []int) {
	var count [tile.PaletteMaxSize]int
	rand := uint32(data[0])
	for i := range centroids {
		centroids[i] = 0
	}
	for i, v := range s.vals[:nv] {
		count[s.idx[i]] += s.cnt[i]
		centroids[s.idx[i]] += v * s.cnt[i]
	}
	for i := range centroids {
		if count[i] == 0 {
			centroids[i] = data[paletteLCGRand16(&rand)%uint32(len(data))]
		} else {
			centroids[i] = (centroids[i] + count[i]>>1) / count[i]
		}
	}
}

// kMeans is svt_av1_k_means_dim1 on the value histogram.
func (s *paletteSearch) kMeans(data []int, nv int, centroids []int, maxItr int) {
	s.calcIndices(nv, centroids)
	thisDist := s.totalDist(nv, centroids)
	var preCentroids [tile.PaletteMaxSize]int
	for range maxItr {
		preDist := thisDist
		copy(preCentroids[:], centroids)
		copy(s.pre[:nv], s.idx[:nv])
		s.calcCentroids(data, nv, centroids)
		s.calcIndices(nv, centroids)
		thisDist = s.totalDist(nv, centroids)
		if thisDist > preDist {
			copy(centroids, preCentroids[:len(centroids)])
			copy(s.idx[:nv], s.pre[:nv])
			break
		}
		if slices.Equal(centroids, preCentroids[:len(centroids)]) {
			break
		}
	}
}

// search is search_palette_luma (kmeans arm, palette_level 3) for the
// bw x bh luma block at src (stride), cache being the block's palette color
// cache. It returns the candidates found.
func (s *paletteSearch) search(src []byte, stride, bw, bh int, cache []uint16, qIndex uint8) []paletteCand {
	clear(s.counted[:])
	colors := 0
	for r := range bh {
		for _, v := range src[r*stride : r*stride+bw] {
			if s.counted[v] == 0 {
				colors++
			}
			s.counted[v]++
		}
	}
	if colors <= 1 || colors > 64 {
		return nil
	}
	nv := 0
	for v, c := range s.counted {
		if c != 0 {
			s.vals[nv], s.cnt[nv] = v, c
			nv++
		}
	}
	n := bw * bh
	data := s.data[:n]
	lb, ub := int(src[0]), int(src[0])
	for r := range bh {
		for c, v := range src[r*stride : r*stride+bw] {
			data[r*bw+c] = int(v)
			lb, ub = min(lb, int(v)), max(ub, int(v))
		}
	}
	maxN := min(colors, tile.PaletteMaxSize)
	found := 0
	var centroids [tile.PaletteMaxSize]int
	for k := maxN; k >= tile.PaletteMinSize; k-- {
		cent := centroids[:k]
		if colors == tile.PaletteMinSize {
			cent[0], cent[1] = lb, ub
		} else {
			for i := range cent {
				cent[i] = lb + (2*i+1)*(ub-lb)/k/2
			}
			s.kMeans(data, nv, cent, 50)
		}
		cand := &s.cands[found]
		if s.paletteRDY(cand, data, nv, cent, cache, qIndex) {
			found++
		}
	}
	return s.cands[:found]
}

// paletteRDY is palette_rd_y with opt_colors: snap to the cache, dedupe,
// final indices and the map (the block is fully on screen here).
func (s *paletteSearch) paletteRDY(cand *paletteCand, data []int, nv int, centroids []int, cache []uint16, qIndex uint8) bool {
	if len(cache) > 0 {
		th := 6 + int(qIndex>>6)
		for i, c := range centroids {
			minDiff, idx := absInt(c-int(cache[0])), 0
			for j := 1; j < len(cache); j++ {
				if d := absInt(c - int(cache[j])); d < minDiff {
					minDiff, idx = d, j
				}
			}
			if minDiff <= th {
				centroids[i] = int(cache[idx])
			}
		}
	}
	slices.Sort(centroids)
	k := 1
	for i := 1; i < len(centroids); i++ {
		if centroids[i] != centroids[i-1] {
			centroids[k] = centroids[i]
			k++
		}
	}
	if k < tile.PaletteMinSize {
		return false
	}
	cand.res = tile.PaletteModeResult{YSize: uint8(k)}
	for i := range k {
		cand.res.YColors[i] = uint16(min(max(centroids[i], 0), 255))
	}
	var lut [256]uint8
	for _, v := range s.vals[:nv] {
		lut[v] = paletteNearest(v, centroids[:k])
	}
	for i, v := range data {
		cand.m[i] = lut[v]
	}
	cand.res.YMap = &cand.m
	return true
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// keyframePalette prices SVT's palette candidates for an n x n keyframe
// block against the intra mode the block already picked (its prediction in
// pred): each candidate's trial luma cost plus its y-mode, palette_mode_info
// and color-map rates against the intra mode's trial cost plus its y-mode
// and has_palette rates. A winning candidate's prediction replaces pred.
func (st *lossyEncodeState) keyframePalette(src SourceFrame420, block tile.BlockVisit, modeCtx *tile.BlockModeContext,
	px, py, n int, mode tile.IntraMode, pred []byte) *paletteCand {
	if !st.allowScreenContentTools || st.color.BitDepth != 8 {
		return nil
	}
	if ok, err := tile.PaletteAllowed(true, block.Size); err != nil || !ok {
		return nil
	}
	if st.pal == nil {
		st.pal = new(paletteSearch)
	}
	var cache [2 * tile.PaletteMaxSize]uint16
	nCache, err := modeCtx.PaletteYCache(int(block.X4), int(block.Y4), block.HaveTop, block.HaveLeft, &cache)
	if err != nil {
		return nil
	}
	cands := st.pal.search(src.Y[py*src.YStride+px:], src.YStride, n, n, cache[:nCache], st.qIndex)
	if len(cands) == 0 {
		return nil
	}
	above, left, err := modeCtx.KeyframeYModeContext(int(block.X4), int(block.Y4))
	if err != nil {
		return nil
	}
	yModeCDF, err := st.intraCDFs.KeyframeYModeCDF(above, left)
	if err != nil {
		return nil
	}
	req := st.paletteRequest(block, tile.IntraModeDC)
	baseReq := st.paletteRequest(block, mode)
	baseRate, err := tile.PaletteModeCost(&st.intraCDFs, modeCtx, baseReq, &tile.PaletteModeResult{})
	if err != nil {
		return nil
	}
	if st.trialBlockLocal {
		st.trialIntraSnap.save(&st.trialCDFs)
		defer st.trialIntraSnap.restore(&st.trialCDFs)
	}
	rateCost := func(r uint32) int64 { return (int64(r)*st.rdMult + 256) >> 9 }
	best := st.trialLumaCost(src.Y, pred, src.YStride, px, py, n) + rateCost(entropy.CostSymbol(yModeCDF, int(mode))+baseRate)
	var winner *paletteCand
	palPred := st.sadScratch[:n*n]
	for i := range cands {
		c := &cands[i]
		modeRate, err := tile.PaletteModeCost(&st.intraCDFs, modeCtx, req, &c.res)
		if err != nil {
			continue
		}
		mapRate, err := tile.PaletteTokensCost(&st.intraCDFs, req, &c.res)
		if err != nil {
			continue
		}
		for j := range palPred {
			palPred[j] = byte(c.res.YColors[c.m[j]])
		}
		cost := st.trialLumaCost(src.Y, palPred, src.YStride, px, py, n) + rateCost(entropy.CostSymbol(yModeCDF, int(tile.IntraModeDC))+modeRate+mapRate)
		if cost < best {
			best, winner = cost, c
			copy(pred, palPred)
		}
	}
	return winner
}

// paletteRequest is the palette syntax request of a keyframe block.
func (st *lossyEncodeState) paletteRequest(block tile.BlockVisit, mode tile.IntraMode) tile.PaletteModeRequest {
	return tile.PaletteModeRequest{
		AllowScreenContentTools: st.allowScreenContentTools,
		Size:                    block.Size,
		LumaMode:                mode,
		X4:                      block.X4,
		Y4:                      block.Y4,
		HaveTop:                 block.HaveTop,
		HaveLeft:                block.HaveLeft,
		BitDepth:                st.color.BitDepth,
		Color:                   st.color,
		ChromaMode:              tile.ChromaIntraModeDC,
		ChromaModeValid:         true,
		HasChroma:               tile.HasChromaBlockAt(block.Size, int(block.X4), int(block.Y4), st.color),
		VisibleW4:               block.VisibleW4,
		VisibleH4:               block.VisibleH4,
	}
}

// writeKeyframePalette codes palette_mode_info for a keyframe block (its
// palette, or has_palette_y = 0) and marks the palette contexts.
func (st *lossyEncodeState) writeKeyframePalette(modeCtx *tile.BlockModeContext, block tile.BlockVisit, mode tile.IntraMode, pal *paletteCand) error {
	res := tile.PaletteModeResult{}
	if pal != nil {
		res = pal.res
		res.YMap = nil
	}
	if err := tile.WritePaletteMode(st.w, &st.intraCDFs, modeCtx, st.paletteRequest(block, mode), &res); err != nil {
		return err
	}
	if err := modeCtx.MarkPaletteY(block.Size, int(block.X4), int(block.Y4), res); err != nil {
		return err
	}
	return modeCtx.MarkPaletteUV(block.Size, int(block.X4), int(block.Y4), tile.PaletteModeResult{})
}
