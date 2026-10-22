package lfmask

import "github.com/thesyncim/goav1/internal/av1/parser"

// LFValues is dav1d's per-frame (or per-superblock, under delta_lf) loop-filter
// level table, f->lf.lvl / ts->lflvlmem: values[seg][dir][ref][mode] where dir
// is 0 = Y vertical, 1 = Y horizontal, 2 = U, 3 = V; ref is 0 for intra and
// 1..7 for LAST..ALTREF; mode is 0 for GLOBALMV(/GLOBAL_GLOBALMV) and 1 for any
// other inter mode. A block's four levels are then plain table loads (dav1d
// decode.c passes &lflvl[seg_id][0][ref+1][!is_globalmv] to
// dav1d_create_lf_mask_inter and &lflvl[seg_id][0][0][0] for intra).
type LFValues [parser.MaxSegments][4][parser.RefFrames][2]uint8

// LFValueParams snapshots the frame-header fields dav1d_calc_lf_values reads.
type LFValueParams struct {
	LevelY        [2]uint8
	LevelU        uint8
	LevelV        uint8
	DeltaLFMulti  bool
	SegEnabled    bool
	SegDeltas     [parser.MaxSegments][4]int16 // delta_lf_y_v, _y_h, _u, _v
	ModeRefDeltas bool
	RefDelta      [parser.RefFrames]int8
	ModeDelta     [2]int8
}

// NewLFValueParams snapshots the loop-filter, segmentation and delta headers.
func NewLFValueParams(lf parser.LoopFilterParams, seg parser.SegmentationParams, delta parser.DeltaParams) LFValueParams {
	p := LFValueParams{
		LevelY:        lf.LevelY,
		LevelU:        lf.LevelU,
		LevelV:        lf.LevelV,
		DeltaLFMulti:  delta.DeltaLFMulti,
		SegEnabled:    seg.Enabled,
		ModeRefDeltas: lf.ModeRefDeltaEnabled,
		RefDelta:      lf.Deltas.Ref,
		ModeDelta:     lf.Deltas.Mode,
	}
	if seg.Enabled {
		for s := range p.SegDeltas {
			d := &seg.Data.Segments[s]
			p.SegDeltas[s] = [4]int16{d.DeltaLFYV, d.DeltaLFYH, d.DeltaLFU, d.DeltaLFV}
		}
	}
	return p
}

func iclip(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// calcLFValue ports dav1d calc_lf_value (src/lf_mask.c:404).
func calcLFValue(out *[parser.RefFrames][2]uint8, baseLvl, lfDelta, segDelta int, p *LFValueParams) {
	base := iclip(iclip(baseLvl+lfDelta, 0, 63)+segDelta, 0, 63)
	if !p.ModeRefDeltas {
		for r := range out {
			out[r] = [2]uint8{uint8(base), uint8(base)}
		}
		return
	}
	sh := 0
	if base >= 32 {
		sh = 1
	}
	v := uint8(iclip(base+int(p.RefDelta[0])*(1<<sh), 0, 63))
	out[0] = [2]uint8{v, v}
	for r := 1; r < parser.RefFrames; r++ {
		for m := 0; m < 2; m++ {
			delta := int(p.ModeDelta[m]) + int(p.RefDelta[r])
			out[r][m] = uint8(iclip(base+delta*(1<<sh), 0, 63))
		}
	}
}

// calcLFValueChroma ports dav1d calc_lf_value_chroma (src/lf_mask.c:427).
func calcLFValueChroma(out *[parser.RefFrames][2]uint8, baseLvl, lfDelta, segDelta int, p *LFValueParams) {
	if baseLvl == 0 {
		*out = [parser.RefFrames][2]uint8{}
		return
	}
	calcLFValue(out, baseLvl, lfDelta, segDelta, p)
}

// CalcLFValues ports dav1d_calc_lf_values (src/lf_mask.c:438): it fills the
// level table for every segment from the frame header and the current
// superblock's delta_lf state lfDelta (dav1d ts->last_delta_lf; without
// delta_lf_multi only lfDelta[0] is consulted).
func CalcLFValues(values *LFValues, p *LFValueParams, lfDelta [4]int8) {
	nSeg := 1
	if p.SegEnabled {
		nSeg = parser.MaxSegments
	}
	if p.LevelY[0] == 0 && p.LevelY[1] == 0 {
		for s := 0; s < nSeg; s++ {
			values[s] = [4][parser.RefFrames][2]uint8{}
		}
		return
	}
	d1, d2, d3 := int(lfDelta[0]), int(lfDelta[0]), int(lfDelta[0])
	if p.DeltaLFMulti {
		d1, d2, d3 = int(lfDelta[1]), int(lfDelta[2]), int(lfDelta[3])
	}
	for s := 0; s < nSeg; s++ {
		var sd [4]int16
		if p.SegEnabled {
			sd = p.SegDeltas[s]
		}
		calcLFValue(&values[s][0], int(p.LevelY[0]), int(lfDelta[0]), int(sd[0]), p)
		calcLFValue(&values[s][1], int(p.LevelY[1]), d1, int(sd[1]), p)
		calcLFValueChroma(&values[s][2], int(p.LevelU), d2, int(sd[2]), p)
		calcLFValueChroma(&values[s][3], int(p.LevelV), d3, int(sd[3]), p)
	}
}
