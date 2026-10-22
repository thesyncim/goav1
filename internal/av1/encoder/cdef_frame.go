package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// cdef_frame.go drives SVT-AV1's per-picture CDEF flow (cdef_search.go) from
// the frame encoders: the signal derivation before the tiles, the search
// after reconstruction, and the cdef_idx signaling via the tile writer log.

// svtCDEFFramePlan is the picture's CDEF setup before the tiles run:
// pcs->ppcs->cdef_level and cdef_search_ctrls after
// update_cdef_filters_on_ref_info (md_config_process.c).
type svtCDEFFramePlan struct {
	// on is cdef_level != 0.
	on bool
	// search marks a picture whose strengths come from cdef_seg_search +
	// finish_cdef_search (neither use_reference_cdef_fs nor use_qp_strength):
	// its tiles record their symbol streams for the cdef_idx splice.
	search bool
	intra  bool
	ctrls  svtCDEFSearchCtrls
	lambda uint64
}

// planSVTCDEF derives the plan for one picture. ref is the LAST reference's
// signaled strengths (unused on I slices). me_based_cdef_skip never fires at
// cdef_recon_level 0 (zero_filter_strength_lvl 0 gives a zero threshold), the
// skip detector is off at the rtc levels, and tune PSNR leaves the
// sharpness CDEF switch off, so none of them can zero cdef_level here.
func planSVTCDEF(enabled, islice, isBase, notHighestLayer, screen bool, ref svtCDEFRefStrengths, refSkipPerc uint8) svtCDEFFramePlan {
	if !enabled {
		return svtCDEFFramePlan{}
	}
	level := svtRTCCDEFSearchLevel(islice, screen)
	p := svtCDEFFramePlan{intra: islice, ctrls: setSVTCDEFSearchControls(level, isBase, notHighestLayer)}
	p.on = p.ctrls.enabled
	// md_config_process.c: the skip detector (level 7 non-base pictures,
	// screen content at rtc p12) turns CDEF off when more than 75% of the
	// LAST reference was coded skip (rc_process.c get_ref_skip_percentage:
	// the list-0 reference's skip_coded_area); otherwise the reference
	// strengths feed update_cdef_filters_on_ref_info.
	switch {
	case islice || !p.on:
	case refSkipPerc > 75 && p.ctrls.useSkipDetector:
		p.on = false
	case p.ctrls.useReferenceCDEFFS || p.ctrls.searchBestRefFS:
		p.on = p.ctrls.updateOnRefInfo(ref)
	}
	p.search = p.on && !p.ctrls.useReferenceCDEFFS && !p.ctrls.useQPStrength
	return p
}

// decide returns the picture's signaled CDEF parameters and per-64x64
// strength indices (nil when every coded block uses index 0), the
// svt_aom_cdef_kernel finish step.
func (p *svtCDEFFramePlan) decide(s *cdefSearcher, src, recon *SourceFrame420, m *threading.FrameWorkLoopFilterMap, qIndex uint8) (CDEFParams, []uint8) {
	damping := 3 + (qIndex >> 6)
	switch {
	case !p.on:
		return CDEFParams{Damping: damping}, nil
	case p.ctrls.useQPStrength:
		y, uv := svtPickCDEFFromQP(qIndex, p.intra)
		out := CDEFParams{Damping: damping}
		out.YStrength[0], out.UVStrength[0] = y, uv
		return out, nil
	case p.ctrls.useReferenceCDEFFS:
		out := CDEFParams{Damping: damping}
		out.YStrength[0], out.UVStrength[0] = uint8(p.ctrls.predYF), uint8(p.ctrls.predUVF)
		return out, nil
	}
	d := s.search(src, recon, m, &p.ctrls, qIndex, p.lambda)
	if cdefDecisionTestHook != nil {
		params := d.params
		cdefDecisionTestHook(&params, d.fbIdx)
		d.params = params
	}
	if d.params.Bits == 0 {
		return d.params, nil
	}
	return d.params, d.fbIdx
}

// cdefDecisionTestHook, set only by tests, rewrites a searched picture's
// decision before it is signaled (forcing cdef_bits > 0 exercises the
// cdef_idx replay end to end).
var cdefDecisionTestHook func(p *CDEFParams, fbIdx []uint8)

// armCDEFLog starts the tile writer's symbol log when the frame searches
// CDEF (the writer was just Reset, which clears any previous log).
func (pc *pframeCoder) armCDEFLog() {
	if pc.cdefLog != nil {
		pc.cdefLog.Reset()
		pc.writer.SetLog(pc.cdefLog)
	}
}

// svtCDEFLogs owns the per-tile symbol logs of searched pictures and the
// cdef_idx table scratch. Logs are indexed by tile, not by coder, so a
// keyframe's (larger) logs size the inter frames' too.
type svtCDEFLogs struct {
	logs   []entropy.WriterLog
	coders []*pframeCoder
	idx    []uint8
}

// arm hands tile t's log to coders[t] for the coming tile run.
func (l *svtCDEFLogs) arm(coders []*pframeCoder) {
	if len(l.logs) < len(coders) {
		next := make([]entropy.WriterLog, len(coders))
		copy(next, l.logs)
		l.logs = next
	}
	for t, pc := range coders {
		pc.cdefLog = &l.logs[t]
	}
}

// disarm stops the coders recording.
func (l *svtCDEFLogs) disarm(coders []*pframeCoder) {
	for _, pc := range coders {
		pc.cdefLog = nil
	}
}

// reserve sizes the logs for an nTiles-wide tiling of a width x height
// picture at three entries per pixel of the tile, so a prewarmed encoder's
// first frames record without growing them: a symbol averages a little
// under one coded bit (a 192x128 keyframe at 2.5 bits per pixel logs 2.2
// entries per pixel), and base-layer inter frames log far fewer.
func (l *svtCDEFLogs) reserve(width, height, nTiles int) {
	if len(l.logs) < nTiles {
		next := make([]entropy.WriterLog, nTiles)
		copy(next, l.logs)
		l.logs = next
	}
	per := 3 * width * height / max(nTiles, 1)
	for t := range l.logs {
		l.logs[t].Reserve(per)
	}
	g := newCDEFFrameGeometry(width, height)
	if n := g.nvfb * g.nhfb * 4; cap(l.idx) < n {
		l.idx = make([]uint8, n)
	}
}

// replay re-emits every tile payload of a recorded picture with the
// cdef_idx literals of fbIdx (frame raster, nhfb wide) spliced in, when the
// search chose more than one strength. Each tile's bytes are rewritten into
// its coder's writer buffer: the log, not the first pass's bytes, is the
// replay's input.
func (l *svtCDEFLogs) replay(payloads []TilePayload, coders []*pframeCoder, tiles TileInfo, miCols uint16, height int, fbIdx []uint8, bits uint8) error {
	if bits == 0 {
		return nil
	}
	g := newCDEFFrameGeometry(int(miCols)*4, height)
	sbRows := (g.miRows + 15) / 16
	for t := range payloads {
		c0, c1 := tileColBounds(tiles, t, miCols)
		sbCols := (int(c1-c0) + 15) / 16
		n := sbCols * sbRows * 4
		if cap(l.idx) < n {
			l.idx = make([]uint8, n)
		}
		idx := l.idx[:n]
		col0 := int(c0) / 16
		for r := 0; r < sbRows; r++ {
			for c := 0; c < sbCols; c++ {
				idx[(r*sbCols+c)*4] = fbIdx[r*g.nhfb+col0+c]
			}
		}
		pc := coders[t]
		buf := pc.writerBuf
		if cap(buf) == 0 {
			buf = make([]byte, 1<<18)
			pc.writerBuf = buf
		}
		data, err := l.logs[t].Replay(buf[:0], idx, int(bits))
		if err != nil {
			return err
		}
		payloads[t].Data = data
	}
	return nil
}

// keyframeTileCoders lists the coders a keyframe's tiles run on: the video
// encoder's e.pc for tile 0 and e.tilePCs[t] after it
// (runKeyframeTileWorkers), else the caller's tilePC (installing a fresh
// per-tile set when it has none, so the logs outlive the tile run).
func keyframeTileCoders(e *VideoEncoder, tilePC *func(t int) *pframeCoder, nTiles int) []*pframeCoder {
	if e != nil {
		if len(e.tilePCs) < nTiles {
			e.tilePCs = make([]pframeCoder, nTiles)
		}
		c := append(e.cdefLogs.coders[:0], &e.pc)
		for t := 1; t < nTiles; t++ {
			c = append(c, &e.tilePCs[t])
		}
		e.cdefLogs.coders = c
		return c
	}
	if *tilePC == nil {
		pcs := make([]pframeCoder, nTiles)
		*tilePC = func(t int) *pframeCoder { return &pcs[t] }
	}
	coders := make([]*pframeCoder, nTiles)
	for t := range coders {
		coders[t] = (*tilePC)(t)
	}
	return coders
}

// interTileCoders lists the coders encodePReusing's tiles run on: e.pc for a
// single tile, e.tilePCs[t] otherwise.
func (e *VideoEncoder) interTileCoders(nTiles int) []*pframeCoder {
	c := e.cdefLogs.coders[:0]
	if nTiles == 1 {
		c = append(c, &e.pc)
	} else {
		for t := 0; t < nTiles; t++ {
			c = append(c, &e.tilePCs[t])
		}
	}
	e.cdefLogs.coders = c
	return c
}

// lpd1RefSkipPerc is the reference's skip_coded_area (0 without one), the
// statistic light-PD1 already publishes with each reference.
func lpd1RefSkipPerc(ref *lpd1Frame) uint8 {
	if ref == nil || !ref.valid {
		return 0
	}
	return ref.skipPerc
}
