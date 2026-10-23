// Ported from SVT-AV1 v4.1.0 (third_party/upstream/svt-av1/Source/Lib/Codec):
//
//   - enc_mode_config.c: set_cdef_search_controls (the per-level candidate
//     tables), set_cdef_recon_controls, and the cdef_search_level choice of
//     svt_aom_sig_deriv_multi_processes_rtc;
//   - md_config_process.c: update_cdef_filters_on_ref_info and
//     me_based_cdef_skip;
//   - cdef_process.c: cdef_seg_search (the per-64x64 filter-block strength
//     trials) and the svt_aom_cdef_kernel finish/apply flow;
//   - enc_cdef.c: svt_sb_compute_cdef_list, svt_aom_compute_cdef_dist_8bit_c,
//     svt_search_one_dual_c, joint_strength_search_dual, finish_cdef_search
//     and svt_pick_cdef_from_qp;
//   - rc_process.c: svt_aom_lambda_assign's full lambda.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package encoder

import (
	"math"
	"sync"
	"sync/atomic"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

const (
	// cdefTotalStrengths is TOTAL_STRENGTHS: CDEF_PRI_STRENGTHS (16) x
	// CDEF_SEC_STRENGTHS (4), the packed-strength index space.
	cdefTotalStrengths = 64
	// cdefSecStrengths is CDEF_SEC_STRENGTHS.
	cdefSecStrengths = 4
	// cdefStrengthBits is CDEF_STRENGTH_BITS.
	cdefStrengthBits = 6
	// cdefDefaultMSEUV is cdef_process.c default_mse_uv: the distortion a
	// chroma candidate the level does not test contributes (times 64).
	cdefDefaultMSEUV = 1040400
	// cdefFB is the 64x64 filter block.
	cdefFB = 64
)

// svtPfGI is enc_mode_config.c pf_gi: primary strength p with secondary 0.
var svtPfGI = [16]int8{0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60}

// svtCDEFSearchCtrls is CdefSearchControls (pcs.h). The first/second pass
// tables hold packed strengths; -1 in a chroma table means "not tested".
type svtCDEFSearchCtrls struct {
	enabled                bool
	firstPassFSNum         int
	defaultSecondPassFSNum int
	defaultFirstPassFS     [cdefTotalStrengths]int8
	defaultSecondPassFS    [cdefTotalStrengths]int8
	defaultFirstPassFSUV   [cdefTotalStrengths]int8
	defaultSecondPassFSUV  [cdefTotalStrengths]int8
	useReferenceCDEFFS     bool
	searchBestRefFS        bool
	subsamplingFactor      uint8
	useSkipDetector        bool
	uvFromY                bool
	useQPStrength          bool
	predYF                 int8
	predUVF                int8
}

// svtCDEFReconCtrls is CdefReconControls.
type svtCDEFReconCtrls struct {
	zeroFSCostBias        uint16
	zeroFilterStrengthLvl uint8
	prevCDEFDistTH        uint16
}

// setSVTCDEFSearchControls is set_cdef_search_controls.
func setSVTCDEFSearchControls(level uint8, isBase, notHighestLayer bool) svtCDEFSearchCtrls {
	var c svtCDEFSearchCtrls
	// secondPass fills default_second_pass_fs with pf + j for j in 1..3 (levels 1-4).
	secondPass := func() {
		sf := 0
		for i := 0; i < c.firstPassFSNum; i++ {
			pf := c.defaultFirstPassFS[i]
			for j := int8(1); j < 4; j++ {
				c.defaultSecondPassFS[sf] = pf + j
				sf++
			}
		}
	}
	uvFirstFromLuma := func() {
		for i := 0; i < c.firstPassFSNum; i++ {
			c.defaultFirstPassFSUV[i] = c.defaultFirstPassFS[i]
		}
	}
	uvSecondOff := func() {
		for i := 0; i < c.defaultSecondPassFSNum; i++ {
			c.defaultSecondPassFSUV[i] = -1
		}
	}
	switch level {
	case 0:
		return c
	case 1:
		c.enabled = true
		c.firstPassFSNum = 16
		c.defaultSecondPassFSNum = c.firstPassFSNum * 3
		for i := range 16 {
			c.defaultFirstPassFS[i] = svtPfGI[i]
		}
		secondPass()
		uvFirstFromLuma()
		for i := 0; i < c.defaultSecondPassFSNum; i++ {
			c.defaultSecondPassFSUV[i] = c.defaultSecondPassFS[i]
		}
		c.subsamplingFactor = 1
	case 2:
		c.enabled = true
		c.firstPassFSNum = 12
		c.defaultSecondPassFSNum = c.firstPassFSNum * 3
		for i, g := range [12]int{0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14} {
			c.defaultFirstPassFS[i] = svtPfGI[g]
		}
		secondPass()
		uvFirstFromLuma()
		uvSecondOff()
		c.subsamplingFactor = 1
	case 3:
		c.enabled = true
		c.firstPassFSNum = 5
		c.defaultSecondPassFSNum = c.firstPassFSNum * 3
		for i, g := range [5]int{0, 4, 8, 12, 15} {
			c.defaultFirstPassFS[i] = svtPfGI[g]
		}
		secondPass()
		uvFirstFromLuma()
		uvSecondOff()
		c.subsamplingFactor = 1
	case 4:
		c.enabled = true
		c.firstPassFSNum = 3
		c.defaultSecondPassFSNum = c.firstPassFSNum * 3
		for i, g := range [3]int{0, 7, 15} {
			c.defaultFirstPassFS[i] = svtPfGI[g]
		}
		secondPass()
		uvFirstFromLuma()
		uvSecondOff()
		c.subsamplingFactor = 1
	case 5:
		c.enabled = true
		c.firstPassFSNum = 3
		c.defaultSecondPassFSNum = c.firstPassFSNum
		for i, g := range [3]int{0, 7, 15} {
			c.defaultFirstPassFS[i] = svtPfGI[g]
			c.defaultSecondPassFS[i] = svtPfGI[g] + 2
		}
		uvFirstFromLuma()
		uvSecondOff()
		c.searchBestRefFS = !notHighestLayer
		c.subsamplingFactor = 1
	case 6, 7, 8:
		c.enabled = true
		c.firstPassFSNum = 2
		c.defaultSecondPassFSNum = c.firstPassFSNum
		c.defaultFirstPassFS[0] = svtPfGI[0]
		c.defaultFirstPassFS[1] = svtPfGI[15]
		c.defaultSecondPassFS[0] = svtPfGI[0] + 2
		c.defaultSecondPassFS[1] = svtPfGI[15] + 2
		if level == 8 {
			c.defaultFirstPassFSUV[0] = -1
			c.defaultFirstPassFSUV[1] = -1
		} else {
			c.defaultFirstPassFSUV[0] = c.defaultFirstPassFS[0]
			c.defaultFirstPassFSUV[1] = c.defaultFirstPassFS[1]
		}
		// "when using search_best_ref_fs, set at least 3 filters"
		c.defaultFirstPassFSUV[2] = -1
		c.defaultSecondPassFSUV[0] = -1
		c.defaultSecondPassFSUV[1] = -1
		c.subsamplingFactor = 4
		switch level {
		case 6:
			c.searchBestRefFS = !notHighestLayer
		case 7:
			c.useReferenceCDEFFS = !notHighestLayer
			c.searchBestRefFS = !isBase
			c.useSkipDetector = !isBase
		case 8:
			c.useReferenceCDEFFS = !isBase
			c.searchBestRefFS = !isBase
			c.uvFromY = true
		}
	case 9:
		c.enabled = true
		c.useQPStrength = true
	}
	// Chroma filters copied from luma are never tested.
	if c.uvFromY && !c.useQPStrength {
		for i := 0; i < c.firstPassFSNum; i++ {
			c.defaultFirstPassFSUV[i] = -1
		}
		for i := 0; i < c.defaultSecondPassFSNum; i++ {
			c.defaultSecondPassFSUV[i] = -1
		}
	}
	return c
}

// setSVTCDEFReconControls is set_cdef_recon_controls.
func setSVTCDEFReconControls(level uint8) svtCDEFReconCtrls {
	switch level {
	case 1:
		return svtCDEFReconCtrls{61, 2, 10}
	case 2:
		return svtCDEFReconCtrls{61, 3, 10}
	case 3:
		return svtCDEFReconCtrls{60, 3, 10}
	case 4:
		return svtCDEFReconCtrls{58, 3, 10}
	}
	return svtCDEFReconCtrls{}
}

// svtRTCCDEFSearchLevel is the cdef_search_level of
// svt_aom_sig_deriv_multi_processes_rtc at preset 12 (above ENC_M11, so the
// final branches): screen content 5 on I slices and 7 otherwise; camera
// content 5 on I slices and 8 otherwise. cdef_recon_level is 0.
func svtRTCCDEFSearchLevel(islice, screen bool) uint8 {
	if screen {
		if islice {
			return 5
		}
		return 7
	}
	if islice {
		return 5
	}
	return 8
}

// svtCDEFRefStrengths is the reference object's ref_cdef_strengths copy
// (rest_process.c): the frame's signaled luma/chroma strengths.
type svtCDEFRefStrengths struct {
	num uint8
	y   [MaxCDEFStrengths]uint8
	uv  [MaxCDEFStrengths]uint8
}

func svtCDEFRefFromParams(p CDEFParams) svtCDEFRefStrengths {
	r := svtCDEFRefStrengths{num: 1 << p.Bits, y: p.YStrength, uv: p.UVStrength}
	return r
}

// updateCDEFFiltersOnRefInfo is update_cdef_filters_on_ref_info for a P
// slice with no list-1 reference (mrp_level 0: ref_list1_count_try is 0).
// It returns false when the frame's cdef_level drops to 0.
func (c *svtCDEFSearchCtrls) updateOnRefInfo(ref svtCDEFRefStrengths) bool {
	if c.useReferenceCDEFFS {
		lowest := uint8(cdefTotalStrengths - 1)
		highest := uint8(0)
		for fs := uint8(0); fs < ref.num; fs++ {
			lowest = min(lowest, ref.y[fs])
			highest = max(highest, ref.y[fs])
		}
		c.predYF = int8(min(63, (int(lowest)+int(highest))/2))
		c.predUVF = 0
		c.firstPassFSNum = 0
		c.defaultSecondPassFSNum = 0
		return c.predYF != 0 || c.predUVF != 0
	}
	if c.searchBestRefFS {
		c.firstPassFSNum = 1
		c.defaultSecondPassFSNum = 0
		if int8(ref.y[0]) != c.defaultFirstPassFS[0] {
			c.defaultFirstPassFS[1] = int8(ref.y[0])
			c.firstPassFSNum++
		}
		if int8(ref.uv[0]) == c.defaultFirstPassFSUV[0] {
			c.defaultFirstPassFSUV[0] = -1
			c.defaultFirstPassFSUV[1] = -1
		}
		return c.firstPassFSNum != 1
	}
	return true
}

// svtCDEFFullLambda is svt_aom_lambda_assign's full_lambda at 8-bit for the
// frame qindex (svt_aom_compute_rd_mult with me_q_index == q_index, so the
// rtc sb-lambda modulation factor is 128; lambda_scale_factors are 128):
// svt_aom_compute_rd_mult_based_on_qindex (def_kf 3.3 for KF_UPDATE, def_arf
// 3.25 for GF/ARF_UPDATE, def_inter 3.2 otherwise, times dc_q squared),
// rd_frame_type_factor[8-bit][gf_update_type] (tlFactor) and the rtc key
// frame boost RTC_KF_LAMBDA_BOOST (100/128).
func svtCDEFFullLambda(qIndex uint8, key, gfUpdate bool, tlFactor int64) uint64 {
	quant, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.PlaneY)
	if err != nil {
		return 1
	}
	q := float64(quant.DC)
	mult := 3.2
	switch {
	case key:
		mult = 3.3
	case gfUpdate:
		mult = 3.25
	}
	rdmult := int64((mult + 0.0015*q) * q * q)
	if key {
		tlFactor = 150
	}
	rdmult = (rdmult * tlFactor) >> 7
	if key {
		rdmult = (rdmult * 100) >> 7
	}
	if rdmult < 1 {
		rdmult = 1
	}
	return uint64(rdmult)
}

// svtRDCost is rd_cost.h RDCOST: ROUND_POWER_OF_TWO(R * RM, 9) + D << 7.
func svtRDCost(lambda uint64, rate int64, dist uint64) uint64 {
	return uint64((rate*int64(lambda)+(1<<8))>>9) + dist<<7
}

// svtPickCDEFFromQP is svt_pick_cdef_from_qp for camera content at 8-bit
// (cdef level 9's use_qp_strength): the libaom quadratic fits from the AC
// quantizer, evaluated in float32 with roundf like the C.
func svtPickCDEFFromQP(qIndex uint8, intra bool) (y, uv uint8) {
	quant, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.PlaneY)
	if err != nil {
		return 0, 0
	}
	q := float32(quant.AC)
	r := func(v float32) int { return int(float32(math.Round(float64(v)))) }
	var yF1, yF2, uvF1, uvF2 int
	if !intra {
		yF1 = r(q*q*-0.0000023593946 + q*0.0068615186 + 0.02709886)
		yF2 = r(q*q*-0.00000057629734 + q*0.0013993345 + 0.03831067)
		uvF1 = r(q*q*-0.0000007095069 + q*0.0034628846 + 0.00887099)
		uvF2 = r(q*q*0.00000023874085 + q*0.00028223585 + 0.05576307)
	} else {
		yF1 = r(q*q*0.0000033731974 + q*0.008070594 + 0.0187634)
		yF2 = r(q*q*0.0000029167343 + q*0.0027798624 + 0.0079405)
		uvF1 = r(q*q*-0.0000130790995 + q*0.012892405 - 0.00748388)
		uvF2 = r(q*q*0.0000032651783 + q*0.00035520183 + 0.00228092)
	}
	clampI := func(v, hi int) int { return max(0, min(v, hi)) }
	yF1, yF2 = clampI(yF1, 15), clampI(yF2, 3)
	uvF1, uvF2 = clampI(uvF1, 15), clampI(uvF2, 3)
	return uint8(yF1*cdefSecStrengths + yF2), uint8(uvF1*cdefSecStrengths + uvF2)
}

// cdefSearcher owns the frame-level CDEF search state (pcs->mse_seg,
// skip_cdef_seg, cdef_dir_data) and per-block scratch, reused across frames.
type cdefSearcher struct {
	mse    [2][][cdefTotalStrengths]uint64
	skipFB []bool
	fbIdx  []uint8

	// cdefFBScratch is lane 0's (the calling goroutine's) scratch.
	cdefFBScratch
	sbMSE   [2][]*[cdefTotalStrengths]uint64
	sbIndex []int
	totMSE  [cdefTotalStrengths][cdefTotalStrengths]uint64

	// The frame's search job and its persistent lanes (search).
	job         cdefSearchJob
	nextRow     atomic.Int32
	wantLanes   int
	lanes       []cdefFBScratch
	laneWork    chan int
	laneStarted int
	laneWG      sync.WaitGroup
	laneDone    sync.WaitGroup
}

// svtCDEFDecision is the frame's chosen CDEF parameters and the per-64x64
// strength index (frame raster, nhfb x nvfb), finish_cdef_search's output.
type svtCDEFDecision struct {
	params CDEFParams
	fbIdx  []uint8
}

// cdefFrameGeometry is the 64x64 filter-block grid of a frame.
type cdefFrameGeometry struct {
	miRows, miCols int
	nvfb, nhfb     int
}

func newCDEFFrameGeometry(width, height int) cdefFrameGeometry {
	miCols := ((width + 7) >> 3) << 1
	miRows := ((height + 7) >> 3) << 1
	return cdefFrameGeometry{
		miRows: miRows, miCols: miCols,
		nvfb: (miRows + 15) / 16, nhfb: (miCols + 15) / 16,
	}
}

func (s *cdefSearcher) bind(g cdefFrameGeometry) {
	n := g.nvfb * g.nhfb
	if cap(s.skipFB) < n {
		s.skipFB = make([]bool, n)
		s.fbIdx = make([]uint8, n)
		s.sbIndex = make([]int, n)
		for p := range s.mse {
			s.mse[p] = make([][cdefTotalStrengths]uint64, n)
			s.sbMSE[p] = make([]*[cdefTotalStrengths]uint64, n)
		}
	}
	s.skipFB = s.skipFB[:n]
	s.fbIdx = s.fbIdx[:n]
}

// computeCDEFList is svt_sb_compute_cdef_list for a 64x64 filter block: the
// 8x8 blocks with at least one non-skip 4x4 mode-info cell, raster order.
func computeCDEFList(m *threading.FrameWorkLoopFilterMap, g cdefFrameGeometry, miRow, miCol int, dlist []cdef.BlockPosition) int {
	maxr := min(g.miRows-miRow, 16)
	maxc := min(g.miCols-miCol, 16)
	if m.NoSkip != nil {
		// The noskip grid holds the same any-non-skip-MI test per 8x8 block
		// (miRow, miCol are 64-aligned).
		word := m.NoSkip.UnitWord(miRow>>4, miCol>>4)
		count := 0
		for r := 0; r < maxr; r += 2 {
			for c := 0; c < maxc; c += 2 {
				if word>>uint((r>>1)*8+c>>1)&1 != 0 {
					dlist[count] = cdef.BlockPosition{BY: uint8(r >> 1), BX: uint8(c >> 1)}
					count++
				}
			}
		}
		return count
	}
	stride := int(m.Stride)
	nonSkip := func(r, c int) bool {
		if r >= int(m.Rows) || c >= stride {
			return false
		}
		rec := &m.Records[r*stride+c]
		return rec.Valid && !rec.SkipTransform
	}
	count := 0
	for r := 0; r < maxr; r += 2 {
		for c := 0; c < maxc; c += 2 {
			mr, mc := miRow+r, miCol+c
			if nonSkip(mr, mc) || nonSkip(mr, mc+1) || nonSkip(mr+1, mc) || nonSkip(mr+1, mc+1) {
				dlist[count] = cdef.BlockPosition{BY: uint8(r >> 1), BX: uint8(c >> 1)}
				count++
			}
		}
	}
	return count
}

// computeCDEFDist8 is svt_aom_compute_cdef_dist_8bit_c: the SSE of the
// filtered blocks of dlist against the source, every subsamplingFactor-th
// row. filtered is the filter-block scratch (stride fstride), src the source
// plane at the filter block's origin.
func computeCDEFDist8(src []byte, sstride int, filtered []byte, fstride int, dlist []cdef.BlockPosition, bwLog2, bhLog2 int, subsampling int) uint64 {
	var sum uint64
	bw := 1 << bwLog2
	bh := 1 << bhLog2
	for _, b := range dlist {
		y0 := int(b.BY) << bhLog2
		x0 := int(b.BX) << bwLog2
		for i := 0; i < bh; i += subsampling {
			so := (y0+i)*sstride + x0
			fo := (y0+i)*fstride + x0
			s := src[so : so+bw]
			f := filtered[fo : fo+bw]
			for j := range s {
				e := int32(s[j]) - int32(f[j])
				sum += uint64(e * e)
			}
		}
	}
	return sum
}

// loadCDEFInput fills the 16-bit filter input of one plane of a filter
// block the way cdef_seg_search's svt_aom_copy_sb8_16 does: the block plus
// its CDEF_VBORDER/CDEF_HBORDER halo where those pixels exist in the frame,
// CDEF_VERY_LARGE elsewhere. Only the rows and columns the filter taps can
// reach are written.
func (s *cdefFBScratch) loadCDEFInput(plane []byte, stride, planeW, planeH, x0, y0, w, h int) {
	const vb, hb = cdef.VerticalBorder, cdef.HorizontalBorder
	origin := vb*cdef.BStride + hb
	// Rows [rTop, rBot) and columns [lo, hi) of each row lie inside the
	// frame; they widen in one LoadSampleRows8Trusted pass (NEON on arm64),
	// everything else is VeryLarge.
	rTop, rBot := max(-vb, -y0), min(h+vb, planeH-y0)
	lo, hi := max(0, hb-x0), min(w+2*hb, planeW-x0+hb)
	for r := -vb; r < h+vb; r++ {
		row := s.in[origin+r*cdef.BStride-hb : origin+r*cdef.BStride+w+hb]
		if r < rTop || r >= rBot {
			for i := range row {
				row[i] = cdef.VeryLarge
			}
			continue
		}
		for i := 0; i < lo; i++ {
			row[i] = cdef.VeryLarge
		}
		for i := hi; i < len(row); i++ {
			row[i] = cdef.VeryLarge
		}
	}
	if rTop < rBot && lo < hi {
		frame.LoadSampleRows8Trusted(s.in[origin+rTop*cdef.BStride-hb+lo:], cdef.BStride,
			plane[(y0+rTop)*stride+x0-hb+lo:], stride, hi-lo, rBot-rTop)
	}
}

// cdefFBScratch is one search lane's per-filter-block scratch.
type cdefFBScratch struct {
	in     [cdef.InputBufferSize]uint16
	tmp    [cdefFB * cdefFB]byte
	dlist  [cdefFB * cdefFB / 64]cdef.BlockPosition
	trials [cdefTotalStrengths]cdef.SearchTrial
	sse    [cdefTotalStrengths]uint64
	dir    cdef.DirectionGrid
	vari   cdef.VarianceGrid
}

// cdefSearchJob is the frame's search input, shared read-only by the lanes.
type cdefSearchJob struct {
	g                   cdefFrameGeometry
	m                   *threading.FrameWorkLoopFilterMap
	subsampling         uint8
	damping             uint8
	total               int
	fsY, fsUV           [cdefTotalStrengths]int8
	chromaTested        bool
	planes, srcPlanes   [3][]byte
	strides, srcStrides [3]int
	planeW, planeH      [3]int
}

// setLanes sets how many lanes the per-filter-block search runs on (SVT's
// cdef_kernel searches picture segments in parallel); fewer than two keeps
// it on the calling goroutine.
func (s *cdefSearcher) setLanes(n int) {
	s.wantLanes = n
}

// startLanes launches the persistent search lanes once (lane 0 is the
// caller); they read the frame's job from s.job and claim filter-block rows.
func (s *cdefSearcher) startLanes(n int) {
	if len(s.lanes) < n {
		next := make([]cdefFBScratch, n)
		s.lanes = next
	}
	if s.laneWork == nil {
		s.laneWork = make(chan int, 64)
	}
	work := s.laneWork
	for s.laneStarted < n-1 {
		s.laneStarted++
		s.laneDone.Add(1)
		go func() {
			defer s.laneDone.Done()
			for idx := range work {
				s.searchClaimedRows(&s.lanes[idx])
				s.laneWG.Done()
			}
		}()
	}
}

// close stops the persistent search lanes.
func (s *cdefSearcher) close() {
	if s.laneWork == nil {
		return
	}
	close(s.laneWork)
	s.laneDone.Wait()
	s.laneWork = nil
	s.laneStarted = 0
	s.lanes = nil
}

// search runs cdef_seg_search over every filter block of the frame and then
// finish_cdef_search. recon is the pre-CDEF reconstruction (rtc deblocking
// is off, so it is the cdef_input_recon), src the source picture. Filter
// block rows are independent (each writes only its own mse/skip entries), so
// they spread over the search lanes and the result does not depend on the
// lane count.
func (s *cdefSearcher) search(src, recon *SourceFrame420, m *threading.FrameWorkLoopFilterMap, ctrls *svtCDEFSearchCtrls, qIndex uint8, lambda uint64) svtCDEFDecision {
	g := newCDEFFrameGeometry(recon.Width, recon.Height)
	s.bind(g)
	j := &s.job
	j.g, j.m, j.subsampling = g, m, ctrls.subsamplingFactor
	j.damping = 3 + (qIndex >> 6)
	first := ctrls.firstPassFSNum
	j.total = first + ctrls.defaultSecondPassFSNum
	j.fsY, j.fsUV = [cdefTotalStrengths]int8{}, [cdefTotalStrengths]int8{}
	for gi := 0; gi < j.total; gi++ {
		if gi < first {
			j.fsY[gi] = ctrls.defaultFirstPassFS[gi]
			j.fsUV[gi] = ctrls.defaultFirstPassFSUV[gi]
		} else {
			j.fsY[gi] = ctrls.defaultSecondPassFS[gi-first]
			j.fsUV[gi] = ctrls.defaultSecondPassFSUV[gi-first]
		}
	}
	// A chroma plane whose candidates are all untested (-1) only contributes
	// default_mse_uv; SVT still copies its input, which nothing reads.
	j.chromaTested = false
	for gi := 0; gi < j.total; gi++ {
		j.chromaTested = j.chromaTested || j.fsUV[gi] != -1
	}
	j.planes = [3][]byte{recon.Y, recon.U, recon.V}
	j.srcPlanes = [3][]byte{src.Y, src.U, src.V}
	j.strides = [3]int{recon.YStride, recon.ChromaStride, recon.ChromaStride}
	j.srcStrides = [3]int{src.YStride, src.ChromaStride, src.ChromaStride}
	j.planeW = [3]int{recon.Width, (recon.Width + 1) >> 1, (recon.Width + 1) >> 1}
	j.planeH = [3]int{recon.Height, (recon.Height + 1) >> 1, (recon.Height + 1) >> 1}
	lanes := min(s.wantLanes, g.nvfb)
	s.nextRow.Store(0)
	if lanes >= 2 {
		s.startLanes(lanes)
		s.laneWG.Add(lanes - 1)
		for i := 1; i < lanes; i++ {
			s.laneWork <- i
		}
		s.searchClaimedRows(&s.cdefFBScratch)
		s.laneWG.Wait()
	} else {
		s.searchClaimedRows(&s.cdefFBScratch)
	}
	s.job.m = nil
	s.job.planes, s.job.srcPlanes = [3][]byte{}, [3][]byte{}
	return s.finish(g, ctrls, qIndex, lambda, j.fsY[:j.total])
}

// searchClaimedRows searches filter-block rows claimed from s.nextRow until
// none remain.
func (s *cdefSearcher) searchClaimedRows(sc *cdefFBScratch) {
	for {
		fbr := int(s.nextRow.Add(1)) - 1
		if fbr >= s.job.g.nvfb {
			return
		}
		s.searchRow(sc, fbr)
	}
}

// searchRow is cdef_seg_search over one row of 64x64 filter blocks.
func (s *cdefSearcher) searchRow(sc *cdefFBScratch, fbr int) {
	j := &s.job
	g := j.g
	total := j.total
	for fbc := 0; fbc < g.nhfb; fbc++ {
		fb := fbr*g.nhfb + fbc
		lr, lc := 16*fbr, 16*fbc
		nhb := min(16, g.miCols-lc)
		nvb := min(16, g.miRows-lr)
		count := computeCDEFList(j.m, g, lr, lc, sc.dlist[:])
		if count == 0 {
			s.skipFB[fb] = true
			continue
		}
		s.skipFB[fb] = false
		dlist := sc.dlist[:count]
		for pli := 0; pli < 3; pli++ {
			if pli > 0 && !j.chromaTested {
				for gi := 0; gi < total; gi++ {
					s.mse[1][fb][gi] = cdefDefaultMSEUV * 64
				}
				break
			}
			dec := 0
			if pli > 0 {
				dec = 1
			}
			x0 := (lc * 4) >> dec
			y0 := (lr * 4) >> dec
			w := (nhb * 4) >> dec
			h := (nvb * 4) >> dec
			sc.loadCDEFInput(j.planes[pli], j.strides[pli], j.planeW[pli], j.planeH[pli], x0, y0, w, h)
			ss := int(j.subsampling)
			if pli == 0 {
				ss = min(ss, 4) // BLOCK_8X8
			} else {
				ss = min(ss, 1) // BLOCK_4X4 (4:2:0)
			}
			if ss < 1 {
				ss = 1
			}
			bLog2 := 3 - dec
			fstride := cdefFB >> dec
			srcPlane := j.srcPlanes[pli][y0*j.srcStrides[pli]+x0:]
			reconPlane := j.planes[pli][y0*j.strides[pli]+x0:]
			if pli == 0 {
				// svt_cdef_filter_fb's cdef_find_dir on the first luma trial
				// with a nonzero strength; the grids then serve every luma and
				// chroma trial of the block.
				sc.copyFB(reconPlane, j.strides[pli], fstride, w, h, 1)
				_ = cdef.FilterFrameBlocksU8Trusted(sc.tmp[:], fstride, nil, 0, dlist, &sc.dir, &sc.vari, cdef.FrameFilterParams{
					XDec: 0, YDec: 0, Plane: cdef.PlaneY, Damping: j.damping,
				})
			}
			if pli == 0 && sc.searchLumaBatched(srcPlane, j.srcStrides[0], dlist, j.fsY[:total], j.damping, ss, &s.mse[0][fb]) {
				continue
			}
			for gi := 0; gi < total; gi++ {
				if pli > 0 && j.fsUV[gi] == -1 {
					s.mse[1][fb][gi] = cdefDefaultMSEUV * 64
					continue
				}
				packed := j.fsY[gi]
				if pli > 0 {
					packed = j.fsUV[gi]
				}
				pri := uint8(packed) / cdefSecStrengths
				sec := uint8(packed) % cdefSecStrengths
				if sec == 3 {
					sec = 4
				}
				var curr uint64
				if pri == 0 && sec == 0 {
					// The (0,0) trial is the unfiltered reconstruction.
					curr = cdefDist8(srcPlane, j.srcStrides[pli], reconPlane, j.strides[pli], dlist, bLog2, bLog2, ss) * uint64(ss)
				} else {
					// svt_cdef_filter_fb with the subsampling factor: only the
					// measured rows are copied and filtered.
					sc.copyFB(reconPlane, j.strides[pli], fstride, w, h, ss)
					_ = cdef.FilterBlocksU8SubsampledTrusted(sc.tmp[:], fstride, sc.in[:], cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder, dlist, &sc.dir, &sc.vari, cdef.FrameFilterParams{
						XDec: uint8(dec), YDec: uint8(dec), Plane: cdef.Plane(pli),
						Level: pri, SecondaryStrength: sec, Damping: j.damping,
					}, ss)
					curr = cdefDist8(srcPlane, j.srcStrides[pli], sc.tmp[:], fstride, dlist, bLog2, bLog2, ss) * uint64(ss)
				}
				switch pli {
				case 0:
					s.mse[0][fb][gi] = curr
				case 1:
					s.mse[1][fb][gi] = curr
				default:
					s.mse[1][fb][gi] += curr
				}
			}
		}
	}
}

// searchLumaBatched evaluates every luma trial of filter block fb in one
// batched pass (cdef.SearchBlocksU8Trusted): per trial, the same
// filter-then-measure sum the per-trial loop computes. It returns false when
// the build has no batched kernel, in which case the caller runs that loop.
func (s *cdefFBScratch) searchLumaBatched(srcPlane []byte, srcStride int, dlist []cdef.BlockPosition, fsY []int8, damping uint8, ss int, mse *[cdefTotalStrengths]uint64) bool {
	total := len(fsY)
	trials := s.trials[:total]
	for gi, packed := range fsY {
		sec := uint8(packed) % cdefSecStrengths
		if sec == 3 {
			sec = 4
		}
		trials[gi] = cdef.SearchTrial{Primary: uint8(packed) / cdefSecStrengths, Secondary: sec}
	}
	if !cdef.SearchBlocksU8Trusted(srcPlane, srcStride, s.in[:], cdef.VerticalBorder*cdef.BStride+cdef.HorizontalBorder,
		dlist, &s.dir, &s.vari, int(damping), trials, ss, s.sse[:total]) {
		return false
	}
	for gi := range fsY {
		mse[gi] = s.sse[gi] * uint64(ss)
	}
	return true
}

// copyFB copies every step-th row of the w x h filter-block reconstruction
// into the trial scratch.
func (s *cdefFBScratch) copyFB(recon []byte, stride, fstride, w, h, step int) {
	for r := 0; r < h; r += step {
		copy(s.tmp[r*fstride:r*fstride+w], recon[r*stride:r*stride+w])
	}
}

// searchOneDual is svt_search_one_dual_c over the used strength range.
func (s *cdefSearcher) searchOneDual(lev0, lev1 []int, nbStrengths int, sbCount, startGI, endGI int) uint64 {
	for j := startGI; j < endGI; j++ {
		clear(s.totMSE[j][startGI:endGI])
	}
	m0, m1 := s.sbMSE[0], s.sbMSE[1]
	for i := 0; i < sbCount; i++ {
		bestMSE := uint64(1) << 63
		for gi := 0; gi < nbStrengths; gi++ {
			curr := m0[i][lev0[gi]] + m1[i][lev1[gi]]
			if curr < bestMSE {
				bestMSE = curr
			}
		}
		for j := startGI; j < endGI; j++ {
			for k := startGI; k < endGI; k++ {
				best := bestMSE
				curr := m0[i][j] + m1[i][k]
				if curr < best {
					best = curr
				}
				s.totMSE[j][k] += best
			}
		}
	}
	bestTot := uint64(1) << 63
	bestID0, bestID1 := 0, 0
	for j := startGI; j < endGI; j++ {
		for k := startGI; k < endGI; k++ {
			if s.totMSE[j][k] < bestTot {
				bestTot = s.totMSE[j][k]
				bestID0, bestID1 = j, k
			}
		}
	}
	lev0[nbStrengths] = bestID0
	lev1[nbStrengths] = bestID1
	return bestTot
}

// jointStrengthSearchDual is joint_strength_search_dual: the greedy pass
// adds one (luma, chroma) pair at a time, then 4*nb refinement passes
// reconsider each selected pair.
func (s *cdefSearcher) jointStrengthSearchDual(lev0, lev1 []int, nbStrengths, sbCount, startGI, endGI int) uint64 {
	bestTot := uint64(1) << 63
	for i := 0; i < nbStrengths; i++ {
		bestTot = s.searchOneDual(lev0, lev1, i, sbCount, startGI, endGI)
	}
	for i := 0; i < 4*nbStrengths; i++ {
		for j := 0; j < nbStrengths-1; j++ {
			lev0[j] = lev0[j+1]
			lev1[j] = lev1[j+1]
		}
		bestTot = s.searchOneDual(lev0, lev1, nbStrengths-1, sbCount, startGI, endGI)
	}
	return bestTot
}

// finish is finish_cdef_search's search branch (cdef_recon_level 0:
// zero_fs_cost_bias is 0, so the (0,0) bias block is not taken).
func (s *cdefSearcher) finish(g cdefFrameGeometry, ctrls *svtCDEFSearchCtrls, qIndex uint8, lambda uint64, filterMap []int8) svtCDEFDecision {
	sbCount := 0
	for fb := range s.skipFB {
		s.fbIdx[fb] = 0
		if s.skipFB[fb] {
			continue
		}
		s.sbMSE[0][sbCount] = &s.mse[0][fb]
		s.sbMSE[1][sbCount] = &s.mse[1][fb]
		s.sbIndex[sbCount] = fb
		sbCount++
	}
	endGI := len(filterMap)
	var p CDEFParams
	bestTot := uint64(1) << 63
	nbStrengthBits := 0
	for i := 0; i <= 3; i++ {
		var lev0, lev1 [MaxCDEFStrengths]int
		nb := 1 << i
		tot := s.jointStrengthSearchDual(lev0[:], lev1[:], nb, sbCount, 0, endGI)
		totalBits := sbCount*i + nb*cdefStrengthBits*2
		rate := int64(totalBits) << 9 // av1_cost_literal
		cost := svtRDCost(lambda, rate, tot*16)
		if cost < bestTot {
			bestTot = cost
			nbStrengthBits = i
			for j := 0; j < nb; j++ {
				p.YStrength[j] = uint8(lev0[j])
				if ctrls.uvFromY {
					p.UVStrength[j] = uint8(lev0[j])
				} else {
					p.UVStrength[j] = uint8(lev1[j])
				}
			}
		}
	}
	nb := 1 << nbStrengthBits
	for i := nb; i < MaxCDEFStrengths; i++ {
		p.YStrength[i], p.UVStrength[i] = 0, 0
	}
	p.Bits = uint8(nbStrengthBits)
	for i := 0; i < sbCount; i++ {
		bestGI := 0
		bestMSE := uint64(1) << 63
		for gi := 0; gi < nb; gi++ {
			curr := s.sbMSE[0][i][p.YStrength[gi]] + s.sbMSE[1][i][p.UVStrength[gi]]
			if curr < bestMSE {
				bestGI = gi
				bestMSE = curr
			}
		}
		s.fbIdx[s.sbIndex[i]] = uint8(bestGI)
	}
	for i := 0; i < nb; i++ {
		p.YStrength[i] = uint8(filterMap[p.YStrength[i]])
		p.UVStrength[i] = uint8(filterMap[p.UVStrength[i]])
	}
	p.Damping = 3 + (qIndex >> 6)
	return svtCDEFDecision{params: p, fbIdx: s.fbIdx}
}
