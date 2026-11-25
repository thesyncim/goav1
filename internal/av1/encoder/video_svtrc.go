package encoder

import "fmt"

// video_svtrc.go wires SVT-AV1's one-pass CBR rate control (svt_rc.go) into
// the 8-bit VideoEncoder: one controller for every temporal layer, the
// reference slots' base q / layer / intra flag for the active-best
// inheritance and the non-base q floor, and the cyclic refresh segments for
// the correction-factor update.

// svtRCRef is what SVT reads from a reference object for rate control
// (ref_base_q_idx, tmp_layer_idx, slice_type).
type svtRCRef struct {
	valid bool
	q     int
	layer int
	isI   bool
}

const (
	svtRefLast = 0
	svtRefT1   = 1
)

func newVideoEncoderCBRWithColor(width, height int, rc RateControlConfig, color SequenceColorConfig) (*VideoEncoder, error) {
	if _, err := rateControlPerFrameBits(rc); err != nil {
		return nil, err
	}
	e, err := newVideoEncoderWithColor(width, height, rc.MinQIndex/2+rc.MaxQIndex/2, color)
	if err != nil {
		return nil, err
	}
	e.armSVTRC(rc)
	return e, nil
}

func rateControlPerFrameBits(rc RateControlConfig) (int, error) {
	if rc.TargetBitsPerSecond <= 0 || rc.FramesPerSecond <= 0 {
		return 0, fmt.Errorf("encoder: invalid rate control target %d bps @ %d fps", rc.TargetBitsPerSecond, rc.FramesPerSecond)
	}
	if rc.MinQIndex == 0 || rc.MaxQIndex <= rc.MinQIndex {
		return 0, fmt.Errorf("encoder: invalid qindex range [%d, %d]", rc.MinQIndex, rc.MaxQIndex)
	}
	if _, err := rc.buffers().resolve(); err != nil {
		return 0, err
	}
	perFrameBits := rc.TargetBitsPerSecond / rc.FramesPerSecond
	if perFrameBits <= 0 {
		perFrameBits = 1
	}
	return perFrameBits, nil
}

// armSVTRC (re)initializes the controller: SVT re-runs set_rc_buffer_sizes
// and svt_av1_rc_init on every rate change (seq_param_changed).
func (e *VideoEncoder) armSVTRC(rc RateControlConfig) {
	e.rcEnabled = true
	e.crfEnabled = false
	e.rcTargetBits = rc.TargetBitsPerSecond
	e.rcFramesPerSec = rc.FramesPerSecond
	e.rcPerFrameBits, _ = rateControlPerFrameBits(rc)
	e.rcMinQ = rc.MinQIndex
	e.rcMaxQ = rc.MaxQIndex
	if e.qIndex < e.rcMinQ {
		e.qIndex = e.rcMinQ
	} else if e.qIndex > e.rcMaxQ {
		e.qIndex = e.rcMaxQ
	}
	e.svtRC.init(rc.TargetBitsPerSecond, rc.FramesPerSecond, rc.MinQIndex, rc.MaxQIndex, e.width, e.height, rc.buffers())
	e.cr.reset()
}

// SetTwoPass configures 2-pass VBR rate control (pass=1 collects FirstPassStats;
// pass=2 allocates frame Q from stats toward targetBitrate @ framerate).
func (e *VideoEncoder) SetTwoPass(pass int, stats *FirstPassStats, targetBitrate, framerate int, minQ, maxQ uint8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	switch pass {
	case 0:
		e.twoPass = twoPassState{}
		return nil
	case 1:
		e.rcEnabled = false
		e.crfEnabled = false
		e.twoPass.initPass1()
		return nil
	case 2:
		if stats == nil || len(stats.Frames) == 0 {
			return fmt.Errorf("encoder: 2-pass pass=2 requires non-empty FirstPassStats")
		}
		if targetBitrate <= 0 || framerate <= 0 {
			return fmt.Errorf("encoder: 2-pass pass=2 requires positive TargetBitrate and Framerate")
		}
		e.rcEnabled = false
		e.crfEnabled = false
		e.twoPass.initPass2(*stats, targetBitrate, framerate, minQ, maxQ)
		return nil
	default:
		return fmt.Errorf("encoder: Pass must be 0, 1, or 2, got %d", pass)
	}
}

// FirstPassStats returns the complexity statistics collected during pass 1.
func (e *VideoEncoder) FirstPassStats() FirstPassStats {
	if e == nil {
		return FirstPassStats{}
	}
	out := e.twoPass.stats
	out.Frames = append([]FirstPassFrameStats(nil), e.twoPass.stats.Frames...)
	return out
}

// SetCRF switches future frames to Constant Rate Factor (CRF) encoding on the
// standard AV1 1..63 quality scale (lower = higher quality). CRF adapts each
// frame's quantizer from temporal layer hierarchy and HME motion/complexity.
func (e *VideoEncoder) SetCRF(crf uint8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	if crf == 0 || crf > 63 {
		return fmt.Errorf("encoder: CRF must be in [1, 63], got %d", crf)
	}
	e.rcEnabled = false
	e.crfEnabled = true
	e.crfValue = crf
	e.qIndex = uint8(min(int(crf)*4, 255))
	e.rcTargetBits = 0
	e.rcFramesPerSec = 0
	e.rcPerFrameBits = 0
	e.svtRC = svtRateControl{}
	return nil
}

// SetQIndex switches future frames to fixed-quality encoding without
// disturbing reference state.
func (e *VideoEncoder) SetQIndex(qIndex uint8) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	if qIndex == 0 {
		return fmt.Errorf("encoder: qindex must be non-zero")
	}
	e.rcEnabled = false
	e.crfEnabled = false
	e.qIndex = qIndex
	e.rcTargetBits = 0
	e.rcFramesPerSec = 0
	e.rcPerFrameBits = 0
	e.svtRC = svtRateControl{}
	return nil
}

// SetRateControlConfig updates the CBR target used for future frames
// without disturbing reference state. The controller restarts as SVT-AV1's
// does on a rate change event.
func (e *VideoEncoder) SetRateControlConfig(rc RateControlConfig) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	if _, err := rateControlPerFrameBits(rc); err != nil {
		return err
	}
	e.armSVTRC(rc)
	return nil
}

// UpdateRateControlConfig changes the CBR settings for future frames. SVT-AV1
// handles a rate change event (seq_param_changed) by re-initializing its
// rate control, so this is equivalent to SetRateControlConfig.
func (e *VideoEncoder) UpdateRateControlConfig(rc RateControlConfig) error {
	return e.SetRateControlConfig(rc)
}

// svtRCHierLevels is hierarchical_levels for the encoder's structure: flat
// IPP mimics SVT's flat structure on top of HIERARCHICAL_LEVELS_AUTO (2 at
// low-delay CBR), layered streams have temporalLayers-1.
func (e *VideoEncoder) svtRCHierLevels() int {
	if e.temporalLayers <= 1 {
		return 2
	}
	return e.temporalLayers - 1
}

// rcKeyQ picks a keyframe's qindex (the fixed q without rate control).
func (e *VideoEncoder) rcKeyQ(width, height int, screen bool) uint8 {
	e.svtCurOn = false
	if e.twoPass.pass == 2 {
		e.qIndex = e.twoPass.pickPass2Q(true)
		return e.qIndex
	}
	if e.crfEnabled {
		q := max(min(int(e.crfValue)*4-12, 255), 1)
		e.qIndex = uint8(q)
		return e.qIndex
	}
	if !e.rcEnabled {
		return e.qIndex
	}
	f := svtRCFrame{
		key:         true,
		screen:      screen,
		flat:        e.temporalLayers <= 1,
		hierLevels:  e.svtRCHierLevels(),
		frameOffset: 0,
		width:       width,
		height:      height,
	}
	e.svtRC.target(&f)
	q := e.svtRC.pickQ(&f)
	e.svtCur, e.svtCurQ, e.svtCurCR, e.svtCurOn = f, q, svtCRStats{}, true
	e.qIndex = q
	return q
}

// rcInterQ picks an inter frame's qindex: rc_init_frame_stats' ME average,
// svt_aom_one_pass_rt_rate_alloc and svt_av1_rc_calc_qindex_rate_control.
// ref names the reference slot the frame predicts from.
func (e *VideoEncoder) rcInterQ(temporalID uint8, ref int, frameOffset int, width, height int, screen bool) uint8 {
	e.svtCurOn = false
	if e.twoPass.pass == 2 {
		e.qIndex = e.twoPass.pickPass2Q(false)
		return e.qIndex
	}
	if e.crfEnabled {
		q := int(e.crfValue) * 4
		switch temporalID {
		case 1:
			q += 10
		case 2:
			q += 20
		}
		if e.hme.fme.valid && len(e.hme.fme.me8x8Dist) > 0 {
			var sum uint64
			for _, d := range e.hme.fme.me8x8Dist {
				sum += d
			}
			avg := sum / uint64(len(e.hme.fme.me8x8Dist))
			switch {
			case avg > 6000:
				q += 12
			case avg > 3000:
				q += 6
			case avg < 600:
				q -= 8
			}
		}
		q = max(min(q, 255), 1)
		e.qIndex = uint8(q)
		return e.qIndex
	}
	if !e.rcEnabled {
		return e.qIndex
	}
	r := e.svtRef[ref]
	f := svtRCFrame{
		screen:        screen,
		flat:          e.temporalLayers <= 1,
		hierLevels:    e.svtRCHierLevels(),
		temporalLayer: int(temporalID),
		frameOffset:   frameOffset,
		refValid:      r.valid,
		refBaseQ:      r.q,
		refLayer:      r.layer,
		refIsI:        r.isI,
		width:         width,
		height:        height,
		gfUpdate:      frameOffset%lpd1MaxGFInterval == 0,
	}
	e.svtRC.beginFrame(&f, &e.hme)
	e.svtRC.target(&f)
	q := e.svtRC.pickQ(&f)
	e.svtCur, e.svtCurQ, e.svtCurCR, e.svtCurOn = f, q, svtCRStats{}, true
	return q
}

// rcUpdate feeds the coded frame's size back (svt_av1_rc_postencode_update)
// and records the picture in the reference slots it refreshed.
func (e *VideoEncoder) rcUpdate(frameBits int, temporalID uint8) {
	if e.twoPass.pass == 1 {
		var cost uint64
		if e.hme.fme.valid {
			for _, d := range e.hme.fme.me8x8Dist {
				cost += d
			}
		}
		e.twoPass.recordPass1(e.frameIndex == 1, cost, frameBits/8)
		return
	}
	if e.twoPass.pass == 2 {
		e.twoPass.updatePass2(frameBits / 8)
		return
	}
	if !e.rcEnabled || !e.svtCurOn {
		return
	}
	e.svtCurOn = false
	f := &e.svtCur
	e.svtRC.postencode(f, int(e.svtCurQ), frameBits, e.svtCurCR)
	if f.key {
		e.svtRef[svtRefLast] = svtRCRef{valid: true, q: int(e.svtCurQ), isI: true}
		e.svtRef[svtRefT1] = e.svtRef[svtRefLast]
	}
	e.qIndex = e.svtCurQ
}

// noteRCRefresh records an inter picture in the reference slot it
// refreshes (LAST for the base layer, the T1 slot for the L1T3 middle
// layer).
func (e *VideoEncoder) noteRCRefresh(slot int, q uint8, temporalID uint8) {
	e.svtRef[slot] = svtRCRef{valid: true, q: int(q), layer: int(temporalID)}
}

// svtScreen is the picture's sc_class1 (inherited from the last keyframe).
func (e *VideoEncoder) svtScreen() bool {
	return e.sc.class1
}

// meControls is svt_aom_sig_deriv_me for the picture's sc_class1. The
// sequence qp is 35 under CBR (static_config.qp keeps DEFAULT_QP) and the
// user qp under CQP.
func (e *VideoEncoder) meControls(width, height int) svtMECtrl {
	qp := 35
	if !e.rcEnabled {
		qp = min(int(e.qIndex)>>2, 63)
	}
	return svtMEControls(e.svtScreen(), width, height, qp)
}
