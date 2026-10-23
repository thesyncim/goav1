package encoder

import "github.com/thesyncim/goav1/internal/av1/tile"

// zeromv_skip.go ports libaom's realtime static-superblock exit (libaom
// v3.13 av1/encoder): av1_source_content_sb's per-SB source SAD class
// (encodeframe_utils.c), set_force_zeromv_skip_for_sb's early partition
// exit (var_based_part.c) and the force_zeromv_skip_for_blk mode/skip
// shortcut of nonrd_pickmode.c. A 64x64 superblock whose source barely
// changed and whose zero-motion LAST reconstruction is already close to it
// is coded as one 64x64 GLOBALMV LAST skip block with no ME, partition
// search, mode decision or transform. SVT-AV1 has no equivalent.
//
// Settings: libaom speed 10 screen content (part_early_exit_zeromv 1,
// set_zeromv_skip_based_on_source_sad 1 from the rt defaults). Screen
// content only: libaom's speed-11 video arm (kVeryLowSad, doubled
// thresholds) measured -0.10..-0.14 dB on camera clips for <=5% cycles.
// Only the superblock-level exit is ported; the block-level
// force_zeromv_skip_for_sb = 2 path is not.

// zmvSourceSAD is libaom's SOURCE_SAD.
const (
	zmvZeroSad uint8 = iota
	zmvVeryLowSad
	zmvLowSad
	zmvMedSad
	zmvHighSad
)

// zmvSourceSADClass is av1_source_content_sb's source_sad_nonrd from the
// 64x64 source SSE against the previous source.
func zmvSourceSADClass(sse uint64, increase bool) uint8 {
	veryLow, low, high := uint64(10000), uint64(100000), uint64(1000000)
	if increase {
		veryLow, low, high = veryLow<<1, low<<1, high<<1
	}
	switch {
	case sse == 0:
		return zmvZeroSad
	case sse < veryLow:
		return zmvVeryLowSad
	case sse < low:
		return zmvLowSad
	case sse > high:
		return zmvHighSad
	}
	return zmvMedSad
}

// zmvThreshExit64 is cpi->zeromv_skip_thresh_exit_part[BLOCK_64X64]
// (encodeframe.c: FORCE_ZMV_SKIP_128X128_BLK_DIFF 10000 scaled by the
// square root of the area ratio, capped at 4 per pixel).
const zmvThreshExit64 = 5000

// zmvCtrl is one picture's zero-mv skip controls.
type zmvCtrl struct {
	on        bool
	maxClass  uint8 // is_set_force_zeromv_skip_based_on_src_sad
	increase  bool  // increase_source_sad_thresh
	earlyExit int   // part_early_exit_zeromv
}

// zmvScreenControls is the speed-10 screen-content control set.
var zmvScreenControls = zmvCtrl{maxClass: zmvZeroSad, earlyExit: 1}

// classifySourceSAD fills h.srcSAD with each full 64x64 superblock's
// source SAD class against the active reference's source (prepare, before
// finish moves the current picture into a slot).
func (h *hmeState) classifySourceSAD(src SourceFrame420, increase bool) {
	h.srcSADOK = false
	if !h.havePrev || h.prev == nil || h.prev.w != src.Width || h.prev.h != src.Height {
		return
	}
	cols, rows := (src.Width+63)/64, (src.Height+63)/64
	if cap(h.srcSAD) < cols*rows {
		h.srcSAD = make([]uint8, cols*rows)
	}
	h.srcSAD = h.srcSAD[:cols*rows]
	if cap(h.zmvForce) < cols*rows {
		h.zmvForce = make([]bool, cols*rows)
	}
	h.zmvForce = h.zmvForce[:cols*rows]
	clear(h.zmvForce)
	p := h.prev
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			px, py := c*64, r*64
			class := zmvMedSad
			if px+64 <= src.Width && py+64 <= src.Height {
				s := src.Y[py*src.YStride+px:]
				q := p.pix[p.org+py*p.stride+px:]
				sse, _ := svtPD0PixelStats(s, src.YStride, q, p.stride, 64)
				class = zmvSourceSADClass(uint64(sse), increase)
			}
			h.srcSAD[r*cols+c] = class
		}
	}
	h.srcSADCols = cols
	h.srcSADOK = true
}

// zmvDecideSB is set_force_zeromv_skip_for_sb for the 64x64 superblock at
// (px, py): the source-SAD gate, then the zero-motion LAST SADs of the
// three planes against thresh_exit_part. It records the decision for the
// superblock's block coding.
func (st *lossyEncodeState) zmvDecideSB(src, ref SourceFrame420, px, py int) bool {
	c := &st.zmvCtrl
	h := st.hme
	if !c.on || h == nil || !h.srcSADOK || px+64 > src.Width || py+64 > src.Height ||
		ref.Width != src.Width || ref.Height != src.Height {
		return false
	}
	idx := (py/64)*h.srcSADCols + px/64
	if idx >= len(h.srcSAD) {
		return false
	}
	class := h.srcSAD[idx]
	if class > c.maxClass {
		return false
	}
	// Only the base segment: a cyclic-refresh boosted superblock keeps
	// ramping up its quality.
	if st.sbQIndex != nil && st.sbQIndex[(py/64)*st.sbQCols+px/64] != st.qIndex {
		return false
	}
	shift := 0
	if c.increase {
		shift = 1
	}
	threshY := uint32(zmvThreshExit64) << shift
	threshUV := uint32((3*zmvThreshExit64)>>2) << shift
	if class >= zmvVeryLowSad && c.earlyExit == 1 {
		threshUV >>= 3
	}
	ySAD := uint32(sadRectDualBlock(src.Y[py*src.YStride+px:], src.YStride, ref.Y[py*ref.YStride+px:], ref.YStride, 64, 64))
	if ySAD >= threshY {
		return false
	}
	if !st.color.MonoChrome {
		cx, cy := chromaXForColor(px, st.color), chromaYForColor(py, st.color)
		cw, ch := 64, 64
		if st.color.SubsamplingX {
			cw = 32
		}
		if st.color.SubsamplingY {
			ch = 32
		}
		for _, pl := range [2][2][]byte{{src.U, ref.U}, {src.V, ref.V}} {
			uv := uint32(sadRectDualBlock(pl[0][cy*src.ChromaStride+cx:], src.ChromaStride, pl[1][cy*ref.ChromaStride+cx:], ref.ChromaStride, cw, ch))
			if uv >= threshUV {
				return false
			}
		}
	}
	h.zmvForce[idx] = true
	return true
}

// zmvForced reports whether the 64x64 block at (px, py) is the superblock
// zmvDecideSB forced to zero-motion skip (force_zeromv_skip_for_blk).
func (st *lossyEncodeState) zmvForced(block tile.BlockSize, px, py int) bool {
	h := st.hme
	if !st.zmvCtrl.on || block != tile.BlockSize64x64 || h == nil || !h.srcSADOK {
		return false
	}
	idx := (py/64)*h.srcSADCols + px/64
	return idx < len(h.zmvForce) && h.zmvForce[idx]
}

// setZMVControls arms (or, with the zero value, disarms) the zero-mv exit
// on every tile coder for the picture about to be coded.
func (e *VideoEncoder) setZMVControls(c zmvCtrl) {
	e.pc.st.zmvCtrl = c
	for t := range e.tilePCs {
		e.tilePCs[t].st.zmvCtrl = c
	}
}
