package tile

import (
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// This file ports dav1d's spatial/temporal reference-MV machinery
// (src/refmvs.h, src/refmvs.c: refmvs_block, refmvs_tile,
// dav1d_refmvs_tile_sbrow_init, dav1d_refmvs_find, add_spatial_candidate,
// scan_row, scan_col, add_temporal_candidate and the extended-candidate
// helpers) together with the per-block splat helpers from src/decode.c
// (splat_oneref_mv, splat_tworef_mv, splat_intrabc_mv, splat_intraref).
//
// Every decoded block of an inter (or intrabc) frame stores one 12-byte
// refmvsBlock per 4x4 unit into a tile-owned row buffer laid out exactly like
// dav1d's refmvs_frame.r: 35 rows of frame-width 4x4 cells, addressed through
// a 37-entry row table that keeps the current superblock row plus the odd rows
// -5/-3/-1 of the superblock row above. The candidate search then reads that
// frame-wide layout directly, so no per-superblock neighbor snapshots or
// cross-superblock carriers are needed.

// refmvsInvalidMV is dav1d's INVALID_MV (y = x = INT16_MIN) in the packed
// row-low/col-high layout used by refmvsMV.
const refmvsInvalidMV refmvsMV = 0x80008000

// refmvsMV is dav1d's `union mv`: the row component in the low 16 bits and the
// column component in the high 16 bits, compared as one 32-bit value.
type refmvsMV uint32

func packRefmvsMV(v motion.Vector) refmvsMV {
	return refmvsMV(uint32(uint16(v.Row)) | uint32(uint16(v.Col))<<16)
}

func (m refmvsMV) vector() motion.Vector {
	return motion.Vector{Row: int16(uint16(m)), Col: int16(uint16(m >> 16))}
}

func (m refmvsMV) row() int32 { return int32(int16(uint16(m))) }
func (m refmvsMV) col() int32 { return int32(int16(uint16(m >> 16))) }

func makeRefmvsMV(row, col int32) refmvsMV {
	return refmvsMV(uint32(uint16(int16(row))) | uint32(uint16(int16(col)))<<16)
}

func (m refmvsMV) negate() refmvsMV {
	return makeRefmvsMV(-m.row(), -m.col())
}

// refmvsBlock is dav1d's refmvs_block: the MV pair, the reference pair in
// dav1d's +1 encoding (0 = intra/intrabc, -1 = none), the block size and the
// mf flags (1 = globalmv with a min side >= 8 px, 2 = newmv).
type refmvsBlock struct {
	mv  [2]refmvsMV
	ref refmvsRefPair
	bs  BlockSize
	mf  uint8
}

// refmvsRefPair is dav1d's union refmvs_refpair: ref[0] in the low byte,
// ref[1] in the high byte, compared as one 16-bit pair.
type refmvsRefPair uint16

func makeRefmvsRefPair(ref0, ref1 int8) refmvsRefPair {
	return refmvsRefPair(uint8(ref0)) | refmvsRefPair(uint8(ref1))<<8
}

func (p refmvsRefPair) ref0() int8 { return int8(p) }
func (p refmvsRefPair) ref1() int8 { return int8(p >> 8) }

// at is ref.ref[n].
func (p refmvsRefPair) at(n int) int8 { return int8(p >> (uint(n&1) * 8)) }

// refmvsCandidate is dav1d's refmvs_candidate.
type refmvsCandidate struct {
	mv     [2]refmvsMV
	weight int32
}

// refmvsFrame carries the per-frame state of dav1d's refmvs_frame consumed by
// dav1d_refmvs_find. Pointer-bearing fields (temporal, mfmv) sit first so the
// GC pointer prefix ends immediately after mfmv.rpRef.
type refmvsFrame struct {
	temporal *TemporalMotionField
	// mfmv is the temporal projection setup of dav1d_refmvs_init_frame; when
	// projectTmvs is set the tile projects each superblock row into rpProj
	// (load_tmvs) instead of reading a whole-frame temporal field.
	mfmv refmvsMfmv

	iw4, ih4 int32

	// signBias and pocDiff are indexed by ref-1 (dav1d's rf->sign_bias and
	// rf->pocdiff).
	signBias [7]uint8
	pocDiff  [7]int8

	// useRefFrameMVs is the frame header's use_ref_frame_mvs flag, which
	// seeds the globalmv context; temporal reports whether projected temporal
	// samples are available (dav1d's rf->use_ref_frame_mvs).
	useRefFrameMVs bool
	projectTmvs    bool

	allowHighPrecisionMV bool
	forceIntegerMV       bool

	// gmType holds the frame's global motion type per reference (index ref-1).
	gmType [7]parser.GlobalMotionType

	// hasNonIdentityGM is true if any reference has non-identity global motion
	// (so blockReferenceGlobalMVsForBlock cannot be skipped as zero).
	hasNonIdentityGM bool
	// hasWarpGM is true if any reference has global motion type > TRANSLATION
	// (so isGlobalMVBlock can return true for global warp promotion).
	hasWarpGM bool
}

const (
	refmvsRowsPerBuffer = 35
	refmvsRowTable      = 32 + 5
)

// refmvsTile is dav1d's refmvs_tile plus the tile-owned row storage that
// dav1d keeps in refmvs_frame.r. rows holds offsets into buf (−1 for dav1d's
// NULL rows). Slices and frame sit first so rows and scalar geometry fall
// outside the GC pointer prefix.
type refmvsTile struct {
	buf []refmvsBlock
	// rpProj is dav1d's rp_proj for this tile: 16 rows of projected
	// temporal samples (8x8 units), rpStride entries per row.
	rpProj []TemporalMotionEntry
	frame  refmvsFrame

	stride   int32
	rpStride int32
	rows     [refmvsRowTable]int32

	tileColStart, tileColEnd int32
	tileRowStart, tileRowEnd int32

	bound bool
	// tall marks rows bound to a frame-tall EncoderRefmvsRows store (one
	// slab per superblock row, as dav1d's frame-thread buffer) instead of the
	// tile-private 35-row buffer.
	tall bool

	// tl4x4Filter is dav1d's t->tl_4x4_filter: the interpolation filters of
	// the first 4x4 block of an 8x8 split, used for the top-left cell of the
	// sub8x8 chroma prediction.
	tl4x4Filter motion.InterpFilters
}

// bindFrame sizes the row storage for a frame of the given coded width (in
// pixels) and records the tile bounds. rp_stride matches dav1d's
// ((width + 127) & ~127) >> 3; the row stride is twice that.
func (rt *refmvsTile) bindFrame(frame refmvsFrame, codedWidth int32, tileColStart, tileColEnd, tileRowStart, tileRowEnd int32) {
	rpStride := ((codedWidth + 127) &^ 127) >> 3
	stride := rpStride * 2
	need := int(stride) * refmvsRowsPerBuffer
	if cap(rt.buf) < need {
		rt.buf = make([]refmvsBlock, need)
	}
	rt.buf = rt.buf[:need]
	rt.stride = stride
	rt.tall = false
	if frame.projectTmvs {
		if need := int(rpStride) * 16; cap(rt.rpProj) < need {
			rt.rpProj = make([]TemporalMotionEntry, need)
		}
		rt.rpProj = rt.rpProj[:int(rpStride)*16]
		rt.rpStride = rpStride
	}
	rt.frame = frame
	rt.tileColStart = tileColStart
	rt.tileColEnd = min(tileColEnd, frame.iw4)
	rt.tileRowStart = tileRowStart
	rt.tileRowEnd = min(tileRowEnd, frame.ih4)
	rt.bound = true
	if refmvsPoisonEnabled {
		refmvsPoison(rt.buf)
	}
}

// PreallocBlockLoopRefmvsScratch sizes the refmvs row storage for frames up
// to miCols 4x4 columns wide so steady-state and cold decodes stay
// allocation-free.
func PreallocBlockLoopRefmvsScratch(s *BlockLoopScratch, miCols int) {
	if s == nil || miCols <= 0 {
		return
	}
	stride := ((int32(miCols)*4 + 127) &^ 127) >> 3 * 2
	if need := int(stride) * refmvsRowsPerBuffer; cap(s.Refmvs.buf) < need {
		s.Refmvs.buf = make([]refmvsBlock, need)
	}
	if need := int(stride) / 2 * 16; cap(s.Refmvs.rpProj) < need {
		s.Refmvs.rpProj = make([]TemporalMotionEntry, need)
	}
}

// sbrowInit ports dav1d_refmvs_tile_sbrow_init for one tile superblock row.
func (rt *refmvsTile) sbrowInit(sby int32, sbsz int32) {
	if rt.tall {
		rt.tallSbrowInit(sby, sbsz)
		return
	}
	off := (sbsz * sby) & 16
	r := int32(0)
	for i := int32(0); i < sbsz; i++ {
		rt.rows[off+5+i] = r
		r += rt.stride
	}
	rt.rows[off+0] = r
	r += rt.stride
	rt.rows[off+1] = -1
	rt.rows[off+2] = r
	r += rt.stride
	rt.rows[off+3] = -1
	rt.rows[off+4] = r
	if sby&1 != 0 {
		rt.rows[off+0], rt.rows[off+sbsz+0] = rt.rows[off+sbsz+0], rt.rows[off+0]
		rt.rows[off+2], rt.rows[off+sbsz+2] = rt.rows[off+sbsz+2], rt.rows[off+2]
		rt.rows[off+4], rt.rows[off+sbsz+4] = rt.rows[off+sbsz+4], rt.rows[off+4]
	}
}

// row returns the row slice for row-table index idx ((by4 & 31) + 5 + dy).
func (rt *refmvsTile) row(idx int32) []refmvsBlock {
	o := rt.rows[idx]
	return rt.buf[o : o+rt.stride]
}

// splat ports dav1d's splat_mv (splat_mv_neon): store one refmvs block into
// every 4x4 cell of the block. mv packs mv[0] and mv[1]; rest packs ref[0],
// ref[1], bs and mf, low byte first. As in the asm, the row pointers come
// from the row table (sliced once for the block's rows) and each width has
// its own straight-line store body: narrow blocks store each cell from these
// registers (one 8- and one 4-byte store), wider ones store the 48-byte
// four-cell pattern a whole number of times.
func (rt *refmvsTile) splat(mv uint64, rest uint32, by4, bx4, bw4, bh4 int32) {
	base := (by4 & 31) + 5
	rows := rt.rows[base : base+bh4]
	buf := rt.buf
	switch bw4 {
	case 1:
		for _, o := range rows {
			buf[o+bx4].set(mv, rest)
		}
		return
	case 2:
		for _, o := range rows {
			r := (*[2]refmvsBlock)(buf[o+bx4:])
			r[0].set(mv, rest)
			r[1].set(mv, rest)
		}
		return
	}
	var pat [4]refmvsBlock
	for i := range pat {
		pat[i].set(mv, rest)
	}
	switch bw4 {
	case 4:
		for _, o := range rows {
			*(*[4]refmvsBlock)(buf[o+bx4:]) = pat
		}
	case 8:
		for _, o := range rows {
			r := (*[8]refmvsBlock)(buf[o+bx4:])
			*(*[4]refmvsBlock)(r[0:]) = pat
			*(*[4]refmvsBlock)(r[4:]) = pat
		}
	case 16:
		for _, o := range rows {
			r := (*[16]refmvsBlock)(buf[o+bx4:])
			*(*[4]refmvsBlock)(r[0:]) = pat
			*(*[4]refmvsBlock)(r[4:]) = pat
			*(*[4]refmvsBlock)(r[8:]) = pat
			*(*[4]refmvsBlock)(r[12:]) = pat
		}
	default: // 32
		for _, o := range rows {
			r := (*[32]refmvsBlock)(buf[o+bx4:])
			for i := 0; i < 32; i += 4 {
				*(*[4]refmvsBlock)(r[i:]) = pat
			}
		}
	}
}

func (b *refmvsBlock) set(mv uint64, rest uint32) {
	b.ref, b.bs, b.mf = refmvsRefPair(rest), BlockSize(rest>>16), uint8(rest>>24)
	b.mv[0], b.mv[1] = refmvsMV(mv), refmvsMV(mv>>32)
}

func refmvsSplatRest(ref0, ref1 int8, bs BlockSize, mf uint8) uint32 {
	return uint32(uint8(ref0)) | uint32(uint8(ref1))<<8 | uint32(bs)<<16 | uint32(mf)<<24
}

// splatInterBlock ports splat_oneref_mv / splat_tworef_mv.
func (rt *refmvsTile) splatInterBlock(size BlockSize, by4, bx4 int32, motionResult InterMotionResult) {
	dims := blockDimensions[size]
	bw4, bh4 := int32(dims.W4), int32(dims.H4)
	refs := motionResult.References
	mode := motionResult.Mode
	mv := uint64(packRefmvsMV(motionResult.MV[0]))
	ref0, ref1 := int8(refs.Ref0)+1, int8(-1)
	mf := uint8(0)
	if refs.Compound {
		ref1 = int8(refs.Ref1) + 1
		mv |= uint64(packRefmvsMV(motionResult.MV[1])) << 32
		cm := mode.CompoundMode
		if cm == CompoundInterModeGlobalGlobal {
			mf |= 1
		}
		if (1<<cm)&0xbc != 0 {
			mf |= 2
		}
	} else {
		if motionResult.InterIntra {
			ref1 = 0
		}
		if mode.Mode == InterModeGlobalMV && min(bw4, bh4) >= 2 {
			mf |= 1
		}
		if mode.Mode == InterModeNewMV {
			mf |= 2
		}
	}
	rt.splat(mv, refmvsSplatRest(ref0, ref1, size, mf), by4, bx4, bw4, bh4)
}

// splatIntraBlock ports splat_intraref.
func (rt *refmvsTile) splatIntraBlock(size BlockSize, by4, bx4 int32) {
	dims := blockDimensions[size]
	rt.splat(uint64(refmvsInvalidMV), refmvsSplatRest(0, -1, size, 0), by4, bx4, int32(dims.W4), int32(dims.H4))
}

// refmvsResult is the output of refmvsFind: dav1d's mvstack, n_mvs and ctx.
type refmvsResult struct {
	stack [8]refmvsCandidate
	count int32
	ctx   int32
	// rawCtx is the packed (refmv_ctx << 4) | (globalmv_ctx << 3) | newmv_ctx
	// mode context before the compound mapping (libaom's mode_context).
	rawCtx int32
}

func addSpatialCandidate(stack *[8]refmvsCandidate, cnt *int32, weight int32, b *refmvsBlock, ref refmvsRefPair, gmv *[2]refmvsMV, haveNewMVMatch *int32, haveRefMVMatch *int32) {
	if b.mv[0] == refmvsInvalidMV {
		return // intra block, no intrabc
	}
	if ref.ref1() == -1 {
		ref0 := ref.ref0()
		var candMV refmvsMV
		if b.ref.at(0) == ref0 {
			candMV = b.mv[0]
		} else if b.ref.at(1) == ref0 {
			candMV = b.mv[1]
		} else {
			return
		}
		if b.mf&1 != 0 && gmv[0] != refmvsInvalidMV {
			candMV = gmv[0]
		}
		*haveRefMVMatch = 1
		*haveNewMVMatch |= int32(b.mf >> 1)
		last := *cnt
		for m := int32(0); m < last; m++ {
			if stack[m].mv[0] == candMV {
				stack[m].weight += weight
				return
			}
		}
		if last < 8 {
			stack[last].mv[0] = candMV
			stack[last].weight = weight
			*cnt = last + 1
		}
		return
	} else if b.ref == ref {
		cand := b.mv
		if b.mf&1 != 0 {
			if gmv[0] != refmvsInvalidMV {
				cand[0] = gmv[0]
			}
			if gmv[1] != refmvsInvalidMV {
				cand[1] = gmv[1]
			}
		}
		*haveRefMVMatch = 1
		*haveNewMVMatch |= int32(b.mf >> 1)
		last := *cnt
		for n := int32(0); n < last; n++ {
			if stack[n].mv == cand {
				stack[n].weight += weight
				return
			}
		}
		if last < 8 {
			stack[last].mv = cand
			stack[last].weight = weight
			*cnt = last + 1
		}
	}
}

// scanRow ports dav1d's scan_row. row is the refmvs row slice and bx4 the
// first scanned column.
func scanRow(stack *[8]refmvsCandidate, cnt *int32, ref refmvsRefPair, gmv *[2]refmvsMV, row []refmvsBlock, bx4, bw4, w4, maxRows, step int32, haveNewMVMatch, haveRefMVMatch *int32) int32 {
	candB := &row[bx4]
	firstDim := blockDimensions[candB.bs]
	candBW4 := int32(firstDim.W4)
	length := max(step, min(bw4, candBW4))
	if bw4 <= candBW4 {
		weight := int32(2)
		if bw4 != 1 {
			weight = max(2, min(2*maxRows, int32(firstDim.H4)))
		}
		addSpatialCandidate(stack, cnt, length*weight, candB, ref, gmv, haveNewMVMatch, haveRefMVMatch)
		return weight >> 1
	}
	for x := int32(0); ; {
		addSpatialCandidate(stack, cnt, length*2, candB, ref, gmv, haveNewMVMatch, haveRefMVMatch)
		x += length
		if x >= w4 {
			return 1
		}
		candB = &row[bx4+x]
		candBW4 = int32(blockDimensions[candB.bs].W4)
		length = max(step, candBW4)
	}
}

// scanCol ports dav1d's scan_col. rowBase is the row-table index of the first
// scanned row and bx4 the scanned column.
func (rt *refmvsTile) scanCol(stack *[8]refmvsCandidate, cnt *int32, ref refmvsRefPair, gmv *[2]refmvsMV, rowBase, bh4, h4, bx4, maxCols, step int32, haveNewMVMatch, haveRefMVMatch *int32) int32 {
	candB := &rt.buf[rt.rows[rowBase]+bx4]
	firstDim := blockDimensions[candB.bs]
	candBH4 := int32(firstDim.H4)
	length := max(step, min(bh4, candBH4))
	if bh4 <= candBH4 {
		weight := int32(2)
		if bh4 != 1 {
			weight = max(2, min(2*maxCols, int32(firstDim.W4)))
		}
		addSpatialCandidate(stack, cnt, length*weight, candB, ref, gmv, haveNewMVMatch, haveRefMVMatch)
		return weight >> 1
	}
	for y := int32(0); ; {
		addSpatialCandidate(stack, cnt, length*2, candB, ref, gmv, haveNewMVMatch, haveRefMVMatch)
		y += length
		if y >= h4 {
			return 1
		}
		candB = &rt.buf[rt.rows[rowBase+y]+bx4]
		candBH4 = int32(blockDimensions[candB.bs].H4)
		length = max(step, candBH4)
	}
}

var refmvsDivMult = [32]int32{
	0, 16384, 8192, 5461, 4096, 3276, 2730, 2340,
	2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
	1024, 963, 910, 862, 819, 780, 744, 712,
	682, 655, 630, 606, 585, 564, 546, 528,
}

// refmvsMVProjection ports dav1d's mv_projection.
func refmvsMVProjection(mv motion.Vector, num, den int32) refmvsMV {
	frac := num * refmvsDivMult[den&31]
	y := int32(mv.Row) * frac
	x := int32(mv.Col) * frac
	return makeRefmvsMV(
		min(max((y+8192+(y>>31))>>14, -0x3fff), 0x3fff),
		min(max((x+8192+(x>>31))>>14, -0x3fff), 0x3fff),
	)
}

// fixMVPrecision ports dav1d's fix_mv_precision.
func (rf *refmvsFrame) fixMVPrecision(m refmvsMV) refmvsMV {
	if rf.forceIntegerMV {
		// fix_int_mv_precision: (v - (v >> 15) + 3) & ~7
		r, c := m.row(), m.col()
		return makeRefmvsMV((r-(r>>15)+3)&^7, (c-(c>>15)+3)&^7)
	}
	if !rf.allowHighPrecisionMV {
		r, c := m.row(), m.col()
		return makeRefmvsMV((r-(r>>15))&^1, (c-(c>>15))&^1)
	}
	return m
}

func (rf *refmvsFrame) addTemporalCandidate(stack *[8]refmvsCandidate, cnt *int32, rb *TemporalMotionEntry, ref refmvsRefPair, globalMVCtx *int32, gmv *[2]refmvsMV) {
	if !rb.Valid {
		return
	}
	den := int32(rb.RefFrameOffset)
	mv := rf.fixMVPrecision(refmvsMVProjection(rb.MV, int32(rf.pocDiff[ref.ref0()-1]), den))
	last := *cnt
	if ref.ref1() == -1 {
		if globalMVCtx != nil {
			dx := mv.col() - gmv[0].col()
			dy := mv.row() - gmv[0].row()
			if dx < 0 {
				dx = -dx
			}
			if dy < 0 {
				dy = -dy
			}
			if dx|dy >= 16 {
				*globalMVCtx = 1
			} else {
				*globalMVCtx = 0
			}
		}
		for n := int32(0); n < last; n++ {
			if stack[n].mv[0] == mv {
				stack[n].weight += 2
				return
			}
		}
		if last < 8 {
			stack[last].mv[0] = mv
			stack[last].weight = 2
			*cnt = last + 1
		}
		return
	}
	mvp := [2]refmvsMV{mv, rf.fixMVPrecision(refmvsMVProjection(rb.MV, int32(rf.pocDiff[ref.ref1()-1]), den))}
	for n := int32(0); n < last; n++ {
		if stack[n].mv == mvp {
			stack[n].weight += 2
			return
		}
	}
	if last < 8 {
		stack[last].mv = mvp
		stack[last].weight = 2
		*cnt = last + 1
	}
}

// addCompoundExtendedCandidate ports dav1d's add_compound_extended_candidate.
// same points at mvstack[*cnt]; diff is same[2:]. sameCount is dav1d's
// same_count[4] (entries 2/3 are diff_count[0/1]).
func addCompoundExtendedCandidate(same []refmvsCandidate, sameCount *[4]int32, candB *refmvsBlock, sign0, sign1 uint8, ref refmvsRefPair, signBias *[7]uint8) {
	diff := same[2:]
	for n := range 2 {
		candRef := candB.ref.at(n)
		if candRef <= 0 {
			break
		}
		candMV := candB.mv[n]
		if candRef == ref.ref0() {
			if sameCount[0] < 2 {
				same[sameCount[0]].mv[0] = candMV
				sameCount[0]++
			}
			if sameCount[3] < 2 {
				if sign1^signBias[candRef-1] != 0 {
					candMV = candMV.negate()
				}
				diff[sameCount[3]].mv[1] = candMV
				sameCount[3]++
			}
		} else if candRef == ref.ref1() {
			if sameCount[1] < 2 {
				same[sameCount[1]].mv[1] = candMV
				sameCount[1]++
			}
			if sameCount[2] < 2 {
				if sign0^signBias[candRef-1] != 0 {
					candMV = candMV.negate()
				}
				diff[sameCount[2]].mv[0] = candMV
				sameCount[2]++
			}
		} else {
			iCandMV := candMV.negate()
			if sameCount[2] < 2 {
				v := candMV
				if sign0^signBias[candRef-1] != 0 {
					v = iCandMV
				}
				diff[sameCount[2]].mv[0] = v
				sameCount[2]++
			}
			if sameCount[3] < 2 {
				v := candMV
				if sign1^signBias[candRef-1] != 0 {
					v = iCandMV
				}
				diff[sameCount[3]].mv[1] = v
				sameCount[3]++
			}
		}
	}
}

// addSingleExtendedCandidate ports dav1d's add_single_extended_candidate.
func addSingleExtendedCandidate(stack *[8]refmvsCandidate, cnt *int32, candB *refmvsBlock, sign uint8, signBias *[7]uint8) {
	for n := range 2 {
		candRef := candB.ref.at(n)
		if candRef <= 0 {
			break
		}
		candMV := candB.mv[n]
		if sign^signBias[candRef-1] != 0 {
			candMV = candMV.negate()
		}
		last := *cnt
		m := int32(0)
		for ; m < last; m++ {
			if candMV == stack[m].mv[0] {
				break
			}
		}
		if m == last {
			stack[m].mv[0] = candMV
			stack[m].weight = 2
			*cnt = last + 1
		}
	}
}

// find ports dav1d_refmvs_find into out, which must be zero: as dav1d it
// writes only the stack entries it fills. ref is the target reference pair in
// dav1d's +1 encoding, tgmv the per-reference global MVs evaluated at this block
// (get_gmv_2d), topHasRight dav1d's EDGE_I444_TOP_HAS_RIGHT edge flag, and
// by4/bx4 the block's absolute 4x4 position.
func (rt *refmvsTile) find(out *refmvsResult, ref refmvsRefPair, size BlockSize, topHasRight bool, by4, bx4 int32, tgmvIn [2]motion.Vector) {
	rf := &rt.frame
	dims := blockDimensions[size]
	bw4 := int32(dims.W4)
	w4 := min(min(bw4, 16), rt.tileColEnd-bx4)
	bh4 := int32(dims.H4)
	h4 := min(min(bh4, 16), rt.tileRowEnd-by4)
	var gmv, tgmv [2]refmvsMV
	stack := &out.stack
	// out is zero (both callers pass a fresh refmvsResult): no stack clear.

	cnt := int32(0)
	if ref.ref0() > 0 {
		tgmv[0] = packRefmvsMV(tgmvIn[0])
		gmv[0] = refmvsInvalidMV
		if rf.gmType[ref.ref0()-1] > parser.GlobalMotionTranslation {
			gmv[0] = tgmv[0]
		}
	} else {
		tgmv[0] = 0
		gmv[0] = refmvsInvalidMV
	}
	if ref.ref1() > 0 {
		tgmv[1] = packRefmvsMV(tgmvIn[1])
		gmv[1] = refmvsInvalidMV
		if rf.gmType[ref.ref1()-1] > parser.GlobalMotionTranslation {
			gmv[1] = tgmv[1]
		}
	}

	rowBase := (by4 & 31) + 5

	// top
	haveNewMV, haveColMVs, haveRowMVs := int32(0), int32(0), int32(0)
	maxRows, nRows := int32(0), int32(-1)
	var bTop []refmvsBlock
	if by4 > rt.tileRowStart {
		maxRows = min((by4-rt.tileRowStart+1)>>1, 2+boolInt32(bh4 > 1))
		bTop = rt.row(rowBase - 1)
		step := int32(1)
		if bw4 >= 16 {
			step = 4
		}
		nRows = scanRow(stack, &cnt, ref, &gmv, bTop, bx4, bw4, w4, maxRows, step, &haveNewMV, &haveRowMVs)
	}

	// left
	maxCols, nCols := int32(0), int32(-1)
	if bx4 > rt.tileColStart {
		maxCols = min((bx4-rt.tileColStart+1)>>1, 2+boolInt32(bw4 > 1))
		step := int32(1)
		if bh4 >= 16 {
			step = 4
		}
		nCols = rt.scanCol(stack, &cnt, ref, &gmv, rowBase, bh4, h4, bx4-1, maxCols, step, &haveNewMV, &haveColMVs)
	}

	// top/right
	if nRows != -1 && topHasRight && max(bw4, bh4) <= 16 && bw4+bx4 < rt.tileColEnd {
		addSpatialCandidate(stack, &cnt, 4, &bTop[bx4+bw4], ref, &gmv, &haveNewMV, &haveRowMVs)
	}

	nearestMatch := haveColMVs + haveRowMVs
	nearestCnt := cnt
	for n := int32(0); n < nearestCnt; n++ {
		stack[n].weight += 640
	}

	// temporal
	globalMVCtx := boolInt32(rf.useRefFrameMVs)
	if entries, stride, rowBase := rt.temporalRows(by4 >> 1); entries != nil {
		by8, bx8 := by4>>1, bx4>>1
		rbi := rowBase + bx8
		rb := rbi
		stepH, stepV := int32(1), int32(1)
		if bw4 >= 16 {
			stepH = 2
		}
		if bh4 >= 16 {
			stepV = 2
		}
		w8 := min((w4+1)>>1, 8)
		h8 := min((h4+1)>>1, 8)
		if e := &entries[rb]; e.Valid {
			rf.addTemporalCandidate(stack, &cnt, e, ref, &globalMVCtx, &tgmv)
		}
		for x := stepH; x < w8; x += stepH {
			if e := &entries[rb+x]; e.Valid {
				rf.addTemporalCandidate(stack, &cnt, e, ref, nil, nil)
			}
		}
		rb += stride * stepV
		for y := stepV; y < h8; y += stepV {
			for x := int32(0); x < w8; x += stepH {
				if e := &entries[rb+x]; e.Valid {
					rf.addTemporalCandidate(stack, &cnt, e, ref, nil, nil)
				}
			}
			rb += stride * stepV
		}
		if min(bw4, bh4) >= 2 && max(bw4, bh4) < 16 {
			bh8, bw8 := bh4>>1, bw4>>1
			rb = rbi + bh8*stride
			hasBottom := by8+bh8 < min(rt.tileRowEnd>>1, (by8&^7)+8)
			if hasBottom && bx8-1 >= max(rt.tileColStart>>1, bx8&^7) {
				if e := &entries[rb-1]; e.Valid {
					rf.addTemporalCandidate(stack, &cnt, e, ref, nil, nil)
				}
			}
			if bx8+bw8 < min(rt.tileColEnd>>1, (bx8&^7)+8) {
				if hasBottom {
					if e := &entries[rb+bw8]; e.Valid {
						rf.addTemporalCandidate(stack, &cnt, e, ref, nil, nil)
					}
				}
				if by8+bh8-1 < min(rt.tileRowEnd>>1, (by8&^7)+8) {
					if e := &entries[rb+bw8-stride]; e.Valid {
						rf.addTemporalCandidate(stack, &cnt, e, ref, nil, nil)
					}
				}
			}
		}
	}

	// top/left (which, confusingly, is part of "secondary" references)
	var dummyNewMV int32
	if nRows != -1 && nCols != -1 {
		addSpatialCandidate(stack, &cnt, 4, &bTop[bx4-1], ref, &gmv, &dummyNewMV, &haveRowMVs)
	}

	// "secondary" (non-direct neighbour) top & left edges
	for n := int32(2); n <= 3; n++ {
		if uint32(n) > uint32(nRows) && n <= maxRows {
			step := int32(2)
			if bw4 >= 16 {
				step = 4
			}
			row := rt.row((((by4 & 31) - 2*n + 1) | 1) + 5)
			nRows += scanRow(stack, &cnt, ref, &gmv, row, bx4|1, bw4, w4, 1+maxRows-n, step, &dummyNewMV, &haveRowMVs)
		}
		if uint32(n) > uint32(nCols) && n <= maxCols {
			step := int32(2)
			if bh4 >= 16 {
				step = 4
			}
			nCols += rt.scanCol(stack, &cnt, ref, &gmv, ((by4&31)|1)+5, bh4, h4, (bx4-n*2+1)|1, 1+maxCols-n, step, &dummyNewMV, &haveColMVs)
		}
	}

	refMatchCount := haveColMVs + haveRowMVs

	// context build-up
	var refmvCtx, newmvCtx int32
	switch nearestMatch {
	case 0:
		refmvCtx = min(2, refMatchCount)
		newmvCtx = boolInt32(refMatchCount > 0)
	case 1:
		refmvCtx = min(refMatchCount*3, 4)
		newmvCtx = 3 - haveNewMV
	case 2:
		refmvCtx = 5
		newmvCtx = 5 - haveNewMV
	}

	// sorting (nearest, then "secondary")
	for length := nearestCnt; length != 0; {
		last := int32(0)
		for n := int32(1); n < length; n++ {
			if stack[n-1].weight < stack[n].weight {
				stack[n-1], stack[n] = stack[n], stack[n-1]
				last = n
			}
		}
		length = last
	}
	for length := cnt; length > nearestCnt; {
		last := nearestCnt
		for n := nearestCnt + 1; n < length; n++ {
			if stack[n-1].weight < stack[n].weight {
				stack[n-1], stack[n] = stack[n], stack[n-1]
				last = n
			}
		}
		length = last
	}

	left := -(bx4 + bw4 + 4) * 4 * 8
	right := (rf.iw4 - bx4 + 4) * 4 * 8
	top := -(by4 + bh4 + 4) * 4 * 8
	bottom := (rf.ih4 - by4 + 4) * 4 * 8

	if ref.ref1() > 0 {
		if cnt < 2 {
			sign0 := rf.signBias[ref.ref0()-1]
			sign1 := rf.signBias[ref.ref1()-1]
			sz4 := min(w4, h4)
			same := stack[cnt:]
			var sameCount [4]int32

			// non-self references in top
			if nRows != -1 {
				for x := int32(0); x < sz4; {
					candB := &bTop[bx4+x]
					addCompoundExtendedCandidate(same, &sameCount, candB, sign0, sign1, ref, &rf.signBias)
					x += int32(blockDimensions[candB.bs].W4)
				}
			}
			// non-self references in left
			if nCols != -1 {
				for y := int32(0); y < sz4; {
					candB := &rt.buf[rt.rows[rowBase+y]+bx4-1]
					addCompoundExtendedCandidate(same, &sameCount, candB, sign0, sign1, ref, &rf.signBias)
					y += int32(blockDimensions[candB.bs].H4)
				}
			}

			diff := same[2:]
			// merge together
			for n := range 2 {
				m := sameCount[n]
				if m >= 2 {
					continue
				}
				l := sameCount[2+n]
				if l != 0 {
					same[m].mv[n] = diff[0].mv[n]
					m++
					if m == 2 {
						continue
					}
					if l == 2 {
						same[1].mv[n] = diff[1].mv[n]
						continue
					}
				}
				for ; m < 2; m++ {
					same[m].mv[n] = tgmv[n]
				}
			}

			// if the first extended was the same as the non-extended one,
			// then replace it with the second extended one
			n := cnt
			if n == 1 && stack[0].mv == same[0].mv {
				stack[1].mv = stack[2].mv
			}
			for ; n < 2; n++ {
				stack[n].weight = 2
			}
			cnt = 2
		}

		// clamping
		for n := int32(0); n < cnt; n++ {
			for i := range 2 {
				m := stack[n].mv[i]
				stack[n].mv[i] = makeRefmvsMV(min(max(m.row(), top), bottom), min(max(m.col(), left), right))
			}
		}

		out.rawCtx = (refmvCtx << 4) | (globalMVCtx << 3) | newmvCtx
		switch refmvCtx >> 1 {
		case 0:
			out.ctx = min(newmvCtx, 1)
		case 1:
			out.ctx = 1 + min(newmvCtx, 3)
		case 2:
			out.ctx = min(max(3+newmvCtx, 4), 7)
		}
		out.count = cnt
		return
	} else if cnt < 2 && ref.ref0() > 0 {
		sign := rf.signBias[ref.ref0()-1]
		sz4 := min(w4, h4)

		// non-self references in top
		if nRows != -1 {
			for x := int32(0); x < sz4 && cnt < 2; {
				candB := &bTop[bx4+x]
				addSingleExtendedCandidate(stack, &cnt, candB, sign, &rf.signBias)
				x += int32(blockDimensions[candB.bs].W4)
			}
		}
		// non-self references in left
		if nCols != -1 {
			for y := int32(0); y < sz4 && cnt < 2; {
				candB := &rt.buf[rt.rows[rowBase+y]+bx4-1]
				addSingleExtendedCandidate(stack, &cnt, candB, sign, &rf.signBias)
				y += int32(blockDimensions[candB.bs].H4)
			}
		}
	}

	// clamping
	for n := int32(0); n < cnt; n++ {
		m := stack[n].mv[0]
		stack[n].mv[0] = makeRefmvsMV(min(max(m.row(), top), bottom), min(max(m.col(), left), right))
	}
	for n := cnt; n < 2; n++ {
		stack[n].mv[0] = tgmv[0]
	}
	out.ctx = (refmvCtx << 4) | (globalMVCtx << 3) | newmvCtx
	out.rawCtx = out.ctx
	out.count = cnt
}

func boolInt32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// refmvsTopHasRight is dav1d's EDGE_I444_TOP_HAS_RIGHT intra-edge flag for a
// block, carried down the partition walk by the intra edge tree.
func refmvsTopHasRight(sbSizeMIB uint8, block *BlockVisit) bool {
	has := block.Edge&edgeI444TopHasRight != 0
	if refmvsOracleEnabled {
		refmvsOracleTopHasRight(sbSizeMIB, block, has)
	}
	return has
}

// refmvsPocDiff ports dav1d's get_poc_diff.
func refmvsPocDiff(bits uint8, a, b uint8) int32 {
	if bits == 0 {
		return 0
	}
	mask := int32(1) << (bits - 1)
	diff := int32(a) - int32(b)
	return (diff & (mask - 1)) - (diff & mask)
}

// bindRefmvsForLoop prepares the tile's refmvs state for one block-loop
// request (one tile), porting the parts of dav1d_refmvs_init_frame consumed by
// dav1d_refmvs_find. It returns false when the request does not decode inter
// mode syntax, in which case the legacy context path is used unchanged.
func bindRefmvsForLoop(rt *refmvsTile, req *BlockLoopRequest) bool {
	rt.bound = false
	if req.FrameType != parser.FrameTypeInter && req.FrameType != parser.FrameTypeSwitch {
		return false
	}
	if !req.DecodePredictionModes || !req.DecodeInterModes || !req.DecodeMotionVectors ||
		req.FrameMICols == 0 || req.FrameMIRows == 0 {
		return false
	}
	frame := refmvsFrame{
		iw4:                  int32(req.FrameMICols),
		ih4:                  int32(req.FrameMIRows),
		useRefFrameMVs:       req.UseRefFrameMVS,
		allowHighPrecisionMV: req.AllowHighPrecisionMV,
		forceIntegerMV:       req.ForceIntegerMV,
	}
	if req.UseRefFrameMVS && req.TemporalMVs != nil {
		frame.temporal = req.TemporalMVs
	}
	if req.UseRefFrameMVS && req.TemporalReferences != nil {
		frame.projectTmvs = true
		frame.mfmv.init(req.TemporalReferences, req.OrderHintBits, req.CurrentOrderHint, &req.ReferenceOrderHints,
			int32(req.FrameMIRows)>>1, int32(req.FrameMICols)>>1)
		if refmvsOracleEnabled {
			frame.temporal = refmvsOracleWholeField(req)
		}
	}
	for i := range 7 {
		if req.RefSignBias[i] {
			frame.signBias[i] = 1
		}
		d := refmvsPocDiff(req.OrderHintBits, req.CurrentOrderHint, req.ReferenceOrderHints[i])
		frame.pocDiff[i] = int8(min(max(d, -31), 31))
		frame.gmType[i] = req.GlobalMotionTypes[i]
		p := &req.GlobalMotion[i]
		if p.Type != parser.GlobalMotionIdentity || p.Matrix[2] != 1<<globalMotionWarpedModelPrecBits {
			frame.hasNonIdentityGM = true
		}
		if req.GlobalMotionTypes[i] > parser.GlobalMotionTranslation {
			frame.hasWarpGM = true
		}
	}
	rt.bindFrame(frame, int32(req.FrameMICols)*4,
		int32(req.Walk.MIColStart), int32(req.Walk.MIColEnd),
		int32(req.Walk.MIRowStart), int32(req.Walk.MIRowEnd))
	return true
}

// saveTmvs ports dav1d_refmvs_save_tmvs / save_tmvs_c: after a tile
// superblock row is decoded, cache its projectable motion vectors for future
// frames' temporal MV prediction. Each 8x8 cell takes the odd 4x4 record
// (r[2y+1][2x+1]) of the rows; the second reference wins over the first, and
// only references in the past (refFrameSide 0, dav1d's mfmv_sign) with
// |mv| < 4096 are kept.
func (rt *refmvsTile) saveTmvs(dst *ReferenceMVFrame, refFrameSide *[referenceFrameCount]int8, colStart8, colEnd8, rowStart8, rowEnd8 int32) {
	rowEnd8 = min(rowEnd8, int32(dst.Rows))
	colEnd8 = min(colEnd8, int32(dst.Cols))
	stride := int32(dst.Stride)
	for y := rowStart8; y < rowEnd8; y++ {
		b := rt.row(6 + (y&15)*2)
		rp := dst.Entries[y*stride : y*stride+colEnd8]
		for x := colStart8; x < colEnd8; {
			cand := &b[x*2+1]
			end := min(x+(int32(blockDimensions[cand.bs].W4)+1)>>1, colEnd8)
			tmv := ReferenceMVEntry{Ref: ReferenceFrameNone}
			if r := cand.ref.ref1(); r > 0 && refFrameSide[r-1] == 0 && refmvsSaveable(cand.mv[1]) {
				tmv = ReferenceMVEntry{Ref: ReferenceFrame(r - 1), MV: cand.mv[1].vector(), Valid: true}
			} else if r := cand.ref.ref0(); r > 0 && refFrameSide[r-1] == 0 && refmvsSaveable(cand.mv[0]) {
				tmv = ReferenceMVEntry{Ref: ReferenceFrame(r - 1), MV: cand.mv[0].vector(), Valid: true}
			}
			for ; x < end; x++ {
				rp[x] = tmv
			}
		}
	}
}

// refmvsSaveable is save_tmvs_c's (abs(mv.y) | abs(mv.x)) < 4096.
func refmvsSaveable(m refmvsMV) bool {
	r, c := m.row(), m.col()
	if r < 0 {
		r = -r
	}
	if c < 0 {
		c = -c
	}
	return r|c < 4096
}
