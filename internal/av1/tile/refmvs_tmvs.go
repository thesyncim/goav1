package tile

import "unsafe"

// This file ports dav1d's temporal-MV projection: the mfmv setup of
// dav1d_refmvs_init_frame and load_tmvs_c, which projects the saved motion
// of up to three reference frames into a 16-row rp_proj window one tile
// superblock row at a time, right before that row is decoded.

// refmvsInvalidRef2Cur is dav1d's INVALID_REF2CUR.
const refmvsInvalidRef2Cur = -32

// refmvsMfmv is the temporal-projection part of dav1d's refmvs_frame.
type refmvsMfmv struct {
	n       int32
	ref     [3]uint8 // mfmv_ref, 0-based reference index
	ref2cur [3]int32
	ref2ref [3][7]int32
	// rpRef holds the reference frames' saved MV_REF side data (dav1d's
	// rp_ref): nil when the reference is intra-only, unavailable or of a
	// different size.
	rpRef [7]*ReferenceMVFrame
	// iw8 and ih8 are the frame size in 8x8 units.
	iw8, ih8 int32
}

// init ports the temporal MV setup of dav1d_refmvs_init_frame. poc is the
// current frame's order hint and refPoc the references' order hints.
func (m *refmvsMfmv) init(refs *[referenceFrameCount]TemporalMotionReferenceFrame, bits uint8, poc uint8, refPoc *[referenceFrameCount]uint8, ih8, iw8 int32) {
	*m = refmvsMfmv{iw8: iw8, ih8: ih8}
	if bits == 0 {
		return
	}
	for i := range refs {
		f := refs[i].Frame
		if f == nil || refs[i].IntraOnly || int32(f.Rows) != ih8 || int32(f.Cols) != iw8 || f.Validate() != nil {
			continue
		}
		m.rpRef[i] = f
	}
	add := func(ref uint8) {
		m.ref[m.n] = ref
		m.n++
	}
	total := int32(2)
	if m.rpRef[0] != nil && refs[0].RefOrderHints[6] != refPoc[3] { // alt-of-last != gold
		add(0) // last
		total = 3
	}
	if m.rpRef[4] != nil && refmvsPocDiff(bits, refPoc[4], poc) > 0 {
		add(4) // bwd
	}
	if m.rpRef[5] != nil && refmvsPocDiff(bits, refPoc[5], poc) > 0 {
		add(5) // altref2
	}
	if m.n < total && m.rpRef[6] != nil && refmvsPocDiff(bits, refPoc[6], poc) > 0 {
		add(6) // altref
	}
	if m.n < total && m.rpRef[1] != nil {
		add(1) // last2
	}
	for n := int32(0); n < m.n; n++ {
		ref := m.ref[n]
		rpoc := refPoc[ref]
		diff1 := refmvsPocDiff(bits, rpoc, poc)
		if diff1 < -31 || diff1 > 31 {
			m.ref2cur[n] = refmvsInvalidRef2Cur
			continue
		}
		if ref < 4 {
			m.ref2cur[n] = -diff1
		} else {
			m.ref2cur[n] = diff1
		}
		for k := range 7 {
			diff2 := refmvsPocDiff(bits, rpoc, refs[ref].RefOrderHints[k])
			// unsigned comparison also catches the < 0 case
			if uint32(diff2) > 31 {
				diff2 = 0
			}
			m.ref2ref[n][k] = diff2
		}
	}
}

// temporalRows returns the projected temporal samples refmvs_find reads for
// the 8x8 row by8: the rp_proj window when the tile projects per superblock
// row, else a whole-frame field.
func (rt *refmvsTile) temporalRows(by8 int32) ([]TemporalMotionEntry, int32, int32) {
	rf := &rt.frame
	if rf.projectTmvs {
		if rf.mfmv.n == 0 {
			return nil, 0, 0
		}
		return rt.rpProj, rt.rpStride, (by8 & 15) * rt.rpStride
	}
	if tmv := rf.temporal; tmv != nil {
		stride := int32(tmv.Stride)
		return tmv.Entries, stride, by8 * stride
	}
	return nil, 0, 0
}

func refmvsAbs(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}

// loadTmvs ports dav1d's load_tmvs (load_tmvs_c and the arm64
// load_tmvs_neon) for one tile superblock row (8x8 units). As in the asm, a
// source row is walked as one slice, the projection runs on the div_mult
// table with the clamp folded into the >> 6 offset, and runs of identical
// entries are skipped with whole-entry compares. It stays out of line, a
// leaf like the asm, so its loop state keeps registers instead of sharing
// the block loop's.
//
//go:noinline
func (rt *refmvsTile) loadTmvs(colStart8, colEnd8, rowStart8, rowEnd8 int32) {
	m := &rt.frame.mfmv
	rowEnd8 = min(rowEnd8, m.ih8)
	colEnd8 = min(colEnd8, m.iw8)
	colStart8i := max(colStart8-8, 0)
	colEnd8i := min(colEnd8+8, m.iw8)
	stride := rt.rpStride
	proj := rt.rpProj
	for y := rowStart8; y < rowEnd8; y++ {
		row := proj[(y&15)*stride:]
		clear(row[colStart8:colEnd8])
	}
	for n := int32(0); n < m.n; n++ {
		ref2cur := m.ref2cur[n]
		if ref2cur == refmvsInvalidRef2Cur {
			continue
		}
		ref := m.ref[n]
		refSign := int32(ref) - 4
		ref2refs := &m.ref2ref[n]
		src := m.rpRef[ref&7]
		srcStride := int32(src.Stride)
		for y := rowStart8; y < rowEnd8; y++ {
			ySBAlign := y &^ 7
			yProjStart := max(ySBAlign, rowStart8)
			yProjEnd := min(ySBAlign+8, rowEnd8)
			r := src.Entries[y*srcStride+colStart8i : y*srcStride+colEnd8i]
			for i := 0; i < len(r); {
				rb := r[i]
				if !rb.Valid {
					i++
					continue
				}
				// dav1d compares rb->ref and rb->mv.n; one 64-bit
				// compare covers the whole entry.
				rbBits := refMVEntryBits(&r[i])
				ref2ref := ref2refs[uint8(rb.Ref)]
				if ref2ref == 0 {
					i++
					continue
				}
				// mv_projection with apply_sign(iclip(abs(offset) >> 6,
				// 0, 0x3fff >> 6), offset ^ ref_sign)
				frac := ref2cur * refmvsDivMult[ref2ref&31]
				oy := int32(rb.MV.Row) * frac
				ox := int32(rb.MV.Col) * frac
				oy = (oy + 8192 + oy>>31) >> 14
				ox = (ox + 8192 + ox>>31) >> 14
				dy := min(refmvsAbs(oy)>>6, 0x3fff>>6)
				dx := min(refmvsAbs(ox)>>6, 0x3fff>>6)
				if oy^refSign < 0 {
					dy = -dy
				}
				if ox^refSign < 0 {
					dx = -dx
				}
				posY := y + dy
				if posY >= yProjStart && posY < yProjEnd {
					x := colStart8i + int32(i)
					posX := x + dx
					dst := proj[(posY&15)*stride:]
					entry := TemporalMotionEntry{MV: rb.MV, RefFrameOffset: uint8(ref2ref), Valid: true}
					for {
						xSBAlign := x &^ 7
						if posX >= max(xSBAlign-8, colStart8) && posX < min(xSBAlign+16, colEnd8) {
							dst[posX] = entry
						}
						i++
						if i >= len(r) || refMVEntryBits(&r[i]) != rbBits {
							break
						}
						x++
						posX++
					}
				} else {
					for i++; i < len(r) && refMVEntryBits(&r[i]) == rbBits; i++ {
					}
				}
			}
		}
	}
	if refmvsOracleEnabled {
		refmvsOracleTmvs(rt, colStart8, colEnd8, rowStart8, rowEnd8)
	}
}

// refMVEntryBits loads a ReferenceMVEntry as one word; the struct has no
// implicit padding and its pad bytes stay zero, so equal words are equal
// entries.
func refMVEntryBits(e *ReferenceMVEntry) uint64 {
	return *(*uint64)(unsafe.Pointer(e))
}
