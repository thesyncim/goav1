// Ported from libaom: aom/src/obu_util.c (sketch helpers for Annex B emit /
// ingest) and the AV1 specification Annex B framing rules.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package obu

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// AppendAnnexBOBU appends a single Annex B size-prefixed OBU element to dst.
//
// The unit raw bytes are emitted verbatim (preserving the obu_has_size_field
// bit and embedded obu_size leb128 if present). The leb128 obu_unit_size that
// precedes the element is written from the length of raw.
//
// This helper is suitable for converting one OBU at a time. For nesting under
// frame_units and temporal_units use AppendAnnexBTemporalUnit which writes
// the surrounding length prefixes.
func AppendAnnexBOBU(dst []byte, raw []byte) []byte {
	dst = bitstream.AppendLEB128(dst, uint32(len(raw)))
	dst = append(dst, raw...)
	return dst
}

// AppendAnnexBFrameUnit appends a frame_unit (one or more OBU units) to dst
// using the Annex B framing per spec section 5.2. obus is treated as the
// concatenation of one or more OBU raw byte slices.
func AppendAnnexBFrameUnit(dst []byte, obus ...[]byte) []byte {
	size := 0
	for _, o := range obus {
		size += bitstream.LEB128Len(uint32(len(o))) + len(o)
	}
	dst = bitstream.AppendLEB128(dst, uint32(size))
	for _, o := range obus {
		dst = AppendAnnexBOBU(dst, o)
	}
	return dst
}

// AppendAnnexBTemporalUnit appends a temporal_unit (a slice of frame_units, each
// of which is a slice of OBU raw byte slices) to dst using the Annex B framing
// per spec section 5.2. The temporal_unit_size, frame_unit_size, and
// obu_unit_size leb128 prefixes are emitted in order.
func AppendAnnexBTemporalUnit(dst []byte, frameUnits [][][]byte) []byte {
	tuSize := 0
	for _, fu := range frameUnits {
		fuSize := 0
		for _, o := range fu {
			fuSize += bitstream.LEB128Len(uint32(len(o))) + len(o)
		}
		tuSize += bitstream.LEB128Len(uint32(fuSize)) + fuSize
	}
	dst = bitstream.AppendLEB128(dst, uint32(tuSize))
	for _, fu := range frameUnits {
		dst = AppendAnnexBFrameUnit(dst, fu...)
	}
	return dst
}

// AnnexBFrameSpan describes one frame_unit boundary in the source low-overhead
// stream: [Start, End) bytes inside src. Exported for caller-provided scratch
// reuse from LowOverheadToAnnexB.
type AnnexBFrameSpan struct {
	Start int
	End   int
}

// LowOverheadToAnnexB converts a low-overhead OBU stream (the kind emitted by
// IVF / Section 5 temporal-unit framing) into Annex B framing per spec section
// 5.2.
//
// Each OBU_TEMPORAL_DELIMITER in src begins a new temporal_unit. Frame_units
// are formed by grouping each frame header / frame OBU with the OBUs that
// follow it up to the next frame header / temporal delimiter (mirroring the
// section 5 grouping used by libaom).
//
// The returned slice is dst extended in place. scratch is an optional caller
// scratch slice used for frame-span bookkeeping; when its capacity is at
// least the number of frame_units in src, LowOverheadToAnnexB makes no
// auxiliary allocations.
func LowOverheadToAnnexB(dst []byte, src []byte, scratch []AnnexBFrameSpan) ([]byte, error) {
	scratch = scratch[:0]

	// Walk the source once, recording frame_unit boundaries. A TD ends the
	// previous TU; emit immediately after we see one.
	it := NewLowOverheadIterator(src)
	off := 0
	curFrameStart := -1
	tuFrameStart := 0 // index into scratch where the current TU begins.

	// frameUnitEmitSize computes the encoded frame_unit content size (sum of
	// leb128(obu_len) + obu_len for every OBU inside the [start, end) span)
	// while validating each header has obu_has_size_field set.
	frameUnitEmitSize := func(fr AnnexBFrameSpan) (int, error) {
		emit := 0
		subOff := fr.Start
		for subOff < fr.End {
			h, hLen, perr := ParseHeader(src[subOff:fr.End])
			if perr != nil {
				return 0, perr
			}
			if !h.HasSizeField {
				return 0, ErrMissingSizeField
			}
			size, sLen, perr := bitstream.ReadLEB128(src[subOff+hLen : fr.End])
			if perr != nil {
				return 0, perr
			}
			obuLen := hLen + sLen + int(size)
			if obuLen <= 0 || subOff+obuLen > fr.End {
				return 0, ErrShortPayload
			}
			emit += bitstream.LEB128Len(uint32(obuLen)) + obuLen
			subOff += obuLen
		}
		if subOff != fr.End {
			return 0, ErrSizeMismatch
		}
		return emit, nil
	}

	emitTU := func() error {
		if tuFrameStart == len(scratch) {
			return nil
		}
		// Compute tu_size = sum(frame_unit_size + leb128(frame_unit_size))
		// where frame_unit_size is the *encoded* frame unit content length.
		// Each frame_unit is walked twice: once to size it (also validates
		// the per-OBU headers) and once to emit it. The double walk avoids
		// any per-TU heap allocation for the per-frame size cache.
		tuSize := 0
		for i := tuFrameStart; i < len(scratch); i++ {
			emit, perr := frameUnitEmitSize(scratch[i])
			if perr != nil {
				return perr
			}
			tuSize += bitstream.LEB128Len(uint32(emit)) + emit
		}
		dst = bitstream.AppendLEB128(dst, uint32(tuSize))
		for i := tuFrameStart; i < len(scratch); i++ {
			fr := scratch[i]
			emit, _ := frameUnitEmitSize(fr)
			dst = bitstream.AppendLEB128(dst, uint32(emit))
			subOff := fr.Start
			for subOff < fr.End {
				h, hLen, _ := ParseHeader(src[subOff:fr.End])
				_ = h
				size, sLen, _ := bitstream.ReadLEB128(src[subOff+hLen : fr.End])
				obuLen := hLen + sLen + int(size)
				dst = AppendAnnexBOBU(dst, src[subOff:subOff+obuLen])
				subOff += obuLen
			}
		}
		// Reset for the next TU.
		scratch = scratch[:tuFrameStart]
		return nil
	}

	closeFrame := func(end int) {
		if curFrameStart < 0 {
			return
		}
		scratch = append(scratch, AnnexBFrameSpan{Start: curFrameStart, End: end})
		curFrameStart = -1
	}

	for {
		start := off
		unit, ok, err := it.Next()
		if err != nil {
			return dst, err
		}
		if !ok {
			closeFrame(off)
			if err := emitTU(); err != nil {
				return dst, err
			}
			break
		}
		off = start + len(unit.Raw)

		switch unit.Header.Type {
		case TypeTemporalDelimiter:
			closeFrame(start)
			if err := emitTU(); err != nil {
				return dst, err
			}
			tuFrameStart = len(scratch)
			curFrameStart = start
		case TypeFrameHeader, TypeFrame, TypeRedundantFrameHeader:
			if curFrameStart < 0 {
				curFrameStart = start
				break
			}
			// A frame header inside an open frame_unit starts a new frame.
			closeFrame(start)
			curFrameStart = start
		default:
			if curFrameStart < 0 {
				curFrameStart = start
			}
		}
	}

	return dst, nil
}

// AnnexBToLowOverhead converts an Annex B stream into the concatenated
// low-overhead OBU stream consumable by NewLowOverheadIterator.
//
// The returned slice is dst extended in place. Each emitted OBU has its
// obu_has_size_field bit forced on and an obu_size leb128 prefix written
// before the payload bytes so that the result is directly consumable by
// NewLowOverheadIterator (matching NormalizeLowOverhead's behaviour for a
// single OBU). Annex B size prefixes are stripped.
//
// AnnexBToLowOverhead performs no allocation beyond growing dst.
func AnnexBToLowOverhead(dst []byte, src []byte) ([]byte, error) {
	it := NewAnnexBIterator(src)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			return dst, err
		}
		if !ok {
			return dst, nil
		}
		header := unit.OBU.Header
		if header.HasSizeField {
			// Source already carries obu_size; emit raw bytes unchanged.
			dst = append(dst, unit.Raw...)
			continue
		}
		// Restore the size field. Grow dst to fit header + leb128 + payload.
		header.HasSizeField = true
		var hdrBuf [2]byte
		hLen, perr := PutHeader(hdrBuf[:], header)
		if perr != nil {
			return dst, perr
		}
		payload := unit.OBU.Payload
		dst = append(dst, hdrBuf[:hLen]...)
		dst = bitstream.AppendLEB128(dst, uint32(len(payload)))
		dst = append(dst, payload...)
	}
}
