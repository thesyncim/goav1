package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// frame_obu.go packs a frame as one OBU_FRAME (frame_obu: frame_header_obu,
// byte_alignment, tile_group_obu), the way SVT-AV1 (and libaom) emit every
// frame that is not a show-existing frame. It replaces the separate
// OBU_FRAME_HEADER + OBU_TILE_GROUP pair, saving an OBU header, a size field
// and the header's trailing bit.

// interFrameOBUHeaderBytes is the byte length of an inter frame's
// uncompressed header zero-padded to a byte boundary (byte_alignment).
func interFrameOBUHeaderBytes(seq SequenceHeader, header InterFrameHeaderParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeInterFrameHeaderFields(&w, seq, header); err != nil {
		return 0, err
	}
	if err := w.writeZeroByteAlignment(); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func intraFrameOBUHeaderBytes(seq SequenceHeader, header IntraFrameHeaderParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeIntraFrameHeaderFields(&w, seq, header); err != nil {
		return 0, err
	}
	if err := w.writeZeroByteAlignment(); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

// appendFrameOBU appends one OBU_FRAME whose header (the intra header when
// intra is non-nil, else the inter one) occupies headerBytes, followed by the
// tile group payload group.
func appendFrameOBU(dst []byte, headerBytes int, group []byte, temporalID, spatialID uint8, seq *SequenceHeader, intra *IntraFrameHeaderParams, inter *InterFrameHeaderParams) ([]byte, error) {
	if temporalID > 7 || spatialID > 3 {
		return dst, ErrInvalidFrame
	}
	payloadSize := headerBytes + len(group)
	ext := temporalID != 0 || spatialID != 0
	hdr := 1
	if ext {
		hdr = 2
	}
	size := hdr + bitstream.LEB128Len(uint32(payloadSize)) + payloadSize
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+size]
	n, err := obu.PutHeader(out[off:], obu.Header{
		Type:         obu.TypeFrame,
		Extension:    ext,
		HasSizeField: true,
		TemporalID:   temporalID,
		SpatialID:    spatialID,
	})
	if err != nil {
		return dst, err
	}
	off += n
	n, err = bitstream.PutLEB128(out[off:], uint32(payloadSize))
	if err != nil {
		return dst, err
	}
	off += n
	w := newBitWriter(out[off : off+headerBytes])
	var err2 error
	if intra != nil {
		err2 = writeIntraFrameHeaderFields(&w, *seq, *intra)
	} else {
		err2 = writeInterFrameHeaderFields(&w, *seq, *inter)
	}
	if err2 != nil {
		return dst, err2
	}
	if err := w.writeZeroByteAlignment(); err != nil {
		return dst, err
	}
	copy(out[off+headerBytes:], group)
	return out, nil
}

// AppendLowOverheadInterFrameOBU appends one inter OBU_FRAME carrying the
// tile group payload group.
func AppendLowOverheadInterFrameOBU(dst []byte, seq SequenceHeader, header InterFrameHeaderParams, temporalID, spatialID uint8, group []byte) ([]byte, error) {
	headerBytes, err := interFrameOBUHeaderBytes(seq, header)
	if err != nil {
		return dst, err
	}
	return appendFrameOBU(dst, headerBytes, group, temporalID, spatialID, &seq, nil, &header)
}

// AppendLowOverheadIntraFrameTemporalUnit appends a temporal delimiter, the
// sequence header and one intra OBU_FRAME carrying group.
func AppendLowOverheadIntraFrameTemporalUnit(dst []byte, seq SequenceHeader, header IntraFrameHeaderParams, group []byte) ([]byte, error) {
	headerBytes, err := intraFrameOBUHeaderBytes(seq, header)
	if err != nil {
		return dst, err
	}
	out, err := AppendLowOverheadOBU(dst, OBU{Type: obu.TypeTemporalDelimiter})
	if err != nil {
		return dst, err
	}
	out, err = AppendLowOverheadSequenceHeaderOBU(out, seq)
	if err != nil {
		return dst, err
	}
	out, err = appendFrameOBU(out, headerBytes, group, 0, 0, &seq, &header, nil)
	if err != nil {
		return dst, err
	}
	return out, nil
}
