package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const MaxCDEFStrengths = 8

// CDEFParams is the encoder-owned view of cdef_params().
type CDEFParams struct {
	Damping    uint8
	Bits       uint8
	YStrength  [MaxCDEFStrengths]uint8
	UVStrength [MaxCDEFStrengths]uint8
}

func CDEFParamsPayloadSize(seq SequenceHeader, size IntraFrameSize, allLossless bool, cdef CDEFParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeCDEFParamsPayload(&w, seq, size.AllowIntrabc, allLossless, cdef); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendCDEFParamsPayload(dst []byte, seq SequenceHeader, size IntraFrameSize, allLossless bool, cdef CDEFParams) ([]byte, error) {
	payloadSize, err := CDEFParamsPayloadSize(seq, size, allLossless, cdef)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeCDEFParamsPayload(&w, seq, size.AllowIntrabc, allLossless, cdef); err != nil {
		return dst, err
	}
	return out, nil
}

func CDEFParamsInterPayloadSize(seq SequenceHeader, size InterFrameSize, allLossless bool, cdef CDEFParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeCDEFParamsPayload(&w, seq, false, allLossless, cdef); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendCDEFParamsInterPayload(dst []byte, seq SequenceHeader, size InterFrameSize, allLossless bool, cdef CDEFParams) ([]byte, error) {
	payloadSize, err := CDEFParamsInterPayloadSize(seq, size, allLossless, cdef)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeCDEFParamsPayload(&w, seq, false, allLossless, cdef); err != nil {
		return dst, err
	}
	return out, nil
}

func writeCDEFParamsPayload(w *bitWriter, seq SequenceHeader, allowIntrabc bool, allLossless bool, cdef CDEFParams) error {
	if err := validateCDEFParams(seq, allowIntrabc, allLossless, cdef); err != nil {
		return err
	}
	if allLossless || !seq.EnableCDEF || allowIntrabc {
		return nil
	}
	if err := w.writeBits(uint64(cdef.Damping-3), 2); err != nil {
		return err
	}
	if err := w.writeBits(uint64(cdef.Bits), 2); err != nil {
		return err
	}
	strengthCount := uint8(1) << cdef.Bits
	for i := uint8(0); i < strengthCount; i++ {
		if err := w.writeBits(uint64(cdef.YStrength[i]), 6); err != nil {
			return err
		}
		if !seq.ColorConfig.MonoChrome {
			if err := w.writeBits(uint64(cdef.UVStrength[i]), 6); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateCDEFParams(seq SequenceHeader, allowIntrabc bool, allLossless bool, cdef CDEFParams) error {
	if allLossless || !seq.EnableCDEF || allowIntrabc {
		if cdef != (CDEFParams{}) {
			return ErrInvalidFrame
		}
		return nil
	}
	if cdef.Damping < 3 || cdef.Damping > 6 || cdef.Bits > 3 {
		return ErrInvalidFrame
	}
	strengthCount := uint8(1) << cdef.Bits
	for i := uint8(0); i < MaxCDEFStrengths; i++ {
		if cdef.YStrength[i] > 63 || cdef.UVStrength[i] > 63 {
			return ErrInvalidFrame
		}
		if i >= strengthCount && (cdef.YStrength[i] != 0 || cdef.UVStrength[i] != 0) {
			return ErrInvalidFrame
		}
		if seq.ColorConfig.MonoChrome && cdef.UVStrength[i] != 0 {
			return ErrInvalidFrame
		}
	}
	return nil
}
