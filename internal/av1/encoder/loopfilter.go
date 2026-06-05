package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const LoopFilterModeDeltas = 2

// LoopFilterDeltas is the persistent loop-filter ref/mode delta table.
type LoopFilterDeltas struct {
	Ref  [encoderRefFrames]int8
	Mode [LoopFilterModeDeltas]int8
}

// LoopFilterParams is the encoder-owned view of loop_filter_params().
type LoopFilterParams struct {
	LevelY [2]uint8
	LevelU uint8
	LevelV uint8

	Sharpness uint8

	ModeRefDeltaEnabled bool
	ModeRefDeltaUpdate  bool
	Deltas              LoopFilterDeltas
}

func LoopFilterParamsPayloadSize(seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) (int, error) {
	w := newSizingBitWriter()
	if err := writeLoopFilterParamsPayload(&w, seq, prefix, size.AllowIntrabc, allLossless, lf, previous); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendLoopFilterParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) ([]byte, error) {
	payloadSize, err := LoopFilterParamsPayloadSize(seq, prefix, size, allLossless, lf, previous)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeLoopFilterParamsPayload(&w, seq, prefix, size.AllowIntrabc, allLossless, lf, previous); err != nil {
		return dst, err
	}
	return out, nil
}

func LoopFilterParamsInterPayloadSize(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) (int, error) {
	w := newSizingBitWriter()
	if err := writeLoopFilterParamsPayload(&w, seq, prefix, false, allLossless, lf, previous); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendLoopFilterParamsInterPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) ([]byte, error) {
	payloadSize, err := LoopFilterParamsInterPayloadSize(seq, prefix, size, allLossless, lf, previous)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeLoopFilterParamsPayload(&w, seq, prefix, false, allLossless, lf, previous); err != nil {
		return dst, err
	}
	return out, nil
}

func writeLoopFilterParamsPayload(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix, allowIntrabc bool, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) error {
	base, err := validateLoopFilterParams(seq, prefix, allowIntrabc, allLossless, lf, previous)
	if err != nil {
		return err
	}
	if allLossless || allowIntrabc {
		return nil
	}

	if err := w.writeBits(uint64(lf.LevelY[0]), 6); err != nil {
		return err
	}
	if err := w.writeBits(uint64(lf.LevelY[1]), 6); err != nil {
		return err
	}
	if !seq.ColorConfig.MonoChrome && (lf.LevelY[0] != 0 || lf.LevelY[1] != 0) {
		if err := w.writeBits(uint64(lf.LevelU), 6); err != nil {
			return err
		}
		if err := w.writeBits(uint64(lf.LevelV), 6); err != nil {
			return err
		}
	}
	if err := w.writeBits(uint64(lf.Sharpness), 3); err != nil {
		return err
	}
	if err := w.writeBool(lf.ModeRefDeltaEnabled); err != nil {
		return err
	}
	if !lf.ModeRefDeltaEnabled {
		return nil
	}
	if err := w.writeBool(lf.ModeRefDeltaUpdate); err != nil {
		return err
	}
	if !lf.ModeRefDeltaUpdate {
		return nil
	}
	return writeLoopFilterDeltaUpdates(w, base, lf.Deltas)
}

func writeLoopFilterDeltaUpdates(w *bitWriter, base LoopFilterDeltas, deltas LoopFilterDeltas) error {
	for i := uint8(0); i < encoderRefFrames; i++ {
		update := deltas.Ref[i] != base.Ref[i]
		if err := w.writeBool(update); err != nil {
			return err
		}
		if update {
			if err := writeSignedBits(w, int16(deltas.Ref[i]), 7); err != nil {
				return err
			}
		}
	}
	for i := uint8(0); i < LoopFilterModeDeltas; i++ {
		update := deltas.Mode[i] != base.Mode[i]
		if err := w.writeBool(update); err != nil {
			return err
		}
		if update {
			if err := writeSignedBits(w, int16(deltas.Mode[i]), 7); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateLoopFilterParams(seq SequenceHeader, prefix FrameHeaderPrefix, allowIntrabc bool, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) (LoopFilterDeltas, error) {
	if lf.LevelY[0] > 63 || lf.LevelY[1] > 63 || lf.LevelU > 63 || lf.LevelV > 63 || lf.Sharpness > 7 {
		return LoopFilterDeltas{}, ErrInvalidFrame
	}
	base, err := loopFilterDeltaBase(prefix, previous)
	if err != nil {
		return LoopFilterDeltas{}, err
	}
	if allLossless || allowIntrabc {
		if lf.LevelY[0] != 0 || lf.LevelY[1] != 0 || lf.LevelU != 0 || lf.LevelV != 0 || lf.Sharpness != 0 ||
			!lf.ModeRefDeltaEnabled || !lf.ModeRefDeltaUpdate || lf.Deltas != defaultLoopFilterDeltas() {
			return LoopFilterDeltas{}, ErrInvalidFrame
		}
		return base, nil
	}
	if seq.ColorConfig.MonoChrome && (lf.LevelU != 0 || lf.LevelV != 0) {
		return LoopFilterDeltas{}, ErrInvalidFrame
	}
	if lf.LevelY[0] == 0 && lf.LevelY[1] == 0 && (lf.LevelU != 0 || lf.LevelV != 0) {
		return LoopFilterDeltas{}, ErrInvalidFrame
	}
	if !lf.ModeRefDeltaEnabled {
		if lf.ModeRefDeltaUpdate || lf.Deltas != base {
			return LoopFilterDeltas{}, ErrInvalidFrame
		}
		return base, nil
	}
	if !lf.ModeRefDeltaUpdate && lf.Deltas != base {
		return LoopFilterDeltas{}, ErrInvalidFrame
	}
	if lf.ModeRefDeltaUpdate {
		if err := validateLoopFilterDeltas(lf.Deltas); err != nil {
			return LoopFilterDeltas{}, err
		}
	}
	return base, nil
}

func loopFilterDeltaBase(prefix FrameHeaderPrefix, previous *LoopFilterDeltas) (LoopFilterDeltas, error) {
	if prefix.PrimaryRefFrame == EncoderPrimaryRefNone {
		return defaultLoopFilterDeltas(), nil
	}
	if previous == nil {
		return LoopFilterDeltas{}, ErrInvalidFrame
	}
	return *previous, nil
}

func validateLoopFilterDeltas(deltas LoopFilterDeltas) error {
	for i := uint8(0); i < encoderRefFrames; i++ {
		if err := validateSignedFeature(int16(deltas.Ref[i]), 7); err != nil {
			return err
		}
	}
	for i := uint8(0); i < LoopFilterModeDeltas; i++ {
		if err := validateSignedFeature(int16(deltas.Mode[i]), 7); err != nil {
			return err
		}
	}
	return nil
}

func defaultLoopFilterDeltas() LoopFilterDeltas {
	return LoopFilterDeltas{
		Ref:  [encoderRefFrames]int8{1, 0, 0, 0, -1, 0, -1, -1},
		Mode: [LoopFilterModeDeltas]int8{0, 0},
	}
}
