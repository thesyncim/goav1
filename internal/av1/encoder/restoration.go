package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const RestorationUnitMax = 256

type RestorationType uint8

const (
	RestorationNone RestorationType = iota
	RestorationSwitchable
	RestorationWiener
	RestorationSGRProj
)

// RestorationParams is the encoder-owned view of loop_restoration_params().
type RestorationParams struct {
	Type [3]RestorationType

	UnitSizeYLog2  uint8
	UnitSizeUVLog2 uint8
	UnitSizeY      uint16
	UnitSizeUV     uint16
}

func RestorationParamsPayloadSize(seq SequenceHeader, size IntraFrameSize, allLossless bool, restoration RestorationParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeRestorationParamsPayload(&w, seq, size.AllowIntrabc, size.SuperResDenominator != 8, allLossless, restoration); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendRestorationParamsPayload(dst []byte, seq SequenceHeader, size IntraFrameSize, allLossless bool, restoration RestorationParams) ([]byte, error) {
	payloadSize, err := RestorationParamsPayloadSize(seq, size, allLossless, restoration)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeRestorationParamsPayload(&w, seq, size.AllowIntrabc, size.SuperResDenominator != 8, allLossless, restoration); err != nil {
		return dst, err
	}
	return out, nil
}

func RestorationParamsInterPayloadSize(seq SequenceHeader, size InterFrameSize, allLossless bool, restoration RestorationParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeRestorationParamsPayload(&w, seq, false, size.SuperResDenominator != 8, allLossless, restoration); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendRestorationParamsInterPayload(dst []byte, seq SequenceHeader, size InterFrameSize, allLossless bool, restoration RestorationParams) ([]byte, error) {
	payloadSize, err := RestorationParamsInterPayloadSize(seq, size, allLossless, restoration)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeRestorationParamsPayload(&w, seq, false, size.SuperResDenominator != 8, allLossless, restoration); err != nil {
		return dst, err
	}
	return out, nil
}

func writeRestorationParamsPayload(w *bitWriter, seq SequenceHeader, allowIntrabc bool, superResEnabled bool, allLossless bool, restoration RestorationParams) error {
	if err := validateRestorationParams(seq, allowIntrabc, superResEnabled, allLossless, restoration); err != nil {
		return err
	}
	if (allLossless && !superResEnabled) || !seq.EnableRestoration || allowIntrabc {
		return nil
	}
	if err := w.writeBits(uint64(restoration.Type[0]), 2); err != nil {
		return err
	}
	if !seq.ColorConfig.MonoChrome {
		if err := w.writeBits(uint64(restoration.Type[1]), 2); err != nil {
			return err
		}
		if err := w.writeBits(uint64(restoration.Type[2]), 2); err != nil {
			return err
		}
	}
	if !restorationAnyActive(restoration) {
		return nil
	}
	base := uint8(6)
	if seq.Use128x128Superblock {
		base = 7
	}
	if restoration.UnitSizeYLog2 == base {
		if err := w.writeBool(false); err != nil {
			return err
		}
	} else {
		if err := w.writeBool(true); err != nil {
			return err
		}
		if !seq.Use128x128Superblock {
			if err := w.writeBool(restoration.UnitSizeYLog2 == base+2); err != nil {
				return err
			}
		}
	}
	if !seq.ColorConfig.MonoChrome &&
		(restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) &&
		seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY {
		return w.writeBool(restoration.UnitSizeUVLog2+1 == restoration.UnitSizeYLog2)
	}
	return nil
}

func validateRestorationParams(seq SequenceHeader, allowIntrabc bool, superResEnabled bool, allLossless bool, restoration RestorationParams) error {
	if (allLossless && !superResEnabled) || !seq.EnableRestoration || allowIntrabc {
		if restoration != (RestorationParams{}) {
			return ErrInvalidFrame
		}
		return nil
	}
	for i := uint8(0); i < 3; i++ {
		if restoration.Type[i] > RestorationSGRProj {
			return ErrInvalidFrame
		}
	}
	if seq.ColorConfig.MonoChrome && (restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) {
		return ErrInvalidFrame
	}
	if !restorationAnyActive(restoration) {
		return validateRestorationNoneUnits(restoration)
	}
	base := uint8(6)
	if seq.Use128x128Superblock {
		base = 7
	}
	max := base + 2
	if seq.Use128x128Superblock {
		max = base + 1
	}
	if restoration.UnitSizeYLog2 < base || restoration.UnitSizeYLog2 > max {
		return ErrInvalidFrame
	}
	if restoration.UnitSizeY != 0 && restoration.UnitSizeY != 1<<restoration.UnitSizeYLog2 {
		return ErrInvalidFrame
	}
	wantUV := restoration.UnitSizeYLog2
	chromaActive420 := !seq.ColorConfig.MonoChrome &&
		(restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) &&
		seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY
	if chromaActive420 {
		if restoration.UnitSizeUVLog2 != restoration.UnitSizeYLog2 &&
			restoration.UnitSizeUVLog2+1 != restoration.UnitSizeYLog2 {
			return ErrInvalidFrame
		}
		wantUV = restoration.UnitSizeUVLog2
	} else if restoration.UnitSizeUVLog2 != 0 && restoration.UnitSizeUVLog2 != restoration.UnitSizeYLog2 {
		return ErrInvalidFrame
	}
	if restoration.UnitSizeUV != 0 && restoration.UnitSizeUV != 1<<wantUV {
		return ErrInvalidFrame
	}
	return nil
}

func validateRestorationNoneUnits(restoration RestorationParams) error {
	if restoration.UnitSizeYLog2 == 0 && restoration.UnitSizeUVLog2 == 0 &&
		restoration.UnitSizeY == 0 && restoration.UnitSizeUV == 0 {
		return nil
	}
	if restoration.UnitSizeYLog2 == 8 && restoration.UnitSizeUVLog2 == 8 &&
		restoration.UnitSizeY == RestorationUnitMax && restoration.UnitSizeUV == RestorationUnitMax {
		return nil
	}
	return ErrInvalidFrame
}

func restorationAnyActive(restoration RestorationParams) bool {
	return restoration.Type[0] != RestorationNone ||
		restoration.Type[1] != RestorationNone ||
		restoration.Type[2] != RestorationNone
}
