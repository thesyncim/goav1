package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	MaxFilmGrainYPoints  = parser.MaxFilmGrainYPoints
	MaxFilmGrainUVPoints = parser.MaxFilmGrainUVPoints
	MaxFilmGrainYCoeffs  = parser.MaxFilmGrainYCoeffs
	MaxFilmGrainUVCoeffs = parser.MaxFilmGrainUVCoeffs
)

type FilmGrainParams struct {
	Apply  bool
	Update bool
	Seed   uint16
	RefIdx uint8

	ChromaScalingFromLuma bool
	NumYPoints            uint8
	YPoints               [MaxFilmGrainYPoints][2]uint8
	NumCbPoints           uint8
	CbPoints              [MaxFilmGrainUVPoints][2]uint8
	NumCrPoints           uint8
	CrPoints              [MaxFilmGrainUVPoints][2]uint8

	ScalingShift    uint8
	ARCoeffLag      uint8
	ARCoeffsY       [MaxFilmGrainYCoeffs]int8
	ARCoeffsCb      [MaxFilmGrainUVCoeffs]int8
	ARCoeffsCr      [MaxFilmGrainUVCoeffs]int8
	ARCoeffShift    uint8
	GrainScaleShift uint8

	CbMult     uint8
	CbLumaMult uint8
	CbOffset   uint16
	CrMult     uint8
	CrLumaMult uint8
	CrOffset   uint16

	Overlap               bool
	ClipToRestrictedRange bool
}

func FilmGrainParamsPayloadSize(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params FilmGrainParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeFilmGrainParamsPayload(&w, seq, prefix, size, refs, params); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendFilmGrainParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params FilmGrainParams) ([]byte, error) {
	payloadSize, err := FilmGrainParamsPayloadSize(seq, prefix, size, refs, params)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeFilmGrainParamsPayload(&w, seq, prefix, size, refs, params); err != nil {
		return dst, err
	}
	return out, nil
}

func writeFilmGrainParamsPayload(w *bitWriter, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params FilmGrainParams) error {
	if err := validateFilmGrainParams(seq, prefix, size, refs, params); err != nil {
		return err
	}
	if filmGrainParamsSkipped(seq, prefix) {
		return nil
	}
	if err := w.writeBool(params.Apply); err != nil {
		return err
	}
	if !params.Apply {
		return nil
	}
	if err := w.writeBits(uint64(params.Seed), 16); err != nil {
		return err
	}
	if prefix.FrameType == FrameHeaderTypeInter {
		if err := w.writeBool(params.Update); err != nil {
			return err
		}
	}
	if !params.Update {
		return w.writeBits(uint64(params.RefIdx), 3)
	}
	return writeFilmGrainUpdate(w, seq, params)
}

func validateFilmGrainParams(seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params FilmGrainParams) error {
	if err := validateSequenceHeader(seq); err != nil {
		return err
	}
	if !prefix.FrameType.valid() {
		return ErrInvalidFrame
	}
	if filmGrainParamsSkipped(seq, prefix) {
		if params != (FilmGrainParams{}) {
			return ErrInvalidFrame
		}
		return nil
	}
	if !params.Apply {
		if params != (FilmGrainParams{}) {
			return ErrInvalidFrame
		}
		return nil
	}
	if prefix.FrameType != FrameHeaderTypeInter && !params.Update {
		return ErrInvalidFrame
	}
	if !params.Update {
		found := false
		for i := uint8(0); i < parser.InterRefsPerFrame; i++ {
			if size.RefFrameIdx[i] == params.RefIdx {
				found = true
				break
			}
		}
		if !found || refs == nil || params.RefIdx >= parser.RefFrames {
			return ErrInvalidFrame
		}
		ref := refs.Frames[params.RefIdx]
		if !ref.Valid || !ref.FilmGrain.ParamsPresent {
			return ErrInvalidFrame
		}
		return nil
	}
	return validateFilmGrainUpdate(seq, params)
}

func filmGrainParamsSkipped(seq SequenceHeader, prefix FrameHeaderPrefix) bool {
	return prefix.ShowExistingFrame || !seq.FilmGrainParamsPresent || (!prefix.ShowFrame && !prefix.ShowableFrame)
}

func writeFilmGrainUpdate(w *bitWriter, seq SequenceHeader, params FilmGrainParams) error {
	if err := w.writeBits(uint64(params.NumYPoints), 4); err != nil {
		return err
	}
	if err := writeFilmGrainYPoints(w, params.NumYPoints, &params.YPoints); err != nil {
		return err
	}
	if !seq.ColorConfig.MonoChrome {
		if err := w.writeBool(params.ChromaScalingFromLuma); err != nil {
			return err
		}
	}
	if filmGrainWritesUVPoints(seq, params) {
		if err := w.writeBits(uint64(params.NumCbPoints), 4); err != nil {
			return err
		}
		if err := writeFilmGrainUVPoints(w, params.NumCbPoints, &params.CbPoints); err != nil {
			return err
		}
		if err := w.writeBits(uint64(params.NumCrPoints), 4); err != nil {
			return err
		}
		if err := writeFilmGrainUVPoints(w, params.NumCrPoints, &params.CrPoints); err != nil {
			return err
		}
	}
	if err := w.writeBits(uint64(params.ScalingShift-8), 2); err != nil {
		return err
	}
	if err := w.writeBits(uint64(params.ARCoeffLag), 2); err != nil {
		return err
	}
	numYPos := int(2 * params.ARCoeffLag * (params.ARCoeffLag + 1))
	if params.NumYPoints != 0 {
		if err := writeFilmGrainCoeffs(w, numYPos, params.ARCoeffsY[:]); err != nil {
			return err
		}
	}
	numUVPos := numYPos
	if params.NumYPoints != 0 {
		numUVPos++
	}
	if params.NumCbPoints != 0 || params.ChromaScalingFromLuma {
		if err := writeFilmGrainCoeffs(w, numUVPos, params.ARCoeffsCb[:]); err != nil {
			return err
		}
	}
	if params.NumCrPoints != 0 || params.ChromaScalingFromLuma {
		if err := writeFilmGrainCoeffs(w, numUVPos, params.ARCoeffsCr[:]); err != nil {
			return err
		}
	}
	if err := w.writeBits(uint64(params.ARCoeffShift-6), 2); err != nil {
		return err
	}
	if err := w.writeBits(uint64(params.GrainScaleShift), 2); err != nil {
		return err
	}
	if params.NumCbPoints != 0 {
		if err := writeFilmGrainChromaParams(w, params.CbMult, params.CbLumaMult, params.CbOffset); err != nil {
			return err
		}
	}
	if params.NumCrPoints != 0 {
		if err := writeFilmGrainChromaParams(w, params.CrMult, params.CrLumaMult, params.CrOffset); err != nil {
			return err
		}
	}
	if err := w.writeBool(params.Overlap); err != nil {
		return err
	}
	return w.writeBool(params.ClipToRestrictedRange)
}

func validateFilmGrainUpdate(seq SequenceHeader, params FilmGrainParams) error {
	if params.NumYPoints > MaxFilmGrainYPoints || params.NumCbPoints > MaxFilmGrainUVPoints || params.NumCrPoints > MaxFilmGrainUVPoints {
		return ErrInvalidFrame
	}
	if seq.ColorConfig.MonoChrome && (params.ChromaScalingFromLuma || params.NumCbPoints != 0 || params.NumCrPoints != 0) {
		return ErrInvalidFrame
	}
	if !filmGrainWritesUVPoints(seq, params) && (params.NumCbPoints != 0 || params.NumCrPoints != 0) {
		return ErrInvalidFrame
	}
	if seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY && ((params.NumCbPoints == 0) != (params.NumCrPoints == 0)) {
		return ErrInvalidFrame
	}
	if params.ScalingShift < 8 || params.ScalingShift > 11 || params.ARCoeffLag > 3 || params.ARCoeffShift < 6 || params.ARCoeffShift > 9 || params.GrainScaleShift > 3 {
		return ErrInvalidFrame
	}
	if params.CbOffset > 511 || params.CrOffset > 511 {
		return ErrInvalidFrame
	}
	if err := validateFilmGrainPoints(params.NumYPoints, params.YPoints[:]); err != nil {
		return err
	}
	if err := validateFilmGrainPoints(params.NumCbPoints, params.CbPoints[:]); err != nil {
		return err
	}
	return validateFilmGrainPoints(params.NumCrPoints, params.CrPoints[:])
}

func filmGrainWritesUVPoints(seq SequenceHeader, params FilmGrainParams) bool {
	return !seq.ColorConfig.MonoChrome &&
		!params.ChromaScalingFromLuma &&
		(!seq.ColorConfig.SubsamplingX || !seq.ColorConfig.SubsamplingY || params.NumYPoints != 0)
}

func validateFilmGrainPoints(count uint8, points [][2]uint8) error {
	var previous uint8
	for i := uint8(0); i < count; i++ {
		x := points[i][0]
		if i != 0 && previous >= x {
			return ErrInvalidFrame
		}
		previous = x
	}
	return nil
}

func writeFilmGrainYPoints(w *bitWriter, count uint8, points *[MaxFilmGrainYPoints][2]uint8) error {
	for i := uint8(0); i < count; i++ {
		if err := w.writeBits(uint64(points[i][0]), 8); err != nil {
			return err
		}
		if err := w.writeBits(uint64(points[i][1]), 8); err != nil {
			return err
		}
	}
	return nil
}

func writeFilmGrainUVPoints(w *bitWriter, count uint8, points *[MaxFilmGrainUVPoints][2]uint8) error {
	for i := uint8(0); i < count; i++ {
		if err := w.writeBits(uint64(points[i][0]), 8); err != nil {
			return err
		}
		if err := w.writeBits(uint64(points[i][1]), 8); err != nil {
			return err
		}
	}
	return nil
}

func writeFilmGrainCoeffs(w *bitWriter, count int, src []int8) error {
	for i := 0; i < count; i++ {
		if err := w.writeBits(uint64(uint8(int16(src[i])+128)), 8); err != nil {
			return err
		}
	}
	return nil
}

func writeFilmGrainChromaParams(w *bitWriter, mult uint8, lumaMult uint8, offset uint16) error {
	if err := w.writeBits(uint64(mult), 8); err != nil {
		return err
	}
	if err := w.writeBits(uint64(lumaMult), 8); err != nil {
		return err
	}
	return w.writeBits(uint64(offset), 9)
}
