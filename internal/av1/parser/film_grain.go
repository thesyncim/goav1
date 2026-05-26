package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const (
	MaxFilmGrainYPoints  = 14
	MaxFilmGrainUVPoints = 10
	MaxFilmGrainYCoeffs  = 24
	MaxFilmGrainUVCoeffs = 25
)

// FilmGrainParams is the film_grain_params() state carried by a frame header.
// Array sizes mirror the AV1 limits so parsers and later synthesis code can
// pass parameters without heap-backed slices.
type FilmGrainParams struct {
	ParamsPresent bool
	Apply         bool
	Update        bool
	Seed          uint16
	RefIdx        uint8

	BitDepth uint8

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
	BitsRead              int
}

// ParseFilmGrainParams parses film_grain_params() after global_motion_params().
func ParseFilmGrainParams(payload []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size FrameSize, refs *ReferenceState, globalMotion GlobalMotionParams) (FilmGrainParams, error) {
	r := bitstream.NewReader(payload)
	if err := r.SkipBits(globalMotion.BitsRead); err != nil {
		return FilmGrainParams{}, err
	}

	params := FilmGrainParams{
		ParamsPresent: seq.FilmGrainParamsPresent,
		BitDepth:      seq.ColorConfig.BitDepth,
		BitsRead:      r.BitsRead(),
	}
	if prefix.ShowExistingFrame || !seq.FilmGrainParamsPresent || (!prefix.ShowFrame && !prefix.ShowableFrame) {
		return params, nil
	}

	apply, err := r.ReadBool()
	if err != nil {
		return FilmGrainParams{}, err
	}
	params.Apply = apply
	params.BitsRead = r.BitsRead()
	if !params.Apply {
		return params, nil
	}

	seed, err := r.ReadBits(16)
	if err != nil {
		return FilmGrainParams{}, err
	}
	params.Seed = uint16(seed)
	params.Update = true
	if prefix.FrameType == FrameTypeInter {
		params.Update, err = r.ReadBool()
		if err != nil {
			return FilmGrainParams{}, err
		}
	}
	if !params.Update {
		return parseFilmGrainReference(&r, seq, size, refs, params)
	}
	if err := parseFilmGrainUpdate(&r, seq, &params); err != nil {
		return FilmGrainParams{}, err
	}
	params.BitsRead = r.BitsRead()
	return params, nil
}

func parseFilmGrainReference(r *bitstream.Reader, seq SequenceHeader, size FrameSize, refs *ReferenceState, params FilmGrainParams) (FilmGrainParams, error) {
	v, err := r.ReadBits(3)
	if err != nil {
		return FilmGrainParams{}, err
	}
	refIdx := uint8(v)
	found := false
	for i := range InterRefsPerFrame {
		if size.RefFrameIdx[i] == refIdx {
			found = true
			break
		}
	}
	if !found {
		return FilmGrainParams{}, ErrInvalidFrameHeader
	}
	if refs == nil {
		return FilmGrainParams{}, ErrReferenceFrameNeeded
	}
	ref := refs.Frames[refIdx]
	if !ref.Valid || !ref.FilmGrain.ParamsPresent {
		return FilmGrainParams{}, ErrReferenceFrameNeeded
	}

	seed := params.Seed
	params = ref.FilmGrain
	params.Apply = true
	params.Update = false
	params.Seed = seed
	params.RefIdx = refIdx
	params.ParamsPresent = seq.FilmGrainParamsPresent
	params.BitDepth = seq.ColorConfig.BitDepth
	params.BitsRead = r.BitsRead()
	return params, nil
}

func parseFilmGrainUpdate(r *bitstream.Reader, seq SequenceHeader, params *FilmGrainParams) error {
	v, err := r.ReadBits(4)
	if err != nil {
		return err
	}
	params.NumYPoints = uint8(v)
	if params.NumYPoints > MaxFilmGrainYPoints {
		return ErrInvalidFrameHeader
	}
	if err := readFilmGrainYPoints(r, params.NumYPoints, &params.YPoints); err != nil {
		return err
	}

	if !seq.ColorConfig.MonoChrome {
		params.ChromaScalingFromLuma, err = r.ReadBool()
		if err != nil {
			return err
		}
	}
	if !seq.ColorConfig.MonoChrome &&
		!params.ChromaScalingFromLuma &&
		(!seq.ColorConfig.SubsamplingX || !seq.ColorConfig.SubsamplingY || params.NumYPoints != 0) {
		v, err = r.ReadBits(4)
		if err != nil {
			return err
		}
		params.NumCbPoints = uint8(v)
		if params.NumCbPoints > MaxFilmGrainUVPoints {
			return ErrInvalidFrameHeader
		}
		if err := readFilmGrainUVPoints(r, params.NumCbPoints, &params.CbPoints); err != nil {
			return err
		}

		v, err = r.ReadBits(4)
		if err != nil {
			return err
		}
		params.NumCrPoints = uint8(v)
		if params.NumCrPoints > MaxFilmGrainUVPoints {
			return ErrInvalidFrameHeader
		}
		if err := readFilmGrainUVPoints(r, params.NumCrPoints, &params.CrPoints); err != nil {
			return err
		}
	}
	if seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY &&
		((params.NumCbPoints == 0) != (params.NumCrPoints == 0)) {
		return ErrInvalidFrameHeader
	}

	v, err = r.ReadBits(2)
	if err != nil {
		return err
	}
	params.ScalingShift = uint8(v + 8)
	v, err = r.ReadBits(2)
	if err != nil {
		return err
	}
	params.ARCoeffLag = uint8(v)

	numYPos := int(2 * params.ARCoeffLag * (params.ARCoeffLag + 1))
	if params.NumYPoints != 0 {
		if err := readFilmGrainCoeffs(r, numYPos, params.ARCoeffsY[:]); err != nil {
			return err
		}
	}
	numUVPos := numYPos
	if params.NumYPoints != 0 {
		numUVPos++
	}
	if params.NumCbPoints != 0 || params.ChromaScalingFromLuma {
		if err := readFilmGrainCoeffs(r, numUVPos, params.ARCoeffsCb[:]); err != nil {
			return err
		}
	}
	if params.NumCrPoints != 0 || params.ChromaScalingFromLuma {
		if err := readFilmGrainCoeffs(r, numUVPos, params.ARCoeffsCr[:]); err != nil {
			return err
		}
	}

	v, err = r.ReadBits(2)
	if err != nil {
		return err
	}
	params.ARCoeffShift = uint8(v + 6)
	v, err = r.ReadBits(2)
	if err != nil {
		return err
	}
	params.GrainScaleShift = uint8(v)

	if params.NumCbPoints != 0 {
		if err := readFilmGrainChromaParams(r, &params.CbMult, &params.CbLumaMult, &params.CbOffset); err != nil {
			return err
		}
	}
	if params.NumCrPoints != 0 {
		if err := readFilmGrainChromaParams(r, &params.CrMult, &params.CrLumaMult, &params.CrOffset); err != nil {
			return err
		}
	}
	if params.Overlap, err = r.ReadBool(); err != nil {
		return err
	}
	params.ClipToRestrictedRange, err = r.ReadBool()
	return err
}

func readFilmGrainYPoints(r *bitstream.Reader, count uint8, points *[MaxFilmGrainYPoints][2]uint8) error {
	var previous uint8
	for i := range count {
		x, err := r.ReadBits(8)
		if err != nil {
			return err
		}
		if i != 0 && previous >= uint8(x) {
			return ErrInvalidFrameHeader
		}
		y, err := r.ReadBits(8)
		if err != nil {
			return err
		}
		points[i][0] = uint8(x)
		points[i][1] = uint8(y)
		previous = uint8(x)
	}
	return nil
}

func readFilmGrainUVPoints(r *bitstream.Reader, count uint8, points *[MaxFilmGrainUVPoints][2]uint8) error {
	var previous uint8
	for i := range count {
		x, err := r.ReadBits(8)
		if err != nil {
			return err
		}
		if i != 0 && previous >= uint8(x) {
			return ErrInvalidFrameHeader
		}
		y, err := r.ReadBits(8)
		if err != nil {
			return err
		}
		points[i][0] = uint8(x)
		points[i][1] = uint8(y)
		previous = uint8(x)
	}
	return nil
}

func readFilmGrainCoeffs(r *bitstream.Reader, count int, dst []int8) error {
	for i := range count {
		v, err := r.ReadBits(8)
		if err != nil {
			return err
		}
		dst[i] = int8(int16(v) - 128)
	}
	return nil
}

func readFilmGrainChromaParams(r *bitstream.Reader, mult *uint8, lumaMult *uint8, offset *uint16) error {
	v, err := r.ReadBits(8)
	if err != nil {
		return err
	}
	*mult = uint8(v)
	v, err = r.ReadBits(8)
	if err != nil {
		return err
	}
	*lumaMult = uint8(v)
	v, err = r.ReadBits(9)
	if err != nil {
		return err
	}
	*offset = uint16(v)
	return nil
}
