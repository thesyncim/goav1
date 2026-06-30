package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

const (
	MVJoints       = 4
	MVClasses      = 11
	MVClass0Bits   = 1
	MVClass0Size   = 1 << MVClass0Bits
	MVOffsetBits   = MVClasses + MVClass0Bits - 2
	MVFPSize       = 4
	MVInUseBits    = 14
	MVUpper        = 1 << MVInUseBits
	MVLower        = -MVUpper
	MVComponents   = 2
	mvComponentRow = 0
	mvComponentCol = 1
)

// MVJoint is libaom's MV_JOINT_TYPE/dav1d's MVJoint.
type MVJoint uint8

const (
	MVJointZero MVJoint = iota
	MVJointHorizontal
	MVJointVertical
	MVJointBoth
)

// MVSubpelPrecision mirrors libaom's MvSubpelPrecision values.
type MVSubpelPrecision int8

const (
	MVSubpelNone MVSubpelPrecision = -1
	MVSubpelLow  MVSubpelPrecision = 0
	MVSubpelHigh MVSubpelPrecision = 1
)

// MVComponentCDFs contains one component of libaom's nmv_component.
type MVComponentCDFs struct {
	Classes  entropy.CDF
	Class0FP [MVClass0Size]entropy.CDF
	FP       entropy.CDF
	Sign     entropy.CDF
	Class0HP entropy.CDF
	HP       entropy.CDF
	Class0   entropy.CDF
	Bits     [MVOffsetBits]entropy.CDF
}

// MVCDFs contains libaom's nmv_context / dav1d's CdfMvContext.
type MVCDFs struct {
	Joint      entropy.CDF
	Components [MVComponents]MVComponentCDFs
}

type MVComponentResult struct {
	Diff          int16
	IntegerPart   uint16
	Class         uint8
	Fraction      uint8
	HighPrecision uint8
	Sign          bool
	Class0        bool
}

type MVResidualResult struct {
	Joint          MVJoint
	Precision      MVSubpelPrecision
	Diff           motion.Vector
	Components     [MVComponents]MVComponentResult
	ComponentValid [MVComponents]bool
}

type InterMotionRequest struct {
	References InterReferencesResult
	Mode       InterModeResult

	ReferenceMVs InterMVReferenceSet
	GlobalMVs    [2]motion.Vector

	Precision MVSubpelPrecision
}

type InterMotionDecodeResult struct {
	Motion InterMotionResult

	Residuals     [2]MVResidualResult
	ResidualValid [2]bool
}

func MVPrecision(allowHighPrecision bool, forceInteger bool) MVSubpelPrecision {
	if forceInteger {
		return MVSubpelNone
	}
	if allowHighPrecision {
		return MVSubpelHigh
	}
	return MVSubpelLow
}

func (joint MVJoint) Valid() bool {
	return joint < MVJoint(MVJoints)
}

func (joint MVJoint) HasVertical() bool {
	return joint == MVJointVertical || joint == MVJointBoth
}

func (joint MVJoint) HasHorizontal() bool {
	return joint == MVJointHorizontal || joint == MVJointBoth
}

func (precision MVSubpelPrecision) Valid() bool {
	return precision >= MVSubpelNone && precision <= MVSubpelHigh
}

func (precision MVSubpelPrecision) UsesSubpel() bool {
	return precision > MVSubpelNone
}

func (precision MVSubpelPrecision) UsesHighPrecision() bool {
	return precision > MVSubpelLow
}

// InitDefault seeds c with libaom/dav1d's default nmv_context.
func (c *MVCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next MVCDFs
	if err := next.Joint.Init([]uint16{4096, 11264, 19328}); err != nil {
		return err
	}
	for i := range next.Components {
		if err := next.Components[i].InitDefault(); err != nil {
			return err
		}
	}
	*c = next
	return nil
}

// InitDefault seeds c with libaom/dav1d's default nmv_component.
func (c *MVComponentCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next MVComponentCDFs
	if err := next.Classes.Init([]uint16{28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767}); err != nil {
		return err
	}
	if err := next.Class0FP[0].Init([]uint16{16384, 24576, 26624}); err != nil {
		return err
	}
	if err := next.Class0FP[1].Init([]uint16{12288, 21248, 24128}); err != nil {
		return err
	}
	if err := next.FP.Init([]uint16{8192, 17408, 21248}); err != nil {
		return err
	}
	if err := next.Sign.Init([]uint16{128 * 128}); err != nil {
		return err
	}
	if err := next.Class0HP.Init([]uint16{160 * 128}); err != nil {
		return err
	}
	if err := next.HP.Init([]uint16{128 * 128}); err != nil {
		return err
	}
	if err := next.Class0.Init([]uint16{216 * 128}); err != nil {
		return err
	}
	for i, cumulative := range [...]uint16{
		128 * 136,
		128 * 140,
		128 * 148,
		128 * 160,
		128 * 176,
		128 * 192,
		128 * 224,
		128 * 234,
		128 * 234,
		128 * 240,
	} {
		if err := next.Bits[i].Init([]uint16{cumulative}); err != nil {
			return err
		}
	}
	*c = next
	return nil
}

func (c *MVCDFs) JointCDF() (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.Joint, MVJoints)
}

func (c *MVComponentCDFs) ClassesCDF() (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.Classes, MVClasses)
}

func (c *MVComponentCDFs) Class0FPCDF(index int) (*entropy.CDF, error) {
	if c == nil || index < 0 || index >= MVClass0Size {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.Class0FP[index], MVFPSize)
}

func (c *MVComponentCDFs) FPCDF() (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.FP, MVFPSize)
}

func (c *MVComponentCDFs) SignCDF() (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.Sign, 2)
}

func (c *MVComponentCDFs) Class0HPCDF() (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.Class0HP, 2)
}

func (c *MVComponentCDFs) HPCDF() (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.HP, 2)
}

func (c *MVComponentCDFs) Class0CDF() (*entropy.CDF, error) {
	if c == nil {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.Class0, MVClass0Size)
}

func (c *MVComponentCDFs) BitCDF(index int) (*entropy.CDF, error) {
	if c == nil || index < 0 || index >= MVOffsetBits {
		return nil, entropy.ErrInvalidCDF
	}
	return mvCDF(&c.Bits[index], 2)
}

func (s *DecodeState) ReadMotionVector(cdfs *MVCDFs, ref motion.Vector, precision MVSubpelPrecision) (motion.Vector, MVResidualResult, error) {
	if s == nil || !precision.Valid() {
		return motion.Vector{}, MVResidualResult{}, ErrInvalidDecodeState
	}
	if cdfs == nil || cdfs.Joint.Symbols() != MVJoints {
		return motion.Vector{}, MVResidualResult{}, entropy.ErrInvalidCDF
	}
	reader := s.Reader.Cursor()
	symbol := reader.ReadCDF4Unchecked(&cdfs.Joint)
	joint := MVJoint(symbol)
	if !joint.Valid() {
		reader.CommitStateTo(&s.Reader)
		return motion.Vector{}, MVResidualResult{}, ErrInvalidDecodeState
	}
	result := MVResidualResult{Joint: joint, Precision: precision}
	if joint.HasVertical() {
		diff, component, err := readMVComponentDiffCursor(&reader, &cdfs.Components[mvComponentRow], precision)
		if err != nil {
			reader.CommitStateTo(&s.Reader)
			return motion.Vector{}, MVResidualResult{}, err
		}
		componentDiff, ok := motionVectorComponentFromInt32(diff)
		if !ok {
			reader.CommitStateTo(&s.Reader)
			return motion.Vector{}, MVResidualResult{}, ErrInvalidDecodeState
		}
		result.Diff.Row = componentDiff
		result.Components[mvComponentRow] = component
		result.ComponentValid[mvComponentRow] = true
	}
	if joint.HasHorizontal() {
		diff, component, err := readMVComponentDiffCursor(&reader, &cdfs.Components[mvComponentCol], precision)
		if err != nil {
			reader.CommitStateTo(&s.Reader)
			return motion.Vector{}, MVResidualResult{}, err
		}
		componentDiff, ok := motionVectorComponentFromInt32(diff)
		if !ok {
			reader.CommitStateTo(&s.Reader)
			return motion.Vector{}, MVResidualResult{}, ErrInvalidDecodeState
		}
		result.Diff.Col = componentDiff
		result.Components[mvComponentCol] = component
		result.ComponentValid[mvComponentCol] = true
	}
	mv, ok := motionVectorFromInt32(
		int32(ref.Row)+int32(result.Diff.Row),
		int32(ref.Col)+int32(result.Diff.Col),
	)
	if !ok {
		reader.CommitStateTo(&s.Reader)
		return motion.Vector{}, MVResidualResult{}, ErrInvalidDecodeState
	}
	reader.CommitStateTo(&s.Reader)
	return mv, result, nil
}

func (s *DecodeState) ReadMVComponentDiff(cdfs *MVComponentCDFs, precision MVSubpelPrecision) (int32, MVComponentResult, error) {
	if s == nil || !precision.Valid() {
		return 0, MVComponentResult{}, ErrInvalidDecodeState
	}
	reader := s.Reader.Cursor()
	diff, result, err := readMVComponentDiffCursor(&reader, cdfs, precision)
	reader.CommitStateTo(&s.Reader)
	return diff, result, err
}

// readMVComponentDiffCursor ports dav1d's read_mv_component_diff(): keep the
// arithmetic decoder state hot while reading sign, class, integer, fractional,
// and high-precision component syntax.
func readMVComponentDiffCursor(reader *entropy.Cursor, cdfs *MVComponentCDFs, precision MVSubpelPrecision) (int32, MVComponentResult, error) {
	if reader == nil || cdfs == nil || !precision.Valid() ||
		cdfs.Sign.Symbols() != 2 ||
		cdfs.Classes.Symbols() != MVClasses {
		return 0, MVComponentResult{}, entropy.ErrInvalidCDF
	}
	signSymbol := reader.ReadBinaryCDFUnchecked(&cdfs.Sign)
	mvClass := reader.ReadCDFSymbolsUnchecked(&cdfs.Classes, MVClasses)

	result := MVComponentResult{
		Sign:          signSymbol != 0,
		Class:         uint8(mvClass),
		Class0:        mvClass == 0,
		Fraction:      3,
		HighPrecision: 1,
	}
	mag := 0
	integerPart := 0
	if result.Class0 {
		if cdfs.Class0.Symbols() != MVClass0Size {
			return 0, MVComponentResult{}, entropy.ErrInvalidCDF
		}
		integerPart = reader.ReadBinaryCDFUnchecked(&cdfs.Class0)
	} else {
		n := mvClass + MVClass0Bits - 1
		for i := range n {
			if cdfs.Bits[i].Symbols() != 2 {
				return 0, MVComponentResult{}, entropy.ErrInvalidCDF
			}
			bit := reader.ReadBinaryCDFUnchecked(&cdfs.Bits[i])
			integerPart |= bit << i
		}
		mag = MVClass0Size << (mvClass + 2)
	}

	if precision.UsesSubpel() {
		if result.Class0 {
			if integerPart < 0 || integerPart >= MVClass0Size ||
				cdfs.Class0FP[integerPart].Symbols() != MVFPSize {
				return 0, MVComponentResult{}, entropy.ErrInvalidCDF
			}
			result.Fraction = uint8(reader.ReadCDF4Unchecked(&cdfs.Class0FP[integerPart]))
		} else {
			if cdfs.FP.Symbols() != MVFPSize {
				return 0, MVComponentResult{}, entropy.ErrInvalidCDF
			}
			result.Fraction = uint8(reader.ReadCDF4Unchecked(&cdfs.FP))
		}
		if precision.UsesHighPrecision() {
			if result.Class0 {
				if cdfs.Class0HP.Symbols() != 2 {
					return 0, MVComponentResult{}, entropy.ErrInvalidCDF
				}
				result.HighPrecision = uint8(reader.ReadBinaryCDFUnchecked(&cdfs.Class0HP))
			} else {
				if cdfs.HP.Symbols() != 2 {
					return 0, MVComponentResult{}, entropy.ErrInvalidCDF
				}
				result.HighPrecision = uint8(reader.ReadBinaryCDFUnchecked(&cdfs.HP))
			}
		}
	}

	mag += ((integerPart << 3) | (int(result.Fraction) << 1) | int(result.HighPrecision)) + 1
	diff := int32(mag)
	if result.Sign {
		diff = -diff
	}
	componentDiff, ok := motionVectorComponentFromInt32(diff)
	if !ok {
		return 0, MVComponentResult{}, ErrInvalidDecodeState
	}
	result.IntegerPart = uint16(integerPart)
	result.Diff = componentDiff
	return diff, result, nil
}

func (s *DecodeState) ReadInterMotion(cdfs *MVCDFs, req InterMotionRequest) (InterMotionDecodeResult, error) {
	if s == nil || !req.Precision.Valid() || req.Mode.Compound != req.References.Compound {
		return InterMotionDecodeResult{}, ErrInvalidDecodeState
	}
	if !req.References.Ref[0].Valid() {
		return InterMotionDecodeResult{}, ErrInvalidDecodeState
	}
	if req.References.Compound {
		if !req.References.Ref[1].Valid() || !req.Mode.CompoundMode.Valid() {
			return InterMotionDecodeResult{}, ErrInvalidDecodeState
		}
	} else if req.References.Ref[1] != ReferenceFrameNone || !req.Mode.Mode.Valid() {
		return InterMotionDecodeResult{}, ErrInvalidDecodeState
	}

	result := InterMotionDecodeResult{
		Motion: InterMotionResult{
			References: req.References,
			Mode:       req.Mode,
		},
	}
	if req.Mode.Compound {
		components, err := req.Mode.CompoundMode.Components()
		if err != nil {
			return InterMotionDecodeResult{}, err
		}
		if err := s.assignInterMVComponent(cdfs, req, components.First, 0, &result); err != nil {
			return InterMotionDecodeResult{}, err
		}
		if err := s.assignInterMVComponent(cdfs, req, components.Second, 1, &result); err != nil {
			return InterMotionDecodeResult{}, err
		}
	} else if err := s.assignInterMVComponent(cdfs, req, req.Mode.Mode, 0, &result); err != nil {
		return InterMotionDecodeResult{}, err
	}

	if !motionVectorValid(result.Motion.MV[0]) {
		return InterMotionDecodeResult{}, ErrInvalidDecodeState
	}
	if req.References.Compound && !motionVectorValid(result.Motion.MV[1]) {
		return InterMotionDecodeResult{}, ErrInvalidDecodeState
	}
	return result, nil
}

func (s *DecodeState) assignInterMVComponent(cdfs *MVCDFs, req InterMotionRequest, mode InterMode, index int, result *InterMotionDecodeResult) error {
	if index < 0 || index >= len(result.Motion.MV) || !mode.Valid() {
		return ErrInvalidDecodeState
	}
	switch mode {
	case InterModeNearestMV:
		result.Motion.MV[index] = req.ReferenceMVs.Nearest[index]
	case InterModeNearMV:
		result.Motion.MV[index] = req.ReferenceMVs.Near[index]
	case InterModeGlobalMV:
		result.Motion.MV[index] = req.GlobalMVs[index]
	case InterModeNewMV:
		mv, residual, err := s.ReadMotionVector(cdfs, req.ReferenceMVs.Residual[index], req.Precision)
		if err != nil {
			return err
		}
		result.Motion.MV[index] = mv
		result.Residuals[index] = residual
		result.ResidualValid[index] = true
	default:
		return ErrInvalidDecodeState
	}
	return nil
}

func mvCDF(cdf *entropy.CDF, symbols int) (*entropy.CDF, error) {
	if cdf == nil || cdf.Symbols() != symbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, nil
}

func motionVectorValid(v motion.Vector) bool {
	return int32(v.Row) > MVLower && int32(v.Row) < MVUpper &&
		int32(v.Col) > MVLower && int32(v.Col) < MVUpper
}

func motionVectorFromInt32(row int32, col int32) (motion.Vector, bool) {
	if row <= MVLower || row >= MVUpper || col <= MVLower || col >= MVUpper {
		return motion.Vector{}, false
	}
	return motion.Vector{Row: int16(row), Col: int16(col)}, true
}

func motionVectorComponentFromInt32(v int32) (int16, bool) {
	const storageUpper = 1 << 15
	if v < -storageUpper || v >= storageUpper {
		return 0, false
	}
	return int16(v), true
}
