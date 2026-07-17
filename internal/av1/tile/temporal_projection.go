package tile

import "github.com/thesyncim/goav1/internal/av1/motion"

// motionFieldProjectMVInRange projects a vector after the caller has proved
// num is in [-motionFieldMaxFrameDistance, motionFieldMaxFrameDistance] and
// den is in [1, motionFieldMaxFrameDistance]. It retains the general helper's
// result shape so the caller's hot loop has the same compact call sequence.
func motionFieldProjectMVInRange(ref motion.Vector, num int, den int) (motion.Vector, error) {
	row := roundPowerOfTwoSigned(int64(ref.Row)*int64(num)*int64(motionFieldDivMult[den]), 14)
	col := roundPowerOfTwoSigned(int64(ref.Col)*int64(num)*int64(motionFieldDivMult[den]), 14)
	return motion.Vector{
		Row: int16(clampInt64(row, motionFieldMVLower+1, motionFieldMVUpper-1)),
		Col: int16(clampInt64(col, motionFieldMVLower+1, motionFieldMVUpper-1)),
	}, nil
}
