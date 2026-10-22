//go:build !goav1_refmvs_oracle

package tile

import "github.com/thesyncim/goav1/internal/av1/motion"

// refmvsPoisonEnabled and refmvsOracleEnabled are compiled out of production
// builds; the goav1_refmvs_oracle tag turns on poisoned row storage and the
// legacy-vs-dav1d candidate-stack differential.
const (
	refmvsPoisonEnabled = false
	refmvsOracleEnabled = false
)

func refmvsPoison([]refmvsBlock) {}

func refmvsOracleCompare(*BlockModeContext, *refmvsTile, *refmvsResult, *BlockLoopRequest, BlockVisit, InterReferencesResult, [2]motion.Vector) {
}

func refmvsOracleTopHasRight(uint8, BlockVisit, bool) {}

func refmvsOracleTmvs(*refmvsTile, int32, int32, int32, int32) {}

func refmvsOracleWholeField(*BlockLoopRequest) *TemporalMotionField { return nil }

func refmvsOracleEncoderStack(*BlockModeContext, *ReferenceMVStackRequest, InterReferencesResult, *ReferenceMVStackResult) {
}
