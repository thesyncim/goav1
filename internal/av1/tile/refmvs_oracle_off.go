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

// TemporalProjectionOracle reports whether the goav1_refmvs_oracle build is
// active, in which case frame decoders still set up the whole-frame temporal
// field so the per-row load_tmvs projection can be checked against it.
const TemporalProjectionOracle = false
