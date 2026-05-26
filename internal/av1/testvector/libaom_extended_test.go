//go:build goav1_oracle

package testvector

import (
	"os"
	"testing"
)

// TestLibaomExtendedFrameWorkDryRun exercises the SuiteLevelExtended cohort
// through the framework dry-run pipeline. The cohort is strictly opt-in: this
// test only runs when GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1. It is never
// part of the default fast gate or the testvectors workflow, and is intended
// to surface latent decoder gaps in:
//
//   - 10-bit streams across the full quantizer range (q=32, q=63);
//   - sub-superblock-aligned (34x34, 64x64, 66x66) and multi-superblock
//     (208x208, 226x226) frame sizes;
//   - additional SVC permutations (L2T1, L2T2) which also exercise
//     the multi-tile path (Cols=3 base, Cols=4 enhancement) - libaom
//     v3.14.0 ships no dedicated av1-1-b8-XX-tiles.ivf, so the SVC
//     vectors are the only committed multi-tile bitstreams. See
//     VectorLabelMultiTile in remote.go and the SVC L1T2 entry in
//     libaom_manifest.go for the non-SVC tile-path probe at
//     SuiteLevelRelevant.
//
// Each vector executes runLibaomFrameWorkDryRun, which gates strictly on
// frame 0's MD5 by default and logs a per-frame progress line for later
// frames. Set GOAV1_STRICT_MD5=1 to upgrade the check to every frame.
func TestLibaomExtendedFrameWorkDryRun(t *testing.T) {
	if os.Getenv("GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN") != "1" {
		t.Skip("set GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1 to run the opt-in extended-vector framework dry-run")
	}
	manifest := LibaomRemoteManifest()
	selected := manifest.SelectRemote(SuiteLevelExtended, 0, nil)
	if len(selected) == 0 {
		t.Fatal("extended cohort selected no vectors")
	}
	for _, vector := range selected {
		t.Run(vector.Name, func(t *testing.T) {
			runLibaomFrameWorkDryRun(t, vector)
		})
	}
}
