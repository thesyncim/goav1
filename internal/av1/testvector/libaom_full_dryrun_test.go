//go:build goav1_oracle

package testvector

import (
	"os"
	"testing"
)

// TestLibaomFullFrameWorkDryRun exercises every checksum-pinned vector in the
// committed libaom manifest through the framework dry-run pipeline.
func TestLibaomFullFrameWorkDryRun(t *testing.T) {
	if os.Getenv("GOAV1_FULL_LIBAOM_FRAMEWORK_DRYRUN") != "1" {
		t.Skip("set GOAV1_FULL_LIBAOM_FRAMEWORK_DRYRUN=1 to run the opt-in full-vector framework dry-run")
	}
	manifest := LibaomRemoteManifest()
	selected := manifest.SelectRemote(SuiteLevelFull, 0, nil)
	if len(selected) == 0 {
		t.Fatal("full cohort selected no vectors")
	}
	ran := 0
	for _, vector := range selected {
		ran++
		t.Run(vector.Name, func(t *testing.T) {
			runLibaomFrameWorkDryRun(t, vector)
		})
	}
	t.Logf("full_vectors=%d", ran)
	if ran == 0 {
		t.Fatal("full cohort ran no vectors")
	}
}
