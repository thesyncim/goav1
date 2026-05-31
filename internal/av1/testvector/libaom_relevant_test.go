//go:build goav1_oracle

package testvector

import (
	"os"
	"testing"
)

// TestLibaomRelevantSupportedFrameWorkDryRun exercises the SuiteLevelRelevant
// vectors whose active postfilters are reference-safe in the current publishable
// output path. Film-grain vectors stay out of this gate until display-grain
// synthesis is split from reference publication.
func TestLibaomRelevantSupportedFrameWorkDryRun(t *testing.T) {
	if os.Getenv("GOAV1_RELEVANT_SUPPORTED_LIBAOM_FRAMEWORK_DRYRUN") != "1" {
		t.Skip("set GOAV1_RELEVANT_SUPPORTED_LIBAOM_FRAMEWORK_DRYRUN=1 to run the opt-in relevant-vector framework dry-run")
	}
	manifest := LibaomRemoteManifest()
	selected := manifest.SelectRemote(SuiteLevelRelevant, 0, nil)
	if len(selected) == 0 {
		t.Fatal("relevant cohort selected no vectors")
	}
	ran := 0
	skippedFilmGrain := 0
	for _, vector := range selected {
		if vector.Labels&VectorLabelFilmGrain != 0 {
			skippedFilmGrain++
			continue
		}
		ran++
		t.Run(vector.Name, func(t *testing.T) {
			runLibaomFrameWorkDryRun(t, vector)
		})
	}
	t.Logf("relevant_supported_vectors=%d skipped_film_grain=%d", ran, skippedFilmGrain)
	if ran == 0 {
		t.Fatal("relevant supported cohort ran no vectors")
	}
}
