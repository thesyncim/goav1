//go:build goav1_encstats

package encoder

import (
	"fmt"
	"os"
	"sync/atomic"
)

// lpd1StatsEnabled compiles the per-SB light-PD1 level histogram into
// goav1_encstats builds only.
const lpd1StatsEnabled = true

// lpd1LevelCounts[0] counts REGULAR_PD1 SBs, [1+l] LPD1_LVL_l.
var lpd1LevelCounts [1 + lpd1Levels]atomic.Uint64

func noteLPD1SBLevel(level lpd1PD1Level) {
	lpd1LevelCounts[int(level)+1].Add(1)
}

// dumpLPD1Stats prints the cumulative histogram (frame-serial callers).
func dumpLPD1Stats(frame uint64) {
	fmt.Fprintf(os.Stderr, "lpd1 frame=%d regular=%d", frame, lpd1LevelCounts[0].Load())
	for l := 0; l < lpd1Levels; l++ {
		fmt.Fprintf(os.Stderr, " lvl%d=%d", l, lpd1LevelCounts[l+1].Load())
	}
	fmt.Fprintln(os.Stderr)
}
