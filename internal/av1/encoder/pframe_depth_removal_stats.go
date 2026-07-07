//go:build goav1_encstats

package encoder

import (
	"fmt"
	"os"
	"sync/atomic"
)

// depthRemovalStatsEnabled compiles depth-removal instrumentation into
// goav1_encstats builds only. Production builds use the stub file below and
// the guarded call sites compile away.
const depthRemovalStatsEnabled = true

// dev16 histogram bucket upper bounds (exclusive); the last bucket is open.
var depthRemovalDev16Buckets = [...]int64{5, 10, 25, 50, 100, 250, 500}

type depthRemovalStatsT struct {
	sbs         atomic.Uint64
	disallowSBs atomic.Uint64 // below16 fired
	costArmSBs  atomic.Uint64
	devArmSBs   atomic.Uint64
	below32SBs  atomic.Uint64
	cost32SBs   atomic.Uint64
	dev32SBs    atomic.Uint64
	below64SBs  atomic.Uint64

	dev16HistBase [len(depthRemovalDev16Buckets) + 1]atomic.Uint64
	dev16HistLeaf [len(depthRemovalDev16Buckets) + 1]atomic.Uint64
	dev32HistBase [len(depthRemovalDev16Buckets) + 1]atomic.Uint64
	dev32HistLeaf [len(depthRemovalDev16Buckets) + 1]atomic.Uint64

	// Per-16x16 removal attribution: a "removed" slot is one the variance
	// tree wanted to split below 16x16 (val16 > thresholds[3]) that some arm
	// clamped to PartitionNone.
	wouldSplit16  atomic.Uint64
	removed16Cost atomic.Uint64 // TRUE cost arm only
	removed16Dev  atomic.Uint64 // TRUE dev arm only
	removed16Prox atomic.Uint64 // landed 18b20370 proxy only
	removed16Mult atomic.Uint64 // more than one arm fired

	// Phase-2 per-depth removal attribution. pend32 counts 32x32 quadrants
	// whose partition decision would come out non-None absent the new arms;
	// removed32* attribute the quadrants the below-32 arm clamped (counted
	// only when the 64 arm did not already remove the whole SB). pend64
	// counts SBs with any split pending at all; removed64 the SBs the
	// below-64 (cost-only) arm collapsed to a single 64x64.
	pend32        atomic.Uint64
	removed32Cost atomic.Uint64
	removed32Dev  atomic.Uint64
	removed32Mult atomic.Uint64
	pend64        atomic.Uint64
	removed64     atomic.Uint64
}

var depthRemovalStats depthRemovalStatsT

func depthRemovalDev16Bucket(dev16 int64) int {
	for i, hi := range depthRemovalDev16Buckets {
		if dev16 < hi {
			return i
		}
	}
	return len(depthRemovalDev16Buckets)
}

// noteDepthRemovalSB records one swept SB's gate outcome. wouldSplit and
// proxyFired describe the 16 16x16 slots in [lvl1][lvl2] order; pending32
// marks quadrants whose 32-level decision would be non-None absent the new
// arms, pending64 whether any split is pending at all; applied32/applied64
// whether the Phase-2 clamps landed.
func noteDepthRemovalSB(dec realtimeDepthRemovalDecision, isBase bool, wouldSplit, proxyFired *[4][4]bool,
	pending32 *[4]bool, pending64, applied32, applied64 bool) {
	s := &depthRemovalStats
	s.sbs.Add(1)
	if dec.below16 {
		s.disallowSBs.Add(1)
	}
	if dec.cost16Arm {
		s.costArmSBs.Add(1)
	}
	if dec.dev16Arm {
		s.devArmSBs.Add(1)
	}
	if dec.below32 {
		s.below32SBs.Add(1)
	}
	if dec.cost32Arm {
		s.cost32SBs.Add(1)
	}
	if dec.dev32Arm {
		s.dev32SBs.Add(1)
	}
	if dec.below64 {
		s.below64SBs.Add(1)
	}
	b := depthRemovalDev16Bucket(dec.dev16)
	b32 := depthRemovalDev16Bucket(dec.dev32To16)
	if isBase {
		s.dev16HistBase[b].Add(1)
		s.dev32HistBase[b32].Add(1)
	} else {
		s.dev16HistLeaf[b].Add(1)
		s.dev32HistLeaf[b32].Add(1)
	}
	if pending64 {
		s.pend64.Add(1)
		if applied64 {
			s.removed64.Add(1)
		}
	}
	for i := range pending32 {
		if !pending32[i] {
			continue
		}
		s.pend32.Add(1)
		if applied32 && !applied64 {
			switch {
			case dec.cost32Arm && dec.dev32Arm:
				s.removed32Mult.Add(1)
			case dec.cost32Arm:
				s.removed32Cost.Add(1)
			case dec.dev32Arm:
				s.removed32Dev.Add(1)
			}
		}
	}
	for i := range wouldSplit {
		for j := range wouldSplit[i] {
			if !wouldSplit[i][j] {
				continue
			}
			s.wouldSplit16.Add(1)
			arms := 0
			if dec.cost16Arm {
				arms++
			}
			if dec.dev16Arm {
				arms++
			}
			if proxyFired[i][j] {
				arms++
			}
			switch {
			case arms > 1:
				s.removed16Mult.Add(1)
			case dec.cost16Arm:
				s.removed16Cost.Add(1)
			case dec.dev16Arm:
				s.removed16Dev.Add(1)
			case proxyFired[i][j]:
				s.removed16Prox.Add(1)
			}
		}
	}
}

func histSnapshotReset(h *[len(depthRemovalDev16Buckets) + 1]atomic.Uint64) [len(depthRemovalDev16Buckets) + 1]uint64 {
	var out [len(depthRemovalDev16Buckets) + 1]uint64
	for i := range h {
		out[i] = h[i].Swap(0)
	}
	return out
}

// dumpDepthRemovalStats prints and resets the per-frame counters. Called from
// the frame-serial tail of encodePReusing, after every lane has joined.
func dumpDepthRemovalStats(frame uint64) {
	s := &depthRemovalStats
	sbs := s.sbs.Swap(0)
	if sbs == 0 {
		return
	}
	base := histSnapshotReset(&s.dev16HistBase)
	leaf := histSnapshotReset(&s.dev16HistLeaf)
	base32 := histSnapshotReset(&s.dev32HistBase)
	leaf32 := histSnapshotReset(&s.dev32HistLeaf)
	fmt.Fprintf(os.Stderr,
		"depthrm frame=%d sbs=%d below16=%d cost16=%d dev16arm=%d below32=%d cost32=%d dev32arm=%d below64=%d "+
			"dev16base=%v dev16leaf=%v dev32base=%v dev32leaf=%v "+
			"wouldsplit16=%d removed16(cost=%d dev=%d proxy=%d multi=%d) "+
			"pend32=%d removed32(cost=%d dev=%d multi=%d) pend64=%d removed64=%d\n",
		frame, sbs, s.disallowSBs.Swap(0), s.costArmSBs.Swap(0), s.devArmSBs.Swap(0),
		s.below32SBs.Swap(0), s.cost32SBs.Swap(0), s.dev32SBs.Swap(0), s.below64SBs.Swap(0),
		base, leaf, base32, leaf32,
		s.wouldSplit16.Swap(0), s.removed16Cost.Swap(0), s.removed16Dev.Swap(0),
		s.removed16Prox.Swap(0), s.removed16Mult.Swap(0),
		s.pend32.Swap(0), s.removed32Cost.Swap(0), s.removed32Dev.Swap(0), s.removed32Mult.Swap(0),
		s.pend64.Swap(0), s.removed64.Swap(0))
}
