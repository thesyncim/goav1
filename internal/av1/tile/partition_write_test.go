package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

// TestWritePartitionRoundTrip is the oracle gate for the partition writer: a
// random sequence of partition decisions (interior, both single-split edges, and
// the forced-split corner) coded with adapting CDFs must decode back through
// ReadPartition exactly, with the decoder CDFs adapting in lockstep.
func TestWritePartitionRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	type rec struct {
		level  BlockLevel
		ctx    int
		hH, hV bool
		part   Partition
	}
	const n = 5000
	recs := make([]rec, 0, n)

	var encCDFs PartitionCDFs
	if err := encCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<14))

	for range n {
		level := BlockLevel(rng.Intn(int(blockLevelCount)))
		ctx := rng.Intn(PartitionContexts)
		var hH, hV bool
		var part Partition
		if level == BlockLevel8x8 {
			hH, hV = true, true
			part = Partition(rng.Intn(int(partitionSymbolCount[level])))
		} else {
			switch rng.Intn(4) {
			case 0: // interior
				hH, hV = true, true
				part = Partition(rng.Intn(int(partitionSymbolCount[level])))
			case 1: // top edge: only horizontal split available
				hH, hV = true, false
				part = splitOr(rng, PartitionH)
			case 2: // left edge: only vertical split available
				hH, hV = false, true
				part = splitOr(rng, PartitionV)
			default: // corner: neither -> forced split
				hH, hV = false, false
				part = PartitionSplit
			}
		}
		if err := WritePartition(&w, &encCDFs, level, ctx, hH, hV, part); err != nil {
			t.Fatalf("WritePartition(level=%d ctx=%d hH=%v hV=%v part=%d): %v", level, ctx, hH, hV, part, err)
		}
		recs = append(recs, rec{level, ctx, hH, hV, part})
	}

	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decCDFs PartitionCDFs
	if err := decCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	for i, r := range recs {
		got, err := state.ReadPartition(&decCDFs, r.level, r.ctx, r.hH, r.hV)
		if err != nil {
			t.Fatalf("rec %d ReadPartition: %v", i, err)
		}
		if got != r.part {
			t.Fatalf("rec %d (level=%d ctx=%d hH=%v hV=%v): got %d want %d", i, r.level, r.ctx, r.hH, r.hV, got, r.part)
		}
	}
}

func splitOr(rng *rand.Rand, nonSplit Partition) Partition {
	if rng.Intn(2) == 0 {
		return PartitionSplit
	}
	return nonSplit
}
