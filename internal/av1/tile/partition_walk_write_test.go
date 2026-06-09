package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

// TestWalkBlocksWriteRoundTrip is the oracle gate for the encoder partition
// walk: random partition trees over regions with partial superblocks (frame
// edges) coded by WalkBlocksWrite must replay through the decoder's WalkBlocks
// with the identical leaf-visit sequence, partition stats, and final partition
// context, CDFs adapting in lockstep.
func TestWalkBlocksWriteRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		req  BlockWalkRequest
	}{
		// 96x80 px: partial 64x64 superblocks on both edges.
		{name: "64root-partial", req: BlockWalkRequest{Root: BlockLevel64x64, MIColEnd: 24, MIRowEnd: 20}},
		// 128x128 px exact with a 128 root: exercises the T-split alphabet.
		{name: "128root-exact", req: BlockWalkRequest{Root: BlockLevel128x128, MIColEnd: 32, MIRowEnd: 32}},
		// 64x64 px exact with a 64 root.
		{name: "64root-exact", req: BlockWalkRequest{Root: BlockLevel64x64, MIColEnd: 16, MIRowEnd: 16}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(31))

			var encCDFs PartitionCDFs
			if err := encCDFs.InitDefault(); err != nil {
				t.Fatal(err)
			}
			var encCtx PartitionContext
			w := entropy.NewWriter(make([]byte, 0, 1<<14))
			var encVisits []BlockVisit
			decide := func(level BlockLevel, ctx int, _ uint32, _ uint32, haveRight bool, haveBottom bool) (Partition, error) {
				switch {
				case haveRight && haveBottom:
					return Partition(rng.Intn(int(partitionSymbolCount[level]))), nil
				case haveRight:
					return splitOr(rng, PartitionH), nil
				default:
					return splitOr(rng, PartitionV), nil
				}
			}
			encStats, err := WalkBlocksWrite(&w, &encCDFs, &encCtx, tc.req, decide, func(v BlockVisit) error {
				encVisits = append(encVisits, v)
				return nil
			})
			if err != nil {
				t.Fatalf("WalkBlocksWrite: %v", err)
			}
			buf, err := w.Finish()
			if err != nil {
				t.Fatal(err)
			}

			var decCDFs PartitionCDFs
			if err := decCDFs.InitDefault(); err != nil {
				t.Fatal(err)
			}
			var decCtx PartitionContext
			var state DecodeState
			if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
				t.Fatal(err)
			}
			var decVisits []BlockVisit
			decStats, err := state.WalkBlocks(&decCDFs, &decCtx, tc.req, func(v BlockVisit) error {
				decVisits = append(decVisits, v)
				return nil
			})
			if err != nil {
				t.Fatalf("WalkBlocks: %v", err)
			}

			if encStats != decStats {
				t.Fatalf("stats: enc %+v dec %+v", encStats, decStats)
			}
			if len(encVisits) != len(decVisits) {
				t.Fatalf("visit count: enc %d dec %d", len(encVisits), len(decVisits))
			}
			for i := range encVisits {
				if encVisits[i] != decVisits[i] {
					t.Fatalf("visit %d: enc %+v dec %+v", i, encVisits[i], decVisits[i])
				}
			}
			if encCtx != decCtx {
				t.Fatal("partition context diverged")
			}
		})
	}
}
