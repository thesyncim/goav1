package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// PartitionDecider chooses the partition for one interior tree node during an
// encoding walk. It receives the same level/context/availability the decoder's
// ReadPartition sees; the returned partition must be valid for that node (forced
// PARTITION_SPLIT nodes are not consulted — the walker handles them).
type PartitionDecider func(level BlockLevel, ctx int, haveRight bool, haveBottom bool) (Partition, error)

// WalkBlocksWrite runs the encoder-side partition walk: it traverses the MI
// region with the decoder's own walkBlocks state machine, asks decide for each
// coded node, writes the partition symbol through WritePartition, and calls
// visit for every leaf block in bitstream order. Because traversal, context
// marking, and availability derivation are the decoder's exact code, a stream
// produced here replays identically through WalkBlocks by construction.
func WalkBlocksWrite(w *entropy.Writer, cdfs *PartitionCDFs, ctx *PartitionContext, req BlockWalkRequest, decide PartitionDecider, visit BlockVisitor) (BlockWalkStats, error) {
	if w == nil || cdfs == nil || decide == nil {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	return walkBlocks(ctx, req, func(level BlockLevel, context int, haveRight bool, haveBottom bool) (Partition, error) {
		if !haveRight && !haveBottom {
			// Forced split: ReadPartition codes nothing, so neither do we.
			if level >= BlockLevel8x8 {
				return 0, ErrInvalidDecodeState
			}
			return PartitionSplit, nil
		}
		partition, err := decide(level, context, haveRight, haveBottom)
		if err != nil {
			return 0, err
		}
		if err := WritePartition(w, cdfs, level, context, haveRight, haveBottom, partition); err != nil {
			return 0, err
		}
		return partition, nil
	}, visit)
}
