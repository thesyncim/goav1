package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

// block_loop_write.go is the encoder-side superblock loop: it iterates root
// blocks in the decoder's exact order and reuses the decoder's own carrier
// load/store (blockLoopLoadRootContext / blockLoopStoreRootContext) so the
// partition, block-mode, and coefficient entropy contexts evolve across
// superblocks precisely as decodeBlockLoopWithCoeffControllerPtr restores them
// on the decode side.

// BlockLoopWriteVisitor encodes one leaf block. The scratch's Mode and CoeffCtx
// are the live per-superblock contexts (already loaded from the carrier); the
// visitor must mark them exactly as the decoder does after reading.
type BlockLoopWriteVisitor func(block BlockVisit, scratch *BlockLoopScratch) error

// WalkBlockLoopWrite runs the encoder block loop over the full walk region:
// for each root (superblock) in raster order it loads the edge contexts from
// the carrier, walks that root's partition tree with WalkBlocksWrite (coding
// the partition symbols), invokes visit for every leaf block, and stores the
// updated contexts back. The per-root walk uses the same neighbor bounds the
// decoder's loop passes, so HaveTop/HaveLeft and all context derivation match
// by construction.
func WalkBlockLoopWrite(w *entropy.Writer, partCDFs *PartitionCDFs, scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, req BlockWalkRequest, sbSizeMIB uint8, decide PartitionDecider, visit BlockLoopWriteVisitor) error {
	if w == nil || partCDFs == nil || scratch == nil || visit == nil {
		return ErrInvalidDecodeState
	}
	rootSize := uint32(req.Root.Size4x4())
	if rootSize == 0 {
		return ErrInvalidDecodeState
	}
	miColStart := uint32(req.MIColStart)
	miRowStart := uint32(req.MIRowStart)
	miColEnd := uint32(req.MIColEnd)
	miRowEnd := uint32(req.MIRowEnd)
	neighborMIColStart := req.neighborMIColStart()
	neighborMIRowStart := req.neighborMIRowStart()
	// The diagonal corner carriers feed the ref-MV outer scan's (-1,-1) cell
	// across superblock boundaries (see the decoder loop in block_loop.go);
	// without them the encoder's stack drops the above-left corner candidate
	// and its MV predictors desync from the decoder for blocks whose corner
	// neighbor is 16x16 or larger.
	ensureIntrabcDiagonalCarriers(carrier)
	for miRow := miRowStart; miRow < miRowEnd; miRow += rootSize {
		promotePendingDiagonalCarriers(carrier)
		for miCol := miColStart; miCol < miColEnd; miCol += rootSize {
			rootColIndex := int((miCol - miColStart) / rootSize)
			if err := blockLoopLoadRootContext(scratch, carrier, rootColIndex, miRow > neighborMIRowStart, miCol > neighborMIColStart, sbSizeMIB); err != nil {
				return err
			}
			rootReq := BlockWalkRequest{
				Root:               req.Root,
				MIColStart:         uint16(miCol),
				MIRowStart:         uint16(miRow),
				MIColEnd:           uint16(minUint32(miColEnd, miCol+rootSize)),
				MIRowEnd:           uint16(minUint32(miRowEnd, miRow+rootSize)),
				UseNeighborBounds:  true,
				NeighborMIColStart: uint16(neighborMIColStart),
				NeighborMIRowStart: uint16(neighborMIRowStart),
			}
			if _, err := WalkBlocksWrite(w, partCDFs, &scratch.Partition, rootReq, decide, func(block BlockVisit) error {
				return visit(block, scratch)
			}); err != nil {
				return err
			}
			if err := blockLoopStoreRootContext(scratch, carrier, rootColIndex, sbSizeMIB); err != nil {
				return err
			}
			captureDiagonalCornerToPending(carrier, rootColIndex+1, &scratch.Mode, sbSizeMIB)
		}
	}
	return nil
}
