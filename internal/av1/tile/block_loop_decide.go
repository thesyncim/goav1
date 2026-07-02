package tile

// block_loop_decide.go is the symbol-free twin of block_loop_write.go: it runs
// the encoder superblock loop with the decoder's exact traversal, carrier
// load/store, and availability derivation, but codes nothing. It backs the
// encoder's decision pass in the libaom encode/pack split
// (av1/encoder/encodeframe.c encode_sb stores decisions without touching the
// bitstream; av1/encoder/bitstream.c write_modes packs them later — SVT-AV1's
// MODE DECISION / ENC-DEC vs ENTROPY CODING stage split is the same shape).

// WalkBlocksDecide traverses the MI region with the decoder's walkBlocks state
// machine, asking decide for each coded node exactly where WalkBlocksWrite
// would, but writes no partition symbols. Forced-split nodes are handled
// identically (decide is not consulted).
func WalkBlocksDecide(ctx *PartitionContext, req BlockWalkRequest, decide PartitionDecider, visit BlockVisitor) (BlockWalkStats, error) {
	if decide == nil {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	return walkBlocks(ctx, req, func(level BlockLevel, context int, miCol uint32, miRow uint32, haveRight bool, haveBottom bool) (Partition, error) {
		if !haveRight && !haveBottom {
			// Forced split: ReadPartition codes nothing, so neither does the
			// write walk; keep the decide walk in lockstep.
			if level >= BlockLevel8x8 {
				return 0, ErrInvalidDecodeState
			}
			return PartitionSplit, nil
		}
		return decide(level, context, miCol, miRow, haveRight, haveBottom)
	}, visit)
}

// BlockLoopWavefrontCarriers owns the cross-row edge state for concurrent
// SB-row decide walks: the shared per-column above contexts (each column slot
// is written by row r exactly once, after row r+1 can no longer read it — the
// top-right wavefront dependency makes the single shared array race-free, the
// same discipline as libaom's shared above_context under row-mt) and two
// diagonal parity planes replacing the serial walk's pending/promote scheme:
// row r reads plane r%2 and captures its bottom-right corners into plane
// (r+1)%2, which under the top-right dependency delivers exactly the values
// the serial promote would.
type BlockLoopWavefrontCarriers struct {
	above []BlockLoopRootAboveContext
	diag  [2][]diagonalCornerSlot
}

// Reset clears the carriers for a fresh frame walk over rootCols root columns,
// retaining capacity across frames.
func (s *BlockLoopWavefrontCarriers) Reset(rootCols int) {
	if cap(s.above) < rootCols {
		s.above = make([]BlockLoopRootAboveContext, rootCols)
	}
	s.above = s.above[:rootCols]
	for i := range s.above {
		s.above[i] = BlockLoopRootAboveContext{}
	}
	for p := range s.diag {
		if cap(s.diag[p]) < rootCols {
			s.diag[p] = make([]diagonalCornerSlot, rootCols)
		}
		s.diag[p] = s.diag[p][:rootCols]
		for i := range s.diag[p] {
			s.diag[p][i] = diagonalCornerSlot{}
		}
	}
}

// BindRow points carrier at the shared above slots and the row's diagonal
// parity planes. The carrier's Left context stays caller-owned (each row
// worker keeps its own; it is never read at a row's first column and is
// overwritten by every store).
func (s *BlockLoopWavefrontCarriers) BindRow(carrier *BlockLoopContextCarrier, sbRow int) {
	carrier.Above = s.above
	carrier.Diagonal = s.diag[sbRow&1]
	carrier.PendingDiagonal = s.diag[(sbRow+1)&1]
}

// WalkBlockLoopDecideSB runs the decide walk for the single superblock at
// (miRow, miCol): carrier load, partition-tree walk with visits, carrier
// store, and the diagonal-corner capture — one wavefront work unit. The
// caller guarantees the top-right dependency (the SB above-right has been
// stored) before calling; req is the full-tile walk request.
func WalkBlockLoopDecideSB(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, req BlockWalkRequest, sbSizeMIB uint8, miRow, miCol uint32, decide PartitionDecider, visit BlockLoopWriteVisitor) error {
	if scratch == nil || decide == nil || visit == nil {
		return ErrInvalidDecodeState
	}
	rootSize := uint32(req.Root.Size4x4())
	if rootSize == 0 {
		return ErrInvalidDecodeState
	}
	miColStart := uint32(req.MIColStart)
	miColEnd := uint32(req.MIColEnd)
	miRowEnd := uint32(req.MIRowEnd)
	neighborMIColStart := req.neighborMIColStart()
	neighborMIRowStart := req.neighborMIRowStart()
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
	if _, err := WalkBlocksDecide(&scratch.Partition, rootReq, decide, func(block BlockVisit) error {
		return visit(block, scratch)
	}); err != nil {
		return err
	}
	if err := blockLoopStoreRootContext(scratch, carrier, rootColIndex, sbSizeMIB); err != nil {
		return err
	}
	captureDiagonalCornerToPending(carrier, rootColIndex+1, &scratch.Mode, sbSizeMIB)
	return nil
}

// WalkBlockLoopDecide runs the encoder block loop over the full walk region
// exactly like WalkBlockLoopWrite — same carrier load/store per root, same
// neighbor bounds, same diagonal-corner carriers feeding the ref-MV outer
// scan's (-1,-1) cell — but codes no symbols. The visitor performs the
// encoder's search/prediction/reconstruction work and must mark the scratch's
// Mode context exactly as the fused path does, so MV prediction and all
// neighbor-derived decisions see the serial coding order's state.
func WalkBlockLoopDecide(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, req BlockWalkRequest, sbSizeMIB uint8, decide PartitionDecider, visit BlockLoopWriteVisitor) error {
	if scratch == nil || decide == nil || visit == nil {
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
			if _, err := WalkBlocksDecide(&scratch.Partition, rootReq, decide, func(block BlockVisit) error {
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
