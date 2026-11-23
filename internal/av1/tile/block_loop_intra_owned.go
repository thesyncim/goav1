package tile

import "github.com/thesyncim/goav1/internal/av1/parser"

// DecodeBlockLoopWithCoeffControllerPtrOwnedIntra is reserved for the decoder-owned
// full tile walk. Its caller excludes external scratch observers and hooks.
// The selection is call-scoped: every generic entry retains complete motion
// contexts, including key frames. No reusable request or scratch flag is set.
// dav1d decode.c:1263 and :2604 likewise omit ref-MV splats and row setup for
// key/intra-only frames without intra-block copy.
func DecodeBlockLoopWithCoeffControllerPtrOwnedIntra[T BlockLoopCoeffController](s *DecodeState, cdfs BlockLoopCDFs, scratch *BlockLoopScratch, req BlockLoopRequest, coeffController T, visit BlockLoopPointerVisitor) (BlockLoopStats, error) {
	if (req.FrameType != parser.FrameTypeKey && req.FrameType != parser.FrameTypeIntraOnly) || req.AllowIntrabc || req.SkipMode.Enabled || !req.DecodePredictionModes {
		return BlockLoopStats{}, ErrInvalidDecodeState
	}
	return decodeBlockLoopWithCoeffControllerPtr(s, cdfs, scratch, req, coeffController, true, true, visit)
}

// Only the edges and the per-block tx_size neighbor snapshot are read by the
// intra syntax loop. Motion grids and cross-SB histories remain untouched
// until a generic walk invokes resetForSuperblock, which clears every validity
// field before use.
func (c *BlockModeContext) resetForIntraSuperblock() {
	c.A = BlockContext{}
	c.L = BlockContext{}
	c.resetIntraSuperblockState()
}

func (c *BlockModeContext) resetIntraSuperblockState() {
	c.TxNeighborValid = false
	c.TxAboveNeighborIntra = 0
	c.TxAboveNeighborBlockSize = 0
	c.TxLeftNeighborIntra = 0
	c.TxLeftNeighborBlockSize = 0
}

// blockLoopLoadIntraRootContext loads one root's intra edge contexts; like
// blockLoopLoadRootContextMotion it keeps the left edge resident between
// consecutive roots of one walk.
func blockLoopLoadIntraRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int, haveTop bool, haveLeft bool, sbSizeMIB uint8, miRow, miCol uint32) error {
	if carrier == nil || rootColIndex < 0 || rootColIndex >= len(carrier.Above) {
		scratch.flushLeftEdge()
		scratch.Partition = PartitionContext{}
		scratch.Mode.resetForIntraSuperblock()
		scratch.CDEF.Reset()
		scratch.CoeffCtx = CoeffEntropyContext{}
		if carrier != nil {
			return ErrInvalidDecodeState
		}
		return nil
	}
	resident := haveLeft && scratch.leftEdgeResident(carrier, miRow, miCol, true)
	if !resident {
		scratch.flushLeftEdge()
	}
	scratch.leftEdge = blockLoopLeftEdge{}
	scratch.Mode.resetIntraSuperblockState()
	scratch.CDEF.Reset()
	if haveTop {
		edge := &carrier.Above[rootColIndex]
		scratch.Partition.Above = edge.Partition
		scratch.Mode.A = edge.mode.BlockContext
		scratch.CoeffCtx.Above = edge.Coeff
		// Keep the mode/size extension used by smaller-SB neighbor contexts.
		if sbSizeMIB != 0 && rootColIndex+1 < len(carrier.Above) {
			right := &carrier.Above[rootColIndex+1].mode
			for off := 0; off < int(sbSizeMIB) && int(sbSizeMIB)+off < MaxBlockModeSlots; off++ {
				dst := int(sbSizeMIB) + off
				scratch.Mode.A.Intra[dst] = right.Intra[off]
				scratch.Mode.A.Mode[dst] = right.Mode[off]
				scratch.Mode.A.ChromaIntra[dst] = right.ChromaIntra[off]
				scratch.Mode.A.UVMode[dst] = right.UVMode[off]
				scratch.Mode.A.BlockSize[dst] = right.BlockSize[off]
			}
		}
	} else {
		scratch.Mode.A = BlockContext{}
		scratch.Partition.Above = [MaxPartitionSlots]uint8{}
		scratch.CoeffCtx.Above = [3][MaxBlockModeSlots]uint8{}
	}
	if !haveLeft {
		scratch.Mode.L = BlockContext{}
		scratch.Partition.Left = [MaxPartitionSlots]uint8{}
		scratch.CoeffCtx.Left = [3][MaxBlockModeSlots]uint8{}
	} else if !resident {
		edge := &carrier.Left
		scratch.Partition.Left = edge.Partition
		scratch.Mode.L = edge.mode.BlockContext
		scratch.CoeffCtx.Left = edge.Coeff
	}
	return nil
}

func blockLoopStoreIntraRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int, miRow, nextMICol uint32) error {
	if carrier == nil {
		return nil
	}
	if rootColIndex < 0 || rootColIndex >= len(carrier.Above) {
		return ErrInvalidDecodeState
	}
	above := &carrier.Above[rootColIndex]
	above.Partition = scratch.Partition.Above
	above.mode.BlockContext = scratch.Mode.A
	above.Coeff = scratch.CoeffCtx.Above
	scratch.keepLeftEdge(carrier, miRow, nextMICol, true)
	return nil
}

// storeIntraLeftEdge is the left half of blockLoopStoreIntraRootContext.
func storeIntraLeftEdge(left *BlockLoopRootLeftContext, scratch *BlockLoopScratch) {
	left.Partition = scratch.Partition.Left
	left.mode.BlockContext = scratch.Mode.L
	left.Coeff = scratch.CoeffCtx.Left
}

// markOwnedIntra retains the edge sizes used by transform entropy contexts.
// The full MarkIntra API continues to maintain observable motion/ref grids.
func (c *BlockModeContext) markOwnedIntra(size BlockSize, x4 int, y4 int, mode IntraMode) error {
	if err := c.MarkIntraEntry(size, x4, y4, true, mode); err != nil {
		return err
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	for i := range int(dims.W4) {
		c.A.BlockSize[x4+i] = size
	}
	for i := range int(dims.H4) {
		c.L.BlockSize[y4+i] = size
	}
	return nil
}
