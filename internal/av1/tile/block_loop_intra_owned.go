package tile

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

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

// Only these two pointer-free spans are read by the intra syntax loop. Edge
// block sizes and the per-block neighbor snapshot remain necessary for tx_size.
// Motion grids and cross-SB histories remain untouched until a generic walk
// invokes resetForSuperblock, which clears every validity field before use.
func (c *BlockModeContext) resetForIntraSuperblock() {
	clear(unsafe.Slice((*byte)(unsafe.Pointer(c)), unsafe.Offsetof(c.AboveRef)))
	clear(unsafe.Slice((*byte)(unsafe.Pointer(&c.AboveBlockSize)), unsafe.Offsetof(c.GridMotionValid)-unsafe.Offsetof(c.AboveBlockSize)))
}

func blockLoopLoadIntraRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int, haveTop bool, haveLeft bool, sbSizeMIB uint8) error {
	scratch.Partition = PartitionContext{}
	scratch.Mode.resetForIntraSuperblock()
	scratch.CDEF.Reset()
	scratch.CoeffCtx = CoeffEntropyContext{}
	if carrier == nil {
		return nil
	}
	if rootColIndex < 0 || rootColIndex >= len(carrier.Above) {
		return ErrInvalidDecodeState
	}
	if haveTop {
		edge := &carrier.Above[rootColIndex]
		scratch.Partition.Above = edge.Partition
		scratch.Mode.AboveSkip = edge.mode.Skip
		scratch.Mode.AboveSkipMode = edge.mode.SkipMode
		scratch.Mode.AboveSegmentPred = edge.mode.SegmentPred
		scratch.Mode.AboveIntra = edge.mode.Intra
		scratch.Mode.AboveMode = edge.mode.Mode
		scratch.Mode.AboveChromaIntra = edge.mode.ChromaIntra
		scratch.Mode.AboveChromaMode = edge.mode.ChromaMode
		scratch.Mode.AboveTxIntra = edge.mode.TxIntra
		scratch.Mode.AboveTx = edge.mode.Tx
		scratch.Mode.AboveBlockSize = edge.mode.BlockSize
		scratch.Mode.AbovePaletteY = edge.mode.PaletteY
		scratch.Mode.AbovePaletteUV = edge.mode.PaletteUV
		for plane := range 3 {
			scratch.CoeffCtx.Above[plane] = edge.Coeff[plane]
		}
		// Keep the mode/size extension used by smaller-SB neighbor contexts.
		if sbSizeMIB != 0 && rootColIndex+1 < len(carrier.Above) {
			right := &carrier.Above[rootColIndex+1].mode
			for off := 0; off < int(sbSizeMIB) && int(sbSizeMIB)+off < MaxBlockModeSlots; off++ {
				dst := int(sbSizeMIB) + off
				scratch.Mode.AboveIntra[dst] = right.Intra[off]
				scratch.Mode.AboveMode[dst] = right.Mode[off]
				scratch.Mode.AboveChromaIntra[dst] = right.ChromaIntra[off]
				scratch.Mode.AboveChromaMode[dst] = right.ChromaMode[off]
				scratch.Mode.AboveBlockSize[dst] = right.BlockSize[off]
			}
		}
	}
	if haveLeft {
		edge := &carrier.Left
		scratch.Partition.Left = edge.Partition
		scratch.Mode.LeftSkip = edge.mode.Skip
		scratch.Mode.LeftSkipMode = edge.mode.SkipMode
		scratch.Mode.LeftSegmentPred = edge.mode.SegmentPred
		scratch.Mode.LeftIntra = edge.mode.Intra
		scratch.Mode.LeftMode = edge.mode.Mode
		scratch.Mode.LeftChromaIntra = edge.mode.ChromaIntra
		scratch.Mode.LeftChromaMode = edge.mode.ChromaMode
		scratch.Mode.LeftTxIntra = edge.mode.TxIntra
		scratch.Mode.LeftTx = edge.mode.Tx
		scratch.Mode.LeftBlockSize = edge.mode.BlockSize
		scratch.Mode.LeftPaletteY = edge.mode.PaletteY
		scratch.Mode.LeftPaletteUV = edge.mode.PaletteUV
		for plane := range 3 {
			scratch.CoeffCtx.Left[plane] = edge.Coeff[plane]
		}
	}
	return nil
}

func blockLoopStoreIntraRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int) error {
	if carrier == nil {
		return nil
	}
	if rootColIndex < 0 || rootColIndex >= len(carrier.Above) {
		return ErrInvalidDecodeState
	}
	above := &carrier.Above[rootColIndex]
	above.Partition = scratch.Partition.Above
	above.mode.Skip = scratch.Mode.AboveSkip
	above.mode.SkipMode = scratch.Mode.AboveSkipMode
	above.mode.SegmentPred = scratch.Mode.AboveSegmentPred
	above.mode.Intra = scratch.Mode.AboveIntra
	above.mode.Mode = scratch.Mode.AboveMode
	above.mode.ChromaIntra = scratch.Mode.AboveChromaIntra
	above.mode.ChromaMode = scratch.Mode.AboveChromaMode
	above.mode.TxIntra = scratch.Mode.AboveTxIntra
	above.mode.Tx = scratch.Mode.AboveTx
	above.mode.BlockSize = scratch.Mode.AboveBlockSize
	above.mode.PaletteY = scratch.Mode.AbovePaletteY
	above.mode.PaletteUV = scratch.Mode.AbovePaletteUV
	for plane := range 3 {
		above.Coeff[plane] = scratch.CoeffCtx.Above[plane]
	}
	left := &carrier.Left
	left.Partition = scratch.Partition.Left
	left.mode.Skip = scratch.Mode.LeftSkip
	left.mode.SkipMode = scratch.Mode.LeftSkipMode
	left.mode.SegmentPred = scratch.Mode.LeftSegmentPred
	left.mode.Intra = scratch.Mode.LeftIntra
	left.mode.Mode = scratch.Mode.LeftMode
	left.mode.ChromaIntra = scratch.Mode.LeftChromaIntra
	left.mode.ChromaMode = scratch.Mode.LeftChromaMode
	left.mode.TxIntra = scratch.Mode.LeftTxIntra
	left.mode.Tx = scratch.Mode.LeftTx
	left.mode.BlockSize = scratch.Mode.LeftBlockSize
	left.mode.PaletteY = scratch.Mode.LeftPaletteY
	left.mode.PaletteUV = scratch.Mode.LeftPaletteUV
	for plane := range 3 {
		left.Coeff[plane] = scratch.CoeffCtx.Left[plane]
	}
	return nil
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
		c.AboveBlockSize[x4+i] = size
	}
	for i := range int(dims.H4) {
		c.LeftBlockSize[y4+i] = size
	}
	return nil
}
