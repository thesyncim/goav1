package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	EncoderDecisionBlockLevelCount        = int(tile.BlockLevel8x8) + 1
	EncoderDecisionPartitionCount         = int(tile.PartitionV4) + 1
	EncoderDecisionBlockSizeCount         = int(tile.BlockSize4x4) + 1
	EncoderDecisionInterModeCount         = int(tile.InterModeNewMV) + 1
	EncoderDecisionCompoundInterModeCount = int(tile.CompoundInterModeNewNew) + 1
	EncoderDecisionReferenceFrameCount    = int(tile.ReferenceFrameAltref) + 1
	EncoderDecisionTransformTypeCount     = int(transform.TypeCount)

	EncoderDecisionBlockSize8x8   = int(tile.BlockSize8x8)
	EncoderDecisionBlockSize16x16 = int(tile.BlockSize16x16)
	EncoderDecisionBlockSize32x32 = int(tile.BlockSize32x32)
	EncoderDecisionBlockSize64x64 = int(tile.BlockSize64x64)
	EncoderDecisionBlockSize16x8  = int(tile.BlockSize16x8)
	EncoderDecisionBlockSize8x16  = int(tile.BlockSize8x16)
	EncoderDecisionBlockSize32x16 = int(tile.BlockSize32x16)
	EncoderDecisionBlockSize16x32 = int(tile.BlockSize16x32)

	EncoderDecisionReferenceLast   = int(tile.ReferenceFrameLast)
	EncoderDecisionReferenceGolden = int(tile.ReferenceFrameGolden)

	EncoderDecisionTransformDCTDCT   = int(transform.TypeDCTDCT)
	EncoderDecisionTransformADSTDCT  = int(transform.TypeADSTDCT)
	EncoderDecisionTransformDCTADST  = int(transform.TypeDCTADST)
	EncoderDecisionTransformADSTADST = int(transform.TypeADSTADST)
	EncoderDecisionTransformIDTX     = int(transform.TypeIDTX)

	// EncoderDecisionTransformSizeCount covers every tile.TransformSize
	// value (TX_* then RTX_* order; 64x16 is the last).
	EncoderDecisionTransformSizeCount = int(tile.TransformSize64x16) + 1
)

// EncoderDecisionStats is an optional encoder-side diagnostic snapshot. It
// counts coded syntax decisions made by the realtime encoder; it is not a
// quality metric and should be read beside decoded-output RD metrics.
type EncoderDecisionStats struct {
	Frames      uint64
	Keyframes   uint64
	InterFrames uint64
	Tiles       uint64

	PartitionDecisions uint64
	Partitions         [EncoderDecisionPartitionCount]uint64
	PartitionsByLevel  [EncoderDecisionBlockLevelCount][EncoderDecisionPartitionCount]uint64

	Blocks         uint64
	InterBlocks    uint64
	IntraBlocks    uint64
	SkipBlocks     uint64
	CodedBlocks    uint64
	CompoundBlocks uint64
	BlockSizes     [EncoderDecisionBlockSizeCount]uint64

	PrimaryReferenceBlocks [EncoderDecisionReferenceFrameCount]uint64
	ReferenceUses          [EncoderDecisionReferenceFrameCount]uint64
	InterModes             [EncoderDecisionInterModeCount]uint64
	CompoundInterModes     [EncoderDecisionCompoundInterModeCount]uint64

	SplitTXBlocks  uint64
	SingleTXBlocks uint64
	LumaTXBs       uint64
	TXTypes        [EncoderDecisionTransformTypeCount]uint64
	NonDCTTXBs     uint64

	// InterLumaTXBsBySize / InterChromaTXBsBySize count the transform blocks
	// actually coded by inter blocks (skip blocks code none), indexed by
	// tile.TransformSize. Unlike LumaTXBs above these use the exact leaf
	// count of the realtime var-tx plan (a split 64x64 codes 16 leaves).
	InterLumaTXBsBySize   [EncoderDecisionTransformSizeCount]uint64
	InterChromaTXBsBySize [EncoderDecisionTransformSizeCount]uint64

	// InterpFilters counts the switchable interpolation filter pairs the
	// base-frame filter search coded (REGULAR, SMOOTH, SHARP), indexed by
	// the AV1 switchable filter symbol.
	InterpFilters [3]uint64
}

func (s *EncoderDecisionStats) Reset() {
	*s = EncoderDecisionStats{}
}

func (s *EncoderDecisionStats) add(other EncoderDecisionStats) {
	s.Frames += other.Frames
	s.Keyframes += other.Keyframes
	s.InterFrames += other.InterFrames
	s.Tiles += other.Tiles
	s.PartitionDecisions += other.PartitionDecisions
	for i := range s.Partitions {
		s.Partitions[i] += other.Partitions[i]
	}
	for i := range s.PartitionsByLevel {
		for j := range s.PartitionsByLevel[i] {
			s.PartitionsByLevel[i][j] += other.PartitionsByLevel[i][j]
		}
	}
	s.Blocks += other.Blocks
	s.InterBlocks += other.InterBlocks
	s.IntraBlocks += other.IntraBlocks
	s.SkipBlocks += other.SkipBlocks
	s.CodedBlocks += other.CodedBlocks
	s.CompoundBlocks += other.CompoundBlocks
	for i := range s.BlockSizes {
		s.BlockSizes[i] += other.BlockSizes[i]
	}
	for i := range s.PrimaryReferenceBlocks {
		s.PrimaryReferenceBlocks[i] += other.PrimaryReferenceBlocks[i]
		s.ReferenceUses[i] += other.ReferenceUses[i]
	}
	for i := range s.InterModes {
		s.InterModes[i] += other.InterModes[i]
	}
	for i := range s.CompoundInterModes {
		s.CompoundInterModes[i] += other.CompoundInterModes[i]
	}
	s.SplitTXBlocks += other.SplitTXBlocks
	s.SingleTXBlocks += other.SingleTXBlocks
	s.LumaTXBs += other.LumaTXBs
	for i := range s.TXTypes {
		s.TXTypes[i] += other.TXTypes[i]
	}
	s.NonDCTTXBs += other.NonDCTTXBs
	for i := range s.InterLumaTXBsBySize {
		s.InterLumaTXBsBySize[i] += other.InterLumaTXBsBySize[i]
		s.InterChromaTXBsBySize[i] += other.InterChromaTXBsBySize[i]
	}
	for i := range s.InterpFilters {
		s.InterpFilters[i] += other.InterpFilters[i]
	}
}

// noteInterTXBs records the exact per-size transform-block counts one coded
// (non-skip) inter block emits: lumaCount leaves of lumaTX plus chromaCount
// blocks of chromaTX.
func (s *EncoderDecisionStats) noteInterTXBs(lumaTX tile.TransformSize, lumaCount int, chromaTX tile.TransformSize, chromaCount int) {
	if s == nil {
		return
	}
	if int(lumaTX) < len(s.InterLumaTXBsBySize) && lumaCount > 0 {
		s.InterLumaTXBsBySize[lumaTX] += uint64(lumaCount)
	}
	if int(chromaTX) < len(s.InterChromaTXBsBySize) && chromaCount > 0 {
		s.InterChromaTXBsBySize[chromaTX] += uint64(chromaCount)
	}
}

// noteInterTXBSizes feeds the exact TXB-size histogram for one coded
// (non-skip) inter block; the fused and split decision paths share it.
func (st *lossyEncodeState) noteInterTXBSizes(blockSize tile.BlockSize, splitTX bool, txPlan realtimeInterTXPlan, bw, bh int) {
	if st.decisionStats == nil {
		return
	}
	var lumaTX tile.TransformSize
	lumaCount := 1
	if splitTX {
		lumaTX = txPlan.leafTX
		lumaCount = (bw / txPlan.leafSize) * (bh / txPlan.leafSize)
	} else {
		lumaTX, _ = st.interLumaTXAndScan(bw, bh)
	}
	chromaTX := tile.TransformSize4x4
	chromaCount := 0
	if !st.color.MonoChrome {
		if tx, err := tile.MaxTransformSize(blockSize, st.color, 1); err == nil {
			chromaTX, chromaCount = tx, 2
		}
	}
	st.decisionStats.noteInterTXBs(lumaTX, lumaCount, chromaTX, chromaCount)
}

func (s *EncoderDecisionStats) noteInterpFilter(f int) {
	if s == nil || f < 0 || f >= len(s.InterpFilters) {
		return
	}
	s.InterpFilters[f]++
}

func (s *EncoderDecisionStats) notePartition(level tile.BlockLevel, partition tile.Partition) {
	if s == nil {
		return
	}
	s.PartitionDecisions++
	if int(partition) < len(s.Partitions) {
		s.Partitions[partition]++
	}
	if int(level) < len(s.PartitionsByLevel) && int(partition) < len(s.PartitionsByLevel[level]) {
		s.PartitionsByLevel[level][partition]++
	}
}

func (s *EncoderDecisionStats) noteIntraBlock(size tile.BlockSize) {
	if s == nil {
		return
	}
	s.noteBlockSize(size)
	s.Blocks++
	s.IntraBlocks++
	s.CodedBlocks++
	s.SingleTXBlocks++
	s.noteTXB(transform.TypeDCTDCT)
}

func (s *EncoderDecisionStats) noteInterBlock(size tile.BlockSize, skip bool, splitTX bool, refs tile.InterReferencesResult, mode tile.InterModeResult, txType transform.Type) {
	if s == nil {
		return
	}
	s.noteBlockSize(size)
	s.Blocks++
	s.InterBlocks++
	if skip {
		s.SkipBlocks++
	} else {
		s.CodedBlocks++
		if splitTX {
			s.SplitTXBlocks++
			for range 4 {
				s.noteTXB(transform.TypeDCTDCT)
			}
		} else {
			s.SingleTXBlocks++
			s.noteTXB(txType)
		}
	}
	if refs.Compound {
		s.CompoundBlocks++
		if int(mode.CompoundMode) < len(s.CompoundInterModes) {
			s.CompoundInterModes[mode.CompoundMode]++
		}
	} else if int(mode.Mode) < len(s.InterModes) {
		s.InterModes[mode.Mode]++
	}
	if refs.Ref[0].Valid() {
		s.PrimaryReferenceBlocks[refs.Ref[0]]++
		s.ReferenceUses[refs.Ref[0]]++
	}
	if refs.Compound && refs.Ref[1].Valid() {
		s.ReferenceUses[refs.Ref[1]]++
	}
}

func (s *EncoderDecisionStats) noteBlockSize(size tile.BlockSize) {
	if int(size) < len(s.BlockSizes) {
		s.BlockSizes[size]++
	}
}

func (s *EncoderDecisionStats) noteTXB(txType transform.Type) {
	if !txType.Valid() {
		return
	}
	s.LumaTXBs++
	s.TXTypes[txType]++
	if txType != transform.TypeDCTDCT {
		s.NonDCTTXBs++
	}
}
