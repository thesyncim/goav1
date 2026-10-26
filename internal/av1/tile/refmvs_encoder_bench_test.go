package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// BenchmarkEncoderReferenceMVStack times dav1d_refmvs_find through the
// encoder entry on a 720p frame of 64x64 LAST blocks with varied vectors
// (64x64 is three quarters of the realtime encoder's light-PD1 queries).
// The blocks of the third superblock row are queried; every neighbour is
// splatted, so each query finds about four candidates.
func BenchmarkEncoderReferenceMVStack(b *testing.B) {
	const cols, rows = 320, 180
	var s BlockLoopScratch
	BindEncoderRefmvs(&s, EncoderRefmvsFrame{
		FrameMICols: cols, FrameMIRows: rows,
		Tile: BlockWalkRequest{MIColEnd: cols, MIRowEnd: rows},
	})
	refs := InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}
	var blocks []BlockVisit
	for sby := int32(0); sby < 3; sby++ {
		s.Refmvs.sbrowInit(sby, 16)
		for bx := uint16(0); bx < cols; bx += 16 {
			blk := BlockVisit{MICol: bx, MIRow: uint16(sby * 16), Size: BlockSize64x64,
				HaveTop: sby > 0, HaveLeft: bx > 0, Edge: edgeI444TopHasRight}
			if sby == 2 {
				blocks = append(blocks, blk)
			}
			mv := motion.Vector{Row: int16(bx%48) - 8, Col: int16(sby*4) - int16(bx%32)}
			s.Refmvs.splatInterBlock(blk.Size, int32(blk.MIRow), int32(blk.MICol),
				InterMotionResult{MV: [2]motion.Vector{mv}, References: refs, Mode: InterModeResult{Mode: InterModeNewMV}})
		}
	}
	var dst ReferenceMVStackResult
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blk := &blocks[i%len(blocks)]
		s.EncoderReferenceMVStack(&dst, refs, blk)
	}
}
