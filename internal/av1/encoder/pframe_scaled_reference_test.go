package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestEncodeScaledReferencePFrameParsesReferenceSizedHeader(t *testing.T) {
	ref := scaledReferenceTestFrame(32, 32)
	src := scaledReferenceTestSource(t, ref, 64, 64)

	tu, recon, err := EncodeScaledReferencePFrame(src, ref, 72)
	if err != nil {
		t.Fatalf("EncodeScaledReferencePFrame: %v", err)
	}
	if recon.Width != src.Width || recon.Height != src.Height ||
		len(recon.Y) != len(src.Y) || len(recon.U) != len(src.U) || len(recon.V) != len(src.V) {
		t.Fatalf("recon=%+v src=%+v", recon, src)
	}

	it := obu.NewLowOverheadIterator(tu)
	unit, ok, err := it.Next()
	if err != nil || !ok || unit.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("temporal delimiter ok=%v err=%v unit=%+v", ok, err, unit.Header)
	}
	frameHeader, ok, err := it.Next()
	if err != nil || !ok || frameHeader.Header.Type != obu.TypeFrameHeader {
		t.Fatalf("frame header ok=%v err=%v unit=%+v", ok, err, frameHeader.Header)
	}

	seq := losslessKeyframeSequence(src.Width, src.Height)
	parsedSeq := parseEncoderSequenceHeader(t, seq)
	prefix, err := parser.ParseFrameHeaderPrefix(frameHeader.Payload, parsedSeq)
	if err != nil {
		t.Fatalf("ParseFrameHeaderPrefix: %v", err)
	}
	refs := referenceStateForFrame(ref.Width, ref.Height)
	size, err := parser.ParseFrameSize(frameHeader.Payload, parsedSeq, prefix, &refs, 0, 0)
	if err != nil {
		t.Fatalf("ParseFrameSize: %v", err)
	}
	if prefix.FrameType != parser.FrameTypeInter || !prefix.ShowFrame || !prefix.ErrorResilientMode {
		t.Fatalf("prefix=%+v", prefix)
	}
	if size.UpscaledWidth != uint32(src.Width) || size.Height != uint32(src.Height) ||
		size.RefFrameIdx[0] != 0 || size.RefreshFrameFlags != 0x01 {
		t.Fatalf("size=%+v", size)
	}

	tileGroup, ok, err := it.Next()
	if err != nil || !ok || tileGroup.Header.Type != obu.TypeTileGroup || len(tileGroup.Payload) == 0 {
		t.Fatalf("tile group ok=%v err=%v unit=%+v payload=%d", ok, err, tileGroup.Header, len(tileGroup.Payload))
	}
	if extra, ok, err := it.Next(); err != nil || ok {
		t.Fatalf("extra ok=%v err=%v unit=%+v", ok, err, extra.Header)
	}
}

func TestEncodeScaledReferenceTileUses8x8Leaves(t *testing.T) {
	ref := scaledReferenceTestFrame(32, 32)
	src := scaledReferenceTestSource(t, ref, 64, 64)
	recon := SourceFrame420{
		Y:            make([]byte, len(src.Y)),
		U:            make([]byte, len(src.U)),
		V:            make([]byte, len(src.V)),
		YStride:      src.YStride,
		ChromaStride: src.ChromaStride,
		Width:        src.Width,
		Height:       src.Height,
	}

	var pc pframeCoder
	pc.decisionStatsEnabled = true
	payload, err := pc.encodeTile(src, ref, nil, &recon, 72, nil, parser.ReferenceModeSingle, 0, uint16(src.Width/4))
	if err != nil {
		t.Fatalf("encodeTile: %v", err)
	}
	if len(payload) == 0 {
		t.Fatal("empty tile payload")
	}
	stats := pc.decisionStats
	wantBlocks := uint64((src.Width / 8) * (src.Height / 8))
	if stats.Blocks != wantBlocks || stats.BlockSizes[tile.BlockSize8x8] != wantBlocks {
		t.Fatalf("stats=%+v want %d 8x8 blocks", stats, wantBlocks)
	}
	if stats.BlockSizes[tile.BlockSize16x16] != 0 ||
		stats.BlockSizes[tile.BlockSize32x32] != 0 ||
		stats.BlockSizes[tile.BlockSize64x64] != 0 {
		t.Fatalf("scaled reference used merged blocks: %+v", stats.BlockSizes)
	}
	if stats.InterBlocks == 0 || stats.PrimaryReferenceBlocks[tile.ReferenceFrameLast] == 0 {
		t.Fatalf("missing LAST inter blocks: %+v", stats)
	}
}

func scaledReferenceTestFrame(width, height int) SourceFrame420 {
	cw, ch := width/2, height/2
	f := SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	for y := range height {
		for x := range width {
			f.Y[y*width+x] = byte(31 + 3*x + 5*y + x*y)
		}
	}
	for y := range ch {
		for x := range cw {
			f.U[y*cw+x] = byte(83 + 7*x + 2*y)
			f.V[y*cw+x] = byte(129 + 5*x + 11*y)
		}
	}
	return f
}

func scaledReferenceTestSource(t *testing.T, ref SourceFrame420, width, height int) SourceFrame420 {
	t.Helper()
	cw, ch := width/2, height/2
	src := SourceFrame420{
		Y:            make([]byte, width*height),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      width,
		ChromaStride: cw,
		Width:        width,
		Height:       height,
	}
	copyScaledReferencePlane(t, src.Y, src.YStride, width, height, ref.Y, ref.YStride, ref.Width, ref.Height, 8, 8, false, false)
	copyScaledReferencePlane(t, src.U, src.ChromaStride, cw, ch, ref.U, ref.ChromaStride, ref.Width/2, ref.Height/2, 4, 4, true, true)
	copyScaledReferencePlane(t, src.V, src.ChromaStride, cw, ch, ref.V, ref.ChromaStride, ref.Width/2, ref.Height/2, 4, 4, true, true)
	return src
}

func copyScaledReferencePlane(t *testing.T, dst []byte, dstStride, curWidth, curHeight int, ref []byte, refStride, refWidth, refHeight int, bw, bh int, ssX, ssY bool) {
	t.Helper()
	tmp := make([]byte, bw*bh)
	for py := 0; py < curHeight; py += bh {
		for px := 0; px < curWidth; px += bw {
			if err := predictIntoScaled(tmp, ref, refStride, refWidth, refHeight, curWidth, curHeight, px, py, bw, bh, motion.Vector{}, ssX, ssY, nil); err != nil {
				t.Fatalf("predict block %dx%d at %d,%d: %v", bw, bh, px, py, err)
			}
			for y := range bh {
				copy(dst[(py+y)*dstStride+px:], tmp[y*bw:(y+1)*bw])
			}
		}
	}
}
