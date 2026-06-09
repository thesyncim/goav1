package tile

import (
	"errors"
	"testing"
)

type scriptedPartitionReader struct {
	partitions []Partition
	calls      []scriptedPartitionCall
}

type scriptedPartitionCall struct {
	level      BlockLevel
	ctx        int
	haveRight  bool
	haveBottom bool
}

func (r *scriptedPartitionReader) read(level BlockLevel, ctx int, _ uint32, _ uint32, haveRight bool, haveBottom bool) (Partition, error) {
	r.calls = append(r.calls, scriptedPartitionCall{
		level:      level,
		ctx:        ctx,
		haveRight:  haveRight,
		haveBottom: haveBottom,
	})
	if len(r.partitions) == 0 {
		return 0, ErrInvalidDecodeState
	}
	partition := r.partitions[0]
	r.partitions = r.partitions[1:]
	return partition, nil
}

func TestWalkBlocksNonePartition(t *testing.T) {
	var cdfs PartitionCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var ctx PartitionContext
	var visits []BlockVisit
	stats, err := state.WalkBlocks(&cdfs, &ctx, BlockWalkRequest{
		Root:       BlockLevel64x64,
		MIColStart: 0,
		MIRowStart: 0,
		MIColEnd:   16,
		MIRowEnd:   16,
	}, func(visit BlockVisit) error {
		visits = append(visits, visit)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockWalkStats{PartitionReads: 1, Blocks: 1}) {
		t.Fatalf("stats=%+v want reads=1 blocks=1", stats)
	}
	if len(visits) != 1 || visits[0].Size != BlockSize64x64 ||
		visits[0].MICol != 0 || visits[0].MIRow != 0 ||
		visits[0].MIColEnd != 16 || visits[0].MIRowEnd != 16 ||
		visits[0].VisibleW4 != 16 || visits[0].VisibleH4 != 16 ||
		visits[0].X4 != 0 || visits[0].Y4 != 0 ||
		visits[0].HaveTop || visits[0].HaveLeft {
		t.Fatalf("visits=%+v", visits)
	}
	if ctx.Above[0] != 0x10 || ctx.Left[0] != 0x10 {
		t.Fatalf("partition context above=%#x left=%#x want none markers", ctx.Above[0], ctx.Left[0])
	}
}

func TestWalkBlocksScriptedSplitOrderMatchesDav1d(t *testing.T) {
	reader := scriptedPartitionReader{partitions: []Partition{
		PartitionSplit,
		PartitionNone,
		PartitionH,
		PartitionV,
		PartitionTTopSplit,
	}}
	var ctx PartitionContext
	var visits []BlockVisit
	stats, err := walkBlocks(&ctx, BlockWalkRequest{
		Root:       BlockLevel64x64,
		MIColStart: 0,
		MIRowStart: 0,
		MIColEnd:   16,
		MIRowEnd:   16,
	}, reader.read, func(visit BlockVisit) error {
		visits = append(visits, visit)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockWalkStats{PartitionReads: 5, Blocks: 8}) {
		t.Fatalf("stats=%+v want reads=5 blocks=8", stats)
	}
	want := []BlockVisit{
		{MICol: 0, MIRow: 0, MIColEnd: 8, MIRowEnd: 8, X4: 0, Y4: 0, Level: BlockLevel32x32, Partition: PartitionNone, Size: BlockSize32x32, VisibleW4: 8, VisibleH4: 8},
		{MICol: 8, MIRow: 0, MIColEnd: 16, MIRowEnd: 4, X4: 8, Y4: 0, Level: BlockLevel32x32, Partition: PartitionH, Size: BlockSize32x16, VisibleW4: 8, VisibleH4: 4, HaveLeft: true},
		{MICol: 8, MIRow: 4, MIColEnd: 16, MIRowEnd: 8, X4: 8, Y4: 4, Level: BlockLevel32x32, Partition: PartitionH, Size: BlockSize32x16, VisibleW4: 8, VisibleH4: 4, HaveTop: true, HaveLeft: true},
		{MICol: 0, MIRow: 8, MIColEnd: 4, MIRowEnd: 16, X4: 0, Y4: 8, Level: BlockLevel32x32, Partition: PartitionV, Size: BlockSize16x32, VisibleW4: 4, VisibleH4: 8, HaveTop: true},
		{MICol: 4, MIRow: 8, MIColEnd: 8, MIRowEnd: 16, X4: 4, Y4: 8, Level: BlockLevel32x32, Partition: PartitionV, Size: BlockSize16x32, VisibleW4: 4, VisibleH4: 8, HaveTop: true, HaveLeft: true},
		{MICol: 8, MIRow: 8, MIColEnd: 12, MIRowEnd: 12, X4: 8, Y4: 8, Level: BlockLevel32x32, Partition: PartitionTTopSplit, Size: BlockSize16x16, VisibleW4: 4, VisibleH4: 4, HaveTop: true, HaveLeft: true},
		{MICol: 12, MIRow: 8, MIColEnd: 16, MIRowEnd: 12, X4: 12, Y4: 8, Level: BlockLevel32x32, Partition: PartitionTTopSplit, Size: BlockSize16x16, VisibleW4: 4, VisibleH4: 4, HaveTop: true, HaveLeft: true},
		{MICol: 8, MIRow: 12, MIColEnd: 16, MIRowEnd: 16, X4: 8, Y4: 12, Level: BlockLevel32x32, Partition: PartitionTTopSplit, Size: BlockSize32x16, VisibleW4: 8, VisibleH4: 4, HaveTop: true, HaveLeft: true},
	}
	if len(visits) != len(want) {
		t.Fatalf("visit count=%d want %d visits=%+v", len(visits), len(want), visits)
	}
	for i := range want {
		if visits[i] != want[i] {
			t.Fatalf("visit %d=%+v want %+v", i, visits[i], want[i])
		}
	}
	if len(reader.calls) != 5 {
		t.Fatalf("reader calls=%d want 5", len(reader.calls))
	}
	if reader.calls[0] != (scriptedPartitionCall{level: BlockLevel64x64, ctx: 0, haveRight: true, haveBottom: true}) {
		t.Fatalf("root call=%+v", reader.calls[0])
	}
	if reader.calls[1].level != BlockLevel32x32 || reader.calls[1].ctx != 0 {
		t.Fatalf("first child call=%+v", reader.calls[1])
	}
}

func TestWalkBlocksBoundaryForcedSplit(t *testing.T) {
	var cdfs PartitionCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var ctx PartitionContext
	var visits []BlockVisit
	stats, err := state.WalkBlocks(&cdfs, &ctx, BlockWalkRequest{
		Root:       BlockLevel64x64,
		MIColStart: 0,
		MIRowStart: 0,
		MIColEnd:   2,
		MIRowEnd:   2,
	}, func(visit BlockVisit) error {
		visits = append(visits, visit)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockWalkStats{PartitionReads: 4, Blocks: 1}) {
		t.Fatalf("stats=%+v want forced splits to one 8x8 block", stats)
	}
	if len(visits) != 1 || visits[0].Level != BlockLevel8x8 || visits[0].Size != BlockSize8x8 {
		t.Fatalf("visits=%+v want one 8x8 leaf", visits)
	}
	if visits[0].MIColEnd != 2 || visits[0].MIRowEnd != 2 || visits[0].VisibleW4 != 2 || visits[0].VisibleH4 != 2 {
		t.Fatalf("clipped visit=%+v", visits[0])
	}
}

func TestWalkBlocksNeighborBoundsCanSpanRootRequests(t *testing.T) {
	reader := scriptedPartitionReader{partitions: []Partition{PartitionNone}}
	var ctx PartitionContext
	var got BlockVisit
	stats, err := walkBlocks(&ctx, BlockWalkRequest{
		Root:               BlockLevel64x64,
		MIColStart:         16,
		MIRowStart:         0,
		MIColEnd:           32,
		MIRowEnd:           16,
		UseNeighborBounds:  true,
		NeighborMIColStart: 0,
		NeighborMIRowStart: 0,
	}, reader.read, func(visit BlockVisit) error {
		got = visit
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockWalkStats{PartitionReads: 1, Blocks: 1}) {
		t.Fatalf("stats=%+v", stats)
	}
	if !got.HaveLeft || got.HaveTop || got.X4 != 0 || got.Y4 != 0 {
		t.Fatalf("visit=%+v want left neighbor across root and root-local x/y", got)
	}
}

func TestWalkBlocksScriptedH4V4EdgeClipping(t *testing.T) {
	reader := scriptedPartitionReader{partitions: []Partition{PartitionH4, PartitionV4}}
	var ctx PartitionContext
	var visits []BlockVisit
	stats, err := walkBlocks(&ctx, BlockWalkRequest{
		Root:       BlockLevel32x32,
		MIColStart: 0,
		MIRowStart: 0,
		MIColEnd:   16,
		MIRowEnd:   7,
	}, reader.read, func(visit BlockVisit) error {
		visits = append(visits, visit)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if stats != (BlockWalkStats{PartitionReads: 2, Blocks: 8}) {
		t.Fatalf("stats=%+v want two roots and clipped leaves", stats)
	}
	want := []BlockVisit{
		{MICol: 0, MIRow: 0, MIColEnd: 8, MIRowEnd: 2, X4: 0, Y4: 0, Level: BlockLevel32x32, Partition: PartitionH4, Size: BlockSize32x8, VisibleW4: 8, VisibleH4: 2},
		{MICol: 0, MIRow: 2, MIColEnd: 8, MIRowEnd: 4, X4: 0, Y4: 2, Level: BlockLevel32x32, Partition: PartitionH4, Size: BlockSize32x8, VisibleW4: 8, VisibleH4: 2, HaveTop: true},
		{MICol: 0, MIRow: 4, MIColEnd: 8, MIRowEnd: 6, X4: 0, Y4: 4, Level: BlockLevel32x32, Partition: PartitionH4, Size: BlockSize32x8, VisibleW4: 8, VisibleH4: 2, HaveTop: true},
		{MICol: 0, MIRow: 6, MIColEnd: 8, MIRowEnd: 7, X4: 0, Y4: 6, Level: BlockLevel32x32, Partition: PartitionH4, Size: BlockSize32x8, VisibleW4: 8, VisibleH4: 1, HaveTop: true},
		{MICol: 8, MIRow: 0, MIColEnd: 10, MIRowEnd: 7, X4: 0, Y4: 0, Level: BlockLevel32x32, Partition: PartitionV4, Size: BlockSize8x32, VisibleW4: 2, VisibleH4: 7, HaveLeft: true},
		{MICol: 10, MIRow: 0, MIColEnd: 12, MIRowEnd: 7, X4: 2, Y4: 0, Level: BlockLevel32x32, Partition: PartitionV4, Size: BlockSize8x32, VisibleW4: 2, VisibleH4: 7, HaveLeft: true},
		{MICol: 12, MIRow: 0, MIColEnd: 14, MIRowEnd: 7, X4: 4, Y4: 0, Level: BlockLevel32x32, Partition: PartitionV4, Size: BlockSize8x32, VisibleW4: 2, VisibleH4: 7, HaveLeft: true},
		{MICol: 14, MIRow: 0, MIColEnd: 16, MIRowEnd: 7, X4: 6, Y4: 0, Level: BlockLevel32x32, Partition: PartitionV4, Size: BlockSize8x32, VisibleW4: 2, VisibleH4: 7, HaveLeft: true},
	}
	for i := range want {
		if visits[i] != want[i] {
			t.Fatalf("visit %d=%+v want %+v", i, visits[i], want[i])
		}
	}
}

func TestWalkBlocksRejectsInvalidInputs(t *testing.T) {
	var nilState *DecodeState
	errBoom := errors.New("boom")
	_, err := walkBlocks(nil, BlockWalkRequest{Root: BlockLevel64x64, MIColEnd: 16, MIRowEnd: 16}, func(BlockLevel, int, uint32, uint32, bool, bool) (Partition, error) {
		return PartitionNone, nil
	}, func(BlockVisit) error { return nil })
	if !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil context err=%v want %v", err, ErrInvalidDecodeState)
	}
	_, err = nilState.WalkBlocks(nil, &PartitionContext{}, BlockWalkRequest{Root: BlockLevel64x64, MIColEnd: 16, MIRowEnd: 16}, func(BlockVisit) error { return nil })
	if !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}
	_, err = walkBlocks(&PartitionContext{}, BlockWalkRequest{Root: BlockLevel64x64, MIColStart: 1, MIColEnd: 16, MIRowEnd: 16}, func(BlockLevel, int, uint32, uint32, bool, bool) (Partition, error) {
		return PartitionNone, nil
	}, func(BlockVisit) error { return nil })
	if !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("unaligned region err=%v want %v", err, ErrInvalidDecodeState)
	}
	_, err = walkBlocks(&PartitionContext{}, BlockWalkRequest{Root: BlockLevel64x64, MIColEnd: 16, MIRowEnd: 16}, func(BlockLevel, int, uint32, uint32, bool, bool) (Partition, error) {
		return PartitionNone, nil
	}, func(BlockVisit) error { return errBoom })
	if !errors.Is(err, errBoom) {
		t.Fatalf("visitor err=%v want %v", err, errBoom)
	}
}

func FuzzWalkBlocksScripted(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockLevel64x64), uint8(16), uint8(16))
	f.Add([]byte{0x03, 0x00, 0x01, 0x02, 0x04}, uint8(BlockLevel64x64), uint8(16), uint8(16))
	f.Add([]byte{0x08, 0x09}, uint8(BlockLevel32x32), uint8(16), uint8(8))

	f.Fuzz(func(t *testing.T, script []byte, rawRoot uint8, rawCols uint8, rawRows uint8) {
		if len(script) == 0 || len(script) > 128 {
			return
		}
		root := max(BlockLevel(rawRoot%uint8(blockLevelCount)), BlockLevel64x64)
		rootSize := uint32(root.Size4x4())
		if rootSize == 0 {
			return
		}
		cols := uint32(rawCols%uint8(rootSize*2) + 1)
		rows := uint32(rawRows%uint8(rootSize*2) + 1)
		cols = (cols + 1) &^ 1
		rows = (rows + 1) &^ 1
		if cols == 0 || rows == 0 {
			return
		}

		pos := 0
		read := func(level BlockLevel, _ int, _ uint32, _ uint32, haveRight bool, haveBottom bool) (Partition, error) {
			if !haveRight && !haveBottom {
				if level >= BlockLevel8x8 {
					return PartitionNone, nil
				}
				return PartitionSplit, nil
			}
			b := script[pos%len(script)]
			pos++
			if !haveBottom {
				if b&1 == 0 {
					return PartitionH, nil
				}
				return PartitionSplit, nil
			}
			if !haveRight {
				if b&1 == 0 {
					return PartitionV, nil
				}
				return PartitionSplit, nil
			}
			return Partition(int(b) % int(partitionSymbolCount[level])), nil
		}

		var ctx PartitionContext
		stats, err := walkBlocks(&ctx, BlockWalkRequest{
			Root:       root,
			MIColStart: 0,
			MIRowStart: 0,
			MIColEnd:   uint16(cols),
			MIRowEnd:   uint16(rows),
		}, read, func(visit BlockVisit) error {
			if visit.MICol >= visit.MIColEnd || visit.MIRow >= visit.MIRowEnd {
				t.Fatalf("empty visit=%+v", visit)
			}
			if uint32(visit.MIColEnd) > cols || uint32(visit.MIRowEnd) > rows {
				t.Fatalf("visit beyond region=%+v cols=%d rows=%d", visit, cols, rows)
			}
			if int(visit.VisibleW4) != int(visit.MIColEnd-visit.MICol) ||
				int(visit.VisibleH4) != int(visit.MIRowEnd-visit.MIRow) {
				t.Fatalf("visible mismatch visit=%+v", visit)
			}
			if !validBlockSize(visit.Size) {
				t.Fatalf("invalid block size in visit=%+v", visit)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walkBlocks err=%v", err)
		}
		if stats.Blocks == 0 || stats.PartitionReads == 0 {
			t.Fatalf("empty stats=%+v", stats)
		}
	})
}

func validBlockSize(size BlockSize) bool {
	_, ok := size.Dimensions()
	return ok
}
