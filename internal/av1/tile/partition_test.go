package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

func TestPartitionCDFsInitDefaultMatchesDav1d(t *testing.T) {
	var cdfs PartitionCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name  string
		level BlockLevel
		ctx   int
		want  []uint16
	}{
		{
			name:  "128 ctx0",
			level: BlockLevel128x128,
			ctx:   0,
			want:  []uint16{4869, 4549, 4239, 284, 229, 149, 129, 0, 0},
		},
		{
			name:  "64 ctx0",
			level: BlockLevel64x64,
			ctx:   0,
			want:  []uint16{12631, 11221, 9690, 3202, 2931, 2507, 2244, 1876, 1044, 0, 0},
		},
		{
			name:  "8 ctx3",
			level: BlockLevel8x8,
			ctx:   3,
			want:  []uint16{22872, 13985, 6915, 0, 0},
		},
	}
	for _, tt := range tests {
		cdf, err := cdfs.CDF(tt.level, tt.ctx)
		if err != nil {
			t.Fatalf("%s CDF: %v", tt.name, err)
		}
		if cdf.Symbols() != partitionSymbolCount[tt.level] {
			t.Fatalf("%s symbols=%d want %d", tt.name, cdf.Symbols(), partitionSymbolCount[tt.level])
		}
		assertEntropyCDFValues(t, cdf.Values(), tt.want)
	}
}

func TestPartitionLevelAndBlockSizeTablesMatchDav1d(t *testing.T) {
	if RootBlockLevel(false) != BlockLevel64x64 || RootBlockLevel(true) != BlockLevel128x128 {
		t.Fatal("root block level mismatch")
	}
	if BlockLevel128x128.HalfSize4x4() != 16 || BlockLevel64x64.HalfSize4x4() != 8 || BlockLevel8x8.HalfSize4x4() != 1 {
		t.Fatal("half-size table mismatch")
	}

	size, second, ok := PartitionTLeftSplit.BlockSizes(BlockLevel32x32)
	if !ok || size != BlockSize16x16 || second != BlockSize16x32 {
		t.Fatalf("T-left sizes=%d,%d ok=%v", size, second, ok)
	}
	size, second, ok = PartitionSplit.BlockSizes(BlockLevel8x8)
	if !ok || size != BlockSize4x4 || second != 0 {
		t.Fatalf("8x8 split sizes=%d,%d ok=%v", size, second, ok)
	}
	if _, _, ok := PartitionSplit.BlockSizes(BlockLevel64x64); ok {
		t.Fatal("recursive split reported a direct block size")
	}
	dims, ok := BlockSize64x16.Dimensions()
	if !ok || dims != (BlockDimensions{W4: 16, H4: 4, Log2W: 4, Log2H: 2}) {
		t.Fatalf("dims=%+v ok=%v", dims, ok)
	}
}

func TestReadPartitionFullAndBoundary(t *testing.T) {
	var cdfs PartitionCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00, 0x00}, Job{Offset: 0, Size: 2}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partition, err := state.ReadPartition(&cdfs, BlockLevel8x8, 0, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if partition != PartitionNone {
		t.Fatalf("full partition=%d want %d", partition, PartitionNone)
	}
	cdf, err := cdfs.CDF(BlockLevel8x8, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := cdf.Values()[cdf.Symbols()]; got != 1 {
		t.Fatalf("adapt count=%d want 1", got)
	}

	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	partition, err = state.ReadPartition(&cdfs, BlockLevel64x64, 0, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if partition != PartitionH {
		t.Fatalf("top-edge partition=%d want H", partition)
	}
	cdf, err = cdfs.CDF(BlockLevel64x64, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got := cdf.Values()[cdf.Symbols()]; got != 0 {
		t.Fatalf("edge bool adapted CDF count=%d want 0", got)
	}
}

func TestPartitionContextMatchesDav1d(t *testing.T) {
	var ctx PartitionContext
	got, err := ctx.Context(BlockLevel64x64, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("initial ctx=%d want 0", got)
	}

	if err := ctx.Mark(BlockLevel64x64, PartitionH, 0, 0); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < int(BlockLevel64x64.HalfSize4x4()); i++ {
		if ctx.Above[i] != 0x10 || ctx.Left[i] != 0x18 {
			t.Fatalf("slot %d above=%#x left=%#x", i, ctx.Above[i], ctx.Left[i])
		}
	}
	got, err = ctx.Context(BlockLevel64x64, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got != 2 {
		t.Fatalf("ctx=%d want 2", got)
	}

	before := ctx
	if err := ctx.Mark(BlockLevel64x64, PartitionSplit, 0, 0); err != nil {
		t.Fatal(err)
	}
	if ctx != before {
		t.Fatal("recursive split should not update parent context")
	}
	if err := ctx.Mark(BlockLevel8x8, PartitionSplit, 4, 5); err != nil {
		t.Fatal(err)
	}
	if ctx.Above[4] != 0x1f || ctx.Left[5] != 0x1f {
		t.Fatalf("8x8 split above=%#x left=%#x", ctx.Above[4], ctx.Left[5])
	}
}

func TestPartitionRejectsInvalidInputs(t *testing.T) {
	var cdfs PartitionCDFs
	if _, err := cdfs.CDF(BlockLevel64x64, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("uninitialized CDF err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if _, err := cdfs.CDF(blockLevelCount, 0); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad level CDF err=%v want %v", err, entropy.ErrInvalidCDF)
	}
	if _, err := cdfs.CDF(BlockLevel64x64, PartitionContexts); !errors.Is(err, entropy.ErrInvalidCDF) {
		t.Fatalf("bad ctx CDF err=%v want %v", err, entropy.ErrInvalidCDF)
	}

	var state DecodeState
	if err := state.Reset([]byte{0x00}, Job{Offset: 0, Size: 1}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := state.ReadPartition(&cdfs, BlockLevel8x8, 0, false, false); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad forced split err=%v want %v", err, ErrInvalidDecodeState)
	}
	if _, err := state.ReadPartition(&cdfs, BlockLevel8x8, 0, false, true); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad 8x8 edge err=%v want %v", err, ErrInvalidDecodeState)
	}
	partition, err := state.ReadPartition(nil, BlockLevel64x64, 0, false, false)
	if err != nil || partition != PartitionSplit {
		t.Fatalf("forced split partition=%d err=%v", partition, err)
	}
	var nilState *DecodeState
	if _, err := nilState.ReadPartition(&cdfs, BlockLevel64x64, 0, true, true); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("nil state err=%v want %v", err, ErrInvalidDecodeState)
	}

	var ctx PartitionContext
	if _, err := ctx.Context(blockLevelCount, 0, 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad context err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := ctx.Mark(BlockLevel128x128, PartitionH4, 0, 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mark err=%v want %v", err, ErrInvalidDecodeState)
	}
	if err := ctx.Mark(BlockLevel64x64, PartitionH, MaxPartitionSlots, 0); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatalf("bad mark range err=%v want %v", err, ErrInvalidDecodeState)
	}
}

func TestReadPartitionAllocs(t *testing.T) {
	var cdfs PartitionCDFs
	if err := cdfs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	payload := []byte{0x00}
	job := Job{Offset: 0, Size: 1}
	var state DecodeState

	allocs := testing.AllocsPerRun(1000, func() {
		if err := state.Reset(payload, job, DecodeOptions{DisableCDFUpdate: true}); err != nil {
			t.Fatal(err)
		}
		if _, err := state.ReadPartition(&cdfs, BlockLevel8x8, 0, true, true); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ReadPartition allocated: %f", allocs)
	}
}

func TestPartitionCDFsInitDefaultAllocs(t *testing.T) {
	var cdfs PartitionCDFs
	allocs := testing.AllocsPerRun(1000, func() {
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("PartitionCDFs.InitDefault allocated: %f", allocs)
	}
}

func FuzzReadPartition(f *testing.F) {
	f.Add([]byte{0x00}, uint8(BlockLevel64x64), uint8(0), true, true)
	f.Add([]byte{0xff}, uint8(BlockLevel64x64), uint8(1), true, false)
	f.Add([]byte{0xaa, 0x55}, uint8(BlockLevel16x16), uint8(3), false, true)

	f.Fuzz(func(t *testing.T, payload []byte, rawLevel uint8, rawCtx uint8, haveH bool, haveV bool) {
		if len(payload) == 0 || len(payload) > 64 {
			return
		}
		level := BlockLevel(rawLevel % uint8(blockLevelCount))
		ctx := int(rawCtx % PartitionContexts)
		if !haveH && !haveV && level == BlockLevel8x8 {
			haveH = true
		}

		var cdfs PartitionCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		partition, err := state.ReadPartition(&cdfs, level, ctx, haveH, haveV)
		if err != nil {
			if level == BlockLevel8x8 && (!haveH || !haveV) {
				return
			}
			t.Fatalf("ReadPartition err=%v", err)
		}
		if !partition.ValidForLevel(level) {
			t.Fatalf("partition=%d invalid for level=%d", partition, level)
		}
	})
}
