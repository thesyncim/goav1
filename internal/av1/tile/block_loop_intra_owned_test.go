package tile

import (
	"errors"
	"math/rand"
	"reflect"
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// The full root load/store, MarkIntra and resetForSuperblock are unchanged from
// f05383c85c62ba39d0cd0055e66379f6fa0f839c and are the independent old-state oracle.
// Compare only the documented private intra live fields; separately assert that
// omitted payload/validity fields remain poisoned, proving the work is omitted.
var ownedIntraLiveFields = []string{
	"AboveSkip", "LeftSkip", "AboveSkipMode", "LeftSkipMode", "AboveSegmentPred", "LeftSegmentPred",
	"AboveIntra", "LeftIntra", "AboveMode", "LeftMode", "AboveChromaIntra", "LeftChromaIntra", "AboveChromaMode", "LeftChromaMode",
	"AboveTxIntra", "LeftTxIntra", "AboveTx", "LeftTx", "AboveBlockSize", "LeftBlockSize",
	"TxNeighborValid", "TxAboveNeighborIntra", "TxAboveNeighborBlockSize", "TxLeftNeighborIntra", "TxLeftNeighborBlockSize",
	"AbovePaletteY", "LeftPaletteY", "AbovePaletteUV", "LeftPaletteUV",
}

func ownedIntraCompareLive(t *testing.T, got, want *BlockLoopScratch) {
	t.Helper()
	g, w := reflect.ValueOf(&got.Mode).Elem(), reflect.ValueOf(&want.Mode).Elem()
	for _, name := range ownedIntraLiveFields {
		if !reflect.DeepEqual(g.FieldByName(name).Interface(), w.FieldByName(name).Interface()) {
			t.Fatalf("live field %s differs", name)
		}
	}
	if got.Partition != want.Partition || got.CoeffCtx != want.CoeffCtx || got.CDEF != want.CDEF {
		t.Fatal("partition/coefficient/CDEF context differs")
	}
}

// Use fixed byte poison only on pointer-free mode data, then normalize bools.
func ownedIntraPoisonMode(c *BlockModeContext, seed byte) {
	b := unsafe.Slice((*byte)(unsafe.Pointer(c)), unsafe.Sizeof(*c))
	for i := range b {
		b[i] = seed + byte(i*37)
	}
	var normalize func(reflect.Value)
	normalize = func(v reflect.Value) {
		switch v.Kind() {
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				normalize(v.Field(i))
			}
		case reflect.Array:
			for i := 0; i < v.Len(); i++ {
				normalize(v.Index(i))
			}
		case reflect.Bool:
			if v.CanSet() {
				v.SetBool(true)
			}
		}
	}
	normalize(reflect.ValueOf(c).Elem())
}

func TestOwnedIntraRootContextsMatchFullOracle(t *testing.T) {
	for _, sb := range []uint8{16, 32} {
		for _, columns := range []int{1, 2, 4} {
			var seed BlockLoopScratch
			ownedIntraPoisonMode(&seed.Mode, 51)
			for i := range seed.Partition.Above {
				seed.Partition.Above[i] = uint8(i)
				seed.Partition.Left[i] = uint8(i + 1)
			}
			var carrier BlockLoopContextCarrier
			carrier.Above = make([]BlockLoopRootAboveContext, columns)
			PreallocBlockLoopContextCarrierScratch(&carrier, columns)
			for col := range columns {
				if err := blockLoopStoreRootContext(&seed, &carrier, col, sb); err != nil {
					t.Fatal(err)
				}
				captureDiagonalCornerToPending(&carrier, col, &seed.Mode, sb)
			}
			promotePendingDiagonalCarriers(&carrier)
			for col := range columns {
				for _, top := range []bool{false, true} {
					for _, left := range []bool{false, true} {
						got, want := seed, seed
						if err := blockLoopLoadRootContext(&want, &carrier, col, top, left, sb); err != nil {
							t.Fatal(err)
						}
						if err := testLoadIntraRootContext(&got, &carrier, col, top, left, sb); err != nil {
							t.Fatal(err)
						}
						ownedIntraCompareLive(t, &got, &want)
						if got.Mode.GridMotionValid != seed.Mode.GridMotionValid || got.Mode.SBTopInterMotionGrid != seed.Mode.SBTopInterMotionGrid {
							t.Fatal("private root load touched omitted motion state")
						}
						fast := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, columns)}
						full := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, columns)}
						if err := testStoreIntraRootContext(&got, &fast, col); err != nil {
							t.Fatal(err)
						}
						if err := blockLoopStoreRootContext(&want, &full, col, sb); err != nil {
							t.Fatal(err)
						}
						var reloadedGot, reloadedWant BlockLoopScratch
						if err := testLoadIntraRootContext(&reloadedGot, &fast, col, true, true, sb); err != nil {
							t.Fatal(err)
						}
						if err := blockLoopLoadRootContext(&reloadedWant, &full, col, true, true, sb); err != nil {
							t.Fatal(err)
						}
						ownedIntraCompareLive(t, &reloadedGot, &reloadedWant)
					}
				}
			}
		}
	}
}

func TestOwnedIntraMarkMatchesFullOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(7711))
	for size := BlockSize(0); size < blockSizeCount; size++ {
		dims, ok := size.Dimensions()
		if !ok {
			t.Fatal(size)
		}
		for trial := 0; trial < 16; trial++ {
			var got BlockLoopScratch
			ownedIntraPoisonMode(&got.Mode, byte(trial+33))
			want := got
			before := got.Mode
			x, y := rng.Intn(MaxBlockModeSlots-int(dims.W4)+1), rng.Intn(MaxBlockModeSlots-int(dims.H4)+1)
			mode := IntraMode(rng.Intn(int(intraModeCount)))
			if err := want.Mode.MarkIntra(size, x, y, true, mode); err != nil {
				t.Fatal(err)
			}
			if err := got.Mode.markOwnedIntra(size, x, y, mode); err != nil {
				t.Fatal(err)
			}
			ownedIntraCompareLive(t, &got, &want)
			if got.Mode.GridMotionValid != before.GridMotionValid || got.Mode.GridBlockSize != before.GridBlockSize || got.Mode.GridBlockSizeVisited != before.GridBlockSizeVisited || got.Mode.AboveRef != before.AboveRef || got.Mode.LeftCompound != before.LeftCompound {
				t.Fatal("private mark touched omitted motion bookkeeping")
			}
		}
	}
}

func TestOwnedIntraErrorsPreserveLiveState(t *testing.T) {
	for _, tc := range []struct {
		size BlockSize
		x, y int
		mode IntraMode
	}{{blockSizeCount, 0, 0, IntraModeDC}, {BlockSize8x8, -1, 0, IntraModeDC}, {BlockSize8x8, 31, 0, IntraModeDC}, {BlockSize8x8, 0, 31, IntraModeDC}, {BlockSize8x8, 0, 0, intraModeCount}} {
		var got BlockModeContext
		ownedIntraPoisonMode(&got, 17)
		want := got
		ge, we := got.markOwnedIntra(tc.size, tc.x, tc.y, tc.mode), want.MarkIntra(tc.size, tc.x, tc.y, true, tc.mode)
		if !errors.Is(ge, ErrInvalidDecodeState) || !errors.Is(we, ErrInvalidDecodeState) || !reflect.DeepEqual(got, want) {
			t.Fatalf("invalid mark changed state: %+v", tc)
		}
	}
	for _, col := range []int{-1, 1} {
		carrier := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, 1)}
		var got BlockLoopScratch
		ownedIntraPoisonMode(&got.Mode, 29)
		want := got
		ge := testLoadIntraRootContext(&got, &carrier, col, true, true, 16)
		we := blockLoopLoadRootContext(&want, &carrier, col, true, true, 16)
		if !errors.Is(ge, ErrInvalidDecodeState) || !errors.Is(we, ErrInvalidDecodeState) {
			t.Fatal("missing invalid carrier error")
		}
		ownedIntraCompareLive(t, &got, &want)
		before := carrier.Above[0]
		if !errors.Is(testStoreIntraRootContext(&got, &carrier, col), ErrInvalidDecodeState) || carrier.Above[0] != before {
			t.Fatal("invalid store mutated carrier")
		}
	}
}

func TestOwnedIntraThenGenericCarrierTransition(t *testing.T) {
	// Mimic the decoder's per-job Bind: Above/Left are reset, diagonal backing is
	// retained. A following generic inter or IBC job must overwrite every consumed
	// history before reading it, even after a prior private walk failed mid-row.
	for _, sb := range []uint8{16, 32} {
		for _, stop := range []int{1, 2, 5, 6} {
			const cols = 3
			var fast, full BlockLoopScratch
			ownedIntraPoisonMode(&fast.Mode, 71)
			full = fast
			fc := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, cols)}
			wc := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, cols)}
			PreallocBlockLoopContextCarrierScratch(&fc, cols)
			PreallocBlockLoopContextCarrierScratch(&wc, cols)
			// Poison previous generic-job diagonal payload equally.
			for col := range cols {
				captureDiagonalCornerToPending(&fc, col, &fast.Mode, sb)
				captureDiagonalCornerToPending(&wc, col, &full.Mode, sb)
			}
			for index := 0; index < stop; index++ {
				row, col := index/cols, index%cols
				if col == 0 {
					promotePendingDiagonalCarriers(&wc)
				}
				if err := testLoadIntraRootContext(&fast, &fc, col, row > 0, col > 0, sb); err != nil {
					t.Fatal(err)
				}
				if err := blockLoopLoadRootContext(&full, &wc, col, row > 0, col > 0, sb); err != nil {
					t.Fatal(err)
				}
				size := BlockSize64x64
				if sb == 32 {
					size = BlockSize128x128
				}
				if err := fast.Mode.markOwnedIntra(size, 0, 0, IntraModeDC); err != nil {
					t.Fatal(err)
				}
				if err := full.Mode.MarkIntra(size, 0, 0, true, IntraModeDC); err != nil {
					t.Fatal(err)
				}
				if err := testStoreIntraRootContext(&fast, &fc, col); err != nil {
					t.Fatal(err)
				}
				if err := blockLoopStoreRootContext(&full, &wc, col, sb); err != nil {
					t.Fatal(err)
				}
				captureDiagonalCornerToPending(&wc, col+1, &full.Mode, sb)
			}
			clear(fc.Above)
			clear(wc.Above)
			fc.Left = BlockLoopRootLeftContext{}
			wc.Left = BlockLoopRootLeftContext{}
			for index := 0; index < 9; index++ {
				row, col := index/cols, index%cols
				if col == 0 {
					promotePendingDiagonalCarriers(&fc)
					promotePendingDiagonalCarriers(&wc)
				}
				if err := blockLoopLoadRootContext(&fast, &fc, col, row > 0, col > 0, sb); err != nil {
					t.Fatal(err)
				}
				if err := blockLoopLoadRootContext(&full, &wc, col, row > 0, col > 0, sb); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(fast.Mode, full.Mode) {
					t.Fatalf("generic state differs after private stop=%d sb=%d root=%d", stop, sb, index)
				}
				result := InterMotionResult{MV: [2]motion.Vector{{Row: int16(index + 1), Col: int16(-index - 2)}}, References: InterReferencesResult{Ref: [2]ReferenceFrame{ReferenceFrameLast, ReferenceFrameNone}}}
				mark := (*BlockModeContext).MarkInterMotion
				if index&1 != 0 {
					mark = (*BlockModeContext).MarkIntrabcMotion
				}
				if err := mark(&fast.Mode, BlockSize16x16, 0, 0, result, false); err != nil {
					t.Fatal(err)
				}
				if err := mark(&full.Mode, BlockSize16x16, 0, 0, result, false); err != nil {
					t.Fatal(err)
				}
				if err := blockLoopStoreRootContext(&fast, &fc, col, sb); err != nil {
					t.Fatal(err)
				}
				if err := blockLoopStoreRootContext(&full, &wc, col, sb); err != nil {
					t.Fatal(err)
				}
				captureDiagonalCornerToPending(&fc, col+1, &fast.Mode, sb)
				captureDiagonalCornerToPending(&wc, col+1, &full.Mode, sb)
			}
		}
	}
}

func TestOwnedIntraEntryRejectsInterAndIBC(t *testing.T) {
	for _, ft := range []parser.FrameType{parser.FrameTypeInter, parser.FrameTypeSwitch, parser.FrameType(9)} {
		if _, err := DecodeBlockLoopWithCoeffControllerPtrOwnedIntra(nil, BlockLoopCDFs{}, nil, BlockLoopRequest{FrameType: ft, DecodePredictionModes: true}, noBlockLoopCoeffController{}, nil); !errors.Is(err, ErrInvalidDecodeState) {
			t.Fatal(ft, err)
		}
	}
	if _, err := DecodeBlockLoopWithCoeffControllerPtrOwnedIntra(nil, BlockLoopCDFs{}, nil, BlockLoopRequest{FrameType: parser.FrameTypeKey, AllowIntrabc: true, DecodePredictionModes: true}, noBlockLoopCoeffController{}, nil); !errors.Is(err, ErrInvalidDecodeState) {
		t.Fatal(err)
	}
}

func TestOwnedIntraContextsNoAlloc(t *testing.T) {
	var scratch BlockLoopScratch
	carrier := BlockLoopContextCarrier{Above: make([]BlockLoopRootAboveContext, 2)}
	if n := testing.AllocsPerRun(100, func() {
		if err := testLoadIntraRootContext(&scratch, &carrier, 0, true, true, 16); err != nil {
			panic(err)
		}
		if err := scratch.Mode.markOwnedIntra(BlockSize16x16, 0, 0, IntraModeDC); err != nil {
			panic(err)
		}
		if err := testStoreIntraRootContext(&scratch, &carrier, 0); err != nil {
			panic(err)
		}
	}); n != 0 {
		t.Fatalf("allocs=%g", n)
	}
	t.Logf("mode reset bytes: generic=%d owned=%d; five-row history bytes per edge=%d", unsafe.Offsetof(scratch.Mode.GridInterMotion), unsafe.Offsetof(scratch.Mode.AboveRef)+unsafe.Offsetof(scratch.Mode.LeftRef)-unsafe.Offsetof(scratch.Mode.LeftSkip), unsafe.Sizeof(carrier.Left.mode.InterMotionHistory)+unsafe.Sizeof(carrier.Left.mode.MotionValidHistory)+unsafe.Sizeof(carrier.Left.mode.BlockSizeHistory)+unsafe.Sizeof(carrier.Left.mode.BlockSizeVisitedHistory))
}

// testLoadIntraRootContext loads one root in isolation: the position never
// continues a resident left edge, so the left edge comes from carrier.Left.
func testLoadIntraRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int, haveTop bool, haveLeft bool, sbSizeMIB uint8) error {
	return blockLoopLoadIntraRootContext(scratch, carrier, rootColIndex, haveTop, haveLeft, sbSizeMIB, ^uint32(0), ^uint32(0))
}

// testStoreIntraRootContext stores one root and flushes its left edge to
// carrier.Left, the per-root store the isolated tests compare against.
func testStoreIntraRootContext(scratch *BlockLoopScratch, carrier *BlockLoopContextCarrier, rootColIndex int) error {
	if err := blockLoopStoreIntraRootContext(scratch, carrier, rootColIndex, 0, 0); err != nil {
		return err
	}
	scratch.flushLeftEdge()
	return nil
}
