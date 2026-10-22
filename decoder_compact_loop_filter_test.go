package goav1

import (
	"crypto/md5"
	"os"
	"path/filepath"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/lfmask"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestDecoderCompactLoopFilterSelectionAndWorkerFallback(t *testing.T) {
	path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", "profile0-420-8bit-sframe-64x64.ivf")
	ivf, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	decodeFirst := func(workers int) (*Decoder, [md5.Size]byte) {
		t.Helper()
		dec, err := NewDecoderFromIVF(ivf, WithWorkers(workers), WithMaxFrameDelay(1))
		if err != nil {
			t.Fatalf("NewDecoderFromIVF workers=%d: %v", workers, err)
		}
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			dec.Close()
			t.Fatalf("DecodeNext workers=%d: %v", workers, err)
		}
		if !ok || len(frames) != 1 {
			dec.Close()
			t.Fatalf("DecodeNext workers=%d returned ok=%v frames=%d", workers, ok, len(frames))
		}
		return dec, visibleFrameMD5(frames[0])
	}

	serial, serialMD5 := decodeFirst(1)
	defer serial.Close()
	if serial.useExternal {
		t.Fatal("fixture unexpectedly selected an external output route")
	}
	if serial.postFilter.runner.Result.LoopFilter.Plan.Active == false || serial.postFilter.runner.Result.LoopFilter.Plan.Cells == 0 {
		t.Fatal("fixture did not exercise active loop filtering")
	}
	if len(serial.batch.LoopFilterMap.Records) != 0 {
		t.Fatalf("eligible private one-worker route populated %d legacy loop-filter records", len(serial.batch.LoopFilterMap.Records))
	}

	parallel, parallelMD5 := decodeFirst(2)
	defer parallel.Close()
	if parallel.postFilter.runner.Result.LoopFilter.Plan.Active == false || parallel.postFilter.runner.Result.LoopFilter.Plan.Cells == 0 {
		t.Fatal("two-worker fallback did not exercise active loop filtering")
	}
	if len(parallel.batch.LoopFilterMap.Records) == 0 {
		t.Fatal("two-worker fallback omitted the legacy loop-filter record map")
	}
	if parallelMD5 != serialMD5 {
		t.Fatalf("worker fallback changed visible output: one-worker=%x two-worker=%x", serialMD5, parallelMD5)
	}
}

func TestDecoderCompactLoopFilterFallsBackForMultiTileAndSuperRes(t *testing.T) {
	for _, tc := range []struct {
		name         string
		file         string
		wantExternal bool
	}{
		{name: "multi-tile", file: "multitile-2x1-rows-256x256.ivf"},
		{name: "super-res", file: "superres-420-8bit-160x128.ivf", wantExternal: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", tc.file)
			ivf, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			dec, err := NewDecoderFromIVF(ivf)
			if err != nil {
				t.Fatal(err)
			}
			defer dec.Close()
			frames, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("DecodeNext: %v", err)
			}
			if !ok || len(frames) != 1 {
				t.Fatalf("DecodeNext returned ok=%v frames=%d", ok, len(frames))
			}
			if dec.useExternal != tc.wantExternal {
				t.Fatalf("external output route=%v want %v", dec.useExternal, tc.wantExternal)
			}
			var loopFilterActive bool
			if dec.useExternal {
				loopFilterActive = dec.external.supported.runner.Result.LoopFilter.Plan.Active
			} else {
				loopFilterActive = dec.postFilter.runner.Result.LoopFilter.Plan.Active
			}
			if !loopFilterActive && !tc.wantExternal {
				t.Fatal("fixture did not exercise active loop filtering")
			}
			if loopFilterActive {
				if len(dec.batch.LoopFilterMap.Records) == 0 {
					t.Fatal("ineligible route omitted the legacy loop-filter record map")
				}
				t.Logf("fallback verified: external=%v active loop filter, legacy records=%d", dec.useExternal, len(dec.batch.LoopFilterMap.Records))
			} else {
				t.Logf("super-res fixture has no active loop filter; owner predicate test covers fallback gating")
			}
			if internalthreading.FrameWorkLoopFilterCompactStateActive(&dec.compactLF) {
				t.Fatal("ineligible route left compact state active")
			}
		})
	}
}

func TestDecoderCompactLoopFilterOwnerRejectsCopiesAndHooks(t *testing.T) {
	side, compact := compactRootSideData(t)
	batch := &DecoderFrameWorkBatchResidualRunner{}
	batch.compactState = compact
	batch.compactOwner = batch
	batch.compactSideOwner = side
	if err := batch.setOwnedCompactSideData(side); err != nil {
		t.Fatal(err)
	}
	post := &DecoderFrameWorkReusableSupportedPostFilterRunner{}
	runner := DecoderFrameWorkResidualEventRunner{
		Workers:           1,
		BatchRunner:       batch,
		SideData:          side,
		sideRunner:        decoderFrameWorkResidualSideDataRunner{SideData: side, BatchRunner: batch},
		compactState:      compact,
		compactNoSkip:     make([]byte, 8),
		compactPostOwner:  post,
		compactBatchOwner: batch,
		compactSideOwner:  side,
	}
	event := &DecoderEvent{
		Kind:       DecoderEventFrameHeader,
		FrameSize:  FrameSize{CodedWidth: 16, Height: 16},
		TileInfo:   TileInfo{Cols: 1, Rows: 1},
		LoopFilter: LoopFilterParams{LevelY: [2]uint8{1, 1}},
	}
	sequence := SequenceHeader{ColorConfig: ColorConfig{BitDepth: 8, MonoChrome: true}}
	if !runner.ownsCompactLoopFilter(sequence, post, event) {
		t.Fatal("exact owned one-tile route was not accepted")
	}

	copiedSide := *side
	runner.SideData = &copiedSide
	if runner.ownsCompactLoopFilter(sequence, post, event) {
		t.Fatal("copied side-data handle inherited compact ownership")
	}
	runner.SideData = side
	if runner.ownsCompactLoopFilter(sequence, &DecoderFrameWorkReusableCallerPostFilterRunner{}, event) {
		t.Fatal("custom/caller postfilter inherited compact ownership")
	}
	customPost := compactLoopFilterTestPostFilter{}
	if runner.ownsCompactLoopFilter(sequence, customPost, event) {
		t.Fatal("custom postfilter hook inherited compact ownership")
	}
	runner.Workers = 2
	if runner.ownsCompactLoopFilter(sequence, post, event) {
		t.Fatal("multi-worker route inherited compact ownership")
	}
	runner.Workers = 1
	multiTile := *event
	multiTile.Kind = DecoderEventTileGroup
	multiTile.TileGroup = TileGroup{TileCount: 2, Final: true}
	if runner.ownsCompactLoopFilter(sequence, post, &multiTile) {
		t.Fatal("multi-tile route inherited compact ownership")
	}
	superRes := *event
	superRes.FrameSize.SuperResEnabled = true
	if runner.ownsCompactLoopFilter(sequence, post, &superRes) {
		t.Fatal("super-res route inherited compact ownership")
	}
	hbdSequence := sequence
	hbdSequence.ColorConfig.BitDepth = 10
	if runner.ownsCompactLoopFilter(hbdSequence, post, event) {
		t.Fatal("high-bit-depth route inherited compact ownership")
	}
}

func TestDecoderCompactLoopFilterRunErrorInvalidatesAndPublicRebindDeactivates(t *testing.T) {
	side, compact := compactRootSideData(t)
	batch := &DecoderFrameWorkBatchResidualRunner{}
	batch.compactState = compact
	batch.compactOwner = batch
	batch.compactSideOwner = side
	if err := batch.setOwnedCompactSideData(side); err != nil {
		t.Fatal(err)
	}
	if err := batch.RunPtr(nil); err == nil {
		t.Fatal("RunPtr accepted a missing batch")
	}
	if !internalthreading.FrameWorkLoopFilterCompactStateActive(compact) ||
		internalthreading.FrameWorkLoopFilterCompactReady(&side.LoopFilterMasks, compact, 4, 4) {
		t.Fatal("failed owned RunPtr did not leave active state invalidated and fail-closed")
	}
	if err := batch.SetSideData(*side); err != nil {
		t.Fatal(err)
	}
	if internalthreading.FrameWorkLoopFilterCompactStateActive(compact) || batch.compactBoundSide != nil {
		t.Fatal("successful public SetSideData retained private compact authority")
	}
}

func TestDecoderCompactLoopFilterWarmResetAllocations(t *testing.T) {
	path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", "profile0-420-8bit-sframe-64x64.ivf")
	ivf, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := NewDecoderFromIVF(ivf, WithWorkers(1))
	if err != nil {
		t.Fatalf("NewDecoderFromIVF: %v", err)
	}
	defer dec.Close()
	decodeAll := func() {
		if err := dec.Reset(); err != nil {
			t.Fatalf("Reset: %v", err)
		}
		visible := 0
		for {
			frames, ok, err := dec.DecodeNext()
			if err != nil {
				t.Fatalf("DecodeNext: %v", err)
			}
			if !ok {
				break
			}
			visible += len(frames)
		}
		if visible != 8 {
			t.Fatalf("decoded %d visible frames, want 8", visible)
		}
	}
	decodeAll()
	allocs := testing.AllocsPerRun(20, decodeAll)
	t.Logf("one-worker compact-route warm DecodeNext+Reset allocs/run=%f", allocs)
	if allocs != 0 {
		t.Fatalf("warm compact DecodeNext+Reset allocated %f objects/run, want 0", allocs)
	}
}

func compactRootSideData(t *testing.T) (*DecoderFrameWorkSideData, *internalthreading.FrameWorkLoopFilterCompactState) {
	t.Helper()
	side := &DecoderFrameWorkSideData{
		CDEFIndexMap: DecoderFrameWorkCDEFIndexMap{Index: make([]uint8, 1), Read: make([]bool, 1), Stride: 1, Rows: 1},
		LoopFilterMap: DecoderFrameWorkLoopFilterMap{
			Records: make([]DecoderFrameWorkLoopFilterBlockRecord, 16),
			Stride:  4,
			Rows:    4,
		},
		LoopFilterMasks: DecoderFrameWorkLoopFilterMasks{
			Masks:      make([]DecoderFrameWorkLoopFilterFilterMask, 1),
			LevelCache: make([][4]uint8, 16),
			Cols:       4,
			Rows:       4,
			SB128W:     1,
			SB128H:     1,
			Layout:     lfmask.Layout{Mono: true},
		},
		RestorationFrameBuffers: DecoderFrameWorkRestorationFrameBuffers{Plan: tile.RestorationFramePlan{Planes: 1}},
	}
	compact := &internalthreading.FrameWorkLoopFilterCompactState{}
	if err := internalthreading.BindFrameWorkLoopFilterCompactState(&side.LoopFilterMasks, compact, make([]byte, 8), LoopFilterParams{LevelY: [2]uint8{1, 1}}, SegmentationParams{}, DeltaParams{}, true); err != nil {
		t.Fatal(err)
	}
	return side, compact
}

type compactLoopFilterTestPostFilter struct{}

func (compactLoopFilterTestPostFilter) Apply(DecoderFrameWorkPostFilterContext) error { return nil }

func visibleFrameMD5(f *Frame) [md5.Size]byte {
	h := md5.New()
	bps := f.Layout.BytesPerSample
	for _, plane := range []FramePlane{f.Y, f.U, f.V} {
		if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
			continue
		}
		rowBytes := plane.Width * bps
		for row := 0; row < plane.Height; row++ {
			off := row * plane.Stride
			_, _ = h.Write(plane.Pix[off : off+rowBytes])
		}
	}
	var sum [md5.Size]byte
	copy(sum[:], h.Sum(nil))
	return sum
}
