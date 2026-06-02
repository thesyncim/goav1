package threading

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/tile"
	decodework "github.com/thesyncim/goav1/internal/av1/work"
)

func TestPoolExecute(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}

	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	var seen [4]uint16
	err = pool.Execute(batches[:n], jobs[:], func(batch Batch, batchJobs []tile.Job) error {
		if len(batchJobs) != int(batch.Count) {
			t.Fatalf("jobs len=%d count=%d", len(batchJobs), batch.Count)
		}
		for i := range batchJobs {
			seen[int(batch.FirstJob)+i] = batchJobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range len(jobs) {
		if seen[i] != jobs[i].Tile+1 {
			t.Fatalf("seen[%d]=%d job=%+v", i, seen[i], jobs[i])
		}
	}
}

func TestPoolExecuteFrameWork(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}

	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	var output frame.Frame
	var reference frame.Frame
	payload := []byte{0xaa, 0xbb}
	base := FrameWorkBatch{
		Step:       decodework.FrameStep{Kind: decodework.FrameStepTile},
		Output:     &output,
		Payload:    payload,
		References: []*frame.Frame{&reference},
	}
	var seen [4]uint16
	err = pool.ExecuteFrameWork(batches[:n], jobs[:], base, func(ctx FrameWorkBatch) error {
		if ctx.Step.Kind != decodework.FrameStepTile {
			t.Fatalf("step=%+v", ctx.Step)
		}
		if ctx.Output != &output || len(ctx.References) != 1 || ctx.References[0] != &reference {
			t.Fatalf("ctx=%+v", ctx)
		}
		if len(ctx.Payload) != len(payload) || ctx.Payload[0] != payload[0] || ctx.Payload[1] != payload[1] {
			t.Fatalf("payload=%v want %v", ctx.Payload, payload)
		}
		if len(ctx.Jobs) != int(ctx.Batch.Count) {
			t.Fatalf("jobs len=%d count=%d", len(ctx.Jobs), ctx.Batch.Count)
		}
		for i := 0; i < len(ctx.Jobs); i++ {
			seen[int(ctx.Batch.FirstJob)+i] = ctx.Jobs[i].Tile + 1
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range len(jobs) {
		if seen[i] != jobs[i].Tile+1 {
			t.Fatalf("seen[%d]=%d job=%+v", i, seen[i], jobs[i])
		}
	}
}

func TestFrameWorkBatchSurface(t *testing.T) {
	tests := []struct {
		name string
		ctx  FrameWorkBatch
		want int
	}{
		{
			name: "begin",
			ctx: FrameWorkBatch{
				Step: decodework.FrameStep{
					Kind:  decodework.FrameStepBegin,
					Begin: decodework.FramePlan{Surface: 3},
				},
			},
			want: 3,
		},
		{
			name: "tile",
			ctx: FrameWorkBatch{
				Step: decodework.FrameStep{
					Kind: decodework.FrameStepTile,
					Tile: decodework.FrameTilePlan{Surface: 5},
				},
			},
			want: 5,
		},
	}
	for _, tt := range tests {
		got, err := tt.ctx.Surface()
		if err != nil {
			t.Fatalf("%s: Surface err=%v", tt.name, err)
		}
		if got != tt.want {
			t.Fatalf("%s: Surface=%d want %d", tt.name, got, tt.want)
		}
	}

	for _, ctx := range []FrameWorkBatch{
		{Step: decodework.FrameStep{Kind: decodework.FrameStepIgnored}},
		{Step: decodework.FrameStep{Kind: decodework.FrameStepBegin, Begin: decodework.FramePlan{Surface: -1}}},
		{Step: decodework.FrameStep{Kind: decodework.FrameStepTile, Tile: decodework.FrameTilePlan{Surface: -1}}},
	} {
		if _, err := ctx.Surface(); !errors.Is(err, ErrInvalidBatch) {
			t.Fatalf("Surface err=%v want %v for ctx=%+v", err, ErrInvalidBatch, ctx)
		}
	}
}

func TestPoolPropagatesFirstError(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	want := errors.New("decode batch")

	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	err = pool.Execute(batches[:n], jobs[:], func(batch Batch, batchJobs []tile.Job) error {
		if batch.Worker == 1 {
			return want
		}
		return nil
	})
	if !errors.Is(err, want) {
		t.Fatalf("Execute err=%v want %v", err, want)
	}
}

func TestPoolRejectsInvalidInputs(t *testing.T) {
	jobs := testJobs()
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	err = pool.Execute([]Batch{{Worker: 2, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}}, jobs[:], func(Batch, []tile.Job) error {
		return nil
	})
	if !errors.Is(err, ErrInvalidBatch) {
		t.Fatalf("Execute invalid batch err=%v want %v", err, ErrInvalidBatch)
	}

	err = pool.Execute([]Batch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}}, jobs[:], nil)
	if !errors.Is(err, ErrInvalidCallback) {
		t.Fatalf("Execute nil callback err=%v want %v", err, ErrInvalidCallback)
	}

	err = pool.ExecuteFrameWork([]Batch{{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}}, jobs[:], FrameWorkBatch{}, nil)
	if !errors.Is(err, ErrInvalidCallback) {
		t.Fatalf("ExecuteFrameWork nil callback err=%v want %v", err, ErrInvalidCallback)
	}
}

func TestPoolRejectsAfterClose(t *testing.T) {
	jobs := testJobs()
	batch := Batch{Worker: 0, FirstJob: 0, Count: 1, FirstTile: 0, LastTile: 0}
	pool, err := NewPool(1)
	if err != nil {
		t.Fatal(err)
	}
	pool.Close()
	pool.Close()

	err = pool.Execute([]Batch{batch}, jobs[:], func(Batch, []tile.Job) error {
		return nil
	})
	if !errors.Is(err, ErrPoolClosed) {
		t.Fatalf("Execute err=%v want %v", err, ErrPoolClosed)
	}
}

func TestPoolExecuteAllocs(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	allocs := testing.AllocsPerRun(1000, func() {
		err := pool.Execute(batches[:n], jobs[:], func(Batch, []tile.Job) error {
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Pool.Execute allocated: %f", allocs)
	}
}

func TestPoolExecuteFrameWorkAllocs(t *testing.T) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	base := FrameWorkBatch{Step: decodework.FrameStep{Kind: decodework.FrameStepTile}}
	allocs := testing.AllocsPerRun(1000, func() {
		err := pool.ExecuteFrameWork(batches[:n], jobs[:], base, noopFrameWorkBatch)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Pool.ExecuteFrameWork allocated: %f", allocs)
	}
}

func BenchmarkPoolExecute(b *testing.B) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		b.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = pool.Execute(batches[:n], jobs[:], func(Batch, []tile.Job) error {
			return nil
		})
	}
}

func BenchmarkPoolExecuteFrameWork(b *testing.B) {
	jobs := testJobs()
	var batches [2]Batch
	n, err := BuildBatches(batches[:], jobs[:], 2)
	if err != nil {
		b.Fatal(err)
	}
	pool, err := NewPool(2)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()

	base := FrameWorkBatch{Step: decodework.FrameStep{Kind: decodework.FrameStepTile}}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = pool.ExecuteFrameWork(batches[:n], jobs[:], base, noopFrameWorkBatch)
	}
}

func noopFrameWorkBatch(FrameWorkBatch) error {
	return nil
}

// TestFrameWorkPlaneRangeChromaSubsampleMatchesLibaomCeil pins the chroma
// sub-sampling formula used by frameWorkPlaneRange. AV1 / libaom compute the
// chroma extent from a luma extent as ROUND_POWER_OF_TWO(end, ss), i.e.
// (end + ss) >> ss for ss in {0, 1}. For ss=1 (4:2:0 / 4:2:2 / 4:4:0) the
// expanded form (end + 1) >> 1 is mathematically identical to the implemented
// (end >> 1) + (end & 1):
//
//	end even N:  N/2 == (N + 1) >> 1 == (N >> 1) + 0
//	end odd  N:  (N+1)/2 == (N + 1) >> 1 == (N >> 1) + 1
//
// This ceil rounding is REQUIRED at frame edges where CodedWidth/CodedHeight
// can be odd: the partially-covered last luma column (or row) MUST contribute
// its chroma sample to the visible/MD5 extent (frame/buffer.go allocates
// the chroma plane as (W + ssx) >> ssx samples wide). Switching to floor
// rounding (end >> 1) would silently drop the rightmost chroma column on
// odd-CodedWidth frames such as the 1x1 fast-suite 1280x720, leaking a
// stride-padding sample into the MD5.
//
// AV1 tile/SB boundaries are always MI-aligned (multiples of MI_SIZE=4 luma
// pixels => even), so inner boundaries are unaffected by ceil-vs-floor; the
// formula's odd-end branch only fires at the right/bottom frame edge.
func TestFrameWorkPlaneRangeChromaSubsampleMatchesLibaomCeil(t *testing.T) {
	tests := []struct {
		start uint32
		end   uint32
		// wantStart/wantEnd are the chroma extents when subsampled=true.
		wantStart uint32
		wantEnd   uint32
	}{
		// Even (MI-aligned) ends: ceil == floor.
		{start: 0, end: 0, wantStart: 0, wantEnd: 0},
		{start: 0, end: 8, wantStart: 0, wantEnd: 4},
		{start: 0, end: 64, wantStart: 0, wantEnd: 32},
		{start: 64, end: 128, wantStart: 32, wantEnd: 64},
		{start: 0, end: 640, wantStart: 0, wantEnd: 320}, // SVC L2T1 base width
		{start: 0, end: 1280, wantStart: 0, wantEnd: 640},
		// Odd ends (only valid at frame edges): ceil must round up.
		{start: 0, end: 1, wantStart: 0, wantEnd: 1},
		{start: 0, end: 17, wantStart: 0, wantEnd: 9}, // 34x34 +1 visible would be 17
		{start: 0, end: 33, wantStart: 0, wantEnd: 17},
		{start: 0, end: 35, wantStart: 0, wantEnd: 18}, // 35-wide frame edge
		{start: 0, end: 67, wantStart: 0, wantEnd: 34}, // 67-wide frame edge
		{start: 64, end: 67, wantStart: 32, wantEnd: 34},
	}
	for _, tt := range tests {
		// Subsampled axis: must match libaom's ROUND_POWER_OF_TWO(end, 1).
		gotStart, gotEnd := frameWorkPlaneRange(tt.start, tt.end, true)
		libaomEnd := (tt.end + 1) >> 1
		libaomStart := tt.start >> 1
		if gotStart != tt.wantStart || gotEnd != tt.wantEnd {
			t.Fatalf("frameWorkPlaneRange(%d,%d,true)=(%d,%d) want (%d,%d)", tt.start, tt.end, gotStart, gotEnd, tt.wantStart, tt.wantEnd)
		}
		if gotEnd != libaomEnd {
			t.Fatalf("frameWorkPlaneRange(%d,%d,true) end=%d != libaom (end+1)>>1=%d", tt.start, tt.end, gotEnd, libaomEnd)
		}
		// Start uses floor (start >> 1). Tile/SB starts are always MI-aligned
		// (multiples of MI_SIZE=4 luma => even), so floor and ceil agree, but
		// the formula's left edge must NEVER round up, otherwise two adjacent
		// jobs with MI-aligned boundaries would overlap on the chroma column.
		if gotStart != libaomStart {
			t.Fatalf("frameWorkPlaneRange(%d,%d,true) start=%d != floor (start>>1)=%d", tt.start, tt.end, gotStart, libaomStart)
		}
		// Non-subsampled axis: identity.
		gotStart, gotEnd = frameWorkPlaneRange(tt.start, tt.end, false)
		if gotStart != tt.start || gotEnd != tt.end {
			t.Fatalf("frameWorkPlaneRange(%d,%d,false)=(%d,%d) want (%d,%d)", tt.start, tt.end, gotStart, gotEnd, tt.start, tt.end)
		}
	}
}

// TestFrameWorkPlaneRangeAdjacentJobsNoChromaOverlap pins that two jobs whose
// MI-aligned luma extents abut at an even luma column (every valid AV1 tile/SB
// boundary) produce chroma ranges that abut without overlap: job A's chroma
// end equals job B's chroma start. This is the safety property that prevents
// the formula from leaking one chroma column of writes from one SB stripe
// into the next.
func TestFrameWorkPlaneRangeAdjacentJobsNoChromaOverlap(t *testing.T) {
	// Walk every plausible MI-aligned boundary up to a generous limit. MI_SIZE
	// is 4 luma pixels and SB sizes are >= 64, so any tile/SB-aligned boundary
	// is a multiple of 4 (hence even). Cover the broader "multiple of 4" set
	// to be safe.
	for boundary := uint32(0); boundary <= 4096; boundary += 4 {
		// Job A: [0, boundary). Job B: [boundary, boundary+8).
		_, aEnd := frameWorkPlaneRange(0, boundary, true)
		bStart, _ := frameWorkPlaneRange(boundary, boundary+8, true)
		if aEnd != bStart {
			t.Fatalf("boundary luma=%d chroma aEnd=%d bStart=%d (gap or overlap)", boundary, aEnd, bStart)
		}
	}
}
