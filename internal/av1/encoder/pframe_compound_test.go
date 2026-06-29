package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestRealtimeInterTX16TreeMatchesLibaomNonRDCap(t *testing.T) {
	for _, tc := range []struct {
		name       string
		w, h       int
		size       tile.BlockSize
		wantSplit  [2]uint16
		wantLeaves int
	}{
		{name: "32x16", w: 32, h: 16, size: tile.BlockSize32x16, wantSplit: [2]uint16{1, 0}, wantLeaves: 2},
		{name: "16x32", w: 16, h: 32, size: tile.BlockSize16x32, wantSplit: [2]uint16{1, 0}, wantLeaves: 2},
		{name: "32x32", w: 32, h: 32, size: tile.BlockSize32x32, wantSplit: [2]uint16{1, 0}, wantLeaves: 4},
		{name: "64x64", w: 64, h: 64, size: tile.BlockSize64x64, wantSplit: [2]uint16{1, 0x0033}, wantLeaves: 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !realtimeInterTXUses16x16Leaves(tc.w, tc.h) {
				t.Fatalf("realtimeInterTXUses16x16Leaves(%d,%d)=false", tc.w, tc.h)
			}
			tree := realtimeInterTX16Tree(tc.w, tc.h)
			if tree.Split != tc.wantSplit {
				t.Fatalf("split=%#v want %#v", tree.Split, tc.wantSplit)
			}
			var helperLeaves []tile.TransformBlock
			if err := forEachRealtimeInterTX16Leaf(tc.w, tc.h, func(i, dx, dy int) error {
				if dx%16 != 0 || dy%16 != 0 {
					t.Fatalf("leaf %d offset=(%d,%d) not 16-aligned", i, dx, dy)
				}
				helperLeaves = append(helperLeaves, tile.TransformBlock{
					X4:        uint8(dx / 4),
					Y4:        uint8(dy / 4),
					Size:      tile.TransformSize16x16,
					VisibleW4: 4,
					VisibleH4: 4,
				})
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			if len(helperLeaves) != tc.wantLeaves {
				t.Fatalf("helper leaves=%d want %d", len(helperLeaves), tc.wantLeaves)
			}

			maxY, err := tile.MaxTransformSize(tc.size, parser.ColorConfig{}, 0)
			if err != nil {
				t.Fatal(err)
			}
			tree.Y = maxY
			tree.Variable = true
			var replayLeaves []tile.TransformBlock
			err = tree.ForEachLumaTXB(tile.TransformTreeRequest{
				Size:          tc.size,
				VisibleW4:     uint8(tc.w / 4),
				VisibleH4:     uint8(tc.h / 4),
				TransformMode: parser.TransformModeSwitchable,
				Inter:         true,
			}, func(block tile.TransformBlock) error {
				replayLeaves = append(replayLeaves, block)
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(replayLeaves) != tc.wantLeaves {
				t.Fatalf("replay leaves=%d got=%+v want %d", len(replayLeaves), replayLeaves, tc.wantLeaves)
			}
			for i, leaf := range replayLeaves {
				if leaf.Size != tile.TransformSize16x16 || leaf.VisibleW4 != 4 || leaf.VisibleH4 != 4 {
					t.Fatalf("leaf[%d]=%+v want visible 16x16", i, leaf)
				}
				if leaf != helperLeaves[i] {
					t.Fatalf("leaf[%d]=%+v helper %+v", i, leaf, helperLeaves[i])
				}
			}
		})
	}
}

func TestRealtimeInterTXLeafSizeMatchesLibaomCalculateTXSize(t *testing.T) {
	const (
		qAC     = int32(88)
		txLevel = 2
	)
	for _, tc := range []struct {
		name      string
		block     tile.BlockSize
		qIndex    uint8
		sse       uint32
		variance  uint32
		wantLeaf  int
		wantSplit [2]uint16
	}{
		{name: "high variance keeps 8x8", block: tile.BlockSize16x16, qIndex: 72, sse: 1000, variance: 1000, wantLeaf: 8, wantSplit: [2]uint16{1, 0}},
		{name: "dc dominated uses max square", block: tile.BlockSize16x16, qIndex: 72, sse: 2000, variance: 1000, wantLeaf: 16},
		{name: "low ac variance uses max square", block: tile.BlockSize16x16, qIndex: 72, sse: 200, variance: 100, wantLeaf: 16},
		{name: "rect max square is 8x8", block: tile.BlockSize16x8, qIndex: 72, sse: 2000, variance: 1000, wantLeaf: 8, wantSplit: [2]uint16{1, 0}},
		{name: "32x32 max square capped to 16x16", block: tile.BlockSize32x32, qIndex: 72, sse: 5000, variance: 1000, wantLeaf: 16, wantSplit: [2]uint16{1, 0}},
		{name: "larger than 32 forced to 16x16", block: tile.BlockSize64x64, qIndex: 72, sse: 1000, variance: 1000, wantLeaf: 16, wantSplit: [2]uint16{1, 0x0033}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotLeaf, err := realtimeInterTXLeafSizeForBlock(tc.block, tc.qIndex, qAC, txLevel, tc.sse, tc.variance)
			if err != nil {
				t.Fatal(err)
			}
			if gotLeaf != tc.wantLeaf {
				t.Fatalf("leaf=%d want %d", gotLeaf, tc.wantLeaf)
			}
			plan, err := realtimeInterTXPlanForBlock(tc.block, tc.qIndex, qAC, txLevel, tc.sse, tc.variance)
			if err != nil {
				t.Fatal(err)
			}
			if plan.leafSize != tc.wantLeaf {
				t.Fatalf("plan leaf=%d want %d", plan.leafSize, tc.wantLeaf)
			}
			if plan.tree.Split != tc.wantSplit {
				t.Fatalf("split=%#v want %#v", plan.tree.Split, tc.wantSplit)
			}
		})
	}
}

func TestRealtimeTXSizeLevelBasedOnQstepMatchesSpeedFeatures(t *testing.T) {
	for _, tc := range []struct {
		name   string
		w, h   int
		effort int8
		speed  int
		level  int
	}{
		{name: "1080p default speed8 level2", w: 1920, h: 1080, effort: 0, speed: 8, level: 2},
		{name: "1080p fastest speed10 disables qstep level", w: 1920, h: 1080, effort: WebRTCMinEffortLevel, speed: 10, level: 0},
		{name: "1080p max effort speed4 disables qstep level", w: 1920, h: 1080, effort: WebRTCMaxEffortLevel, speed: 4, level: 0},
		{name: "360p default speed8 level2", w: 640, h: 360, effort: 0, speed: 8, level: 2},
		{name: "below 360p default speed8 level1", w: 320, h: 180, effort: 0, speed: 8, level: 1},
		{name: "below 360p fastest speed10 keeps level1", w: 320, h: 180, effort: WebRTCMinEffortLevel, speed: 10, level: 1},
		{name: "720p speed7 disables qstep level", w: 1280, h: 720, effort: 1, speed: 7, level: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := realtimeLibaomSpeedForEffort(tc.effort); got != tc.speed {
				t.Fatalf("speed=%d want %d", got, tc.speed)
			}
			if got := realtimeTXSizeLevelBasedOnQstep(tc.w, tc.h, tc.effort); got != tc.level {
				t.Fatalf("level=%d want %d", got, tc.level)
			}
		})
	}
}

func TestRealtimeSourceContentSBMatchesLibaomThresholds(t *testing.T) {
	makePair := func(delta byte) (SourceFrame420, SourceFrame420) {
		src := SourceFrame420{Y: make([]byte, 64*64), YStride: 64, Width: 64, Height: 64}
		last := SourceFrame420{Y: make([]byte, 64*64), YStride: 64, Width: 64, Height: 64}
		for i := range src.Y {
			last.Y[i] = 80
			src.Y[i] = 80 + delta
		}
		return src, last
	}
	for _, tc := range []struct {
		name         string
		delta        byte
		wantNonRD    realtimeSourceSAD
		wantRD       realtimeSourceSAD
		wantLighting bool
		wantLowSum   bool
	}{
		{name: "zero", delta: 0, wantNonRD: realtimeSourceSADZero, wantRD: realtimeSourceSADLow},
		{name: "very-low", delta: 1, wantNonRD: realtimeSourceSADVeryLow, wantRD: realtimeSourceSADLow, wantLowSum: true},
		{name: "low", delta: 3, wantNonRD: realtimeSourceSADLow, wantRD: realtimeSourceSADMed, wantLighting: true},
		{name: "high", delta: 16, wantNonRD: realtimeSourceSADHigh, wantRD: realtimeSourceSADMed, wantLighting: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src, last := makePair(tc.delta)
			got := realtimeSourceContentSB(src, last, 0, 0, false)
			if got.sourceSADNonRD != tc.wantNonRD || got.sourceSADRD != tc.wantRD ||
				got.lightingChange != tc.wantLighting || got.lowSumDiff != tc.wantLowSum {
				t.Fatalf("content=%+v want nonrd=%d rd=%d lighting=%v lowSum=%v",
					got, tc.wantNonRD, tc.wantRD, tc.wantLighting, tc.wantLowSum)
			}
		})
	}
}

func TestRealtimeSourceVariancePerPixelMatchesAV1VarOffs(t *testing.T) {
	src := make([]byte, 8*8)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if (x+y)&1 == 0 {
				src[y*8+x] = 80
			} else {
				src[y*8+x] = 176
			}
		}
	}
	total, sum := 0, 0
	for _, px := range src {
		d := int(px) - 128
		sum += d
		total += d * d
	}
	variance := uint32(total) - uint32((int64(sum)*int64(sum))>>6)
	want := (variance + 32) >> 6
	if got := realtimeSourceVariancePerPixel(src, 8, 8, 8); got != want {
		t.Fatalf("source variance=%d want %d", got, want)
	}
}

func TestRealtimeVarPartSpeedFeaturesMatchLibaomRT(t *testing.T) {
	for _, tc := range []struct {
		name       string
		w, h       int
		effort     int8
		speed      int
		prefer     int
		qidx       int
		splitShift uint
		disable8x8 bool
	}{
		{name: "1080p speed8", w: 1920, h: 1080, effort: 0, speed: 8, prefer: 0, qidx: 4, splitShift: 8},
		{name: "1080p speed9", w: 1920, h: 1080, effort: -1, speed: 9, prefer: 3, qidx: 0, splitShift: 9},
		{name: "1080p speed10", w: 1920, h: 1080, effort: WebRTCMinEffortLevel, speed: 10, prefer: 3, qidx: 0, splitShift: 10},
		{name: "qvga speed8", w: 320, h: 180, effort: 0, speed: 8, prefer: 1, qidx: 4, splitShift: 8},
		{name: "360p speed6 disables 8x8 by q", w: 640, h: 360, effort: 2, speed: 6, prefer: 0, qidx: 2, splitShift: 7, disable8x8: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := realtimeVarPartSpeedFeatures(tc.w, tc.h, tc.effort)
			if got.speed != tc.speed || got.preferLargeBlocks != tc.prefer ||
				got.varPartBasedOnQidx != tc.qidx || got.splitThresholdShift != tc.splitShift ||
				got.disable8x8ByQidx != tc.disable8x8 {
				t.Fatalf("features=%+v want speed=%d prefer=%d qidx=%d split=%d disable8x8=%v",
					got, tc.speed, tc.prefer, tc.qidx, tc.splitShift, tc.disable8x8)
			}
		})
	}
}

func TestRealtimeVarPartMotionLevelMatchesLibaomRT(t *testing.T) {
	for _, tc := range []struct {
		name   string
		w, h   int
		effort int8
		want   int
	}{
		{name: "speed8", w: 1920, h: 1080, effort: 0, want: 2},
		{name: "speed9", w: 1920, h: 1080, effort: -1, want: 3},
		{name: "speed10 1080p", w: 1920, h: 1080, effort: WebRTCMinEffortLevel, want: 3},
		{name: "speed10 qvga", w: 320, h: 180, effort: WebRTCMinEffortLevel, want: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := realtimeEstimateMotionForVarBasedPartition(tc.w, tc.h, tc.effort); got != tc.want {
				t.Fatalf("est_motion=%d want %d", got, tc.want)
			}
		})
	}
	col, row, large := realtimeIntProSearchWindow64(1920, 1080, realtimeSourceSADHigh)
	if col != 256 || row != 256 || !large {
		t.Fatalf("1080p high-SAD search=(%d,%d,%v), want (256,256,true)", col, row, large)
	}
	col, row, large = realtimeIntProSearchWindow64(1920, 1080, realtimeSourceSADMed)
	if col != 32 || row != 32 || large {
		t.Fatalf("1080p medium-SAD search=(%d,%d,%v), want (32,32,false)", col, row, large)
	}
}

func TestRealtimeVarPartThresholdsMatchLibaomRT(t *testing.T) {
	content := defaultRealtimeContentStateSB()
	got := realtimeVarPartThresholds(1920, 1080, 120, 100, 0, content, 0)
	want := [5]int64{62, 125, 750, 64000, 500}
	if got != want {
		t.Fatalf("speed8 thresholds=%v want %v", got, want)
	}
	content.sourceSADNonRD = realtimeSourceSADLow
	got = realtimeVarPartThresholds(320, 180, 60, 80, 0, content, 0)
	want = [5]int64{50, 12, 52, 933, 400}
	if got != want {
		t.Fatalf("qvga thresholds=%v want %v", got, want)
	}
}

func TestRealtimeVectorMatchMatchesLibaomSearchOrder(t *testing.T) {
	var ref [128]int16
	var src [64]int16
	for i := range ref {
		ref[i] = int16((i*i*17 + i*29 + 7) & 511)
	}
	copy(src[:], ref[32:32+64])
	center, sad := realtimeVectorMatch(ref[:], src[:], 4, 32, 32, false)
	if center != 0 || sad != 0 {
		t.Fatalf("center=%d sad=%d want center=0 sad=0", center, sad)
	}
}

func TestRealtimeIntProMotionEstimation64MatchesLibaomShift(t *testing.T) {
	const width, height = 160, 160
	const px, py = 48, 48
	const wantDX, wantDY = 16, -16
	ref := make([]byte, width*height)
	src := make([]byte, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			ref[y*width+x] = byte((x*37 + y*19 + x*y*3 + (x>>2)*11 + (y>>1)*5) & 255)
			src[y*width+x] = byte((x*13 + y*7 + 91) & 255)
		}
	}
	for y := 0; y < 64; y++ {
		copy(src[(py+y)*width+px:(py+y)*width+px+64], ref[(py+wantDY+y)*width+px+wantDX:(py+wantDY+y)*width+px+wantDX+64])
	}
	dx, dy, sad, zeroSAD, ok := realtimeIntProMotionEstimation64(src, ref, width, width, height, px, py, realtimeSourceSADHigh, 0)
	if !ok {
		t.Fatal("projection ME disabled")
	}
	if dx != wantDX || dy != wantDY || sad != 0 {
		t.Fatalf("motion=(%d,%d) sad=%d want (%d,%d) sad=0", dx, dy, sad, wantDX, wantDY)
	}
	if zeroSAD == 0 {
		t.Fatal("zero-motion SAD unexpectedly zero")
	}
}

func TestRealtimeSetVTPartitioningMatchesLibaomOrder(t *testing.T) {
	var part realtimeVPartVariances
	realtimeFillVariance(100, 0, 0, &part.none)
	realtimeFillVariance(1, 0, 0, &part.vert[0])
	realtimeFillVariance(1, 0, 0, &part.vert[1])
	realtimeFillVariance(100, 0, 0, &part.horz[0])
	realtimeFillVariance(100, 0, 0, &part.horz[1])
	if got := realtimeSetVTPartitioning(&part, 1000, tile.BlockLevel32x32, tile.BlockLevel16x16, realtimePartEvalAll); got != tile.PartitionV {
		t.Fatalf("partition=%d want vertical", got)
	}
	if got := realtimeSetVTPartitioning(&part, 1000, tile.BlockLevel32x32, tile.BlockLevel16x16, realtimePartEvalOnlySplit); got != tile.PartitionSplit {
		t.Fatalf("forced split partition=%d", got)
	}
	if got := realtimeSetVTPartitioning(&part, 1000, tile.BlockLevel32x32, tile.BlockLevel16x16, realtimePartEvalOnlyNone); got != tile.PartitionNone {
		t.Fatalf("forced none partition=%d", got)
	}
}

func TestRealtimeVarianceTreeMatchesLibaomVPartVarMath(t *testing.T) {
	var vt realtimeVarTree16
	realtimeFillVariance(0, 0, 0, &vt.split[0])
	realtimeFillVariance(100, 10, 0, &vt.split[1])
	realtimeFillVariance(0, 0, 0, &vt.split[2])
	realtimeFillVariance(100, 10, 0, &vt.split[3])
	realtimeFillVarianceTree16(&vt)
	if got := realtimeGetVariance(&vt.part.none); got != 6400 {
		t.Fatalf("variance=%d want 6400", got)
	}
	if got := realtimeGetVariance(&vt.part.horz[0]); got != 6400 {
		t.Fatalf("horizontal half variance=%d want 6400", got)
	}
	if got := realtimeGetVariance(&vt.part.vert[0]); got != 0 {
		t.Fatalf("vertical half variance=%d want 0", got)
	}
}

func TestRealtimeReduceMVPelPrecisionLowcomplexLevelMatchesSpeedFeatures(t *testing.T) {
	for _, tc := range []struct {
		name   string
		w, h   int
		effort int8
		want   int
	}{
		{name: "1080p speed8", w: 1920, h: 1080, effort: 0, want: 2},
		{name: "1080p speed10", w: 1920, h: 1080, effort: WebRTCMinEffortLevel, want: 0},
		{name: "720p speed7", w: 1280, h: 720, effort: 1, want: 2},
		{name: "below360 speed10", w: 320, h: 180, effort: WebRTCMinEffortLevel, want: 1},
		{name: "360p speed8", w: 640, h: 360, effort: 0, want: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := realtimeReduceMVPelPrecisionLowcomplexLevel(tc.w, tc.h, tc.effort); got != tc.want {
				t.Fatalf("level=%d want %d", got, tc.want)
			}
		})
	}
}

func TestRealtimeSubpelStopForBlockMatchesLibaomLowcomplexRule(t *testing.T) {
	makeFrame := func(v0, v1 byte) SourceFrame420 {
		f := SourceFrame420{Y: make([]byte, 1280*720), YStride: 1280, Width: 1280, Height: 720}
		for y := 0; y < 720; y++ {
			for x := 0; x < 1280; x++ {
				if (x+y)&1 == 0 {
					f.Y[y*1280+x] = v0
				} else {
					f.Y[y*1280+x] = v1
				}
			}
		}
		return f
	}
	for _, tc := range []struct {
		name string
		src  SourceFrame420
		want realtimeSubpelStop
	}{
		{name: "flat-fullpel", src: makeFrame(128, 128), want: realtimeSubpelStopFull},
		{name: "moderate-halfpel", src: makeFrame(80, 176), want: realtimeSubpelStopHalf},
		{name: "high-quarterpel", src: makeFrame(0, 255), want: realtimeSubpelStopQuarter},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var st lossyEncodeState
			st.qIndex = 72
			st.effortLevel = 0
			st.prepareRealtimeSourceContent(tc.src, tc.src, 20, 12, 0, uint16(tc.src.Width/4))
			if got := st.realtimeSubpelStopForBlock(tc.src, 0, 0, 32); got != tc.want {
				t.Fatalf("stop=%d want %d", got, tc.want)
			}
		})
	}
}

func TestRealtimeInterTXLeafSizeHonorsQstepSpeedFeatureLevel(t *testing.T) {
	const (
		qIndex   = uint8(72)
		qAC      = int32(88)
		sse      = uint32(100)
		variance = uint32(100)
	)
	level0, err := realtimeInterTXLeafSizeForBlock(tile.BlockSize16x16, qIndex, qAC, 0, sse, variance)
	if err != nil {
		t.Fatal(err)
	}
	if level0 != 8 {
		t.Fatalf("level0 leaf=%d want 8", level0)
	}
	level2, err := realtimeInterTXLeafSizeForBlock(tile.BlockSize16x16, qIndex, qAC, 2, sse, variance)
	if err != nil {
		t.Fatal(err)
	}
	if level2 != 16 {
		t.Fatalf("level2 leaf=%d want 16", level2)
	}
}

func TestRealtimeInterTXPlanReplayMatchesTransformTree(t *testing.T) {
	for _, tc := range []struct {
		name       string
		size       tile.BlockSize
		leaf       int
		wantSplit  [2]uint16
		wantLeaves int
	}{
		{name: "8x8 leaf8", size: tile.BlockSize8x8, leaf: 8, wantLeaves: 1},
		{name: "16x8 leaf8", size: tile.BlockSize16x8, leaf: 8, wantSplit: [2]uint16{1, 0}, wantLeaves: 2},
		{name: "8x16 leaf8", size: tile.BlockSize8x16, leaf: 8, wantSplit: [2]uint16{1, 0}, wantLeaves: 2},
		{name: "16x16 leaf8", size: tile.BlockSize16x16, leaf: 8, wantSplit: [2]uint16{1, 0}, wantLeaves: 4},
		{name: "32x16 leaf8", size: tile.BlockSize32x16, leaf: 8, wantSplit: [2]uint16{1, 0x0003}, wantLeaves: 8},
		{name: "16x32 leaf8", size: tile.BlockSize16x32, leaf: 8, wantSplit: [2]uint16{1, 0x0011}, wantLeaves: 8},
		{name: "32x32 leaf8", size: tile.BlockSize32x32, leaf: 8, wantSplit: [2]uint16{1, 0x0033}, wantLeaves: 16},
		{name: "32x16 leaf16", size: tile.BlockSize32x16, leaf: 16, wantSplit: [2]uint16{1, 0}, wantLeaves: 2},
		{name: "32x32 leaf16", size: tile.BlockSize32x32, leaf: 16, wantSplit: [2]uint16{1, 0}, wantLeaves: 4},
		{name: "64x16 leaf16", size: tile.BlockSize64x16, leaf: 16, wantSplit: [2]uint16{1, 0x0003}, wantLeaves: 4},
		{name: "16x64 leaf16", size: tile.BlockSize16x64, leaf: 16, wantSplit: [2]uint16{1, 0x0011}, wantLeaves: 4},
		{name: "64x64 leaf16", size: tile.BlockSize64x64, leaf: 16, wantSplit: [2]uint16{1, 0x0033}, wantLeaves: 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan, err := realtimeInterTXPlanForLeafSize(tc.size, tc.leaf)
			if err != nil {
				t.Fatal(err)
			}
			if plan.tree.Split != tc.wantSplit {
				t.Fatalf("split=%#v want %#v", plan.tree.Split, tc.wantSplit)
			}

			var helperLeaves []tile.TransformBlock
			err = plan.ForEachLeaf(func(i, dx, dy int) error {
				helperLeaves = append(helperLeaves, tile.TransformBlock{
					X4:        uint8(dx / 4),
					Y4:        uint8(dy / 4),
					Size:      plan.leafTX,
					VisibleW4: uint8(tc.leaf / 4),
					VisibleH4: uint8(tc.leaf / 4),
				})
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(helperLeaves) != tc.wantLeaves {
				t.Fatalf("helper leaves=%d want %d", len(helperLeaves), tc.wantLeaves)
			}

			dims, ok := tc.size.Dimensions()
			if !ok {
				t.Fatal("bad block size")
			}
			maxY, err := tile.MaxTransformSize(tc.size, parser.ColorConfig{}, 0)
			if err != nil {
				t.Fatal(err)
			}
			tree := plan.tree
			tree.Y = maxY
			tree.Variable = plan.Variable()
			var replayLeaves []tile.TransformBlock
			err = tree.ForEachLumaTXB(tile.TransformTreeRequest{
				Size:          tc.size,
				VisibleW4:     dims.W4,
				VisibleH4:     dims.H4,
				TransformMode: parser.TransformModeSwitchable,
				Inter:         true,
			}, func(block tile.TransformBlock) error {
				replayLeaves = append(replayLeaves, block)
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if len(replayLeaves) != len(helperLeaves) {
				t.Fatalf("replay leaves=%d helper=%d replay=%+v helper=%+v", len(replayLeaves), len(helperLeaves), replayLeaves, helperLeaves)
			}
			for i := range replayLeaves {
				if replayLeaves[i] != helperLeaves[i] {
					t.Fatalf("leaf[%d]=%+v helper %+v", i, replayLeaves[i], helperLeaves[i])
				}
			}
		})
	}
}

func TestEncodePBlockCompoundLastGolden8x8(t *testing.T) {
	const w, h = 16, 16
	solid := func(y, u, v byte) SourceFrame420 {
		f := SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		for i := range f.Y {
			f.Y[i] = y
		}
		for i := range f.U {
			f.U[i] = u
			f.V[i] = v
		}
		return f
	}
	src := solid(128, 128, 128)
	ref := solid(200, 180, 90)
	golden := solid(56, 76, 166)
	recon := solid(0, 0, 0)

	var pc pframeCoder
	if err := pc.reset(72, 1, nil, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}); err != nil {
		t.Fatal(err)
	}
	st := &pc.st
	pc.writer.Reset(pc.writerBuf[:0])
	st.w = &pc.writer
	st.interTxTypeReq = tile.InterTransformTypeRequest{
		Size:        tile.TransformSize8x8,
		QIndexKnown: true,
		QIndex:      72,
	}
	st.afterSkipInter = func() error {
		return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, transform.TypeDCTDCT)
	}
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	st.sadPerBit = int(0.0418*(dcq/4) + 2.4107)
	st.grid8Cols = 2
	st.mv8Grid = make([]motion.Vector, 4)
	st.sad8Grid = make([]uint32, 4)

	block := tile.BlockVisit{
		MIColEnd:  2,
		MIRowEnd:  2,
		Size:      tile.BlockSize8x8,
		VisibleW4: 2,
		VisibleH4: 2,
	}
	walkReq := tile.BlockWalkRequest{MIColEnd: 4, MIRowEnd: 4}
	if err := st.encodePBlock(src, ref, &golden, &recon, block, &pc.scratch, &pc.refCDFs, &pc.modeCDFs, parser.ReferenceModeSelect, walkReq, 4, 4); err != nil {
		t.Fatal(err)
	}

	got := pc.scratch.Mode.AboveInterMotion[0]
	if !got.References.Compound ||
		got.References.Ref[0] != tile.ReferenceFrameLast ||
		got.References.Ref[1] != tile.ReferenceFrameGolden ||
		got.Mode.CompoundMode != tile.CompoundInterModeGlobalGlobal {
		t.Fatalf("motion = %+v, want GLOBAL_GLOBAL LAST+GOLDEN compound", got)
	}
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if recon.Y[y*w+x] != src.Y[y*w+x] {
				t.Fatalf("recon Y(%d,%d)=%d want %d", x, y, recon.Y[y*w+x], src.Y[y*w+x])
			}
		}
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if recon.U[y*(w/2)+x] != src.U[y*(w/2)+x] || recon.V[y*(w/2)+x] != src.V[y*(w/2)+x] {
				t.Fatalf("recon chroma(%d,%d)=(%d,%d) want (%d,%d)", x, y, recon.U[y*(w/2)+x], recon.V[y*(w/2)+x], src.U[y*(w/2)+x], src.V[y*(w/2)+x])
			}
		}
	}
}

func TestEncodePBlockGoldenSingleLarge(t *testing.T) {
	for _, tc := range []struct {
		name string
		w, h int
		size tile.BlockSize
		tx   tile.TransformSize
	}{
		{name: "16x16", w: 16, h: 16, size: tile.BlockSize16x16, tx: tile.TransformSize16x16},
		{name: "32x32", w: 32, h: 32, size: tile.BlockSize32x32, tx: tile.TransformSize32x32},
		{name: "16x8", w: 16, h: 8, size: tile.BlockSize16x8, tx: tile.TransformSize16x8},
		{name: "8x16", w: 8, h: 16, size: tile.BlockSize8x16, tx: tile.TransformSize8x16},
		{name: "32x16", w: 32, h: 16, size: tile.BlockSize32x16, tx: tile.TransformSize32x16},
		{name: "16x32", w: 16, h: 32, size: tile.BlockSize16x32, tx: tile.TransformSize16x32},
		{name: "64x16", w: 64, h: 16, size: tile.BlockSize64x16, tx: tile.TransformSize64x16},
		{name: "16x64", w: 16, h: 64, size: tile.BlockSize16x64, tx: tile.TransformSize16x64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, h := tc.w, tc.h
			solid := func(y, u, v byte) SourceFrame420 {
				f := SourceFrame420{
					Y:            make([]byte, w*h),
					U:            make([]byte, w*h/4),
					V:            make([]byte, w*h/4),
					YStride:      w,
					ChromaStride: w / 2,
					Width:        w,
					Height:       h,
				}
				for i := range f.Y {
					f.Y[i] = y
				}
				for i := range f.U {
					f.U[i] = u
					f.V[i] = v
				}
				return f
			}
			src := solid(56, 76, 166)
			ref := solid(200, 180, 90)
			golden := solid(56, 76, 166)
			recon := solid(0, 0, 0)

			var pc pframeCoder
			if err := pc.reset(72, 1, nil, parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}); err != nil {
				t.Fatal(err)
			}
			st := &pc.st
			pc.writer.Reset(pc.writerBuf[:0])
			st.w = &pc.writer
			st.interTxTypeReq = tile.InterTransformTypeRequest{
				Size:        tc.tx,
				QIndexKnown: true,
				QIndex:      72,
			}
			st.afterSkipInter = func() error {
				return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, transform.TypeDCTDCT)
			}
			dcq := float64(st.yQuant.DC)
			st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
			st.sadPerBit = int(0.0418*(dcq/4) + 2.4107)
			switch {
			case w == 16 && h == 16:
				st.grid16Cols = 1
				st.mv16Grid = make([]motion.Vector, 1)
				st.sad16Grid = make([]uint32, 1)
			case w == 32 && h == 32:
				st.grid32Cols = 1
				st.mv32Grid = make([]motion.Vector, 1)
				st.sad32Grid = make([]uint32, 1)
			case w >= 32 || h >= 32:
				st.sadCacheEpoch = 1
				st.grid16Cols = max(1, w/16)
				st.mv16Grid = make([]motion.Vector, 2)
				st.sad16Grid = []uint32{sadCachePack(st.sadCacheEpoch, 1<<15), sadCachePack(st.sadCacheEpoch, 1<<15)}
			default:
				st.sadCacheEpoch = 1
				st.grid8Cols = max(1, w/8)
				st.mv8Grid = make([]motion.Vector, 2)
				st.sad8Grid = []uint32{sadCachePack(st.sadCacheEpoch, 1<<14), sadCachePack(st.sadCacheEpoch, 1<<14)}
			}

			miW, miH := uint16(w/4), uint16(h/4)
			block := tile.BlockVisit{
				MIColEnd:  miW,
				MIRowEnd:  miH,
				Size:      tc.size,
				VisibleW4: uint8(miW),
				VisibleH4: uint8(miH),
			}
			walkReq := tile.BlockWalkRequest{MIColEnd: miW, MIRowEnd: miH}
			if err := st.encodePBlock(src, ref, &golden, &recon, block, &pc.scratch, &pc.refCDFs, &pc.modeCDFs, parser.ReferenceModeSingle, walkReq, miW, miH); err != nil {
				t.Fatal(err)
			}

			got := pc.scratch.Mode.AboveInterMotion[0]
			if got.References.Compound ||
				got.References.Ref[0] != tile.ReferenceFrameGolden ||
				got.References.Ref[1] != tile.ReferenceFrameNone ||
				got.Mode.Mode != tile.InterModeGlobalMV {
				t.Fatalf("motion = %+v, want GLOBALMV single GOLDEN", got)
			}
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					if recon.Y[y*w+x] != src.Y[y*w+x] {
						t.Fatalf("recon Y(%d,%d)=%d want %d", x, y, recon.Y[y*w+x], src.Y[y*w+x])
					}
				}
			}
		})
	}
}

func TestCompoundGoldenLikely(t *testing.T) {
	const w, h = 16, 16
	solid := func(y byte) SourceFrame420 {
		f := SourceFrame420{
			Y:            make([]byte, w*h),
			U:            make([]byte, w*h/4),
			V:            make([]byte, w*h/4),
			YStride:      w,
			ChromaStride: w / 2,
			Width:        w,
			Height:       h,
		}
		for i := range f.Y {
			f.Y[i] = y
		}
		for i := range f.U {
			f.U[i] = 128
			f.V[i] = 128
		}
		return f
	}

	ref := solid(200)
	golden := solid(56)
	average := solid(128)
	var st lossyEncodeState
	if !compoundGoldenLikely(&st, average, ref, &golden) {
		t.Fatal("compoundGoldenLikely returned false for a LAST/GOLDEN average")
	}
	if compoundGoldenLikely(&st, ref, ref, &golden) {
		t.Fatal("compoundGoldenLikely returned true when LAST already predicts the frame")
	}
}
