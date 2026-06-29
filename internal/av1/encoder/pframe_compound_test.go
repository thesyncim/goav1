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
