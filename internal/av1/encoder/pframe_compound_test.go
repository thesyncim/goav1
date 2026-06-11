package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

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
	if err := pc.reset(72, 1, nil); err != nil {
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
	st.sad8Grid = []int32{-1, -1, -1, -1}

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
		n    int
		size tile.BlockSize
		tx   tile.TransformSize
	}{
		{name: "16x16", n: 16, size: tile.BlockSize16x16, tx: tile.TransformSize16x16},
		{name: "32x32", n: 32, size: tile.BlockSize32x32, tx: tile.TransformSize32x32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, h := tc.n, tc.n
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
			if err := pc.reset(72, 1, nil); err != nil {
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
			switch tc.n {
			case 16:
				st.grid16Cols = 1
				st.mv16Grid = make([]motion.Vector, 1)
				st.sad16Grid = []int32{-1}
			case 32:
				st.grid32Cols = 1
				st.mv32Grid = make([]motion.Vector, 1)
				st.sad32Grid = []int32{-1}
			}

			mi := uint16(tc.n / 4)
			block := tile.BlockVisit{
				MIColEnd:  mi,
				MIRowEnd:  mi,
				Size:      tc.size,
				VisibleW4: uint8(mi),
				VisibleH4: uint8(mi),
			}
			walkReq := tile.BlockWalkRequest{MIColEnd: mi, MIRowEnd: mi}
			if err := st.encodePBlock(src, ref, &golden, &recon, block, &pc.scratch, &pc.refCDFs, &pc.modeCDFs, parser.ReferenceModeSingle, walkReq, mi, mi); err != nil {
				t.Fatal(err)
			}

			got := pc.scratch.Mode.AboveInterMotion[0]
			if got.References.Compound ||
				got.References.Ref[0] != tile.ReferenceFrameGolden ||
				got.References.Ref[1] != tile.ReferenceFrameNone ||
				got.Mode.Mode != tile.InterModeGlobalMV {
				t.Fatalf("motion = %+v, want GLOBALMV single GOLDEN", got)
			}
			for y := 0; y < tc.n; y++ {
				for x := 0; x < tc.n; x++ {
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
