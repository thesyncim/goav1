package tile

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestWritePaletteRoundTrip writes luma palettes (mode info and color index
// maps, with neighbour palettes seeding the color cache and partially
// visible blocks) and reads them back through ReadPaletteMode /
// ReadPaletteTokens.
func TestWritePaletteRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	color := parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	sizes := []BlockSize{BlockSize8x8, BlockSize16x16, BlockSize32x32, BlockSize16x8, BlockSize8x16, BlockSize64x64, BlockSize32x16}

	type rec struct {
		req PaletteModeRequest
		res PaletteModeResult
		m   [MaxPalettePixels]uint8
	}
	var recs []rec
	var encCDFs IntraModeCDFs
	if err := encCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<20))
	var encCtx BlockModeContext
	const blocks = 400
	for b := range blocks {
		size := sizes[rng.Intn(len(sizes))]
		dims, _ := size.Dimensions()
		bw4, bh4 := int(dims.W4), int(dims.H4)
		x4 := (b * 3) % (MaxBlockModeSlots - bw4 + 1)
		y4 := (b * 5) % (MaxBlockModeSlots - bh4 + 1)
		req := PaletteModeRequest{
			AllowScreenContentTools: true,
			Size:                    size,
			LumaMode:                IntraModeDC,
			X4:                      uint8(x4),
			Y4:                      uint8(y4),
			HaveTop:                 y4 > 0,
			HaveLeft:                x4 > 0,
			BitDepth:                8,
			Color:                   color,
			ChromaMode:              ChromaIntraModeDC,
			ChromaModeValid:         true,
			HasChroma:               true,
		}
		if rng.Intn(3) == 0 {
			req.VisibleW4 = uint8(1 + rng.Intn(bw4))
			req.VisibleH4 = uint8(1 + rng.Intn(bh4))
		}
		var r rec
		r.req = req
		if rng.Intn(4) != 0 {
			n := PaletteMinSize + rng.Intn(PaletteMaxSize-PaletteMinSize+1)
			// Strictly ascending colors, some reused from a small shared set
			// so the neighbours' caches hit.
			used := map[int]bool{}
			for len(used) < n {
				if rng.Intn(2) == 0 {
					used[[...]int{0, 16, 64, 128, 200, 255}[rng.Intn(6)]] = true
				} else {
					used[rng.Intn(256)] = true
				}
			}
			r.res.YSize = uint8(n)
			k := 0
			for v := range 256 {
				if used[v] {
					r.res.YColors[k] = uint16(v)
					k++
				}
			}
			stride := bw4 * 4
			w4, h4 := paletteOnscreen4(bw4, bh4, req.VisibleW4, req.VisibleH4, 0, 0)
			for y := range h4 * 4 {
				for x := range w4 * 4 {
					v := uint8((x/3 + y/5) % n)
					if rng.Intn(8) == 0 {
						v = uint8(rng.Intn(n))
					}
					r.m[y*stride+x] = v
				}
			}
			// pal_idx_finish's extension of the off-screen part.
			for y := range h4 * 4 {
				for x := w4 * 4; x < stride; x++ {
					r.m[y*stride+x] = r.m[y*stride+w4*4-1]
				}
			}
			for y := h4 * 4; y < bh4*4; y++ {
				copy(r.m[y*stride:(y+1)*stride], r.m[(h4*4-1)*stride:h4*4*stride])
			}
			r.res.YMap = &r.m
		}
		if c, err := PaletteModeCost(&encCDFs, &encCtx, req, &r.res); err != nil || (r.res.YSize > 0 && c < 8<<9) {
			t.Fatalf("block %d mode cost %d err %v", b, c, err)
		}
		if err := WritePaletteMode(&w, &encCDFs, &encCtx, req, &r.res); err != nil {
			t.Fatalf("block %d write mode: %v", b, err)
		}
		if err := WritePaletteTokens(&w, &encCDFs, req, &r.res); err != nil {
			t.Fatalf("block %d write tokens: %v", b, err)
		}
		mark := r.res
		mark.YMap = nil
		if err := encCtx.MarkPaletteY(size, x4, y4, mark); err != nil {
			t.Fatal(err)
		}
		if err := encCtx.MarkPaletteUV(size, x4, y4, PaletteModeResult{}); err != nil {
			t.Fatal(err)
		}
		recs = append(recs, r)
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decCDFs IntraModeCDFs
	if err := decCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	var decCtx BlockModeContext
	var scratch PaletteModeScratch
	for i := range recs {
		r := &recs[i]
		var got PaletteModeResult
		if err := state.ReadPaletteMode(&decCDFs, &decCtx, r.req, &got, &scratch); err != nil {
			t.Fatalf("block %d read mode: %v", i, err)
		}
		if got.YSize != r.res.YSize || got.UVSize != 0 || got.YColors != r.res.YColors {
			t.Fatalf("block %d: got size %d colors %v, want %d %v", i, got.YSize, got.YColors, r.res.YSize, r.res.YColors)
		}
		if err := state.ReadPaletteTokens(&decCDFs, r.req, &got, &scratch); err != nil {
			t.Fatalf("block %d read tokens: %v", i, err)
		}
		if got.YSize > 0 {
			dims, _ := r.req.Size.Dimensions()
			n := int(dims.W4) * int(dims.H4) * 16
			if got.YMap == nil || !bytes.Equal(got.YMap[:n], r.m[:n]) {
				t.Fatalf("block %d: color map mismatch", i)
			}
		}
		mark := got
		mark.YMap = nil
		if err := decCtx.MarkPaletteY(r.req.Size, int(r.req.X4), int(r.req.Y4), mark); err != nil {
			t.Fatal(err)
		}
		if err := decCtx.MarkPaletteUV(r.req.Size, int(r.req.X4), int(r.req.Y4), PaletteModeResult{}); err != nil {
			t.Fatal(err)
		}
	}
}
