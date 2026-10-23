package tile

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

// palette_write.go is the encoder side of palette_mode_info and
// palette_tokens for the luma palette: the forward of ReadPaletteMode /
// ReadPaletteTokens (libaom av1/encoder/bitstream.c write_palette_mode_info,
// write_palette_colors_y, delta_encode_palette_colors and
// av1/encoder/tokenize.c av1_tokenize_color_map). Chroma palettes are not
// produced (SVT-AV1 has no chroma palette search), so a UV palette is
// rejected and has_palette_uv codes 0.
//
// Each writer has a cost twin in entropy.CostSymbol units (512 per bit) for
// the encoder's rate-distortion pricing; both walk the same syntax.

// paletteSink either codes the palette syntax into w or, with w nil,
// accumulates its cost, so the writers and their cost twins share one
// traversal (a concrete type, so the cost path stays on the stack).
type paletteSink struct {
	w    *entropy.Writer
	cost uint32
}

func (s *paletteSink) WriteBinaryCDFTrusted(cdf *entropy.CDF, sym int) {
	if s.w != nil {
		s.w.WriteBinaryCDFTrusted(cdf, sym)
		return
	}
	s.cost += entropy.CostSymbol(cdf, sym)
}

func (s *paletteSink) WriteCDF(cdf *entropy.CDF, sym int) {
	if s.w != nil {
		s.w.WriteCDF(cdf, sym)
		return
	}
	s.cost += entropy.CostSymbol(cdf, sym)
}

func (s *paletteSink) WriteLiteral(value uint32, n int) {
	if s.w != nil {
		s.w.WriteLiteral(value, n)
		return
	}
	s.cost += uint32(n) << 9
}

func (s *paletteSink) WriteBit(val int) {
	if s.w != nil {
		s.w.WriteBit(val)
		return
	}
	s.cost += 1 << 9
}

// WritePaletteMode codes palette_mode_info for a block whose luma palette is
// res (YSize 0 codes has_palette_y = 0), with has_palette_uv = 0.
func WritePaletteMode(w *entropy.Writer, cdfs *IntraModeCDFs, ctx *BlockModeContext, req PaletteModeRequest, res *PaletteModeResult) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	s := paletteSink{w: w}
	return writePaletteMode(&s, cdfs, ctx, req, res)
}

// PaletteModeCost prices WritePaletteMode.
func PaletteModeCost(cdfs *IntraModeCDFs, ctx *BlockModeContext, req PaletteModeRequest, res *PaletteModeResult) (uint32, error) {
	var c paletteSink
	err := writePaletteMode(&c, cdfs, ctx, req, res)
	return c.cost, err
}

func writePaletteMode(w *paletteSink, cdfs *IntraModeCDFs, ctx *BlockModeContext, req PaletteModeRequest, res *PaletteModeResult) error {
	if cdfs == nil || ctx == nil || res == nil || res.UVSize != 0 {
		return ErrInvalidDecodeState
	}
	allowed, err := PaletteAllowed(req.AllowScreenContentTools, req.Size)
	if err != nil {
		return err
	}
	if !allowed {
		if res.YSize != 0 {
			return ErrInvalidDecodeState
		}
		return nil
	}
	sizeCtx, err := PaletteBSizeContext(req.Size)
	if err != nil {
		return err
	}
	if req.LumaMode == IntraModeDC {
		modeCtx, err := ctx.PaletteYModeContext(int(req.X4), int(req.Y4), req.HaveTop, req.HaveLeft)
		if err != nil {
			return err
		}
		modeCDF, err := cdfs.PaletteYModeCDF(sizeCtx, modeCtx)
		if err != nil {
			return err
		}
		use := 0
		if res.YSize > 0 {
			use = 1
		}
		w.WriteBinaryCDFTrusted(modeCDF, use)
		if use != 0 {
			if res.YSize < PaletteMinSize || res.YSize > PaletteMaxSize {
				return ErrInvalidDecodeState
			}
			sizeCDF, err := cdfs.PaletteYSizeCDF(sizeCtx)
			if err != nil {
				return err
			}
			w.WriteCDF(sizeCDF, int(res.YSize)-PaletteMinSize)
			if err := writePaletteColorsY(w, ctx, req, res); err != nil {
				return err
			}
		}
	} else if res.YSize != 0 {
		return ErrInvalidDecodeState
	}
	if req.Color.MonoChrome || !req.HasChroma || !req.ChromaModeValid || req.ChromaMode != ChromaIntraModeDC {
		return nil
	}
	uvCtx := 0
	if res.YSize > 0 {
		uvCtx = 1
	}
	modeCDF, err := cdfs.PaletteUVModeCDF(uvCtx)
	if err != nil {
		return err
	}
	w.WriteBinaryCDFTrusted(modeCDF, 0)
	return nil
}

// writePaletteColorsY is write_palette_colors_y: one bit per cache entry
// (while fewer than n colors are taken from the cache) naming the cached
// colors the palette reuses, then the remaining colors in ascending order
// through delta_encode_palette_colors with min_val 1. The palette colors
// must be strictly ascending, as readPaletteColorsY rebuilds them.
func writePaletteColorsY(w *paletteSink, ctx *BlockModeContext, req PaletteModeRequest, res *PaletteModeResult) error {
	n := int(res.YSize)
	colors := res.YColors[:n]
	for i := 1; i < n; i++ {
		if colors[i] <= colors[i-1] {
			return ErrInvalidDecodeState
		}
	}
	var cache [2 * PaletteMaxSize]uint16
	nCache, err := ctx.PaletteYCache(int(req.X4), int(req.Y4), req.HaveTop, req.HaveLeft, &cache)
	if err != nil {
		return err
	}
	var inCache [PaletteMaxSize]bool
	nCached := 0
	for i := 0; i < nCache && nCached < n; i++ {
		use := 0
		for j, c := range colors {
			if c == cache[i] && !inCache[j] {
				inCache[j] = true
				use = 1
				nCached++
				break
			}
		}
		w.WriteBit(use)
	}
	var out [PaletteMaxSize]uint16
	nOut := 0
	for j, c := range colors {
		if !inCache[j] {
			out[nOut] = c
			nOut++
		}
	}
	return deltaEncodePaletteColors(w, out[:nOut], req.BitDepth, 1)
}

// deltaEncodePaletteColors is libaom's delta_encode_palette_colors.
func deltaEncodePaletteColors(w *paletteSink, colors []uint16, bitDepth uint8, minVal int) error {
	num := len(colors)
	if num <= 0 {
		return nil
	}
	w.WriteLiteral(uint32(colors[0]), int(bitDepth))
	if num == 1 {
		return nil
	}
	var deltas [PaletteMaxSize]int
	maxDelta := 0
	for i := 1; i < num; i++ {
		d := int(colors[i]) - int(colors[i-1])
		if d < minVal {
			return ErrInvalidDecodeState
		}
		deltas[i-1] = d
		maxDelta = max(maxDelta, d)
	}
	minBits := int(bitDepth) - 3
	nbits := max(ceilLog2(maxDelta+1-minVal), minBits)
	rng := (1 << bitDepth) - int(colors[0]) - minVal
	w.WriteLiteral(uint32(nbits-minBits), 2)
	for i := 0; i < num-1; i++ {
		w.WriteLiteral(uint32(deltas[i]-minVal), nbits)
		rng -= deltas[i]
		nbits = min(nbits, ceilLog2(rng))
	}
	return nil
}

// ceilLog2 is av1_ceil_log2.
func ceilLog2(n int) int {
	if n < 2 {
		return 0
	}
	return bits.Len(uint(n - 1))
}

// WritePaletteTokens codes the luma color index map of res (palette_tokens):
// the on-screen part of the block in wavefront order, the first index with
// a uniform code and every other through its color-context CDF, the forward
// of ReadPaletteTokens / readPaletteIndices. res.YMap holds the bw x bh map
// at stride bw.
func WritePaletteTokens(w *entropy.Writer, cdfs *IntraModeCDFs, req PaletteModeRequest, res *PaletteModeResult) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	s := paletteSink{w: w}
	return writePaletteTokens(&s, cdfs, req, res)
}

// PaletteTokensCost prices WritePaletteTokens.
func PaletteTokensCost(cdfs *IntraModeCDFs, req PaletteModeRequest, res *PaletteModeResult) (uint32, error) {
	var c paletteSink
	err := writePaletteTokens(&c, cdfs, req, res)
	return c.cost, err
}

func writePaletteTokens(w *paletteSink, cdfs *IntraModeCDFs, req PaletteModeRequest, res *PaletteModeResult) error {
	if cdfs == nil || res == nil || res.UVSize != 0 {
		return ErrInvalidDecodeState
	}
	if res.YSize == 0 {
		return nil
	}
	if res.YMap == nil {
		return ErrInvalidDecodeState
	}
	dims, ok := req.Size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	bw4, bh4 := int(dims.W4), int(dims.H4)
	w4, h4 := paletteOnscreen4(bw4, bh4, req.VisibleW4, req.VisibleH4, 0, 0)
	n := int(res.YSize)
	stride := bw4 * 4
	cols, rows := w4*4, h4*4
	m := res.YMap[:]
	writeUniform(w, uint32(n), uint32(m[0]))
	for i := 1; i < rows+cols-1; i++ {
		start := max(i-rows+1, 0)
		end := min(cols-1, i)
		for j := end; j >= start; j-- {
			row, col := i-j, j
			ctx, order, err := paletteColorIndexContext(m, stride, row, col, n)
			if err != nil {
				return err
			}
			v := m[row*stride+col]
			sym := -1
			for k := range n {
				if order[k] == v {
					sym = k
					break
				}
			}
			if sym < 0 {
				return ErrInvalidDecodeState
			}
			cdf, err := cdfs.PaletteYColorIndexCDF(n, ctx)
			if err != nil {
				return err
			}
			w.WriteCDF(cdf, sym)
		}
	}
	return nil
}

// writeUniform is the forward of Reader.ReadUniform (aom_write_primitive
// quniform's inverse in the decoder's ns(n) form).
func writeUniform(w *paletteSink, n, v uint32) {
	if n <= 1 {
		return
	}
	width := bits.Len32(n)
	m := (uint32(1) << width) - n
	if v < m {
		w.WriteLiteral(v, width-1)
		return
	}
	t := v + m
	w.WriteLiteral(t>>1, width-1)
	w.WriteBit(int(t & 1))
}
