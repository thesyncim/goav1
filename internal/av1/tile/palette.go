package tile

import (
	"fmt"
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

const (
	PaletteMinSize       = 2
	PaletteMaxSize       = 8
	PaletteSizes         = PaletteMaxSize - PaletteMinSize + 1
	PaletteBSizeContexts = 7
	PaletteYModeContexts = 3
	PaletteColorContexts = 5
	MaxPalettePixels     = 64 * 64
)

type PaletteModeResult struct {
	YSize   uint8
	YColors [PaletteMaxSize]uint16
	YMap    *[MaxPalettePixels]uint8
}

type paletteContext struct {
	Size   uint8
	Colors [PaletteMaxSize]uint16
}

func (c *IntraModeCDFs) initPaletteDefaults() error {
	defaultYSize := [PaletteBSizeContexts][PaletteSizes - 1]uint16{
		{7952, 13000, 18149, 21478, 25527, 29241},
		{7139, 11421, 16195, 19544, 23666, 28073},
		{7788, 12741, 17325, 20500, 24315, 28530},
		{8271, 14064, 18246, 21564, 25071, 28533},
		{12725, 19180, 21863, 24839, 27535, 30120},
		{9711, 14888, 16923, 21052, 25661, 27875},
		{14940, 20797, 21678, 24186, 27033, 28999},
	}
	for i := range defaultYSize {
		if err := c.PaletteYSize[i].Init(defaultYSize[i][:]); err != nil {
			return err
		}
	}
	defaultYMode := [PaletteBSizeContexts][PaletteYModeContexts]uint16{
		{31676, 3419, 1261},
		{31912, 2859, 980},
		{31823, 3400, 781},
		{32030, 3561, 904},
		{32309, 7337, 1462},
		{32265, 4015, 1521},
		{32450, 7946, 129},
	}
	for i := range defaultYMode {
		for j := range defaultYMode[i] {
			if err := c.PaletteYMode[i][j].Init([]uint16{defaultYMode[i][j]}); err != nil {
				return err
			}
		}
	}
	defaultColor := [PaletteSizes][PaletteColorContexts][]uint16{
		{
			{28710}, {16384}, {10553}, {27036}, {31603},
		},
		{
			{27877, 30490}, {11532, 25697}, {6544, 30234}, {23018, 28072}, {31915, 32385},
		},
		{
			{25572, 28046, 30045}, {9478, 21590, 27256}, {7248, 26837, 29824}, {19167, 24486, 28349}, {31400, 31825, 32250},
		},
		{
			{24779, 26955, 28576, 30282}, {8669, 20364, 24073, 28093}, {4255, 27565, 29377, 31067}, {19864, 23674, 26716, 29530}, {31646, 31893, 32147, 32426},
		},
		{
			{23132, 25407, 26970, 28435, 30073}, {7443, 17242, 20717, 24762, 27982}, {6300, 24862, 26944, 28784, 30671}, {18916, 22895, 25267, 27435, 29652}, {31270, 31550, 31808, 32059, 32353},
		},
		{
			{23105, 25199, 26464, 27684, 28931, 30318}, {6950, 15447, 18952, 22681, 25567, 28563}, {7560, 23474, 25490, 27203, 28921, 30708}, {18544, 22373, 24457, 26195, 28119, 30045}, {31198, 31451, 31670, 31882, 32123, 32391},
		},
		{
			{21689, 23883, 25163, 26352, 27506, 28827, 30195}, {6892, 15385, 17840, 21606, 24287, 26753, 29204}, {5651, 23182, 25042, 26518, 27982, 29392, 30900}, {19349, 22578, 24418, 25994, 27524, 29031, 30448}, {31028, 31270, 31504, 31705, 31927, 32153, 32392},
		},
	}
	for size := range defaultColor {
		for ctx := range defaultColor[size] {
			if err := c.PaletteYColorIndex[size][ctx].Init(defaultColor[size][ctx]); err != nil {
				return err
			}
		}
	}
	return nil
}

func (c *IntraModeCDFs) PaletteYModeCDF(sizeCtx int, modeCtx int) (*entropy.CDF, error) {
	if c == nil || sizeCtx < 0 || sizeCtx >= PaletteBSizeContexts || modeCtx < 0 || modeCtx >= PaletteYModeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.PaletteYMode[sizeCtx][modeCtx]
	if cdf.Symbols() != 2 {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func (c *IntraModeCDFs) PaletteYSizeCDF(sizeCtx int) (*entropy.CDF, error) {
	if c == nil || sizeCtx < 0 || sizeCtx >= PaletteBSizeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.PaletteYSize[sizeCtx]
	if cdf.Symbols() != PaletteSizes {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func (c *IntraModeCDFs) PaletteYColorIndexCDF(colors int, ctx int) (*entropy.CDF, error) {
	if c == nil || colors < PaletteMinSize || colors > PaletteMaxSize || ctx < 0 || ctx >= PaletteColorContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.PaletteYColorIndex[colors-PaletteMinSize][ctx]
	if cdf.Symbols() != colors {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

func (s *DecodeState) ReadPaletteMode(cdfs *IntraModeCDFs, ctx *BlockModeContext, req PaletteModeRequest, dst *PaletteModeResult, mapScratch *[MaxPalettePixels]uint8) error {
	if s == nil || cdfs == nil || ctx == nil || dst == nil || mapScratch == nil {
		return ErrInvalidDecodeState
	}
	*dst = PaletteModeResult{}
	allowed, err := PaletteAllowed(req.AllowScreenContentTools, req.Size)
	if err != nil || !allowed || req.LumaMode != IntraModeDC {
		return err
	}
	sizeCtx, err := PaletteBSizeContext(req.Size)
	if err != nil {
		return err
	}
	modeCtx, err := ctx.PaletteYModeContext(req.X4, req.Y4, req.HaveTop, req.HaveLeft)
	if err != nil {
		return err
	}
	modeCDF, err := cdfs.PaletteYModeCDF(sizeCtx, modeCtx)
	if err != nil {
		return err
	}
	use, err := s.Reader.ReadCDF(modeCDF)
	if err != nil || use == 0 {
		return err
	}
	sizeCDF, err := cdfs.PaletteYSizeCDF(sizeCtx)
	if err != nil {
		return err
	}
	sizeSym, err := s.Reader.ReadCDF(sizeCDF)
	if err != nil {
		return err
	}
	dst.YSize = uint8(sizeSym + PaletteMinSize)
	if err := s.readPaletteColorsY(ctx, req, dst); err != nil {
		return fmt.Errorf("palette colors y size=%d block=%v x4=%d y4=%d: %w", dst.YSize, req.Size, req.X4, req.Y4, err)
	}
	if err := s.readPaletteColorMapY(cdfs, req, dst, mapScratch); err != nil {
		return fmt.Errorf("palette map y size=%d block=%v x4=%d y4=%d colors=%v: %w", dst.YSize, req.Size, req.X4, req.Y4, dst.YColors, err)
	}
	dst.YMap = mapScratch
	return nil
}

type PaletteModeRequest struct {
	AllowScreenContentTools bool
	Size                    BlockSize
	LumaMode                IntraMode
	X4                      int
	Y4                      int
	HaveTop                 bool
	HaveLeft                bool
	BitDepth                uint8
}

func PaletteAllowed(allowScreenContentTools bool, size BlockSize) (bool, error) {
	dims, ok := size.Dimensions()
	if !ok {
		return false, ErrInvalidDecodeState
	}
	return allowScreenContentTools &&
		dims.W4 >= 2 && dims.H4 >= 2 &&
		int(dims.W4)*4 <= 64 &&
		int(dims.H4)*4 <= 64, nil
}

func PaletteBSizeContext(size BlockSize) (int, error) {
	dims, ok := size.Dimensions()
	if !ok {
		return 0, ErrInvalidDecodeState
	}
	ctx := int(dims.Log2W) + int(dims.Log2H) - 2
	if ctx < 0 || ctx >= PaletteBSizeContexts {
		return 0, ErrInvalidDecodeState
	}
	return ctx, nil
}

func (s *DecodeState) readPaletteColorsY(ctx *BlockModeContext, req PaletteModeRequest, result *PaletteModeResult) error {
	var cache [2 * PaletteMaxSize]uint16
	nCache, err := ctx.PaletteYCache(req.X4, req.Y4, req.HaveTop, req.HaveLeft, &cache)
	if err != nil {
		return err
	}
	n := int(result.YSize)
	idx := 0
	var cached [PaletteMaxSize]uint16
	nCached := 0
	for i := 0; i < nCache && nCached < n; i++ {
		use, err := s.Reader.ReadBit()
		if err != nil {
			return err
		}
		if use != 0 {
			cached[nCached] = cache[i]
			nCached++
		}
	}
	if nCached < n {
		idx = nCached
		v, err := s.Reader.ReadBits(req.BitDepth)
		if err != nil {
			return err
		}
		result.YColors[idx] = uint16(v)
		idx++
		if idx < n {
			minBits := int(req.BitDepth) - 3
			extraBits, err := s.Reader.ReadBits(2)
			if err != nil {
				return err
			}
			bitCount := uint8(minBits + int(extraBits))
			rng := (1 << req.BitDepth) - int(result.YColors[idx-1]) - 1
			for idx < n {
				if rng < 0 {
					return ErrInvalidDecodeState
				}
				delta, err := s.Reader.ReadBits(bitCount)
				if err != nil {
					return err
				}
				value := int(result.YColors[idx-1]) + int(delta) + 1
				maxSample := (1 << req.BitDepth) - 1
				if value > maxSample {
					value = maxSample
				}
				result.YColors[idx] = uint16(value)
				rng -= value - int(result.YColors[idx-1])
				nextBits := bits.Len(uint(rng))
				if rng > 0 {
					nextBits = bits.Len(uint(rng - 1))
				}
				if nextBits < int(bitCount) {
					bitCount = uint8(nextBits)
				}
				idx++
			}
		}
		mergePaletteColors(result.YColors[:], cached[:nCached], n)
	} else {
		copy(result.YColors[:], cached[:n])
	}
	return nil
}

func (s *DecodeState) readPaletteColorMapY(cdfs *IntraModeCDFs, req PaletteModeRequest, result *PaletteModeResult, mapScratch *[MaxPalettePixels]uint8) error {
	dims, ok := req.Size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	cols := int(dims.W4) * 4
	rows := int(dims.H4) * 4
	planeBlockWidth := cols
	planeBlockHeight := rows
	n := int(result.YSize)
	first, err := s.Reader.ReadUniform(uint32(n))
	if err != nil {
		return err
	}
	mapScratch[0] = uint8(first)
	for i := 1; i < rows+cols-1; i++ {
		start := 0
		if i-rows+1 > start {
			start = i - rows + 1
		}
		end := i
		if cols-1 < end {
			end = cols - 1
		}
		for j := end; j >= start; j-- {
			row := i - j
			col := j
			ctx, order, err := paletteColorIndexContext(mapScratch[:], planeBlockWidth, row, col, n)
			if err != nil {
				return err
			}
			cdf, err := cdfs.PaletteYColorIndexCDF(n, ctx)
			if err != nil {
				return err
			}
			sym, err := s.Reader.ReadCDF(cdf)
			if err != nil {
				return err
			}
			mapScratch[row*planeBlockWidth+col] = order[sym]
		}
	}
	for row := rows; row < planeBlockHeight; row++ {
		copy(mapScratch[row*planeBlockWidth:row*planeBlockWidth+planeBlockWidth], mapScratch[(rows-1)*planeBlockWidth:rows*planeBlockWidth])
	}
	return nil
}

func paletteColorIndexContext(colorMap []uint8, stride int, row int, col int, paletteSize int) (int, [PaletteMaxSize]uint8, error) {
	if paletteSize < PaletteMinSize || paletteSize > PaletteMaxSize || (row == 0 && col == 0) {
		return 0, [PaletteMaxSize]uint8{}, ErrInvalidDecodeState
	}
	neighbors := [3]int{-1, -1, -1}
	if col > 0 {
		neighbors[0] = int(colorMap[row*stride+col-1])
	}
	if col > 0 && row > 0 {
		neighbors[1] = int(colorMap[(row-1)*stride+col-1])
	}
	if row > 0 {
		neighbors[2] = int(colorMap[(row-1)*stride+col])
	}
	weights := [3]int{2, 1, 2}
	var scores [PaletteMaxSize]int
	for i, n := range neighbors {
		if n >= 0 {
			scores[n] += weights[i]
		}
	}
	var order [PaletteMaxSize]uint8
	var inverse [PaletteMaxSize]uint8
	for i := 0; i < PaletteMaxSize; i++ {
		order[i] = uint8(i)
		inverse[i] = uint8(i)
	}
	for i := 0; i < 3; i++ {
		maxScore := scores[i]
		maxIdx := i
		for j := i + 1; j < paletteSize; j++ {
			if scores[j] > maxScore {
				maxScore = scores[j]
				maxIdx = j
			}
		}
		if maxIdx != i {
			maxOrder := order[maxIdx]
			for k := maxIdx; k > i; k-- {
				scores[k] = scores[k-1]
				order[k] = order[k-1]
				inverse[order[k]] = uint8(k)
			}
			scores[i] = maxScore
			order[i] = maxOrder
			inverse[order[i]] = uint8(i)
		}
	}
	hash := scores[0] + 2*scores[1] + 2*scores[2]
	lookup := [...]int{-1, -1, 0, -1, -1, 4, 3, 2, 1}
	if hash <= 0 || hash >= len(lookup) || lookup[hash] < 0 {
		return 0, [PaletteMaxSize]uint8{}, ErrInvalidDecodeState
	}
	return lookup[hash], order, nil
}

func mergePaletteColors(colors []uint16, cache []uint16, n int) {
	if len(cache) == 0 {
		return
	}
	var merged [PaletteMaxSize]uint16
	cacheIdx := 0
	transIdx := len(cache)
	for i := 0; i < n; i++ {
		if cacheIdx < len(cache) && (transIdx >= n || cache[cacheIdx] <= colors[transIdx]) {
			merged[i] = cache[cacheIdx]
			cacheIdx++
		} else {
			merged[i] = colors[transIdx]
			transIdx++
		}
	}
	copy(colors, merged[:n])
}

func (c *BlockModeContext) PaletteYModeContext(x4 int, y4 int, haveTop bool, haveLeft bool) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, err
	}
	ctx := 0
	if haveTop && c.AbovePaletteY[x4].Size > 0 {
		ctx++
	}
	if haveLeft && c.LeftPaletteY[y4].Size > 0 {
		ctx++
	}
	return ctx, nil
}

func (c *BlockModeContext) PaletteYCache(x4 int, y4 int, haveTop bool, haveLeft bool, cache *[2 * PaletteMaxSize]uint16) (int, error) {
	if c == nil || cache == nil {
		return 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, err
	}
	var above, left paletteContext
	if haveTop {
		above = c.AbovePaletteY[x4]
	}
	if haveLeft {
		left = c.LeftPaletteY[y4]
	}
	i, j, n := 0, 0, 0
	for i < int(above.Size) && j < int(left.Size) {
		a, l := above.Colors[i], left.Colors[j]
		if l < a {
			paletteAddToCache(cache, &n, l)
			j++
		} else {
			paletteAddToCache(cache, &n, a)
			i++
			if l == a {
				j++
			}
		}
	}
	for i < int(above.Size) {
		paletteAddToCache(cache, &n, above.Colors[i])
		i++
	}
	for j < int(left.Size) {
		paletteAddToCache(cache, &n, left.Colors[j])
		j++
	}
	return n, nil
}

func paletteAddToCache(cache *[2 * PaletteMaxSize]uint16, n *int, value uint16) {
	if *n > 0 && cache[*n-1] == value {
		return
	}
	cache[*n] = value
	*n++
}

func (c *BlockModeContext) MarkPaletteY(size BlockSize, x4 int, y4 int, palette PaletteModeResult) error {
	if c == nil {
		return ErrInvalidDecodeState
	}
	dims, ok := size.Dimensions()
	if !ok || x4 < 0 || y4 < 0 || x4+int(dims.W4) > MaxBlockModeSlots || y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	var ctx paletteContext
	ctx.Size = palette.YSize
	ctx.Colors = palette.YColors
	for i := 0; i < int(dims.W4); i++ {
		c.AbovePaletteY[x4+i] = ctx
	}
	for i := 0; i < int(dims.H4); i++ {
		c.LeftPaletteY[y4+i] = ctx
	}
	return nil
}
