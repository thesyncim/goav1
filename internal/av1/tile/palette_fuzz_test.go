package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

// palettePaletteCapableSizes enumerates the block sizes that PaletteAllowed
// accepts. Restricting fuzz inputs to these keeps the input space focused on
// paths that actually exercise palette decode.
var palettePaletteCapableSizes = [...]BlockSize{
	BlockSize8x8, BlockSize8x16, BlockSize16x8, BlockSize16x16,
	BlockSize16x32, BlockSize32x16, BlockSize32x32, BlockSize32x64,
	BlockSize64x32, BlockSize64x64,
}

// FuzzReadPaletteMode drives the full palette mode symbol decode path. The
// goal is to expose any panic, out-of-range index, or unchecked allocation in
// the readers; success is "returned cleanly" or "returned a sentinel error".
func FuzzReadPaletteMode(f *testing.F) {
	f.Add([]byte{0x00, 0xff, 0x55, 0xaa}, uint8(0), uint8(8), uint8(0), uint8(0), uint8(0), uint8(0), uint8(0))
	f.Add([]byte{0xff, 0x00, 0xa5, 0x5a, 0x12, 0x34, 0x56, 0x78}, uint8(3), uint8(10), uint8(2), uint8(4), uint8(0), uint8(3), uint8(1))
	f.Add([]byte{0x4d, 0x21, 0xc3, 0x7e, 0x90, 0x1a, 0xbc, 0xde, 0x55, 0xaa}, uint8(7), uint8(12), uint8(6), uint8(8), uint8(2), uint8(7), uint8(3))

	f.Fuzz(func(t *testing.T, payload []byte, rawSize uint8, rawBitDepth uint8, rawX4 uint8, rawY4 uint8, rawChroma uint8, flagBits uint8, rawAboveSize uint8) {
		if len(payload) == 0 || len(payload) > 256 {
			return
		}
		size := palettePaletteCapableSizes[int(rawSize)%len(palettePaletteCapableSizes)]
		dims, ok := size.Dimensions()
		if !ok {
			return
		}
		if int(dims.W4) > MaxBlockModeSlots || int(dims.H4) > MaxBlockModeSlots {
			return
		}
		bitDepths := [3]uint8{8, 10, 12}
		bitDepth := bitDepths[int(rawBitDepth)%len(bitDepths)]
		x4 := int(rawX4) % (MaxBlockModeSlots - int(dims.W4) + 1)
		y4 := int(rawY4) % (MaxBlockModeSlots - int(dims.H4) + 1)
		if x4 < 0 || y4 < 0 {
			return
		}

		var cdfs IntraModeCDFs
		if err := cdfs.InitDefault(); err != nil {
			t.Fatal(err)
		}

		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: len(payload)}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}

		// Pre-seed a neighbor palette context so the cache-merge path
		// gets exercised. Sizes/colors are bounded to the legal range.
		ctx := &BlockModeContext{}
		aboveSize := uint8(int(rawAboveSize) % (PaletteMaxSize + 1))
		neighbor := PaletteModeResult{YSize: aboveSize, UVSize: aboveSize}
		for i := range aboveSize {
			value := uint16(i+1) << (bitDepth - 4)
			neighbor.YColors[i] = value
			neighbor.UColors[i] = value
			neighbor.VColors[i] = value
		}
		if aboveSize > 0 && x4 >= int(dims.W4) {
			if err := ctx.MarkPaletteY(size, x4-int(dims.W4), y4, neighbor); err == nil {
				_ = ctx.MarkPaletteUV(size, x4-int(dims.W4), y4, neighbor)
			}
		}

		haveTop := flagBits&1 != 0
		haveLeft := flagBits&2 != 0
		hasChroma := flagBits&4 != 0
		chromaValid := flagBits&8 != 0
		allowScreen := flagBits&16 != 0
		mono := flagBits&32 != 0
		ssX := flagBits&64 != 0
		ssY := flagBits&128 != 0

		req := PaletteModeRequest{
			AllowScreenContentTools: allowScreen,
			Size:                    size,
			LumaMode:                IntraModeDC,
			X4:                      x4,
			Y4:                      y4,
			HaveTop:                 haveTop,
			HaveLeft:                haveLeft,
			BitDepth:                bitDepth,
			Color: parser.ColorConfig{
				MonoChrome:   mono,
				BitDepth:     bitDepth,
				SubsamplingX: ssX,
				SubsamplingY: ssY,
			},
			ChromaMode:      ChromaIntraMode(rawChroma % 13),
			ChromaModeValid: chromaValid,
			HasChroma:       hasChroma,
		}

		var result PaletteModeResult
		var scratch PaletteModeScratch
		err := state.ReadPaletteMode(&cdfs, ctx, req, &result, &scratch)
		if err != nil {
			// Any returned error is acceptable so long as no panic
			// fires. Reader/CDF/decode-state errors are all valid
			// outcomes for arbitrary input.
			_ = err
			return
		}

		// Validate the result shape if a palette was decoded.
		if result.YSize != 0 {
			if result.YSize < PaletteMinSize || result.YSize > PaletteMaxSize {
				t.Fatalf("YSize=%d out of range", result.YSize)
			}
			maxSample := uint16((1 << bitDepth) - 1)
			for i := uint8(0); i < result.YSize; i++ {
				if result.YColors[i] > maxSample {
					t.Fatalf("YColors[%d]=%d exceeds bitdepth max %d", i, result.YColors[i], maxSample)
				}
			}
			if result.YMap != nil {
				for _, v := range result.YMap {
					if v >= result.YSize {
						t.Fatalf("YMap entry=%d >= size=%d", v, result.YSize)
					}
				}
			}
		}
		if result.UVSize != 0 {
			if result.UVSize < PaletteMinSize || result.UVSize > PaletteMaxSize {
				t.Fatalf("UVSize=%d out of range", result.UVSize)
			}
			if result.UVMap != nil {
				for _, v := range result.UVMap {
					if v >= result.UVSize {
						t.Fatalf("UVMap entry=%d >= size=%d", v, result.UVSize)
					}
				}
			}
		}
	})
}

// FuzzPaletteColorIndexContext explores every coordinate, palette size, and
// neighbor coloring; the routine must either return a sentinel error or a
// valid (ctx, order) pair without panicking.
func FuzzPaletteColorIndexContext(f *testing.F) {
	f.Add(make([]byte, 16), uint8(4), uint8(0), uint8(0), uint8(2))
	f.Add([]byte{0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, uint8(4), uint8(1), uint8(1), uint8(3))
	f.Add(make([]byte, 64), uint8(8), uint8(7), uint8(7), uint8(8))

	f.Fuzz(func(t *testing.T, raw []byte, rawStride uint8, rawRow uint8, rawCol uint8, rawSize uint8) {
		if len(raw) == 0 || len(raw) > 4096 {
			return
		}
		stride := int(rawStride%32) + 1
		row := int(rawRow) % 16
		col := int(rawCol) % 16
		paletteSize := int(rawSize)
		if row*stride+col >= len(raw) {
			return
		}
		// Cap neighbor values to a legal palette index range; out-of-
		// range neighbor values would mean a corrupt prior decode and
		// must produce ErrInvalidDecodeState, not a panic.
		colorMap := make([]uint8, len(raw))
		for i, b := range raw {
			colorMap[i] = b % PaletteMaxSize
		}
		ctx, order, err := paletteColorIndexContext(colorMap, stride, row, col, paletteSize)
		if err != nil {
			if !errors.Is(err, ErrInvalidDecodeState) {
				t.Fatalf("unexpected err=%v", err)
			}
			return
		}
		if ctx < 0 || ctx >= PaletteColorContexts {
			t.Fatalf("ctx=%d out of range", ctx)
		}
		if paletteSize < PaletteMinSize || paletteSize > PaletteMaxSize {
			t.Fatalf("paletteSize=%d should have been rejected", paletteSize)
		}
		// Order must be a permutation of 0..PaletteMaxSize-1.
		var seen [PaletteMaxSize]bool
		for i := range PaletteMaxSize {
			v := order[i]
			if int(v) >= PaletteMaxSize {
				t.Fatalf("order[%d]=%d out of range", i, v)
			}
			if seen[v] {
				t.Fatalf("order has duplicate value %d", v)
			}
			seen[v] = true
		}
	})
}

// FuzzPaletteCache fuzzes the cache-merge routine for the Y and UV neighbor
// palette caches. Output must be strictly increasing with bounded length.
func FuzzPaletteCache(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint64(0), uint64(0), uint8(0), uint8(0), uint8(3))
	f.Add(uint8(4), uint8(4), uint64(0x1122334455667788), uint64(0x99aabbccddeeff00), uint8(7), uint8(3), uint8(3))
	f.Add(uint8(8), uint8(8), uint64(0xffffffffffffffff), uint64(0x0102030405060708), uint8(31), uint8(31), uint8(3))

	f.Fuzz(func(t *testing.T, aboveSize uint8, leftSize uint8, abovePack uint64, leftPack uint64, rawX uint8, rawY uint8, flags uint8) {
		aSize := aboveSize % (PaletteMaxSize + 1)
		lSize := leftSize % (PaletteMaxSize + 1)
		x4 := int(rawX) % MaxBlockModeSlots
		y4 := int(rawY) % MaxBlockModeSlots

		buildResult := func(size uint8, pack uint64) PaletteModeResult {
			var result PaletteModeResult
			result.YSize = size
			result.UVSize = size
			// Generate strictly-increasing color sequences to mimic
			// libaom's contract that palette colors are sorted.
			prev := uint16(0)
			for i := range size {
				step := uint16((pack>>(uint(i)*8))&0xff) + 1
				prev += step
				if prev > 0x0fff {
					prev = 0x0fff
				}
				result.YColors[i] = prev
				result.UColors[i] = prev
				result.VColors[i] = prev
			}
			return result
		}

		above := buildResult(aSize, abovePack)
		left := buildResult(lSize, leftPack)

		var ctx BlockModeContext
		const block = BlockSize8x8
		dims, ok := block.Dimensions()
		if !ok {
			t.Fatal("BlockSize8x8 dimensions missing")
		}
		// Cap the target slot so neighbor marks always fit. Both
		// neighbor and the current slot need W4 / H4 headroom.
		if x4+int(dims.W4) > MaxBlockModeSlots {
			x4 = MaxBlockModeSlots - int(dims.W4)
		}
		if y4+int(dims.H4) > MaxBlockModeSlots {
			y4 = MaxBlockModeSlots - int(dims.H4)
		}
		// Mark an above neighbor one row up if it fits.
		if y4 >= int(dims.H4) {
			if err := ctx.MarkPaletteY(block, x4, y4-int(dims.H4), above); err != nil {
				t.Fatal(err)
			}
			if err := ctx.MarkPaletteUV(block, x4, y4-int(dims.H4), above); err != nil {
				t.Fatal(err)
			}
		}
		// Mark a left neighbor one column over if it fits.
		if x4 >= int(dims.W4) {
			if err := ctx.MarkPaletteY(block, x4-int(dims.W4), y4, left); err != nil {
				t.Fatal(err)
			}
			if err := ctx.MarkPaletteUV(block, x4-int(dims.W4), y4, left); err != nil {
				t.Fatal(err)
			}
		}

		haveTop := flags&1 != 0
		haveLeft := flags&2 != 0

		var cache [2 * PaletteMaxSize]uint16
		n, err := ctx.PaletteYCache(x4, y4, haveTop, haveLeft, &cache)
		if err != nil {
			if !errors.Is(err, ErrInvalidDecodeState) {
				t.Fatalf("Y cache err=%v", err)
			}
			return
		}
		if n < 0 || n > 2*PaletteMaxSize {
			t.Fatalf("Y cache len=%d out of range", n)
		}
		for i := 1; i < n; i++ {
			if cache[i] < cache[i-1] {
				t.Fatalf("Y cache not sorted at %d: %d < %d", i, cache[i], cache[i-1])
			}
			if cache[i] == cache[i-1] {
				t.Fatalf("Y cache has adjacent duplicate at %d: %d", i, cache[i])
			}
		}

		n2, err := ctx.PaletteUVCache(x4, y4, haveTop, haveLeft, &cache)
		if err != nil {
			if !errors.Is(err, ErrInvalidDecodeState) {
				t.Fatalf("UV cache err=%v", err)
			}
			return
		}
		if n2 < 0 || n2 > 2*PaletteMaxSize {
			t.Fatalf("UV cache len=%d out of range", n2)
		}
		for i := 1; i < n2; i++ {
			if cache[i] < cache[i-1] {
				t.Fatalf("UV cache not sorted at %d: %d < %d", i, cache[i], cache[i-1])
			}
			if cache[i] == cache[i-1] {
				t.Fatalf("UV cache has adjacent duplicate at %d: %d", i, cache[i])
			}
		}
	})
}
