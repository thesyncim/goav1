package entropy

import (
	"math/rand"
	"slices"
	"testing"
	"unsafe"
)

// frozenCoeffTokens2D is the old AC loop from tile/coeff.go at b9e260f4.
// It deliberately retains the per-symbol Cursor calls as an independent
// state/materialization oracle for the fused reader.
func frozenCoeffTokens2D(reader *Cursor, scan []CoeffScanContext, stride int, levels []uint8, base *[42]CDF, br *[21]CDF, dirty, levelDirty *[1024]int16) (nonzero, levelDirtyNext int) {
	nonzero, levelDirtyNext = 1, 1
	for i := len(scan) - 1; i >= 0; i-- {
		p := scan[i]
		pos := int(p.Pos)
		padded := int(p.Padded)
		s1 := padded + stride
		s1p1 := s1 + 1
		s2 := s1 + stride
		p1 := padded + 1
		p2 := padded + 2
		_ = levels[s2]
		_ = levels[p2]
		// Tracked 2D levels store the raw token in bits 0..5 and
		// min(token, 3) in bits 6..7. Summing the five packed neighbors
		// makes the rounded half-sum of clipped levels available in bits
		// 7 and above. If the clipped sum is below 8, the raw sum cannot
		// carry into that field; otherwise the context saturates at 4.
		h := int(levels[s1]) + int(levels[p1]) + int(levels[s1p1])
		l := h + int(levels[s2]) + int(levels[p2])
		ctx := (l + 64) >> 7
		if ctx > 4 {
			ctx = 4
		}
		ctx += int(p.Lower2DOffset)
		level := reader.ReadCDF4UpdateUnchecked(&base[ctx])
		packedLevel := uint8(level * 0x41)
		if level == 3 {
			// Re-read only on the rare high-token path so h does not need
			// to stay live across the CDF update call. The low six bits
			// recover the raw-token sum (at most 45).
			brH := (int(levels[s1]) + int(levels[p1]) + int(levels[s1p1])) & 63
			mag := (brH + 1) >> 1
			if mag > 6 {
				mag = 6
			}
			brCtx := mag + int(p.BR2DOffset)
			extra := reader.ReadCDF4HighTokenUpdateUnchecked(&br[brCtx])
			level += int(extra)
			packedLevel = uint8(level | 0xc0)
		}
		if level != 0 {
			levels[padded] = packedLevel
			levelDirty[levelDirtyNext] = int16(padded)
			levelDirtyNext++
			dirty[nonzero] = int16((level << 10) | pos)
			nonzero++
		}
	}
	return nonzero, levelDirtyNext
}

func TestCoeffScanContextLayout(t *testing.T) {
	var p CoeffScanContext
	if unsafe.Sizeof(p) != 8 || unsafe.Offsetof(p.Pos) != 0 || unsafe.Offsetof(p.Padded) != 2 || unsafe.Offsetof(p.Lower2DOffset) != 4 || unsafe.Offsetof(p.BR2DOffset) != 5 || unsafe.Offsetof(p.LowerEOBCtx) != 6 || unsafe.Offsetof(p.BREOBCtx) != 7 {
		t.Fatal("coefficient scan descriptor layout changed")
	}
}

func TestCoeffTokens2DLocalStateMatchesFrozenLoop(t *testing.T) {
	random := rand.New(rand.NewSource(0xcdf4ac))
	lengths := []int{0, 1, 7, 8, 9, 128, 4096}
	tokens := []int{0, 1, 3, 14, 62, 254, 1022}
	if traceEntropyReads {
		lengths = []int{0, 9}
		tokens = []int{3, 14}
	}
	sawBR, sawLongBR, sawRefill, sawBeyondEnd := false, false, false, false
	for _, length := range lengths {
		for _, n := range tokens {
			for sample := 0; sample < 6; sample++ {
				var base, wantBase [42]CDF
				var br, wantBR [21]CDF
				for k := range base {
					initCoeffTokenTestCDF(&base[k], random, sample, k)
				}
				for k := range br {
					initCoeffTokenTestCDF(&br[k], random, sample, k)
				}
				wantBase, wantBR = base, br
				width := 4
				for width*width < n+2 {
					width *= 2
				}
				stride := width + 4
				scan := make([]CoeffScanContext, n)
				for k := range scan {
					pos := k + 1
					scan[k] = CoeffScanContext{Pos: uint16(pos), Padded: uint16((pos/width)*stride + pos%width), Lower2DOffset: int8(1 + (k%8)*5), BR2DOffset: int8((k % 3) * 7)}
				}
				levels := make([]uint8, stride*(width+4)+2)
				levels[0], levels[len(levels)-1] = 0x5a, 0xa5
				for k := 1; k < len(levels)-1; k++ {
					tok := uint8(random.Intn(16))
					levels[k] = tok | min(tok, uint8(3))<<6
				}
				wantLevels := slices.Clone(levels)
				src := make([]byte, length)
				random.Read(src)
				if sample == 0 {
					clear(src)
				}
				if sample == 1 {
					for k := range src {
						src[k] = 255
					}
				}
				wantSrc := slices.Clone(src)
				// A reader may source bytes from the same backing as its level
				// scratch. Preserve that alias and its refill/write ordering.
				if sample == 5 {
					limit := min(length, len(levels)-2)
					src, wantSrc = levels[1:limit+1], wantLevels[1:limit+1]
				}
				r := NewReader(src)
				wr := NewReader(wantSrc)
				got, want := r.Cursor(), wr.Cursor()
				var dirty, wantDirty, levelDirty, wantLevelDirty [1024]int16
				for k := range dirty {
					dirty[k] = int16(0x5000 + k)
					levelDirty[k] = int16(0x6000 + k)
				}
				wantDirty, wantLevelDirty = dirty, levelDirty
				gotBR, refBR := &br, &wantBR
				// No unsafe alias: the first 21 base rows can also be BR rows.
				if sample == 4 {
					gotBR = (*[21]CDF)(base[:21])
					refBR = (*[21]CDF)(wantBase[:21])
				}
				gn, gl := got.ReadCoeffTokens2DUpdateUnchecked(scan, stride, levels[1:len(levels)-1], &base, gotBR, &dirty, &levelDirty)
				wn, wl := frozenCoeffTokens2D(&want, scan, stride, wantLevels[1:len(wantLevels)-1], &wantBase, refBR, &wantDirty, &wantLevelDirty)
				if gn != wn || gl != wl || dirty != wantDirty || levelDirty != wantLevelDirty || !slices.Equal(levels, wantLevels) || base != wantBase || br != wantBR {
					t.Fatalf("length=%d tokens=%d sample=%d scratch/CDF differs", length, n, sample)
				}
				got.CommitTo(&r)
				want.CommitTo(&wr)
				if r.State() != wr.State() || r.allowCDFUpdate != wr.allowCDFUpdate || !slices.Equal(src, wantSrc) {
					t.Fatalf("length=%d tokens=%d sample=%d reader state differs: got=%+v want=%+v", length, n, sample, r.State(), wr.State())
				}
				for _, v := range dirty[1:gn] {
					level := uint16(v) >> 10
					sawBR = sawBR || level >= 3
					sawLongBR = sawLongBR || level >= 12
				}
				sawRefill = sawRefill || r.pos > 7
				sawBeyondEnd = sawBeyondEnd || r.cnt > 1000
				if levels[0] != 0x5a || levels[len(levels)-1] != 0xa5 {
					t.Fatal("level guards changed")
				}
			}
		}
	}
	if !sawBR || !sawLongBR || (!traceEntropyReads && !sawRefill) || !sawBeyondEnd {
		t.Fatalf("coverage BR=%v longBR=%v refill=%v pastEnd=%v", sawBR, sawLongBR, sawRefill, sawBeyondEnd)
	}
}

func initCoeffTokenTestCDF(c *CDF, random *rand.Rand, sample, row int) {
	c.symbols = 4
	switch sample {
	case 0:
		c.values[0], c.values[1], c.values[2] = 32767, 32766, 32765
	case 1:
		c.values[0], c.values[1], c.values[2] = 3, 2, 1
	default:
		v := []uint16{uint16(random.Intn(32768)), uint16(random.Intn(32768)), uint16(random.Intn(32768))}
		slices.Sort(v)
		c.values[0], c.values[1], c.values[2] = v[2], v[1], v[0]
	}
	c.values[4] = []uint16{0, 15, 16, 31, 32}[row%5]
}

// Selected lower/upper range thresholds, equality, each count-rate transition,
// and negative-count refill boundaries must match the unchanged symbol core.
func TestCoeffTokens2DThresholdStates(t *testing.T) {
	ranges := []uint16{32768, 33024, 65412, 65535}
	counts := []int16{0, 1, 2, 11, 12, 48, 16384}
	lengths := []int{0, 1, 7, 8, 9}
	if traceEntropyReads {
		ranges = ranges[:1]
		counts = counts[:2]
		lengths = lengths[:1]
	}
	scan := []CoeffScanContext{{Pos: 1, Padded: 1}}
	for _, rng := range ranges {
		for _, adapt := range []uint16{0, 15, 16, 31, 32} {
			for _, shape := range [][3]uint16{{32767, 32766, 32765}, {24576, 16384, 8192}, {3, 2, 1}, {0, 0, 0}} {
				codes := []uint32{0, uint32(rng) - 1}
				for i, v := range shape {
					split := ((uint32(rng) >> 8) * uint32(v>>6) >> 1) + uint32(4*(3-i))
					for _, x := range []uint32{split - 1, split, split + 1} {
						if x < uint32(rng) {
							codes = append(codes, x)
						}
					}
				}
				for _, coded := range codes {
					for _, cnt := range counts {
						for _, length := range lengths {
							src := make([]byte, length)
							for i := range src {
								src[i] = byte(i*79 + 0xa5)
							}
							ref := NewReader(src)
							ref.pos = 0
							ref.rng = rng
							ref.cnt = cnt
							ref.dif = uint64(coded)<<48 | 0x123456789abc
							ref.tellOffs = readerInitTellOffs
							got, want := ref.Cursor(), ref.Cursor()
							var base, refBase [42]CDF
							var br, refBR [21]CDF
							for i := range base {
								base[i].InitUniform(4)
							}
							for i := range br {
								br[i].InitUniform(4)
							}
							base[0].values[0], base[0].values[1], base[0].values[2], base[0].values[4] = shape[0], shape[1], shape[2], adapt
							refBase, refBR = base, br
							var levels, refLevels [24]uint8
							var dirty, refDirty, levelDirty, refLevelDirty [1024]int16
							gn, gl := got.ReadCoeffTokens2DUpdateUnchecked(scan, 4, levels[:], &base, &br, &dirty, &levelDirty)
							wn, wl := frozenCoeffTokens2D(&want, scan, 4, refLevels[:], &refBase, &refBR, &refDirty, &refLevelDirty)
							var gr, wr Reader
							got.CommitTo(&gr)
							want.CommitTo(&wr)
							if gr.State() != wr.State() || base != refBase || br != refBR || gn != wn || gl != wl || dirty != refDirty || levelDirty != refLevelDirty || levels != refLevels {
								t.Fatalf("rng=%d count=%d adapt=%d coded=%d len=%d threshold mismatch", rng, cnt, adapt, coded, length)
							}
						}
					}
				}
			}
		}
	}
}
