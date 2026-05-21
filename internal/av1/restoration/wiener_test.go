package restoration

import (
	"errors"
	"slices"
	"testing"
)

func TestDefaultWienerInfoMatchesLibaom(t *testing.T) {
	want := WienerFilter{3, -7, 15, -22, 15, -7, 3, 0}
	got := DefaultWienerInfo()
	if got.VFilter != want || got.HFilter != want {
		t.Fatalf("DefaultWienerInfo=%+v want %+v", got, want)
	}
	if WienerTap0Min != -5 || WienerTap0Max != 10 ||
		WienerTap1Min != -23 || WienerTap1Max != 8 ||
		WienerTap2Min != -17 || WienerTap2Max != 46 {
		t.Fatalf("tap ranges mismatch")
	}
	roundTests := map[uint8][2]int{
		8:  {3, 11},
		10: {3, 11},
		12: {5, 9},
	}
	for bitDepth, want := range roundTests {
		got0, got1 := wienerRounds(int(bitDepth))
		if got0 != want[0] || got1 != want[1] {
			t.Fatalf("bitDepth=%d rounds=%d,%d want %d,%d", bitDepth, got0, got1, want[0], want[1])
		}
	}
	if !validWienerFilter(NewWienerFilter(WienerTap0Min, WienerTap1Min, WienerTap2Min)) ||
		!validWienerFilter(NewWienerFilter(WienerTap0Max, WienerTap1Max, WienerTap2Max)) {
		t.Fatalf("tap range endpoints should be valid")
	}
}

func TestWienerRestorationPreservesConstantBlocks(t *testing.T) {
	filters := []WienerInfo{
		DefaultWienerInfo(),
		{VFilter: NewWienerFilter(-3, -11, 22), HFilter: NewWienerFilter(6, -9, 18)},
	}
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, size := range []struct {
			width  int
			height int
		}{{1, 1}, {17, 13}, {64, 64}} {
			for _, info := range filters {
				stride := size.width + 2*WienerHalfwin
				origin := WienerHalfwin*stride + WienerHalfwin
				src := make([]uint16, stride*(size.height+2*WienerHalfwin))
				for i := range src {
					src[i] = max / 3
				}
				dst := make([]uint16, size.width*size.height)
				scratchLen, err := WienerScratchLen(size.width, size.height)
				if err != nil {
					t.Fatal(err)
				}
				scratch := make([]uint16, scratchLen)
				if err := ApplyWienerRestoration(src, stride, origin, dst, size.width, size.width, size.height, info, bitDepth, scratch); err != nil {
					t.Fatalf("bitDepth=%d size=%dx%d: %v", bitDepth, size.width, size.height, err)
				}
				for i, got := range dst {
					if got != max/3 {
						t.Fatalf("bitDepth=%d size=%dx%d dst[%d]=%d want %d", bitDepth, size.width, size.height, i, got, max/3)
					}
				}
			}
		}
	}
}

func TestApplyWienerRestorationMatchesReferenceCorpus(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x57)
	for _, tc := range []struct {
		name     string
		bitDepth uint8
		width    int
		height   int
		dstPad   int
		info     WienerInfo
	}{
		{name: "default-8bit", bitDepth: 8, width: 8, height: 8, dstPad: 0, info: DefaultWienerInfo()},
		{name: "custom-10bit", bitDepth: 10, width: 13, height: 9, dstPad: 3, info: WienerInfo{
			VFilter: NewWienerFilter(-4, -13, 29),
			HFilter: NewWienerFilter(7, -5, 13),
		}},
		{name: "large-12bit", bitDepth: 12, width: 64, height: 31, dstPad: 5, info: WienerInfo{
			VFilter: NewWienerFilter(WienerTap0Max, WienerTap1Min, 21),
			HFilter: NewWienerFilter(WienerTap0Min, WienerTap1Max, WienerTap2Max),
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			max := uint16((1 << tc.bitDepth) - 1)
			stride := tc.width + 2*WienerHalfwin + 4
			origin := WienerHalfwin*stride + WienerHalfwin + 2
			src := make([]uint16, stride*(tc.height+2*WienerHalfwin))
			for i := range src {
				src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
			}
			dstStride := tc.width + tc.dstPad
			dst := make([]uint16, dstStride*tc.height)
			for i := range dst {
				dst[i] = max
			}
			scratchLen, err := WienerScratchLen(tc.width, tc.height)
			if err != nil {
				t.Fatal(err)
			}
			scratch := make([]uint16, scratchLen)
			if err := ApplyWienerRestoration(src, stride, origin, dst, dstStride, tc.width, tc.height, tc.info, tc.bitDepth, scratch); err != nil {
				t.Fatal(err)
			}
			for row := 0; row < tc.height; row++ {
				for col := 0; col < tc.width; col++ {
					want := referenceWienerPixel(src, stride, origin, row, col, tc.info, tc.bitDepth)
					if got := dst[row*dstStride+col]; got != want {
						t.Fatalf("row=%d col=%d got=%d want=%d", row, col, got, want)
					}
				}
				for col := tc.width; col < dstStride; col++ {
					if got := dst[row*dstStride+col]; got != max {
						t.Fatalf("padding row=%d col=%d overwritten with %d", row, col, got)
					}
				}
			}
		})
	}
}

func TestWienerScratchLenAndAllocs(t *testing.T) {
	scratchLen, err := WienerScratchLen(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	if scratchLen != 64*(64+2*WienerHalfwin) {
		t.Fatalf("scratchLen=%d", scratchLen)
	}
	scratch := make([]uint16, scratchLen)
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(64+2*WienerHalfwin))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0x99)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	info := WienerInfo{
		VFilter: NewWienerFilter(-5, 8, 46),
		HFilter: NewWienerFilter(10, -23, -17),
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplyWienerRestoration(src, stride, origin, dst, 64, 64, 64, info, 12, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyWienerRestoration allocated: %f", allocs)
	}
}

func TestApplyWienerRestorationRejectsInvalidInputs(t *testing.T) {
	scratchLen, err := WienerScratchLen(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]uint16, scratchLen)
	stride := 8 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(8+2*WienerHalfwin))
	dst := make([]uint16, 8*8)
	badFilter := DefaultWienerInfo()
	badFilter.HFilter[7] = 1
	tests := []struct {
		name string
		fn   func() error
	}{
		{name: "bad-size", fn: func() error {
			return ApplyWienerRestoration(src, stride, origin, dst, 8, 0, 8, DefaultWienerInfo(), 8, scratch)
		}},
		{name: "bad-depth", fn: func() error {
			return ApplyWienerRestoration(src, stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), 9, scratch)
		}},
		{name: "bad-filter", fn: func() error {
			return ApplyWienerRestoration(src, stride, origin, dst, 8, 8, 8, badFilter, 8, scratch)
		}},
		{name: "short-src", fn: func() error {
			return ApplyWienerRestoration(src[:origin], stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), 8, scratch)
		}},
		{name: "short-dst", fn: func() error {
			return ApplyWienerRestoration(src, stride, origin, dst[:8], 8, 8, 8, DefaultWienerInfo(), 8, scratch)
		}},
		{name: "short-scratch", fn: func() error {
			return ApplyWienerRestoration(src, stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), 8, scratch[:scratchLen-1])
		}},
		{name: "sample-out-of-range", fn: func() error {
			bad := slices.Clone(src)
			bad[origin] = 256
			return ApplyWienerRestoration(bad, stride, origin, dst, 8, 8, 8, DefaultWienerInfo(), 8, scratch)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.fn(); !errors.Is(err, ErrInvalidRestoration) {
				t.Fatalf("err=%v want %v", err, ErrInvalidRestoration)
			}
		})
	}
}

func FuzzApplyWienerRestoration(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), int16(3), int16(-7), int16(15), []byte{0, 64, 128, 255})
	f.Add(uint8(31), uint8(17), uint8(2), int16(-5), int16(8), int16(46), []byte{255, 1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawDepth uint8, rawTap0 int16, rawTap1 int16, rawTap2 int16, data []byte) {
		width := int(rawW%32) + 1
		height := int(rawH%32) + 1
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		stride := width + 2*WienerHalfwin
		origin := WienerHalfwin*stride + WienerHalfwin
		src := make([]uint16, stride*(height+2*WienerHalfwin))
		for i := range src {
			src[i] = fuzzRestorationSample(data, i, max)
		}
		dst := make([]uint16, width*height)
		scratchLen, err := WienerScratchLen(width, height)
		if err != nil {
			t.Fatalf("WienerScratchLen err=%v", err)
		}
		scratch := make([]uint16, scratchLen)
		info := WienerInfo{
			VFilter: NewWienerFilter(fuzzWienerTap(rawTap0, WienerTap0Min, WienerTap0Max), fuzzWienerTap(rawTap1, WienerTap1Min, WienerTap1Max), fuzzWienerTap(rawTap2, WienerTap2Min, WienerTap2Max)),
			HFilter: NewWienerFilter(fuzzWienerTap(rawTap2, WienerTap0Min, WienerTap0Max), fuzzWienerTap(rawTap0, WienerTap1Min, WienerTap1Max), fuzzWienerTap(rawTap1, WienerTap2Min, WienerTap2Max)),
		}
		if err := ApplyWienerRestoration(src, stride, origin, dst, width, width, height, info, bitDepth, scratch); err != nil {
			t.Fatalf("ApplyWienerRestoration err=%v", err)
		}
		for i, sample := range dst {
			if sample > max {
				t.Fatalf("dst[%d]=%d max=%d", i, sample, max)
			}
		}
		if len(dst) > 0 {
			want := referenceWienerPixel(src, stride, origin, height/2, width/2, info, bitDepth)
			if got := dst[(height/2)*width+width/2]; got != want {
				t.Fatalf("middle got=%d want=%d", got, want)
			}
		}
	})
}

func BenchmarkApplyWienerRestoration(b *testing.B) {
	scratchLen, err := WienerScratchLen(64, 64)
	if err != nil {
		b.Fatal(err)
	}
	scratch := make([]uint16, scratchLen)
	stride := 64 + 2*WienerHalfwin
	origin := WienerHalfwin*stride + WienerHalfwin
	src := make([]uint16, stride*(64+2*WienerHalfwin))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed ^ 0xabc)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	info := DefaultWienerInfo()
	b.ReportAllocs()
	b.SetBytes(64 * 64 * 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ApplyWienerRestoration(src, stride, origin, dst, 64, 64, 64, info, 12, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func referenceWienerPixel(src []uint16, stride int, origin int, row int, col int, info WienerInfo, bitDepth uint8) uint16 {
	max := int32((1 << bitDepth) - 1)
	round0, round1 := referenceWienerRounds(int(bitDepth))
	center := referenceWienerHorizontal(src, stride, origin, row, col, info.HFilter, int(bitDepth), round0)
	sum := int32(center)<<WienerFilterBits - int32(1<<(int(bitDepth)+round1-1))
	for k := 0; k < WienerWin; k++ {
		v := referenceWienerHorizontal(src, stride, origin, row-WienerHalfwin+k, col, info.HFilter, int(bitDepth), round0)
		sum += int32(v) * int32(info.VFilter[k])
	}
	return uint16(referenceClamp(referenceRoundPowerOfTwo(sum, round1), 0, max))
}

func referenceWienerHorizontal(src []uint16, stride int, origin int, row int, col int, filter WienerFilter, bitDepth int, round0 int) uint16 {
	limit := int32(1 << (bitDepth + 1 + WienerFilterBits - round0))
	sum := int32(src[origin+row*stride+col])<<WienerFilterBits + int32(1<<(bitDepth+WienerFilterBits-1))
	for k := 0; k < WienerWin; k++ {
		sum += int32(src[origin+row*stride+col-WienerHalfwin+k]) * int32(filter[k])
	}
	return uint16(referenceClamp(referenceRoundPowerOfTwo(sum, round0), 0, limit-1))
}

func referenceWienerRounds(bitDepth int) (int, int) {
	round0 := 3
	round1 := 11
	if bitDepth == 12 {
		round0 = 5
		round1 = 9
	}
	return round0, round1
}

func referenceRoundPowerOfTwo(v int32, bits int) int32 {
	if bits <= 0 {
		return v
	}
	return (v + int32(1<<(bits-1))) >> bits
}

func referenceClamp(v int32, lo int32, hi int32) int32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func fuzzWienerTap(raw int16, min int, max int) int16 {
	span := max - min + 1
	v := int(raw) % span
	if v < 0 {
		v += span
	}
	return int16(min + v)
}
