package dsp

import (
	"slices"
	"testing"
)

const libaomBlendDeterministicSeed = 0xbaba

func TestBlendA64MaskKnownVector(t *testing.T) {
	src0 := []uint16{
		0, 64, 128, 255,
		10, 20, 30, 40,
		200, 180, 160, 140,
		255, 0, 255, 0,
	}
	src1 := []uint16{
		255, 128, 64, 0,
		40, 30, 20, 10,
		0, 32, 64, 96,
		0, 255, 0, 255,
	}
	mask := []uint8{
		64, 32, 0, 16,
		48, 24, 8, 56,
		1, 63, 31, 33,
		0, 64, 32, 32,
	}
	dst := make([]uint16, 16)
	if err := BlendA64Mask(dst, 4, src0, 4, src1, 4, mask, 4, 4, 4, false, false, 8); err != nil {
		t.Fatal(err)
	}
	want := []uint16{
		0, 96, 64, 64,
		18, 26, 21, 36,
		3, 178, 111, 119,
		0, 0, 128, 128,
	}
	if !slices.Equal(dst, want) {
		t.Fatalf("dst=%v want %v", dst, want)
	}
}

func TestBlendA64MaskMatchesLibaomCorpus(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		rnd := newBlendRandom(libaomBlendDeterministicSeed)
		for _, width := range []int{4, 8, 16, 32, 64} {
			for _, height := range []int{4, 8, 16, 32, 64} {
				for _, subX := range []bool{false, true} {
					for _, subY := range []bool{false, true} {
						for iter := range 8 {
							dstStride := width + rnd.pseudoUniform(17)
							src0Stride := width + rnd.pseudoUniform(17)
							src1Stride := width + rnd.pseudoUniform(17)
							mStride := maskWidth(width, subX) + rnd.pseudoUniform(17)
							dst := make([]uint16, dstStride*height)
							src0 := make([]uint16, src0Stride*height)
							src1 := make([]uint16, src1Stride*height)
							mask := make([]uint8, mStride*maskHeight(height, subY))
							fillBlendSamples(src0, max, rnd)
							fillBlendSamples(src1, max, rnd)
							fillBlendMask(mask, rnd)
							want := slices.Clone(dst)
							blendA64MaskLibaomReference(want, dstStride, src0, src0Stride, src1, src1Stride, mask, mStride, width, height, subX, subY)
							if err := BlendA64Mask(dst, dstStride, src0, src0Stride, src1, src1Stride, mask, mStride, width, height, subX, subY, bitDepth); err != nil {
								t.Fatalf("bitDepth=%d size=%dx%d subX=%t subY=%t iter=%d: %v", bitDepth, width, height, subX, subY, iter, err)
							}
							if !slices.Equal(dst, want) {
								t.Fatalf("bitDepth=%d size=%dx%d subX=%t subY=%t iter=%d mismatch", bitDepth, width, height, subX, subY, iter)
							}
						}
					}
				}
			}
		}
	}
}

func TestBlendA64MaskSupportsDestinationAliasing(t *testing.T) {
	src0 := []uint16{0, 64, 128, 255}
	src1 := []uint16{255, 128, 64, 0}
	mask := []uint8{64, 32, 16, 0}
	want := slices.Clone(src0)
	blendA64MaskLibaomReference(want, 2, src0, 2, src1, 2, mask, 2, 2, 2, false, false)
	got := slices.Clone(src0)
	if err := BlendA64Mask(got, 2, got, 2, src1, 2, mask, 2, 2, 2, false, false, 8); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("dst/src0 alias got=%v want %v", got, want)
	}

	want = slices.Clone(src1)
	blendA64MaskLibaomReference(want, 2, src0, 2, src1, 2, mask, 2, 2, 2, false, false)
	got = slices.Clone(src1)
	if err := BlendA64Mask(got, 2, src0, 2, got, 2, mask, 2, 2, 2, false, false, 8); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("dst/src1 alias got=%v want %v", got, want)
	}
}

func TestBlendA64MaskRejectsInvalidInputs(t *testing.T) {
	dst := make([]uint16, 16)
	src0 := make([]uint16, 16)
	src1 := make([]uint16, 16)
	mask := make([]uint8, 16)
	tests := []struct {
		name     string
		dst      []uint16
		dstStr   int
		src0     []uint16
		src0Str  int
		src1     []uint16
		src1Str  int
		mask     []uint8
		maskStr  int
		width    int
		height   int
		subX     bool
		subY     bool
		bitDepth uint8
	}{
		{name: "bad-bitdepth", dst: dst, dstStr: 4, src0: src0, src0Str: 4, src1: src1, src1Str: 4, mask: mask, maskStr: 4, width: 4, height: 4, bitDepth: 9},
		{name: "non-power-width", dst: dst, dstStr: 4, src0: src0, src0Str: 4, src1: src1, src1Str: 4, mask: mask, maskStr: 4, width: 3, height: 4, bitDepth: 8},
		{name: "short-dst", dst: dst[:8], dstStr: 4, src0: src0, src0Str: 4, src1: src1, src1Str: 4, mask: mask, maskStr: 4, width: 4, height: 4, bitDepth: 8},
		{name: "short-mask-subsampled", dst: dst, dstStr: 4, src0: src0, src0Str: 4, src1: src1, src1Str: 4, mask: mask, maskStr: 4, width: 4, height: 4, subX: true, bitDepth: 8},
		{name: "mask-out-of-range", dst: dst, dstStr: 4, src0: src0, src0Str: 4, src1: src1, src1Str: 4, mask: []uint8{65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, maskStr: 4, width: 4, height: 4, bitDepth: 8},
		{name: "sample-out-of-range", dst: dst, dstStr: 4, src0: []uint16{256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, src0Str: 4, src1: src1, src1Str: 4, mask: mask, maskStr: 4, width: 4, height: 4, bitDepth: 8},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := BlendA64Mask(tt.dst, tt.dstStr, tt.src0, tt.src0Str, tt.src1, tt.src1Str, tt.mask, tt.maskStr, tt.width, tt.height, tt.subX, tt.subY, tt.bitDepth); err != ErrInvalidBlock {
				t.Fatalf("err=%v want %v", err, ErrInvalidBlock)
			}
		})
	}
}

func TestBlendA64MaskAllocs(t *testing.T) {
	dst := make([]uint16, 64*64)
	src0 := make([]uint16, 64*64)
	src1 := make([]uint16, 64*64)
	mask := make([]uint8, 128*128)
	rnd := newBlendRandom(libaomBlendDeterministicSeed)
	fillBlendSamples(src0, 0xfff, rnd)
	fillBlendSamples(src1, 0xfff, rnd)
	fillBlendMask(mask, rnd)
	allocs := testing.AllocsPerRun(1000, func() {
		if err := BlendA64Mask(dst, 64, src0, 64, src1, 64, mask, 128, 64, 64, true, true, 12); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("BlendA64Mask allocated: %f", allocs)
	}
}

func FuzzBlendA64Mask(f *testing.F) {
	f.Add(uint8(0), uint8(0), uint8(0), uint8(0), []byte{0, 64, 128, 255})
	f.Add(uint8(2), uint8(1), uint8(1), uint8(1), []byte{255, 0, 127, 32, 64})

	f.Fuzz(func(t *testing.T, rawSize uint8, rawBitDepth uint8, rawSubX uint8, rawSubY uint8, data []byte) {
		sizes := [...]int{4, 8, 16, 32}
		width := sizes[rawSize%uint8(len(sizes))]
		height := sizes[(rawSize/4)%uint8(len(sizes))]
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawBitDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		subX := rawSubX&1 != 0
		subY := rawSubY&1 != 0
		dst := make([]uint16, width*height)
		src0 := make([]uint16, width*height)
		src1 := make([]uint16, width*height)
		mask := make([]uint8, maskWidth(width, subX)*maskHeight(height, subY))
		for i := range src0 {
			src0[i] = fuzzBlendSample(data, i, max)
			src1[i] = fuzzBlendSample(data, len(src0)+i, max)
		}
		for i := range mask {
			if len(data) != 0 {
				mask[i] = data[i%len(data)] % (blendA64MaxAlpha + 1)
			}
		}
		if err := BlendA64Mask(dst, width, src0, width, src1, width, mask, maskWidth(width, subX), width, height, subX, subY, bitDepth); err != nil {
			t.Fatalf("BlendA64Mask err=%v", err)
		}
		for i, sample := range dst {
			if sample > max {
				t.Fatalf("dst[%d]=%d exceeds max %d", i, sample, max)
			}
		}
	})
}

func BenchmarkBlendA64Mask(b *testing.B) {
	dst := make([]uint16, 64*64)
	src0 := make([]uint16, 64*64)
	src1 := make([]uint16, 64*64)
	mask := make([]uint8, 128*128)
	rnd := newBlendRandom(libaomBlendDeterministicSeed)
	fillBlendSamples(src0, 0xfff, rnd)
	fillBlendSamples(src1, 0xfff, rnd)
	fillBlendMask(mask, rnd)
	b.ReportAllocs()
	for b.Loop() {
		_ = BlendA64Mask(dst, 64, src0, 64, src1, 64, mask, 128, 64, 64, true, true, 12)
	}
}

func blendA64MaskLibaomReference(dst []uint16, dstStride int, src0 []uint16, src0Stride int, src1 []uint16, src1Stride int, mask []uint8, maskStride int, width int, height int, subX bool, subY bool) {
	for row := range height {
		for col := range width {
			var m int
			switch {
			case !subX && !subY:
				m = int(mask[row*maskStride+col])
			case subX && subY:
				m = roundPowerOfTwoPositive(
					int(mask[(2*row)*maskStride+2*col])+
						int(mask[(2*row+1)*maskStride+2*col])+
						int(mask[(2*row)*maskStride+2*col+1])+
						int(mask[(2*row+1)*maskStride+2*col+1]),
					2,
				)
			case subX:
				m = blendAvg(int(mask[row*maskStride+2*col]), int(mask[row*maskStride+2*col+1]))
			default:
				m = blendAvg(int(mask[(2*row)*maskStride+col]), int(mask[(2*row+1)*maskStride+col]))
			}
			dst[row*dstStride+col] = uint16(roundPowerOfTwoPositive(m*int(src0[row*src0Stride+col])+(blendA64MaxAlpha-m)*int(src1[row*src1Stride+col]), 6))
		}
	}
}

func fillBlendSamples(samples []uint16, max uint16, rnd *blendRandom) {
	for i := range samples {
		samples[i] = uint16(rnd.pseudoUniform(int(max) + 1))
	}
}

func fillBlendMask(mask []uint8, rnd *blendRandom) {
	for i := range mask {
		mask[i] = uint8(rnd.pseudoUniform(blendA64MaxAlpha + 1))
	}
}

func fuzzBlendSample(data []byte, index int, max uint16) uint16 {
	if len(data) == 0 {
		return 0
	}
	lo := uint16(data[index%len(data)])
	hi := uint16(data[(index+1)%len(data)])
	return (lo | hi<<8) & max
}

type blendRandom struct {
	state uint32
}

func newBlendRandom(seed uint32) *blendRandom {
	return &blendRandom{state: seed}
}

func (r *blendRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *blendRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}
