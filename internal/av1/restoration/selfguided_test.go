package restoration

import (
	"errors"
	"slices"
	"testing"
)

const restorationDeterministicSeed = 0xbaba

func TestSGRParameterTableMatchesLibaom(t *testing.T) {
	want := [SGRProjParams]SGRParams{
		{Radius: [2]int{2, 1}, S: [2]int{140, 3236}},
		{Radius: [2]int{2, 1}, S: [2]int{112, 2158}},
		{Radius: [2]int{2, 1}, S: [2]int{93, 1618}},
		{Radius: [2]int{2, 1}, S: [2]int{80, 1438}},
		{Radius: [2]int{2, 1}, S: [2]int{70, 1295}},
		{Radius: [2]int{2, 1}, S: [2]int{58, 1177}},
		{Radius: [2]int{2, 1}, S: [2]int{47, 1079}},
		{Radius: [2]int{2, 1}, S: [2]int{37, 996}},
		{Radius: [2]int{2, 1}, S: [2]int{30, 925}},
		{Radius: [2]int{2, 1}, S: [2]int{25, 863}},
		{Radius: [2]int{0, 1}, S: [2]int{-1, 2589}},
		{Radius: [2]int{0, 1}, S: [2]int{-1, 1618}},
		{Radius: [2]int{0, 1}, S: [2]int{-1, 1177}},
		{Radius: [2]int{0, 1}, S: [2]int{-1, 925}},
		{Radius: [2]int{2, 0}, S: [2]int{56, -1}},
		{Radius: [2]int{2, 0}, S: [2]int{22, -1}},
	}
	if SGRParameterTable != want {
		t.Fatalf("SGRParameterTable=%+v want %+v", SGRParameterTable, want)
	}
	xByXPlus1Checks := map[uint32]int32{
		0:   1,
		1:   128,
		2:   171,
		3:   192,
		4:   205,
		16:  241,
		255: 256,
		256: 256,
	}
	for z, want := range xByXPlus1Checks {
		if got := xByXPlus1Value(z); got != want {
			t.Fatalf("xByXPlus1Value(%d)=%d want %d", z, got, want)
		}
	}
	if oneByX[0] != 4096 || oneByX[24] != 164 {
		t.Fatalf("oneByX sentinels mismatch")
	}
}

func TestDecodeSGRXQMatchesLibaom(t *testing.T) {
	tests := []struct {
		name   string
		params SGRParams
		xqd    [2]int
		want   [2]int
	}{
		{name: "two-radii", params: SGRParameterTable[0], xqd: [2]int{20, -7}, want: [2]int{20, 115}},
		{name: "first-disabled", params: SGRParameterTable[10], xqd: [2]int{99, 13}, want: [2]int{0, 115}},
		{name: "second-disabled", params: SGRParameterTable[14], xqd: [2]int{-12, 44}, want: [2]int{-12, 0}},
	}
	for _, tt := range tests {
		if got := DecodeSGRXQ(tt.xqd, tt.params); got != tt.want {
			t.Fatalf("%s xq=%v want %v", tt.name, got, tt.want)
		}
	}
}

func TestSelfguidedRestorationConstantBlocksStayInRange(t *testing.T) {
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, size := range []struct {
			width  int
			height int
		}{{4, 4}, {17, 19}, {64, 64}} {
			for eps := 0; eps < SGRProjParams; eps++ {
				stride := size.width + 2*SGRProjBorderHorz
				origin := SGRProjBorderVert*stride + SGRProjBorderHorz
				src := make([]uint16, stride*(size.height+2*SGRProjBorderVert))
				for i := range src {
					src[i] = max / 3
				}
				dst := make([]uint16, size.width*size.height)
				scratchLen, err := SelfguidedScratchLen(size.width, size.height)
				if err != nil {
					t.Fatal(err)
				}
				scratch := make([]int32, scratchLen)
				if err := ApplySelfguidedRestoration(src, stride, origin, dst, size.width, size.width, size.height, eps, [2]int{13, -9}, bitDepth, scratch); err != nil {
					t.Fatalf("bitDepth=%d size=%dx%d eps=%d: %v", bitDepth, size.width, size.height, eps, err)
				}
				for i, got := range dst {
					if got > max {
						t.Fatalf("bitDepth=%d size=%dx%d eps=%d dst[%d]=%d max %d", bitDepth, size.width, size.height, eps, i, got, max)
					}
					if got != dst[0] {
						t.Fatalf("bitDepth=%d size=%dx%d eps=%d dst[%d]=%d want constant %d", bitDepth, size.width, size.height, eps, i, got, dst[0])
					}
				}
			}
		}
	}
}

func TestSelfguidedRestorationDeterministicCorpus(t *testing.T) {
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for _, bitDepth := range []uint8{8, 10, 12} {
		max := uint16((1 << bitDepth) - 1)
		for _, tc := range []struct {
			width  int
			height int
			eps    int
			xqd    [2]int
		}{
			{8, 8, 0, [2]int{0, 0}},
			{13, 11, 7, [2]int{19, -5}},
			{64, 16, 10, [2]int{4, 33}},
			{31, 64, 14, [2]int{-20, 0}},
		} {
			stride := tc.width + 2*SGRProjBorderHorz + 5
			origin := SGRProjBorderVert*stride + SGRProjBorderHorz
			src := make([]uint16, stride*(tc.height+2*SGRProjBorderVert))
			for i := range src {
				src[i] = uint16(rnd.pseudoUniform(int(max) + 1))
			}
			dst := make([]uint16, tc.width*tc.height)
			scratchLen, err := SelfguidedScratchLen(tc.width, tc.height)
			if err != nil {
				t.Fatal(err)
			}
			scratch := make([]int32, scratchLen)
			if err := ApplySelfguidedRestoration(src, stride, origin, dst, tc.width, tc.width, tc.height, tc.eps, tc.xqd, bitDepth, scratch); err != nil {
				t.Fatalf("bitDepth=%d tc=%+v: %v", bitDepth, tc, err)
			}
			for i, sample := range dst {
				if sample > max {
					t.Fatalf("bitDepth=%d tc=%+v dst[%d]=%d max=%d", bitDepth, tc, i, sample, max)
				}
			}
			dst2 := make([]uint16, len(dst))
			if err := ApplySelfguidedRestoration(src, stride, origin, dst2, tc.width, tc.width, tc.height, tc.eps, tc.xqd, bitDepth, scratch); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(dst, dst2) {
				t.Fatalf("bitDepth=%d tc=%+v not deterministic", bitDepth, tc)
			}
		}
	}
}

func TestSelfguidedScratchLenAndAllocs(t *testing.T) {
	scratchLen, err := SelfguidedScratchLen(64, 64)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]int32, scratchLen)
	stride := 64 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(64+2*SGRProjBorderVert))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := ApplySelfguidedRestoration(src, stride, origin, dst, 64, 64, 64, 15, [2]int{8, 11}, 12, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplySelfguidedRestoration allocated: %f", allocs)
	}
}

func BenchmarkApplySelfguidedRestoration(b *testing.B) {
	scratchLen, err := SelfguidedScratchLen(64, 64)
	if err != nil {
		b.Fatal(err)
	}
	scratch := make([]int32, scratchLen)
	stride := 64 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(64+2*SGRProjBorderVert))
	dst := make([]uint16, 64*64)
	rnd := newRestorationRandom(restorationDeterministicSeed)
	for i := range src {
		src[i] = uint16(rnd.pseudoUniform(1 << 12))
	}
	b.SetBytes(int64(64 * 64 * 2))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ApplySelfguidedRestoration(src, stride, origin, dst, 64, 64, 64, 15, [2]int{8, 11}, 12, scratch)
	}
}

func TestSelfguidedRestorationRejectsInvalidInputs(t *testing.T) {
	scratchLen, err := SelfguidedScratchLen(8, 8)
	if err != nil {
		t.Fatal(err)
	}
	scratch := make([]int32, scratchLen)
	stride := 8 + 2*SGRProjBorderHorz
	origin := SGRProjBorderVert*stride + SGRProjBorderHorz
	src := make([]uint16, stride*(8+2*SGRProjBorderVert))
	dst := make([]uint16, 8*8)
	tests := []struct {
		name string
		fn   func() error
	}{
		{name: "bad-size", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst, 8, 0, 8, 0, [2]int{}, 8, scratch)
		}},
		{name: "bad-eps", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst, 8, 8, 8, 16, [2]int{}, 8, scratch)
		}},
		{name: "bad-depth", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst, 8, 8, 8, 0, [2]int{}, 9, scratch)
		}},
		{name: "short-src", fn: func() error {
			return ApplySelfguidedRestoration(src[:origin], stride, origin, dst, 8, 8, 8, 0, [2]int{}, 8, scratch)
		}},
		{name: "short-dst", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst[:8], 8, 8, 8, 0, [2]int{}, 8, scratch)
		}},
		{name: "short-scratch", fn: func() error {
			return ApplySelfguidedRestoration(src, stride, origin, dst, 8, 8, 8, 0, [2]int{}, 8, scratch[:scratchLen-1])
		}},
		{name: "sample-out-of-range", fn: func() error {
			bad := slices.Clone(src)
			bad[origin] = 256
			return ApplySelfguidedRestoration(bad, stride, origin, dst, 8, 8, 8, 0, [2]int{}, 8, scratch)
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

func FuzzApplySelfguidedRestoration(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), uint8(0), int16(0), int16(0), []byte{0, 64, 128, 255})
	f.Add(uint8(31), uint8(17), uint8(2), uint8(15), int16(13), int16(-9), []byte{255, 1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawDepth uint8, rawEps uint8, rawX0 int16, rawX1 int16, data []byte) {
		width := int(rawW%ProcUnitSize) + 1
		height := int(rawH%ProcUnitSize) + 1
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		stride := width + 2*SGRProjBorderHorz
		origin := SGRProjBorderVert*stride + SGRProjBorderHorz
		src := make([]uint16, stride*(height+2*SGRProjBorderVert))
		for i := range src {
			src[i] = fuzzRestorationSample(data, i, max)
		}
		dst := make([]uint16, width*height)
		scratchLen, err := SelfguidedScratchLen(width, height)
		if err != nil {
			t.Fatalf("SelfguidedScratchLen err=%v", err)
		}
		scratch := make([]int32, scratchLen)
		err = ApplySelfguidedRestoration(src, stride, origin, dst, width, width, height, int(rawEps%SGRProjParams), [2]int{int(rawX0 % 96), int(rawX1 % 96)}, bitDepth, scratch)
		if err != nil {
			t.Fatalf("ApplySelfguidedRestoration err=%v", err)
		}
		for i, sample := range dst {
			if sample > max {
				t.Fatalf("dst[%d]=%d max=%d", i, sample, max)
			}
		}
	})
}

func fuzzRestorationSample(data []byte, index int, max uint16) uint16 {
	if len(data) == 0 {
		return 0
	}
	lo := uint16(data[index%len(data)])
	hi := uint16(data[(index+1)%len(data)])
	return (lo | hi<<8) & max
}

type restorationRandom struct {
	state uint32
}

func newRestorationRandom(seed uint32) *restorationRandom {
	return &restorationRandom{state: seed}
}

func (r *restorationRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *restorationRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}
