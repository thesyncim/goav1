package filmgrain

import (
	"bytes"
	"math/rand"
	"testing"
)

// The reference walks each pixel and overlap region independently of the new
// byte block/segment path; full buffers include stride padding and end canaries.
func TestApplyRowsU8MatchPerSampleReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0x892f))
	grainY := randomGrainTemplate(rng, LumaGrainSamples, 8)
	grainUV := randomGrainTemplate(rng, ChromaGrainSamples, 8)
	scaling := randomScalingLUT(rng)
	for i := range scaling {
		if i&3 == 0 {
			scaling[i] = 255
		}
	}
	for _, width := range []int{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65} {
		for mode := 0; mode < 16; mode++ {
			height := 32
			if mode&1 != 0 {
				height = 1 + mode
			}
			p := LumaRowParams{Seed: uint16(mode * 3511), Width: width, Height: height, Stride: width + 7, Row: mode % 3,
				BitDepth: 8, ScalingShift: uint8(8 + mode%4), Overlap: mode&2 != 0, ClipToRestrictedRange: mode&4 != 0}
			src16 := randomSamplePlane(rng, height*p.Stride+13, 8)
			src := grainTestBytes(src16)
			before := bytes.Clone(src)
			for _, alias := range []bool{false, true} {
				dst := bytes.Repeat([]byte{0xa5}, len(src))
				want := make([]uint16, len(src))
				for i := range want {
					want[i] = 0xa5
				}
				if alias {
					copy(dst, src)
					copy(want, src16)
				}
				input := src
				if alias {
					input = dst
				}
				applyLumaRowRef(t, want, src16, grainY, scaling, p)
				if err := ApplyLumaRowU8(dst, input, grainY, scaling, p); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(dst, grainTestBytes(want)) {
					t.Fatalf("luma width=%d mode=%d alias=%v", width, mode, alias)
				}
			}
			if !bytes.Equal(src, before) {
				t.Fatal("source modified")
			}

			for _, ss := range [][2]bool{{false, false}, {true, false}, {true, true}} {
				sx, sy := chromaSubsamplingShift(ss[0]), chromaSubsamplingShift(ss[1])
				h := min(height, 32>>sy)
				cp := ChromaRowParams{Seed: p.Seed, Width: width, Height: h, Stride: width + 7, LumaStride: (width << sx) + 9,
					Row: p.Row, BitDepth: 8, ScalingShift: p.ScalingShift, Overlap: p.Overlap, ClipToRestrictedRange: p.ClipToRestrictedRange,
					SubsamplingX: ss[0], SubsamplingY: ss[1], ChromaScalingFromLuma: mode&1 != 0, MatrixIdentity: mode&8 != 0,
					ChromaMult: uint8(mode * 17), ChromaLumaMult: uint8(255 - mode*17), ChromaOffset: uint16(mode * 34)}
				uv16 := randomSamplePlane(rng, h*cp.Stride+13, 8)
				uv := grainTestBytes(uv16)
				lum16 := randomSamplePlane(rng, (h<<sy)*cp.LumaStride, 8)
				lum := grainTestBytes(lum16)
				for _, alias := range []bool{false, true} {
					dst := bytes.Repeat([]byte{0xa5}, len(uv))
					want := make([]uint16, len(uv))
					for i := range want {
						want[i] = 0xa5
					}
					if alias {
						copy(dst, uv)
						copy(want, uv16)
					}
					input := uv
					if alias {
						input = dst
					}
					applyChromaRowRef(t, want, uv16, lum16, grainUV, scaling, cp)
					if err := ApplyChromaRowU8(dst, input, lum, grainUV, scaling, cp); err != nil {
						t.Fatal(err)
					}
					if !bytes.Equal(dst, grainTestBytes(want)) {
						t.Fatalf("chroma width=%d mode=%d ss=%v alias=%v", width, mode, ss, alias)
					}
				}
				if !bytes.Equal(uv, grainTestBytes(uv16)) || !bytes.Equal(lum, grainTestBytes(lum16)) {
					t.Fatal("chroma source modified")
				}
			}
		}
	}
}

func grainTestBytes(samples []uint16) []byte {
	out := make([]byte, len(samples))
	for i, v := range samples {
		out[i] = byte(v)
	}
	return out
}

func TestApplyRowsU8RejectInvalidBeforeMutation(t *testing.T) {
	grain := make([]int16, LumaGrainSamples)
	scaling := make([]uint8, ScalingLUTSize)
	p := LumaRowParams{Width: 8, Height: 2, Stride: 8, BitDepth: 8, ScalingShift: 8}
	for _, change := range []func(*LumaRowParams){
		func(p *LumaRowParams) { p.BitDepth = 10 }, func(p *LumaRowParams) { p.ScalingShift = 7 },
		func(p *LumaRowParams) { p.Stride = int(^uint(0) >> 1) }, func(p *LumaRowParams) { p.Height = 33 },
	} {
		bad := p
		change(&bad)
		dst := bytes.Repeat([]byte{0xab}, 16)
		if err := ApplyLumaRowU8(dst, dst, grain, scaling, bad); err != ErrInvalidParams {
			t.Fatalf("params=%+v err=%v", bad, err)
		}
		if !bytes.Equal(dst, bytes.Repeat([]byte{0xab}, 16)) {
			t.Fatal("invalid request mutated destination")
		}
	}
	if err := ApplyLumaRowU8(make([]byte, 15), make([]byte, 16), grain, scaling, p); err != ErrInvalidParams {
		t.Fatal(err)
	}
	cp := ChromaRowParams{Width: 8, Height: 2, Stride: 8, LumaStride: 16, BitDepth: 8, ScalingShift: 8, SubsamplingX: true}
	for _, change := range []func(*ChromaRowParams){
		func(p *ChromaRowParams) { p.BitDepth = 12 }, func(p *ChromaRowParams) { p.ScalingShift = 12 },
		func(p *ChromaRowParams) { p.LumaStride = int(^uint(0) >> 1) }, func(p *ChromaRowParams) { p.Height = 33 },
	} {
		bad := cp
		change(&bad)
		dst := bytes.Repeat([]byte{0xab}, 16)
		if err := ApplyChromaRowU8(dst, dst, make([]byte, 32), grain, scaling, bad); err != ErrInvalidParams {
			t.Fatalf("params=%+v err=%v", bad, err)
		}
		if !bytes.Equal(dst, bytes.Repeat([]byte{0xab}, 16)) {
			t.Fatal("invalid chroma request mutated destination")
		}
	}
}

// TestRound2ViaHalvingAdd checks the identity the arm64 U8 kernels use for
// dav1d's srshl by -scaling_shift, which archsimd lacks:
// round2(p, s) == ((p >> (s-1)) + 1) >> 1 (sshl, then srhadd with zero) for
// every scaling*grain product an 8-bit frame can produce.
func TestRound2ViaHalvingAdd(t *testing.T) {
	for s := 8; s <= 11; s++ {
		for p := -255 * 128; p <= 255*127; p++ {
			if got, want := ((p>>(s-1))+1)>>1, roundPowerOfTwo(p, s); got != want {
				t.Fatalf("p=%d s=%d: got %d want %d", p, s, got, want)
			}
		}
	}
}

// TestChromaIndexSaturatingSum checks that dav1d's int16 sqadd of
// avg*uv_luma_mult + src*uv_mult (used by the arm64 U8 chroma kernels) gives
// the same clipped scaling index as the exact sum.
func TestChromaIndexSaturatingSum(t *testing.T) {
	for _, lm := range []int{-128, -127, -1, 0, 1, 126, 127} {
		for _, m := range []int{-128, -127, -1, 0, 1, 126, 127} {
			for _, off := range []int{-256, -255, -1, 0, 1, 254, 255} {
				for avg := 0; avg < 256; avg++ {
					for src := 0; src < 256; src++ {
						exact := avg*lm + src*m
						sat := min(max(exact, -32768), 32767)
						if got, want := clipInt((sat>>6)+off, 0, 255), clipInt((exact>>6)+off, 0, 255); got != want {
							t.Fatalf("lm=%d m=%d off=%d avg=%d src=%d: got %d want %d", lm, m, off, avg, src, got, want)
						}
					}
				}
			}
		}
	}
}
