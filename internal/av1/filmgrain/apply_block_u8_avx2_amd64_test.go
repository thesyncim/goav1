//go:build goexperiment.simd && amd64 && !purego

package filmgrain

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestAMD64FilmGrainU8Dispatch checks that the AVX2 block kernels are live
// and that TestApplyRowsU8MatchPerSampleReference also passes on the
// scalar fallback.
func TestAMD64FilmGrainU8Dispatch(t *testing.T) {
	if !fgAVX2 {
		t.Skip("no AVX2")
	}
	t.Run("scalar", func(t *testing.T) {
		fgAVX2 = false
		defer func() { fgAVX2 = true }()
		TestApplyRowsU8MatchPerSampleReference(t)
	})
}

// BenchmarkAMD64FilmGrainU8 compares the AVX2 film grain block kernels with
// the scalar block path on one 720p 4:2:0 frame.
func BenchmarkAMD64FilmGrainU8(b *testing.B) {
	for _, c := range []struct {
		name         string
		chroma, csfl bool
	}{{"luma", false, false}, {"chroma", true, false}, {"chroma_csfl", true, true}} {
		for _, impl := range []string{"generic", "avx2"} {
			b.Run("plane="+c.name+"/impl="+impl, func(b *testing.B) {
				saved := fgAVX2
				if impl == "generic" {
					fgAVX2 = false
				} else if !fgAVX2 {
					b.Skip("no AVX2")
				}
				defer func() { fgAVX2 = saved }()
				benchApplyU8(b, c.chroma, c.csfl)
			})
		}
	}
}

// TestAMD64FilmGrainU8WideMatchesScalar runs full 1280-wide bands (every
// grain offset the RNG produces across many blocks and seeds) through the
// AVX2 kernels and the scalar block path.
func TestAMD64FilmGrainU8WideMatchesScalar(t *testing.T) {
	if !fgAVX2 {
		t.Skip("no AVX2")
	}
	const w = 1280
	rng := rand.New(rand.NewSource(11))
	grainY := randomGrainTemplate(rng, LumaGrainSamples, 8)
	grainUV := randomGrainTemplate(rng, ChromaGrainSamples, 8)
	scaling := randomScalingLUT(rng)
	luma := make([]byte, w*64)
	for i := range luma {
		luma[i] = byte(rng.Intn(256))
	}
	run := func(avx2 bool, f func(dst []byte) error) []byte {
		fgAVX2 = avx2
		defer func() { fgAVX2 = true }()
		dst := make([]byte, w*64)
		if err := f(dst); err != nil {
			t.Fatal(err)
		}
		return dst
	}
	for seed := 0; seed < 24; seed++ {
		for row := 0; row < 3; row++ {
			p := LumaRowParams{Seed: uint16(seed * 7919), Width: w, Height: 32, Stride: w, Row: row, BitDepth: 8,
				ScalingShift: uint8(8 + seed%4), Overlap: seed%3 != 0, ClipToRestrictedRange: seed%5 == 0}
			f := func(dst []byte) error { return ApplyLumaRowU8(dst, luma, grainY, scaling, p) }
			if !bytes.Equal(run(true, f), run(false, f)) {
				t.Fatalf("luma seed %d row %d", seed, row)
			}
			for _, ss := range [][2]bool{{true, true}, {true, false}, {false, false}} {
				sx, sy := chromaSubsamplingShift(ss[0]), chromaSubsamplingShift(ss[1])
				cw := w >> sx
				cp := ChromaRowParams{Seed: p.Seed, Width: cw, Height: 32 >> sy, Stride: cw, LumaStride: w, Row: row, BitDepth: 8,
					ScalingShift: p.ScalingShift, Overlap: p.Overlap, ClipToRestrictedRange: p.ClipToRestrictedRange,
					SubsamplingX: ss[0], SubsamplingY: ss[1], ChromaScalingFromLuma: seed%2 == 0,
					ChromaMult: uint8(seed * 37), ChromaLumaMult: uint8(255 - seed*29), ChromaOffset: uint16(seed * 21)}
				uv := luma[w*32:]
				f := func(dst []byte) error { return ApplyChromaRowU8(dst, uv, luma, grainUV, scaling, cp) }
				if !bytes.Equal(run(true, f), run(false, f)) {
					t.Fatalf("chroma seed %d row %d ss %v", seed, row, ss)
				}
			}
		}
	}
}
