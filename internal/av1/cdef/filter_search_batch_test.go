package cdef

import (
	"math/rand"
	"testing"
)

// searchOracle is the per-trial search as the encoder ran it: copy the
// reconstruction into a scratch, filter every rowStep-th row of each block
// with FilterBlocksU8SubsampledTrusted, then the row-stepped SSE against the
// source (compute_cdef_dist_8bit).
func searchOracle(src []byte, srcStride int, in []uint16, origin int, blocks []BlockPosition, dir *DirectionGrid, vari *VarianceGrid, damping int, trials []SearchTrial, rowStep int) []uint64 {
	out := make([]uint64, len(trials))
	tmp := make([]byte, 64*64)
	for t, tr := range trials {
		for r := 0; r < 64; r++ {
			for c := 0; c < 64; c++ {
				tmp[r*64+c] = byte(in[origin+r*BStride+c])
			}
		}
		if tr.Primary != 0 || tr.Secondary != 0 {
			params := FrameFilterParams{Plane: PlaneY, Level: tr.Primary, SecondaryStrength: tr.Secondary, Damping: uint8(damping)}
			if err := FilterBlocksU8SubsampledTrusted(tmp, 64, in, origin, blocks, dir, vari, params, rowStep); err != nil {
				panic(err)
			}
		}
		var sum uint64
		for _, b := range blocks {
			for i := 0; i < 8; i += rowStep {
				so := (int(b.BY)*8+i)*srcStride + int(b.BX)*8
				fo := (int(b.BY)*8+i)*64 + int(b.BX)*8
				for j := 0; j < 8; j++ {
					e := int32(src[so+j]) - int32(tmp[fo+j])
					sum += uint64(e * e)
				}
			}
		}
		out[t] = sum
	}
	return out
}

// TestSearchBlocksU8MatchesPerTrial: the batched search returns, per trial,
// the per-trial filter-then-measure sums, across random strengths, row
// steps, directions, variances and block lists, including lists longer than
// one kernel chunk.
func TestSearchBlocksU8MatchesPerTrial(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	in := make([]uint16, InputBufferSize)
	origin := VerticalBorder*BStride + HorizontalBorder
	const srcStride = 72
	src := make([]byte, 64*srcStride)
	secondaries := []uint8{0, 1, 2, 4}
	for iter := 0; iter < 300; iter++ {
		for i := range in {
			in[i] = uint16(rng.Intn(256))
			if rng.Intn(40) == 0 {
				in[i] = VeryLarge
			}
		}
		// Block interiors are reconstruction samples; sentinels live in the halo.
		for r := 0; r < 64; r++ {
			for c := 0; c < 64; c++ {
				if in[origin+r*BStride+c] == VeryLarge {
					in[origin+r*BStride+c] = uint16(rng.Intn(256))
				}
			}
		}
		switch rng.Intn(3) {
		case 0: // source near the reconstruction, as real content is
			for r := 0; r < 64; r++ {
				for c := 0; c < 64; c++ {
					v := int(in[origin+r*BStride+c]) + rng.Intn(21) - 10
					src[r*srcStride+c] = byte(min(max(v, 0), 255))
				}
			}
		default:
			for i := range src {
				src[i] = byte(rng.Intn(256))
			}
		}
		var blocks []BlockPosition
		for by := 0; by < 8; by++ {
			for bx := 0; bx < 8; bx++ {
				if rng.Intn(3) != 0 {
					blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
				}
			}
		}
		if len(blocks) == 0 {
			blocks = append(blocks, BlockPosition{BY: uint8(rng.Intn(8)), BX: uint8(rng.Intn(8))})
		}
		var dir DirectionGrid
		var vari VarianceGrid
		for by := range dir {
			for bx := range dir[by] {
				dir[by][bx] = uint8(rng.Intn(8))
				switch rng.Intn(4) {
				case 0:
					vari[by][bx] = 0
				case 1:
					vari[by][bx] = int32(rng.Intn(64))
				default:
					vari[by][bx] = int32(rng.Intn(1 << 20))
				}
			}
		}
		damping := 3 + rng.Intn(4)
		rowStep := []int{1, 2, 4}[rng.Intn(3)]
		nTrials := 1 + rng.Intn(20)
		trials := make([]SearchTrial, nTrials)
		for i := range trials {
			trials[i] = SearchTrial{Primary: uint8(rng.Intn(16)), Secondary: secondaries[rng.Intn(4)]}
			if rng.Intn(6) == 0 {
				trials[i] = SearchTrial{}
			}
		}
		want := searchOracle(src, srcStride, in, origin, blocks, &dir, &vari, damping, trials, rowStep)
		got := make([]uint64, nTrials)
		if !SearchBlocksU8Trusted(src, srcStride, in, origin, blocks, &dir, &vari, damping, trials, rowStep, got) {
			t.Skip("no batched search kernel in this build")
		}
		for i := range trials {
			if got[i] != want[i] {
				t.Fatalf("iter %d trial %d %+v (damping %d, step %d, %d blocks): got %d want %d", iter, i, trials[i], damping, rowStep, len(blocks), got[i], want[i])
			}
		}
	}
}

// TestSearchBlocksU8RejectsOutOfContract: geometry outside the trusted
// contract is refused rather than read.
func TestSearchBlocksU8RejectsOutOfContract(t *testing.T) {
	in := make([]uint16, InputBufferSize)
	origin := VerticalBorder*BStride + HorizontalBorder
	src := make([]byte, 64*64)
	var dir DirectionGrid
	var vari VarianceGrid
	trials := []SearchTrial{{15, 2}}
	sse := make([]uint64, 1)
	ok := SearchBlocksU8Trusted(src, 64, in, origin, []BlockPosition{{BY: 7, BX: 7}}, &dir, &vari, 3, trials, 1, sse)
	if !ok {
		t.Skip("no batched search kernel in this build")
	}
	if SearchBlocksU8Trusted(src[:64*63], 64, in, origin, []BlockPosition{{BY: 7, BX: 7}}, &dir, &vari, 3, trials, 1, sse) {
		t.Fatal("accepted a source plane missing the last block row")
	}
	if SearchBlocksU8Trusted(src, 64, in, origin, []BlockPosition{{BY: 8, BX: 0}}, &dir, &vari, 3, trials, 1, sse) {
		t.Fatal("accepted a block outside the 64x64 filter block")
	}
	if SearchBlocksU8Trusted(src, 64, in, origin, nil, &dir, &vari, 3, trials, 0, sse) {
		t.Fatal("accepted row step 0")
	}
}

func BenchmarkSearchBlocksU8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	in := make([]uint16, InputBufferSize)
	origin := VerticalBorder*BStride + HorizontalBorder
	for i := range in {
		in[i] = uint16(rng.Intn(256))
	}
	src := make([]byte, 64*64)
	for i := range src {
		src[i] = byte(rng.Intn(256))
	}
	var blocks []BlockPosition
	for by := 0; by < 8; by++ {
		for bx := 0; bx < 8; bx++ {
			blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
		}
	}
	var dir DirectionGrid
	var vari VarianceGrid
	for by := range dir {
		for bx := range dir[by] {
			dir[by][bx] = uint8(rng.Intn(8))
			vari[by][bx] = int32(rng.Intn(1 << 16))
		}
	}
	// SVT rtc level 8 on a base-layer inter frame: (0,0), (15,0), (0,2),
	// (15,2) at subsampling 4.
	trials := []SearchTrial{{0, 0}, {15, 0}, {0, 2}, {15, 2}}
	sse := make([]uint64, len(trials))
	b.Run("level8-step4", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SearchBlocksU8Trusted(src, 64, in, origin, blocks, &dir, &vari, 4, trials, 4, sse)
		}
	})
	b.Run("level8-step4-oracle", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			searchOracle(src, 64, in, origin, blocks, &dir, &vari, 4, trials, 4)
		}
	})
	// Level 5 on an I slice: six trials, every row.
	trials5 := []SearchTrial{{0, 0}, {7, 0}, {15, 0}, {0, 2}, {7, 2}, {15, 2}}
	sse5 := make([]uint64, len(trials5))
	b.Run("level5-step1", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SearchBlocksU8Trusted(src, 64, in, origin, blocks, &dir, &vari, 4, trials5, 1, sse5)
		}
	})
}
