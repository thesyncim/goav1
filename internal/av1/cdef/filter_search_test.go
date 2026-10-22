package cdef

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestFilterBlocksU8SubsampledMatchesFullRows: every row the subsampled
// search filter writes equals the same row of the full-block filter, and
// the rows it skips stay untouched.
func TestFilterBlocksU8SubsampledMatchesFullRows(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	in := make([]uint16, InputBufferSize)
	origin := VerticalBorder*BStride + HorizontalBorder
	for trial := 0; trial < 200; trial++ {
		dec := rng.Intn(2)
		size := 64 >> dec
		for i := range in {
			in[i] = uint16(rng.Intn(256))
			if rng.Intn(50) == 0 {
				in[i] = VeryLarge
			}
		}
		for r := 0; r < size; r++ {
			for c := 0; c < size; c++ {
				if in[origin+r*BStride+c] == VeryLarge {
					in[origin+r*BStride+c] = 7
				}
			}
		}
		base := make([]byte, size*size)
		for r := 0; r < size; r++ {
			for c := 0; c < size; c++ {
				base[r*size+c] = byte(in[origin+r*BStride+c])
			}
		}
		var blocks []BlockPosition
		for by := 0; by < 8; by++ {
			for bx := 0; bx < 8; bx++ {
				if rng.Intn(4) != 0 {
					blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
				}
			}
		}
		var dir DirectionGrid
		var vari VarianceGrid
		plane := PlaneY
		if dec == 1 {
			plane = PlaneU
		}
		dirSrc := make([]byte, 64*64)
		for i := range dirSrc {
			dirSrc[i] = byte(rng.Intn(256))
		}
		if err := FilterFrameBlocksU8Trusted(dirSrc, 64, nil, 0, blocks, &dir, &vari, FrameFilterParams{Plane: PlaneY, Damping: 3}); err != nil {
			t.Fatal(err)
		}
		params := FrameFilterParams{XDec: uint8(dec), YDec: uint8(dec), Plane: plane,
			Level: uint8(rng.Intn(16)), SecondaryStrength: uint8([]int{0, 1, 2, 4}[rng.Intn(4)]), Damping: uint8(3 + rng.Intn(4))}
		full := append([]byte(nil), base...)
		if err := FilterFrameBlocksU8TrustedNoPrimaryDirection(full, size, in, origin, blocks, &dir, &vari, params); err != nil {
			t.Fatal(err)
		}
		for _, step := range []int{2, 4} {
			sub := append([]byte(nil), base...)
			if err := FilterBlocksU8SubsampledTrusted(sub, size, in, origin, blocks, &dir, &vari, params, step); err != nil {
				t.Fatal(err)
			}
			for r := 0; r < size; r++ {
				want := base[r*size : r*size+size]
				if (r%(8>>dec))%step == 0 {
					want = full[r*size : r*size+size]
				}
				if !bytes.Equal(sub[r*size:r*size+size], want) {
					t.Fatalf("trial %d dec %d step %d row %d differs", trial, dec, step, r)
				}
			}
		}
	}
}
