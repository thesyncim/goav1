// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Differential coverage for the six/eight-sample 16-bit vertical-edge NEON
// transpose kernels (filter_wide16_vtrn_neon_arm64.s). These drive the arm64
// asm directly, so the file is constrained to that build.
//go:build arm64 && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

// wide16VertTransposeCase describes one six/eight-tap transpose kernel pair
// for the shared gather/scatter differential: taps is the window width in
// samples, modLo..modHi the modifiable tap rows, lowTap the offset (in taps)
// from the lowest tap back to q0.
type wide16VertTransposeCase struct {
	name    string
	taps    int
	modLo   int
	modHi   int
	lowTap  int
	gather  func(*wideVertTransposeCtx)
	scatter func(*wideVertTransposeCtx)
}

func wide16VertTransposeCases() []wide16VertTransposeCase {
	return []wide16VertTransposeCase{
		{"filter6", 6, 1, 4, 3, filter6Vert16GatherNEONAsm, filter6Vert16ScatterNEONAsm},
		{"filter8", 8, 1, 6, 4, filter8Vert16GatherNEONAsm, filter8Vert16ScatterNEONAsm},
	}
}

// TestWide16VertTransposeGatherScatter drives the six/eight-tap gather/scatter
// asm kernels against the scalar transpose loops they replaced, byte for byte,
// across strides and counts. The gather check also asserts that no scratch
// byte outside the taps x positions window is written, and the scatter check
// first fills the modifiable tap rows with fresh random content so the
// scattered columns are distinguishable from the gathered ones.
func TestWide16VertTransposeGatherScatter(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const scratchStride = wide16VertScratchStride
	for _, tc := range wide16VertTransposeCases() {
		const step = 2
		for _, stride := range []int{16 * 2, 96, 640} {
			for count := 1; count <= filter14VertBatchGroups; count++ {
				pix := make([]byte, 8*filter14VertBatchGroups*stride+64)
				for i := range pix {
					pix[i] = byte(rng.Intn(256))
				}
				q0Base := tc.lowTap * step // lowest tap of position 0 at byte 0
				scratch := make([]byte, tc.taps*scratchStride)
				for i := range scratch {
					scratch[i] = 0xAA
				}
				wantScratch := make([]byte, tc.taps*scratchStride)
				for i := range wantScratch {
					wantScratch[i] = 0xAA
				}
				ctx := wideVertTransposeCtx{
					src:     &pix[q0Base-tc.lowTap*step],
					stride:  uintptr(stride),
					scratch: &scratch[0],
					count:   uintptr(count),
				}
				// Reference gather (the old scalar loops).
				for j := 0; j < 8*count; j++ {
					src := q0Base + j*stride - tc.lowTap*step
					for tap := 0; tap < tc.taps; tap++ {
						o := src + tap*step
						d := tap*scratchStride + j*2
						wantScratch[d] = pix[o]
						wantScratch[d+1] = pix[o+1]
					}
				}
				tc.gather(&ctx)
				for i := range scratch {
					if scratch[i] != wantScratch[i] {
						t.Fatalf("%s gather stride=%d count=%d scratch[%d] got=%d want=%d",
							tc.name, stride, count, i, scratch[i], wantScratch[i])
					}
				}
				// Scatter: randomize the modifiable rows, then compare the
				// asm scatter against the scalar loops on a copy.
				for tap := tc.modLo; tap <= tc.modHi; tap++ {
					for j := 0; j < 8*count*2; j++ {
						scratch[tap*scratchStride+j] = byte(rng.Intn(256))
					}
				}
				wantPix := append([]byte(nil), pix...)
				for j := 0; j < 8*count; j++ {
					dst := q0Base + j*stride - tc.lowTap*step
					for tap := tc.modLo; tap <= tc.modHi; tap++ {
						o := dst + tap*step
						d := tap*scratchStride + j*2
						wantPix[o] = scratch[d]
						wantPix[o+1] = scratch[d+1]
					}
				}
				tc.scatter(&ctx)
				for i := range pix {
					if pix[i] != wantPix[i] {
						t.Fatalf("%s scatter stride=%d count=%d idx=%d got=%d want=%d",
							tc.name, stride, count, i, pix[i], wantPix[i])
					}
				}
			}
		}
	}
}

// TestWide16VertNEONBatchLengths sweeps the batched six/eight-tap vertical
// paths across lengths that hit every batch/remainder shape (full batches,
// partial batches, scalar tails) at 10-bit against the pure-Go reference
// (12-bit routes to pure Go before reaching the batched kernels and is
// covered by the dispatch differentials). The dispatch-level differentials
// cover the standard length set; this adds the batch-boundary lengths
// introduced by the four-group scratch.
func TestWide16VertNEONBatchLengths(t *testing.T) {
	lengths := []int{8, 16, 24, 32, 33, 40, 41, 47, 48, 56, 63, 64, 65, 72, 80, 96}
	const strideBytes = 128
	kernels := []struct {
		name string
		ref  func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
		neon func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	}{
		{"filter6", filter6Edge16PureGo, filter6Vert16NEON},
		{"filter8", filter8Edge16PureGo, filter8Vert16NEON},
	}
	var seed int64 = 97000
	for _, k := range kernels {
		for _, c := range wide16Corpus() {
			if c.bitDepth != 10 {
				continue
			}
			scale, params := wide16Params(c.bitDepth, c.limit, c.blimit, c.hev)
			for _, length := range lengths {
				rows := length + 8
				rng := rand.New(rand.NewSource(seed))
				base := make([]byte, strideBytes*rows)
				fillWide16Content(base, rng, int(seed)%5, (1<<c.bitDepth)-1)
				want := append([]byte(nil), base...)
				got := append([]byte(nil), base...)
				k.ref(want, 16*2, 2, strideBytes, length, scale, params)
				k.neon(got, 16*2, 2, strideBytes, length, scale, params)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%s len=%d params=%+v idx=%d got=%d want=%d", k.name, length, params, i, got[i], want[i])
					}
				}
				seed++
			}
		}
	}
}

// TestWide16VertNEONZeroAlloc guards that the batched 10-bit vertical paths
// (asm gather/scatter, stack scratch, remainder tail) allocate nothing.
func TestWide16VertNEONZeroAlloc(t *testing.T) {
	const strideBytes = 128
	const rows = 96
	base := make([]byte, strideBytes*rows)
	rng := rand.New(rand.NewSource(5))
	fillWide16Content(base, rng, 2, 1023)
	scale, params := wide16Params(10, 16, 40, 8)
	if n := testing.AllocsPerRun(100, func() {
		filter6Vert16NEON(base, 16*2, 2, strideBytes, 84, scale, params) // 10 groups + tail of 4
	}); n != 0 {
		t.Fatalf("filter6Vert16NEON allocs=%v", n)
	}
	if n := testing.AllocsPerRun(100, func() {
		filter8Vert16NEON(base, 16*2, 2, strideBytes, 84, scale, params)
	}); n != 0 {
		t.Fatalf("filter8Vert16NEON allocs=%v", n)
	}
}
