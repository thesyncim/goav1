// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Differential coverage for the fourteen-sample vertical-edge NEON transpose
// kernels (filter14_vtrn_neon_arm64.s). These drive the arm64 asm directly, so
// the file is constrained to that build.
//go:build arm64 && !purego

package loopfilter

import (
	"math/rand"
	"testing"
)

// TestFilter14VertTransposeGatherScatter drives the gather/scatter asm kernels
// against the scalar transpose loops they replaced, byte for byte, across
// strides, counts, and both sample widths. The scatter check first fills the
// modifiable tap rows (1..12) of the scratch with fresh random content so the
// scattered columns are distinguishable from the gathered ones.
func TestFilter14VertTransposeGatherScatter(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for _, bytesPerSample := range []int{1, 2} {
		step := bytesPerSample
		groupCols := 8 * bytesPerSample // bytes per group column offset in scratch
		scratchStride := filter14VertBatchGroups * groupCols
		for _, stride := range []int{16 * bytesPerSample, 96, 640} {
			for count := 1; count <= filter14VertBatchGroups; count++ {
				pix := make([]byte, 8*filter14VertBatchGroups*stride+64)
				for i := range pix {
					pix[i] = byte(rng.Intn(256))
				}
				q0Base := 7 * step // p6 of position 0 at byte 0
				scratch := make([]byte, 14*scratchStride)
				wantScratch := make([]byte, 14*scratchStride)
				ctx := wideVertTransposeCtx{
					src:     &pix[q0Base-7*step],
					stride:  uintptr(stride),
					scratch: &scratch[0],
					count:   uintptr(count),
				}
				// Reference gather (the old scalar loops).
				for j := 0; j < 8*count; j++ {
					src := q0Base + j*stride - 7*step
					for tap := 0; tap < 14; tap++ {
						for b := 0; b < bytesPerSample; b++ {
							wantScratch[tap*scratchStride+j*bytesPerSample+b] = pix[src+tap*step+b]
						}
					}
				}
				if bytesPerSample == 1 {
					filter14VertGatherNEONAsm(&ctx)
				} else {
					filter14Vert16GatherNEONAsm(&ctx)
				}
				for tap := 0; tap < 14; tap++ {
					for j := 0; j < 8*count*bytesPerSample; j++ {
						i := tap*scratchStride + j
						if scratch[i] != wantScratch[i] {
							t.Fatalf("gather bps=%d stride=%d count=%d tap=%d col=%d got=%d want=%d",
								bytesPerSample, stride, count, tap, j, scratch[i], wantScratch[i])
						}
					}
				}
				// Scatter: randomize the modifiable rows, then compare the
				// asm scatter against the scalar loops on a copy.
				for tap := 1; tap <= 12; tap++ {
					for j := 0; j < 8*count*bytesPerSample; j++ {
						scratch[tap*scratchStride+j] = byte(rng.Intn(256))
					}
				}
				wantPix := append([]byte(nil), pix...)
				for j := 0; j < 8*count; j++ {
					dst := q0Base + j*stride - 7*step
					for tap := 1; tap <= 12; tap++ {
						for b := 0; b < bytesPerSample; b++ {
							wantPix[dst+tap*step+b] = scratch[tap*scratchStride+j*bytesPerSample+b]
						}
					}
				}
				if bytesPerSample == 1 {
					filter14VertScatterNEONAsm(&ctx)
				} else {
					filter14Vert16ScatterNEONAsm(&ctx)
				}
				for i := range pix {
					if pix[i] != wantPix[i] {
						t.Fatalf("scatter bps=%d stride=%d count=%d idx=%d got=%d want=%d",
							bytesPerSample, stride, count, i, pix[i], wantPix[i])
					}
				}
			}
		}
	}
}

// TestFilter14VertNEONBatchLengths sweeps the batched vertical path across
// lengths that hit every batch/remainder shape (full batches, partial batches,
// scalar tails) at 8-bit against the pure-Go reference. The dispatch-level
// differentials cover the standard length set; this adds the batch-boundary
// lengths introduced by the four-group scratch.
func TestFilter14VertNEONBatchLengths(t *testing.T) {
	lengths := []int{8, 16, 24, 32, 33, 40, 41, 47, 48, 56, 63, 64, 65, 72, 80, 96}
	const stride = 128
	var seed int64 = 90000
	for _, params := range wideFilterParamsCorpus() {
		for _, length := range lengths {
			rows := length + 8
			rng := rand.New(rand.NewSource(seed))
			base := make([]byte, stride*rows)
			fillWideContent(base, rng, int(seed)%5)
			want := append([]byte(nil), base...)
			got := append([]byte(nil), base...)
			filter14EdgePureGo(want, 16, 1, stride, length, 1, params)
			filter14VertNEON(got, 16, 1, stride, length, 1, params)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("len=%d params=%+v idx=%d got=%d want=%d", length, params, i, got[i], want[i])
				}
			}
			seed++
		}
	}
}

// TestFilter14Vert16NEONBatchLengths mirrors the batch-length sweep for the
// 10-bit vertical path (12-bit routes to pure Go before reaching the batched
// kernels and is covered by the dispatch differentials).
func TestFilter14Vert16NEONBatchLengths(t *testing.T) {
	lengths := []int{8, 16, 24, 32, 33, 40, 41, 47, 48, 56, 63, 64, 65, 72, 80, 96}
	const strideBytes = 128
	var seed int64 = 95000
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
			filter14Edge16PureGo(want, 16*2, 2, strideBytes, length, scale, params)
			filter14Vert16NEON(got, 16*2, 2, strideBytes, length, scale, params)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("len=%d params=%+v idx=%d got=%d want=%d", length, params, i, got[i], want[i])
				}
			}
			seed++
		}
	}
}

// TestFilter14VertNEONZeroAlloc guards that the batched 8-bit vertical path
// (asm gather/scatter, stack scratch, remainder tail) allocates nothing.
func TestFilter14VertNEONZeroAlloc(t *testing.T) {
	const stride = 128
	const rows = 96
	base := make([]byte, stride*rows)
	rng := rand.New(rand.NewSource(3))
	fillWideContent(base, rng, 2)
	params := wideFilterParamsCorpus()[4]
	if n := testing.AllocsPerRun(100, func() {
		filter14VertNEON(base, 16, 1, stride, 84, 1, params) // 10 groups + tail of 4
	}); n != 0 {
		t.Fatalf("filter14VertNEON allocs=%v", n)
	}
}
