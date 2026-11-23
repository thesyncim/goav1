// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix && goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
)

// TestPixelStatsRows16StaysInsideGuardedRows is the guard-page check of the
// SIMD-only row statistics kernel (see TestEncoderKernelsStayInsideGuardedBlocks).
func TestPixelStatsRows16StaysInsideGuardedRows(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	for _, before := range []bool{false, true} {
		for _, pad := range []int{0, 24} {
			name := fmt.Sprintf("before=%v pad=%d", before, pad)
			for _, w := range []int{16, 32, 48, 64} {
				for _, h := range []int{1, 7, 16, 64} {
					ss, rs := w+pad, w+3*pad
					s, r := guardBytes(t, rng, blockLen(w, h, ss), before), guardBytes(t, rng, blockLen(w, h, rs), before)
					var e, m uint32
					var c, d int32
					guardpage.Check(t, fmt.Sprintf("%s rows16 %dx%d", name, w, h), func() { e, c = pixelStatsRows16(s, ss, r, rs, w, h) })
					m, d = pixelStatsRows16(slices.Clone(s), ss, slices.Clone(r), rs, w, h)
					if e != m || c != d {
						t.Fatalf("%s rows16 %dx%d differs", name, w, h)
					}
				}
			}
		}
	}
}
