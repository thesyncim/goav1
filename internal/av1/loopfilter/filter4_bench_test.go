// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package loopfilter

import (
	"math/rand"
	"testing"
)

// benchFilter4Edge sets up a horizontal-edge buffer of edgeLen contiguous
// positions (outer == 1, step == stride) with thresholds chosen so most
// positions pass needsFilter4 and split across the hev branch, the worst case
// for the kernel.
func benchFilter4Edge(edgeLen int) ([]byte, int, int, int, filter4Params) {
	const stride = 256
	buf := make([]byte, stride*8)
	rng := rand.New(rand.NewSource(99))
	for i := range buf {
		// Keep adjacent rows close so needsFilter4 passes for most positions.
		buf[i] = byte(100 + rng.Intn(40))
	}
	params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
	q0Base := 3 * stride
	return buf, q0Base, stride, edgeLen, params
}

// BenchmarkFilter4EdgePureGo and BenchmarkFilter4EdgeDispatch bracket the
// per-kernel speedup: the first always runs the pure-Go reference, the second
// runs whatever the dispatcher resolved (NEON on arm64). Both time only the
// inner edge kernel, not the validation wrapper.
func BenchmarkFilter4EdgePureGo(b *testing.B) {
	buf, q0Base, step, length, params := benchFilter4Edge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter4EdgePureGo(buf, q0Base, step, 1, length, params)
	}
}

func BenchmarkFilter4EdgeDispatch(b *testing.B) {
	buf, q0Base, step, length, params := benchFilter4Edge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter4EdgeImpl(buf, q0Base, step, 1, length, params)
	}
}
