// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package frame

import (
	"fmt"
	"testing"
)

// BenchmarkLoadSampleRows8Shapes covers CDEF-sized rectangles and a full
// frame through the same dispatch that production callers use.
func BenchmarkLoadSampleRows8Shapes(b *testing.B) {
	for _, shape := range [...]struct{ width, height int }{{8, 8}, {16, 16}, {32, 32}, {64, 64}, {1280, 720}} {
		b.Run(fmt.Sprintf("%dx%d", shape.width, shape.height), func(b *testing.B) {
			srcStride := shape.width + 16
			dstStride := shape.width + 32
			src := make([]byte, srcStride*shape.height)
			dst := make([]uint16, dstStride*shape.height)
			for i := range src {
				src[i] = byte(i * 13)
			}
			b.SetBytes(int64(shape.width * shape.height))
			b.ReportAllocs()
			for b.Loop() {
				loadSampleRows8(dst, dstStride, src, srcStride, shape.width, shape.height)
			}
		})
	}
}
