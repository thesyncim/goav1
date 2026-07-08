// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

// TestInverseDCT8Col2GoSIMDMatchesScalar proves the Go-native-SIMD 8-point
// inverse DCT column pass is byte-identical to the scalar reference across
// bit-depth ranges (including 12-bit, which overflows int32 intermediates) and
// content that exercises the clipRange clamps.
func TestInverseDCT8Col2GoSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0xdc78))
	ranges := [][2]int32{
		{-(1 << 16), (1 << 16) - 1},
		{-(1 << 18), (1 << 18) - 1},
		{-(1 << 20), (1 << 20) - 1},
	}
	for iter := 0; iter < 20000; iter++ {
		r := ranges[rng.Intn(len(ranges))]
		min, max := r[0], r[1]
		stride := 4 + rng.Intn(5) // callers pass stride == transform width (>= 4)
		a := make([]int32, 8*stride)
		for k := 0; k < 8; k++ {
			for col := 0; col < 2; col++ {
				var v int32
				switch rng.Intn(5) {
				case 0:
					v = min
				case 1:
					v = max
				case 2:
					v = int32(rng.Intn(1<<12) - (1 << 11))
				default:
					v = min + int32(rng.Int63n(int64(max)-int64(min)+1))
				}
				a[k*stride+col] = v
			}
		}
		b := make([]int32, len(a))
		copy(b, a)
		inverseDCT8Col2PureGo(a, stride, min, max)
		inverseDCT8Col2SIMD(b, stride, min, max)
		for k := 0; k < 8; k++ {
			for col := 0; col < 2; col++ {
				i := k*stride + col
				if a[i] != b[i] {
					t.Fatalf("iter=%d range=[%d,%d] stride=%d row=%d col=%d: scalar=%d simd=%d",
						iter, min, max, stride, k, col, a[i], b[i])
				}
			}
		}
	}
}

func benchDCT8Col2(b *testing.B, fn func([]int32, int, int32, int32)) {
	rng := rand.New(rand.NewSource(7))
	const stride = 8
	buf := make([]int32, 8*stride)
	for i := range buf {
		buf[i] = int32(rng.Intn(1<<17) - (1 << 16))
	}
	work := make([]int32, len(buf))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(work, buf)
		fn(work, stride, -(1 << 16), (1<<16)-1)
	}
}

func BenchmarkDCT8Col2_Scalar(b *testing.B) { benchDCT8Col2(b, inverseDCT8Col2PureGo) }
func BenchmarkDCT8Col2_GoSIMD(b *testing.B) { benchDCT8Col2(b, inverseDCT8Col2SIMD) }
func BenchmarkDCT8Col2_ASM(b *testing.B)    { benchDCT8Col2(b, inverseDCT8Col2NEONAdapter) }
