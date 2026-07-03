// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package transform

import "testing"

// These benchmarks call the four-lane AVX2 adapters directly (bypassing the
// cpu.Detected.AVX2 gate) so they measure the kernels even under Rosetta 2,
// which executes AVX2 but does not advertise it. Compare each against the
// matching *PureGo benchmark in colpass/rowpass_simd_bench_test.go.

func BenchmarkDCT32Col4AVX2(b *testing.B) { benchmarkCol4(b, dct32Size, inverseDCT32Col4AVX2Adapter) }
func BenchmarkDCT64Col4AVX2(b *testing.B) { benchmarkCol4(b, dct64Size, inverseDCT64Col4AVX2Adapter) }
func BenchmarkDCT32Row4AVX2(b *testing.B) { benchmarkRow4(b, dct32Size, inverseDCT32Row4AVX2Adapter) }
func BenchmarkDCT64Row4AVX2(b *testing.B) { benchmarkRow4(b, dct64Size, inverseDCT64Row4AVX2Adapter) }
