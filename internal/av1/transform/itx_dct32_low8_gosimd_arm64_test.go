// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package transform

import (
	"math/rand"
	"testing"
)

func TestInverseDCT32Col4Low8SIMDMatchesFullAndScalar(t *testing.T) {
	const prefix = 3
	const suffix = 4
	const canary = int32(0x51a7c0de)
	rng := rand.New(rand.NewSource(0x32_108))

	check := func(t *testing.T, stride int, min, max int32, values [32]int32) {
		t.Helper()
		span := 31*stride + 4
		newBacking := func() ([]int32, []int32) {
			backing := make([]int32, prefix+span+suffix)
			for i := range backing {
				backing[i] = canary
			}
			view := backing[prefix : prefix+span : prefix+span]
			for row := 0; row < 32; row++ {
				for lane := 0; lane < 4; lane++ {
					if row < 8 {
						view[row*stride+lane] = values[row*4+lane]
					} else {
						view[row*stride+lane] = 0
					}
				}
			}
			return backing, view
		}

		wantFull, fullView := newBacking()
		wantScalar, scalarView := newBacking()
		got, low8View := newBacking()
		inverseDCT32Col4SIMD(fullView, stride, min, max)
		inverseDCT32Col4PureGo(scalarView, stride, min, max)
		inverseDCT32Col4Low8SIMD(low8View, stride, min, max)
		for i := range wantFull {
			if wantFull[i] != wantScalar[i] {
				t.Fatalf("full/scalar mismatch stride=%d clamp=[%d,%d] backing[%d]=%d want %d", stride, min, max, i, wantFull[i], wantScalar[i])
			}
			if got[i] != wantFull[i] {
				t.Fatalf("low8/full mismatch stride=%d clamp=[%d,%d] backing[%d]=%d want %d", stride, min, max, i, got[i], wantFull[i])
			}
		}
		for i := 0; i < prefix; i++ {
			if got[i] != canary {
				t.Fatalf("prefix canary[%d]=%#x want %#x", i, got[i], canary)
			}
		}
		for i := prefix + span; i < len(got); i++ {
			if got[i] != canary {
				t.Fatalf("suffix canary[%d]=%#x want %#x", i, got[i], canary)
			}
		}
	}

	// Every signed-int16 value appears in the active input rows across these
	// batches, covering the kernel's full lane input domain.
	for batch := 0; batch < (1<<16)/32; batch++ {
		var values [32]int32
		for i := range values {
			values[i] = int32(int16(int32(-1<<15) + int32(batch*32+i)))
		}
		check(t, 4, minInt16, maxInt16, values)
	}

	for _, stride := range []int{4, 5, 13} {
		for _, clamp := range [][2]int32{{minInt16, maxInt16}, {-1024, 1023}, {-1, 1}, {0, maxInt16}} {
			for iter := 0; iter < 180; iter++ {
				var values [32]int32
				for i := range values {
					values[i] = clamp[0] + int32(rng.Int63n(int64(clamp[1])-int64(clamp[0])+1))
				}
				check(t, stride, clamp[0], clamp[1], values)
			}
		}
	}

	// Exercise every input row/lane independently at signed and rounding
	// boundaries, including values adjacent to int16 extrema.
	impulses := []int32{minInt16, minInt16 + 1, -16385, -9, -8, -7, -1, 0, 1, 7, 8, 9, 16383, maxInt16 - 1, maxInt16}
	for row := 0; row < 8; row++ {
		for lane := 0; lane < 4; lane++ {
			for _, value := range impulses {
				var values [32]int32
				values[row*4+lane] = value
				check(t, 4, minInt16, maxInt16, values)
			}
		}
	}
}

func TestInverseBlockBitDepthRawDCT32Low8RouteAndFallback(t *testing.T) {
	original := inverseDCT32Low8Col4Impl
	defer func() { inverseDCT32Low8Col4Impl = original }()

	type routeCase struct {
		name       string
		size       Size
		typ        Type
		bitDepth   uint8
		activeRows int
		wantRoute  bool
	}
	cases := []routeCase{
		{name: "width8 rows1", size: Size{Width: 8, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 1, wantRoute: true},
		{name: "width16 rows7", size: Size{Width: 16, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 7, wantRoute: true},
		{name: "width32 rows8", size: Size{Width: 32, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 8, wantRoute: true},
		{name: "width64 rows8", size: Size{Width: 64, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 8, wantRoute: true},
		{name: "vertical dct horizontal adst", size: Size{Width: 8, Height: 32}, typ: TypeDCTADST, bitDepth: 8, activeRows: 4, wantRoute: true},
		{name: "vertical dct identity horizontal", size: Size{Width: 8, Height: 32}, typ: TypeVDCT, bitDepth: 8, activeRows: 4, wantRoute: true},
		{name: "vertical identity", size: Size{Width: 32, Height: 32}, typ: TypeHDCT, bitDepth: 8, activeRows: 4},
		{name: "ten bit same column envelope", size: Size{Width: 16, Height: 32}, typ: TypeDCTDCT, bitDepth: 10, activeRows: 8, wantRoute: true},
		{name: "full sentinel", size: Size{Width: 32, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 0},
		{name: "above low rows", size: Size{Width: 32, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 9},
		{name: "wide adjusted scan above low rows", size: Size{Width: 64, Height: 32}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 9},
		{name: "twelve bit wider column range", size: Size{Width: 32, Height: 32}, typ: TypeDCTDCT, bitDepth: 12, activeRows: 8},
		{name: "short vertical", size: Size{Width: 16, Height: 16}, typ: TypeDCTDCT, bitDepth: 8, activeRows: 4},
		{name: "vertical adst", size: Size{Width: 8, Height: 16}, typ: TypeADSTDCT, bitDepth: 8, activeRows: 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.typ.Supported(tc.size) {
				t.Fatalf("test case uses unsupported type/size: %v %+v", tc.typ, tc.size)
			}
			scan := adjustedScanSize(tc.size)
			coeffStride := int(scan.Height) + 3
			coeff := make([]int32, (int(scan.Width)-1)*coeffStride+int(scan.Height))
			rng := rand.New(rand.NewSource(0x32_500 + int64(tc.activeRows) + int64(tc.size.Width)*7))
			active := tc.activeRows
			if active == 0 || active > int(scan.Height) {
				active = int(scan.Height)
			}
			for col := 0; col < int(scan.Width); col++ {
				for row := 0; row < active; row++ {
					coeff[col*coeffStride+row] = int32(rng.Intn(8191) - 4095)
				}
			}
			want := make([]int32, int(tc.size.Width)*int(tc.size.Height))
			got := make([]int32, len(want))
			baseline := inverseDCT32Low8Col4Impl
			inverseDCT32Low8Col4Impl = nil
			wantErr := InverseBlockBitDepthRaw(want, coeff, coeffStride, tc.size, tc.typ, tc.bitDepth, tc.activeRows)
			inverseDCT32Low8Col4Impl = baseline
			if wantErr != nil {
				t.Fatalf("baseline Raw: %v", wantErr)
			}
			calls := 0
			inverseDCT32Low8Col4Impl = func(buf []int32, rowStride int, min, max int32) {
				calls++
				original(buf, rowStride, min, max)
			}
			gotErr := InverseBlockBitDepthRaw(got, coeff, coeffStride, tc.size, tc.typ, tc.bitDepth, tc.activeRows)
			inverseDCT32Low8Col4Impl = baseline
			if gotErr != nil {
				t.Fatalf("candidate Raw: %v", gotErr)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("Raw mismatch at %d: got %d want %d", i, got[i], want[i])
				}
			}
			wantCalls := 0
			if tc.wantRoute {
				wantCalls = int(tc.size.Width) / 4
			}
			if calls != wantCalls {
				t.Fatalf("specialized kernel calls=%d want %d", calls, wantCalls)
			}
		})
	}

	// The optional slot is nil in pure-Go/other architectures; explicitly
	// verify that an eligible public Raw transform still uses the old path.
	size := Size{Width: 32, Height: 32}
	scan := adjustedScanSize(size)
	coeff := make([]int32, int(scan.Width)*int(scan.Height))
	coeff[0] = 17
	out := make([]int32, 32*32)
	inverseDCT32Low8Col4Impl = nil
	if err := InverseBlockBitDepthRaw(out, coeff, int(scan.Height), size, TypeDCTDCT, 8, 1); err != nil {
		t.Fatalf("unbound optional slot fallback: %v", err)
	}
}

func TestInverseBlockBitDepthRawDCT32Low8ReusesPoisonedDestination(t *testing.T) {
	original := inverseDCT32Low8Col4Impl
	if original == nil {
		t.Fatal("ARM64 low8 implementation is not bound")
	}
	defer func() { inverseDCT32Low8Col4Impl = original }()

	size := Size{Width: 32, Height: 32}
	scan := adjustedScanSize(size)
	stride := int(scan.Height)
	coeff := make([]int32, int(scan.Width)*stride)
	for col := 0; col < int(scan.Width); col++ {
		for row := 0; row < stride; row++ {
			coeff[col*stride+row] = int32((col*31+row*47)%4093 - 2046)
		}
	}
	shared := make([]int32, 32*32)
	if err := InverseBlockBitDepthRaw(shared, coeff, stride, size, TypeDCTDCT, 8, 0); err != nil {
		t.Fatalf("initial full Raw: %v", err)
	}
	poisoned := false
	for _, v := range shared {
		poisoned = poisoned || v != 0
	}
	if !poisoned {
		t.Fatal("full transform left the destination entirely zero")
	}

	low := make([]int32, len(coeff))
	for col := 0; col < int(scan.Width); col++ {
		low[col*stride] = int32(col*23 - 300)
	}
	want := make([]int32, 32*32)
	prev := inverseDCT32Low8Col4Impl
	inverseDCT32Low8Col4Impl = nil
	err := InverseBlockBitDepthRaw(want, low, stride, size, TypeDCTDCT, 8, 0)
	inverseDCT32Low8Col4Impl = prev
	if err != nil {
		t.Fatalf("independent full-sentinel Raw oracle: %v", err)
	}
	calls := 0
	inverseDCT32Low8Col4Impl = func(buf []int32, rowStride int, min, max int32) {
		calls++
		original(buf, rowStride, min, max)
	}
	err = InverseBlockBitDepthRaw(shared, low, stride, size, TypeDCTDCT, 8, 1)
	inverseDCT32Low8Col4Impl = prev
	if err != nil {
		t.Fatalf("low8 Raw after full transform: %v", err)
	}
	if calls != 8 {
		t.Fatalf("low8 kernel calls=%d want 8", calls)
	}
	for i := range want {
		if shared[i] != want[i] {
			t.Fatalf("reused destination[%d]=%d want %d", i, shared[i], want[i])
		}
	}
}

func TestInverseBlockBitDepthRawDCT32Low8NoAlloc(t *testing.T) {
	size := Size{Width: 32, Height: 32}
	scan := adjustedScanSize(size)
	coeff := make([]int32, int(scan.Width)*int(scan.Height))
	for col := 0; col < int(scan.Width); col++ {
		for row := 0; row < 8; row++ {
			coeff[col*int(scan.Height)+row] = int32((col*19+row*7)%511 - 255)
		}
	}
	dst := make([]int32, 32*32)
	allocs := testing.AllocsPerRun(100, func() {
		if err := InverseBlockBitDepthRaw(dst, coeff, int(scan.Height), size, TypeDCTDCT, 8, 8); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Raw DCT32 low8 allocs/op=%g want 0", allocs)
	}
}

func TestInverseDCT32Low8AdapterFallsBackWhenClampMovesZero(t *testing.T) {
	const stride = 5
	const span = 31*stride + 4
	input := make([]int32, span)
	for row := 0; row < 8; row++ {
		for lane := 0; lane < 4; lane++ {
			input[row*stride+lane] = int32(row*11 + lane - 20)
		}
	}
	full := append([]int32(nil), input...)
	low8 := append([]int32(nil), input...)
	inverseDCT32Col4SIMDAdapter(full, stride, 1, 32767)
	inverseDCT32Low8Col4SIMDAdapter(low8, stride, 1, 32767)
	for i := range full {
		if low8[i] != full[i] {
			t.Fatalf("clamp=[1,32767] fallback mismatch at %d: got %d want %d", i, low8[i], full[i])
		}
	}
}
