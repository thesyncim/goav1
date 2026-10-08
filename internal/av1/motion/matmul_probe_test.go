// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"simd/archsimd"
	"testing"
)

// TestMotionSIMDOfficialDotProductLaneOrder keeps the official-API USDOT
// composition tied to the instruction contract: each int32 lane sums one
// consecutive group of four unsigned samples times signed taps, then adds its
// independent accumulator lane.
func TestMotionSIMDOfficialDotProductLaneOrder(t *testing.T) {
	var sampleBytes [16]uint8
	var tapBytes [16]int8
	for i := range sampleBytes {
		sampleBytes[i] = uint8(129 + 7*i)
		tapBytes[i] = int8((i*5)%15 - 7)
	}
	seedValues := [4]int32{13, -29, 101, -307}
	samples := archsimd.LoadUint8x16Array(&sampleBytes)
	taps := archsimd.LoadInt8x16Array(&tapBytes)
	seed := archsimd.LoadInt32x4Array(&seedValues)
	var got [4]int32
	simdDotProdUS(seed, samples, taps).StoreArray(&got)
	for lane := range got {
		want := seedValues[lane]
		for tap := range 4 {
			i := lane*4 + tap
			want += int32(sampleBytes[i]) * int32(tapBytes[i])
		}
		if got[lane] != want {
			t.Fatalf("lane %d = %d, want %d", lane, got[lane], want)
		}
	}
}

func TestMotionSIMDOfficialWideningAndLaneJoins(t *testing.T) {
	var a, b [8]int16
	for i := range 4 {
		a[i] = int16(11 + 11*i)
		b[i] = int16(55 + 11*i)
	}
	var got16 [8]int16
	simdConcatInt16x8(archsimd.LoadInt16x8Array(&a), archsimd.LoadInt16x8Array(&b)).StoreArray(&got16)
	for i := range 4 {
		if got16[i] != a[i] || got16[i+4] != b[i] {
			t.Fatalf("int16 join = %v, want low=%v high=%v", got16, a[:4], b[:4])
		}
	}

	var raw [16]uint8
	var kernel [filterTaps]int16
	for i := range raw {
		raw[i] = uint8(31 + 13*i)
	}
	for i := range kernel {
		kernel[i] = int16((i*29)%127 - 61)
	}
	rawVector := archsimd.LoadUint8x16Array(&raw)
	lo := archsimd.BroadcastInt32x4(1234)
	hi := archsimd.BroadcastInt32x4(1234)
	for tap := range filterTaps {
		samples := rawVector.ConcatShiftBytesRight(rawVector, uint64(tap)).ExtendLo8ToUint16().ConvertToInt16()
		coeff := archsimd.BroadcastInt16x8(kernel[tap])
		lo, hi = simdHorizontalConvMAC(lo, hi, samples, coeff)
	}
	var gotLo, gotHi [4]int32
	lo.StoreArray(&gotLo)
	hi.StoreArray(&gotHi)
	for lane := range 8 {
		want := int32(1234)
		for tap := range filterTaps {
			want += int32(raw[lane+tap]) * int32(kernel[tap])
		}
		var got int32
		if lane < 4 {
			got = gotLo[lane]
		} else {
			got = gotHi[lane-4]
		}
		if got != want {
			t.Fatalf("horizontal lane %d = %d, want %d", lane, got, want)
		}
	}
}

func TestMotionSIMDOfficialRoundingAndSaturation(t *testing.T) {
	values := [8]int16{32767, 32766, -32768, -32767, -3, -2, -1, 0}
	var got16 [8]int16
	simdRoundShiftInt16(archsimd.LoadInt16x8Array(&values), 1).StoreArray(&got16)
	for i, x := range values {
		want := int16((int32(x) + 1) >> 1)
		if got16[i] != want {
			t.Fatalf("rounded int16 lane %d = %d, want %d", i, got16[i], want)
		}
	}

	wideLo := archsimd.LoadInt32x4Array(&[4]int32{mathMaxInt32, mathMinInt32, -3, -1})
	wideHi := archsimd.LoadInt32x4Array(&[4]int32{32767, -32768, 5, 7})
	simdRoundShiftNarrowInt32Pair(wideLo, wideHi, 1).StoreArray(&got16)
	want16 := [8]int16{32767, -32768, -1, 0, 16384, -16384, 3, 4}
	for i := range got16 {
		if got16[i] != want16[i] {
			t.Fatalf("rounded narrow lane %d = %d, want %d", i, got16[i], want16[i])
		}
	}

	var low, high [8]int16
	low = [8]int16{-1, 0, 1, 255, 256, 511, 32767, -32768}
	high = [8]int16{10, 20, 30, 40, 50, 60, 70, 80}
	lo8 := simdRoundShiftNarrowUint8(archsimd.LoadInt16x8Array(&low), 0)
	hi8 := simdRoundShiftNarrowUint8(archsimd.LoadInt16x8Array(&high), 0)
	var joined [16]uint8
	simdConcatUint8x16(lo8, hi8).StoreArray(&joined)
	wantBytes := [16]uint8{0, 0, 1, 255, 255, 255, 255, 0, 10, 20, 30, 40, 50, 60, 70, 80}
	if joined != wantBytes {
		t.Fatalf("uint8 join/saturate = %v, want %v", joined, wantBytes)
	}
}

const (
	mathMaxInt32 = int32(1<<31 - 1)
	mathMinInt32 = -1 << 31
)
