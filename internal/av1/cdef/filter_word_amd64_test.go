// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package cdef

import (
	"math/rand"
	"reflect"
	"simd/archsimd"
	"testing"
)

func requireAVX2(tb testing.TB) {
	tb.Helper()
	if !archsimd.X86.AVX2() {
		tb.Skip("CPU does not advertise AVX2")
	}
}

// TestAMD64CDEFWordDispatch asserts the block dispatch slots resolve to the
// 256-bit word kernels on AVX2 CPUs.
func TestAMD64CDEFWordDispatch(t *testing.T) {
	requireAVX2(t)
	if reflect.ValueOf(filterBlockImpl).Pointer() != reflect.ValueOf(filterBlockAVX2).Pointer() {
		t.Fatal("filterBlockImpl is not the AVX2 word kernel")
	}
	if reflect.ValueOf(filterBlockU8Impl).Pointer() != reflect.ValueOf(filterBlockU8AVX2).Pointer() {
		t.Fatal("filterBlockU8Impl is not the AVX2 word kernel")
	}
}

// TestAMD64CDEFWordMatchesPureGo checks the 256-bit word kernels (uint16 and
// 8-bit stores) against the pure-Go references: uniform inputs salted with
// the VeryLarge sentinel, and narrow-spread neighbourhoods that drive the
// [min, max] clamp. Every trial shape is one the word kernels take.
func TestAMD64CDEFWordMatchesPureGo(t *testing.T) {
	requireAVX2(t)
	rng := rand.New(rand.NewSource(0x7ab1e5))
	input := make([]uint16, BStride*24)
	origin := BStride*8 + 16
	shapes := [][2]int{{8, 8}, {8, 4}, {4, 8}, {4, 4}, {8, 2}}
	for trial := 0; trial < 12000; trial++ {
		coeffShift := rng.Intn(3)
		u8 := trial&1 == 0
		if u8 {
			coeffShift = 0
		}
		maxSample := (1 << (8 + coeffShift)) - 1
		clipCorpus := trial%3 == 0
		base, spread := rng.Intn(maxSample+1), 1<<rng.Intn(6)
		for i := range input {
			v := rng.Intn(maxSample + 1)
			if clipCorpus {
				v = min(base+rng.Intn(spread+1), maxSample)
			}
			if rng.Intn(9) == 0 {
				v = VeryLarge
			}
			input[i] = uint16(v)
		}
		if u8 {
			// The 8-bit contract: real samples in the block interior.
			for r := 0; r < 8; r++ {
				for c := 0; c < 8; c++ {
					if input[origin+r*BStride+c] == VeryLarge {
						input[origin+r*BStride+c] = uint16(rng.Intn(256))
					}
				}
			}
		}
		shape := shapes[rng.Intn(len(shapes))]
		pri := uint8(rng.Intn(16))
		sec := []uint8{0, 1, 2, 4}[rng.Intn(4)]
		if clipCorpus {
			pri, sec = uint8(1+rng.Intn(15)), uint8(1+rng.Intn(4))
		}
		params := BlockFilterParams{
			PrimaryStrength:   pri << coeffShift,
			SecondaryStrength: sec << coeffShift,
			Direction:         uint8(rng.Intn(8)),
			PrimaryDamping:    uint8(3 + coeffShift + rng.Intn(4)),
			SecondaryDamping:  uint8(3 + coeffShift + rng.Intn(4)),
			CoeffShift:        uint8(coeffShift),
			Width:             uint8(shape[0]),
			Height:            uint8(shape[1]),
		}
		if !cdefWordShapeOK(params) {
			t.Fatalf("shape %v not taken by the word kernels", shape)
		}
		if u8 {
			want := make([]byte, 16*16)
			got := make([]byte, 16*16)
			for i := range want {
				want[i] = byte(i)
				got[i] = byte(i)
			}
			filterBlockU8PureGo(want, 16, 0, input, origin, params)
			filterBlockU8AVX2(got, 16, 0, input, origin, params)
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("trial %d u8 params=%+v: dst[%d] avx2 %d want %d", trial, params, i, got[i], want[i])
				}
			}
			continue
		}
		want := make([]uint16, 16*16)
		got := make([]uint16, 16*16)
		filterBlockPureGo(want, 16, 0, input, origin, params)
		filterBlockAVX2(got, 16, 0, input, origin, params)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d params=%+v: dst[%d] avx2 %d want %d", trial, params, i, got[i], want[i])
			}
		}
	}
}

// BenchmarkAMD64CDEFBlock times one block through the 256-bit word kernels
// (impl=avx2) against the 128-bit Go SIMD kernels (impl=generic) on the
// uint16 tap buffer.
func BenchmarkAMD64CDEFBlock(b *testing.B) {
	input := make([]uint16, BStride*24)
	for i := range input {
		input[i] = uint16(i * 7 % 256)
	}
	dst := make([]byte, 16*16)
	dst16 := make([]uint16, 16*16)
	for _, s := range []struct {
		name     string
		w, h     uint8
		pri, sec uint8
		hbd      bool
	}{
		{"8x8/prisec", 8, 8, 9, 2, false},
		{"8x8/pri", 8, 8, 9, 0, false},
		{"8x8/sec", 8, 8, 0, 2, false},
		{"4x4/prisec", 4, 4, 9, 2, false},
		{"4x8/prisec", 4, 8, 9, 2, false},
		{"8x8/prisec/hbd", 8, 8, 36, 8, true},
		{"4x4/prisec/hbd", 4, 4, 36, 8, true},
	} {
		p := BlockFilterParams{PrimaryStrength: s.pri, SecondaryStrength: s.sec, Direction: 5, PrimaryDamping: 5, SecondaryDamping: 5, Width: s.w, Height: s.h}
		if s.hbd {
			p.CoeffShift, p.PrimaryDamping, p.SecondaryDamping = 2, 7, 7
		}
		for _, impl := range []string{"generic", "avx2"} {
			b.Run("shape="+s.name+"/impl="+impl, func(b *testing.B) {
				if impl == "avx2" {
					requireAVX2(b)
				}
				for b.Loop() {
					switch {
					case s.hbd && impl == "avx2":
						filterBlockAVX2(dst16, 16, 0, input, BStride*8+16, p)
					case s.hbd:
						filterBlockSIMD(dst16, 16, 0, input, BStride*8+16, p)
					case impl == "avx2":
						filterBlockU8AVX2(dst, 16, 0, input, BStride*8+16, p)
					default:
						filterBlockU8SIMD(dst, 16, 0, input, BStride*8+16, p)
					}
				}
			})
		}
	}
}
