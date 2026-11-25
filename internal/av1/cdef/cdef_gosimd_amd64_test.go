//go:build goexperiment.simd && amd64 && !purego

package cdef

import (
	"math/rand"
	"reflect"
	"simd/archsimd"
	"testing"
)

// directionSIMDBound reports whether init bound the AVX2 direction kernels.
func directionSIMDBound() bool { return archsimd.X86.AVX2() }

// The AVX2 Go SIMD direction kernels are called directly (not through the
// dispatch slot), so they are checked against the scalar reference regardless
// of CPU feature detection.

func TestFindDirectionAVX2MatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x41565832)
	for coeffShift := range 5 {
		max := uint16((1 << (8 + coeffShift)) - 1)
		for _, stride := range []int{8, 9, 13, 16, 31} {
			for iter := range 128 {
				img := make([]uint16, stride*8)
				for i := range img {
					img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
				gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
				if gotDir != wantDir || gotVar != wantVar {
					t.Fatalf("coeffShift=%d stride=%d iter=%d dir,var=%d,%d want %d,%d", coeffShift, stride, iter, gotDir, gotVar, wantDir, wantVar)
				}
			}
		}
	}
}

// TestFindDirectionAVX2Directional exercises every one of the 8 directions with
// deterministic gradient patterns so the argmax and tie-break paths are all hit.
func TestFindDirectionAVX2Directional(t *testing.T) {
	const stride = 8
	patterns := []func(row, col int) uint16{
		func(row, col int) uint16 { return uint16((row + col) * 16) },       // diag 0
		func(row, col int) uint16 { return uint16((2*row + col) * 8) },      // dir 1
		func(row, col int) uint16 { return uint16(row * 32) },               // horizontal 2
		func(row, col int) uint16 { return uint16((2*row - col + 8) * 8) },  // dir 3
		func(row, col int) uint16 { return uint16((row - col + 8) * 16) },   // anti-diag 4
		func(row, col int) uint16 { return uint16((row - 2*col + 16) * 8) }, // dir 5
		func(row, col int) uint16 { return uint16(col * 32) },               // vertical 6
		func(row, col int) uint16 { return uint16((-row + 2*col + 8) * 8) }, // dir 7
		func(row, col int) uint16 { return 128 },                            // constant -> dir 0, var 0
	}
	for coeffShift := range 5 {
		for pi, fill := range patterns {
			img := make([]uint16, stride*8)
			for row := range 8 {
				for col := range 8 {
					img[row*stride+col] = fill(row, col) << coeffShift
				}
			}
			wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
			gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("pattern=%d coeffShift=%d dir,var=%d,%d want %d,%d", pi, coeffShift, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

func TestFindDirectionDualAVX2MatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x44554132)
	for coeffShift := range 5 {
		max := uint16((1 << (8 + coeffShift)) - 1)
		for _, stride := range []int{16, 19, 32} {
			for iter := range 96 {
				img := make([]uint16, stride*8)
				for i := range img {
					img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				wantDir1, wantVar1, wantDir2, wantVar2 := findDirectionDualScalar(img, img[8:], stride, coeffShift)
				gotDir1, gotVar1, gotDir2, gotVar2 := findDirectionDualSIMD(img, img[8:], stride, coeffShift)
				if gotDir1 != wantDir1 || gotVar1 != wantVar1 || gotDir2 != wantDir2 || gotVar2 != wantVar2 {
					t.Fatalf("coeffShift=%d stride=%d iter=%d dual=(%d,%d),(%d,%d) want (%d,%d),(%d,%d)",
						coeffShift, stride, iter,
						gotDir1, gotVar1, gotDir2, gotVar2,
						wantDir1, wantVar1, wantDir2, wantVar2)
				}
			}
		}
	}
}

func TestFindDirectionAVX2ZeroAlloc(t *testing.T) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 37) & 0xfff)
	}
	var dir int32
	var variance int32
	allocs := testing.AllocsPerRun(1000, func() {
		dir, variance = int32(0), int32(0)
		d, v := findDirectionSIMD(img, 8, 4)
		dir, variance = int32(d), v
	})
	_ = dir
	_ = variance
	if allocs != 0 {
		t.Fatalf("findDirectionSIMD allocated: %f", allocs)
	}
}

// TestFindDirectionU8AVX2MatchesScalar pins the AVX2-backed 8-bit direction
// wrapper (single and dual) against the scalar uint8 reference.
func TestFindDirectionU8AVX2MatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x38445236)
	for _, stride := range []int{8, 23, 320} {
		for iter := range 128 {
			img := make([]byte, stride*8+8)
			for i := range img {
				switch iter % 3 {
				case 0:
					img[i] = byte(rnd.generate(256))
				case 1:
					img[i] = byte(rnd.generate(5))
				default:
					img[i] = byte(251 + rnd.generate(5))
				}
			}
			wantDir, wantVar := findDirectionU8Scalar(img, stride)
			gotDir, gotVar := findDirectionU8SIMD(img, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
			if stride >= 16 {
				wd1, wv1, wd2, wv2 := findDirectionDualU8Scalar(img, img[8:], stride)
				gd1, gv1, gd2, gv2 := findDirectionDualU8SIMD(img, img[8:], stride)
				if gd1 != wd1 || gv1 != wv1 || gd2 != wd2 || gv2 != wv2 {
					t.Fatalf("dual: stride=%d iter=%d got=(%d,%d,%d,%d) want=(%d,%d,%d,%d)",
						stride, iter, gd1, gv1, gd2, gv2, wd1, wv1, wd2, wv2)
				}
			}
		}
	}
}

// BenchmarkAMD64CDEFDirection times the 8-bit direction search, dav1d's
// paired-lane AVX2 cdef_dir port.
func BenchmarkAMD64CDEFDirection(b *testing.B) {
	if !directionSIMDBound() {
		b.Skip("CPU does not advertise AVX2")
	}
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x1234)
	const stride = 64
	img := make([]byte, stride*8+16)
	for i := range img {
		img[i] = byte(rnd.generate(256))
	}
	var sink int
	b.Run("shape=8x8x2/impl=avx2", func(b *testing.B) {
		for b.Loop() {
			d1, _, d2, _ := findDirectionDualU8SIMD(img, img[8:], stride)
			sink += d1 + d2
		}
	})
	_ = sink
}

// BenchmarkAMD64CDEFDirection16 times the high-bitdepth direction search,
// dav1d's cdef_dir_16bpc port, at 10 bits.
func BenchmarkAMD64CDEFDirection16(b *testing.B) {
	if !directionSIMDBound() {
		b.Skip("CPU does not advertise AVX2")
	}
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x5678)
	const stride = 64
	img := make([]uint16, stride*8+16)
	for i := range img {
		img[i] = uint16(rnd.generate(1024))
	}
	var sink int
	b.Run("shape=8x8x2/impl=avx2", func(b *testing.B) {
		for b.Loop() {
			d1, _, d2, _ := findDirectionDualSIMD(img, img[8:], stride, 2)
			sink += d1 + d2
		}
	})
	_ = sink
}

// TestAMD64CDEFByteRouteDispatch asserts that an AVX2 CPU takes the byte-lane
// interior route (dav1d's 8bpc edges == 0xf kernels) instead of declining to
// the uint16 tap buffer.
func TestAMD64CDEFByteRouteDispatch(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("CPU does not advertise AVX2")
	}
	if !U8ByteInteriorAvailable() {
		t.Fatal("AVX2 CPU declines the byte-interior CDEF route")
	}
}

// BenchmarkAMD64CDEFUnitU8 times a full 8-bit unit through the byte-lane AVX2
// route (impl=avx2) against the uint16 tap buffer route amd64 used before
// (impl=generic: FilterFrameBlocksU8Trusted with the 128-bit block kernel).
// Directions are precomputed, so only the filter is timed.
func BenchmarkAMD64CDEFUnitU8(b *testing.B) {
	type shape struct {
		name       string
		dec        uint8
		plane      Plane
		level, sec uint8
	}
	for _, s := range []shape{
		{"luma8x8/prisec", 0, PlaneY, 6, 2},
		{"luma8x8/pri", 0, PlaneY, 6, 0},
		{"luma8x8/sec", 0, PlaneY, 0, 2},
		{"chroma4x4/prisec", 1, PlaneU, 5, 1},
		{"chroma4x4/pri", 1, PlaneU, 5, 0},
	} {
		rng := byteUnitRNG(0x1234567)
		unitW, unitH := BlockSize>>s.dec, BlockSize>>s.dec
		bw, bh := 8>>s.dec, 8>>s.dec
		input, wide, origin := makeByteUnitInput(&rng, unitW, unitH, 5)
		var blocks []BlockPosition
		for by := 0; by < unitH/bh; by++ {
			for bx := 0; bx < unitW/bw; bx++ {
				blocks = append(blocks, BlockPosition{BX: uint8(bx), BY: uint8(by)})
			}
		}
		var dirs DirectionGrid
		var vars VarianceGrid
		for i := range dirs {
			for j := range dirs[i] {
				dirs[i][j] = uint8(rng.next() % 8)
				vars[i][j] = 1 << 12
			}
		}
		params := FrameFilterParams{XDec: s.dec, YDec: s.dec, Plane: s.plane, Level: s.level, SecondaryStrength: s.sec, Damping: 5}
		dst := make([]byte, unitW*unitH)
		b.Run("shape="+s.name+"/impl=generic", func(b *testing.B) {
			for b.Loop() {
				_ = FilterFrameBlocksU8TrustedNoPrimaryDirection(dst, unitW, wide, origin, blocks, &dirs, &vars, params)
			}
		})
		b.Run("shape="+s.name+"/impl=avx2", func(b *testing.B) {
			if !U8ByteInteriorAvailable() {
				b.Skip("no AVX2")
			}
			for b.Loop() {
				if !FilterFrameBlocksU8ByteInteriorTrusted(dst, unitW, input, origin, unitW, unitH, blocks, &dirs, &vars, params, false) {
					b.Fatal("declined")
				}
			}
		})
	}
}

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
