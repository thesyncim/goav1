//go:build goexperiment.simd && (arm64 || amd64) && !purego

package cdef

import (
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"testing"
)

// TestFindDirectionSIMDMatchesScalar fuzz-sweeps random blocks over every
// coeffShift and stride and asserts the Go-native SIMD kernel is byte-identical
// to the scalar reference (which matches libaom's cdef_find_dir_c).
func TestFindDirectionSIMDMatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x53494d44)
	for coeffShift := range 5 {
		max := uint16((1 << (8 + coeffShift)) - 1)
		for _, stride := range []int{8, 9, 13, 16, 31} {
			for iter := range 256 {
				img := make([]uint16, stride*8)
				for i := range img {
					img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
				gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
				if gotDir != wantDir || gotVar != wantVar {
					t.Fatalf("coeffShift=%d stride=%d iter=%d dir,var=%d,%d want %d,%d",
						coeffShift, stride, iter, gotDir, gotVar, wantDir, wantVar)
				}
			}
		}
	}
}

// TestFindDirectionSIMDEdges covers all-flat, min/max-contrast and each direction
// dominant so the argmax + tie-break and the variance paths are all exercised.
func TestFindDirectionSIMDEdges(t *testing.T) {
	const stride = 8
	patterns := []func(row, col int) uint16{
		func(row, col int) uint16 { return 128 },                            // flat -> dir 0, var 0
		func(row, col int) uint16 { return 0 },                              // min flat
		func(row, col int) uint16 { return 255 },                            // max flat (shift 0)
		func(row, col int) uint16 { return uint16((row + col) * 16) },       // diag 0
		func(row, col int) uint16 { return uint16((2*row + col) * 8) },      // dir 1
		func(row, col int) uint16 { return uint16(row * 32) },               // horizontal 2
		func(row, col int) uint16 { return uint16((2*row - col + 8) * 8) },  // dir 3
		func(row, col int) uint16 { return uint16((row - col + 8) * 16) },   // anti-diag 4
		func(row, col int) uint16 { return uint16((row - 2*col + 16) * 8) }, // dir 5
		func(row, col int) uint16 { return uint16(col * 32) },               // vertical 6
		func(row, col int) uint16 { return uint16((-row + 2*col + 8) * 8) }, // dir 7
	}
	for coeffShift := range 5 {
		maxSample := uint16((1 << (8 + coeffShift)) - 1)
		for pi, fill := range patterns {
			img := make([]uint16, stride*8)
			for row := range 8 {
				for col := range 8 {
					v := uint32(fill(row, col)) << coeffShift
					if v > uint32(maxSample) {
						v = uint32(maxSample)
					}
					img[row*stride+col] = uint16(v)
				}
			}
			wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
			gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("pattern=%d coeffShift=%d dir,var=%d,%d want %d,%d",
					pi, coeffShift, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

// TestFindDirectionSIMDMaxContrast drives extreme checkerboards (full-range) so
// the partial sums reach their largest magnitudes and any int32 overflow in the
// cost arithmetic must match the scalar reference bit-for-bit.
func TestFindDirectionSIMDMaxContrast(t *testing.T) {
	const stride = 8
	for coeffShift := range 5 {
		maxSample := uint16((1 << (8 + coeffShift)) - 1)
		for variant := 0; variant < 4; variant++ {
			img := make([]uint16, stride*8)
			for row := range 8 {
				for col := range 8 {
					var hi bool
					switch variant {
					case 0:
						hi = (row+col)&1 == 0
					case 1:
						hi = row&1 == 0
					case 2:
						hi = col&1 == 0
					case 3:
						hi = row < 4
					}
					if hi {
						img[row*stride+col] = maxSample
					}
				}
			}
			wantDir, wantVar := findDirectionScalar(img, stride, coeffShift)
			gotDir, gotVar := findDirectionSIMD(img, stride, coeffShift)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("variant=%d coeffShift=%d dir,var=%d,%d want %d,%d",
					variant, coeffShift, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

func TestFindDirectionSIMDZeroAlloc(t *testing.T) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 37) & 0xfff)
	}
	allocs := testing.AllocsPerRun(1000, func() {
		_, _ = findDirectionSIMD(img, 8, 4)
	})
	if allocs != 0 {
		t.Fatalf("findDirectionSIMD allocated: %f", allocs)
	}
}

func BenchmarkFindDirectionSIMD(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xfff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = findDirectionSIMD(img, 8, 4)
	}
}

func BenchmarkFindDirectionSIMD8(b *testing.B) {
	img := make([]uint16, 64)
	for i := range img {
		img[i] = uint16((i * 41) & 0xff)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _ = findDirectionSIMD(img, 8, 0)
	}
}

// TestFindDirectionU8SIMDMatchesScalar checks the Go-native SIMD u8 kernel
// against the independent scalar reference (direction + variance) across
// strides and content regimes, including flat, near-max and full-random blocks.
func TestFindDirectionU8SIMDMatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x11223344)
	for _, stride := range []int{8, 17, 160, 640} {
		for iter := range 256 {
			img := make([]byte, stride*8+8)
			for i := range img {
				switch iter % 4 {
				case 0:
					img[i] = byte(rnd.generate(256))
				case 1:
					img[i] = byte(rnd.generate(6))
				case 2:
					img[i] = byte(250 + rnd.generate(6))
				default:
					if (i/stride)%2 == 0 {
						img[i] = 255
					}
				}
			}
			wantDir, wantVar := findDirectionU8Scalar(img, stride)
			gotDir, gotVar := findDirectionU8SIMD(img, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("SIMD stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

// TestFindDirectionU8SIMDTightTail pins the row-7 hi-half load: the block's
// last row ends exactly at the end of the backing slice (no byte after it),
// so any overreading load would fault or diverge.
func TestFindDirectionU8SIMDTightTail(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x777)
	for _, stride := range []int{8, 16, 33} {
		for iter := 0; iter < 64; iter++ {
			img := make([]byte, 7*stride+8) // last row has exactly 8 bytes
			for i := range img {
				img[i] = byte(rnd.generate(256))
			}
			wantDir, wantVar := findDirectionU8Scalar(img, stride)
			gotDir, gotVar := findDirectionU8SIMD(img, stride)
			if gotDir != wantDir || gotVar != wantVar {
				t.Fatalf("stride=%d iter=%d got=(%d,%d) want=(%d,%d)", stride, iter, gotDir, gotVar, wantDir, wantVar)
			}
		}
	}
}

func TestFindDirectionSIMDDispatchBound(t *testing.T) {
	if !directionSIMDBound() {
		t.Skip("CPU does not advertise AVX2")
	}
	nameOf := func(v interface{}) string {
		return runtime.FuncForPC(reflect.ValueOf(v).Pointer()).Name()
	}
	if got, want := nameOf(findDirectionImpl), nameOf(findDirectionSIMD); got != want {
		t.Errorf("findDirectionImpl = %s, want %s", got, want)
	}
	if got, want := nameOf(findDirectionDualImpl), nameOf(findDirectionDualSIMD); got != want {
		t.Errorf("findDirectionDualImpl = %s, want %s", got, want)
	}
	if got, want := nameOf(findDirectionU8Impl), nameOf(findDirectionU8SIMD); got != want {
		t.Errorf("findDirectionU8Impl = %s, want %s", got, want)
	}
	if got, want := nameOf(findDirectionDualU8Impl), nameOf(findDirectionDualU8SIMD); got != want {
		t.Errorf("findDirectionDualU8Impl = %s, want %s", got, want)
	}
}

func TestFindDirectionDualSIMDMatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x4455414c)
	for coeffShift := range 5 {
		max := uint16((1 << (8 + coeffShift)) - 1)
		for _, stride := range []int{16, 19, 32} {
			for iter := range 64 {
				img := make([]uint16, stride*8)
				for i := range img {
					img[i] = uint16(rnd.pseudoUniform(int(max) + 1))
				}
				want1, wantVar1, want2, wantVar2 := findDirectionDualScalar(img, img[8:], stride, coeffShift)
				got1, gotVar1, got2, gotVar2 := findDirectionDualSIMD(img, img[8:], stride, coeffShift)
				if got1 != want1 || gotVar1 != wantVar1 || got2 != want2 || gotVar2 != wantVar2 {
					t.Fatalf("coeffShift=%d stride=%d iter=%d got=(%d,%d),(%d,%d) want=(%d,%d),(%d,%d)",
						coeffShift, stride, iter, got1, gotVar1, got2, gotVar2,
						want1, wantVar1, want2, wantVar2)
				}
			}
		}
	}
}

func TestFindDirectionDualU8SIMDMatchesScalar(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x55445541)
	for _, stride := range []int{16, 19, 32} {
		for iter := range 64 {
			img := make([]byte, stride*8)
			for i := range img {
				img[i] = byte(rnd.generate(256))
			}
			want1, wantVar1, want2, wantVar2 := findDirectionDualU8Scalar(img, img[8:], stride)
			got1, gotVar1, got2, gotVar2 := findDirectionDualU8SIMD(img, img[8:], stride)
			if got1 != want1 || gotVar1 != wantVar1 || got2 != want2 || gotVar2 != wantVar2 {
				t.Fatalf("stride=%d iter=%d got=(%d,%d),(%d,%d) want=(%d,%d),(%d,%d)",
					stride, iter, got1, gotVar1, got2, gotVar2,
					want1, wantVar1, want2, wantVar2)
			}
		}
	}
}

func TestFindDirectionU8SIMDZeroAlloc(t *testing.T) {
	img := make([]byte, 640*8+8)
	rnd := newCDEFRandom(1)
	for i := range img {
		img[i] = byte(rnd.generate(256))
	}
	if a := testing.AllocsPerRun(50, func() { findDirectionU8SIMD(img, 640) }); a != 0 {
		t.Errorf("findDirectionU8SIMD allocated %.1f objects/run, want 0", a)
	}
}

func BenchmarkFindDirectionU8SIMD(b *testing.B) {
	img := make([]byte, 640*8+8)
	rnd := newCDEFRandom(2)
	for i := range img {
		img[i] = byte(rnd.generate(256))
	}
	b.ReportAllocs()
	for b.Loop() {
		findDirectionU8SIMD(img, 640)
	}
}

// TestFilterBlockSIMDMatchesPureGo proves the Go SIMD CDEF block filter
// bit-exact with the pure-Go reference across directions, strengths,
// dampings, coefficient shifts, block shapes, and inputs salted with the
// VeryLarge border sentinel.
func TestFilterBlockSIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(97))
	input := make([]uint16, BStride*24)
	for trial := 0; trial < 4000; trial++ {
		maxSample := 255
		coeffShift := rng.Intn(3)
		if coeffShift > 0 {
			maxSample = (1 << (8 + coeffShift)) - 1
		}
		for i := range input {
			if rng.Intn(13) == 0 {
				input[i] = VeryLarge
			} else {
				input[i] = uint16(rng.Intn(maxSample + 1))
			}
		}
		var width, height int
		switch rng.Intn(3) {
		case 0:
			width, height = 8, 8
		case 1:
			width, height = 8, 4
		default:
			width, height = 4, 4
		}
		params := BlockFilterParams{
			PrimaryStrength:   uint8(rng.Intn(16) << coeffShift),
			SecondaryStrength: uint8([]int{0, 1, 2, 4}[rng.Intn(4)] << coeffShift),
			Direction:         uint8(rng.Intn(8)),
			PrimaryDamping:    uint8(3 + coeffShift + rng.Intn(3)),
			SecondaryDamping:  uint8(3 + coeffShift + rng.Intn(3)),
			CoeffShift:        uint8(coeffShift),
			Width:             uint8(width),
			Height:            uint8(height),
		}
		origin := BStride*8 + 16
		want := make([]uint16, 16*16)
		got := make([]uint16, 16*16)
		filterBlockPureGo(want, 16, 0, input, origin, params)
		filterBlockSIMD(got, 16, 0, input, origin, params)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d params=%+v: dst[%d] simd %d want %d", trial, params, i, got[i], want[i])
			}
		}
	}
}

// TestFilterBlockSIMDStrengthCasesMatchPureGo pins the single-strength
// shapes (primary-only and secondary-only), which the kernel serves with the
// same row loop as the fused case but with the other side disabled.
func TestFilterBlockSIMDStrengthCasesMatchPureGo(t *testing.T) {
	cases := []struct {
		name      string
		primary   uint8
		secondary uint8
	}{
		{name: "primary-only", primary: 13, secondary: 0},
		{name: "secondary-only", primary: 0, secondary: 4},
		{name: "fused", primary: 9, secondary: 2},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			shapes := []struct {
				width  int
				height int
			}{
				{width: 8, height: 8},
				{width: 4, height: 8},
				{width: 4, height: 4},
			}
			for coeffShift := uint8(0); coeffShift <= 2; coeffShift++ {
				for direction := uint8(0); direction < 8; direction++ {
					for _, shape := range shapes {
						seed := cdefDeterministicSeed ^
							uint32(direction) ^
							(uint32(coeffShift) << 8) ^
							(uint32(shape.width) << 16) ^
							(uint32(shape.height) << 20)
						input := makeCDEFBlockInput(newCDEFRandom(seed), 8+int(coeffShift), int(direction)&0xf, int(coeffShift)+1)
						params := BlockFilterParams{
							PrimaryStrength:   tc.primary << coeffShift,
							SecondaryStrength: tc.secondary << coeffShift,
							Direction:         direction,
							PrimaryDamping:    4 + coeffShift,
							SecondaryDamping:  4 + coeffShift,
							CoeffShift:        coeffShift,
							Width:             uint8(shape.width),
							Height:            uint8(shape.height),
						}
						origin := cdefBlockOrigin()
						want := make([]uint16, 16*16)
						got := make([]uint16, 16*16)
						filterBlockPureGo(want, 16, 0, input, origin, params)
						filterBlockSIMD(got, 16, 0, input, origin, params)
						for i := range want {
							if want[i] != got[i] {
								t.Fatalf("%s %dx%d coeffShift=%d direction=%d dst[%d] simd %d want %d", tc.name, shape.width, shape.height, coeffShift, direction, i, got[i], want[i])
							}
						}
					}
				}
			}
		})
	}
}

func benchFilterSIMD(b *testing.B, fn func([]uint16, int, int, []uint16, int, BlockFilterParams)) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 9, SecondaryStrength: 2, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, fn)
}

func benchFilterSIMDParams(b *testing.B, params BlockFilterParams, fn func([]uint16, int, int, []uint16, int, BlockFilterParams)) {
	input := make([]uint16, BStride*24)
	for i := range input {
		input[i] = uint16(i * 7 % 256)
	}
	dst := make([]uint16, 16*16)
	b.ReportAllocs()
	for b.Loop() {
		fn(dst, 16, 0, input, BStride*8+16, params)
	}
}

func BenchmarkFilterBlock8x8SIMD(b *testing.B)   { benchFilterSIMD(b, filterBlockSIMD) }
func BenchmarkFilterBlock8x8PureGo(b *testing.B) { benchFilterSIMD(b, filterBlockPureGo) }

func BenchmarkFilterBlock8x8PrimaryOnlySIMD(b *testing.B) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, filterBlockSIMD)
}

func BenchmarkFilterBlock8x8SecondaryOnlySIMD(b *testing.B) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 8, Height: 8,
	}, filterBlockSIMD)
}

func BenchmarkFilterBlock4x4PrimaryOnlySIMD(b *testing.B) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 13, SecondaryStrength: 0, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, filterBlockSIMD)
}

func BenchmarkFilterBlock4x4SecondaryOnlySIMD(b *testing.B) {
	benchFilterSIMDParams(b, BlockFilterParams{
		PrimaryStrength: 0, SecondaryStrength: 4, Direction: 5,
		PrimaryDamping: 5, SecondaryDamping: 4, Width: 4, Height: 4,
	}, filterBlockSIMD)
}

// TestFilterBlockSIMDClipCorpus targets the clamp path. Uniform random taps
// almost never overshoot, because the fused weights sum to 24 against a
// divisor of 16 and only a near-flat neighbourhood with strong strengths
// exposes the [min, max] clamp. Narrow-spread neighbourhoods with strong
// strengths and injected VeryLarge halo samples hit it directly, including the
// sentinel-skip in the maximum fold.
func TestFilterBlockSIMDClipCorpus(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5ca1ab1e))
	input := make([]uint16, BStride*24)
	origin := cdefBlockOrigin()
	for trial := 0; trial < 6000; trial++ {
		coeffShift := rng.Intn(3)
		maxSample := (1 << (8 + coeffShift)) - 1
		base := rng.Intn(maxSample + 1)
		spread := 1 << rng.Intn(6)
		for i := range input {
			v := min(base+rng.Intn(spread+1), maxSample)
			if rng.Intn(8) == 0 {
				v = VeryLarge
			}
			input[i] = uint16(v)
		}
		var width, height int
		switch rng.Intn(3) {
		case 0:
			width, height = 8, 8
		case 1:
			width, height = 8, 4
		default:
			width, height = 4, 4
		}
		params := BlockFilterParams{
			PrimaryStrength:   uint8(1+rng.Intn(15)) << coeffShift,
			SecondaryStrength: uint8(1+rng.Intn(4)) << coeffShift,
			Direction:         uint8(rng.Intn(8)),
			PrimaryDamping:    uint8(3 + coeffShift + rng.Intn(4)),
			SecondaryDamping:  uint8(3 + coeffShift + rng.Intn(4)),
			CoeffShift:        uint8(coeffShift),
			Width:             uint8(width),
			Height:            uint8(height),
		}
		want := make([]uint16, 16*16)
		got := make([]uint16, 16*16)
		filterBlockPureGo(want, 16, 0, input, origin, params)
		filterBlockSIMD(got, 16, 0, input, origin, params)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d params=%+v spread=%d base=%d: dst[%d] simd %d want %d", trial, params, spread, base, i, got[i], want[i])
			}
		}
	}
}

type byteUnitRNG uint32

func (r *byteUnitRNG) next() uint32 {
	x := uint32(*r)
	x ^= x << 13
	x ^= x >> 17
	x ^= x << 5
	*r = byteUnitRNG(x)
	return x
}

// makeByteUnitInput fills a unit plus its 2-pixel halo with pattern pixels
// and returns the byte buffer, its uint16 widening, and the unit origin.
// Bytes outside the halo hold junk that the kernels may load but must not use.
func makeByteUnitInput(rng *byteUnitRNG, unitWidth, unitHeight int, pattern int) ([]byte, []uint16, int) {
	origin := VerticalBorder*BStride + HorizontalBorder
	end := origin + (unitHeight+1)*BStride + unitWidth + 16
	input := make([]byte, end)
	for i := range input {
		input[i] = byte(rng.next())
	}
	for row := -VerticalBorder; row < unitHeight+VerticalBorder; row++ {
		for col := -2; col < unitWidth+2; col++ {
			r := rng.next()
			var v byte
			switch pattern {
			case 0:
				v = byte(r)
			case 1:
				v = byte(128 + int(r%9) - 4)
			case 2:
				v = byte(r & 1 * 255)
				if r&6 == 0 {
					v = byte(r >> 8)
				}
			case 3:
				v = byte(250 + r%6)
			case 4:
				v = byte(r % 6)
			default:
				v = byte(uint32(row*7+col*3) + r%3)
			}
			input[origin+row*BStride+col] = v
		}
	}
	wide := make([]uint16, len(input))
	for i, b := range input {
		wide[i] = uint16(b)
	}
	return input, wide, origin
}

func TestFilterFrameBlocksU8ByteInteriorParity(t *testing.T) {
	requireByteInterior(t)
	rng := byteUnitRNG(0x9e3779b9)
	decs := [][2]uint8{{0, 0}, {1, 1}, {1, 0}, {0, 1}}
	for iter := 0; iter < 3000; iter++ {
		dec := decs[iter%len(decs)]
		plane := PlaneY
		if dec != [2]uint8{0, 0} || iter%7 == 3 {
			plane = Plane(1 + iter%2)
		}
		unitWidth := BlockSize >> dec[0]
		unitHeight := BlockSize >> dec[1]
		blockWidth := 8 >> dec[0]
		blockHeight := 8 >> dec[1]
		input, wide, origin := makeByteUnitInput(&rng, unitWidth, unitHeight, iter%6)
		var blocks []BlockPosition
		dense := iter%3 == 0
		for by := 0; by < unitHeight/blockHeight; by++ {
			for bx := 0; bx < unitWidth/blockWidth; bx++ {
				if dense || rng.next()%3 != 0 {
					blocks = append(blocks, BlockPosition{BX: uint8(bx), BY: uint8(by)})
				}
			}
		}
		if len(blocks) == 0 {
			continue
		}
		secs := [4]uint8{0, 1, 2, 4}
		params := FrameFilterParams{
			XDec: dec[0], YDec: dec[1], Plane: plane,
			Level:             uint8(rng.next() % 16),
			SecondaryStrength: secs[rng.next()%4],
			Damping:           uint8(3 + rng.next()%4),
		}
		var dirs DirectionGrid
		var vars VarianceGrid
		for i := range dirs {
			for j := range dirs[i] {
				dirs[i][j] = uint8(rng.next() % 8)
				switch rng.next() % 4 {
				case 0:
					vars[i][j] = 0
				case 1:
					vars[i][j] = int32(rng.next() % 64)
				default:
					vars[i][j] = int32(rng.next() % (1 << 20))
				}
			}
		}
		const pad = 3
		dstStride := unitWidth + pad
		dstLen := (unitHeight-1)*dstStride + unitWidth
		got := make([]byte, dstLen)
		for i := range got {
			got[i] = byte(rng.next())
		}
		want := append([]byte(nil), got...)
		if !FilterFrameBlocksU8ByteInteriorTrusted(got, dstStride, input, origin, unitWidth, unitHeight, blocks, &dirs, &vars, params, false) {
			t.Fatalf("iter %d: declined %+v", iter, params)
		}
		damping := int(params.Damping)
		if plane != PlaneY {
			damping--
		}
		u := unitFilterParams{
			primaryStrength:   int(params.Level),
			secondaryStrength: int(params.SecondaryStrength),
			damping:           damping,
			bwLog2:            3 - int(dec[0]),
			bhLog2:            3 - int(dec[1]),
			blockWidth:        blockWidth,
			blockHeight:       blockHeight,
			lumaAdjust:        plane == PlaneY && params.Level != 0,
		}
		if u.primaryStrength != 0 || u.secondaryStrength != 0 {
			if err := filterUnitBlocksU8PureGo(want, dstStride, wide, origin, blocks, &dirs, &vars, u); err != nil {
				t.Fatal(err)
			}
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("iter %d params %+v: idx %d (row %d col %d) got %d want %d", iter, params, i, i/dstStride, i%dstStride, got[i], want[i])
			}
		}
	}
}

// TestFilterFrameBlocksU8ByteInteriorDerivesDirections checks the luma
// direction pass (run on dst before any write) and the chroma 4:2:2 map.
func TestFilterFrameBlocksU8ByteInteriorDerivesDirections(t *testing.T) {
	requireByteInterior(t)
	rng := byteUnitRNG(0x51ed270b)
	for iter := 0; iter < 200; iter++ {
		for _, dec := range [][2]uint8{{0, 0}, {1, 0}} {
			plane := PlaneY
			if dec[0] != 0 {
				plane = PlaneU
			}
			unitWidth := BlockSize >> dec[0]
			unitHeight := BlockSize >> dec[1]
			input, wide, origin := makeByteUnitInput(&rng, unitWidth, unitHeight, iter%6)
			dstStride := unitWidth
			dst := make([]byte, unitWidth*unitHeight)
			for y := 0; y < unitHeight; y++ {
				copy(dst[y*dstStride:], input[origin+y*BStride:origin+y*BStride+unitWidth])
			}
			want := append([]byte(nil), dst...)
			var blocks []BlockPosition
			for by := 0; by < unitHeight/(8>>dec[1]); by++ {
				for bx := 0; bx < unitWidth/(8>>dec[0]); bx++ {
					blocks = append(blocks, BlockPosition{BX: uint8(bx), BY: uint8(by)})
				}
			}
			params := FrameFilterParams{XDec: dec[0], YDec: dec[1], Plane: plane, Level: uint8(rng.next() % 16), SecondaryStrength: [4]uint8{0, 1, 2, 4}[rng.next()%4], Damping: 6}
			var dg, wdg DirectionGrid
			var vg, wvg VarianceGrid
			for i := range dg {
				for j := range dg[i] {
					dg[i][j] = uint8(rng.next() % 8)
				}
			}
			wdg = dg
			if !FilterFrameBlocksU8ByteInteriorTrusted(dst, dstStride, input, origin, unitWidth, unitHeight, blocks, &dg, &vg, params, true) {
				t.Fatal("declined")
			}
			if err := FilterFrameBlocksU8Trusted(want, dstStride, wide, origin, blocks, &wdg, &wvg, params); err != nil {
				t.Fatal(err)
			}
			if dg != wdg || vg != wvg {
				t.Fatalf("iter %d dec %v: grids differ", iter, dec)
			}
			for i := range dst {
				if dst[i] != want[i] {
					t.Fatalf("iter %d dec %v params %+v: idx %d got %d want %d", iter, dec, params, i, dst[i], want[i])
				}
			}
		}
	}
}

func TestFilterFrameBlocksU8ByteInteriorDeclines(t *testing.T) {
	requireByteInterior(t)
	rng := byteUnitRNG(7)
	input, _, origin := makeByteUnitInput(&rng, 64, 64, 0)
	blocks := []BlockPosition{{BX: 0, BY: 0}}
	dst := make([]byte, 64*64)
	var dirs DirectionGrid
	var vars VarianceGrid
	base := FrameFilterParams{Level: 3, SecondaryStrength: 2, Damping: 4}
	for name, tc := range map[string]struct {
		in     []byte
		w, h   int
		params FrameFilterParams
	}{
		"short-input":  {input[:len(input)-1], 64, 64, base},
		"misaligned":   {input, 60, 64, base},
		"secondary-3":  {input, 64, 64, FrameFilterParams{Level: 3, SecondaryStrength: 3, Damping: 4}},
		"coeff-shift":  {input, 64, 64, FrameFilterParams{Level: 3, SecondaryStrength: 2, Damping: 4, CoeffShift: 2}},
		"neg-damping":  {input, 64, 64, FrameFilterParams{Plane: PlaneU, Level: 3, Damping: 0}},
		"level-too-hi": {input, 64, 64, FrameFilterParams{Level: 16, Damping: 4}},
	} {
		before := append([]byte(nil), dst...)
		if FilterFrameBlocksU8ByteInteriorTrusted(dst, 64, tc.in, origin, tc.w, tc.h, blocks, &dirs, &vars, tc.params, true) {
			t.Fatalf("%s: accepted", name)
		}
		if fmt.Sprint(before) != fmt.Sprint(dst) {
			t.Fatalf("%s: mutated dst", name)
		}
	}
}

func TestFilterFrameBlocksU8ByteInteriorZeroAlloc(t *testing.T) {
	requireByteInterior(t)
	rng := byteUnitRNG(11)
	input, _, origin := makeByteUnitInput(&rng, 64, 64, 0)
	blocks := make([]BlockPosition, 0, 64)
	for by := range 8 {
		for bx := range 8 {
			blocks = append(blocks, BlockPosition{BX: uint8(bx), BY: uint8(by)})
		}
	}
	dst := make([]byte, 64*64)
	var dirs DirectionGrid
	var vars VarianceGrid
	params := FrameFilterParams{Level: 5, SecondaryStrength: 2, Damping: 5}
	allocs := testing.AllocsPerRun(50, func() {
		FilterFrameBlocksU8ByteInteriorTrusted(dst, 64, input, origin, 64, 64, blocks, &dirs, &vars, params, true)
	})
	if allocs != 0 {
		t.Fatalf("byte interior route allocates: %v", allocs)
	}
}

// requireByteInterior skips on builds whose CPU lacks the byte kernels (amd64
// without AVX2). Where they are available every test above must take them:
// the entry declining is a failure.
func requireByteInterior(t *testing.T) {
	t.Helper()
	if !U8ByteInteriorAvailable() {
		t.Skip("byte-interior CDEF kernels unavailable on this CPU")
	}
}

// TestFilterFrameBlocksU8ByteInteriorSecondaryOnlyEntryParity checks the
// secondary-only byte entry (arm64 NEON, amd64 AVX2) against the widened
// pure-Go filter, dense and sparse block lists, all strengths and dampings.
// Its availability must match the general byte route so the decoder picks
// the same route on every SIMD build.
func TestFilterFrameBlocksU8ByteInteriorSecondaryOnlyEntryParity(t *testing.T) {
	requireByteInterior(t)
	if !U8SecondaryOnlyByteInteriorAvailable() {
		t.Fatal("secondary-only byte route must be available wherever the byte-interior route is")
	}
	rng := byteUnitRNG(0x2545f491)
	decs := [][2]uint8{{0, 0}, {1, 1}, {1, 0}, {0, 1}}
	for iter := 0; iter < 1200; iter++ {
		dec := decs[iter%len(decs)]
		plane := PlaneY
		if dec != [2]uint8{0, 0} || iter%5 == 2 {
			plane = Plane(1 + iter%2)
		}
		unitWidth := BlockSize >> dec[0]
		unitHeight := BlockSize >> dec[1]
		blockWidth := 8 >> dec[0]
		blockHeight := 8 >> dec[1]
		input, wide, origin := makeByteUnitInput(&rng, unitWidth, unitHeight, iter%6)
		var blocks []BlockPosition
		dense := iter%3 == 0
		for by := 0; by < unitHeight/blockHeight; by++ {
			for bx := 0; bx < unitWidth/blockWidth; bx++ {
				if dense || rng.next()%3 != 0 {
					blocks = append(blocks, BlockPosition{BX: uint8(bx), BY: uint8(by)})
				}
			}
		}
		if len(blocks) == 0 {
			continue
		}
		secs := [3]uint8{1, 2, 4}
		params := FrameFilterParams{
			XDec: dec[0], YDec: dec[1], Plane: plane,
			SecondaryStrength: secs[rng.next()%3],
			Damping:           uint8(3 + rng.next()%4),
		}
		const pad = 5
		dstStride := unitWidth + pad
		dstLen := (unitHeight-1)*dstStride + unitWidth
		got := make([]byte, dstLen)
		for i := range got {
			got[i] = byte(rng.next())
		}
		want := append([]byte(nil), got...)
		if !FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(got, dstStride, input, origin, unitWidth, unitHeight, blocks, params) {
			t.Fatalf("iter %d: declined %+v", iter, params)
		}
		damping := int(params.Damping)
		if plane != PlaneY {
			damping--
		}
		var dirs DirectionGrid
		var vars VarianceGrid
		u := unitFilterParams{
			secondaryStrength: int(params.SecondaryStrength),
			damping:           damping,
			bwLog2:            3 - int(dec[0]),
			bhLog2:            3 - int(dec[1]),
			blockWidth:        blockWidth,
			blockHeight:       blockHeight,
		}
		if err := filterUnitBlocksU8PureGo(want, dstStride, wide, origin, blocks, &dirs, &vars, u); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("iter %d params %+v: idx %d (row %d col %d) got %d want %d", iter, params, i, i/dstStride, i%dstStride, got[i], want[i])
			}
		}
	}
}

// TestFilterBlockU8SIMDMatchesPureGo pins the Go SIMD 8-bit block
// filter against the pure-Go reference. It covers every legal 8-bit primary
// strength (0..15), the full secondary range accepted by this package (0..4),
// all directions, all halo-sentinel patterns, all u8 CDEF block shapes, and
// representative damping values without going through cpu.Detected dispatch.
func TestFilterBlockU8SIMDMatchesPureGo(t *testing.T) {
	const (
		dstStride = 19
		dstOrigin = 5
	)
	origin := cdefBlockOrigin()
	shapes := [...]struct{ width, height int }{
		{8, 8}, {8, 4}, {4, 8}, {4, 4},
	}
	for boundary := 0; boundary < 16; boundary++ {
		input := makeCDEFBlockInput(newCDEFRandom(cdefDeterministicSeed^0x38415658), 8, boundary, boundary+2)
		for _, shape := range shapes {
			for dir := 0; dir <= 7; dir++ {
				for pri := 0; pri <= 15; pri++ {
					for sec := 0; sec <= 4; sec++ {
						if pri == 0 && sec == 0 {
							continue
						}
						for _, damping := range []int{3, 4, 5, 6} {
							params := BlockFilterParams{
								PrimaryStrength:   uint8(pri),
								SecondaryStrength: uint8(sec),
								Direction:         uint8(dir),
								PrimaryDamping:    uint8(damping),
								SecondaryDamping:  uint8(damping),
								CoeffShift:        0,
								Width:             uint8(shape.width),
								Height:            uint8(shape.height),
							}
							want := makeGuardedU8Dst(dstOrigin + dstStride*8 + 8)
							got := makeGuardedU8Dst(len(want))
							filterBlockU8PureGo(want, dstStride, dstOrigin, input, origin, params)
							filterBlockU8SIMD(got, dstStride, dstOrigin, input, origin, params)
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("boundary=%d shape=%dx%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
										boundary, shape.width, shape.height, dir, pri, sec, damping, i, got[i], want[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

// TestFilterUnitBlocksU8SIMDMatchesPureGo proves the unit loop
// remains byte-identical when every per-block call is forced through the Go
// SIMD u8 kernel directly. This covers partial units and skip masks without
// relying on the runtime dispatch slot.
func TestFilterUnitBlocksU8SIMDMatchesPureGo(t *testing.T) {
	rnd := newCDEFRandom(cdefDeterministicSeed ^ 0x41563832)
	cases := make([]cdefU8UnitCase, 0, 5120)
	for _, geom := range []struct {
		plane      Plane
		xDec, yDec int
	}{
		{PlaneY, 0, 0},
		{PlaneU, 1, 1},
		{PlaneU, 1, 0},
		{PlaneU, 0, 0},
	} {
		unitWFull := BlockSize >> geom.xDec
		unitHFull := BlockSize >> geom.yDec
		blockW := 8 >> geom.xDec
		blockH := 8 >> geom.yDec
		for _, size := range []struct{ w, h int }{
			{unitWFull, unitHFull},
			{unitWFull - blockW, unitHFull - blockH},
			{blockW, blockH},
			{unitWFull, blockH * 2},
		} {
			for level := 0; level <= 15; level++ {
				for sec := 0; sec <= 4; sec++ {
					for _, damping := range []int{3, 4, 5, 6} {
						cases = append(cases, cdefU8UnitCase{
							plane: geom.plane, xDec: geom.xDec, yDec: geom.yDec,
							unitW: size.w, unitH: size.h,
							level: level, sec: sec, damping: damping,
							skipMask: uint64(rnd.generate(1<<16)) | uint64(rnd.generate(1<<16))<<16,
							halo:     uint8(rnd.generate(16)),
						})
					}
				}
			}
		}
	}
	for i, tc := range cases {
		runCDEFU8UnitSIMDDifferential(t, rnd, tc, i)
	}
}

func makeGuardedU8Dst(n int) []byte {
	dst := make([]byte, n)
	for i := range dst {
		dst[i] = byte((i*37 + 113) & 0xff)
	}
	return dst
}

func runCDEFU8UnitSIMDDifferential(t *testing.T, rnd *cdefRandom, tc cdefU8UnitCase, caseIdx int) {
	t.Helper()
	blockW := 8 >> tc.xDec
	blockH := 8 >> tc.yDec
	blockCols := (tc.unitW + blockW - 1) / blockW
	blockRows := (tc.unitH + blockH - 1) / blockH

	const margin = 24
	stride := tc.unitW + 2*margin + 5
	height := tc.unitH + 2*margin
	frame := make([]byte, stride*height)
	for i := range frame {
		frame[i] = byte(rnd.generate(256))
	}
	unitOrigin := margin*stride + margin

	input := make([]uint16, InputBufferSize)
	for i := range input {
		input[i] = uint16(rnd.generate(256))
	}
	origin := cdefBlockOrigin()
	for row := 0; row < tc.unitH; row++ {
		for col := 0; col < tc.unitW; col++ {
			input[origin+row*BStride+col] = uint16(frame[unitOrigin+row*stride+col])
		}
	}
	applyCDEFU8UnitHalo(input, tc.unitW, tc.unitH, tc.halo)

	blocks := make([]BlockPosition, 0, blockCols*blockRows)
	bit := 0
	for by := 0; by < blockRows; by++ {
		for bx := 0; bx < blockCols; bx++ {
			if tc.skipMask&(1<<uint(bit%64)) == 0 {
				blocks = append(blocks, BlockPosition{BY: uint8(by), BX: uint8(bx)})
			}
			bit++
		}
	}
	if len(blocks) == 0 {
		blocks = append(blocks, BlockPosition{BY: 0, BX: 0})
	}

	var dirs DirectionGrid
	var vars VarianceGrid
	for by := range NBlocks {
		for bx := range NBlocks {
			dirs[by][bx] = uint8(rnd.generate(8))
			vars[by][bx] = int32(rnd.generate(1 << 20))
		}
	}
	wantDirs := dirs
	gotDirs := dirs
	wantVars := vars
	gotVars := vars
	u := unitFilterParams{
		primaryStrength:   tc.level,
		secondaryStrength: tc.sec,
		damping:           tc.damping,
		coeffShift:        0,
		bwLog2:            3 - tc.xDec,
		bhLog2:            3 - tc.yDec,
		blockWidth:        blockW,
		blockHeight:       blockH,
		lumaAdjust:        tc.plane == PlaneY,
	}

	want := make([]byte, len(frame))
	copy(want, frame)
	got := make([]byte, len(frame))
	copy(got, frame)
	if err := filterUnitBlocksU8WithBlockForTest(want[unitOrigin:], stride, input, origin, blocks, &wantDirs, &wantVars, u, filterBlockU8PureGo); err != nil {
		t.Fatalf("case %d: pure-Go unit filter: %v", caseIdx, err)
	}
	if err := filterUnitBlocksU8WithBlockForTest(got[unitOrigin:], stride, input, origin, blocks, &gotDirs, &gotVars, u, filterBlockU8SIMD); err != nil {
		t.Fatalf("case %d: SIMD unit filter: %v", caseIdx, err)
	}
	if gotDirs != wantDirs {
		t.Fatalf("case %d (%+v): direction grids differ", caseIdx, tc)
	}
	if gotVars != wantVars {
		t.Fatalf("case %d (%+v): variance grids differ", caseIdx, tc)
	}
	for i := range got {
		if got[i] != want[i] {
			row := i / stride
			col := i % stride
			t.Fatalf("case %d (%+v): byte mismatch at row=%d col=%d got=%d want=%d", caseIdx, tc, row, col, got[i], want[i])
		}
	}
}

func filterUnitBlocksU8WithBlockForTest(dst []byte, dstStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, u unitFilterParams, filterBlock func([]byte, int, int, []uint16, int, BlockFilterParams)) error {
	for _, block := range blocks {
		by := int(block.BY)
		bx := int(block.BX)
		strength := u.primaryStrength
		if u.lumaAdjust {
			strength = adjustStrength(u.primaryStrength, variances[by][bx])
		}
		if strength == 0 && u.secondaryStrength == 0 {
			continue
		}
		dir := 0
		if u.primaryStrength != 0 {
			dir = int(directions[by][bx])
		}
		srcOrigin := inputOrigin + ((by * BStride) << u.bhLog2) + (bx << u.bwLog2)
		dstOrigin := (by<<u.bhLog2)*dstStride + (bx << u.bwLog2)
		params := BlockFilterParams{
			PrimaryStrength:   uint8(strength),
			SecondaryStrength: uint8(u.secondaryStrength),
			Direction:         uint8(dir),
			PrimaryDamping:    uint8(u.damping),
			SecondaryDamping:  uint8(u.damping),
			CoeffShift:        0,
			Width:             uint8(u.blockWidth),
			Height:            uint8(u.blockHeight),
		}
		filterBlock(dst, dstStride, dstOrigin, input, srcOrigin, params)
	}
	return nil
}

// compareFilterBlockU8SIMD checks one u8 block against the pure-Go reference.
func compareFilterBlockU8SIMD(t *testing.T, input []uint16, origin int, params BlockFilterParams) {
	t.Helper()
	const dstStride = 16
	want := make([]byte, dstStride*8)
	got := make([]byte, dstStride*8)
	filterBlockU8PureGo(want, dstStride, 0, input, origin, params)
	filterBlockU8SIMD(got, dstStride, 0, input, origin, params)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("w=%d h=%d dir=%d pri=%d sec=%d damp=%d idx=%d got=%d want=%d",
				params.Width, params.Height, params.Direction, params.PrimaryStrength,
				params.SecondaryStrength, params.PrimaryDamping, i, got[i], want[i])
		}
	}
}

// TestFilterBlockU8SIMDExtremes drives all-zero, all-255, and alternating
// 0/255 neighbourhoods with VeryLarge halos through every direction and both
// single-strength shapes; the extremes sit on the clamp and saturation edges.
func TestFilterBlockU8SIMDExtremes(t *testing.T) {
	origin := cdefBlockOrigin()
	fills := []func(int) uint16{
		func(int) uint16 { return 0 },
		func(int) uint16 { return 255 },
		func(i int) uint16 {
			if i&1 == 0 {
				return 0
			}
			return 255
		},
	}
	for _, fill := range fills {
		input := make([]uint16, InputBufferSize)
		for i := range input {
			input[i] = fill(i)
		}
		for i := range input {
			row, col := i/BStride, i%BStride
			if row < VerticalBorder || row >= VerticalBorder+8 || col < HorizontalBorder || col >= HorizontalBorder+8 {
				input[i] = VeryLarge
			}
		}
		for dir := 0; dir < 8; dir++ {
			for _, shape := range [...]struct{ width, height int }{{8, 8}, {4, 4}} {
				for _, strengths := range [][2]uint8{{15, 0}, {0, 4}, {15, 4}} {
					compareFilterBlockU8SIMD(t, input, origin, BlockFilterParams{
						PrimaryStrength: strengths[0], SecondaryStrength: strengths[1],
						Direction: uint8(dir), PrimaryDamping: 3, SecondaryDamping: 3,
						Width: uint8(shape.width), Height: uint8(shape.height),
					})
				}
			}
		}
	}
}

// TestFilterBlockU8SIMDClipCorpus is the 8-bit counterpart of
// TestFilterBlockSIMDClipCorpus: narrow-spread neighbourhoods at 8-bit depth
// with strong strengths, so the fused clamp and the sentinel skip are reached.
func TestFilterBlockU8SIMDClipCorpus(t *testing.T) {
	rng := rand.New(rand.NewSource(0x8b175ca1))
	input := make([]uint16, BStride*24)
	origin := cdefBlockOrigin()
	for trial := 0; trial < 6000; trial++ {
		base := rng.Intn(256)
		spread := 1 << rng.Intn(6)
		for i := range input {
			v := min(base+rng.Intn(spread+1), 255)
			if rng.Intn(8) == 0 {
				v = VeryLarge
			}
			input[i] = uint16(v)
		}
		width, height := 8, 8
		if rng.Intn(2) == 0 {
			width, height = 4, 4
		}
		params := BlockFilterParams{
			PrimaryStrength:   uint8(1 + rng.Intn(15)),
			SecondaryStrength: uint8(1 + rng.Intn(4)),
			Direction:         uint8(rng.Intn(8)),
			PrimaryDamping:    uint8(3 + rng.Intn(4)),
			SecondaryDamping:  uint8(3 + rng.Intn(4)),
			Width:             uint8(width),
			Height:            uint8(height),
		}
		compareFilterBlockU8SIMD(t, input, origin, params)
	}
}
