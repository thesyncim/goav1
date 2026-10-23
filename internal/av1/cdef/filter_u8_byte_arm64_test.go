// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import (
	"fmt"
	"testing"
)

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
