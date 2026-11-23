// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package cdef

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/thesyncim/goav1/internal/guardpage"
)

// TestCDEFUnitStaysInsideGuardedBuffers filters whole units whose destination
// and tap buffers are exactly as long as each entry point's contract allows,
// ending at (or starting right after) an inaccessible page: the widened
// route on an input of the unit plus its two-row halo, and the byte routes on
// inputOrigin+(unitHeight+1)*BStride+unitWidth+16 bytes.
func TestCDEFUnitStaysInsideGuardedBuffers(t *testing.T) {
	rnd := newCDEFRandom(0x6e7d)
	byteRuns := 0
	for _, dec := range []uint8{0, 1} {
		plane := PlaneY
		if dec == 1 {
			plane = PlaneU
		}
		unitW, unitH := BlockSize>>dec, BlockSize>>dec
		bw, bh := 8>>dec, 8>>dec
		var blocks []BlockPosition
		for by := range unitH / bh {
			for bx := range unitW / bw {
				blocks = append(blocks, BlockPosition{BX: uint8(bx), BY: uint8(by)})
			}
		}
		for _, before := range []bool{false, true} {
			for _, st := range [][2]uint8{{6, 2}, {6, 0}, {0, 2}, {0, 1}, {15, 4}} {
				params := FrameFilterParams{XDec: dec, YDec: dec, Plane: plane, Level: st[0], SecondaryStrength: st[1], Damping: 5}
				name := fmt.Sprintf("dec=%d before=%v level=%d sec=%d", dec, before, st[0], st[1])
				alloc := func(n int) []byte {
					if before {
						return guardpage.Before[byte](t, n)
					}
					return guardpage.After[byte](t, n)
				}
				dstStride := unitW
				dst := alloc((unitH-1)*dstStride + unitW)
				for i := range dst {
					dst[i] = byte(rnd.generate(256))
				}
				ref := bytes.Clone(dst)

				// Widened route: the unit plus the two-row, eight-column halo.
				origin := VerticalBorder*BStride + HorizontalBorder
				n16 := (unitH + 2*VerticalBorder) * BStride
				var wide []uint16
				if before {
					wide = guardpage.Before[uint16](t, n16)
				} else {
					wide = guardpage.After[uint16](t, n16)
				}
				for i := range wide {
					wide[i] = uint16(rnd.generate(256))
				}
				var dirs, refDirs DirectionGrid
				var vars, refVars VarianceGrid
				guardpage.Check(t, name+" widened", func() {
					if err := FilterFrameBlocksU8Trusted(dst, dstStride, wide, origin, blocks, &dirs, &vars, params); err != nil {
						t.Fatal(err)
					}
				})
				refWide := make([]uint16, InputBufferSize)
				copy(refWide, wide)
				if err := FilterFrameBlocksU8Trusted(ref, dstStride, refWide, origin, blocks, &refDirs, &refVars, params); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(dst, ref) || dirs != refDirs || vars != refVars {
					t.Fatalf("%s: widened route differs on guarded buffers", name)
				}

				// Byte routes, at the smallest origin each accepts on the
				// leading side and the smallest length on the trailing side.
				bOrigin := origin
				if before {
					bOrigin = 2*BStride + 2
				}
				nb := bOrigin + (unitH+1)*BStride + unitW + 16
				in := alloc(nb)
				for i := range in {
					in[i] = byte(rnd.generate(256))
				}
				refIn := bytes.Clone(in)
				dst2 := alloc((unitH-1)*dstStride + unitW)
				ref2 := bytes.Clone(dst2)
				var ok, refOK bool
				guardpage.Check(t, name+" byte", func() {
					ok = FilterFrameBlocksU8ByteInteriorTrusted(dst2, dstStride, in, bOrigin, unitW, unitH, blocks, &dirs, &vars, params, true)
				})
				refOK = FilterFrameBlocksU8ByteInteriorTrusted(ref2, dstStride, refIn, bOrigin, unitW, unitH, blocks, &refDirs, &refVars, params, true)
				if ok != refOK || !bytes.Equal(dst2, ref2) {
					t.Fatalf("%s: byte route differs on guarded buffers", name)
				}
				if ok {
					byteRuns++
				}
				guardpage.Check(t, name+" secondary byte", func() {
					ok = FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(dst2, dstStride, in, bOrigin, unitW, unitH, blocks, params)
				})
				refOK = FilterFrameBlocksU8SecondaryOnlyByteInteriorTrusted(ref2, dstStride, refIn, bOrigin, unitW, unitH, blocks, params)
				if ok != refOK || !bytes.Equal(dst2, ref2) {
					t.Fatalf("%s: secondary byte route differs on guarded buffers", name)
				}
			}
		}
	}
	if U8ByteInteriorAvailable() && byteRuns == 0 {
		t.Fatal("the byte route declined every unit")
	}
}
