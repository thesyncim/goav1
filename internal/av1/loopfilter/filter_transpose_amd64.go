// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && amd64 && !purego

package loopfilter

// lfGatherU8 transposes the 8x8 byte tile at po (rows ps apart) of pix into
// the scratch tile at so, readBytes taps per position.
func lfGatherU8(pix []byte, po, ps int, scr *lfScratch, so int, readBytes int) {
	for y := 0; y < 8; y++ {
		for x := 0; x < readBytes; x++ {
			scr[so+x*lfScrRow8+y] = pix[po+y*ps+x]
		}
		for x := readBytes; x < 8; x++ {
			scr[so+x*lfScrRow8+y] = 0
		}
	}
}

// lfScatterU8 writes scratch tap rows 0..writeBytes-1 of the tile at so back
// to the plane tile at po.
func lfScatterU8(scr *lfScratch, so int, pix []byte, po, ps int, writeBytes int) {
	for x := 0; x < 8; x++ {
		for y := 0; y < writeBytes; y++ {
			pix[po+x*ps+y] = scr[so+y*lfScrRow8+x]
		}
	}
}

// lfGatherU16 is lfGatherU8 for two-byte samples.
func lfGatherU16(pix []byte, po, ps int, scr *lfScratch, so int, readWidth int) {
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			d := so + 2*y + x*lfScrRow16
			if x < readWidth {
				s := po + 2*x + y*ps
				scr[d], scr[d+1] = pix[s], pix[s+1]
			} else {
				scr[d], scr[d+1] = 0, 0
			}
		}
	}
}

// lfScatterU16 is lfScatterU8 for two-byte samples.
func lfScatterU16(scr *lfScratch, so int, pix []byte, po, ps int, writeWidth int) {
	for y := 0; y < 8; y++ {
		for x := 0; x < writeWidth; x++ {
			d := po + 2*x + y*ps
			s := so + 2*y + x*lfScrRow16
			pix[d], pix[d+1] = scr[s], scr[s+1]
		}
	}
}
