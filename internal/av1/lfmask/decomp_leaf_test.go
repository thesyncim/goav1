// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package lfmask

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

// decompTxLeafRef is the per-byte leaf fill decompTxLeaf replaces.
func decompTxLeafRef(txa *[2][2][32][32]uint8, yBase, xBase int, td tile.TransformDimensions) {
	lw := min(int(td.Log2W), 2)
	lh := min(int(td.Log2H), 2)
	w, h := int(td.W4), int(td.H4)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			txa[0][0][yBase+y][xBase+x] = uint8(lw)
			txa[1][0][yBase+y][xBase+x] = uint8(lh)
		}
		txa[0][1][yBase+y][xBase] = uint8(w)
	}
	for x := 0; x < w; x++ {
		txa[1][1][yBase][xBase+x] = uint8(h)
	}
}

// TestDecompTxLeafMatchesBytewise checks the replicated-store leaf against
// the per-byte fill for every transform size at every aligned position.
func TestDecompTxLeafMatchesBytewise(t *testing.T) {
	for tx := tile.TransformSize(0); ; tx++ {
		td, ok := tx.Dimensions()
		if !ok {
			break
		}
		for yBase := 0; yBase+int(td.H4) <= 32; yBase += int(td.H4) {
			for xBase := 0; xBase+int(td.W4) <= 32; xBase += int(td.W4) {
				var got, want [2][2][32][32]uint8
				for i := range got {
					for j := range got[i] {
						for k := range got[i][j] {
							for l := range got[i][j][k] {
								got[i][j][k][l] = uint8(i*7 + j*5 + k + l)
							}
						}
					}
				}
				want = got
				decompTxLeaf(&got, yBase, xBase, &td)
				decompTxLeafRef(&want, yBase, xBase, td)
				if got != want {
					t.Fatalf("tx %d at (%d,%d): leaf fill differs", tx, xBase, yBase)
				}
			}
		}
	}
}
