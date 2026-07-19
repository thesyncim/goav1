// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package filmgrain

// NEON-accelerated scaling-LUT gather+interpolate for 10-bit. The .s file
// implements the eight-wide inner loop that fills scale[] for a run of image
// samples, reproducing scalar scaleLUT10 (scaling.go) bit-for-bit:
//
//	idx  = min(src, 1023); x = idx>>2; rem = idx&3
//	start = lut[x]; end = lut[min(x+1,255)]
//	scale = uint8(start + roundPowerOfTwo((end-start)*rem, 2))
//
// The gather itself is a scalar lane load per sample (umov the index to a GPR,
// add the LUT base, ld1 one byte into a lane) exactly as dav1d's gather_neon
// does (third_party/upstream/dav1d/src/arm/64/filmgrain16.S); the interpolation
// and clamp are then done eight lanes wide, which is where the win comes from.
// Capping the second index at 255 (min(x+1,255)) makes end==start when x==255,
// so the interpolation collapses to lut[255] with no out-of-range LUT read —
// matching scaleLUT10's x==255 special case without a per-lane branch. The
// uint8 truncation of the scalar (start+round can leave [0,255]) is reproduced
// with a final AND #0x00ff.
//
// Only 10-bit is vectorized: the 8-bit path is a bare lut[index] gather with no
// interpolation to amortize the per-lane umov/ld1 cost, so a NEON gather
// measured slower than the scalar load and 8-bit stays scalar in apply.go.
//
// The Go wrapper handles the (<8)-lane tail with the scalar reference, so the
// asm only runs for full eight-sample groups.

// buildScaleNEONCtx is the asm calling context. Field order and sizes are part
// of the ABI shared with build_scale_neon_arm64.s; do not reorder.
type buildScaleNEONCtx struct {
	scale *uint16
	src   *uint16
	lut   *uint8

	groups uintptr // number of full 8-sample groups
}

//go:noescape
func buildScaleRow10NEONAsm(ctx *buildScaleNEONCtx)

func buildScaleRow10NEON(scale []uint16, src []uint16, lut []uint8) {
	n := len(scale)
	groups := n / 8
	if groups > 0 {
		ctx := buildScaleNEONCtx{
			scale:  &scale[0],
			src:    &src[0],
			lut:    &lut[0],
			groups: uintptr(groups),
		}
		buildScaleRow10NEONAsm(&ctx)
	}
	for i := groups * 8; i < n; i++ {
		scale[i] = uint16(scaleLUT10(lut, int(src[i])))
	}
}
