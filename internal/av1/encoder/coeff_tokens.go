package encoder

import "github.com/thesyncim/goav1/internal/av1/entropy"

// coeff_tokens.go holds the coefficient-coding primitives that are the first
// pieces of the forward of the decoder's tile.ReadCoefficientsTXB. They are
// source-shaped ports of libaom av1/encoder/encodetxb.c and av1/common/
// txb_common.c. The full transform-block coefficient writer (av1_write_coeffs_txb)
// builds on these plus the nz-map/br context derivation (av1_get_nz_map_contexts,
// get_br_ctx), ported in a later slice.

// eobGroupStart is av1_eob_group_start (txb_common.c): eobGroupStart[t] is the
// first eob value in EOB position-token group t. eobOffsetBits is
// av1_eob_offset_bits (txb_common.c): the number of extra offset bits coded for
// group t (the first via an adaptive eob_extra symbol, the rest as raw bits).
var (
	eobGroupStart = [12]int{0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513}
	eobOffsetBits = [12]int{0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
)

// eobToPosSmall / eobToPosLarge map an eob value to its position token, ported
// from the static tables in encodetxb.c.
var (
	eobToPosSmall = [33]int8{
		0, 1, 2,
		3, 3,
		4, 4, 4, 4,
		5, 5, 5, 5, 5, 5, 5, 5,
		6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
	}
	eobToPosLarge = [17]int8{
		6,
		7,
		8, 8,
		9, 9, 9, 9,
		10, 10, 10, 10, 10, 10, 10, 10,
		11,
	}
)

// eobPosToken returns the EOB position token for eob (the scan index of the last
// non-zero coefficient plus one, >= 1) together with the offset of eob within
// that token's group. Port of av1_get_eob_pos_token (encodetxb.c). The offset
// satisfies eobGroupStart[token]+extra == eob and 0 <= extra < 1<<eobOffsetBits[token].
func eobPosToken(eob int) (token int, extra int) {
	if eob < 33 {
		token = int(eobToPosSmall[eob])
	} else {
		e := min((eob-1)>>5, 16) // AOMMIN((eob-1)>>5, 16)
		token = int(eobToPosLarge[e])
	}
	return token, eob - eobGroupStart[token]
}

// writeGolomb codes a non-negative coefficient-level tail with the AV1 Exp-Golomb
// code as equiprobable bits, the exact inverse of the decoder's
// readCoeffGolombCursor. Port of write_golomb (encodetxb.c).
func writeGolomb(w *entropy.Writer, level int) {
	x := level + 1
	length := 0
	for i := x; i != 0; i >>= 1 {
		length++
	}
	for range length - 1 {
		w.WriteLiteral(0, 1)
	}
	for i := length - 1; i >= 0; i-- {
		w.WriteLiteral(uint32((x>>uint(i))&1), 1)
	}
}
