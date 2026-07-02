// Ported from libaom:
//   av1/encoder/cost.c (av1_prob_cost)
//   av1/encoder/cost.h (av1_cost_symbol, av1_cost_literal)
//   aom_dsp/prob.h (get_prob)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package entropy

import "math/bits"

// ProbCostShift is libaom's AV1_PROB_COST_SHIFT: the factor to scale from
// cost in bits to cost in av1_prob_cost units (512 per bit).
const ProbCostShift = 9

// av1ProbCost is libaom's av1_prob_cost (av1/encoder/cost.c): the cost in
// 512-per-bit units of a binary symbol whose probability is (128+i)/256.
var av1ProbCost = [128]uint16{
	512, 506, 501, 495, 489, 484, 478, 473, 467, 462, 456, 451, 446, 441, 435,
	430, 425, 420, 415, 410, 405, 400, 395, 390, 385, 380, 375, 371, 366, 361,
	356, 352, 347, 343, 338, 333, 329, 324, 320, 316, 311, 307, 302, 298, 294,
	289, 285, 281, 277, 273, 268, 264, 260, 256, 252, 248, 244, 240, 236, 232,
	228, 224, 220, 216, 212, 209, 205, 201, 197, 194, 190, 186, 182, 179, 175,
	171, 168, 164, 161, 157, 153, 150, 146, 143, 139, 136, 132, 129, 125, 122,
	119, 115, 112, 109, 105, 102, 99, 95, 92, 89, 86, 82, 79, 76, 73,
	70, 66, 63, 60, 57, 54, 51, 48, 45, 42, 38, 35, 32, 29, 26,
	23, 20, 18, 15, 12, 9, 6, 3,
}

// CostSymbol prices coding symbol with cdf in 512-per-bit units, the port of
// libaom's av1_cost_symbol over the probability mass av1_cost_tokens_from_cdf
// extracts from AV1 inverse-CDF storage. Callers pass a valid symbol index;
// out-of-range probabilities clamp exactly as upstream.
func CostSymbol(cdf *CDF, symbol int) uint32 {
	if cdf == nil || symbol < 0 || symbol >= cdf.Symbols() {
		return 0
	}
	vals := cdf.Values()
	prev := CDFProbTop
	if symbol > 0 {
		prev = int(vals[symbol-1])
	}
	// av1_cost_tokens_from_cdf: p15 = AOM_ICDF(cdf[i]) - prev_cdf, floored at
	// EC_MIN_PROB; av1_cost_symbol clamps into [1, CDF_PROB_TOP-1].
	p15 := prev - int(vals[symbol])
	if p15 < ecMinProb {
		p15 = ecMinProb
	}
	if p15 > CDFProbTop-1 {
		p15 = CDFProbTop - 1
	}
	// shift = CDF_PROB_BITS - 1 - get_msb(p15)
	shift := CDFProbBits - bits.Len(uint(p15))
	// prob = get_prob(p15 << shift, CDF_PROB_TOP): p15<<shift is in
	// [2^14, 2^15-1], so prob lands in [128, 256] and only needs the upper
	// clip get_prob applies.
	prob := ((p15<<shift)*256 + CDFProbTop/2) / CDFProbTop
	if prob > 255 {
		prob = 255
	}
	return uint32(av1ProbCost[prob-128]) + uint32(shift)<<ProbCostShift
}
