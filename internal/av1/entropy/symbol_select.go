package entropy

// Branchless symbol search and CDF adaptation for the small-alphabet CDFs that
// dominate coefficient and mode decode. dav1d's arm64 msac_decode_symbol_adapt
// (src/arm/64/msac.S) compares the coded value against every bound at once and
// adapts all CDF entries under a lane mask; these are the scalar forms of the
// same idea. The symbol is data-dependent and often poorly predicted, so the
// bounds and adaptations are computed for every outcome and the symbol only
// drives conditional selects (CSEL), never a branch.

// selectSymbol4 is the symbol search of dav1d_msac_decode_symbol_adapt4 with
// n_symbols = 3, given the three cdfBound values: it returns the decoded symbol
// and its [lower, upper) interval. The inverse CDF is non-increasing and the
// EC_MIN_PROB offsets strictly decrease, so l0 > l1 > l2 and a hit on a bound
// implies the hits before it.
func selectSymbol4(rng, coded, l0, l1, l2 uint32) (sym uint, lower, upper uint32) {
	upper, lower = rng, l0
	if coded < l0 {
		upper, lower, sym = l0, l1, 1
	}
	if coded < l1 {
		upper, lower, sym = l1, l2, 2
	}
	if coded < l2 {
		upper, lower, sym = l2, 0, 3
	}
	return sym, lower, upper
}

// adaptCDFEntry is one entry of update_cdf: toward 32768 when the decoded
// symbol is above the entry's index (up), toward 0 otherwise.
func adaptCDFEntry(v uint32, rate uint, up bool) uint32 {
	next := v - v>>rate
	raised := v + (CDFProbTop-v)>>rate
	if up {
		next = raised
	}
	return next
}

// nextCDFCount is update_cdf's count + (count < 32).
func nextCDFCount(count uint16) uint16 {
	if count < MaxCDFCount {
		count++
	}
	return count
}

// cdfBound is the msac bound v = ((rng >> 8) * (cdf >> EC_PROB_SHIFT) >>
// (7 - EC_PROB_SHIFT)) + EC_MIN_PROB * (n_symbols - val).
func cdfBound(rngHi, c, minProb uint32) uint32 {
	return ((rngHi * (c >> ecProbShift)) >> (7 - ecProbShift)) + minProb
}
