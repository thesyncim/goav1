package entropy

const (
	CDFProbBits = 15
	CDFProbTop  = 1 << CDFProbBits
	MaxSymbols  = 16
	MaxCDFCount = 32
)

// InitCDF converts cumulative probabilities into AV1 inverse-CDF storage.
// cumulative must contain symbols-1 strictly increasing values in [1, 32768].
func InitCDF(dst []uint16, cumulative []uint16) error {
	symbols := len(cumulative) + 1
	if symbols < 2 || symbols > MaxSymbols || len(dst) < symbols+1 {
		return ErrInvalidCDF
	}

	previous := uint16(0)
	for i := 0; i < len(cumulative); i++ {
		v := cumulative[i]
		if v <= previous || v > CDFProbTop {
			return ErrInvalidCDF
		}
		dst[i] = uint16(CDFProbTop) - v
		previous = v
	}
	dst[symbols-1] = 0
	dst[symbols] = 0
	return nil
}

// InitUniformCDF writes an evenly-spaced AV1 inverse CDF with a zero update
// count into dst.
func InitUniformCDF(dst []uint16, symbols int) error {
	if symbols < 2 || symbols > MaxSymbols || len(dst) < symbols+1 {
		return ErrInvalidCDF
	}
	for i := 0; i < symbols-1; i++ {
		cumulative := ((i + 1) * CDFProbTop) / symbols
		dst[i] = uint16(CDFProbTop - cumulative)
	}
	dst[symbols-1] = 0
	dst[symbols] = 0
	return nil
}

// ValidateCDF checks the AV1 inverse-CDF shape for symbols entries plus the
// trailing adaptation count.
func ValidateCDF(cdf []uint16, symbols int) error {
	if symbols < 2 || symbols > MaxSymbols || len(cdf) < symbols+1 {
		return ErrInvalidCDF
	}
	if cdf[symbols] > MaxCDFCount || cdf[symbols-1] != 0 {
		return ErrInvalidCDF
	}
	for i := 0; i < symbols-1; i++ {
		if cdf[i] >= CDFProbTop || cdf[i] < cdf[i+1] {
			return ErrInvalidCDF
		}
	}
	return nil
}

// UpdateCDF adapts cdf after decoding symbol. cdf uses AV1 inverse-CDF storage:
// symbols probability entries followed by the update count at cdf[symbols].
func UpdateCDF(cdf []uint16, symbols int, symbol int) error {
	if symbol < 0 || symbol >= symbols {
		return ErrInvalidSymbol
	}
	if err := ValidateCDF(cdf, symbols); err != nil {
		return err
	}
	updateCDF(cdf, symbols, symbol)
	return nil
}

func updateCDF(cdf []uint16, symbols int, symbol int) {
	count := cdf[symbols]
	rate := uint(4 + (count >> 4))
	if symbols > 3 {
		rate++
	}

	for i := 0; i < symbols-1; i++ {
		if i < symbol {
			cdf[i] += (uint16(CDFProbTop) - cdf[i]) >> rate
		} else {
			cdf[i] -= cdf[i] >> rate
		}
	}
	if count < MaxCDFCount {
		cdf[symbols] = count + 1
	}
}
