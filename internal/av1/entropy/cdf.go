package entropy

const (
	CDFProbBits = 15
	CDFProbTop  = 1 << CDFProbBits
	MaxSymbols  = 16
	MaxCDFCount = 32

	// DeltaSmall is the AV1 DELTA_Q_SMALL/DELTA_LF_SMALL threshold.
	DeltaSmall = 3
)

// CDF is caller-owned AV1 inverse-CDF state with fixed backing storage.
type CDF struct {
	symbols uint8
	values  [MaxSymbols + 1]uint16
}

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

// Init converts cumulative probabilities into c's fixed inverse-CDF storage.
func (c *CDF) Init(cumulative []uint16) error {
	if c == nil {
		return ErrInvalidCDF
	}
	var next CDF
	if err := InitCDF(next.values[:], cumulative); err != nil {
		return err
	}
	next.symbols = uint8(len(cumulative) + 1)
	*c = next
	return nil
}

// InitUniform writes an evenly-spaced inverse-CDF into c's fixed storage.
func (c *CDF) InitUniform(symbols int) error {
	if c == nil {
		return ErrInvalidCDF
	}
	var next CDF
	if err := InitUniformCDF(next.values[:], symbols); err != nil {
		return err
	}
	next.symbols = uint8(symbols)
	*c = next
	return nil
}

// InitDefaultDelta seeds the default AV1 delta-q/delta-lf CDF.
func (c *CDF) InitDefaultDelta() error {
	if c == nil {
		return ErrInvalidCDF
	}
	return c.Init([]uint16{28160, 32120, 32677})
}

// Reset clears c to the zero invalid state.
func (c *CDF) Reset() {
	if c != nil {
		*c = CDF{}
	}
}

// Symbols returns the number of symbols represented by c.
func (c *CDF) Symbols() int {
	if c == nil {
		return 0
	}
	return int(c.symbols)
}

// Values returns c's inverse-CDF values plus the trailing adaptation count.
func (c *CDF) Values() []uint16 {
	if c == nil {
		return nil
	}
	symbols := int(c.symbols)
	if symbols < 2 || symbols > MaxSymbols {
		return nil
	}
	return c.values[:symbols+1]
}

// Validate checks c's inverse-CDF shape.
func (c *CDF) Validate() error {
	if c == nil {
		return ErrInvalidCDF
	}
	return ValidateCDF(c.Values(), c.Symbols())
}

// Update adapts c after decoding symbol.
func (c *CDF) Update(symbol int) error {
	if c == nil {
		return ErrInvalidCDF
	}
	if c.Symbols() < 2 || c.Symbols() > MaxSymbols {
		return ErrInvalidCDF
	}
	return UpdateCDF(c.Values(), c.Symbols(), symbol)
}

// CopyFrom replaces c with src after validating src.
func (c *CDF) CopyFrom(src *CDF) error {
	if c == nil || src == nil {
		return ErrInvalidCDF
	}
	if err := src.Validate(); err != nil {
		return err
	}
	*c = *src
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
