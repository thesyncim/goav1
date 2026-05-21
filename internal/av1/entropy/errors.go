package entropy

import "errors"

var (
	ErrInvalidBitCount    = errors.New("entropy: invalid bit count")
	ErrInvalidCDF         = errors.New("entropy: invalid cdf")
	ErrInvalidProbability = errors.New("entropy: invalid probability")
	ErrInvalidRange       = errors.New("entropy: invalid range")
	ErrInvalidSymbol      = errors.New("entropy: invalid symbol")
)
