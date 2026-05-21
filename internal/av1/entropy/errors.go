package entropy

import "errors"

var (
	ErrInvalidCDF    = errors.New("entropy: invalid cdf")
	ErrInvalidSymbol = errors.New("entropy: invalid symbol")
)
