package obu

import "errors"

var (
	ErrShortHeader      = errors.New("obu: short header")
	ErrForbiddenBit     = errors.New("obu: forbidden bit set")
	ErrReservedBit      = errors.New("obu: reserved bit set")
	ErrInvalidType      = errors.New("obu: invalid type")
	ErrMissingSizeField = errors.New("obu: missing size field")
	ErrShortPayload     = errors.New("obu: short payload")
)
