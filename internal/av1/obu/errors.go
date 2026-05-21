package obu

import "errors"

var (
	ErrShortHeader              = errors.New("obu: short header")
	ErrForbiddenBit             = errors.New("obu: forbidden bit set")
	ErrReservedBit              = errors.New("obu: reserved bit set")
	ErrInvalidType              = errors.New("obu: invalid type")
	ErrInvalidAnnexB            = errors.New("obu: invalid annex b")
	ErrMissingSizeField         = errors.New("obu: missing size field")
	ErrMissingTemporalDelimiter = errors.New("obu: missing temporal delimiter")
	ErrSizeMismatch             = errors.New("obu: size field mismatch")
	ErrShortPayload             = errors.New("obu: short payload")
)
