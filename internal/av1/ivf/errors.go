package ivf

import "errors"

var (
	ErrShortHeader       = errors.New("ivf: short header")
	ErrInvalidSignature  = errors.New("ivf: invalid signature")
	ErrUnsupportedCodec  = errors.New("ivf: unsupported codec")
	ErrUnsupportedHeader = errors.New("ivf: unsupported header")
	ErrShortFrameHeader  = errors.New("ivf: short frame header")
	ErrShortFramePayload = errors.New("ivf: short frame payload")
)
