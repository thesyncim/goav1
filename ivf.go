package goav1

import internalivf "github.com/thesyncim/goav1/internal/av1/ivf"

const (
	IVFFileHeaderSize  = internalivf.FileHeaderSize
	IVFFrameHeaderSize = internalivf.FrameHeaderSize
)

type IVFHeader = internalivf.Header
type IVFFrame = internalivf.Frame
type IVFIterator = internalivf.Iterator

var (
	ErrIVFShortHeader       = internalivf.ErrShortHeader
	ErrIVFInvalidSignature  = internalivf.ErrInvalidSignature
	ErrIVFUnsupportedCodec  = internalivf.ErrUnsupportedCodec
	ErrIVFUnsupportedHeader = internalivf.ErrUnsupportedHeader
	ErrIVFShortFrameHeader  = internalivf.ErrShortFrameHeader
	ErrIVFShortFramePayload = internalivf.ErrShortFramePayload
)

func ParseIVFHeader(src []byte) (IVFHeader, int, error) {
	return internalivf.ParseHeader(src)
}

func NewIVFIterator(src []byte) (IVFIterator, error) {
	return internalivf.NewIterator(src)
}
