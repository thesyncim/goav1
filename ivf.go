package goav1

import internalivf "github.com/thesyncim/goav1/internal/av1/ivf"

// IVF container size constants.
//
//   - IVFFileHeaderSize is the byte length of the DKIF/AV01 file header.
//   - IVFFrameHeaderSize is the byte length of each per-frame header.
const (
	IVFFileHeaderSize  = internalivf.FileHeaderSize
	IVFFrameHeaderSize = internalivf.FrameHeaderSize
)

// IVFHeader holds the parsed DKIF/AV01 IVF file header fields: width, height,
// timebase, frame count, and a placeholder unused field. Its layout mirrors
// the on-disk header.
type IVFHeader = internalivf.Header

// IVFFrame is one demuxed AV1 frame: an alias of the caller-owned source
// payload together with the frame's index, byte offset in the file, and
// presentation timestamp.
type IVFFrame = internalivf.Frame

// IVFIterator walks an in-memory IVF stream frame by frame without allocating.
// Each Next call returns an IVFFrame whose payload aliases the source slice
// supplied to NewIVFIterator.
type IVFIterator = internalivf.Iterator

// IVF parse errors. ErrIVFShortHeader, ErrIVFInvalidSignature,
// ErrIVFUnsupportedCodec, and ErrIVFUnsupportedHeader are returned by
// ParseIVFHeader; ErrIVFShortFrameHeader and ErrIVFShortFramePayload are
// returned by IVFIterator.Next when the input is truncated.
var (
	ErrIVFShortHeader       = internalivf.ErrShortHeader
	ErrIVFInvalidSignature  = internalivf.ErrInvalidSignature
	ErrIVFUnsupportedCodec  = internalivf.ErrUnsupportedCodec
	ErrIVFUnsupportedHeader = internalivf.ErrUnsupportedHeader
	ErrIVFShortFrameHeader  = internalivf.ErrShortFrameHeader
	ErrIVFShortFramePayload = internalivf.ErrShortFramePayload
)

// ParseIVFHeader parses an IVF file header from the front of src and returns
// the parsed header together with the number of bytes consumed. It does not
// allocate. Errors include ErrIVFShortHeader, ErrIVFInvalidSignature,
// ErrIVFUnsupportedCodec, and ErrIVFUnsupportedHeader.
func ParseIVFHeader(src []byte) (IVFHeader, int, error) {
	return internalivf.ParseHeader(src)
}

// NewIVFIterator parses the IVF file header at the start of src and returns
// an iterator that walks the remaining frames. The returned iterator aliases
// src; the caller must keep src alive for the iterator's lifetime.
func NewIVFIterator(src []byte) (IVFIterator, error) {
	return internalivf.NewIterator(src)
}
