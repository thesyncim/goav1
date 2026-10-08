package goav1

import (
	"encoding/binary"
	"errors"
	"io"
	"strconv"
	"testing"

	internalivf "github.com/thesyncim/goav1/internal/av1/ivf"
)

func TestReaderAtDecoderPayloadSourceTruncatedFrameHeaderParity(t *testing.T) {
	backing := decoderBoundaryIVFWithFrame(0)
	for available := 1; available < IVFFrameHeaderSize; available++ {
		t.Run(strconv.Itoa(available), func(t *testing.T) {
			declaredSize := int64(IVFFileHeaderSize + available)

			// Iterator.Next checks the bytes remaining in the declared slice before
			// reading the frame header (internal/av1/ivf/reader.go).
			iterator, err := internalivf.NewIterator(backing[:declaredSize])
			if err != nil {
				t.Fatalf("NewIterator: %v", err)
			}
			_, _, wantErr := iterator.Next()
			if !errors.Is(wantErr, ErrIVFShortFrameHeader) {
				t.Fatalf("slice iterator err=%v want %v", wantErr, ErrIVFShortFrameHeader)
			}

			reader := &decoderBoundaryTrackingReaderAt{data: backing, limit: declaredSize}
			_, _, gotErr := newReaderAtDecoderPayloadSource(reader, declaredSize)
			if !errors.Is(gotErr, wantErr) {
				t.Fatalf("ReaderAt err=%v want slice iterator err %v", gotErr, wantErr)
			}
			if reader.readBeyondLimit {
				t.Fatalf("ReaderAt read beyond declared size %d: requests=%v", declaredSize, reader.requests)
			}
		})
	}
}

func TestReaderAtDecoderPayloadSourceTruncatedFramePayloadParity(t *testing.T) {
	const payloadSize = 3
	backing := decoderBoundaryIVFWithFrame(payloadSize)

	for available := 0; available < payloadSize; available++ {
		t.Run(strconv.Itoa(available), func(t *testing.T) {
			declaredSize := int64(IVFFileHeaderSize + IVFFrameHeaderSize + available)

			iterator, err := internalivf.NewIterator(backing[:declaredSize])
			if err != nil {
				t.Fatalf("NewIterator: %v", err)
			}
			_, _, wantErr := iterator.Next()
			if !errors.Is(wantErr, ErrIVFShortFramePayload) {
				t.Fatalf("slice iterator err=%v want %v", wantErr, ErrIVFShortFramePayload)
			}

			reader := &decoderBoundaryTrackingReaderAt{data: backing, limit: declaredSize}
			_, _, gotErr := newReaderAtDecoderPayloadSource(reader, declaredSize)
			if !errors.Is(gotErr, wantErr) {
				t.Fatalf("ReaderAt err=%v want slice iterator err %v", gotErr, wantErr)
			}
			if reader.readBeyondLimit {
				t.Fatalf("ReaderAt read beyond declared size %d: requests=%v", declaredSize, reader.requests)
			}
		})
	}
}

func TestReaderAtDecoderPayloadSourceTruncatedFileHeaderDoesNotReadPastSize(t *testing.T) {
	backing := decoderBoundaryIVFWithFrame(0)
	for available := 0; available < IVFFileHeaderSize; available++ {
		t.Run(strconv.Itoa(available), func(t *testing.T) {
			declaredSize := int64(available)
			reader := &decoderBoundaryTrackingReaderAt{data: backing, limit: declaredSize}
			_, _, gotErr := newReaderAtDecoderPayloadSource(reader, declaredSize)
			if !errors.Is(gotErr, ErrIVFShortHeader) {
				t.Fatalf("ReaderAt err=%v want %v", gotErr, ErrIVFShortHeader)
			}
			if reader.readBeyondLimit {
				t.Fatalf("ReaderAt read beyond declared size %d: requests=%v", declaredSize, reader.requests)
			}
		})
	}
}

func TestReaderAtDecoderPayloadSourcePreservesShortUnderlyingRead(t *testing.T) {
	backing := decoderBoundaryIVFWithFrame(0)
	declaredSize := int64(len(backing))
	reader := &decoderBoundaryTrackingReaderAt{
		data:  backing[:IVFFileHeaderSize+5],
		limit: declaredSize,
	}
	_, _, err := newReaderAtDecoderPayloadSource(reader, declaredSize)
	if !errors.Is(err, ErrIVFShortFrameHeader) {
		t.Fatalf("ReaderAt err=%v want %v", err, ErrIVFShortFrameHeader)
	}
	if reader.readBeyondLimit {
		t.Fatalf("ReaderAt read beyond declared size %d: requests=%v", declaredSize, reader.requests)
	}
}

func FuzzReaderAtDecoderPayloadSourceMatchesSliceIterator(f *testing.F) {
	frame := decoderBoundaryIVFWithFrame(3)
	f.Add(frame, uint16(len(frame)))
	f.Add(frame, uint16(IVFFileHeaderSize+1))
	f.Add(frame, uint16(IVFFileHeaderSize+IVFFrameHeaderSize-1))
	f.Add(frame, uint16(IVFFileHeaderSize+IVFFrameHeaderSize+1))

	// This parser supports the standard 32-byte IVF header. A different header
	// size is explicitly unsupported by both the ReaderAt and slice paths.
	extendedHeader := append([]byte(nil), frame...)
	binary.LittleEndian.PutUint16(extendedHeader[6:8], IVFFileHeaderSize+1)
	f.Add(extendedHeader, uint16(len(extendedHeader)))
	f.Add([]byte{}, uint16(0))

	f.Fuzz(func(t *testing.T, input []byte, sizeSelector uint16) {
		const maxInputSize = 8 << 10
		if len(input) > maxInputSize {
			input = input[:maxInputSize]
		}
		declaredSize := int(sizeSelector) % (len(input) + 1)
		visible := input[:declaredSize]

		iterator, sliceErr := internalivf.NewIterator(visible)
		reader := &decoderBoundaryTrackingReaderAt{
			data:  input,
			limit: int64(declaredSize),
		}
		source, _, readerErr := newReaderAtDecoderPayloadSource(reader, int64(declaredSize))
		if reader.readBeyondLimit {
			t.Fatalf("ReaderAt read beyond declared size %d: requests=%v", declaredSize, reader.requests)
		}

		if sliceErr != nil {
			if !errors.Is(readerErr, sliceErr) {
				t.Fatalf("ReaderAt err=%v want slice iterator err %v", readerErr, sliceErr)
			}
			return
		}

		frameCount := 0
		var sliceFrameErr error
		for {
			_, ok, err := iterator.Next()
			if err != nil {
				sliceFrameErr = err
				break
			}
			if !ok {
				break
			}
			frameCount++
		}
		if sliceFrameErr != nil {
			if !errors.Is(readerErr, sliceFrameErr) {
				t.Fatalf("ReaderAt err=%v want slice iterator err %v", readerErr, sliceFrameErr)
			}
			return
		}
		if frameCount == 0 {
			// The ReaderAt source additionally requires at least one frame; the
			// slice iterator represents an empty stream as ordinary end-of-input.
			if readerErr == nil {
				t.Fatal("ReaderAt source accepted an IVF stream with no frames")
			}
			return
		}
		if readerErr != nil {
			t.Fatalf("ReaderAt err=%v after slice iterator parsed %d frames", readerErr, frameCount)
		}
		if len(source.records) != frameCount {
			t.Fatalf("ReaderAt indexed %d frames, slice iterator parsed %d", len(source.records), frameCount)
		}
	})
}

func decoderBoundaryIVFWithFrame(payloadSize uint32) []byte {
	data := make([]byte, IVFFileHeaderSize+IVFFrameHeaderSize+int(payloadSize))
	copy(data[:4], "DKIF")
	binary.LittleEndian.PutUint16(data[6:8], IVFFileHeaderSize)
	copy(data[8:12], "AV01")
	binary.LittleEndian.PutUint16(data[12:14], 16)
	binary.LittleEndian.PutUint16(data[14:16], 16)
	binary.LittleEndian.PutUint32(data[16:20], 30)
	binary.LittleEndian.PutUint32(data[20:24], 1)
	binary.LittleEndian.PutUint32(data[24:28], 1)
	binary.LittleEndian.PutUint32(data[IVFFileHeaderSize:], payloadSize)
	for i := IVFFileHeaderSize + IVFFrameHeaderSize; i < len(data); i++ {
		data[i] = byte(i)
	}
	return data
}

type decoderBoundaryTrackingReaderAt struct {
	data            []byte
	limit           int64
	requests        [][2]int64
	readBeyondLimit bool
}

func (r *decoderBoundaryTrackingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	end := off + int64(len(p))
	r.requests = append(r.requests, [2]int64{off, end})
	if off < 0 || end < off {
		return 0, errors.New("invalid read range")
	}
	if end > r.limit {
		r.readBeyondLimit = true
	}
	if off >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[int(off):])
	if n != len(p) {
		return n, io.EOF
	}
	return n, nil
}
