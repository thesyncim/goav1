// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package rtp

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

const maxOBUPrefixBytes = 2 + bitstream.MaxLEB128Bytes

// FrameOBU describes one normalized OBU in an assembled low-overhead frame.
//
// The exported fields are stable caller-visible metadata. The unexported fields
// carry the exact prefix and raw-skip state used by the second assembly pass.
type FrameOBU struct {
	Header obu.Header

	Offset      int
	Length      int
	PrefixSize  int
	PayloadSize int

	prefix  [7]byte
	rawSkip int
}

// End reports the exclusive end offset of the normalized low-overhead OBU in
// the assembled frame.
func (o FrameOBU) End() int {
	return o.Offset + o.Length
}

// PayloadOffset reports the payload start offset inside the assembled frame.
func (o FrameOBU) PayloadOffset() int {
	return o.Offset + o.PrefixSize
}

// PayloadEnd reports the exclusive payload end offset inside the assembled
// frame.
func (o FrameOBU) PayloadEnd() int {
	return o.End()
}

type frameAssembleScan struct {
	obus []FrameOBU

	count int
	total int

	open      bool
	rawSize   int
	prefix    [maxOBUPrefixBytes]byte
	prefixLen int
}

type frameAssembleWrite struct {
	dst  []byte
	obus []FrameOBU

	used     int
	obuIndex int

	open          bool
	rawSeen       int
	payloadCopied int
	current       *FrameOBU
}

// AssembleFrame writes a sequence of AV1 RTP payloads as one low-overhead AV1
// frame into dst.
//
// payloads must include each RTP payload's aggregation header. obus is
// caller-owned scratch and receives metadata for each output OBU. The function
// follows libwebrtc's AV1 frame assembly contract: RTP OBU fragments are joined,
// OBU size fields are restored, and malformed continuation or OBU size state is
// rejected. No memory is allocated.
func AssembleFrame(dst []byte, payloads [][]byte, obus []FrameOBU) (n int, obuCount int, err error) {
	scan := frameAssembleScan{obus: obus}
	if err := scanPayloads(payloads, &scan); err != nil {
		return 0, scan.count, err
	}
	if scan.count == 0 {
		return 0, 0, ErrEmptyFrame
	}
	if len(dst) < scan.total {
		return 0, scan.count, ErrShortBuffer
	}
	n, err = writeAssembledFrame(dst, payloads, obus[:scan.count])
	if err != nil {
		return n, scan.count, err
	}
	return n, scan.count, nil
}

// AssembleFrameSize reports the output bytes and OBU metadata slots required by
// AssembleFrame for payloads. It runs the same validation scan as AssembleFrame
// without requiring caller-owned dst or FrameOBU scratch.
func AssembleFrameSize(payloads [][]byte) (n int, obuCount int, err error) {
	var scan frameAssembleScan
	if err := scanPayloads(payloads, &scan); err != nil {
		return 0, scan.count, err
	}
	if scan.count == 0 {
		return 0, 0, ErrEmptyFrame
	}
	return scan.total, scan.count, nil
}

func scanPayloads(payloads [][]byte, scan *frameAssembleScan) error {
	expectContinuation := false
	for i := range payloads {
		payload := payloads[i]
		header, off, err := ParseAggregationHeader(payload)
		if err != nil {
			return err
		}
		if err := checkContinuation(header.ContinuesPrevious, expectContinuation); err != nil {
			return err
		}

		if off == len(payload) {
			if header.ElementCount != 1 {
				return ErrInvalidElementCount
			}
			if !header.ContinuesPrevious {
				if err := scan.startOBU(); err != nil {
					return err
				}
			}
			if !header.ContinuesNext {
				if err := scan.finishOBU(); err != nil {
					return err
				}
			}
			expectContinuation = header.ContinuesNext
			continue
		}

		for elementIndex := 1; off < len(payload); elementIndex++ {
			continuesPrevious := header.ContinuesPrevious && elementIndex == 1
			if !continuesPrevious {
				if err := scan.startOBU(); err != nil {
					return err
				}
			} else if !scan.open {
				return ErrUnexpectedContinuation
			}

			fragment, next, err := nextElementFragment(payload, off, elementIndex, int(header.ElementCount))
			if err != nil {
				return err
			}
			scan.addFragment(fragment)
			off = next

			if !(header.ContinuesNext && off == len(payload)) {
				if err := scan.finishOBU(); err != nil {
					return err
				}
			}
		}
		expectContinuation = header.ContinuesNext
	}

	if expectContinuation || scan.open {
		return ErrFragmentInterrupted
	}
	return nil
}

func writeAssembledFrame(dst []byte, payloads [][]byte, obus []FrameOBU) (int, error) {
	expectContinuation := false
	write := frameAssembleWrite{dst: dst, obus: obus}

	for i := range payloads {
		payload := payloads[i]
		header, off, err := ParseAggregationHeader(payload)
		if err != nil {
			return write.used, err
		}
		if err := checkContinuation(header.ContinuesPrevious, expectContinuation); err != nil {
			return write.used, err
		}

		if off == len(payload) {
			if header.ElementCount != 1 {
				return write.used, ErrInvalidElementCount
			}
			if !header.ContinuesPrevious {
				if err := write.startOBU(); err != nil {
					return write.used, err
				}
			}
			if !header.ContinuesNext {
				if err := write.finishOBU(); err != nil {
					return write.used, err
				}
			}
			expectContinuation = header.ContinuesNext
			continue
		}

		for elementIndex := 1; off < len(payload); elementIndex++ {
			continuesPrevious := header.ContinuesPrevious && elementIndex == 1
			if !continuesPrevious {
				if err := write.startOBU(); err != nil {
					return write.used, err
				}
			} else if !write.open {
				return write.used, ErrUnexpectedContinuation
			}

			fragment, next, err := nextElementFragment(payload, off, elementIndex, int(header.ElementCount))
			if err != nil {
				return write.used, err
			}
			if err := write.addFragment(fragment); err != nil {
				return write.used, err
			}
			off = next

			if !(header.ContinuesNext && off == len(payload)) {
				if err := write.finishOBU(); err != nil {
					return write.used, err
				}
			}
		}
		expectContinuation = header.ContinuesNext
	}

	if expectContinuation || write.open {
		return write.used, ErrFragmentInterrupted
	}
	if write.obuIndex != len(obus) {
		return write.used, ErrOBUBufferTooSmall
	}
	return write.used, nil
}

func nextElementFragment(payload []byte, off int, elementIndex int, expectedElements int) ([]byte, int, error) {
	if expectedElements != 0 && elementIndex == expectedElements {
		return payload[off:], len(payload), nil
	}

	size, sizeLen, err := bitstream.ReadLEB128(payload[off:])
	if err != nil {
		return nil, off, err
	}
	start := off + sizeLen
	end := start + int(size)
	if end < start || end > len(payload) {
		return nil, off, ErrShortPayload
	}
	return payload[start:end], end, nil
}

func checkContinuation(continuesPrevious bool, expected bool) error {
	if continuesPrevious == expected {
		return nil
	}
	if continuesPrevious {
		return ErrUnexpectedContinuation
	}
	return ErrFragmentInterrupted
}

func (w *frameAssembleWrite) startOBU() error {
	if w.open {
		return ErrFragmentInterrupted
	}
	if w.obuIndex >= len(w.obus) {
		return ErrOBUBufferTooSmall
	}

	w.current = &w.obus[w.obuIndex]
	if len(w.dst)-w.used < w.current.PrefixSize {
		return ErrShortBuffer
	}
	copy(w.dst[w.used:w.used+w.current.PrefixSize], w.current.prefix[:w.current.PrefixSize])
	w.used += w.current.PrefixSize
	w.rawSeen = 0
	w.payloadCopied = 0
	w.open = true
	return nil
}

func (w *frameAssembleWrite) addFragment(fragment []byte) error {
	if !w.open {
		return ErrUnexpectedContinuation
	}

	start := 0
	if w.rawSeen < w.current.rawSkip {
		skip := w.current.rawSkip - w.rawSeen
		if skip >= len(fragment) {
			w.rawSeen += len(fragment)
			return nil
		}
		start = skip
	}

	n := len(fragment) - start
	if n > w.current.PayloadSize-w.payloadCopied {
		return obu.ErrSizeMismatch
	}
	if len(w.dst)-w.used < n {
		return ErrShortBuffer
	}
	copy(w.dst[w.used:w.used+n], fragment[start:])
	w.used += n
	w.rawSeen += len(fragment)
	w.payloadCopied += n
	return nil
}

func (w *frameAssembleWrite) finishOBU() error {
	if !w.open {
		return ErrUnexpectedContinuation
	}
	if w.payloadCopied != w.current.PayloadSize {
		return obu.ErrSizeMismatch
	}
	w.open = false
	w.obuIndex++
	return nil
}

func (s *frameAssembleScan) startOBU() error {
	if s.open {
		return ErrFragmentInterrupted
	}
	s.open = true
	s.rawSize = 0
	s.prefixLen = 0
	return nil
}

func (s *frameAssembleScan) addFragment(fragment []byte) {
	if len(fragment) == 0 {
		return
	}
	if s.prefixLen < len(s.prefix) {
		n := copy(s.prefix[s.prefixLen:], fragment)
		s.prefixLen += n
	}
	s.rawSize += len(fragment)
}

func (s *frameAssembleScan) finishOBU() error {
	if !s.open {
		return ErrUnexpectedContinuation
	}
	if s.obus != nil && s.count >= len(s.obus) {
		return ErrOBUBufferTooSmall
	}
	if s.rawSize == 0 {
		return ErrZeroLengthElement
	}

	info, err := s.calculateOBUInfo()
	if err != nil {
		return err
	}
	info.Offset = s.total
	info.Length = info.PrefixSize + info.PayloadSize
	if s.obus != nil {
		s.obus[s.count] = info
	}
	s.count++
	s.total += info.Length
	s.open = false
	return nil
}

func (s *frameAssembleScan) calculateOBUInfo() (FrameOBU, error) {
	header, headerLen, err := obu.ParseHeader(s.prefix[:s.prefixLen])
	if err != nil {
		return FrameOBU{}, err
	}
	if s.rawSize < headerLen {
		return FrameOBU{}, obu.ErrShortPayload
	}

	rawSkip := headerLen
	payloadSize := s.rawSize - headerLen
	if header.HasSizeField {
		size, sizeLen, err := bitstream.ReadLEB128(s.prefix[headerLen:s.prefixLen])
		if err != nil {
			return FrameOBU{}, err
		}
		rawSkip = headerLen + sizeLen
		if s.rawSize < rawSkip {
			return FrameOBU{}, obu.ErrShortPayload
		}
		payloadSize = s.rawSize - rawSkip
		if uint32(payloadSize) != size {
			return FrameOBU{}, obu.ErrSizeMismatch
		}
	}
	if uint64(payloadSize) > bitstream.MaxLEB128Value {
		return FrameOBU{}, bitstream.ErrLEB128Overflow
	}

	header.HasSizeField = true
	info := FrameOBU{
		Header:      header,
		PrefixSize:  headerLen,
		PayloadSize: payloadSize,
		rawSkip:     rawSkip,
	}
	n, err := obu.PutHeader(info.prefix[:], header)
	if err != nil {
		return FrameOBU{}, err
	}
	info.PrefixSize = n
	n, err = bitstream.PutLEB128(info.prefix[info.PrefixSize:], uint32(payloadSize))
	if err != nil {
		return FrameOBU{}, err
	}
	info.PrefixSize += n
	return info, nil
}
