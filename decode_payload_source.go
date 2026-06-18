package goav1

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

type decoderPayloadRecord struct {
	offset int64
	size   uint32
}

type decoderPayloadSource struct {
	payloads   [][]byte
	readerAt   io.ReaderAt
	records    []decoderPayloadRecord
	maxPayload int
}

func newSliceDecoderPayloadSource(payloads [][]byte) decoderPayloadSource {
	maxPayload := 0
	for _, payload := range payloads {
		if len(payload) > maxPayload {
			maxPayload = len(payload)
		}
	}
	return decoderPayloadSource{payloads: payloads, maxPayload: maxPayload}
}

// NewDecoderFromIVFReaderAt indexes an IVF stream from r, builds a Decoder over
// it, and returns the parsed IVF header. Unlike NewDecoderFromIVF, it does not
// retain the full IVF file or copy every payload into memory; DecodeNext reads
// each indexed payload through r into one reusable decoder-owned buffer.
//
// The caller must keep r alive and immutable until the Decoder is closed.
func NewDecoderFromIVFReaderAt(r io.ReaderAt, size int64, opts ...Option) (*Decoder, IVFHeader, error) {
	if r == nil {
		return nil, IVFHeader{}, errors.New("goav1: nil IVF reader")
	}
	if size < 0 {
		return nil, IVFHeader{}, fmt.Errorf("goav1: negative IVF size %d", size)
	}
	source, header, err := newReaderAtDecoderPayloadSource(r, size)
	if err != nil {
		return nil, header, err
	}
	dec, err := newDecoderFromPayloadSource(source, opts...)
	if err != nil {
		return nil, header, err
	}
	return dec, header, nil
}

func newReaderAtDecoderPayloadSource(r io.ReaderAt, size int64) (decoderPayloadSource, IVFHeader, error) {
	var headerBuf [IVFFileHeaderSize]byte
	if err := decoderReadFullAt(r, headerBuf[:], 0); err != nil {
		if decoderShortRead(err) {
			return decoderPayloadSource{}, IVFHeader{}, ErrIVFShortHeader
		}
		return decoderPayloadSource{}, IVFHeader{}, err
	}
	header, n, err := ParseIVFHeader(headerBuf[:])
	if err != nil {
		return decoderPayloadSource{}, header, err
	}
	if size < int64(n) {
		return decoderPayloadSource{}, header, ErrIVFShortHeader
	}

	capHint := decoderIVFFrameCapHint(header.FrameCount, size-int64(n))
	records := make([]decoderPayloadRecord, 0, capHint)
	maxPayload := 0
	off := int64(n)
	for off < size {
		var frameHeader [IVFFrameHeaderSize]byte
		if err := decoderReadFullAt(r, frameHeader[:], off); err != nil {
			if decoderShortRead(err) {
				return decoderPayloadSource{}, header, ErrIVFShortFrameHeader
			}
			return decoderPayloadSource{}, header, err
		}
		payloadSize32 := binary.LittleEndian.Uint32(frameHeader[:4])
		payloadSize64 := int64(payloadSize32)
		if payloadSize64 > int64(decoderMaxInt()) {
			return decoderPayloadSource{}, header, ErrIVFShortFramePayload
		}
		start := off + IVFFrameHeaderSize
		end := start + payloadSize64
		if start < off || end < start || end > size {
			return decoderPayloadSource{}, header, ErrIVFShortFramePayload
		}
		payloadSize := int(payloadSize64)
		records = append(records, decoderPayloadRecord{
			offset: start,
			size:   payloadSize32,
		})
		if payloadSize > maxPayload {
			maxPayload = payloadSize
		}
		off = end
	}
	if len(records) == 0 {
		return decoderPayloadSource{}, header, errors.New("goav1: ivf stream produced no frames")
	}
	return decoderPayloadSource{
		readerAt:   r,
		records:    records,
		maxPayload: maxPayload,
	}, header, nil
}

func decoderIVFFrameCapHint(frameCount uint32, payloadBytes int64) int {
	if frameCount == 0 {
		return 0
	}
	maxFrames := payloadBytes/IVFFrameHeaderSize + 1
	if int64(frameCount) > maxFrames || int64(frameCount) > int64(decoderMaxInt()) {
		return 0
	}
	return int(frameCount)
}

func (s decoderPayloadSource) len() int {
	if s.readerAt != nil {
		return len(s.records)
	}
	return len(s.payloads)
}

func (s decoderPayloadSource) payloadBufferLen() int {
	if s.readerAt == nil {
		return 0
	}
	return s.maxPayload
}

func (s decoderPayloadSource) totalPayloadLen() (int, error) {
	total := 0
	if s.readerAt == nil {
		for _, payload := range s.payloads {
			next, ok := decoderArenaAdd2(total, len(payload))
			if !ok {
				return 0, ErrFrameShortBuffer
			}
			total = next
		}
		return total, nil
	}
	for _, record := range s.records {
		next, ok := decoderArenaAdd2(total, int(record.size))
		if !ok {
			return 0, ErrFrameShortBuffer
		}
		total = next
	}
	return total, nil
}

func (s decoderPayloadSource) makeProbeBuffer() []byte {
	if s.readerAt == nil || s.maxPayload == 0 {
		return nil
	}
	return make([]byte, s.maxPayload)
}

func (s decoderPayloadSource) payloadAt(i int, buf []byte) ([]byte, error) {
	if s.readerAt == nil {
		if i < 0 || i >= len(s.payloads) {
			return nil, ErrDecoderInvalidFrameWorkState
		}
		return s.payloads[i], nil
	}
	if i < 0 || i >= len(s.records) {
		return nil, ErrDecoderInvalidFrameWorkState
	}
	record := s.records[i]
	size := int(record.size)
	if len(buf) < size {
		return nil, ErrFrameShortBuffer
	}
	payload := buf[:size]
	if err := decoderReadFullAt(s.readerAt, payload, record.offset); err != nil {
		if decoderShortRead(err) {
			return nil, ErrIVFShortFramePayload
		}
		return nil, err
	}
	return payload, nil
}

func (s decoderPayloadSource) forEachPayload(buf []byte, fn func([]byte) error) error {
	if s.readerAt == nil {
		for _, payload := range s.payloads {
			if err := fn(payload); err != nil {
				return err
			}
		}
		return nil
	}
	if len(buf) < s.maxPayload {
		return ErrFrameShortBuffer
	}
	for i := range s.records {
		payload, err := s.payloadAt(i, buf)
		if err != nil {
			return err
		}
		if err := fn(payload); err != nil {
			return err
		}
	}
	return nil
}

func decoderFrameWorkResidualLowOverheadStreamsPlanFromSource(stream DecoderStream, source decoderPayloadSource, workers int, payloadBuf []byte, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamPlan, error) {
	var plan DecoderFrameWorkResidualStreamPlan
	outputs := 0
	err := source.forEachPayload(payloadBuf, func(src []byte) error {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(src)
		nextSize := DecoderFrameWorkResidualStreamScratchSize{Events: eventCount}
		plan.Size = plan.Size.Max(nextSize)
		if err != nil {
			return err
		}
		if len(events) < eventCount {
			return ErrDecoderEventBufferTooSmall
		}

		sequence, _ := stream.SequenceHeader()
		count, err := stream.PushLowOverhead(src, events[:eventCount])
		if err != nil {
			return err
		}
		eventSize, err := DecoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
		if err != nil {
			return err
		}
		nextBind := DecoderFrameWorkResidualEventsBindPlan(sequence, events[:count])
		if nextBind.HasEvent() {
			plan.Bind = nextBind
		} else if !plan.Bind.HasEvent() {
			plan.Bind.Sequence = nextBind.Sequence
		}
		nextSize.Event = eventSize
		outputs += eventSize.Outputs
		plan.Size = plan.Size.Max(nextSize)
		plan.Size.Event.Outputs = outputs
		return nil
	})
	return plan, err
}

func decoderFrameWorkResidualRTPPayloadsStreamPlanFromSource(stream DecoderStream, source decoderPayloadSource, workers int, events []DecoderEvent, spans []TileSpan, jobs []TileJob, batches []TileBatch) (DecoderFrameWorkResidualStreamPlan, error) {
	rtpBufferBytes, err := source.totalPayloadLen()
	if err != nil {
		return DecoderFrameWorkResidualStreamPlan{}, err
	}
	rtpBuffer := make([]byte, rtpBufferBytes)
	rtpSpans := make([]RTPObuSpan, source.len()*16+64)

	var plan DecoderFrameWorkResidualStreamPlan
	outputs := 0
	used := 0
	err = source.forEachPayload(nil, func(payload []byte) error {
		plannedUsed, eventCount, err := stream.PushRTPPayloadSize(used, payload)
		nextSize := DecoderFrameWorkResidualStreamScratchSize{
			Events:    eventCount,
			RTPBuffer: plannedUsed,
			RTPSpans:  eventCount,
		}
		plan.Size = plan.Size.Max(nextSize)
		if err != nil {
			return err
		}
		if len(rtpBuffer) < plannedUsed {
			return ErrRTPShortBuffer
		}
		if len(rtpSpans) < eventCount {
			rtpSpans = make([]RTPObuSpan, eventCount)
		}
		if len(events) < eventCount {
			events = make([]DecoderEvent, eventCount)
		}

		sequence, _ := stream.SequenceHeader()
		actualUsed, count, err := stream.PushRTPPayload(rtpBuffer[:plannedUsed], used, rtpSpans[:eventCount], events[:eventCount], payload)
		if err != nil {
			return err
		}
		if actualUsed != plannedUsed || count != eventCount {
			return ErrDecoderInvalidFrameWorkState
		}
		eventSize, err := DecoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
		if err != nil {
			return err
		}
		nextBind := DecoderFrameWorkResidualEventsBindPlan(sequence, events[:count])
		if nextBind.HasEvent() {
			plan.Bind = nextBind
		} else if !plan.Bind.HasEvent() {
			plan.Bind.Sequence = nextBind.Sequence
		}
		nextSize.Event = eventSize
		outputs += eventSize.Outputs
		plan.Size = plan.Size.Max(nextSize)
		plan.Size.Event.Outputs = outputs
		used = actualUsed
		if !stream.InRTPFragment() {
			used = 0
		}
		return nil
	})
	return plan, err
}

func decoderExternalOutputFormatFromSource(source decoderPayloadSource, align int, payloadBuf []byte) (bool, FrameFormat, error) {
	var stream DecoderStream
	var format FrameFormat
	var events []DecoderEvent
	have := false
	err := source.forEachPayload(payloadBuf, func(payload []byte) error {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
		if err != nil {
			return err
		}
		if len(events) < eventCount {
			events = make([]DecoderEvent, eventCount)
		}
		sequence, _ := stream.SequenceHeader()
		count, err := stream.PushLowOverhead(payload, events[:eventCount])
		if err != nil {
			return err
		}
		for i := range count {
			event := events[i]
			sequence = decoderFrameWorkResidualEventSequence(sequence, event)
			if !decoderFrameWorkResidualEventBindCandidate(event) ||
				!event.FrameSize.SuperResEnabled ||
				event.FrameSize.UpscaledWidth <= event.FrameSize.CodedWidth {
				continue
			}
			next, err := FrameOutputFormatFromHeaders(sequence, event.FrameSize, align)
			if err != nil {
				return err
			}
			if have && next != format {
				return ErrFrameInvalidFormat
			}
			format = next
			have = true
		}
		return nil
	})
	return have, format, err
}

func decoderExternalOutputFormatFromRTPSource(source decoderPayloadSource, align int) (bool, FrameFormat, error) {
	var format FrameFormat
	have := false
	err := decoderForEachRTPEventFromSource(source, func(sequence SequenceHeader, event DecoderEvent) error {
		if !decoderFrameWorkResidualEventBindCandidate(event) ||
			!event.FrameSize.SuperResEnabled ||
			event.FrameSize.UpscaledWidth <= event.FrameSize.CodedWidth {
			return nil
		}
		next, err := FrameOutputFormatFromHeaders(sequence, event.FrameSize, align)
		if err != nil {
			return err
		}
		if have && next != format {
			return ErrFrameInvalidFormat
		}
		format = next
		have = true
		return nil
	})
	return have, format, err
}

func decoderPostFilterScratchArenaUpperBoundFromSource(source decoderPayloadSource, align int, events []DecoderEvent, payloadBuf []byte) (DecoderFrameWorkPostFilterRequestScratchSize, error) {
	var stream DecoderStream
	var arena DecoderFrameWorkPostFilterRequestScratchSize
	err := source.forEachPayload(payloadBuf, func(payload []byte) error {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
		if err != nil {
			return err
		}
		if len(events) < eventCount {
			return ErrDecoderEventBufferTooSmall
		}
		sequence, _ := stream.SequenceHeader()
		count, err := stream.PushLowOverhead(payload, events[:eventCount])
		if err != nil {
			return err
		}
		for i := range count {
			event := events[i]
			sequence = decoderFrameWorkResidualEventSequence(sequence, event)
			if !decoderFrameWorkResidualEventBindCandidate(event) {
				continue
			}
			size, err := decoderPostFilterScratchSizeUpperBound(sequence, event, align)
			if err != nil {
				return err
			}
			arena = arena.Max(decoderExternalPostFilterScratchLen(size))
		}
		return nil
	})
	return arena, err
}

func decoderPostFilterScratchArenaUpperBoundFromRTPSource(source decoderPayloadSource, align int) (DecoderFrameWorkPostFilterRequestScratchSize, error) {
	var arena DecoderFrameWorkPostFilterRequestScratchSize
	err := decoderForEachRTPEventFromSource(source, func(sequence SequenceHeader, event DecoderEvent) error {
		if !decoderFrameWorkResidualEventBindCandidate(event) {
			return nil
		}
		size, err := decoderPostFilterScratchSizeUpperBound(sequence, event, align)
		if err != nil {
			return err
		}
		arena = arena.Max(decoderExternalPostFilterScratchLen(size))
		return nil
	})
	return arena, err
}

func decoderTemporalMotionScratchFrameSizeUpperBoundFromSource(source decoderPayloadSource, events []DecoderEvent, payloadBuf []byte) (FrameSize, error) {
	var stream DecoderStream
	var maxSize FrameSize
	err := source.forEachPayload(payloadBuf, func(payload []byte) error {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
		if err != nil {
			return err
		}
		if len(events) < eventCount {
			return ErrDecoderEventBufferTooSmall
		}
		count, err := stream.PushLowOverhead(payload, events[:eventCount])
		if err != nil {
			return err
		}
		for i := range count {
			event := events[i]
			if !decoderFrameWorkResidualEventBindCandidate(event) ||
				event.FrameHeader.ShowExistingFrame {
				continue
			}
			if event.FrameSize.CodedWidth > maxSize.CodedWidth {
				maxSize.CodedWidth = event.FrameSize.CodedWidth
			}
			if event.FrameSize.Height > maxSize.Height {
				maxSize.Height = event.FrameSize.Height
			}
		}
		return nil
	})
	return maxSize, err
}

func decoderTemporalMotionScratchFrameSizeUpperBoundFromRTPSource(source decoderPayloadSource) (FrameSize, error) {
	var maxSize FrameSize
	err := decoderForEachRTPEventFromSource(source, func(_ SequenceHeader, event DecoderEvent) error {
		if !decoderFrameWorkResidualEventBindCandidate(event) ||
			event.FrameHeader.ShowExistingFrame {
			return nil
		}
		if event.FrameSize.CodedWidth > maxSize.CodedWidth {
			maxSize.CodedWidth = event.FrameSize.CodedWidth
		}
		if event.FrameSize.Height > maxSize.Height {
			maxSize.Height = event.FrameSize.Height
		}
		return nil
	})
	return maxSize, err
}

func decoderForEachRTPEventFromSource(source decoderPayloadSource, fn func(SequenceHeader, DecoderEvent) error) error {
	rtpBufferBytes, err := source.totalPayloadLen()
	if err != nil {
		return err
	}
	rtpBuffer := make([]byte, rtpBufferBytes)
	rtpSpans := make([]RTPObuSpan, source.len()*16+64)
	events := make([]DecoderEvent, source.len()*16+64)
	var stream DecoderStream
	used := 0
	return source.forEachPayload(nil, func(payload []byte) error {
		plannedUsed, eventCount, err := stream.PushRTPPayloadSize(used, payload)
		if err != nil {
			return err
		}
		if len(rtpBuffer) < plannedUsed {
			return ErrRTPShortBuffer
		}
		if len(rtpSpans) < eventCount {
			rtpSpans = make([]RTPObuSpan, eventCount)
		}
		if len(events) < eventCount {
			events = make([]DecoderEvent, eventCount)
		}
		sequence, _ := stream.SequenceHeader()
		actualUsed, count, err := stream.PushRTPPayload(rtpBuffer[:plannedUsed], used, rtpSpans[:eventCount], events[:eventCount], payload)
		if err != nil {
			return err
		}
		if actualUsed != plannedUsed || count != eventCount {
			return ErrDecoderInvalidFrameWorkState
		}
		for i := range count {
			event := events[i]
			sequence = decoderFrameWorkResidualEventSequence(sequence, event)
			if err := fn(sequence, event); err != nil {
				return err
			}
		}
		used = actualUsed
		if !stream.InRTPFragment() {
			used = 0
		}
		return nil
	})
}

func decoderReadFullAt(r io.ReaderAt, dst []byte, off int64) error {
	if len(dst) == 0 {
		return nil
	}
	n, err := r.ReadAt(dst, off)
	if n == len(dst) {
		return nil
	}
	if err != nil {
		return err
	}
	return io.ErrUnexpectedEOF
}

func decoderShortRead(err error) bool {
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
}

func decoderMaxInt() int {
	return int(^uint(0) >> 1)
}
