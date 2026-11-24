package goav1

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	internalthreading "github.com/thesyncim/goav1/internal/av1/threading"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
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
	if size < IVFFileHeaderSize {
		return decoderPayloadSource{}, IVFHeader{}, ErrIVFShortHeader
	}
	// ReadAt is an interface call, so its destination escapes: one heap
	// buffer serves the file header and then every frame header, instead of
	// one escaping array per frame.
	hdrBuf := make([]byte, max(IVFFileHeaderSize, IVFFrameHeaderSize))
	headerBuf := hdrBuf[:IVFFileHeaderSize]
	if err := decoderReadFullAt(r, headerBuf, 0); err != nil {
		if decoderShortRead(err) {
			return decoderPayloadSource{}, IVFHeader{}, ErrIVFShortHeader
		}
		return decoderPayloadSource{}, IVFHeader{}, err
	}
	header, n, err := ParseIVFHeader(headerBuf)
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
		if size-off < IVFFrameHeaderSize {
			return decoderPayloadSource{}, header, ErrIVFShortFrameHeader
		}
		frameHeader := hdrBuf[:IVFFrameHeaderSize]
		if err := decoderReadFullAt(r, frameHeader, off); err != nil {
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
				return 0, internalframe.ErrShortBuffer
			}
			total = next
		}
		return total, nil
	}
	for _, record := range s.records {
		next, ok := decoderArenaAdd2(total, int(record.size))
		if !ok {
			return 0, internalframe.ErrShortBuffer
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
			return nil, internaldecoder.ErrInvalidFrameWorkState
		}
		return s.payloads[i], nil
	}
	if i < 0 || i >= len(s.records) {
		return nil, internaldecoder.ErrInvalidFrameWorkState
	}
	record := s.records[i]
	size := int(record.size)
	if len(buf) < size {
		return nil, internalframe.ErrShortBuffer
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
		return internalframe.ErrShortBuffer
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

func decoderFrameWorkResidualLowOverheadStreamsPlanFromSource(stream internaldecoder.Stream, source decoderPayloadSource, workers int, frameSizeLimit int, payloadBuf []byte, events []internaldecoder.Event, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch) (decoderFrameWorkResidualStreamPlan, error) {
	var plan decoderFrameWorkResidualStreamPlan
	outputs := 0
	err := source.forEachPayload(payloadBuf, func(src []byte) error {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(src)
		nextSize := decoderFrameWorkResidualStreamScratchSize{Events: eventCount}
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
		if err := decoderCheckFrameSizeLimit(frameSizeLimit, events[:count]); err != nil {
			return err
		}
		eventSize, err := decoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
		if err != nil {
			return err
		}
		nextBind := decoderFrameWorkResidualEventsBindPlan(sequence, events[:count])
		if plan.Bind.prefer(nextBind) {
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

func decoderFrameWorkResidualRTPPayloadsStreamPlanFromSource(stream internaldecoder.Stream, source decoderPayloadSource, workers int, frameSizeLimit int, events []internaldecoder.Event, spans []internalparser.TileSpan, jobs []internaltile.Job, batches []internalthreading.Batch) (decoderFrameWorkResidualStreamPlan, error) {
	rtpBufferBytes, err := source.totalPayloadLen()
	if err != nil {
		return decoderFrameWorkResidualStreamPlan{}, err
	}
	rtpBuffer := make([]byte, rtpBufferBytes)
	rtpSpans := make([]RTPObuSpan, decoderMaxPayloadEvents)

	var plan decoderFrameWorkResidualStreamPlan
	outputs := 0
	used := 0
	err = source.forEachPayload(nil, func(payload []byte) error {
		plannedUsed, eventCount, err := stream.PushRTPPayloadSize(used, payload)
		nextSize := decoderFrameWorkResidualStreamScratchSize{
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
		if len(rtpSpans) < eventCount || len(events) < eventCount {
			return ErrDecoderEventBufferTooSmall
		}

		sequence, _ := stream.SequenceHeader()
		actualUsed, count, err := stream.PushRTPPayload(rtpBuffer[:plannedUsed], used, rtpSpans[:eventCount], events[:eventCount], payload)
		if err != nil {
			return err
		}
		if actualUsed != plannedUsed || count != eventCount {
			return internaldecoder.ErrInvalidFrameWorkState
		}
		if err := decoderCheckFrameSizeLimit(frameSizeLimit, events[:count]); err != nil {
			return err
		}
		eventSize, err := decoderFrameWorkResidualEventsScratchLen(sequence, events[:count], workers, spans, jobs, batches)
		if err != nil {
			return err
		}
		nextBind := decoderFrameWorkResidualEventsBindPlan(sequence, events[:count])
		if plan.Bind.prefer(nextBind) {
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
	var formats []FrameFormat
	err := decoderForEachLowOverheadEventFromSource(source, payloadBuf, func(sequence SequenceHeader, event *internaldecoder.Event) error {
		return decoderAddExternalOutputFormat(&formats, sequence, event, align)
	})
	if err != nil || len(formats) == 0 {
		return false, FrameFormat{}, err
	}
	format, err := decoderLargestFrameFormat(formats)
	return err == nil, format, err
}

func decoderExternalOutputFormatFromRTPSource(source decoderPayloadSource, align int) (bool, FrameFormat, error) {
	var formats []FrameFormat
	err := decoderForEachRTPEventFromSource(source, func(sequence SequenceHeader, event internaldecoder.Event) error {
		return decoderAddExternalOutputFormat(&formats, sequence, &event, align)
	})
	if err != nil || len(formats) == 0 {
		return false, FrameFormat{}, err
	}
	format, err := decoderLargestFrameFormat(formats)
	return err == nil, format, err
}

func decoderAddExternalOutputFormat(formats *[]FrameFormat, sequence SequenceHeader, event *internaldecoder.Event, align int) error {
	if !decoderFrameWorkResidualEventBindCandidate(event) {
		return nil
	}
	next, ok, err := decoderExternalOutputFormatForEvent(sequence, event, align)
	if err != nil || !ok {
		return err
	}
	*formats = decoderAddFrameFormat(*formats, next)
	return nil
}

// decoderCodedPoolFormatFromSource returns the frame-pool slot format for a
// stream: the largest layout over every sample format its coded frames use,
// each at that format's largest frame size. A new coded video sequence may
// change bit depth, chroma layout or superblock size; every slot fits any
// picture of the stream, as dav1d's picture pool serves every picture of a
// stream whatever its format.
func decoderCodedPoolFormatFromSource(source decoderPayloadSource, kind decoderPayloadKind, align int, payloadBuf []byte) (FrameFormat, error) {
	var formats []FrameFormat
	add := func(sequence SequenceHeader, event *internaldecoder.Event) error {
		if !decoderFrameWorkResidualEventBindCandidate(event) {
			return nil
		}
		format, err := frameCodedFormatFromHeaders(sequence, event.FrameSize, align)
		if err != nil {
			return err
		}
		formats = decoderAddFrameFormat(formats, format)
		return nil
	}
	var err error
	switch kind {
	case decoderPayloadLowOverhead:
		err = decoderForEachLowOverheadEventFromSource(source, payloadBuf, add)
	case decoderPayloadRTP:
		err = decoderForEachRTPEventFromSource(source, func(sequence SequenceHeader, event internaldecoder.Event) error {
			return add(sequence, &event)
		})
	default:
		err = internaldecoder.ErrInvalidFrameWorkState
	}
	if err != nil {
		return FrameFormat{}, err
	}
	if len(formats) == 0 {
		return FrameFormat{}, internalframe.ErrInvalidFormat
	}
	return decoderLargestFrameFormat(formats)
}

// decoderAddFrameFormat folds next into formats, which holds one entry per
// sample format (everything but the frame size) at the largest width and
// height seen for it.
func decoderAddFrameFormat(formats []FrameFormat, next FrameFormat) []FrameFormat {
	for i := range formats {
		f := formats[i]
		f.Width, f.Height = next.Width, next.Height
		if f == next {
			formats[i].Width = max(formats[i].Width, next.Width)
			formats[i].Height = max(formats[i].Height, next.Height)
			return formats
		}
	}
	return append(formats, next)
}

// decoderLargestFrameFormat returns the entry of formats with the largest
// frame layout. Pool slots are sized for it, and every other entry fits one.
func decoderLargestFrameFormat(formats []FrameFormat) (FrameFormat, error) {
	best, bestSize := FrameFormat{}, -1
	for _, f := range formats {
		layout, err := internalframe.RequiredSize(f)
		if err != nil {
			return FrameFormat{}, err
		}
		if layout.Size > bestSize {
			best, bestSize = f, layout.Size
		}
	}
	return best, nil
}

// decoderForEachLowOverheadEventFromSource parses every low-overhead payload
// of source and calls fn with each event and the sequence header it decodes
// under.
func decoderForEachLowOverheadEventFromSource(source decoderPayloadSource, payloadBuf []byte, fn func(SequenceHeader, *internaldecoder.Event) error) error {
	var stream internaldecoder.Stream
	events := make([]internaldecoder.Event, decoderMaxPayloadEvents)
	return source.forEachPayload(payloadBuf, func(payload []byte) error {
		eventCount, err := decoderFrameWorkResidualLowOverheadEventLen(payload)
		if err != nil {
			return err
		}
		if eventCount > decoderMaxPayloadEvents {
			return ErrDecoderEventBufferTooSmall
		}
		sequence, _ := stream.SequenceHeader()
		count, err := stream.PushLowOverhead(payload, events[:eventCount])
		if err != nil {
			return err
		}
		for i := range count {
			sequence = decoderFrameWorkResidualEventSequence(sequence, &events[i])
			if err := fn(sequence, &events[i]); err != nil {
				return err
			}
		}
		return nil
	})
}

func decoderExternalOutputFormatForEvent(sequence SequenceHeader, event *internaldecoder.Event, align int) (FrameFormat, bool, error) {
	if event.Kind == internaldecoder.EventTileList {
		if event.TileListErr != nil {
			return FrameFormat{}, false, event.TileListErr
		}
		geometry, err := internalparser.TileListOutputGeometryForGrid(event.TileList, event.TileInfo, sequence.Use128x128Superblock)
		if err != nil {
			return FrameFormat{}, false, err
		}
		format, err := tileListOutputFrameFormat(sequence, geometry, align)
		if err != nil {
			return FrameFormat{}, false, err
		}
		return format, true, nil
	}
	if !event.FrameSize.SuperResEnabled || event.FrameSize.UpscaledWidth <= event.FrameSize.CodedWidth {
		return FrameFormat{}, false, nil
	}
	format, err := frameOutputFormatFromHeaders(sequence, event.FrameSize, align)
	if err != nil {
		return FrameFormat{}, false, err
	}
	return format, true, nil
}

func decoderPostFilterScratchArenaUpperBoundFromSource(source decoderPayloadSource, align int, workerPool *internalthreading.Pool, events []internaldecoder.Event, payloadBuf []byte) (decoderFrameWorkPostFilterRequestScratchSize, error) {
	var scratch decoderUpperBoundScratch
	var stream internaldecoder.Stream
	var arena decoderFrameWorkPostFilterRequestScratchSize
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
			sequence = decoderFrameWorkResidualEventSequence(sequence, &event)
			if !decoderFrameWorkResidualEventBindCandidate(&event) {
				continue
			}
			size, err := decoderPostFilterScratchSizeUpperBound(sequence, &event, align, workerPool, &scratch)
			if err != nil {
				return err
			}
			arena = arena.Max(decoderExternalPostFilterScratchLen(size))
		}
		return nil
	})
	return arena, err
}

func decoderPostFilterScratchArenaUpperBoundFromRTPSource(source decoderPayloadSource, align int, workerPool *internalthreading.Pool) (decoderFrameWorkPostFilterRequestScratchSize, error) {
	var scratch decoderUpperBoundScratch
	var arena decoderFrameWorkPostFilterRequestScratchSize
	err := decoderForEachRTPEventFromSource(source, func(sequence SequenceHeader, event internaldecoder.Event) error {
		if !decoderFrameWorkResidualEventBindCandidate(&event) {
			return nil
		}
		size, err := decoderPostFilterScratchSizeUpperBound(sequence, &event, align, workerPool, &scratch)
		if err != nil {
			return err
		}
		arena = arena.Max(decoderExternalPostFilterScratchLen(size))
		return nil
	})
	return arena, err
}

func decoderTemporalMotionScratchFrameSizeUpperBoundFromSource(source decoderPayloadSource, events []internaldecoder.Event, payloadBuf []byte) (FrameSize, error) {
	var stream internaldecoder.Stream
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
			if !decoderFrameWorkResidualEventBindCandidate(&event) ||
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
	err := decoderForEachRTPEventFromSource(source, func(_ SequenceHeader, event internaldecoder.Event) error {
		if !decoderFrameWorkResidualEventBindCandidate(&event) ||
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

func decoderForEachRTPEventFromSource(source decoderPayloadSource, fn func(SequenceHeader, internaldecoder.Event) error) error {
	rtpBufferBytes, err := source.totalPayloadLen()
	if err != nil {
		return err
	}
	rtpBuffer := make([]byte, rtpBufferBytes)
	rtpSpans := make([]RTPObuSpan, decoderMaxPayloadEvents)
	events := make([]internaldecoder.Event, decoderMaxPayloadEvents)
	var stream internaldecoder.Stream
	used := 0
	return source.forEachPayload(nil, func(payload []byte) error {
		plannedUsed, eventCount, err := stream.PushRTPPayloadSize(used, payload)
		if err != nil {
			return err
		}
		if len(rtpBuffer) < plannedUsed {
			return ErrRTPShortBuffer
		}
		if eventCount > decoderMaxPayloadEvents {
			return ErrDecoderEventBufferTooSmall
		}
		sequence, _ := stream.SequenceHeader()
		actualUsed, count, err := stream.PushRTPPayload(rtpBuffer[:plannedUsed], used, rtpSpans[:eventCount], events[:eventCount], payload)
		if err != nil {
			return err
		}
		if actualUsed != plannedUsed || count != eventCount {
			return internaldecoder.ErrInvalidFrameWorkState
		}
		for i := range count {
			event := events[i]
			sequence = decoderFrameWorkResidualEventSequence(sequence, &event)
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
