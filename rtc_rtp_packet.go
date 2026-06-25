package goav1

// EncoderWebRTCRTPPacketHeaderConfig describes the RTP fixed-header fields and
// dependency-descriptor header-extension mapping used to wrap encoded AV1 RTP
// payload bodies from RTCFrame.AppendRTPPackets.
type EncoderWebRTCRTPPacketHeaderConfig struct {
	PayloadType    uint8
	SequenceNumber uint16
	Timestamp      uint32
	SSRC           uint32

	// DependencyDescriptorExtensionID is the negotiated extmap ID for WebRTC's
	// dependency descriptor RTP header extension.
	DependencyDescriptorExtensionID uint8
	// MIDExtensionID and MID, when set, add the negotiated RTP MID extension to
	// every packet.
	MIDExtensionID uint8
	MID            string
	// RTPStreamIDExtensionID and RTPStreamID, when set, add the negotiated RID
	// RTP header extension to every packet.
	RTPStreamIDExtensionID uint8
	RTPStreamID            string
	// RepairedRTPStreamIDExtensionID and RepairedRTPStreamID, when set, add the
	// negotiated repaired-RID RTP header extension to every packet.
	RepairedRTPStreamIDExtensionID uint8
	RepairedRTPStreamID            string
	// TransportWideCCExtensionID, when set, adds WebRTC's transport-wide
	// congestion-control sequence extension to every packet. The payload value
	// starts at TransportWideCCSequenceNumber and increments once per packet.
	TransportWideCCExtensionID    uint8
	TransportWideCCSequenceNumber uint16
	// TransportWideCC02ExtensionID, when set, adds WebRTC's transport-wide-cc-02
	// extension to every packet. TransportWideCC02.SequenceNumber is incremented
	// once per packet while the feedback-request fields are preserved.
	TransportWideCC02ExtensionID uint8
	TransportWideCC02            RTPTransportWideCC02
	// VideoOrientationExtensionID, when set, adds WebRTC's coordination-of-video
	// orientation RTP header extension to every packet.
	VideoOrientationExtensionID uint8
	VideoOrientation            RTPCoordinationOfVideoOrientation
	// PlayoutDelayExtensionID, when set, adds WebRTC's playout-delay RTP header
	// extension to every packet.
	PlayoutDelayExtensionID uint8
	PlayoutDelay            RTPPlayoutDelay
	// AbsoluteSendTimeExtensionID, when set, adds WebRTC's absolute-send-time
	// RTP header extension to every packet.
	AbsoluteSendTimeExtensionID uint8
	AbsoluteSendTime            uint32
	// AbsoluteCaptureTimeExtensionID, when set, adds WebRTC's absolute-capture-
	// time RTP header extension to every packet.
	AbsoluteCaptureTimeExtensionID uint8
	AbsoluteCaptureTime            RTPAbsoluteCaptureTime
	// VideoContentTypeExtensionID, when set, adds WebRTC's video-content-type
	// RTP header extension to every packet.
	VideoContentTypeExtensionID uint8
	VideoContentType            RTPVideoContentType
	// VideoTimingExtensionID, when set, adds WebRTC's video-timing RTP header
	// extension to every packet.
	VideoTimingExtensionID uint8
	VideoTiming            RTPVideoTiming
	// ColorSpaceExtensionID, when set, adds WebRTC's color-space RTP header
	// extension to every packet.
	ColorSpaceExtensionID uint8
	ColorSpace            RTPColorSpace
	// HeaderExtensionProfile selects RFC 8285 one-byte or two-byte extension
	// element framing. Zero defaults to the two-byte profile because dependency
	// descriptors with attached structures often exceed one-byte's 16 byte
	// element limit.
	HeaderExtensionProfile uint16
}

// EncoderWebRTCRTPPacketHeaderSpan describes one complete RTP packet appended by
// AppendEncoderWebRTCRTPPacketsWithHeaders.
type EncoderWebRTCRTPPacketHeaderSpan struct {
	Offset int
	Length int

	HeaderSize    int
	PayloadOffset int
	PayloadLength int

	DependencyDescriptorOffset int
	DependencyDescriptorLength int

	SequenceNumber uint16
	Marker         bool
}

// EncoderWebRTCRTPPacketsWithHeadersSizeInfo reports the complete RTP packet
// byte counts needed by AppendEncoderWebRTCRTPPacketsWithHeaders.
type EncoderWebRTCRTPPacketsWithHeadersSizeInfo struct {
	Packets        int
	Bytes          int
	MaxPacketBytes int
	MaxHeaderBytes int
}

// EncoderWebRTCRTPPacketsWithHeadersSize validates a packet span plan and
// reports the exact destination size required to wrap all AV1 RTP payload bodies
// and negotiated RTP header extensions into complete RTP packets.
func EncoderWebRTCRTPPacketsWithHeadersSize(config EncoderWebRTCRTPPacketHeaderConfig, rtpPayloads []byte, descriptors []byte, packetSpans []EncoderWebRTCRTPPacketSpan) (EncoderWebRTCRTPPacketsWithHeadersSizeInfo, error) {
	var size EncoderWebRTCRTPPacketsWithHeadersSizeInfo
	if err := validateEncoderWebRTCRTPPacketHeaderConfig(config); err != nil {
		return size, err
	}
	profile := encoderWebRTCRTPPacketHeaderExtensionProfile(config.HeaderExtensionProfile)
	maxInt := int(^uint(0) >> 1)

	for i := range packetSpans {
		payload, descriptor, err := encoderWebRTCRTPPacketSpanSlices(rtpPayloads, descriptors, packetSpans[i])
		if err != nil {
			return size, err
		}
		descriptorLen := 0
		if config.DependencyDescriptorExtensionID != 0 {
			if len(descriptor) == 0 {
				return size, ErrEncoderInvalidFrame
			}
			descriptorLen = len(descriptor)
		}
		packetSize, err := encoderWebRTCRTPPacketHeaderSize(profile, config, i, len(payload), descriptorLen)
		if err != nil {
			return size, err
		}
		if packetSize > maxInt-size.Bytes {
			return size, ErrEncoderInvalidFrame
		}
		headerSize := packetSize - len(payload)
		size.Packets++
		size.Bytes += packetSize
		if packetSize > size.MaxPacketBytes {
			size.MaxPacketBytes = packetSize
		}
		if headerSize > size.MaxHeaderBytes {
			size.MaxHeaderBytes = headerSize
		}
	}
	return size, nil
}

// AppendEncoderWebRTCRTPPacketsWithHeaders wraps AV1 RTP payload bodies and
// negotiated RTP header extensions into complete RTP packets. The function writes
// RFC 8285 extension elements and fixed RTP headers; SRTP, pacing,
// retransmission, and network transport remain caller-owned.
func AppendEncoderWebRTCRTPPacketsWithHeaders(dst []byte, headerSpans []EncoderWebRTCRTPPacketHeaderSpan, config EncoderWebRTCRTPPacketHeaderConfig, rtpPayloads []byte, descriptors []byte, packetSpans []EncoderWebRTCRTPPacketSpan) ([]byte, int, error) {
	if len(headerSpans) < len(packetSpans) {
		return dst, 0, ErrRTPPacketPlanTooSmall
	}
	if _, err := EncoderWebRTCRTPPacketsWithHeadersSize(config, rtpPayloads, descriptors, packetSpans); err != nil {
		return dst, 0, err
	}
	profile := encoderWebRTCRTPPacketHeaderExtensionProfile(config.HeaderExtensionProfile)

	out := dst
	for i := range packetSpans {
		span := packetSpans[i]
		payload, descriptor, err := encoderWebRTCRTPPacketSpanSlices(rtpPayloads, descriptors, span)
		if err != nil {
			return dst, 0, err
		}
		descriptorLen := 0
		if config.DependencyDescriptorExtensionID != 0 {
			descriptorLen = len(descriptor)
		}
		start := len(out)
		packetSize, err := encoderWebRTCRTPPacketHeaderSize(profile, config, i, len(payload), descriptorLen)
		if err != nil {
			return dst, 0, err
		}
		out = appendZeroedBytes(out, packetSize)
		packet := out[start : start+packetSize]

		var extensionPayloads encoderWebRTCRTPPacketHeaderExtensionPayloads
		elements, elementCount, err := encoderWebRTCRTPPacketHeaderExtensionElements(config, i, descriptor, &extensionPayloads)
		if err != nil {
			return dst, 0, err
		}
		extHeaderLen, err := RTPHeaderExtensionElementsSize(profile, elements[:elementCount])
		if err != nil {
			return dst, 0, err
		}
		extPayloadStart := RTPHeaderMinSize + 4
		n, err := PutRTPHeaderExtensionElements(packet[extPayloadStart:extPayloadStart+extHeaderLen], profile, elements[:elementCount])
		if err != nil {
			return dst, 0, err
		}
		descriptorOffset := -1
		if config.DependencyDescriptorExtensionID != 0 {
			descriptorOffset = start + extPayloadStart + encoderWebRTCRTPPacketHeaderExtensionElementPrefixLen(profile)
		}
		header := RTPHeader{
			Marker:           span.Marker,
			PayloadType:      config.PayloadType,
			SequenceNumber:   config.SequenceNumber + uint16(i),
			Timestamp:        config.Timestamp,
			SSRC:             config.SSRC,
			ExtensionProfile: profile,
			ExtensionPayload: packet[extPayloadStart : extPayloadStart+n],
		}
		headerSize, err := PutRTPHeader(packet, header)
		if err != nil {
			return dst, 0, err
		}
		copy(packet[headerSize:], payload)
		headerSpans[i] = EncoderWebRTCRTPPacketHeaderSpan{
			Offset:                     start,
			Length:                     packetSize,
			HeaderSize:                 headerSize,
			PayloadOffset:              start + headerSize,
			PayloadLength:              len(payload),
			DependencyDescriptorOffset: descriptorOffset,
			DependencyDescriptorLength: descriptorLen,
			SequenceNumber:             header.SequenceNumber,
			Marker:                     span.Marker,
		}
	}
	return out, len(packetSpans), nil
}

func appendZeroedBytes(dst []byte, n int) []byte {
	if n <= 0 {
		return dst
	}
	oldLen := len(dst)
	if n <= cap(dst)-oldLen {
		dst = dst[:oldLen+n]
		clear(dst[oldLen:])
		return dst
	}
	return append(dst, make([]byte, n)...)
}

func validateEncoderWebRTCRTPPacketHeaderConfig(config EncoderWebRTCRTPPacketHeaderConfig) error {
	if config.PayloadType > 127 {
		return ErrRTPInvalidHeader
	}
	var ids encoderWebRTCRTPPacketHeaderExtensionIDs
	if config.DependencyDescriptorExtensionID != 0 {
		if err := ids.add(config.DependencyDescriptorExtensionID); err != nil {
			return err
		}
	}
	if err := validateEncoderWebRTCRTPPacketHeaderStringExtension(&ids, config.MIDExtensionID, config.MID, ValidateRTPMID); err != nil {
		return err
	}
	if err := validateEncoderWebRTCRTPPacketHeaderStringExtension(&ids, config.RTPStreamIDExtensionID, config.RTPStreamID, ValidateRTPStreamID); err != nil {
		return err
	}
	if err := validateEncoderWebRTCRTPPacketHeaderStringExtension(&ids, config.RepairedRTPStreamIDExtensionID, config.RepairedRTPStreamID, ValidateRTPStreamID); err != nil {
		return err
	}
	if config.TransportWideCCExtensionID == 0 {
		if config.TransportWideCCSequenceNumber != 0 {
			return ErrRTPInvalidHeaderExtension
		}
	} else if err := ids.add(config.TransportWideCCExtensionID); err != nil {
		return err
	}
	if config.TransportWideCC02ExtensionID == 0 {
		if config.TransportWideCC02 != (RTPTransportWideCC02{}) {
			return ErrRTPInvalidHeaderExtension
		}
	} else {
		if err := ids.add(config.TransportWideCC02ExtensionID); err != nil {
			return err
		}
		if err := ValidateRTPTransportWideCC02(config.TransportWideCC02); err != nil {
			return err
		}
	}
	if err := validateEncoderWebRTCRTPPacketHeaderValueExtension(&ids, config.VideoOrientationExtensionID, config.VideoOrientation, ValidateRTPCoordinationOfVideoOrientation); err != nil {
		return err
	}
	if err := validateEncoderWebRTCRTPPacketHeaderValueExtension(&ids, config.PlayoutDelayExtensionID, config.PlayoutDelay, ValidateRTPPlayoutDelay); err != nil {
		return err
	}
	if config.AbsoluteSendTimeExtensionID == 0 {
		if config.AbsoluteSendTime != 0 {
			return ErrRTPInvalidHeaderExtension
		}
	} else {
		if err := ids.add(config.AbsoluteSendTimeExtensionID); err != nil {
			return err
		}
		if config.AbsoluteSendTime > RTPAbsoluteSendTimeMaxValue {
			return ErrRTPInvalidHeaderExtension
		}
	}
	if err := validateEncoderWebRTCRTPPacketHeaderValueExtension(&ids, config.AbsoluteCaptureTimeExtensionID, config.AbsoluteCaptureTime, ValidateRTPAbsoluteCaptureTime); err != nil {
		return err
	}
	if err := validateEncoderWebRTCRTPPacketHeaderValueExtension(&ids, config.VideoContentTypeExtensionID, config.VideoContentType, ValidateRTPVideoContentType); err != nil {
		return err
	}
	if err := validateEncoderWebRTCRTPPacketHeaderValueExtension(&ids, config.VideoTimingExtensionID, config.VideoTiming, ValidateRTPVideoTiming); err != nil {
		return err
	}
	if err := validateEncoderWebRTCRTPPacketHeaderValueExtension(&ids, config.ColorSpaceExtensionID, config.ColorSpace, ValidateRTPColorSpace); err != nil {
		return err
	}
	if ids.n == 0 {
		return ErrRTPInvalidHeaderExtension
	}
	profile := encoderWebRTCRTPPacketHeaderExtensionProfile(config.HeaderExtensionProfile)
	_, err := rtpHeaderExtensionProfileKind(profile)
	return err
}

func encoderWebRTCRTPPacketHeaderExtensionProfile(profile uint16) uint16 {
	if profile == 0 {
		return RTPExtensionProfileTwoByte
	}
	return profile
}

func encoderWebRTCRTPPacketHeaderExtensionElementPrefixLen(profile uint16) int {
	kind, err := rtpHeaderExtensionProfileKind(profile)
	if err != nil {
		return 0
	}
	if kind == rtpHeaderExtensionProfileOneByte {
		return 1
	}
	return 2
}

func encoderWebRTCRTPPacketHeaderSize(profile uint16, config EncoderWebRTCRTPPacketHeaderConfig, packetIndex int, payloadLen int, descriptorLen int) (int, error) {
	kind, err := rtpHeaderExtensionProfileKind(profile)
	if err != nil {
		return 0, err
	}
	extLen, err := encoderWebRTCRTPPacketHeaderExtensionPayloadSize(kind, config, packetIndex, descriptorLen)
	if err != nil {
		return 0, err
	}
	extPadded, err := rtpPaddedExtensionPayloadLen(extLen)
	if err != nil {
		return 0, err
	}
	if payloadLen < 0 {
		return 0, ErrEncoderInvalidFrame
	}
	return RTPHeaderMinSize + 4 + extPadded + payloadLen, nil
}

type encoderWebRTCRTPPacketHeaderExtensionIDs struct {
	seen [16]uint16
	n    int
}

func (ids *encoderWebRTCRTPPacketHeaderExtensionIDs) add(id uint8) error {
	if id == 0 {
		return ErrRTPInvalidHeaderExtension
	}
	for i := 0; i < ids.n; i++ {
		if ids.seen[i] == uint16(id) {
			return ErrRTPInvalidHeaderExtension
		}
	}
	if ids.n >= len(ids.seen) {
		return ErrRTPInvalidHeaderExtension
	}
	ids.seen[ids.n] = uint16(id)
	ids.n++
	return nil
}

func validateEncoderWebRTCRTPPacketHeaderStringExtension(
	ids *encoderWebRTCRTPPacketHeaderExtensionIDs,
	id uint8,
	value string,
	validate func(string) error,
) error {
	if id == 0 {
		if value != "" {
			return ErrRTPInvalidHeaderExtension
		}
		return nil
	}
	if err := ids.add(id); err != nil {
		return err
	}
	return validate(value)
}

func validateEncoderWebRTCRTPPacketHeaderValueExtension[T comparable](
	ids *encoderWebRTCRTPPacketHeaderExtensionIDs,
	id uint8,
	value T,
	validate func(T) error,
) error {
	var zero T
	if id == 0 {
		if value != zero {
			return ErrRTPInvalidHeaderExtension
		}
		return nil
	}
	if err := ids.add(id); err != nil {
		return err
	}
	return validate(value)
}

func encoderWebRTCRTPPacketHeaderExtensionPayloadSize(kind int, config EncoderWebRTCRTPPacketHeaderConfig, packetIndex int, descriptorLen int) (int, error) {
	if packetIndex < 0 {
		return 0, ErrEncoderInvalidFrame
	}
	size := 0
	if config.DependencyDescriptorExtensionID != 0 {
		if descriptorLen <= 0 {
			return 0, ErrEncoderInvalidFrame
		}
		n, err := rtpHeaderExtensionElementSize(kind, config.DependencyDescriptorExtensionID, descriptorLen)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.MIDExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.MIDExtensionID, len(config.MID))
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.RTPStreamIDExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.RTPStreamIDExtensionID, len(config.RTPStreamID))
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.RepairedRTPStreamIDExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.RepairedRTPStreamIDExtensionID, len(config.RepairedRTPStreamID))
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.TransportWideCCExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.TransportWideCCExtensionID, RTPTransportWideCCHeaderExtensionSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.TransportWideCC02ExtensionID != 0 {
		cc := config.TransportWideCC02
		cc.SequenceNumber += uint16(packetIndex)
		ccSize, err := RTPTransportWideCC02Size(cc)
		if err != nil {
			return 0, err
		}
		n, err := rtpHeaderExtensionElementSize(kind, config.TransportWideCC02ExtensionID, ccSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.VideoOrientationExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.VideoOrientationExtensionID, RTPCoordinationOfVideoOrientationHeaderExtensionSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.PlayoutDelayExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.PlayoutDelayExtensionID, RTPPlayoutDelayHeaderExtensionSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.AbsoluteSendTimeExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.AbsoluteSendTimeExtensionID, RTPAbsoluteSendTimeHeaderExtensionSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.AbsoluteCaptureTimeExtensionID != 0 {
		captureSize, err := RTPAbsoluteCaptureTimeSize(config.AbsoluteCaptureTime)
		if err != nil {
			return 0, err
		}
		n, err := rtpHeaderExtensionElementSize(kind, config.AbsoluteCaptureTimeExtensionID, captureSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.VideoContentTypeExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.VideoContentTypeExtensionID, RTPVideoContentTypeHeaderExtensionSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.VideoTimingExtensionID != 0 {
		n, err := rtpHeaderExtensionElementSize(kind, config.VideoTimingExtensionID, RTPVideoTimingHeaderExtensionSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	if config.ColorSpaceExtensionID != 0 {
		colorSpaceSize, err := RTPColorSpaceSize(config.ColorSpace)
		if err != nil {
			return 0, err
		}
		n, err := rtpHeaderExtensionElementSize(kind, config.ColorSpaceExtensionID, colorSpaceSize)
		if err != nil {
			return 0, err
		}
		size += n
	}
	return size, nil
}

type encoderWebRTCRTPPacketHeaderExtensionPayloads struct {
	MID                 [RTPHeaderExtensionSDESMaxLen]byte
	RTPStreamID         [RTPHeaderExtensionSDESMaxLen]byte
	RepairedRTPStreamID [RTPHeaderExtensionSDESMaxLen]byte
	TransportWideCC     [RTPTransportWideCCHeaderExtensionSize]byte
	TransportWideCC02   [RTPTransportWideCC02HeaderExtensionSize]byte
	VideoOrientation    [RTPCoordinationOfVideoOrientationHeaderExtensionSize]byte
	PlayoutDelay        [RTPPlayoutDelayHeaderExtensionSize]byte
	AbsoluteSendTime    [RTPAbsoluteSendTimeHeaderExtensionSize]byte
	AbsoluteCaptureTime [RTPAbsoluteCaptureTimeHeaderExtensionSize]byte
	VideoContentType    [RTPVideoContentTypeHeaderExtensionSize]byte
	VideoTiming         [RTPVideoTimingHeaderExtensionSize]byte
	ColorSpace          [RTPColorSpaceHeaderExtensionSize]byte
}

func encoderWebRTCRTPPacketHeaderExtensionElements(
	config EncoderWebRTCRTPPacketHeaderConfig,
	packetIndex int,
	descriptor []byte,
	payloads *encoderWebRTCRTPPacketHeaderExtensionPayloads,
) ([13]RTPHeaderExtensionElement, int, error) {
	var elements [13]RTPHeaderExtensionElement
	if packetIndex < 0 || payloads == nil {
		return elements, 0, ErrEncoderInvalidFrame
	}
	count := 0
	if config.DependencyDescriptorExtensionID != 0 {
		if len(descriptor) == 0 {
			return elements, 0, ErrEncoderInvalidFrame
		}
		elements[count] = RTPHeaderExtensionElement{
			ID:      config.DependencyDescriptorExtensionID,
			Payload: descriptor,
		}
		count++
	}

	if config.MIDExtensionID != 0 {
		n, err := PutRTPMIDHeaderExtension(payloads.MID[:], config.MID)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.MIDExtensionID, Payload: payloads.MID[:n]}
		count++
	}
	if config.RTPStreamIDExtensionID != 0 {
		n, err := PutRTPStreamIDHeaderExtension(payloads.RTPStreamID[:], config.RTPStreamID)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.RTPStreamIDExtensionID, Payload: payloads.RTPStreamID[:n]}
		count++
	}
	if config.RepairedRTPStreamIDExtensionID != 0 {
		n, err := PutRTPRepairedStreamIDHeaderExtension(payloads.RepairedRTPStreamID[:], config.RepairedRTPStreamID)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.RepairedRTPStreamIDExtensionID, Payload: payloads.RepairedRTPStreamID[:n]}
		count++
	}
	if config.TransportWideCCExtensionID != 0 {
		n, err := PutRTPTransportWideCCHeaderExtension(payloads.TransportWideCC[:], config.TransportWideCCSequenceNumber+uint16(packetIndex))
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.TransportWideCCExtensionID, Payload: payloads.TransportWideCC[:n]}
		count++
	}
	if config.TransportWideCC02ExtensionID != 0 {
		cc := config.TransportWideCC02
		cc.SequenceNumber += uint16(packetIndex)
		n, err := PutRTPTransportWideCC02HeaderExtension(payloads.TransportWideCC02[:], cc)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.TransportWideCC02ExtensionID, Payload: payloads.TransportWideCC02[:n]}
		count++
	}
	if config.VideoOrientationExtensionID != 0 {
		n, err := PutRTPCoordinationOfVideoOrientationHeaderExtension(payloads.VideoOrientation[:], config.VideoOrientation)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.VideoOrientationExtensionID, Payload: payloads.VideoOrientation[:n]}
		count++
	}
	if config.PlayoutDelayExtensionID != 0 {
		n, err := PutRTPPlayoutDelayHeaderExtension(payloads.PlayoutDelay[:], config.PlayoutDelay)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.PlayoutDelayExtensionID, Payload: payloads.PlayoutDelay[:n]}
		count++
	}
	if config.AbsoluteSendTimeExtensionID != 0 {
		n, err := PutRTPAbsoluteSendTimeHeaderExtension(payloads.AbsoluteSendTime[:], config.AbsoluteSendTime)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.AbsoluteSendTimeExtensionID, Payload: payloads.AbsoluteSendTime[:n]}
		count++
	}
	if config.AbsoluteCaptureTimeExtensionID != 0 {
		n, err := PutRTPAbsoluteCaptureTimeHeaderExtension(payloads.AbsoluteCaptureTime[:], config.AbsoluteCaptureTime)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.AbsoluteCaptureTimeExtensionID, Payload: payloads.AbsoluteCaptureTime[:n]}
		count++
	}
	if config.VideoContentTypeExtensionID != 0 {
		n, err := PutRTPVideoContentTypeHeaderExtension(payloads.VideoContentType[:], config.VideoContentType)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.VideoContentTypeExtensionID, Payload: payloads.VideoContentType[:n]}
		count++
	}
	if config.VideoTimingExtensionID != 0 {
		n, err := PutRTPVideoTimingHeaderExtension(payloads.VideoTiming[:], config.VideoTiming)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.VideoTimingExtensionID, Payload: payloads.VideoTiming[:n]}
		count++
	}
	if config.ColorSpaceExtensionID != 0 {
		n, err := PutRTPColorSpaceHeaderExtension(payloads.ColorSpace[:], config.ColorSpace)
		if err != nil {
			return elements, 0, err
		}
		elements[count] = RTPHeaderExtensionElement{ID: config.ColorSpaceExtensionID, Payload: payloads.ColorSpace[:n]}
		count++
	}
	return elements, count, nil
}

func encoderWebRTCRTPPacketSpanSlices(rtpPayloads []byte, descriptors []byte, span EncoderWebRTCRTPPacketSpan) ([]byte, []byte, error) {
	payload, ok := sliceByOffsetLen(rtpPayloads, span.PayloadOffset, span.PayloadLength)
	if !ok {
		return nil, nil, ErrEncoderInvalidFrame
	}
	descriptor, ok := sliceByOffsetLen(descriptors, span.DescriptorOffset, span.DescriptorLength)
	if !ok {
		return nil, nil, ErrEncoderInvalidFrame
	}
	return payload, descriptor, nil
}

func sliceByOffsetLen(src []byte, off int, n int) ([]byte, bool) {
	if off < 0 || n < 0 || off > len(src) || n > len(src)-off {
		return nil, false
	}
	return src[off : off+n], true
}
