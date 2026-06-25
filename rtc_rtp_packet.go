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
// and dependency descriptors into complete RTP packets.
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
		if len(descriptor) == 0 {
			return size, ErrEncoderInvalidFrame
		}
		packetSize, err := encoderWebRTCRTPPacketHeaderSize(profile, config, i, len(payload), len(descriptor))
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
// dependency descriptors from RTCFrame.AppendRTPPackets into complete RTP
// packets. The function writes RFC 8285 dependency-descriptor extension elements
// and fixed RTP headers; SRTP, pacing, retransmission, and network transport
// remain caller-owned.
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
		start := len(out)
		packetSize, err := encoderWebRTCRTPPacketHeaderSize(profile, config, i, len(payload), len(descriptor))
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
		descriptorOffset := start + extPayloadStart + encoderWebRTCRTPPacketHeaderExtensionElementPrefixLen(profile)
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
			DependencyDescriptorLength: len(descriptor),
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
	if config.DependencyDescriptorExtensionID == 0 {
		return ErrRTPInvalidHeaderExtension
	}
	var ids encoderWebRTCRTPPacketHeaderExtensionIDs
	if err := ids.add(config.DependencyDescriptorExtensionID); err != nil {
		return err
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

func encoderWebRTCRTPPacketHeaderExtensionPayloadSize(kind int, config EncoderWebRTCRTPPacketHeaderConfig, packetIndex int, descriptorLen int) (int, error) {
	if packetIndex < 0 || descriptorLen <= 0 {
		return 0, ErrEncoderInvalidFrame
	}
	size, err := rtpHeaderExtensionElementSize(kind, config.DependencyDescriptorExtensionID, descriptorLen)
	if err != nil {
		return 0, err
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
	return size, nil
}

type encoderWebRTCRTPPacketHeaderExtensionPayloads struct {
	MID                 [RTPHeaderExtensionSDESMaxLen]byte
	RTPStreamID         [RTPHeaderExtensionSDESMaxLen]byte
	RepairedRTPStreamID [RTPHeaderExtensionSDESMaxLen]byte
	TransportWideCC     [RTPTransportWideCCHeaderExtensionSize]byte
	TransportWideCC02   [RTPTransportWideCC02HeaderExtensionSize]byte
}

func encoderWebRTCRTPPacketHeaderExtensionElements(
	config EncoderWebRTCRTPPacketHeaderConfig,
	packetIndex int,
	descriptor []byte,
	payloads *encoderWebRTCRTPPacketHeaderExtensionPayloads,
) ([6]RTPHeaderExtensionElement, int, error) {
	var elements [6]RTPHeaderExtensionElement
	if packetIndex < 0 || len(descriptor) == 0 || payloads == nil {
		return elements, 0, ErrEncoderInvalidFrame
	}
	count := 0
	elements[count] = RTPHeaderExtensionElement{
		ID:      config.DependencyDescriptorExtensionID,
		Payload: descriptor,
	}
	count++

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
