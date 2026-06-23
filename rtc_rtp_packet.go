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

// AppendEncoderWebRTCRTPPacketsWithHeaders wraps AV1 RTP payload bodies and
// dependency descriptors from RTCFrame.AppendRTPPackets into complete RTP
// packets. The function writes RFC 8285 dependency-descriptor extension elements
// and fixed RTP headers; SRTP, pacing, retransmission, and network transport
// remain caller-owned.
func AppendEncoderWebRTCRTPPacketsWithHeaders(dst []byte, headerSpans []EncoderWebRTCRTPPacketHeaderSpan, config EncoderWebRTCRTPPacketHeaderConfig, rtpPayloads []byte, descriptors []byte, packetSpans []EncoderWebRTCRTPPacketSpan) ([]byte, int, error) {
	if len(headerSpans) < len(packetSpans) {
		return dst, 0, ErrRTPPacketPlanTooSmall
	}
	if err := validateEncoderWebRTCRTPPacketHeaderConfig(config); err != nil {
		return dst, 0, err
	}
	profile := encoderWebRTCRTPPacketHeaderExtensionProfile(config.HeaderExtensionProfile)

	for i := range packetSpans {
		payload, descriptor, err := encoderWebRTCRTPPacketSpanSlices(rtpPayloads, descriptors, packetSpans[i])
		if err != nil {
			return dst, 0, err
		}
		if len(descriptor) == 0 {
			return dst, 0, ErrEncoderInvalidFrame
		}
		extLen, err := RTPHeaderExtensionElementsSize(profile, []RTPHeaderExtensionElement{{
			ID:      config.DependencyDescriptorExtensionID,
			Payload: descriptor,
		}})
		if err != nil {
			return dst, 0, err
		}
		if _, err := rtpPaddedExtensionPayloadLen(extLen); err != nil {
			return dst, 0, err
		}
		if _, err := encoderWebRTCRTPPacketHeaderSize(profile, config, len(payload), len(descriptor)); err != nil {
			return dst, 0, err
		}
	}

	out := dst
	for i := range packetSpans {
		span := packetSpans[i]
		payload, descriptor, _ := encoderWebRTCRTPPacketSpanSlices(rtpPayloads, descriptors, span)
		start := len(out)
		packetSize, _ := encoderWebRTCRTPPacketHeaderSize(profile, config, len(payload), len(descriptor))
		out = append(out, make([]byte, packetSize)...)
		packet := out[start : start+packetSize]

		extHeaderLen, _ := RTPHeaderExtensionElementsSize(profile, []RTPHeaderExtensionElement{{
			ID:      config.DependencyDescriptorExtensionID,
			Payload: descriptor,
		}})
		extPayloadStart := RTPHeaderMinSize + 4
		n, err := PutRTPHeaderExtensionElements(packet[extPayloadStart:extPayloadStart+extHeaderLen], profile, []RTPHeaderExtensionElement{{
			ID:      config.DependencyDescriptorExtensionID,
			Payload: descriptor,
		}})
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

func validateEncoderWebRTCRTPPacketHeaderConfig(config EncoderWebRTCRTPPacketHeaderConfig) error {
	if config.PayloadType > 127 {
		return ErrRTPInvalidHeader
	}
	if config.DependencyDescriptorExtensionID == 0 {
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

func encoderWebRTCRTPPacketHeaderSize(profile uint16, config EncoderWebRTCRTPPacketHeaderConfig, payloadLen int, descriptorLen int) (int, error) {
	kind, err := rtpHeaderExtensionProfileKind(profile)
	if err != nil {
		return 0, err
	}
	extLen, err := rtpHeaderExtensionElementSize(kind, config.DependencyDescriptorExtensionID, descriptorLen)
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
