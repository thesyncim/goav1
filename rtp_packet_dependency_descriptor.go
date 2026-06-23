package goav1

// RTPPacketDependencyDescriptor is the result of parsing a complete RTP packet
// and its negotiated WebRTC dependency descriptor header extension.
type RTPPacketDependencyDescriptor struct {
	Packet RTPPacket

	Descriptor        RTPDependencyDescriptor
	DescriptorPayload []byte
}

// ParseRTPPacketDependencyDescriptor parses a complete RTP packet and extracts
// the dependency descriptor carried by dependencyDescriptorExtensionID. state is
// updated exactly like RTPDependencyDescriptorState.Parse; pass the same state
// across packets for compact descriptors that refer to a prior attached
// structure. state may be nil only for descriptors that carry enough structure
// to parse independently.
func ParseRTPPacketDependencyDescriptor(packet []byte, dependencyDescriptorExtensionID uint8, state *RTPDependencyDescriptorState) (RTPPacketDependencyDescriptor, error) {
	parsedPacket, err := ParseRTPPacket(packet)
	if err != nil {
		return RTPPacketDependencyDescriptor{}, err
	}
	if parsedPacket.Header.ExtensionProfile == 0 && len(parsedPacket.Header.ExtensionPayload) == 0 {
		return RTPPacketDependencyDescriptor{}, ErrRTPHeaderExtensionNotFound
	}
	element, ok, err := FindRTPHeaderExtensionElement(parsedPacket.Header.ExtensionProfile, parsedPacket.Header.ExtensionPayload, dependencyDescriptorExtensionID)
	if err != nil {
		return RTPPacketDependencyDescriptor{}, err
	}
	if !ok {
		return RTPPacketDependencyDescriptor{}, ErrRTPHeaderExtensionNotFound
	}
	descriptor, consumed, err := state.Parse(element.Payload)
	if err != nil {
		return RTPPacketDependencyDescriptor{}, err
	}
	if consumed != len(element.Payload) {
		return RTPPacketDependencyDescriptor{}, ErrRTPInvalidDependencyDescriptor
	}
	return RTPPacketDependencyDescriptor{
		Packet:            parsedPacket,
		Descriptor:        descriptor,
		DescriptorPayload: element.Payload,
	}, nil
}
