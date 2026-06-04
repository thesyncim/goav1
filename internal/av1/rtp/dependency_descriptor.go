// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package rtp

const DependencyDescriptorMandatorySize = 3

// DependencyDescriptorMandatory is the fixed 24-bit prefix of WebRTC's RTP
// dependency descriptor header extension.
type DependencyDescriptorMandatory struct {
	FirstPacketInFrame bool
	LastPacketInFrame  bool
	TemplateID         uint8
	FrameNumber        uint16
}

func PutDependencyDescriptorMandatory(dst []byte, descriptor DependencyDescriptorMandatory) (int, error) {
	if len(dst) < DependencyDescriptorMandatorySize {
		return 0, ErrShortBuffer
	}
	if descriptor.TemplateID >= 64 {
		return 0, ErrInvalidDependencyDescriptor
	}
	var b byte
	if descriptor.FirstPacketInFrame {
		b |= 0x80
	}
	if descriptor.LastPacketInFrame {
		b |= 0x40
	}
	b |= descriptor.TemplateID & 0x3f
	dst[0] = b
	dst[1] = byte(descriptor.FrameNumber >> 8)
	dst[2] = byte(descriptor.FrameNumber)
	return DependencyDescriptorMandatorySize, nil
}

func ParseDependencyDescriptorMandatory(src []byte) (DependencyDescriptorMandatory, int, error) {
	if len(src) < DependencyDescriptorMandatorySize {
		return DependencyDescriptorMandatory{}, 0, ErrShortPayload
	}
	return DependencyDescriptorMandatory{
		FirstPacketInFrame: src[0]&0x80 != 0,
		LastPacketInFrame:  src[0]&0x40 != 0,
		TemplateID:         src[0] & 0x3f,
		FrameNumber:        uint16(src[1])<<8 | uint16(src[2]),
	}, DependencyDescriptorMandatorySize, nil
}
