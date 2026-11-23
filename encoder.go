package goav1

import (
	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
)

const (
	EncoderWebRTCMaxSpatialLayers = internalencoder.WebRTCMaxSpatialLayers

	EncoderWebRTCMinEffortLevel = internalencoder.WebRTCMinEffortLevel
	EncoderWebRTCMaxEffortLevel = internalencoder.WebRTCMaxEffortLevel
)

const (
	EncoderProfile0 = internalencoder.Profile0
)

type EncoderScalabilityMode = internalencoder.ScalabilityMode

const (
	EncoderScalabilityModeL1T1 = internalencoder.ScalabilityModeL1T1
	EncoderScalabilityModeL1T2 = internalencoder.ScalabilityModeL1T2
	EncoderScalabilityModeL1T3 = internalencoder.ScalabilityModeL1T3

	EncoderScalabilityModeL2T3_KEY_SHIFT = internalencoder.ScalabilityModeL2T3_KEY_SHIFT

	EncoderScalabilityModeL3T2_KEY_SHIFT = internalencoder.ScalabilityModeL3T2_KEY_SHIFT

	EncoderScalabilityModeL3T3_KEY_SHIFT = internalencoder.ScalabilityModeL3T3_KEY_SHIFT
)

type EncoderRational = internalencoder.Rational
type EncoderResolution = internalencoder.Resolution
type EncoderRateControlMode = internalencoder.RateControlMode
type EncoderContentHint = internalencoder.ContentHint

type EncoderConfig = internalencoder.Config

type EncoderWebRTCRTPPacketDependencyDescriptorOptions struct {
	AttachStructureOnFirstPacket            bool
	ActiveDecodeTargetsPresentOnFirstPacket bool
	ActiveDecodeTargetsMask                 uint32
}

type EncoderWebRTCRTPPacketSpan struct {
	PayloadOffset    int
	PayloadLength    int
	DescriptorOffset int
	DescriptorLength int
	Marker           bool
}

const (
	EncoderRateControlCBR = internalencoder.RateControlCBR
	EncoderRateControlCQP = internalencoder.RateControlCQP

	EncoderContentAuto   = internalencoder.ContentAuto
	EncoderContentCamera = internalencoder.ContentCamera
	EncoderContentScreen = internalencoder.ContentScreen
)

var (
	ErrEncoderInvalidFrame = internalencoder.ErrInvalidFrame
)

// EncoderWebRTCScalabilityModes returns the pinned-libwebrtc WebRTC SVC
// scalabilityMode values supported by the WebRTC encoder/control surfaces.
func EncoderWebRTCScalabilityModes() []EncoderScalabilityMode {
	out := make([]EncoderScalabilityMode, 0, internalencoder.WebRTCScalabilityModeCount())
	return internalencoder.AppendWebRTCScalabilityModes(out)
}

func SetWebRTCEncoderSVCConfig(config EncoderConfig, requestedTemporalLayers uint8, requestedSpatialLayers uint8) (EncoderConfig, error) {
	return internalencoder.SetWebRTCSVCConfig(config, requestedTemporalLayers, requestedSpatialLayers)
}

// EncoderWebRTCRTPFrameDuration returns the exact RTP timestamp duration of
// one encoded WebRTC picture after applying the same config normalization used
// by the WebRTC encoder. The returned rational is expressed in RTP timestamp
// ticks, not seconds. For the default 90 kHz RTP timebase, 30 fps returns
// 3000/1 and 30000/1001 fps returns 3003/1.
func EncoderWebRTCRTPFrameDuration(config EncoderConfig) (EncoderRational, error) {
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return EncoderRational{}, err
	}
	num := int64(normalized.MaxFramerate.Den) * int64(normalized.RTPTimebase.Den)
	den := int64(normalized.MaxFramerate.Num) * int64(normalized.RTPTimebase.Num)
	div := encoderWebRTCRTPDurationGCD(num, den)
	num /= div
	den /= div
	if num > 1<<31-1 || den > 1<<31-1 {
		return EncoderRational{}, internalencoder.ErrInvalidConfig
	}
	return EncoderRational{Num: int32(num), Den: int32(den)}, nil
}

func encoderWebRTCRTPDurationGCD(a int64, b int64) int64 {
	for b != 0 {
		a, b = b, a%b
	}
	if a < 0 {
		return -a
	}
	return a
}

func appendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(dst []byte, packetizer *RTPPacketizer, structure internalencoder.WebRTCFrameDependencyStructure, info internalencoder.WebRTCGenericFrameInfo, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) ([]byte, bool, error) {
	flags, ok := NextRTPPacketDependencyDescriptorFlags(packetizer)
	if !ok {
		return dst, false, nil
	}
	out, err := internalencoder.AppendWebRTCDependencyDescriptorWithOptions(dst, structure, info, encoderWebRTCRTPPacketDependencyDescriptorOptions(flags, options))
	if err != nil {
		return dst, true, err
	}
	return out, true, nil
}

func appendEncoderWebRTCFrameControlRTPPacketDependencyDescriptorWithOptions(dst []byte, packetizer *RTPPacketizer, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) ([]byte, bool, error) {
	if control.AttachDependencyStructure {
		options.AttachStructureOnFirstPacket = true
	}
	return appendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(dst, packetizer, structure, control.GenericFrameInfo, options)
}

func appendEncoderWebRTCFrameControlRTPPacketWithOptions(payloadDst []byte, descriptorDst []byte, packetizer *RTPPacketizer, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (payload []byte, descriptor []byte, marker bool, ok bool, err error) {
	descriptor, ok, err = appendEncoderWebRTCFrameControlRTPPacketDependencyDescriptorWithOptions(descriptorDst, packetizer, control, structure, options)
	if err != nil || !ok {
		return payloadDst, descriptorDst, false, ok, err
	}
	payloadSize, sizeOK := packetizer.NextPacketSize()
	if !sizeOK {
		return payloadDst, descriptorDst, false, false, nil
	}
	if cap(payloadDst)-len(payloadDst) < payloadSize {
		return payloadDst, descriptorDst, false, true, ErrRTPShortBuffer
	}
	off := len(payloadDst)
	out := payloadDst[:off+payloadSize]
	n, marker, ok, err := packetizer.NextPacket(out[off:])
	if err != nil {
		return payloadDst, descriptorDst, false, true, err
	}
	if !ok {
		return payloadDst, descriptorDst, false, false, nil
	}
	return out[:off+n], descriptor, marker, true, nil
}

func encoderWebRTCRTPPacketDependencyDescriptorOptions(flags RTPPacketDependencyDescriptorFlags, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) internalencoder.WebRTCDependencyDescriptorOptions {
	return internalencoder.WebRTCDependencyDescriptorOptions{
		FirstPacketInFrame:         flags.FirstPacketInFrame,
		LastPacketInFrame:          flags.LastPacketInFrame,
		AttachStructure:            options.AttachStructureOnFirstPacket && flags.FirstPacketInFrame,
		ActiveDecodeTargetsPresent: options.ActiveDecodeTargetsPresentOnFirstPacket && flags.FirstPacketInFrame,
		ActiveDecodeTargetsMask:    options.ActiveDecodeTargetsMask,
	}
}

func encoderPlaneFits(p FramePlane, bytesPerSample int) bool {
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return false
	}
	if p.Width < 0 || p.Height < 0 || p.Stride < 0 {
		return false
	}
	if p.Width == 0 || p.Height == 0 {
		return len(p.Pix) == 0
	}
	maxInt := int(^uint(0) >> 1)
	if p.Width > maxInt/bytesPerSample {
		return false
	}
	rowBytes := p.Width * bytesPerSample
	if p.Stride < rowBytes {
		return false
	}
	if p.Height-1 > maxInt/p.Stride {
		return false
	}
	lastRow := (p.Height - 1) * p.Stride
	if lastRow > maxInt-rowBytes {
		return false
	}
	need := lastRow + rowBytes
	return need >= rowBytes && len(p.Pix) >= need
}
