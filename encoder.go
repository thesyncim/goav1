package goav1

import (
	internalbitstream "github.com/thesyncim/goav1/internal/av1/bitstream"
	internalencoder "github.com/thesyncim/goav1/internal/av1/encoder"
	internalframe "github.com/thesyncim/goav1/internal/av1/frame"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
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

type webRTCEncoder struct {
	config EncoderConfig
	state  internalencoder.WebRTCEncoderState
}

type EncoderWebRTCRTPPacketDependencyDescriptorOptions struct {
	AttachStructureOnFirstPacket            bool
	ActiveDecodeTargetsPresentOnFirstPacket bool
	ActiveDecodeTargetsMask                 uint32
}

// encoderWebRTCVideoLayersAllocationConfig controls how encoder configuration
// is converted into WebRTC's video-layers-allocation RTP header extension.
//
// The zero value describes one RTP stream, all configured spatial layers active,
// and resolution/framerate omitted. For multi-temporal modes, callers must set
// TargetBitratesKbps[spatial][temporal] to the cumulative bitrate ladder in
// kbps; goav1 does not invent a temporal bitrate split from a spatial target.
type encoderWebRTCVideoLayersAllocationConfig struct {
	RTPStreamID    int
	RTPStreamCount int

	// SpatialLayerRTPStreamIDs maps each configured spatial layer to an RTP
	// stream ID. The zero value maps every layer to stream 0.
	SpatialLayerRTPStreamIDs [EncoderWebRTCMaxSpatialLayers]int

	// ActiveSpatialLayerMaskSet makes ActiveSpatialLayerMask authoritative. When
	// false, all configured spatial layers are active. When true, bit N selects
	// spatial layer N; a zero mask writes the paused one-byte allocation form.
	ActiveSpatialLayerMaskSet bool
	ActiveSpatialLayerMask    uint8

	// TargetBitratesKbps holds cumulative target bitrates in kbps by spatial
	// layer then temporal layer. Single-temporal layers may omit this and use the
	// normalized encoder spatial-layer target bitrate.
	TargetBitratesKbps [EncoderWebRTCMaxSpatialLayers][internalencoder.WebRTCMaxTemporalLayers]uint32

	IncludeResolutionAndFramerate bool
}

// encoderWebRTCVideoLayersAllocationScratch owns the backing storage used by
// encoderWebRTCVideoLayersAllocationInto. The returned RTPVideoLayersAllocation
// aliases this scratch until the next call that reuses it.
type encoderWebRTCVideoLayersAllocationScratch struct {
	ActiveSpatialLayers [EncoderWebRTCMaxSpatialLayers]RTPVideoLayersAllocationLayer
	TargetBitratesKbps  [EncoderWebRTCMaxSpatialLayers][internalencoder.WebRTCMaxTemporalLayers]uint32
}

type encoderFrameSamplePlanes struct {
	Y internalframe.SamplePlane
	U internalframe.SamplePlane
	V internalframe.SamplePlane
}

type encoderWebRTCPictureSampleScratchSize struct {
	Samples  int
	FrameNum int
}

type encoderWebRTCPictureSampleScratch struct {
	Samples []uint16
}

type encoderWebRTCPictureSamplePlanes struct {
	Frames   [EncoderWebRTCMaxSpatialLayers]encoderFrameSamplePlanes
	FrameNum uint8
}

type encoderWebRTCPictureTemporalUnitRTPScratchSize struct {
	Packetizer         RTPPacketizerScratchSize
	MaxPayloadBytes    int
	MaxDescriptorBytes int
}

type encoderWebRTCPictureTemporalUnitFramesRTPScratchSize struct {
	FrameOBUBytes      int
	FrameSpans         int
	PacketSpans        int
	Packetizer         RTPPacketizerScratchSize
	MaxPayloadBytes    int
	MaxDescriptorBytes int
}

type encoderWebRTCPictureTemporalUnitFramesRTPScratch struct {
	FrameOBU    []byte
	FrameSpans  []encoderWebRTCFrameRTPPacketSpan
	PacketSpans []EncoderWebRTCRTPPacketSpan
	OBUs        []RTPPacketizerOBU
	Packets     []RTPPacketPlan
	Work        []RTPPacketPlan
}

type encoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo struct {
	PacketCount     int
	PayloadBytes    int
	DescriptorBytes int
}

type encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo struct {
	FrameOBUBytes int
	RTP           encoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo
}

type encoderWebRTCFrameOBUSpan struct {
	Offset int
	Length int
}

type encoderWebRTCFrameRTPPacketSpan struct {
	FrameOBUOffset int
	FrameOBULength int
	PacketOffset   int
	PacketCount    int
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

// encoderScalabilityModes returns every explicit encoder scalabilityMode value
// accepted by encoder config, including implemented key-shift variants that
// are not in the pinned WebRTC catalogue. Use EncoderWebRTCScalabilityModes
// when validating browser setParameters()/libwebrtc catalogue values.
func encoderScalabilityModes() []EncoderScalabilityMode {
	out := make([]EncoderScalabilityMode, 0, internalencoder.ScalabilityModeCount())
	return internalencoder.AppendScalabilityModes(out)
}

// EncoderWebRTCScalabilityModes returns the pinned-libwebrtc WebRTC SVC
// scalabilityMode values supported by the WebRTC encoder/control surfaces.
func EncoderWebRTCScalabilityModes() []EncoderScalabilityMode {
	out := make([]EncoderScalabilityMode, 0, internalencoder.WebRTCScalabilityModeCount())
	return internalencoder.AppendWebRTCScalabilityModes(out)
}

// encoderWebRTCSVCScalabilityModes returns every W3C WebRTC-SVC scalabilityMode
// value implemented by the encoder/control surfaces, including W3C KEY_SHIFT
// variants that are not present in the pinned libwebrtc catalogue.
func encoderWebRTCSVCScalabilityModes() []EncoderScalabilityMode {
	out := make([]EncoderScalabilityMode, 0, internalencoder.WebRTCSVCScalabilityModeCount())
	return internalencoder.AppendWebRTCSVCScalabilityModes(out)
}

// validateEncoderWebRTCActiveScalabilityModes validates the active
// scalabilityMode values for one WebRTC sender. Pass only active encodings; a
// sender with no active encodings is accepted. Per the WebRTC-SVC setParameters
// rule, multiple active encodings cannot include a single-SSRC simulcast S mode
// such as S2T1 or S3T3h.
func validateEncoderWebRTCActiveScalabilityModes(modes ...EncoderScalabilityMode) error {
	return internalencoder.ValidateWebRTCActiveScalabilityModes(modes)
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

// encoderWebRTCVideoLayersAllocation builds WebRTC's video-layers-allocation
// RTP header-extension payload model from an encoder config and explicit
// allocation controls. The result is ready for RTPVideoLayersAllocationSize,
// PutRTPVideoLayersAllocationHeaderExtension, or
// EncoderWebRTCRTPPacketHeaderConfig.VideoLayersAllocation.
func encoderWebRTCVideoLayersAllocation(config EncoderConfig, allocationConfig encoderWebRTCVideoLayersAllocationConfig) (RTPVideoLayersAllocation, error) {
	var scratch encoderWebRTCVideoLayersAllocationScratch
	allocation, err := encoderWebRTCVideoLayersAllocationInto(config, allocationConfig, &scratch)
	if err != nil {
		return RTPVideoLayersAllocation{}, err
	}
	return cloneRTPVideoLayersAllocation(allocation), nil
}

// encoderWebRTCVideoLayersAllocationInto is encoderWebRTCVideoLayersAllocation
// with caller-owned scratch for production control loops that refresh layer
// allocation metadata when bitrate, framerate, active layers, or scalability
// settings change.
func encoderWebRTCVideoLayersAllocationInto(config EncoderConfig, allocationConfig encoderWebRTCVideoLayersAllocationConfig, scratch *encoderWebRTCVideoLayersAllocationScratch) (RTPVideoLayersAllocation, error) {
	if scratch == nil {
		return RTPVideoLayersAllocation{}, internalencoder.ErrInvalidConfig
	}
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return RTPVideoLayersAllocation{}, err
	}
	spatialLayers := int(normalized.SpatialLayerCount)
	temporalLayers := int(normalized.TemporalLayerCount)
	if spatialLayers <= 0 || spatialLayers > EncoderWebRTCMaxSpatialLayers ||
		temporalLayers <= 0 || temporalLayers > internalencoder.WebRTCMaxTemporalLayers {
		return RTPVideoLayersAllocation{}, internalencoder.ErrInvalidConfig
	}

	rtpStreamCount := allocationConfig.RTPStreamCount
	if rtpStreamCount == 0 {
		rtpStreamCount = 1
	}
	if rtpStreamCount < 1 || rtpStreamCount > 4 ||
		allocationConfig.RTPStreamID < 0 || allocationConfig.RTPStreamID >= rtpStreamCount {
		return RTPVideoLayersAllocation{}, internalencoder.ErrInvalidConfig
	}

	allSpatialMask := uint8((1 << uint(spatialLayers)) - 1)
	activeMask := allSpatialMask
	if allocationConfig.ActiveSpatialLayerMaskSet {
		activeMask = allocationConfig.ActiveSpatialLayerMask
		if activeMask&^allSpatialMask != 0 {
			return RTPVideoLayersAllocation{}, internalencoder.ErrInvalidConfig
		}
	}
	if activeMask == 0 {
		if allocationConfig.RTPStreamID != 0 || rtpStreamCount != 1 {
			return RTPVideoLayersAllocation{}, internalencoder.ErrInvalidConfig
		}
		return RTPVideoLayersAllocation{RTPStreamCount: 1}, nil
	}

	framerate := 0
	if allocationConfig.IncludeResolutionAndFramerate {
		framerate, err = encoderWebRTCVideoLayersAllocationFramerate(normalized.MaxFramerate)
		if err != nil {
			return RTPVideoLayersAllocation{}, err
		}
	}

	allocation := RTPVideoLayersAllocation{
		RTPStreamID:               allocationConfig.RTPStreamID,
		RTPStreamCount:            rtpStreamCount,
		ActiveSpatialLayers:       scratch.ActiveSpatialLayers[:0],
		HasResolutionAndFramerate: allocationConfig.IncludeResolutionAndFramerate,
	}
	for streamID := 0; streamID < rtpStreamCount; streamID++ {
		for spatialID := 0; spatialID < spatialLayers; spatialID++ {
			if activeMask&(1<<uint(spatialID)) == 0 {
				continue
			}
			layerStreamID := allocationConfig.SpatialLayerRTPStreamIDs[spatialID]
			if layerStreamID < 0 || layerStreamID >= rtpStreamCount {
				return RTPVideoLayersAllocation{}, internalencoder.ErrInvalidConfig
			}
			if layerStreamID != streamID {
				continue
			}
			layer, err := encoderWebRTCVideoLayersAllocationLayer(normalized, allocationConfig, spatialID, temporalLayers, layerStreamID, framerate, scratch.TargetBitratesKbps[spatialID][:])
			if err != nil {
				return RTPVideoLayersAllocation{}, err
			}
			allocation.ActiveSpatialLayers = append(allocation.ActiveSpatialLayers, layer)
		}
	}
	if len(allocation.ActiveSpatialLayers) == 0 {
		return RTPVideoLayersAllocation{}, internalencoder.ErrInvalidConfig
	}
	if err := ValidateRTPVideoLayersAllocation(allocation); err != nil {
		return RTPVideoLayersAllocation{}, err
	}
	return allocation, nil
}

func cloneRTPVideoLayersAllocation(allocation RTPVideoLayersAllocation) RTPVideoLayersAllocation {
	out := allocation
	if len(allocation.ActiveSpatialLayers) == 0 {
		return out
	}
	out.ActiveSpatialLayers = make([]RTPVideoLayersAllocationLayer, len(allocation.ActiveSpatialLayers))
	copy(out.ActiveSpatialLayers, allocation.ActiveSpatialLayers)
	for i := range out.ActiveSpatialLayers {
		targets := allocation.ActiveSpatialLayers[i].TargetBitratesKbps
		if len(targets) != 0 {
			out.ActiveSpatialLayers[i].TargetBitratesKbps = append([]uint32(nil), targets...)
		}
	}
	return out
}

func encoderWebRTCVideoLayersAllocationLayer(config EncoderConfig, allocationConfig encoderWebRTCVideoLayersAllocationConfig, spatialID int, temporalLayers int, rtpStreamID int, framerate int, targetScratch []uint32) (RTPVideoLayersAllocationLayer, error) {
	targets, err := encoderWebRTCVideoLayersAllocationTargets(config, allocationConfig, spatialID, temporalLayers, targetScratch)
	if err != nil {
		return RTPVideoLayersAllocationLayer{}, err
	}
	layerConfig := config.SpatialLayers[spatialID]
	layer := RTPVideoLayersAllocationLayer{
		RTPStreamID:        rtpStreamID,
		SpatialID:          spatialID,
		TargetBitratesKbps: targets,
	}
	if allocationConfig.IncludeResolutionAndFramerate {
		layer.Width = int(layerConfig.Resolution.Width)
		layer.Height = int(layerConfig.Resolution.Height)
		layer.Framerate = framerate
	}
	return layer, nil
}

func encoderWebRTCVideoLayersAllocationTargets(config EncoderConfig, allocationConfig encoderWebRTCVideoLayersAllocationConfig, spatialID int, temporalLayers int, targetScratch []uint32) ([]uint32, error) {
	if len(targetScratch) < temporalLayers {
		return nil, internalencoder.ErrInvalidConfig
	}
	targets := targetScratch[:temporalLayers]
	var previous uint32
	for temporalID := 0; temporalID < temporalLayers; temporalID++ {
		target := allocationConfig.TargetBitratesKbps[spatialID][temporalID]
		if target == 0 && temporalLayers == 1 {
			layerTarget := config.SpatialLayers[spatialID].TargetBitrateKbps
			if layerTarget <= 0 {
				layerTarget = config.TargetBitrateKbps
			}
			if layerTarget > 0 {
				target = uint32(layerTarget)
			}
		}
		if target == 0 ||
			target > uint32(internalencoder.WebRTCMaxBitrateKbps) ||
			(temporalID > 0 && target < previous) {
			return nil, internalencoder.ErrInvalidConfig
		}
		targets[temporalID] = target
		previous = target
	}
	return targets, nil
}

func encoderWebRTCVideoLayersAllocationFramerate(rate EncoderRational) (int, error) {
	if !rate.Valid() {
		return 0, internalencoder.ErrInvalidConfig
	}
	fps := int((int64(rate.Num) + int64(rate.Den)/2) / int64(rate.Den))
	if fps < 1 || fps > 0xff {
		return 0, internalencoder.ErrInvalidConfig
	}
	return fps, nil
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

func encoderWebRTCRTPDependencyDescriptorMandatoryForFrame(structure internalencoder.WebRTCFrameDependencyStructure, info internalencoder.WebRTCGenericFrameInfo, firstPacketInFrame bool, lastPacketInFrame bool) (RTPDependencyDescriptorMandatory, error) {
	templateID, err := internalencoder.WebRTCTemplateIDForFrame(structure, info)
	if err != nil {
		return RTPDependencyDescriptorMandatory{}, err
	}
	return RTPDependencyDescriptorMandatory{
		FirstPacketInFrame: firstPacketInFrame,
		LastPacketInFrame:  lastPacketInFrame,
		TemplateID:         templateID,
		FrameNumber:        uint16(info.FrameID),
	}, nil
}

func encoderWebRTCRTPPacketDependencyDescriptorSizeWithOptions(packetizer *RTPPacketizer, structure internalencoder.WebRTCFrameDependencyStructure, info internalencoder.WebRTCGenericFrameInfo, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (int, bool, error) {
	flags, ok := NextRTPPacketDependencyDescriptorFlags(packetizer)
	if !ok {
		return 0, false, nil
	}
	size, err := internalencoder.WebRTCDependencyDescriptorSizeWithOptions(structure, info, encoderWebRTCRTPPacketDependencyDescriptorOptions(flags, options))
	if err != nil {
		return 0, true, err
	}
	return size, true, nil
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

func encoderWebRTCFrameControlRTPPacketDependencyDescriptorSizeWithOptions(packetizer *RTPPacketizer, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (int, bool, error) {
	if control.AttachDependencyStructure {
		options.AttachStructureOnFirstPacket = true
	}
	return encoderWebRTCRTPPacketDependencyDescriptorSizeWithOptions(packetizer, structure, control.GenericFrameInfo, options)
}

func appendEncoderWebRTCFrameControlRTPPacketDependencyDescriptorWithOptions(dst []byte, packetizer *RTPPacketizer, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) ([]byte, bool, error) {
	if control.AttachDependencyStructure {
		options.AttachStructureOnFirstPacket = true
	}
	return appendEncoderWebRTCRTPPacketDependencyDescriptorWithOptions(dst, packetizer, structure, control.GenericFrameInfo, options)
}

func encoderWebRTCFrameControlRTPPacketSize(packetizer *RTPPacketizer, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure) (payloadSize int, descriptorSize int, ok bool, err error) {
	return encoderWebRTCFrameControlRTPPacketSizeWithOptions(packetizer, control, structure, EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket: control.AttachDependencyStructure,
	})
}

func encoderWebRTCFrameControlRTPPacketSizeWithOptions(packetizer *RTPPacketizer, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure, options EncoderWebRTCRTPPacketDependencyDescriptorOptions) (payloadSize int, descriptorSize int, ok bool, err error) {
	payloadSize, ok = packetizer.NextPacketSize()
	if !ok {
		return 0, 0, false, nil
	}
	descriptorSize, ok, err = encoderWebRTCFrameControlRTPPacketDependencyDescriptorSizeWithOptions(packetizer, control, structure, options)
	if err != nil {
		return 0, 0, true, err
	}
	if !ok {
		return 0, 0, false, nil
	}
	return payloadSize, descriptorSize, true, nil
}

func appendEncoderWebRTCFrameControlRTPPacket(payloadDst []byte, descriptorDst []byte, packetizer *RTPPacketizer, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure) (payload []byte, descriptor []byte, marker bool, ok bool, err error) {
	return appendEncoderWebRTCFrameControlRTPPacketWithOptions(payloadDst, descriptorDst, packetizer, control, structure, EncoderWebRTCRTPPacketDependencyDescriptorOptions{
		AttachStructureOnFirstPacket: control.AttachDependencyStructure,
	})
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

func encoderWebRTCPictureTemporalUnitFrameNum(unit internalencoder.WebRTCPictureTemporalUnit) uint8 {
	if unit.Key {
		return unit.KeyUnit.FrameNum
	}
	if unit.Delta {
		return unit.DeltaUnit.FrameNum
	}
	return 0
}

func encoderWebRTCPictureTemporalUnitFrameControl(unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8) (internalencoder.WebRTCFrameControl, internalencoder.WebRTCFrameDependencyStructure, error) {
	if unit.Key == unit.Delta {
		return internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	if unit.Key {
		if frameIndex >= unit.KeyUnit.Control.FrameNum {
			return internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
		}
		return unit.KeyUnit.Control.Frames[frameIndex], unit.KeyUnit.Control.DependencyStructure, nil
	}
	if frameIndex >= unit.DeltaUnit.Control.FrameNum || !state.DependencyStructureState.Valid {
		return internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	return unit.DeltaUnit.Control.Frames[frameIndex], state.DependencyStructureState.Structure, nil
}

func encoderWebRTCPictureTemporalUnitDependencyDescriptorSize(unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, attachStructure bool) (int, error) {
	control, structure, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return 0, err
	}
	return internalencoder.WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, attachStructure)
}

func encoderWebRTCPictureTemporalUnitMaxDependencyDescriptorSize(unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8) (int, error) {
	control, structure, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return 0, err
	}
	return internalencoder.WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, control.AttachDependencyStructure)
}

func appendEncoderWebRTCPictureTemporalUnitDependencyDescriptor(dst []byte, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, firstPacketInFrame bool, lastPacketInFrame bool, attachStructure bool) ([]byte, error) {
	control, structure, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return dst, err
	}
	return internalencoder.AppendWebRTCDependencyDescriptor(dst, structure, control.GenericFrameInfo, firstPacketInFrame, lastPacketInFrame, attachStructure)
}

func encoderWebRTCPictureTemporalUnitFrameOBUSize(payload []byte, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8) (int, internalencoder.WebRTCFrameControl, internalencoder.WebRTCFrameDependencyStructure, error) {
	control, structure, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return 0, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	size, err := internalencoder.LowOverheadOBUSize(internalencoder.OBU{
		Type:       internalobu.TypeFrame,
		TemporalID: control.Settings.TemporalID,
		SpatialID:  control.Settings.SpatialID,
		Payload:    payload,
	})
	if err != nil {
		return 0, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	return size, control, structure, nil
}

func appendEncoderWebRTCPictureTemporalUnitFrameOBU(dst []byte, payload []byte, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8) ([]byte, internalencoder.WebRTCFrameControl, internalencoder.WebRTCFrameDependencyStructure, error) {
	_, control, structure, err := encoderWebRTCPictureTemporalUnitFrameOBUSize(payload, unit, state, frameIndex)
	if err != nil {
		return dst, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	out, err := internalencoder.AppendLowOverheadOBU(dst, internalencoder.OBU{
		Type:       internalobu.TypeFrame,
		TemporalID: control.Settings.TemporalID,
		SpatialID:  control.Settings.SpatialID,
		Payload:    payload,
	})
	if err != nil {
		return dst, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	return out, control, structure, nil
}

func encoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads [][]byte, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState) (int, error) {
	frameNum := encoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return 0, ErrEncoderInvalidFrame
	}
	size := 0
	for i := uint8(0); i < frameNum; i++ {
		frameSize, _, _, err := encoderWebRTCPictureTemporalUnitFrameOBUSize(framePayloads[i], unit, state, i)
		if err != nil {
			return 0, err
		}
		size += frameSize
	}
	return size, nil
}

func appendEncoderWebRTCPictureTemporalUnitFramesOBU(dst []byte, spans []encoderWebRTCFrameOBUSpan, framePayloads [][]byte, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState) ([]byte, int, error) {
	frameNum := encoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return dst, 0, ErrEncoderInvalidFrame
	}
	if len(spans) < int(frameNum) {
		return dst, 0, internalbitstream.ErrShortBuffer
	}
	out := dst
	for i := uint8(0); i < frameNum; i++ {
		start := len(out)
		next, _, _, err := appendEncoderWebRTCPictureTemporalUnitFrameOBU(out, framePayloads[i], unit, state, i)
		if err != nil {
			return dst, 0, err
		}
		out = next
		spans[i] = encoderWebRTCFrameOBUSpan{
			Offset: start,
			Length: len(out) - start,
		}
	}
	return out, int(frameNum), nil
}

func encoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads [][]byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU) (encoderWebRTCPictureTemporalUnitFramesRTPScratchSize, error) {
	frameNum := encoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return encoderWebRTCPictureTemporalUnitFramesRTPScratchSize{}, ErrEncoderInvalidFrame
	}
	frameOBUBytes, err := encoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads, unit, state)
	size := encoderWebRTCPictureTemporalUnitFramesRTPScratchSize{
		FrameOBUBytes: frameOBUBytes,
		FrameSpans:    int(frameNum),
	}
	if err != nil {
		return size, err
	}
	if cap(frameOBUScratch)-len(frameOBUScratch) < frameOBUBytes {
		return size, internalbitstream.ErrShortBuffer
	}

	frameOBUs := frameOBUScratch[:0]
	for i := uint8(0); i < frameNum; i++ {
		frameStart := len(frameOBUs)
		nextFrameOBUs, _, _, err := appendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUs, framePayloads[i], unit, state, i)
		if err != nil {
			return size, err
		}
		frameOBUs = nextFrameOBUs
		scratch, err := encoderWebRTCPictureTemporalUnitRTPScratchLen(frameOBUs[frameStart:], limits, unit, state, i, obuScratch)
		if err != nil {
			return size, err
		}
		size.PacketSpans += scratch.Packetizer.Packets
		if scratch.Packetizer.OBUs > size.Packetizer.OBUs {
			size.Packetizer.OBUs = scratch.Packetizer.OBUs
		}
		if scratch.Packetizer.Packets > size.Packetizer.Packets {
			size.Packetizer.Packets = scratch.Packetizer.Packets
		}
		if scratch.Packetizer.Work > size.Packetizer.Work {
			size.Packetizer.Work = scratch.Packetizer.Work
		}
		if scratch.MaxPayloadBytes > size.MaxPayloadBytes {
			size.MaxPayloadBytes = scratch.MaxPayloadBytes
		}
		if scratch.MaxDescriptorBytes > size.MaxDescriptorBytes {
			size.MaxDescriptorBytes = scratch.MaxDescriptorBytes
		}
	}
	return size, nil
}

func bindEncoderWebRTCPictureTemporalUnitFramesRTPScratch(size encoderWebRTCPictureTemporalUnitFramesRTPScratchSize, scratch encoderWebRTCPictureTemporalUnitFramesRTPScratch) (encoderWebRTCPictureTemporalUnitFramesRTPScratch, error) {
	if size.FrameOBUBytes < 0 || size.FrameSpans < 0 || size.PacketSpans < 0 ||
		size.Packetizer.OBUs < 0 || size.Packetizer.Packets < 0 || size.Packetizer.Work < 0 {
		return encoderWebRTCPictureTemporalUnitFramesRTPScratch{}, internalbitstream.ErrShortBuffer
	}
	if cap(scratch.FrameOBU) < size.FrameOBUBytes || len(scratch.FrameSpans) < size.FrameSpans ||
		len(scratch.PacketSpans) < size.PacketSpans || len(scratch.OBUs) < size.Packetizer.OBUs ||
		len(scratch.Packets) < size.Packetizer.Packets || len(scratch.Work) < size.Packetizer.Work {
		return encoderWebRTCPictureTemporalUnitFramesRTPScratch{}, internalbitstream.ErrShortBuffer
	}
	scratch.FrameOBU = scratch.FrameOBU[:0]
	scratch.FrameSpans = scratch.FrameSpans[:size.FrameSpans]
	scratch.PacketSpans = scratch.PacketSpans[:size.PacketSpans]
	scratch.OBUs = scratch.OBUs[:size.Packetizer.OBUs]
	scratch.Packets = scratch.Packets[:size.Packetizer.Packets]
	scratch.Work = scratch.Work[:size.Packetizer.Work]
	return scratch, nil
}

func encoderWebRTCPictureTemporalUnitFrameRTPPacketsSize(framePayload []byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo, internalencoder.WebRTCFrameControl, internalencoder.WebRTCFrameDependencyStructure, error) {
	frameOBUSize, control, structure, err := encoderWebRTCPictureTemporalUnitFrameOBUSize(framePayload, unit, state, frameIndex)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	frameOBU, _, _, err := appendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUScratch[:0], framePayload, unit, state, frameIndex)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	rtpSize, rtpControl, rtpStructure, err := encoderWebRTCPictureTemporalUnitRTPPacketsSize(frameOBU, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	if rtpControl != control || rtpStructure != structure {
		return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{
		FrameOBUBytes: frameOBUSize,
		RTP:           rtpSize,
	}, control, structure, nil
}

func appendEncoderWebRTCPictureTemporalUnitFrameRTPPackets(payloadDst []byte, descriptorDst []byte, spans []EncoderWebRTCRTPPacketSpan, frameOBUScratch []byte, framePayload []byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (frameOBU []byte, rtpPayloads []byte, descriptors []byte, packetCount int, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure, err error) {
	frameOBU, control, structure, err = appendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUScratch[:0], framePayload, unit, state, frameIndex)
	if err != nil {
		return frameOBUScratch[:0], payloadDst, descriptorDst, 0, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	rtpPayloads, descriptors, packetCount, rtpControl, rtpStructure, err := appendEncoderWebRTCPictureTemporalUnitRTPPackets(payloadDst, descriptorDst, spans, frameOBU, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return frameOBU, payloadDst, descriptorDst, 0, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	if rtpControl != control || rtpStructure != structure {
		return frameOBU, payloadDst, descriptorDst, 0, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrEncoderInvalidFrame
	}
	return frameOBU, rtpPayloads, descriptors, packetCount, control, structure, nil
}

func encoderWebRTCPictureTemporalUnitFramesRTPPacketsSize(framePayloads [][]byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo, error) {
	frameNum := encoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, ErrEncoderInvalidFrame
	}
	frameOBUBytes, err := encoderWebRTCPictureTemporalUnitFramesOBUSize(framePayloads, unit, state)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, err
	}
	frameOBUs := frameOBUScratch[:0]
	size := encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{FrameOBUBytes: frameOBUBytes}
	for i := uint8(0); i < frameNum; i++ {
		frameStart := len(frameOBUs)
		nextFrameOBUs, _, _, err := appendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUs, framePayloads[i], unit, state, i)
		if err != nil {
			return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, err
		}
		frameOBUs = nextFrameOBUs
		rtpSize, _, _, err := encoderWebRTCPictureTemporalUnitRTPPacketsSize(frameOBUs[frameStart:], limits, unit, state, i, obuScratch, packetScratch, workScratch)
		if err != nil {
			return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, err
		}
		size.RTP.PacketCount += rtpSize.PacketCount
		size.RTP.PayloadBytes += rtpSize.PayloadBytes
		size.RTP.DescriptorBytes += rtpSize.DescriptorBytes
	}
	return size, nil
}

func appendEncoderWebRTCPictureTemporalUnitFramesRTPPackets(frameOBUDst []byte, payloadDst []byte, descriptorDst []byte, frameSpans []encoderWebRTCFrameRTPPacketSpan, packetSpans []EncoderWebRTCRTPPacketSpan, framePayloads [][]byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (frameOBUs []byte, rtpPayloads []byte, descriptors []byte, frameCount int, packetCount int, err error) {
	frameNum := encoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(framePayloads) != int(frameNum) {
		return frameOBUDst, payloadDst, descriptorDst, 0, 0, ErrEncoderInvalidFrame
	}
	if len(frameSpans) < int(frameNum) {
		return frameOBUDst, payloadDst, descriptorDst, 0, 0, ErrRTPPacketPlanTooSmall
	}

	frameOBUs = frameOBUDst
	rtpPayloads = payloadDst
	descriptors = descriptorDst
	for i := uint8(0); i < frameNum; i++ {
		frameStart := len(frameOBUs)
		packetStart := packetCount
		nextFrameOBUs, _, _, err := appendEncoderWebRTCPictureTemporalUnitFrameOBU(frameOBUs, framePayloads[i], unit, state, i)
		if err != nil {
			return frameOBUDst, payloadDst, descriptorDst, 0, 0, err
		}
		frameOBUs = nextFrameOBUs
		nextPayloads, nextDescriptors, wrotePackets, _, _, err := appendEncoderWebRTCPictureTemporalUnitRTPPackets(rtpPayloads, descriptors, packetSpans[packetCount:], frameOBUs[frameStart:], limits, unit, state, i, obuScratch, packetScratch, workScratch)
		if err != nil {
			return frameOBUDst, payloadDst, descriptorDst, 0, 0, err
		}
		frameSpans[i] = encoderWebRTCFrameRTPPacketSpan{
			FrameOBUOffset: frameStart,
			FrameOBULength: len(frameOBUs) - frameStart,
			PacketOffset:   packetStart,
			PacketCount:    wrotePackets,
		}
		rtpPayloads = nextPayloads
		descriptors = nextDescriptors
		packetCount += wrotePackets
		frameCount++
	}
	return frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, nil
}

func appendEncoderWebRTCPictureTemporalUnitFramesRTPPacketsWithScratch(payloadDst []byte, descriptorDst []byte, scratch encoderWebRTCPictureTemporalUnitFramesRTPScratch, framePayloads [][]byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState) (frameOBUs []byte, rtpPayloads []byte, descriptors []byte, frameCount int, packetCount int, err error) {
	return appendEncoderWebRTCPictureTemporalUnitFramesRTPPackets(
		scratch.FrameOBU,
		payloadDst,
		descriptorDst,
		scratch.FrameSpans,
		scratch.PacketSpans,
		framePayloads,
		limits,
		unit,
		state,
		scratch.OBUs,
		scratch.Packets,
		scratch.Work,
	)
}

func encoderWebRTCNextTemporalUnitFramesRTPScratchLen(framePayloads [][]byte, limits RTPPayloadSizeLimits, config EncoderConfig, state internalencoder.WebRTCEncoderState, forceKeyFrame bool, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU) (encoderWebRTCPictureTemporalUnitFramesRTPScratchSize, internalencoder.WebRTCPictureTemporalUnit, internalencoder.WebRTCEncoderState, error) {
	unit, next, err := internalencoder.WebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitFramesRTPScratchSize{}, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.WebRTCEncoderState{}, err
	}
	size, err := encoderWebRTCPictureTemporalUnitFramesRTPScratchLen(framePayloads, limits, unit, next, frameOBUScratch, obuScratch)
	if err != nil {
		return size, unit, next, err
	}
	return size, unit, next, nil
}

func encoderWebRTCNextTemporalUnitFramesRTPPacketsSize(framePayloads [][]byte, limits RTPPayloadSizeLimits, config EncoderConfig, state internalencoder.WebRTCEncoderState, forceKeyFrame bool, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo, internalencoder.WebRTCPictureTemporalUnit, internalencoder.WebRTCEncoderState, error) {
	unit, next, err := internalencoder.WebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.WebRTCEncoderState{}, err
	}
	size, err := encoderWebRTCPictureTemporalUnitFramesRTPPacketsSize(framePayloads, limits, unit, next, frameOBUScratch, obuScratch, packetScratch, workScratch)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitFrameRTPPacketsSizeInfo{}, unit, next, err
	}
	return size, unit, next, nil
}

func appendEncoderWebRTCNextTemporalUnitFramesRTPPacketsWithScratch(payloadDst []byte, descriptorDst []byte, scratch encoderWebRTCPictureTemporalUnitFramesRTPScratch, framePayloads [][]byte, limits RTPPayloadSizeLimits, config EncoderConfig, state internalencoder.WebRTCEncoderState, forceKeyFrame bool) (frameOBUs []byte, rtpPayloads []byte, descriptors []byte, frameCount int, packetCount int, unit internalencoder.WebRTCPictureTemporalUnit, next internalencoder.WebRTCEncoderState, err error) {
	unit, next, err = internalencoder.WebRTCNextTemporalUnitForState(config, state, forceKeyFrame)
	if err != nil {
		return scratch.FrameOBU, payloadDst, descriptorDst, 0, 0, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.WebRTCEncoderState{}, err
	}
	frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, err = appendEncoderWebRTCPictureTemporalUnitFramesRTPPacketsWithScratch(
		payloadDst,
		descriptorDst,
		scratch,
		framePayloads,
		limits,
		unit,
		next,
	)
	if err != nil {
		return scratch.FrameOBU, payloadDst, descriptorDst, 0, 0, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.WebRTCEncoderState{}, err
	}
	return frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, unit, next, nil
}

func encoderWebRTCPictureTemporalUnitRTPScratchLen(payload []byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, obuScratch []RTPPacketizerOBU) (encoderWebRTCPictureTemporalUnitRTPScratchSize, error) {
	packetizer, err := RTPPacketizerScratchLen(payload, limits, obuScratch)
	size := encoderWebRTCPictureTemporalUnitRTPScratchSize{Packetizer: packetizer}
	if err != nil {
		return size, err
	}
	descriptor, err := encoderWebRTCPictureTemporalUnitMaxDependencyDescriptorSize(unit, state, frameIndex)
	if err != nil {
		return size, err
	}
	size.MaxDescriptorBytes = descriptor
	if packetizer.OBUs != 0 {
		size.MaxPayloadBytes = limits.MaxPayloadLen
	}
	return size, nil
}

func newEncoderWebRTCPictureTemporalUnitRTPPacketizer(payload []byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (RTPPacketizer, internalencoder.WebRTCFrameControl, internalencoder.WebRTCFrameDependencyStructure, error) {
	control, structure, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return RTPPacketizer{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	packetizer, err := NewRTPPacketizer(
		payload,
		limits,
		control.Settings.Type == internalencoder.FrameTypeKey,
		frameIndex+1 == encoderWebRTCPictureTemporalUnitFrameNum(unit),
		obuScratch,
		packetScratch,
		workScratch,
	)
	if err != nil {
		return RTPPacketizer{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	return packetizer, control, structure, nil
}

func encoderWebRTCPictureTemporalUnitRTPPacketsSize(payload []byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (encoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo, internalencoder.WebRTCFrameControl, internalencoder.WebRTCFrameDependencyStructure, error) {
	packetizer, control, structure, err := newEncoderWebRTCPictureTemporalUnitRTPPacketizer(payload, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	size := encoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo{
		PacketCount:  packetizer.NumPackets(),
		PayloadBytes: RTPPacketizerRemainingPayloadSize(&packetizer),
	}
	if size.PacketCount == 0 {
		return size, control, structure, nil
	}
	firstDescriptor, err := internalencoder.WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, control.AttachDependencyStructure)
	if err != nil {
		return encoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	size.DescriptorBytes = firstDescriptor
	if size.PacketCount > 1 {
		descriptor, err := internalencoder.WebRTCDependencyDescriptorSize(structure, control.GenericFrameInfo, false)
		if err != nil {
			return encoderWebRTCPictureTemporalUnitRTPPacketsSizeInfo{}, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
		}
		size.DescriptorBytes += descriptor * (size.PacketCount - 1)
	}
	return size, control, structure, nil
}

func appendEncoderWebRTCPictureTemporalUnitFirstRTPPacket(payloadDst []byte, descriptorDst []byte, payload []byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (rtpPayload []byte, descriptor []byte, marker bool, ok bool, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure, err error) {
	packetizer, control, structure, err := newEncoderWebRTCPictureTemporalUnitRTPPacketizer(payload, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return payloadDst, descriptorDst, false, false, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	rtpPayload, descriptor, marker, ok, err = appendEncoderWebRTCFrameControlRTPPacket(payloadDst, descriptorDst, &packetizer, control, structure)
	if err != nil {
		return payloadDst, descriptorDst, false, ok, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	return rtpPayload, descriptor, marker, ok, control, structure, nil
}

func appendEncoderWebRTCPictureTemporalUnitRTPPackets(payloadDst []byte, descriptorDst []byte, spans []EncoderWebRTCRTPPacketSpan, payload []byte, limits RTPPayloadSizeLimits, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8, obuScratch []RTPPacketizerOBU, packetScratch []RTPPacketPlan, workScratch []RTPPacketPlan) (rtpPayloads []byte, descriptors []byte, packetCount int, control internalencoder.WebRTCFrameControl, structure internalencoder.WebRTCFrameDependencyStructure, err error) {
	packetizer, control, structure, err := newEncoderWebRTCPictureTemporalUnitRTPPacketizer(payload, limits, unit, state, frameIndex, obuScratch, packetScratch, workScratch)
	if err != nil {
		return payloadDst, descriptorDst, 0, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
	}
	rtpPayloads = payloadDst
	descriptors = descriptorDst
	for {
		if packetCount >= len(spans) {
			if packetizer.NumPackets() == 0 {
				return rtpPayloads, descriptors, packetCount, control, structure, nil
			}
			return payloadDst, descriptorDst, 0, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, ErrRTPPacketPlanTooSmall
		}
		payloadStart := len(rtpPayloads)
		descriptorStart := len(descriptors)
		nextPayloads, nextDescriptors, marker, ok, err := appendEncoderWebRTCFrameControlRTPPacket(rtpPayloads, descriptors, &packetizer, control, structure)
		if err != nil {
			return payloadDst, descriptorDst, 0, internalencoder.WebRTCFrameControl{}, internalencoder.WebRTCFrameDependencyStructure{}, err
		}
		if !ok {
			return rtpPayloads, descriptors, packetCount, control, structure, nil
		}
		spans[packetCount] = EncoderWebRTCRTPPacketSpan{
			PayloadOffset:    payloadStart,
			PayloadLength:    len(nextPayloads) - payloadStart,
			DescriptorOffset: descriptorStart,
			DescriptorLength: len(nextDescriptors) - descriptorStart,
			Marker:           marker,
		}
		rtpPayloads = nextPayloads
		descriptors = nextDescriptors
		packetCount++
	}
}

func encoderWebRTCPictureTemporalUnitRTPPacketSize(packetizer *RTPPacketizer, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8) (payloadSize int, descriptorSize int, ok bool, err error) {
	control, structure, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return 0, 0, false, err
	}
	return encoderWebRTCFrameControlRTPPacketSize(packetizer, control, structure)
}

func appendEncoderWebRTCPictureTemporalUnitRTPPacket(payloadDst []byte, descriptorDst []byte, packetizer *RTPPacketizer, unit internalencoder.WebRTCPictureTemporalUnit, state internalencoder.WebRTCEncoderState, frameIndex uint8) (payload []byte, descriptor []byte, marker bool, ok bool, err error) {
	control, structure, err := encoderWebRTCPictureTemporalUnitFrameControl(unit, state, frameIndex)
	if err != nil {
		return payloadDst, descriptorDst, false, false, err
	}
	return appendEncoderWebRTCFrameControlRTPPacket(payloadDst, descriptorDst, packetizer, control, structure)
}

func newWebRTCEncoder(config EncoderConfig, state internalencoder.WebRTCEncoderState) (webRTCEncoder, error) {
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return webRTCEncoder{}, err
	}
	if _, err := internalencoder.SequenceHeaderForConfig(normalized); err != nil {
		return webRTCEncoder{}, err
	}
	return webRTCEncoder{config: normalized, state: state}, nil
}

func (e *webRTCEncoder) Config() EncoderConfig {
	if e == nil {
		return EncoderConfig{}
	}
	return e.config
}

// RTPFrameDuration returns the exact RTP timestamp duration of one encoded
// picture under the encoder's current normalized config.
func (e *webRTCEncoder) RTPFrameDuration() (EncoderRational, error) {
	if e == nil {
		return EncoderRational{}, internalencoder.ErrInvalidConfig
	}
	return EncoderWebRTCRTPFrameDuration(e.config)
}

// SetConfig atomically updates the WebRTC control config. Changes that alter
// the AV1 sequence header or dependency structure reset reference/dependency
// state so the next temporal unit is a key unit while preserving frame number
// and order-hint continuity.
func (e *webRTCEncoder) SetConfig(config EncoderConfig) error {
	if e == nil {
		return internalencoder.ErrInvalidConfig
	}
	normalized, err := internalencoder.SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return err
	}
	seq, err := internalencoder.SequenceHeaderForConfig(normalized)
	if err != nil {
		return err
	}
	prevSeq, prevSeqErr := internalencoder.SequenceHeaderForConfig(e.config)
	newStructure, err := internalencoder.WebRTCFrameDependencyStructureForConfig(normalized)
	if err != nil {
		return err
	}
	prevStructure, prevStructureErr := internalencoder.WebRTCFrameDependencyStructureForConfig(e.config)
	if prevSeqErr != nil || prevSeq != seq ||
		prevStructureErr != nil || prevStructure != newStructure {
		e.state = encoderWebRTCStateForNextKey(e.state)
	}
	e.config = normalized
	return nil
}

// ApplyRTCPReceiverEstimatedMaximumBitrate parses a raw compound RTCP packet
// stream, applies the last WebRTC REMB estimate to the encoder bitrate config
// when present, and returns the parsed packet suffix. Caller-owned scratch keeps
// the feedback path allocation-free once sized.
func (e *webRTCEncoder) ApplyRTCPReceiverEstimatedMaximumBitrate(compound []byte, packetScratch []RTCPPacket, ssrcScratch []uint32) (applied bool, packets []RTCPPacket, err error) {
	if e == nil {
		return false, packetScratch, internalencoder.ErrInvalidConfig
	}
	updated, ok, packets, err := EncoderWebRTCRTCPCompoundPacketsApplyReceiverEstimatedMaximumBitrate(
		e.config,
		compound,
		packetScratch,
		ssrcScratch,
	)
	if err != nil || !ok {
		return ok, packets, err
	}
	if err := e.SetConfig(updated); err != nil {
		return true, packets, err
	}
	return true, packets, nil
}

// RTCPRequiresKeyFrame parses a raw compound RTCP packet stream and reports
// whether PLI, FIR, or valid AV1 LRR feedback asks the caller to force the next
// encoded picture as a key picture.
func (e *webRTCEncoder) RTCPRequiresKeyFrame(compound []byte, packetScratch []RTCPPacket, firScratch []RTCPFullIntraRequestEntry, lrrScratch []AV1RTCPLayerRefreshRequestEntry) (forceKey bool, packets []RTCPPacket, err error) {
	if e == nil {
		return false, packetScratch, internalencoder.ErrInvalidConfig
	}
	return EncoderWebRTCRTCPCompoundPacketsRequireKeyFrame(
		e.config,
		compound,
		packetScratch,
		firScratch,
		lrrScratch,
	)
}

// RTCPLayerRefreshTarget parses a raw compound RTCP packet stream and returns
// the highest active decode-target cap requested by valid AV1 LRR feedback.
func (e *webRTCEncoder) RTCPLayerRefreshTarget(compound []byte, packetScratch []RTCPPacket, lrrScratch []AV1RTCPLayerRefreshRequestEntry) (target AV1RTCPLayerRefreshLayerIndex, ok bool, packets []RTCPPacket, err error) {
	if e == nil {
		return AV1RTCPLayerRefreshLayerIndex{}, false, packetScratch, internalencoder.ErrInvalidConfig
	}
	return EncoderWebRTCRTCPCompoundPacketsLayerRefreshTarget(
		e.config,
		compound,
		packetScratch,
		lrrScratch,
	)
}

func encoderWebRTCStateForNextKey(state internalencoder.WebRTCEncoderState) internalencoder.WebRTCEncoderState {
	return internalencoder.WebRTCEncoderState{
		NextOrderHint: state.NextOrderHint,
		NextFrameID:   state.NextFrameID,
	}
}

func (e *webRTCEncoder) State() internalencoder.WebRTCEncoderState {
	if e == nil {
		return internalencoder.WebRTCEncoderState{}
	}
	return e.state
}

func (e *webRTCEncoder) ResetState(state internalencoder.WebRTCEncoderState) error {
	if e == nil {
		return internalencoder.ErrInvalidConfig
	}
	e.state = state
	return nil
}

func (e *webRTCEncoder) LowOverheadPictureHeaderTemporalUnitSize(forceKeyFrame bool) (int, internalencoder.WebRTCPictureTemporalUnit, error) {
	if e == nil {
		return 0, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	size, unit, _, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(e.config, e.state, forceKeyFrame)
	return size, unit, err
}

func (e *webRTCEncoder) AppendLowOverheadPictureHeaderTemporalUnit(dst []byte, forceKeyFrame bool) ([]byte, internalencoder.WebRTCPictureTemporalUnit, error) {
	if e == nil {
		return dst, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	out, unit, next, err := internalencoder.AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(dst, e.config, e.state, forceKeyFrame)
	if err != nil {
		return dst, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	e.state = next
	return out, unit, nil
}

func (e *webRTCEncoder) LowOverheadPictureHeaderTemporalUnitForFramesSize(frames []Frame, forceKeyFrame bool) (int, internalencoder.WebRTCPictureTemporalUnit, error) {
	if e == nil {
		return 0, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	size, unit, _, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(e.config, e.state, forceKeyFrame)
	if err != nil {
		return 0, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	if err := validateEncoderPictureFrames(frames, unit, e.config); err != nil {
		return 0, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	return size, unit, nil
}

func (e *webRTCEncoder) AppendLowOverheadPictureHeaderTemporalUnitForFrames(dst []byte, frames []Frame, forceKeyFrame bool) ([]byte, internalencoder.WebRTCPictureTemporalUnit, error) {
	if e == nil {
		return dst, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	_, unit, _, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(e.config, e.state, forceKeyFrame)
	if err != nil {
		return dst, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	if err := validateEncoderPictureFrames(frames, unit, e.config); err != nil {
		return dst, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	out, appended, next, err := internalencoder.AppendLowOverheadWebRTCPictureHeaderTemporalUnitForState(dst, e.config, e.state, forceKeyFrame)
	if err != nil {
		return dst, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	if appended != unit {
		return dst, internalencoder.WebRTCPictureTemporalUnit{}, ErrEncoderInvalidFrame
	}
	e.state = next
	return out, unit, nil
}

func (e *webRTCEncoder) PictureSampleScratchSize(frames []Frame, forceKeyFrame bool) (encoderWebRTCPictureSampleScratchSize, internalencoder.WebRTCPictureTemporalUnit, error) {
	if e == nil {
		return encoderWebRTCPictureSampleScratchSize{}, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	_, unit, _, err := internalencoder.LowOverheadWebRTCPictureHeaderTemporalUnitForStateSize(e.config, e.state, forceKeyFrame)
	if err != nil {
		return encoderWebRTCPictureSampleScratchSize{}, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	if err := validateEncoderPictureFrames(frames, unit, e.config); err != nil {
		return encoderWebRTCPictureSampleScratchSize{}, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	size, err := encoderPictureSampleScratchSize(frames)
	if err != nil {
		return encoderWebRTCPictureSampleScratchSize{}, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	return size, unit, nil
}

func (e *webRTCEncoder) LoadPictureSamplePlanes(scratch encoderWebRTCPictureSampleScratch, frames []Frame, forceKeyFrame bool) (encoderWebRTCPictureSamplePlanes, internalencoder.WebRTCPictureTemporalUnit, error) {
	if e == nil {
		return encoderWebRTCPictureSamplePlanes{}, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	size, unit, err := e.PictureSampleScratchSize(frames, forceKeyFrame)
	if err != nil {
		return encoderWebRTCPictureSamplePlanes{}, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	if len(scratch.Samples) < size.Samples {
		return encoderWebRTCPictureSamplePlanes{}, internalencoder.WebRTCPictureTemporalUnit{}, internalframe.ErrShortBuffer
	}
	planes, err := loadEncoderPictureSamplePlanes(scratch.Samples[:size.Samples], frames)
	if err != nil {
		return encoderWebRTCPictureSamplePlanes{}, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	return planes, unit, nil
}

func (e *webRTCEncoder) PictureTemporalUnitFramesRTPScratchSize(framePayloads [][]byte, limits RTPPayloadSizeLimits, forceKeyFrame bool, frameOBUScratch []byte, obuScratch []RTPPacketizerOBU) (encoderWebRTCPictureTemporalUnitFramesRTPScratchSize, internalencoder.WebRTCPictureTemporalUnit, error) {
	if e == nil {
		return encoderWebRTCPictureTemporalUnitFramesRTPScratchSize{}, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	size, unit, _, err := encoderWebRTCNextTemporalUnitFramesRTPScratchLen(framePayloads, limits, e.config, e.state, forceKeyFrame, frameOBUScratch, obuScratch)
	if err != nil {
		return size, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	return size, unit, nil
}

func (e *webRTCEncoder) AppendPictureTemporalUnitFramesRTPPacketsWithScratch(payloadDst []byte, descriptorDst []byte, scratch encoderWebRTCPictureTemporalUnitFramesRTPScratch, framePayloads [][]byte, limits RTPPayloadSizeLimits, forceKeyFrame bool) (frameOBUs []byte, rtpPayloads []byte, descriptors []byte, frameCount int, packetCount int, unit internalencoder.WebRTCPictureTemporalUnit, err error) {
	if e == nil {
		return scratch.FrameOBU, payloadDst, descriptorDst, 0, 0, internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, unit, next, err := appendEncoderWebRTCNextTemporalUnitFramesRTPPacketsWithScratch(
		payloadDst,
		descriptorDst,
		scratch,
		framePayloads,
		limits,
		e.config,
		e.state,
		forceKeyFrame,
	)
	if err != nil {
		return scratch.FrameOBU, payloadDst, descriptorDst, 0, 0, internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	e.state = next
	return frameOBUs, rtpPayloads, descriptors, frameCount, packetCount, unit, nil
}

func (e *webRTCEncoder) NextTemporalUnit(forceKeyFrame bool) (internalencoder.WebRTCPictureTemporalUnit, error) {
	if e == nil {
		return internalencoder.WebRTCPictureTemporalUnit{}, internalencoder.ErrInvalidConfig
	}
	unit, next, err := internalencoder.WebRTCNextTemporalUnitForState(e.config, e.state, forceKeyFrame)
	if err != nil {
		return internalencoder.WebRTCPictureTemporalUnit{}, err
	}
	e.state = next
	return unit, nil
}

func validateEncoderPictureFrames(frames []Frame, unit internalencoder.WebRTCPictureTemporalUnit, config EncoderConfig) error {
	frameNum := encoderWebRTCPictureTemporalUnitFrameNum(unit)
	if frameNum == 0 || len(frames) != int(frameNum) {
		return ErrEncoderInvalidFrame
	}
	seq, err := internalencoder.SequenceHeaderForConfig(config)
	if err != nil {
		return err
	}
	for i := uint8(0); i < frameNum; i++ {
		settings, ok := encoderPictureUnitFrameSettings(unit, i)
		if !ok || !encoderFrameMatchesSettings(frames[i], settings, seq) {
			return ErrEncoderInvalidFrame
		}
	}
	return nil
}

func encoderPictureSampleScratchSize(frames []Frame) (encoderWebRTCPictureSampleScratchSize, error) {
	total := 0
	for i := range frames {
		frame := frames[i]
		layout, err := internalframe.RequiredSize(frame.Format)
		if err != nil {
			return encoderWebRTCPictureSampleScratchSize{}, err
		}
		need, err := internalframe.SamplePlaneLen(frame.Y, layout.BytesPerSample)
		if err != nil {
			return encoderWebRTCPictureSampleScratchSize{}, err
		}
		if !checkedAddEncoderInt(&total, need) {
			return encoderWebRTCPictureSampleScratchSize{}, ErrEncoderInvalidFrame
		}
		if frame.Format.MonoChrome {
			continue
		}
		need, err = internalframe.SamplePlaneLen(frame.U, layout.BytesPerSample)
		if err != nil {
			return encoderWebRTCPictureSampleScratchSize{}, err
		}
		if !checkedAddEncoderInt(&total, need) {
			return encoderWebRTCPictureSampleScratchSize{}, ErrEncoderInvalidFrame
		}
		need, err = internalframe.SamplePlaneLen(frame.V, layout.BytesPerSample)
		if err != nil {
			return encoderWebRTCPictureSampleScratchSize{}, err
		}
		if !checkedAddEncoderInt(&total, need) {
			return encoderWebRTCPictureSampleScratchSize{}, ErrEncoderInvalidFrame
		}
	}
	return encoderWebRTCPictureSampleScratchSize{Samples: total, FrameNum: len(frames)}, nil
}

func loadEncoderPictureSamplePlanes(samples []uint16, frames []Frame) (encoderWebRTCPictureSamplePlanes, error) {
	var out encoderWebRTCPictureSamplePlanes
	if len(frames) > EncoderWebRTCMaxSpatialLayers {
		return encoderWebRTCPictureSamplePlanes{}, ErrEncoderInvalidFrame
	}
	off := 0
	for i := range frames {
		frame := frames[i]
		layout, err := internalframe.RequiredSize(frame.Format)
		if err != nil {
			return encoderWebRTCPictureSamplePlanes{}, err
		}
		yNeed, err := internalframe.SamplePlaneLen(frame.Y, layout.BytesPerSample)
		if err != nil {
			return encoderWebRTCPictureSamplePlanes{}, err
		}
		y, err := internalframe.LoadSamplePlane(samples[off:off+yNeed], frame.Y, layout.BytesPerSample)
		if err != nil {
			return encoderWebRTCPictureSamplePlanes{}, err
		}
		off += yNeed
		out.Frames[i].Y = y
		if !frame.Format.MonoChrome {
			uNeed, err := internalframe.SamplePlaneLen(frame.U, layout.BytesPerSample)
			if err != nil {
				return encoderWebRTCPictureSamplePlanes{}, err
			}
			u, err := internalframe.LoadSamplePlane(samples[off:off+uNeed], frame.U, layout.BytesPerSample)
			if err != nil {
				return encoderWebRTCPictureSamplePlanes{}, err
			}
			off += uNeed
			vNeed, err := internalframe.SamplePlaneLen(frame.V, layout.BytesPerSample)
			if err != nil {
				return encoderWebRTCPictureSamplePlanes{}, err
			}
			v, err := internalframe.LoadSamplePlane(samples[off:off+vNeed], frame.V, layout.BytesPerSample)
			if err != nil {
				return encoderWebRTCPictureSamplePlanes{}, err
			}
			off += vNeed
			out.Frames[i].U = u
			out.Frames[i].V = v
		}
	}
	out.FrameNum = uint8(len(frames))
	return out, nil
}

func checkedAddEncoderInt(dst *int, add int) bool {
	if add < 0 {
		return false
	}
	maxInt := int(^uint(0) >> 1)
	if *dst > maxInt-add {
		return false
	}
	*dst += add
	return true
}

func encoderPictureUnitFrameSettings(unit internalencoder.WebRTCPictureTemporalUnit, frameIndex uint8) (internalencoder.FrameEncodeSettings, bool) {
	if unit.Key == unit.Delta {
		return internalencoder.FrameEncodeSettings{}, false
	}
	if unit.Key {
		if frameIndex >= unit.KeyUnit.FrameNum {
			return internalencoder.FrameEncodeSettings{}, false
		}
		return unit.KeyUnit.Frames[frameIndex], true
	}
	if frameIndex >= unit.DeltaUnit.FrameNum {
		return internalencoder.FrameEncodeSettings{}, false
	}
	return unit.DeltaUnit.Frames[frameIndex], true
}

func encoderFrameMatchesSettings(f Frame, settings internalencoder.FrameEncodeSettings, seq internalencoder.SequenceHeader) bool {
	if !settings.Output || settings.Resolution.Width <= 0 || settings.Resolution.Height <= 0 {
		return false
	}
	if f.Format.Width != int(settings.Resolution.Width) || f.Format.Height != int(settings.Resolution.Height) ||
		f.Format.BitDepth != seq.ColorConfig.BitDepth || f.Format.MonoChrome != seq.ColorConfig.MonoChrome ||
		f.Format.SubsamplingX != seq.ColorConfig.SubsamplingX || f.Format.SubsamplingY != seq.ColorConfig.SubsamplingY {
		return false
	}
	layout, err := internalframe.RequiredSize(f.Format)
	if err != nil || layout.BytesPerSample <= 0 {
		return false
	}
	if f.Y.Width != f.Format.Width || f.Y.Height != f.Format.Height ||
		!encoderPlaneFits(f.Y, layout.BytesPerSample) {
		return false
	}
	if f.Format.MonoChrome {
		return len(f.U.Pix) == 0 && len(f.V.Pix) == 0 && f.U.Stride == 0 && f.V.Stride == 0
	}
	return f.U.Width == layout.ChromaWidth && f.U.Height == layout.ChromaHeight &&
		f.V.Width == layout.ChromaWidth && f.V.Height == layout.ChromaHeight &&
		encoderPlaneFits(f.U, layout.BytesPerSample) && encoderPlaneFits(f.V, layout.BytesPerSample)
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
