package goav1

import (
	"errors"
	"strconv"
	"strings"

	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	// AV1RTPMediaType is the RTP media type registered for AV1 video.
	AV1RTPMediaType = "video/AV1"
	// AV1RTPEncodingName is the SDP rtpmap encoding name for AV1.
	AV1RTPEncodingName = "AV1"
	// AV1RTPClockRate is the fixed RTP clock rate for AV1 video.
	AV1RTPClockRate = 90000

	// AV1RTPDependencyDescriptorURI is the SDP extmap URI for WebRTC's AV1
	// dependency descriptor RTP header extension.
	AV1RTPDependencyDescriptorURI = "https://aomediacodec.github.io/av1-rtp-spec/#dependency-descriptor-rtp-header-extension"

	// AV1SDPFmtpProfile is the AV1 SDP fmtp key for seq_profile.
	AV1SDPFmtpProfile = "profile"
	// AV1SDPFmtpLevelIdx is the AV1 SDP fmtp key for seq_level_idx.
	AV1SDPFmtpLevelIdx = "level-idx"
	// AV1SDPFmtpTier is the AV1 SDP fmtp key for seq_tier.
	AV1SDPFmtpTier = "tier"

	// AV1SDPDefaultProfile is the profile inferred when fmtp omits profile.
	AV1SDPDefaultProfile = 0
	// AV1SDPDefaultLevelIdx is the level inferred when fmtp omits level-idx.
	AV1SDPDefaultLevelIdx = 5
	// AV1SDPDefaultTier is the tier inferred when fmtp omits tier.
	AV1SDPDefaultTier = 0

	// AV1SDPRTCPFeedbackNACK is the SDP rtcp-fb value for generic NACK.
	AV1SDPRTCPFeedbackNACK = "nack"
	// AV1SDPRTCPFeedbackPLI is the SDP rtcp-fb value for Picture Loss
	// Indication feedback.
	AV1SDPRTCPFeedbackPLI = "nack pli"
	// AV1SDPRTCPFeedbackFIR is the SDP rtcp-fb value for Full Intra Request
	// feedback.
	AV1SDPRTCPFeedbackFIR = "ccm fir"
	// AV1SDPRTCPFeedbackLRR is the SDP rtcp-fb value for Layer Refresh
	// Request feedback.
	AV1SDPRTCPFeedbackLRR = "ccm lrr"

	// AV1SDPRIDDirectionSend is the RID direction for streams sent by the
	// endpoint declaring the RID.
	AV1SDPRIDDirectionSend = "send"
	// AV1SDPRIDDirectionReceive is the RID direction for streams received by
	// the endpoint declaring the RID.
	AV1SDPRIDDirectionReceive = "recv"

	// AV1SDPRIDPayloadTypes is the RID parameter for payload-type binding.
	AV1SDPRIDPayloadTypes = "pt"
	// AV1SDPRIDMaxWidth is the RID restriction for max-width.
	AV1SDPRIDMaxWidth = "max-width"
	// AV1SDPRIDMaxHeight is the RID restriction for max-height.
	AV1SDPRIDMaxHeight = "max-height"
	// AV1SDPRIDMaxFrameRate is the RID restriction for max-fps.
	AV1SDPRIDMaxFrameRate = "max-fps"
	// AV1SDPRIDMaxFrameSize is the RID restriction for max-fs.
	AV1SDPRIDMaxFrameSize = "max-fs"
	// AV1SDPRIDMaxBitrate is the RID restriction for max-br.
	AV1SDPRIDMaxBitrate = "max-br"
	// AV1SDPRIDMaxPixelsPerSecond is the RID restriction for max-pps.
	AV1SDPRIDMaxPixelsPerSecond = "max-pps"
	// AV1SDPRIDMaxBitsPerPixel is the RID restriction for max-bpp.
	AV1SDPRIDMaxBitsPerPixel = "max-bpp"
	// AV1SDPRIDDepend is the RID restriction for dependent RID identifiers.
	AV1SDPRIDDepend = "depend"
)

// ErrSDPInvalidConfig is returned when AV1 SDP/fmtp parameters are malformed
// or describe profile/level/tier values this package cannot emit or parse.
var ErrSDPInvalidConfig = errors.New("goav1: invalid SDP configuration")

// AV1SDPFmtpParameters holds the AV1 RTP fmtp profile, level-idx, and tier
// parameters. Per the AV1 RTP payload format, the same fields can describe a
// stream's bitstream properties or a receiver's maximum supported properties.
type AV1SDPFmtpParameters struct {
	// Profile is the highest AV1 seq_profile used or supported.
	Profile int
	// LevelIdx is the highest AV1 seq_level_idx used or supported.
	LevelIdx int
	// Tier is the highest AV1 seq_tier used or supported.
	Tier int
}

// AV1SDPRIDRestrictions holds the AV1 interpretation of RFC 8851 RID video
// restrictions. Zero numeric fields mean the corresponding optional
// restriction was not declared, or was declared without a concrete value.
type AV1SDPRIDRestrictions struct {
	// MaxWidth is max-width in samples.
	MaxWidth int
	// MaxHeight is max-height in samples.
	MaxHeight int
	// MaxFrameRate is max-fps in temporal units per second.
	MaxFrameRate int
	// MaxFrameSize is max-fs in samples per frame.
	MaxFrameSize int
	// MaxBitrate is max-br in media payload bits per second.
	MaxBitrate int
	// MaxPixelsPerSecond is max-pps in decoded samples per second.
	MaxPixelsPerSecond int
	// MaxBitsPerPixelX10000 is max-bpp multiplied by 10000. For example, a
	// max-bpp value of 1.5 is stored as 15000.
	MaxBitsPerPixelX10000 int
	// DependsOn is the optional depend RID list.
	DependsOn []string
}

// AV1SDPRID holds a complete SDP a=rid attribute.
type AV1SDPRID struct {
	// ID is the RID identifier, unique within a media section.
	ID string
	// Direction is "send" or "recv".
	Direction string
	// PayloadTypes is the optional pt= list. Nil means all payload types in the
	// media section are allowed by this RID.
	PayloadTypes []string
	// Restrictions are the RID restrictions attached to this stream.
	Restrictions AV1SDPRIDRestrictions
}

// DefaultAV1SDPFmtpParameters returns the profile/level/tier inferred by the
// AV1 RTP payload format when an SDP fmtp line omits these parameters.
func DefaultAV1SDPFmtpParameters() AV1SDPFmtpParameters {
	return AV1SDPFmtpParameters{
		Profile:  AV1SDPDefaultProfile,
		LevelIdx: AV1SDPDefaultLevelIdx,
		Tier:     AV1SDPDefaultTier,
	}
}

// AV1SDPFmtpParametersForSequence returns the profile/level/tier parameters
// needed to advertise the supplied AV1 sequence header. For scalable streams it
// uses the highest operating-point level/tier present in the sequence.
func AV1SDPFmtpParametersForSequence(seq SequenceHeader) (AV1SDPFmtpParameters, error) {
	if seq.SeqProfile > uint8(EncoderProfile2) {
		return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
	}
	count := int(seq.OperatingPointsCount)
	if count <= 0 || count > len(seq.OperatingPoints) {
		return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
	}
	out := AV1SDPFmtpParameters{Profile: int(seq.SeqProfile)}
	for i := 0; i < count; i++ {
		op := seq.OperatingPoints[i]
		current := AV1SDPFmtpParameters{
			Profile:  out.Profile,
			LevelIdx: int(op.SeqLevelIdx),
			Tier:     int(op.SeqTier),
		}
		if err := current.Validate(); err != nil {
			return AV1SDPFmtpParameters{}, err
		}
		if av1SDPFmtpLevelTierCompare(current.LevelIdx, current.Tier,
			out.LevelIdx, out.Tier) > 0 {
			out.LevelIdx = current.LevelIdx
			out.Tier = current.Tier
		}
	}
	return out, nil
}

// Validate rejects fmtp parameters outside the AV1 bitstream syntax ranges.
func (p AV1SDPFmtpParameters) Validate() error {
	if p.Profile < int(EncoderProfile0) || p.Profile > int(EncoderProfile2) {
		return ErrSDPInvalidConfig
	}
	if p.LevelIdx < 0 || p.LevelIdx > EncoderSequenceLevelMax ||
		!internalparser.ValidSequenceLevelIndex(uint8(p.LevelIdx)) {
		return ErrSDPInvalidConfig
	}
	if p.Tier < 0 || p.Tier > 1 {
		return ErrSDPInvalidConfig
	}
	if p.LevelIdx <= 7 && p.Tier != 0 {
		return ErrSDPInvalidConfig
	}
	return nil
}

// Allows reports whether p, interpreted as receiver capabilities, can accept
// a stream described by stream.
func (p AV1SDPFmtpParameters) Allows(stream AV1SDPFmtpParameters) (bool, error) {
	if err := p.Validate(); err != nil {
		return false, err
	}
	if err := stream.Validate(); err != nil {
		return false, err
	}
	if stream.Profile > p.Profile {
		return false, nil
	}
	return av1SDPFmtpLevelTierCompare(stream.LevelIdx, stream.Tier,
		p.LevelIdx, p.Tier) <= 0, nil
}

// AllowsSequence reports whether p, interpreted as receiver capabilities, can
// accept a stream described by seq.
func (p AV1SDPFmtpParameters) AllowsSequence(seq SequenceHeader) (bool, error) {
	stream, err := AV1SDPFmtpParametersForSequence(seq)
	if err != nil {
		return false, err
	}
	return p.Allows(stream)
}

// AppendFmtp appends a semicolon-separated AV1 fmtp parameter string in the
// order profile, level-idx, tier.
func (p AV1SDPFmtpParameters) AppendFmtp(dst []byte) ([]byte, error) {
	if err := p.Validate(); err != nil {
		return dst, err
	}
	dst = append(dst, AV1SDPFmtpProfile...)
	dst = append(dst, '=')
	dst = strconv.AppendInt(dst, int64(p.Profile), 10)
	dst = append(dst, ';', ' ')
	dst = append(dst, AV1SDPFmtpLevelIdx...)
	dst = append(dst, '=')
	dst = strconv.AppendInt(dst, int64(p.LevelIdx), 10)
	dst = append(dst, ';', ' ')
	dst = append(dst, AV1SDPFmtpTier...)
	dst = append(dst, '=')
	dst = strconv.AppendInt(dst, int64(p.Tier), 10)
	return dst, nil
}

// Fmtp returns a semicolon-separated AV1 fmtp parameter string.
func (p AV1SDPFmtpParameters) Fmtp() (string, error) {
	buf, err := p.AppendFmtp(nil)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// Validate rejects nonsensical RID restriction values. Zero numeric values are
// allowed because RID restrictions are optional and RFC 8851 also permits a
// restriction name without a concrete value during offer negotiation.
func (r AV1SDPRIDRestrictions) Validate() error {
	if r.MaxWidth < 0 || r.MaxHeight < 0 || r.MaxFrameRate < 0 ||
		r.MaxFrameSize < 0 || r.MaxBitrate < 0 || r.MaxPixelsPerSecond < 0 {
		return ErrSDPInvalidConfig
	}
	if r.MaxBitsPerPixelX10000 < 0 || r.MaxBitsPerPixelX10000 > 480000 {
		return ErrSDPInvalidConfig
	}
	for _, id := range r.DependsOn {
		if !av1SDPValidRIDID(id) {
			return ErrSDPInvalidConfig
		}
	}
	return nil
}

// AllowsFrame reports whether width x height at fps fits the concrete RID
// spatial, frame-rate, frame-size, and pixel-rate restrictions. Bitrate and
// bits-per-pixel restrictions require encoded-size information and are checked
// by AllowsEncodedFrame.
func (r AV1SDPRIDRestrictions) AllowsFrame(width int, height int, fps int) (bool, error) {
	if err := r.Validate(); err != nil {
		return false, err
	}
	if width <= 0 || height <= 0 || fps <= 0 {
		return false, ErrSDPInvalidConfig
	}
	samples := int64(width) * int64(height)
	if samples <= 0 {
		return false, ErrSDPInvalidConfig
	}
	if r.MaxWidth > 0 && width > r.MaxWidth {
		return false, nil
	}
	if r.MaxHeight > 0 && height > r.MaxHeight {
		return false, nil
	}
	if r.MaxFrameRate > 0 && fps > r.MaxFrameRate {
		return false, nil
	}
	if r.MaxFrameSize > 0 && samples > int64(r.MaxFrameSize) {
		return false, nil
	}
	if r.MaxPixelsPerSecond > 0 &&
		samples*int64(fps) > int64(r.MaxPixelsPerSecond) {
		return false, nil
	}
	return true, nil
}

// AllowsEncodedFrame reports whether width x height at fps, bitrate, and
// current encoded frame size fit every concrete RID restriction. bitrate is in
// media-payload bits per second and frameBits is the coded frame size in bits.
func (r AV1SDPRIDRestrictions) AllowsEncodedFrame(
	width int,
	height int,
	fps int,
	bitrate int,
	frameBits int,
) (bool, error) {
	allowed, err := r.AllowsFrame(width, height, fps)
	if err != nil || !allowed {
		return allowed, err
	}
	if r.MaxBitrate > 0 {
		if bitrate <= 0 {
			return false, ErrSDPInvalidConfig
		}
		if bitrate > r.MaxBitrate {
			return false, nil
		}
	}
	if r.MaxBitsPerPixelX10000 > 0 {
		if frameBits <= 0 {
			return false, ErrSDPInvalidConfig
		}
		samples := int64(width) * int64(height)
		if int64(frameBits)*10000 > int64(r.MaxBitsPerPixelX10000)*samples {
			return false, nil
		}
	}
	return true, nil
}

// AppendRestrictions appends a semicolon-separated RID restriction list in
// RFC 8851 order.
func (r AV1SDPRIDRestrictions) AppendRestrictions(dst []byte) ([]byte, error) {
	if err := r.Validate(); err != nil {
		return dst, err
	}
	first := true
	appendParam := func(key string, value string) {
		if !first {
			dst = append(dst, ';')
		}
		first = false
		dst = append(dst, key...)
		dst = append(dst, '=')
		dst = append(dst, value...)
	}
	if r.MaxWidth > 0 {
		appendParam(AV1SDPRIDMaxWidth, strconv.Itoa(r.MaxWidth))
	}
	if r.MaxHeight > 0 {
		appendParam(AV1SDPRIDMaxHeight, strconv.Itoa(r.MaxHeight))
	}
	if r.MaxFrameRate > 0 {
		appendParam(AV1SDPRIDMaxFrameRate, strconv.Itoa(r.MaxFrameRate))
	}
	if r.MaxFrameSize > 0 {
		appendParam(AV1SDPRIDMaxFrameSize, strconv.Itoa(r.MaxFrameSize))
	}
	if r.MaxBitrate > 0 {
		appendParam(AV1SDPRIDMaxBitrate, strconv.Itoa(r.MaxBitrate))
	}
	if r.MaxPixelsPerSecond > 0 {
		appendParam(AV1SDPRIDMaxPixelsPerSecond, strconv.Itoa(r.MaxPixelsPerSecond))
	}
	if r.MaxBitsPerPixelX10000 > 0 {
		appendParam(AV1SDPRIDMaxBitsPerPixel,
			av1SDPFormatBPPX10000(r.MaxBitsPerPixelX10000))
	}
	if len(r.DependsOn) > 0 {
		appendParam(AV1SDPRIDDepend, strings.Join(r.DependsOn, ","))
	}
	return dst, nil
}

// Restrictions returns a semicolon-separated RID restriction list.
func (r AV1SDPRIDRestrictions) Restrictions() (string, error) {
	buf, err := r.AppendRestrictions(nil)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// Validate rejects malformed complete RID attributes.
func (r AV1SDPRID) Validate() error {
	if !av1SDPValidRIDID(r.ID) {
		return ErrSDPInvalidConfig
	}
	switch strings.ToLower(strings.TrimSpace(r.Direction)) {
	case AV1SDPRIDDirectionSend, AV1SDPRIDDirectionReceive:
	default:
		return ErrSDPInvalidConfig
	}
	for _, payloadType := range r.PayloadTypes {
		if strings.TrimSpace(payloadType) == "" || strings.Contains(payloadType, ",") {
			return ErrSDPInvalidConfig
		}
	}
	return r.Restrictions.Validate()
}

// AppendSDP appends a complete a=rid SDP attribute.
func (r AV1SDPRID) AppendSDP(dst []byte) ([]byte, error) {
	if err := r.Validate(); err != nil {
		return dst, err
	}
	dst = append(dst, "a=rid:"...)
	dst = append(dst, r.ID...)
	dst = append(dst, ' ')
	dst = append(dst, strings.ToLower(strings.TrimSpace(r.Direction))...)
	paramsStart := len(dst)
	if len(r.PayloadTypes) > 0 {
		dst = append(dst, ' ')
		dst = append(dst, AV1SDPRIDPayloadTypes...)
		dst = append(dst, '=')
		for i, payloadType := range r.PayloadTypes {
			if i > 0 {
				dst = append(dst, ',')
			}
			dst = append(dst, strings.TrimSpace(payloadType)...)
		}
	}
	restrictions, err := r.Restrictions.Restrictions()
	if err != nil {
		return dst, err
	}
	if restrictions != "" {
		if len(dst) == paramsStart {
			dst = append(dst, ' ')
		} else {
			dst = append(dst, ';')
		}
		dst = append(dst, restrictions...)
	}
	return dst, nil
}

// SDP returns a complete a=rid SDP attribute.
func (r AV1SDPRID) SDP() (string, error) {
	buf, err := r.AppendSDP(nil)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// ParseAV1SDPFmtp parses AV1 RTP fmtp parameters. Missing profile,
// level-idx, or tier values are filled with the AV1 RTP payload format
// defaults. Unknown fmtp parameters are ignored so callers can pass complete
// fmtp attribute values from peers.
func ParseAV1SDPFmtp(fmtp string) (AV1SDPFmtpParameters, error) {
	out := DefaultAV1SDPFmtpParameters()
	var sawProfile bool
	var sawLevelIdx bool
	var sawTier bool
	for _, part := range strings.Split(fmtp, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, value, ok := strings.Cut(part, "=")
		if !ok {
			return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.TrimSpace(value)
		switch key {
		case AV1SDPFmtpProfile:
			if sawProfile {
				return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
			}
			n, err := strconv.Atoi(value)
			if err != nil {
				return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
			}
			out.Profile = n
			sawProfile = true
		case AV1SDPFmtpLevelIdx:
			if sawLevelIdx {
				return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
			}
			n, err := strconv.Atoi(value)
			if err != nil {
				return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
			}
			out.LevelIdx = n
			sawLevelIdx = true
		case AV1SDPFmtpTier:
			if sawTier {
				return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
			}
			n, err := strconv.Atoi(value)
			if err != nil {
				return AV1SDPFmtpParameters{}, ErrSDPInvalidConfig
			}
			out.Tier = n
			sawTier = true
		}
	}
	if err := out.Validate(); err != nil {
		return AV1SDPFmtpParameters{}, err
	}
	return out, nil
}

// ParseAV1SDPRIDRestrictions parses a semicolon-separated RFC 8851 RID
// restriction list using the AV1 RTP restriction semantics. Unknown RID
// restrictions are rejected so receive-capability checks do not silently accept
// a stream against a constraint this package cannot evaluate.
func ParseAV1SDPRIDRestrictions(restrictions string) (AV1SDPRIDRestrictions, error) {
	out, _, err := av1SDPParseRIDParams(restrictions)
	return out, err
}

// ParseAV1SDPRID parses a complete RFC 8851 a=rid attribute. The input may
// include or omit the leading "a=rid:" prefix.
func ParseAV1SDPRID(line string) (AV1SDPRID, error) {
	line = strings.TrimSpace(line)
	line = strings.TrimPrefix(line, "a=rid:")
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return AV1SDPRID{}, ErrSDPInvalidConfig
	}
	out := AV1SDPRID{
		ID:        fields[0],
		Direction: strings.ToLower(strings.TrimSpace(fields[1])),
	}
	if len(fields) > 2 {
		restrictions, payloadTypes, err := av1SDPParseRIDParams(strings.Join(fields[2:], " "))
		if err != nil {
			return AV1SDPRID{}, err
		}
		out.Restrictions = restrictions
		out.PayloadTypes = payloadTypes
	}
	if err := out.Validate(); err != nil {
		return AV1SDPRID{}, err
	}
	return out, nil
}

// AV1SDPNegotiates reports whether an SDP blob contains an active video
// section that binds an AV1/90000 payload type to valid AV1 fmtp parameters.
func AV1SDPNegotiates(sdp string) bool {
	return av1SDPHas(sdp, av1SDPDirectionIsActive, nil)
}

// AV1SDPNegotiatesParams reports whether an SDP blob contains an active video
// section whose AV1 fmtp parameters allow stream.
func AV1SDPNegotiatesParams(sdp string, stream AV1SDPFmtpParameters) bool {
	return av1SDPHasParams(sdp, av1SDPDirectionIsActive, stream)
}

// AV1SDPNegotiatesRTCPFeedback reports whether an SDP blob contains an active
// video section that binds AV1/90000 and declares the supplied rtcp-fb value
// for that payload type, either directly or with the wildcard payload type.
func AV1SDPNegotiatesRTCPFeedback(sdp string, feedback string) bool {
	return av1SDPHasRTCPFeedback(sdp, av1SDPDirectionIsActive, feedback)
}

// AV1SDPOffersReceive reports whether an SDP offer contains a video section
// that can receive AV1/90000.
func AV1SDPOffersReceive(sdp string) bool {
	return av1SDPHas(sdp, av1SDPDirectionAllowsReceive, nil)
}

// AV1SDPOffersReceiveParams reports whether an SDP offer contains a video
// section that can receive AV1/90000 and whose AV1 fmtp parameters allow
// stream.
func AV1SDPOffersReceiveParams(sdp string, stream AV1SDPFmtpParameters) bool {
	return av1SDPHasParams(sdp, av1SDPDirectionAllowsReceive, stream)
}

// AV1SDPOffersReceiveRTCPFeedback reports whether an SDP offer contains a
// video section that can receive AV1/90000 and declares the supplied rtcp-fb
// value for that payload type, either directly or with the wildcard payload
// type.
func AV1SDPOffersReceiveRTCPFeedback(sdp string, feedback string) bool {
	return av1SDPHasRTCPFeedback(sdp, av1SDPDirectionAllowsReceive, feedback)
}

// AV1SDPOffersReceiveFrame reports whether an SDP offer contains a video
// section that can receive AV1/90000 and does not declare AV1-applicable RID
// restrictions below width x height at fps.
func AV1SDPOffersReceiveFrame(sdp string, width int, height int, fps int) bool {
	return av1SDPHasFrame(sdp, av1SDPDirectionAllowsReceive,
		AV1SDPRIDDirectionReceive, width, height, fps)
}

// AV1SDPOffersReceiveSequence reports whether an SDP offer contains a video
// section that can receive the AV1 stream described by seq.
func AV1SDPOffersReceiveSequence(sdp string, seq SequenceHeader) bool {
	stream, err := AV1SDPFmtpParametersForSequence(seq)
	if err != nil {
		return false
	}
	return AV1SDPOffersReceiveParams(sdp, stream)
}

// AV1SDPAnswersSend reports whether an SDP answer contains a video section
// that can send AV1/90000.
func AV1SDPAnswersSend(sdp string) bool {
	return av1SDPHas(sdp, av1SDPDirectionAllowsSend, nil)
}

// AV1SDPAnswersSendRTCPFeedback reports whether an SDP answer contains a video
// section that can send AV1/90000 and declares the supplied rtcp-fb value for
// that payload type, either directly or with the wildcard payload type.
func AV1SDPAnswersSendRTCPFeedback(sdp string, feedback string) bool {
	return av1SDPHasRTCPFeedback(sdp, av1SDPDirectionAllowsSend, feedback)
}

// AV1SDPAnswersSendFrame reports whether an SDP answer contains a video
// section that can send AV1/90000 and does not declare AV1-applicable RID
// restrictions below width x height at fps.
func AV1SDPAnswersSendFrame(sdp string, width int, height int, fps int) bool {
	return av1SDPHasFrame(sdp, av1SDPDirectionAllowsSend,
		AV1SDPRIDDirectionSend, width, height, fps)
}

func av1SDPHasParams(
	sdp string,
	directionOK func(string) bool,
	stream AV1SDPFmtpParameters,
) bool {
	if err := stream.Validate(); err != nil {
		return false
	}
	return av1SDPHas(sdp, directionOK,
		func(section av1SDPMediaSection, payloadType string) bool {
			caps := DefaultAV1SDPFmtpParameters()
			if params, ok := section.fmtpParams[payloadType]; ok {
				var err error
				caps, err = ParseAV1SDPFmtp(params)
				if err != nil {
					return false
				}
			}
			allowed, err := caps.Allows(stream)
			return err == nil && allowed
		})
}

func av1SDPHasFrame(
	sdp string,
	directionOK func(string) bool,
	ridDirection string,
	width int,
	height int,
	fps int,
) bool {
	return av1SDPHas(sdp, directionOK,
		func(section av1SDPMediaSection, payloadType string) bool {
			return section.allowsAV1Frame(payloadType, ridDirection,
				width, height, fps)
		})
}

func av1SDPHasRTCPFeedback(
	sdp string,
	directionOK func(string) bool,
	feedback string,
) bool {
	feedback = strings.ToLower(strings.TrimSpace(feedback))
	if feedback == "" {
		return false
	}
	return av1SDPHas(sdp, directionOK,
		func(section av1SDPMediaSection, payloadType string) bool {
			return section.hasRTCPFeedback(payloadType, feedback)
		})
}

func av1SDPHas(
	sdp string,
	directionOK func(string) bool,
	payloadOK func(av1SDPMediaSection, string) bool,
) bool {
	sessionDirection := "sendrecv"
	section := av1SDPMediaSection{direction: sessionDirection}
	haveSection := false
	for _, raw := range strings.Split(sdp, "\n") {
		line := strings.TrimSpace(strings.ToLower(raw))
		if strings.HasPrefix(line, "m=") {
			if haveSection && section.hasAV1(directionOK, payloadOK) {
				return true
			}
			media, active, payloadTypes := av1SDPMediaPayloadTypes(line)
			section = av1SDPMediaSection{
				media:           media,
				portActive:      active,
				payloadTypes:    payloadTypes,
				direction:       sessionDirection,
				av1PayloadTypes: make(map[string]bool),
				fmtpParams:      make(map[string]string),
				rtcpFeedback:    make(map[string][]string),
				ridSeen:         make(map[string]bool),
				ridDuplicate:    make(map[string]bool),
			}
			haveSection = true
			continue
		}
		if direction, ok := av1SDPDirection(line); ok {
			if haveSection {
				section.direction = direction
			} else {
				sessionDirection = direction
			}
			continue
		}
		if !haveSection || !section.parsesVideoPayloadAttributes() {
			continue
		}
		switch {
		case strings.HasPrefix(line, "a=rtpmap:"):
			fields := strings.Fields(strings.TrimPrefix(line, "a=rtpmap:"))
			if len(fields) >= 2 && fields[1] == "av1/90000" &&
				section.payloadTypes[fields[0]] {
				section.av1PayloadTypes[fields[0]] = true
			}
		case strings.HasPrefix(line, "a=fmtp:"):
			fields := strings.Fields(strings.TrimPrefix(line, "a=fmtp:"))
			if len(fields) >= 2 && section.payloadTypes[fields[0]] {
				section.fmtpParams[fields[0]] = strings.Join(fields[1:], " ")
			}
		case strings.HasPrefix(line, "a=rtcp-fb:"):
			fields := strings.Fields(strings.TrimPrefix(line, "a=rtcp-fb:"))
			if len(fields) >= 2 && (fields[0] == "*" || section.payloadTypes[fields[0]]) {
				section.rtcpFeedback[fields[0]] = append(section.rtcpFeedback[fields[0]],
					strings.Join(fields[1:], " "))
			}
		case strings.HasPrefix(line, "a=rid:"):
			rid, err := ParseAV1SDPRID(line)
			if err == nil && section.filterRIDPayloadTypes(&rid) {
				if section.ridSeen[rid.ID] {
					section.ridDuplicate[rid.ID] = true
				}
				section.ridSeen[rid.ID] = true
				section.rids = append(section.rids, rid)
			}
		}
	}
	return haveSection && section.hasAV1(directionOK, payloadOK)
}

type av1SDPMediaSection struct {
	media           string
	portActive      bool
	payloadTypes    map[string]bool
	direction       string
	av1PayloadTypes map[string]bool
	fmtpParams      map[string]string
	rtcpFeedback    map[string][]string
	rids            []AV1SDPRID
	ridSeen         map[string]bool
	ridDuplicate    map[string]bool
}

func (s av1SDPMediaSection) parsesVideoPayloadAttributes() bool {
	return s.media == "video" && s.portActive
}

func (s av1SDPMediaSection) hasAV1(
	directionOK func(string) bool,
	payloadOK func(av1SDPMediaSection, string) bool,
) bool {
	if !s.parsesVideoPayloadAttributes() || !directionOK(s.direction) {
		return false
	}
	for payloadType := range s.av1PayloadTypes {
		if params, ok := s.fmtpParams[payloadType]; ok {
			if _, err := ParseAV1SDPFmtp(params); err != nil {
				continue
			}
		}
		if payloadOK == nil || payloadOK(s, payloadType) {
			return true
		}
	}
	return false
}

func (s av1SDPMediaSection) hasRTCPFeedback(payloadType string, feedback string) bool {
	return av1SDPRTCPFeedbackListContains(s.rtcpFeedback[payloadType], feedback) ||
		av1SDPRTCPFeedbackListContains(s.rtcpFeedback["*"], feedback)
}

func (s av1SDPMediaSection) allowsAV1Frame(
	payloadType string,
	ridDirection string,
	width int,
	height int,
	fps int,
) bool {
	matchedRID := false
	for _, rid := range s.rids {
		if s.ridDuplicate[rid.ID] ||
			rid.Direction != ridDirection ||
			!av1SDPRIDPayloadTypesAllow(rid.PayloadTypes, payloadType) {
			continue
		}
		matchedRID = true
		allowed, err := rid.Restrictions.AllowsFrame(width, height, fps)
		if err == nil && allowed {
			return true
		}
	}
	return !matchedRID
}

func (s av1SDPMediaSection) filterRIDPayloadTypes(rid *AV1SDPRID) bool {
	if len(rid.PayloadTypes) == 0 {
		return true
	}
	filtered := rid.PayloadTypes[:0]
	for _, payloadType := range rid.PayloadTypes {
		if s.payloadTypes[payloadType] {
			filtered = append(filtered, payloadType)
		}
	}
	rid.PayloadTypes = filtered
	return len(rid.PayloadTypes) > 0
}

func av1SDPRIDPayloadTypesAllow(payloadTypes []string, payloadType string) bool {
	if len(payloadTypes) == 0 {
		return true
	}
	for _, current := range payloadTypes {
		if current == payloadType {
			return true
		}
	}
	return false
}

func av1SDPParseRIDParams(params string) (AV1SDPRIDRestrictions, []string, error) {
	var out AV1SDPRIDRestrictions
	var payloadTypes []string
	var sawPayloadTypes bool
	var sawMaxWidth bool
	var sawMaxHeight bool
	var sawMaxFrameRate bool
	var sawMaxFrameSize bool
	var sawMaxBitrate bool
	var sawMaxPixelsPerSecond bool
	var sawMaxBitsPerPixel bool
	var sawDepend bool
	for _, part := range strings.Split(params, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, value, hasValue := strings.Cut(part, "=")
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.TrimSpace(value)
		switch key {
		case AV1SDPRIDPayloadTypes:
			if sawPayloadTypes || !hasValue || value == "" {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			for _, payloadType := range strings.Split(value, ",") {
				payloadType = strings.TrimSpace(payloadType)
				if payloadType == "" {
					return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
				}
				payloadTypes = append(payloadTypes, payloadType)
			}
			sawPayloadTypes = true
		case AV1SDPRIDMaxWidth:
			n, ok, err := av1SDPParseOptionalRIDInt(hasValue, value)
			if sawMaxWidth || err != nil {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			if ok {
				out.MaxWidth = n
			}
			sawMaxWidth = true
		case AV1SDPRIDMaxHeight:
			n, ok, err := av1SDPParseOptionalRIDInt(hasValue, value)
			if sawMaxHeight || err != nil {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			if ok {
				out.MaxHeight = n
			}
			sawMaxHeight = true
		case AV1SDPRIDMaxFrameRate:
			n, ok, err := av1SDPParseOptionalRIDInt(hasValue, value)
			if sawMaxFrameRate || err != nil {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			if ok {
				out.MaxFrameRate = n
			}
			sawMaxFrameRate = true
		case AV1SDPRIDMaxFrameSize:
			n, ok, err := av1SDPParseOptionalRIDInt(hasValue, value)
			if sawMaxFrameSize || err != nil {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			if ok {
				out.MaxFrameSize = n
			}
			sawMaxFrameSize = true
		case AV1SDPRIDMaxBitrate:
			n, ok, err := av1SDPParseOptionalRIDInt(hasValue, value)
			if sawMaxBitrate || err != nil {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			if ok {
				out.MaxBitrate = n
			}
			sawMaxBitrate = true
		case AV1SDPRIDMaxPixelsPerSecond:
			n, ok, err := av1SDPParseOptionalRIDInt(hasValue, value)
			if sawMaxPixelsPerSecond || err != nil {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			if ok {
				out.MaxPixelsPerSecond = n
			}
			sawMaxPixelsPerSecond = true
		case AV1SDPRIDMaxBitsPerPixel:
			n, ok, err := av1SDPParseOptionalRIDBPP(hasValue, value)
			if sawMaxBitsPerPixel || err != nil {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			if ok {
				out.MaxBitsPerPixelX10000 = n
			}
			sawMaxBitsPerPixel = true
		case AV1SDPRIDDepend:
			if sawDepend || !hasValue || value == "" {
				return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
			}
			for _, id := range strings.Split(value, ",") {
				id = strings.TrimSpace(id)
				if !av1SDPValidRIDID(id) {
					return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
				}
				out.DependsOn = append(out.DependsOn, id)
			}
			sawDepend = true
		default:
			return AV1SDPRIDRestrictions{}, nil, ErrSDPInvalidConfig
		}
	}
	if err := out.Validate(); err != nil {
		return AV1SDPRIDRestrictions{}, nil, err
	}
	return out, payloadTypes, nil
}

func av1SDPParseOptionalRIDInt(hasValue bool, value string) (int, bool, error) {
	if !hasValue {
		return 0, false, nil
	}
	if value == "" {
		return 0, false, ErrSDPInvalidConfig
	}
	n, err := strconv.Atoi(value)
	if err != nil || n <= 0 {
		return 0, false, ErrSDPInvalidConfig
	}
	return n, true, nil
}

func av1SDPParseOptionalRIDBPP(hasValue bool, value string) (int, bool, error) {
	if !hasValue {
		return 0, false, nil
	}
	scaled, err := av1SDPParseBPPX10000(value)
	if err != nil {
		return 0, false, err
	}
	return scaled, true, nil
}

func av1SDPParseBPPX10000(value string) (int, error) {
	integer, fraction, ok := strings.Cut(strings.TrimSpace(value), ".")
	if !ok || integer == "" || fraction == "" || len(fraction) > 4 ||
		!av1SDPAllDigits(integer) || !av1SDPAllDigits(fraction) {
		return 0, ErrSDPInvalidConfig
	}
	intPart, err := strconv.Atoi(integer)
	if err != nil {
		return 0, ErrSDPInvalidConfig
	}
	for len(fraction) < 4 {
		fraction += "0"
	}
	fracPart, err := strconv.Atoi(fraction)
	if err != nil {
		return 0, ErrSDPInvalidConfig
	}
	scaled := intPart*10000 + fracPart
	if scaled < 1 || scaled > 480000 {
		return 0, ErrSDPInvalidConfig
	}
	return scaled, nil
}

func av1SDPFormatBPPX10000(value int) string {
	integer := value / 10000
	fraction := strconv.Itoa(value % 10000)
	for len(fraction) < 4 {
		fraction = "0" + fraction
	}
	fraction = strings.TrimRight(fraction, "0")
	if fraction == "" {
		fraction = "0"
	}
	return strconv.Itoa(integer) + "." + fraction
}

func av1SDPAllDigits(value string) bool {
	for _, ch := range value {
		if ch < '0' || ch > '9' {
			return false
		}
	}
	return value != ""
}

func av1SDPValidRIDID(value string) bool {
	if value == "" {
		return false
	}
	for _, ch := range value {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
			(ch >= '0' && ch <= '9') || ch == '-' || ch == '_' {
			continue
		}
		return false
	}
	return true
}

func av1SDPRTCPFeedbackListContains(values []string, feedback string) bool {
	for _, value := range values {
		if strings.ToLower(strings.TrimSpace(value)) == feedback {
			return true
		}
	}
	return false
}

func av1SDPMediaPayloadTypes(line string) (string, bool, map[string]bool) {
	fields := strings.Fields(strings.TrimPrefix(line, "m="))
	if len(fields) < 4 {
		return "", false, nil
	}
	payloadTypes := make(map[string]bool, len(fields)-3)
	for _, payloadType := range fields[3:] {
		payloadTypes[payloadType] = true
	}
	return fields[0], !av1SDPMediaPortIsZero(fields[1]), payloadTypes
}

func av1SDPMediaPortIsZero(port string) bool {
	first, _, _ := strings.Cut(port, "/")
	first = strings.TrimLeft(first, "0")
	return first == ""
}

func av1SDPDirection(line string) (string, bool) {
	switch line {
	case "a=sendrecv", "a=sendonly", "a=recvonly", "a=inactive":
		return strings.TrimPrefix(line, "a="), true
	default:
		return "", false
	}
}

func av1SDPDirectionIsActive(direction string) bool {
	return direction != "inactive"
}

func av1SDPDirectionAllowsReceive(direction string) bool {
	return direction == "" || direction == "sendrecv" || direction == "recvonly"
}

func av1SDPDirectionAllowsSend(direction string) bool {
	return direction == "" || direction == "sendrecv" || direction == "sendonly"
}

func av1SDPFmtpLevelTierCompare(levelA int, tierA int, levelB int, tierB int) int {
	if levelA < levelB {
		return -1
	}
	if levelA > levelB {
		return 1
	}
	if tierA < tierB {
		return -1
	}
	if tierA > tierB {
		return 1
	}
	return 0
}
