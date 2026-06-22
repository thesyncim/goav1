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
