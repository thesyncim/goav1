package encoder

import (
	"math"

	"github.com/thesyncim/goav1/internal/av1/obu"
)

const (
	MaxLayers         = 32
	MaxTemporalLayers = 8
	MaxSpatialLayers  = 4

	WebRTCMaxSpatialLayers   = 4
	WebRTCMaxTemporalLayers  = 4
	WebRTCReferenceBuffers   = 8
	WebRTCMaxFrameReferences = 3
	WebRTCRtpTicksPerSecond  = 90000
	WebRTCMinEffortLevel     = -2
	WebRTCMaxEffortLevel     = 4
	WebRTCMaxQuantizer       = 63
	WebRTCMaxDimension       = 65536
	WebRTCMaxFramePixels     = 1 << 30
	WebRTCMaxBitrateKbps     = 2000000

	WebRTCMinSpatialLayerLongSide  = 240
	WebRTCMinSpatialLayerShortSide = 135
)

type Profile uint8

const (
	Profile0 Profile = iota
	Profile1
	Profile2
)

func (p Profile) Valid() bool {
	return p <= Profile2
}

func (p Profile) String() string {
	switch p {
	case Profile1:
		return "1"
	case Profile2:
		return "2"
	default:
		return "0"
	}
}

func ParseProfile(profile string) (Profile, bool) {
	switch profile {
	case "0":
		return Profile0, true
	case "1":
		return Profile1, true
	case "2":
		return Profile2, true
	default:
		return 0, false
	}
}

type ScalabilityMode uint8

const (
	ScalabilityModeL1T1 ScalabilityMode = iota
	ScalabilityModeL1T2
	ScalabilityModeL1T3
	ScalabilityModeL2T1
	ScalabilityModeL2T1h
	ScalabilityModeL2T1_KEY
	ScalabilityModeL2T2
	ScalabilityModeL2T2h
	ScalabilityModeL2T2_KEY
	ScalabilityModeL2T2_KEY_SHIFT
	ScalabilityModeL2T3
	ScalabilityModeL2T3h
	ScalabilityModeL2T3_KEY
	ScalabilityModeL2T3_KEY_SHIFT
	ScalabilityModeL3T1
	ScalabilityModeL3T1h
	ScalabilityModeL3T1_KEY
	ScalabilityModeL3T2
	ScalabilityModeL3T2h
	ScalabilityModeL3T2_KEY
	ScalabilityModeL3T2_KEY_SHIFT
	ScalabilityModeL3T3
	ScalabilityModeL3T3h
	ScalabilityModeL3T3_KEY
	ScalabilityModeL3T3_KEY_SHIFT
	ScalabilityModeS2T1
	ScalabilityModeS2T1h
	ScalabilityModeS2T2
	ScalabilityModeS2T2h
	ScalabilityModeS2T3
	ScalabilityModeS2T3h
	ScalabilityModeS3T1
	ScalabilityModeS3T1h
	ScalabilityModeS3T2
	ScalabilityModeS3T2h
	ScalabilityModeS3T3
	ScalabilityModeS3T3h

	scalabilityModeCount
)

type scalabilityInfo struct {
	name      string
	spatial   uint8
	temporal  uint8
	key       bool
	keyShift  bool
	smallStep bool
	simulcast bool
}

var scalabilityModes = [...]scalabilityInfo{
	{name: "L1T1", spatial: 1, temporal: 1},
	{name: "L1T2", spatial: 1, temporal: 2},
	{name: "L1T3", spatial: 1, temporal: 3},
	{name: "L2T1", spatial: 2, temporal: 1},
	{name: "L2T1h", spatial: 2, temporal: 1, smallStep: true},
	{name: "L2T1_KEY", spatial: 2, temporal: 1, key: true},
	{name: "L2T2", spatial: 2, temporal: 2},
	{name: "L2T2h", spatial: 2, temporal: 2, smallStep: true},
	{name: "L2T2_KEY", spatial: 2, temporal: 2, key: true},
	{name: "L2T2_KEY_SHIFT", spatial: 2, temporal: 2, key: true, keyShift: true},
	{name: "L2T3", spatial: 2, temporal: 3},
	{name: "L2T3h", spatial: 2, temporal: 3, smallStep: true},
	{name: "L2T3_KEY", spatial: 2, temporal: 3, key: true},
	{name: "L2T3_KEY_SHIFT", spatial: 2, temporal: 3, key: true, keyShift: true},
	{name: "L3T1", spatial: 3, temporal: 1},
	{name: "L3T1h", spatial: 3, temporal: 1, smallStep: true},
	{name: "L3T1_KEY", spatial: 3, temporal: 1, key: true},
	{name: "L3T2", spatial: 3, temporal: 2},
	{name: "L3T2h", spatial: 3, temporal: 2, smallStep: true},
	{name: "L3T2_KEY", spatial: 3, temporal: 2, key: true},
	{name: "L3T2_KEY_SHIFT", spatial: 3, temporal: 2, key: true, keyShift: true},
	{name: "L3T3", spatial: 3, temporal: 3},
	{name: "L3T3h", spatial: 3, temporal: 3, smallStep: true},
	{name: "L3T3_KEY", spatial: 3, temporal: 3, key: true},
	{name: "L3T3_KEY_SHIFT", spatial: 3, temporal: 3, key: true, keyShift: true},
	{name: "S2T1", spatial: 2, temporal: 1, simulcast: true},
	{name: "S2T1h", spatial: 2, temporal: 1, smallStep: true, simulcast: true},
	{name: "S2T2", spatial: 2, temporal: 2, simulcast: true},
	{name: "S2T2h", spatial: 2, temporal: 2, smallStep: true, simulcast: true},
	{name: "S2T3", spatial: 2, temporal: 3, simulcast: true},
	{name: "S2T3h", spatial: 2, temporal: 3, smallStep: true, simulcast: true},
	{name: "S3T1", spatial: 3, temporal: 1, simulcast: true},
	{name: "S3T1h", spatial: 3, temporal: 1, smallStep: true, simulcast: true},
	{name: "S3T2", spatial: 3, temporal: 2, simulcast: true},
	{name: "S3T2h", spatial: 3, temporal: 2, smallStep: true, simulcast: true},
	{name: "S3T3", spatial: 3, temporal: 3, simulcast: true},
	{name: "S3T3h", spatial: 3, temporal: 3, smallStep: true, simulcast: true},
}

func (m ScalabilityMode) Valid() bool {
	return m < scalabilityModeCount
}

func (m ScalabilityMode) String() string {
	if !m.Valid() {
		return ""
	}
	return scalabilityModes[m].name
}

func ParseScalabilityMode(mode string) (ScalabilityMode, bool) {
	switch mode {
	case "L1T1":
		return ScalabilityModeL1T1, true
	case "L1T2":
		return ScalabilityModeL1T2, true
	case "L1T3":
		return ScalabilityModeL1T3, true
	case "L2T1":
		return ScalabilityModeL2T1, true
	case "L2T1h":
		return ScalabilityModeL2T1h, true
	case "L2T1_KEY":
		return ScalabilityModeL2T1_KEY, true
	case "L2T2":
		return ScalabilityModeL2T2, true
	case "L2T2h":
		return ScalabilityModeL2T2h, true
	case "L2T2_KEY":
		return ScalabilityModeL2T2_KEY, true
	case "L2T2_KEY_SHIFT":
		return ScalabilityModeL2T2_KEY_SHIFT, true
	case "L2T3":
		return ScalabilityModeL2T3, true
	case "L2T3h":
		return ScalabilityModeL2T3h, true
	case "L2T3_KEY":
		return ScalabilityModeL2T3_KEY, true
	case "L2T3_KEY_SHIFT":
		return ScalabilityModeL2T3_KEY_SHIFT, true
	case "L3T1":
		return ScalabilityModeL3T1, true
	case "L3T1h":
		return ScalabilityModeL3T1h, true
	case "L3T1_KEY":
		return ScalabilityModeL3T1_KEY, true
	case "L3T2":
		return ScalabilityModeL3T2, true
	case "L3T2h":
		return ScalabilityModeL3T2h, true
	case "L3T2_KEY":
		return ScalabilityModeL3T2_KEY, true
	case "L3T2_KEY_SHIFT":
		return ScalabilityModeL3T2_KEY_SHIFT, true
	case "L3T3":
		return ScalabilityModeL3T3, true
	case "L3T3h":
		return ScalabilityModeL3T3h, true
	case "L3T3_KEY":
		return ScalabilityModeL3T3_KEY, true
	case "L3T3_KEY_SHIFT":
		return ScalabilityModeL3T3_KEY_SHIFT, true
	case "S2T1":
		return ScalabilityModeS2T1, true
	case "S2T1h":
		return ScalabilityModeS2T1h, true
	case "S2T2":
		return ScalabilityModeS2T2, true
	case "S2T2h":
		return ScalabilityModeS2T2h, true
	case "S2T3":
		return ScalabilityModeS2T3, true
	case "S2T3h":
		return ScalabilityModeS2T3h, true
	case "S3T1":
		return ScalabilityModeS3T1, true
	case "S3T1h":
		return ScalabilityModeS3T1h, true
	case "S3T2":
		return ScalabilityModeS3T2, true
	case "S3T2h":
		return ScalabilityModeS3T2h, true
	case "S3T3":
		return ScalabilityModeS3T3, true
	case "S3T3h":
		return ScalabilityModeS3T3h, true
	default:
		return 0, false
	}
}

func WebRTCScalabilityModeCount() int {
	return int(scalabilityModeCount)
}

func AppendWebRTCScalabilityModes(dst []ScalabilityMode) []ScalabilityMode {
	for mode := ScalabilityMode(0); mode < scalabilityModeCount; mode++ {
		dst = append(dst, mode)
	}
	return dst
}

// ValidateWebRTCActiveScalabilityModes validates the active scalabilityMode
// values for one WebRTC sender. The WebRTC-SVC API permits multiple active
// encodings only when none of those encodings uses a single-SSRC S mode.
func ValidateWebRTCActiveScalabilityModes(modes []ScalabilityMode) error {
	hasSingleSSRCSimulcast := false
	for _, mode := range modes {
		if !mode.Valid() {
			return ErrInvalidConfig
		}
		if mode.IsSimulcast() {
			hasSingleSSRCSimulcast = true
		}
	}
	if len(modes) > 1 && hasSingleSSRCSimulcast {
		return ErrInvalidConfig
	}
	return nil
}

// WebRTCScalabilityModeIDC maps a W3C WebRTC scalabilityMode value to its
// predefined AV1 metadata_scalability() scalability_mode_idc value. ok is
// false when AV1 has no predefined IDC for the WebRTC mode. The metadata OBU
// writers still signal those shapes with MetadataScalabilityModeSS plus an
// explicit scalability_structure().
func WebRTCScalabilityModeIDC(mode ScalabilityMode) (idc uint8, ok bool) {
	switch mode {
	case ScalabilityModeL1T2:
		return obu.ScalabilityModeL1T2, true
	case ScalabilityModeL1T3:
		return obu.ScalabilityModeL1T3, true
	case ScalabilityModeL2T1:
		return obu.ScalabilityModeL2T1, true
	case ScalabilityModeL2T2:
		return obu.ScalabilityModeL2T2, true
	case ScalabilityModeL2T3:
		return obu.ScalabilityModeL2T3, true
	case ScalabilityModeS2T1:
		return obu.ScalabilityModeS2T1, true
	case ScalabilityModeS2T2:
		return obu.ScalabilityModeS2T2, true
	case ScalabilityModeS2T3:
		return obu.ScalabilityModeS2T3, true
	case ScalabilityModeL2T1h:
		return obu.ScalabilityModeL2T1h, true
	case ScalabilityModeL2T2h:
		return obu.ScalabilityModeL2T2h, true
	case ScalabilityModeL2T3h:
		return obu.ScalabilityModeL2T3h, true
	case ScalabilityModeS2T1h:
		return obu.ScalabilityModeS2T1h, true
	case ScalabilityModeS2T2h:
		return obu.ScalabilityModeS2T2h, true
	case ScalabilityModeS2T3h:
		return obu.ScalabilityModeS2T3h, true
	case ScalabilityModeL3T1:
		return obu.ScalabilityModeL3T1, true
	case ScalabilityModeL3T2:
		return obu.ScalabilityModeL3T2, true
	case ScalabilityModeL3T3:
		return obu.ScalabilityModeL3T3, true
	case ScalabilityModeS3T1:
		return obu.ScalabilityModeS3T1, true
	case ScalabilityModeS3T2:
		return obu.ScalabilityModeS3T2, true
	case ScalabilityModeS3T3:
		return obu.ScalabilityModeS3T3, true
	case ScalabilityModeL2T2_KEY:
		return obu.ScalabilityModeL3T2_KEY, true
	case ScalabilityModeL2T3_KEY:
		return obu.ScalabilityModeL3T3_KEY, true
	case ScalabilityModeL3T2_KEY:
		return obu.ScalabilityModeL4T5_KEY, true
	case ScalabilityModeL3T3_KEY:
		return obu.ScalabilityModeL4T7_KEY, true
	case ScalabilityModeL2T2_KEY_SHIFT:
		return obu.ScalabilityModeL3T2_KEY_SHIFT, true
	case ScalabilityModeL2T3_KEY_SHIFT:
		return obu.ScalabilityModeL3T3_KEY_SHIFT, true
	case ScalabilityModeL3T2_KEY_SHIFT:
		return obu.ScalabilityModeL4T5_KEY_SHIFT, true
	case ScalabilityModeL3T3_KEY_SHIFT:
		return obu.ScalabilityModeL4T7_KEY_SHIFT, true
	default:
		return 0, false
	}
}

func (m ScalabilityMode) AV1ScalabilityModeIDC() (idc uint8, ok bool) {
	return WebRTCScalabilityModeIDC(m)
}

func (m ScalabilityMode) Layers() (spatial uint8, temporal uint8, key bool, ok bool) {
	if !m.Valid() {
		return 0, 0, false, false
	}
	info := scalabilityModes[m]
	return info.spatial, info.temporal, info.key, true
}

func (m ScalabilityMode) UsesSmallResolutionStep() bool {
	return m.Valid() && scalabilityModes[m].smallStep
}

func (m ScalabilityMode) IsSimulcast() bool {
	return m.Valid() && scalabilityModes[m].simulcast
}

func (m ScalabilityMode) UsesSharedReferenceSlots() bool {
	spatial, _, _, ok := m.Layers()
	return ok && spatial > 1 && !m.IsSimulcast()
}

func (m ScalabilityMode) UsesKeyFrameInterLayerDependency() bool {
	return m.Valid() && scalabilityModes[m].key
}

func (m ScalabilityMode) UsesKeyFrameInterLayerDependencyShift() bool {
	return m.Valid() && scalabilityModes[m].keyShift
}

func DefaultScalabilityMode(temporalLayers uint8, spatialLayers uint8) (ScalabilityMode, bool) {
	if spatialLayers == 0 {
		spatialLayers = 1
	}
	if temporalLayers == 0 {
		temporalLayers = 1
	}
	key := spatialLayers > 1
	return scalabilityModeFor(spatialLayers, temporalLayers, key, false, false, false)
}

func LimitScalabilityModeSpatialLayers(mode ScalabilityMode, limit uint8) (ScalabilityMode, bool) {
	if !mode.Valid() || limit == 0 {
		return 0, false
	}
	info := scalabilityModes[mode]
	if info.spatial <= limit {
		return mode, true
	}
	if limit > 3 {
		limit = 3
	}
	if limit == 1 {
		return scalabilityModeFor(1, info.temporal, false, false, false, false)
	}
	keyShift := info.keyShift && info.temporal > 1
	return scalabilityModeFor(limit, info.temporal, info.key, info.smallStep, info.simulcast, keyShift)
}

func scalabilityModeFor(spatial uint8, temporal uint8, key bool, smallStep bool, simulcast bool, keyShift bool) (ScalabilityMode, bool) {
	switch spatial {
	case 1:
		if key || smallStep || simulcast || keyShift {
			return 0, false
		}
		switch temporal {
		case 1:
			return ScalabilityModeL1T1, true
		case 2:
			return ScalabilityModeL1T2, true
		case 3:
			return ScalabilityModeL1T3, true
		default:
			return 0, false
		}
	case 2:
		return scalabilityModeFor2Spatial(temporal, key, smallStep, simulcast, keyShift)
	case 3:
		return scalabilityModeFor3SpatialShift(temporal, key, smallStep, simulcast, keyShift)
	default:
		return 0, false
	}
}

func scalabilityModeFor2Spatial(temporal uint8, key bool, smallStep bool, simulcast bool, keyShift bool) (ScalabilityMode, bool) {
	if simulcast {
		if key || keyShift {
			return 0, false
		}
		switch temporal {
		case 1:
			if smallStep {
				return ScalabilityModeS2T1h, true
			}
			return ScalabilityModeS2T1, true
		case 2:
			if smallStep {
				return ScalabilityModeS2T2h, true
			}
			return ScalabilityModeS2T2, true
		case 3:
			if smallStep {
				return ScalabilityModeS2T3h, true
			}
			return ScalabilityModeS2T3, true
		default:
			return 0, false
		}
	}
	if smallStep {
		if key || keyShift {
			return 0, false
		}
		switch temporal {
		case 1:
			return ScalabilityModeL2T1h, true
		case 2:
			return ScalabilityModeL2T2h, true
		case 3:
			return ScalabilityModeL2T3h, true
		default:
			return 0, false
		}
	}
	if keyShift {
		if key {
			switch temporal {
			case 2:
				return ScalabilityModeL2T2_KEY_SHIFT, true
			case 3:
				return ScalabilityModeL2T3_KEY_SHIFT, true
			}
		}
		return 0, false
	}
	switch temporal {
	case 1:
		if key {
			return ScalabilityModeL2T1_KEY, true
		}
		return ScalabilityModeL2T1, true
	case 2:
		if key {
			return ScalabilityModeL2T2_KEY, true
		}
		return ScalabilityModeL2T2, true
	case 3:
		if key {
			return ScalabilityModeL2T3_KEY, true
		}
		return ScalabilityModeL2T3, true
	default:
		return 0, false
	}
}

func scalabilityModeFor3Spatial(temporal uint8, key bool, smallStep bool, simulcast bool) (ScalabilityMode, bool) {
	return scalabilityModeFor3SpatialShift(temporal, key, smallStep, simulcast, false)
}

func scalabilityModeFor3SpatialShift(temporal uint8, key bool, smallStep bool, simulcast bool, keyShift bool) (ScalabilityMode, bool) {
	if simulcast {
		if key || keyShift {
			return 0, false
		}
		switch temporal {
		case 1:
			if smallStep {
				return ScalabilityModeS3T1h, true
			}
			return ScalabilityModeS3T1, true
		case 2:
			if smallStep {
				return ScalabilityModeS3T2h, true
			}
			return ScalabilityModeS3T2, true
		case 3:
			if smallStep {
				return ScalabilityModeS3T3h, true
			}
			return ScalabilityModeS3T3, true
		default:
			return 0, false
		}
	}
	if smallStep {
		if key || keyShift {
			return 0, false
		}
		switch temporal {
		case 1:
			return ScalabilityModeL3T1h, true
		case 2:
			return ScalabilityModeL3T2h, true
		case 3:
			return ScalabilityModeL3T3h, true
		default:
			return 0, false
		}
	}
	if keyShift {
		if key {
			switch temporal {
			case 2:
				return ScalabilityModeL3T2_KEY_SHIFT, true
			case 3:
				return ScalabilityModeL3T3_KEY_SHIFT, true
			}
		}
		return 0, false
	}
	switch temporal {
	case 1:
		if key {
			return ScalabilityModeL3T1_KEY, true
		}
		return ScalabilityModeL3T1, true
	case 2:
		if key {
			return ScalabilityModeL3T2_KEY, true
		}
		return ScalabilityModeL3T2, true
	case 3:
		if key {
			return ScalabilityModeL3T3_KEY, true
		}
		return ScalabilityModeL3T3, true
	default:
		return 0, false
	}
}

type Rational struct {
	Num int32
	Den int32
}

func (r Rational) Valid() bool {
	return r.Num > 0 && r.Den > 0
}

type Resolution struct {
	Width  int32
	Height int32
}

func (r Resolution) Valid() bool {
	return r.Width > 0 && r.Height > 0
}

type RateControlMode uint8

const (
	RateControlCBR RateControlMode = iota
	RateControlCQP
)

func (m RateControlMode) Valid() bool {
	return m == RateControlCBR || m == RateControlCQP
}

type ContentHint uint8

const (
	ContentCamera ContentHint = iota
	ContentScreen
)

func (h ContentHint) Valid() bool {
	return h == ContentCamera || h == ContentScreen
}

type SpatialLayer struct {
	Resolution        Resolution
	MinBitrateKbps    int32
	TargetBitrateKbps int32
	MaxBitrateKbps    int32
	MaxFramerate      Rational
	ScalingFactor     Rational
	TemporalLayers    uint8
	Active            bool
}

type Config struct {
	Resolution         Resolution
	Profile            Profile
	BitDepth           uint8
	ColorConfig        SequenceColorConfig
	ColorConfigSet     bool
	MaxFramerate       Rational
	MinBitrateKbps     int32
	MaxBitrateKbps     int32
	TargetBitrateKbps  int32
	RateControl        RateControlMode
	Quantizer          uint8
	MaxThreads         int32
	Speed              int8
	Content            ContentHint
	Scalability        ScalabilityMode
	SpatialLayerCount  uint8
	TemporalLayerCount uint8
	SpatialLayers      [MaxSpatialLayers]SpatialLayer
	KeyFrameInterval   int32
	RTPTimebase        Rational
	DependencyMetadata bool
	LowOverheadOBU     bool
	RTPPacketization   bool
}

func SetWebRTCSVCConfig(config Config, requestedTemporalLayers uint8, requestedSpatialLayers uint8) (Config, error) {
	config, err := normalizeConfig(config)
	if err != nil {
		return Config{}, err
	}
	requestedLayers := config.SpatialLayers

	mode := config.Scalability
	if mode == ScalabilityModeL1T1 && (requestedTemporalLayers != 0 || requestedSpatialLayers != 0) {
		var ok bool
		mode, ok = DefaultScalabilityMode(requestedTemporalLayers, requestedSpatialLayers)
		if !ok {
			return Config{}, ErrUnsupported
		}
	}

	limit := LimitedSpatialLayers(config.Resolution)
	var ok bool
	mode, ok = LimitScalabilityModeSpatialLayers(mode, limit)
	if !ok {
		return Config{}, ErrUnsupported
	}

	spatialLayers, temporalLayers, _, ok := mode.Layers()
	if !ok || spatialLayers > MaxSpatialLayers || temporalLayers > WebRTCMaxTemporalLayers {
		return Config{}, ErrUnsupported
	}

	config.Scalability = mode
	config.SpatialLayerCount = spatialLayers
	config.TemporalLayerCount = temporalLayers
	config.SpatialLayers = [MaxSpatialLayers]SpatialLayer{}
	for i := uint8(0); i < spatialLayers; i++ {
		scale := scalabilityLayerScalingFactor(mode, i, spatialLayers)
		layer := &config.SpatialLayers[i]
		layer.Resolution = scaleResolution(config.Resolution, scale)
		layer.ScalingFactor = scale
		layer.MaxFramerate = config.MaxFramerate
		layer.TemporalLayers = temporalLayers
		layer.Active = true
	}

	if spatialLayers == 1 {
		layer := &config.SpatialLayers[0]
		layer.MinBitrateKbps = config.MinBitrateKbps
		layer.MaxBitrateKbps = config.MaxBitrateKbps
		layer.TargetBitrateKbps = config.TargetBitrateKbps
		if layer.TargetBitrateKbps == 0 {
			layer.TargetBitrateKbps = (config.MinBitrateKbps + config.MaxBitrateKbps) / 2
		}
		return config, nil
	}

	for i := uint8(0); i < spatialLayers; i++ {
		layer := &config.SpatialLayers[i]
		minBitrate, maxBitrate := defaultWebRTCSpatialLayerBitrates(layer.Resolution)
		targetBitrate := (minBitrate + maxBitrate) / 2
		minBitrate, maxBitrate, targetBitrate, err = normalizeWebRTCSpatialLayerBitrates(
			requestedLayers[i],
			minBitrate,
			maxBitrate,
			targetBitrate,
		)
		if err != nil {
			return Config{}, err
		}
		layer.MinBitrateKbps = minBitrate
		layer.MaxBitrateKbps = maxBitrate
		layer.TargetBitrateKbps = targetBitrate
	}
	return config, nil
}

func defaultWebRTCSpatialLayerBitrates(resolution Resolution) (minBitrateKbps int32, maxBitrateKbps int32) {
	numPixels := int64(resolution.Width) * int64(resolution.Height)
	minBitrateKbps = int32((480.0*math.Sqrt(float64(numPixels)) - 95000.0) / 1000.0)
	if minBitrateKbps < 20 {
		minBitrateKbps = 20
	}
	maxBitrateKbps = int32(50.0 + 1.6*float64(numPixels)/1000.0)
	return minBitrateKbps, maxBitrateKbps
}

func normalizeWebRTCSpatialLayerBitrates(requested SpatialLayer, defaultMin int32, defaultMax int32, defaultTarget int32) (minBitrateKbps int32, maxBitrateKbps int32, targetBitrateKbps int32, err error) {
	minBitrateKbps = defaultMin
	maxBitrateKbps = defaultMax
	targetBitrateKbps = defaultTarget
	if requested.MinBitrateKbps != 0 {
		minBitrateKbps = requested.MinBitrateKbps
	}
	if requested.MaxBitrateKbps != 0 {
		maxBitrateKbps = requested.MaxBitrateKbps
	}
	if requested.TargetBitrateKbps != 0 {
		targetBitrateKbps = requested.TargetBitrateKbps
	}
	if minBitrateKbps < 0 || maxBitrateKbps < 0 || targetBitrateKbps < 0 ||
		minBitrateKbps > maxBitrateKbps ||
		maxBitrateKbps > WebRTCMaxBitrateKbps ||
		targetBitrateKbps > WebRTCMaxBitrateKbps ||
		targetBitrateKbps < minBitrateKbps ||
		targetBitrateKbps > maxBitrateKbps {
		return 0, 0, 0, ErrInvalidConfig
	}
	return minBitrateKbps, maxBitrateKbps, targetBitrateKbps, nil
}

func normalizeConfig(config Config) (Config, error) {
	if !config.Resolution.Valid() || !config.Profile.Valid() || !config.RateControl.Valid() || !config.Content.Valid() || !config.Scalability.Valid() {
		return Config{}, ErrInvalidConfig
	}
	if config.Resolution.Width > WebRTCMaxDimension || config.Resolution.Height > WebRTCMaxDimension {
		return Config{}, ErrInvalidConfig
	}
	if int64(config.Resolution.Width)*int64(config.Resolution.Height) > WebRTCMaxFramePixels {
		return Config{}, ErrInvalidConfig
	}
	var err error
	config, err = normalizeConfigColor(config)
	if err != nil {
		return Config{}, ErrInvalidConfig
	}
	if config.MaxFramerate == (Rational{}) {
		config.MaxFramerate = Rational{Num: 30, Den: 1}
	} else if !config.MaxFramerate.Valid() {
		return Config{}, ErrInvalidConfig
	}
	if int64(config.MaxFramerate.Num) < int64(config.MaxFramerate.Den) {
		return Config{}, ErrInvalidConfig
	}
	if config.RTPTimebase == (Rational{}) {
		config.RTPTimebase = Rational{Num: 1, Den: WebRTCRtpTicksPerSecond}
	} else if !config.RTPTimebase.Valid() {
		return Config{}, ErrInvalidConfig
	}
	if config.Speed < WebRTCMinEffortLevel || config.Speed > WebRTCMaxEffortLevel {
		return Config{}, ErrInvalidConfig
	}
	if config.MinBitrateKbps < 0 || config.MaxBitrateKbps < 0 || config.TargetBitrateKbps < 0 || config.MinBitrateKbps > config.MaxBitrateKbps {
		return Config{}, ErrInvalidConfig
	}
	if config.MaxBitrateKbps > WebRTCMaxBitrateKbps || config.TargetBitrateKbps > WebRTCMaxBitrateKbps {
		return Config{}, ErrInvalidConfig
	}
	if config.TargetBitrateKbps != 0 && (config.TargetBitrateKbps < config.MinBitrateKbps || config.TargetBitrateKbps > config.MaxBitrateKbps) {
		return Config{}, ErrInvalidConfig
	}
	if config.RateControl == RateControlCQP && config.Quantizer > WebRTCMaxQuantizer {
		return Config{}, ErrInvalidConfig
	}
	if config.RateControl != RateControlCQP && config.Quantizer != 0 {
		return Config{}, ErrInvalidConfig
	}
	if config.MaxThreads < 0 || config.KeyFrameInterval < 0 {
		return Config{}, ErrInvalidConfig
	}
	return config, nil
}

func normalizeConfigColor(config Config) (Config, error) {
	bitDepth := config.BitDepth
	if config.ColorConfigSet && config.ColorConfig.BitDepth != 0 {
		bitDepth = config.ColorConfig.BitDepth
	}
	if bitDepth == 0 {
		bitDepth = 8
	}
	switch bitDepth {
	case 8, 10, 12:
	default:
		return Config{}, ErrInvalidConfig
	}

	if config.ColorConfig != (SequenceColorConfig{}) && !config.ColorConfigSet {
		return Config{}, ErrInvalidConfig
	}
	color := config.ColorConfig
	if !config.ColorConfigSet {
		color = defaultSequenceColorConfig(config.Profile, bitDepth)
	} else {
		if color.BitDepth == 0 {
			color.BitDepth = bitDepth
		} else if color.BitDepth != bitDepth {
			return Config{}, ErrInvalidConfig
		}
	}
	if color.MonoChrome {
		color.SubsamplingX = true
		color.SubsamplingY = true
	}
	if err := validateSequenceColorConfig(config.Profile, color); err != nil {
		return Config{}, err
	}
	config.BitDepth = color.BitDepth
	config.ColorConfig = color
	config.ColorConfigSet = true
	return config, nil
}

func defaultSequenceColorConfig(profile Profile, bitDepth uint8) SequenceColorConfig {
	return SequenceColorConfig{
		BitDepth:     bitDepth,
		ColorRange:   false,
		SubsamplingX: profile != Profile1,
		SubsamplingY: profile == Profile0,
	}
}

func LimitedSpatialLayers(resolution Resolution) uint8 {
	if !resolution.Valid() {
		return 0
	}
	minWidth := float64(WebRTCMinSpatialLayerShortSide)
	minHeight := float64(WebRTCMinSpatialLayerLongSide)
	if resolution.Width >= resolution.Height {
		minWidth = WebRTCMinSpatialLayerLongSide
		minHeight = WebRTCMinSpatialLayerShortSide
	}
	layersFitHorz := uint8(math.Floor(1 + math.Max(0, math.Log2(float64(resolution.Width)/minWidth))))
	layersFitVert := uint8(math.Floor(1 + math.Max(0, math.Log2(float64(resolution.Height)/minHeight))))
	if layersFitHorz < layersFitVert {
		return layersFitHorz
	}
	return layersFitVert
}

func scalabilityLayerScalingFactor(mode ScalabilityMode, layer uint8, spatialLayers uint8) Rational {
	if spatialLayers == 0 || layer >= spatialLayers {
		return Rational{}
	}
	stepsFromTop := spatialLayers - 1 - layer
	if mode.UsesSmallResolutionStep() {
		return Rational{Num: powInt32(2, stepsFromTop), Den: powInt32(3, stepsFromTop)}
	}
	return Rational{Num: 1, Den: int32(uint32(1) << stepsFromTop)}
}

func powInt32(base int32, exp uint8) int32 {
	v := int32(1)
	for i := uint8(0); i < exp; i++ {
		v *= base
	}
	return v
}

func scaleResolution(resolution Resolution, scale Rational) Resolution {
	return Resolution{
		Width:  int32(int64(resolution.Width) * int64(scale.Num) / int64(scale.Den)),
		Height: int32(int64(resolution.Height) * int64(scale.Num) / int64(scale.Den)),
	}
}

type FrameType uint8

const (
	FrameTypeKey FrameType = iota
	FrameTypeStart
	FrameTypeDelta
)

func (t FrameType) Valid() bool {
	return t == FrameTypeKey || t == FrameTypeStart || t == FrameTypeDelta
}

func (t FrameType) resetsReferences() bool {
	return t == FrameTypeKey
}

func (t FrameType) requiresUpdateBuffer() bool {
	return t == FrameTypeKey || t == FrameTypeStart
}

type FrameEncodeSettings struct {
	Type             FrameType
	Resolution       Resolution
	SpatialID        uint8
	TemporalID       uint8
	UpdateBuffer     uint8
	UpdateBufferSet  bool
	ReferenceBuffers [WebRTCMaxFrameReferences]uint8
	ReferenceCount   uint8
	EffortLevel      int8
	RateControl      RateControlMode
	Quantizer        uint8
	Output           bool
}

type ReferenceBufferState struct {
	Valid       [WebRTCReferenceBuffers]bool
	Resolutions [WebRTCReferenceBuffers]Resolution
}

func ValidateTemporalUnitFrames(frames []FrameEncodeSettings, state ReferenceBufferState, rcMode RateControlMode) (ReferenceBufferState, error) {
	if len(frames) == 0 || !rcMode.Valid() {
		return ReferenceBufferState{}, ErrInvalidFrame
	}
	for i := range frames {
		if err := validateFrameSettings(frames, i, state, rcMode); err != nil {
			return ReferenceBufferState{}, err
		}
	}
	out := state
	for i := range frames {
		settings := frames[i]
		if settings.Type.resetsReferences() {
			out = ReferenceBufferState{}
		}
		if settings.UpdateBufferSet {
			out.Valid[settings.UpdateBuffer] = true
			out.Resolutions[settings.UpdateBuffer] = settings.Resolution
		}
	}
	return out, nil
}

func validateFrameSettings(frames []FrameEncodeSettings, index int, state ReferenceBufferState, rcMode RateControlMode) error {
	settings := frames[index]
	if !settings.Output || !settings.Type.Valid() || !settings.Resolution.Valid() || !settings.RateControl.Valid() {
		return ErrInvalidFrame
	}
	if settings.EffortLevel < WebRTCMinEffortLevel || settings.EffortLevel > WebRTCMaxEffortLevel {
		return ErrInvalidFrame
	}
	if settings.SpatialID >= WebRTCMaxSpatialLayers || settings.TemporalID >= WebRTCMaxTemporalLayers {
		return ErrInvalidFrame
	}
	if index > 0 && frames[index-1].SpatialID >= settings.SpatialID {
		return ErrInvalidFrame
	}
	if settings.Type.requiresUpdateBuffer() && !settings.UpdateBufferSet {
		return ErrInvalidFrame
	}
	if settings.Type.requiresUpdateBuffer() && settings.ReferenceCount != 0 {
		return ErrInvalidFrame
	}
	if settings.UpdateBufferSet && settings.UpdateBuffer >= WebRTCReferenceBuffers {
		return ErrInvalidFrame
	}
	if settings.ReferenceCount > WebRTCMaxFrameReferences {
		return ErrInvalidFrame
	}
	if settings.RateControl != rcMode {
		return ErrInvalidFrame
	}
	if settings.RateControl == RateControlCQP && settings.Quantizer > WebRTCMaxQuantizer {
		return ErrInvalidFrame
	}
	for j := uint8(0); j < settings.ReferenceCount; j++ {
		ref := settings.ReferenceBuffers[j]
		if ref >= WebRTCReferenceBuffers || duplicateReference(settings.ReferenceBuffers, settings.ReferenceCount, j) {
			return ErrInvalidFrame
		}
		refResolution, ok := temporalUnitReferenceResolution(frames, index, ref, state)
		if !ok {
			return ErrInvalidFrame
		}
		if _, ok := SupportedResolutionScaling(refResolution, settings.Resolution); !ok {
			return ErrInvalidFrame
		}
	}
	return nil
}

func duplicateReference(refs [WebRTCMaxFrameReferences]uint8, count uint8, index uint8) bool {
	for i := index + 1; i < count; i++ {
		if refs[index] == refs[i] {
			return true
		}
	}
	return false
}

func temporalUnitReferenceResolution(frames []FrameEncodeSettings, index int, ref uint8, state ReferenceBufferState) (Resolution, bool) {
	var resolution Resolution
	var ok bool
	var keyframeOnPreviousLayer bool
	for i := 0; i < index; i++ {
		settings := frames[i]
		if settings.Type.resetsReferences() {
			keyframeOnPreviousLayer = true
			resolution = Resolution{}
			ok = false
		}
		if settings.UpdateBufferSet && settings.UpdateBuffer == ref {
			resolution = settings.Resolution
			ok = true
		}
	}
	if !ok && !keyframeOnPreviousLayer && state.Valid[ref] {
		resolution = state.Resolutions[ref]
		ok = true
	}
	return resolution, ok
}

var supportedResolutionScalingFactors = [...]Rational{
	{Num: 8, Den: 1},
	{Num: 4, Den: 1},
	{Num: 2, Den: 1},
	{Num: 3, Den: 2},
	{Num: 1, Den: 1},
	{Num: 2, Den: 3},
	{Num: 1, Den: 2},
	{Num: 1, Den: 4},
	{Num: 1, Den: 8},
}

func SupportedResolutionScaling(from Resolution, to Resolution) (Rational, bool) {
	if !from.Valid() || !to.Valid() {
		return Rational{}, false
	}
	for _, factor := range supportedResolutionScalingFactors {
		if int64(from.Width)*int64(factor.Num)/int64(factor.Den) == int64(to.Width) &&
			int64(from.Height)*int64(factor.Num)/int64(factor.Den) == int64(to.Height) {
			return factor, true
		}
		if int64(to.Width)*int64(factor.Den)/int64(factor.Num) == int64(from.Width) &&
			int64(to.Height)*int64(factor.Den)/int64(factor.Num) == int64(from.Height) {
			return factor, true
		}
	}
	return Rational{}, false
}
