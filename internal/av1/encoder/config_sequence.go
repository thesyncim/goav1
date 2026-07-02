package encoder

const webRTCDefaultOrderHintBits = 7

// SequenceHeaderForConfig maps a WebRTC encoder config into the sequence
// header descriptor consumed by AppendSequenceHeaderPayload and
// AppendLowOverheadSequenceHeaderOBU.
func SequenceHeaderForConfig(config Config) (SequenceHeader, error) {
	config, err := SetWebRTCSVCConfig(config, config.TemporalLayerCount, config.SpatialLayerCount)
	if err != nil {
		return SequenceHeader{}, err
	}

	seq := SequenceHeader{
		Profile:               config.Profile,
		OperatingPointsCount:  sequenceOperatingPointsForLayers(config.SpatialLayerCount, config.TemporalLayerCount),
		MaxFrameWidth:         uint32(config.Resolution.Width),
		MaxFrameHeight:        uint32(config.Resolution.Height),
		EnableFilterIntra:     true,
		EnableIntraEdgeFilter: true,
		// Dual filters stay off as SVT-AV1 configures the sequence
		// (Source/Lib/Codec/sequence_control_set.c enable_dual_filter = 0):
		// the switchable filter search codes one X = Y pair per block.
		EnableDualFilter:           false,
		EnableOrderHint:            true,
		SeqForceScreenContentTools: SequenceSelectScreenContentTools,
		SeqForceIntegerMV:          SequenceSelectIntegerMV,
		OrderHintBits:              webRTCDefaultOrderHintBits,
		EnableCDEF:                 true,
		ColorConfig:                config.ColorConfig,
	}
	if !videoColorSupportsInLoopFilters(parserColorConfig(config.ColorConfig)) {
		seq.EnableCDEF = false
	}

	for i := uint8(0); i < seq.OperatingPointsCount; i++ {
		seq.OperatingPoints[i].SeqLevelIdx = SequenceLevelMax
	}
	fillSequenceOperatingPointIDC(&seq, config.SpatialLayerCount, config.TemporalLayerCount)

	if err := validateSequenceHeader(seq); err != nil {
		return SequenceHeader{}, err
	}
	return seq, nil
}

func sequenceOperatingPointsForLayers(spatialLayers uint8, temporalLayers uint8) uint8 {
	if spatialLayers <= 1 && temporalLayers <= 1 {
		return 1
	}
	return spatialLayers * temporalLayers
}

func fillSequenceOperatingPointIDC(seq *SequenceHeader, spatialLayers uint8, temporalLayers uint8) {
	if seq.OperatingPointsCount == 1 {
		seq.OperatingPoints[0].IDC = 0
		return
	}

	// Source-shaped from libaom av1_set_svc_seq_params(): operating_point_idc
	// is a uint16_t carrying 8 temporal bits plus 4 spatial bits.
	var index uint8
	for sl := uint8(0); sl < spatialLayers; sl++ {
		spatialMask := uint16((uint32(1)<<(spatialLayers-sl))-1) << 8
		for tl := uint8(0); tl < temporalLayers; tl++ {
			temporalMask := uint16((uint32(1) << (temporalLayers - tl)) - 1)
			seq.OperatingPoints[index].IDC = spatialMask | temporalMask
			index++
		}
	}
}
