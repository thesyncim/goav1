package encoder

// svt_rtc_filters.go ports SVT-AV1's (v4.1.0) in-loop filter levels for the
// realtime preset 12 the video encoder follows
// (enc_mode_config.c svt_aom_sig_deriv_mode_decision_config_rtc):
//
//   - Deblocking: get_dlf_level_rtc returns dlf_level 0 above ENC_M9
//     (hierarchical) and above ENC_M10 (flat), on every layer, intra
//     pictures included; dlf_level_modulation only lowers it. SVT's own
//     preset-12 rtc streams signal loop_filter_level 0 on every frame.
//   - Restoration: svt_aom_get_wn_filter_level_rtc is 0 above ENC_M6 and
//     svt_aom_get_sg_filter_level_rtc 0 above ENC_M3, so enable_restoration
//     is off (this encoder codes no restoration).

// svtRTCDeblockLevel maps the encoder's q-derived deblocking strength to
// what SVT preset 12 rtc signals: the deblocking filter is off.
func svtRTCDeblockLevel(level uint8) uint8 {
	_ = level
	return 0
}
