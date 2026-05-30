// Ported from libaom:
//   av1/decoder/obu.c (is_obu_in_current_operating_point,
//                      get_num_layers_from_operating_point_idc)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

// AV1 scalability constants per spec section 6.4.1 (operating_point_idc()):
//
//   - operating_point_idc is 12 bits wide.
//   - The low 8 bits ([0..7]) carry the in-temporal-layer bitmask
//     (MAX_NUM_TEMPORAL_LAYERS == 8).
//   - The high 4 bits ([8..11]) carry the in-spatial-layer bitmask
//     (MAX_NUM_SPATIAL_LAYERS == 4).
//   - operating_point_idc == 0 is the "any layer" sentinel: every
//     extension-OBU is in scope regardless of its (temporal_id, spatial_id).
//
// These match libaom's MAX_NUM_TEMPORAL_LAYERS / MAX_NUM_SPATIAL_LAYERS
// macros and the obu_header extension byte's 3+2-bit layout (see
// internal/av1/obu.ParseHeader).
const (
	// MaxOperatingPointTemporalLayers is the number of distinct
	// temporal_id values that operating_point_idc can describe.
	MaxOperatingPointTemporalLayers = 8

	// MaxOperatingPointSpatialLayers is the number of distinct
	// spatial_id values that operating_point_idc can describe.
	MaxOperatingPointSpatialLayers = 4

	// OperatingPointIDCBits is the width of the operating_point_idc
	// syntax element in bits.
	OperatingPointIDCBits = 12

	// OperatingPointIDCAnyLayer is the sentinel value that selects every
	// (temporal_id, spatial_id) combination, matching the operating_point_idc
	// == 0 special case in AV1 spec section 6.4.1.
	OperatingPointIDCAnyLayer uint16 = 0
)

// OperatingPointIDCMatches reports whether a layer-tagged OBU with the
// supplied (temporal_id, spatial_id) participates in the operating point
// described by idc.
//
// idc == OperatingPointIDCAnyLayer (i.e. 0) matches every layer. Otherwise
// the OBU participates only if both the temporal_id bit and the spatial_id
// bit (shifted by MaxOperatingPointTemporalLayers) are set, matching
// libaom's is_obu_in_current_operating_point().
//
// Out-of-range layer IDs (temporal_id >= MaxOperatingPointTemporalLayers or
// spatial_id >= MaxOperatingPointSpatialLayers) never match a non-zero idc;
// they only match the any-layer sentinel.
func OperatingPointIDCMatches(idc uint16, temporalID uint8, spatialID uint8) bool {
	if idc == OperatingPointIDCAnyLayer {
		return true
	}
	if temporalID >= MaxOperatingPointTemporalLayers {
		return false
	}
	if spatialID >= MaxOperatingPointSpatialLayers {
		return false
	}
	if (idc>>temporalID)&1 == 0 {
		return false
	}
	if (idc>>(uint16(spatialID)+MaxOperatingPointTemporalLayers))&1 == 0 {
		return false
	}
	return true
}

// OperatingPointTemporalLayerCount returns the number of distinct
// temporal layers selected by idc, matching libaom's
// get_num_layers_from_operating_point_idc().
//
// When idc == OperatingPointIDCAnyLayer the count is 1 (a single-layer
// stream).
func OperatingPointTemporalLayerCount(idc uint16) uint8 {
	if idc == OperatingPointIDCAnyLayer {
		return 1
	}
	var n uint8
	for j := range uint8(MaxOperatingPointTemporalLayers) {
		n += uint8((idc >> j) & 1)
	}
	return n
}

// OperatingPointSpatialLayerCount returns the number of distinct spatial
// layers selected by idc, matching libaom's
// get_num_layers_from_operating_point_idc().
//
// When idc == OperatingPointIDCAnyLayer the count is 1.
func OperatingPointSpatialLayerCount(idc uint16) uint8 {
	if idc == OperatingPointIDCAnyLayer {
		return 1
	}
	var n uint8
	for j := range uint8(MaxOperatingPointSpatialLayers) {
		n += uint8((idc >> (j + MaxOperatingPointTemporalLayers)) & 1)
	}
	return n
}

// SelectOperatingPoint scans seq.OperatingPoints[:seq.OperatingPointsCount]
// in declaration order and returns the index of the first operating point
// whose operating_point_idc covers the supplied (temporal_id, spatial_id),
// matching the "first match wins" convention libaom uses in its
// is_obu_in_current_operating_point() loop.
//
// The boolean result is false when no operating point covers the layer
// pair. Callers that pin a specific operating point should use
// OperatingPointIDCMatches against that point's IDC directly; this helper
// is the convenience entry point for "pick the right operating point for
// this OBU" workflows.
func SelectOperatingPoint(seq SequenceHeader, temporalID uint8, spatialID uint8) (int, bool) {
	count := int(seq.OperatingPointsCount)
	if count <= 0 || count > len(seq.OperatingPoints) {
		return 0, false
	}
	for i := range count {
		if OperatingPointIDCMatches(seq.OperatingPoints[i].IDC, temporalID, spatialID) {
			return i, true
		}
	}
	return 0, false
}
