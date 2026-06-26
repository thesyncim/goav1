// Ported from libaom: av1/decoder/obu.c (read_metadata_* helpers)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package obu

import (
	"errors"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
)

// MetadataType is the AV1 metadata_type leb128 field (spec section 6.7.1).
type MetadataType uint32

const (
	// MetadataTypeReserved0 is the reserved leb128 value 0.
	MetadataTypeReserved0 MetadataType = 0
	// MetadataTypeHDRCLL carries content light level info (max_cll, max_fall).
	MetadataTypeHDRCLL MetadataType = 1
	// MetadataTypeHDRMDCV carries the mastering display color volume.
	MetadataTypeHDRMDCV MetadataType = 2
	// MetadataTypeScalability carries the scalability descriptor.
	MetadataTypeScalability MetadataType = 3
	// MetadataTypeITUTT35 carries ITU-T T.35 registered payloads (CEA-708,
	// AFD, HDR10+, Dolby Vision, etc.).
	MetadataTypeITUTT35 MetadataType = 4
	// MetadataTypeTimecode carries the SMPTE-style timecode payload.
	MetadataTypeTimecode MetadataType = 5
)

func (t MetadataType) String() string {
	switch t {
	case MetadataTypeITUTT35:
		return "METADATA_TYPE_ITUT_T35"
	case MetadataTypeHDRCLL:
		return "METADATA_TYPE_HDR_CLL"
	case MetadataTypeHDRMDCV:
		return "METADATA_TYPE_HDR_MDCV"
	case MetadataTypeScalability:
		return "METADATA_TYPE_SCALABILITY"
	case MetadataTypeTimecode:
		return "METADATA_TYPE_TIMECODE"
	default:
		return "METADATA_TYPE_RESERVED"
	}
}

// Metadata parse errors.
var (
	ErrMetadataShortPayload     = errors.New("obu: metadata short payload")
	ErrMetadataMissingTrailing  = errors.New("obu: metadata missing trailing byte")
	ErrMetadataInvalidTrailing  = errors.New("obu: metadata invalid trailing byte")
	ErrMetadataMissingCountry   = errors.New("obu: metadata itu_t_t35 missing country code")
	ErrMetadataMissingCountryEx = errors.New("obu: metadata itu_t_t35 missing country code extension")
	ErrMetadataInvalidReserved  = errors.New("obu: metadata reserved type carries zero payload")
)

// MetadataITUTT35 is the parsed ITU-T T.35 metadata payload.
//
// Payload aliases the caller-provided buffer, the same as Unit.Payload.
type MetadataITUTT35 struct {
	CountryCode          uint8
	CountryCodeExtension uint8
	HasCountryExtension  bool
	Payload              []byte
}

// MetadataHDRCLL is the parsed HDR Content Light Level metadata payload
// (spec section 6.7.3).
type MetadataHDRCLL struct {
	MaxCLL  uint16
	MaxFALL uint16
}

// MetadataHDRMDCV is the parsed HDR Mastering Display Color Volume metadata
// payload (spec section 6.7.4).
//
// Indices 0..2 of PrimaryChromaticityX/Y correspond to the R, G, B color
// primaries respectively, matching the order defined by the spec.
type MetadataHDRMDCV struct {
	PrimaryChromaticityX    [3]uint16
	PrimaryChromaticityY    [3]uint16
	WhitePointChromaticityX uint16
	WhitePointChromaticityY uint16
	LuminanceMax            uint32
	LuminanceMin            uint32
}

// MetadataScalability is the parsed scalability metadata payload
// (spec section 6.7.5). When ModeIDC equals ScalabilityModeSS, Structure
// holds the optional scalability_structure() payload.
type MetadataScalability struct {
	ModeIDC      uint8
	Structure    ScalabilityStructure
	HasStructure bool
}

// ScalabilityMode* are the predefined scalability_mode_idc values from AV1
// metadata_scalability().
const (
	ScalabilityModeL1T2 uint8 = iota
	ScalabilityModeL1T3
	ScalabilityModeL2T1
	ScalabilityModeL2T2
	ScalabilityModeL2T3
	ScalabilityModeS2T1
	ScalabilityModeS2T2
	ScalabilityModeS2T3
	ScalabilityModeL2T1h
	ScalabilityModeL2T2h
	ScalabilityModeL2T3h
	ScalabilityModeS2T1h
	ScalabilityModeS2T2h
	ScalabilityModeS2T3h
	ScalabilityModeSS
	ScalabilityModeL3T1
	ScalabilityModeL3T2
	ScalabilityModeL3T3
	ScalabilityModeS3T1
	ScalabilityModeS3T2
	ScalabilityModeS3T3
	ScalabilityModeL3T2_KEY
	ScalabilityModeL3T3_KEY
	ScalabilityModeL4T5_KEY
	ScalabilityModeL4T7_KEY
	ScalabilityModeL3T2_KEY_SHIFT
	ScalabilityModeL3T3_KEY_SHIFT
	ScalabilityModeL4T5_KEY_SHIFT
	ScalabilityModeL4T7_KEY_SHIFT
)

// ScalabilityStructure is the parsed scalability_structure() payload.
type ScalabilityStructure struct {
	SpatialLayersCountMinus1        uint8
	SpatialLayerDimensionsPresent   bool
	SpatialLayerDescriptionPresent  bool
	TemporalGroupDescriptionPresent bool
	SpatialLayerMaxWidth            [4]uint16
	SpatialLayerMaxHeight           [4]uint16
	SpatialLayerRefID               [4]uint8
	TemporalGroupSize               uint8
	TemporalGroup                   [255]ScalabilityTemporalGroupEntry
}

// ScalabilityTemporalGroupEntry is one entry of the temporal-group description.
type ScalabilityTemporalGroupEntry struct {
	TemporalID          uint8
	TemporalSwitchingUp bool
	SpatialSwitchingUp  bool
	RefCount            uint8
	RefPicDiff          [7]uint8
}

// MetadataTimecode is the parsed timecode metadata payload
// (spec section 6.7.6).
type MetadataTimecode struct {
	CountingType     uint8
	FullTimestamp    bool
	Discontinuity    bool
	CountDropped     bool
	NFrames          uint16
	SecondsFlag      bool
	HasSeconds       bool
	HasMinutes       bool
	HasHours         bool
	Seconds          uint8
	Minutes          uint8
	Hours            uint8
	TimeOffsetLength uint8
	TimeOffsetValue  uint32
}

// Metadata is the parsed AV1 Metadata OBU payload.
//
// Type selects which of the embedded variant fields is populated.
type Metadata struct {
	Type        MetadataType
	ITUTT35     MetadataITUTT35
	HDRCLL      MetadataHDRCLL
	HDRMDCV     MetadataHDRMDCV
	Scalability MetadataScalability
	Timecode    MetadataTimecode
}

// ParseMetadata parses an AV1 Metadata OBU payload (the bytes after the OBU
// header and obu_size leb128, i.e. the same slice carried in Unit.Payload).
//
// The returned Metadata aliases payload when the variant exposes byte slices
// (ITU-T T.35), and never allocates.
func ParseMetadata(payload []byte) (Metadata, error) {
	var meta Metadata
	if len(payload) == 0 {
		return meta, ErrMetadataShortPayload
	}

	value, n, err := bitstream.ReadLEB128(payload)
	if err != nil {
		return meta, err
	}
	meta.Type = MetadataType(value)
	rest := payload[n:]

	switch meta.Type {
	case MetadataTypeITUTT35:
		t35, err := parseMetadataITUTT35(rest)
		if err != nil {
			return meta, err
		}
		meta.ITUTT35 = t35
	case MetadataTypeHDRCLL:
		cll, err := parseMetadataHDRCLL(rest)
		if err != nil {
			return meta, err
		}
		meta.HDRCLL = cll
	case MetadataTypeHDRMDCV:
		mdcv, err := parseMetadataHDRMDCV(rest)
		if err != nil {
			return meta, err
		}
		meta.HDRMDCV = mdcv
	case MetadataTypeScalability:
		sc, err := parseMetadataScalability(rest)
		if err != nil {
			return meta, err
		}
		meta.Scalability = sc
	case MetadataTypeTimecode:
		tc, err := parseMetadataTimecode(rest)
		if err != nil {
			return meta, err
		}
		meta.Timecode = tc
	default:
		// Reserved or user-private metadata types: per spec the OBU is
		// ignored but trailing bits must still be present (at least one
		// nonzero byte). libaom enforces the same check.
		if lastNonZero(rest) == 0 {
			return meta, ErrMetadataInvalidReserved
		}
	}
	return meta, nil
}

// parseMetadataITUTT35 parses the metadata_itut_t35() payload per spec
// section 6.7.2. The returned Payload slice aliases src.
func parseMetadataITUTT35(src []byte) (MetadataITUTT35, error) {
	var t35 MetadataITUTT35
	if len(src) == 0 {
		return t35, ErrMetadataMissingCountry
	}
	t35.CountryCode = src[0]
	countrySize := 1
	if t35.CountryCode == 0xFF {
		if len(src) < 2 {
			return t35, ErrMetadataMissingCountryEx
		}
		t35.CountryCodeExtension = src[1]
		t35.HasCountryExtension = true
		countrySize = 2
	}

	// itu_t_t35_payload_bytes is byte aligned. The last nonzero byte of the
	// trailing-bits region must be 0x80, matching libaom's check.
	endIndex := lastNonZeroIndex(src)
	if endIndex < countrySize {
		return t35, ErrMetadataMissingTrailing
	}
	if src[endIndex] != 0x80 {
		return t35, ErrMetadataInvalidTrailing
	}
	t35.Payload = src[countrySize:endIndex]
	return t35, nil
}

// parseMetadataHDRCLL parses metadata_hdr_cll() per spec section 6.7.3.
func parseMetadataHDRCLL(src []byte) (MetadataHDRCLL, error) {
	const payloadSize = 4
	if len(src) < payloadSize {
		return MetadataHDRCLL{}, ErrMetadataShortPayload
	}
	if err := requireTrailing(src[payloadSize:]); err != nil {
		return MetadataHDRCLL{}, err
	}
	return MetadataHDRCLL{
		MaxCLL:  uint16(src[0])<<8 | uint16(src[1]),
		MaxFALL: uint16(src[2])<<8 | uint16(src[3]),
	}, nil
}

// parseMetadataHDRMDCV parses metadata_hdr_mdcv() per spec section 6.7.4.
func parseMetadataHDRMDCV(src []byte) (MetadataHDRMDCV, error) {
	const payloadSize = 24
	if len(src) < payloadSize {
		return MetadataHDRMDCV{}, ErrMetadataShortPayload
	}
	if err := requireTrailing(src[payloadSize:]); err != nil {
		return MetadataHDRMDCV{}, err
	}
	var m MetadataHDRMDCV
	for i := range 3 {
		m.PrimaryChromaticityX[i] = uint16(src[i*4])<<8 | uint16(src[i*4+1])
		m.PrimaryChromaticityY[i] = uint16(src[i*4+2])<<8 | uint16(src[i*4+3])
	}
	m.WhitePointChromaticityX = uint16(src[12])<<8 | uint16(src[13])
	m.WhitePointChromaticityY = uint16(src[14])<<8 | uint16(src[15])
	m.LuminanceMax = uint32(src[16])<<24 | uint32(src[17])<<16 | uint32(src[18])<<8 | uint32(src[19])
	m.LuminanceMin = uint32(src[20])<<24 | uint32(src[21])<<16 | uint32(src[22])<<8 | uint32(src[23])
	return m, nil
}

// parseMetadataScalability parses metadata_scalability() per spec section
// 6.7.5 followed by trailing-bit validation.
func parseMetadataScalability(src []byte) (MetadataScalability, error) {
	var sc MetadataScalability
	r := bitstream.NewReader(src)
	mode, err := r.ReadBits(8)
	if err != nil {
		return sc, err
	}
	sc.ModeIDC = uint8(mode)
	if sc.ModeIDC == ScalabilityModeSS {
		structure, err := parseScalabilityStructure(&r)
		if err != nil {
			return sc, err
		}
		sc.Structure = structure
		sc.HasStructure = true
	}
	if err := r.ReadTrailingBits(); err != nil {
		return sc, err
	}
	return sc, nil
}

func parseScalabilityStructure(r *bitstream.Reader) (ScalabilityStructure, error) {
	var s ScalabilityStructure
	v, err := r.ReadBits(2)
	if err != nil {
		return s, err
	}
	s.SpatialLayersCountMinus1 = uint8(v)
	if v, err := r.ReadBool(); err != nil {
		return s, err
	} else {
		s.SpatialLayerDimensionsPresent = v
	}
	if v, err := r.ReadBool(); err != nil {
		return s, err
	} else {
		s.SpatialLayerDescriptionPresent = v
	}
	if v, err := r.ReadBool(); err != nil {
		return s, err
	} else {
		s.TemporalGroupDescriptionPresent = v
	}
	if _, err := r.ReadBits(3); err != nil { // scalability_structure_reserved_3bits
		return s, err
	}

	layers := int(s.SpatialLayersCountMinus1) + 1
	if s.SpatialLayerDimensionsPresent {
		for i := range layers {
			w, err := r.ReadBits(16)
			if err != nil {
				return s, err
			}
			h, err := r.ReadBits(16)
			if err != nil {
				return s, err
			}
			s.SpatialLayerMaxWidth[i] = uint16(w)
			s.SpatialLayerMaxHeight[i] = uint16(h)
		}
	}
	if s.SpatialLayerDescriptionPresent {
		for i := range layers {
			ref, err := r.ReadBits(8)
			if err != nil {
				return s, err
			}
			s.SpatialLayerRefID[i] = uint8(ref)
		}
	}
	if s.TemporalGroupDescriptionPresent {
		size, err := r.ReadBits(8)
		if err != nil {
			return s, err
		}
		s.TemporalGroupSize = uint8(size)
		for i := 0; i < int(s.TemporalGroupSize); i++ {
			var entry ScalabilityTemporalGroupEntry
			tid, err := r.ReadBits(3)
			if err != nil {
				return s, err
			}
			entry.TemporalID = uint8(tid)
			if v, err := r.ReadBool(); err != nil {
				return s, err
			} else {
				entry.TemporalSwitchingUp = v
			}
			if v, err := r.ReadBool(); err != nil {
				return s, err
			} else {
				entry.SpatialSwitchingUp = v
			}
			refCount, err := r.ReadBits(3)
			if err != nil {
				return s, err
			}
			entry.RefCount = uint8(refCount)
			for j := 0; j < int(entry.RefCount); j++ {
				diff, err := r.ReadBits(8)
				if err != nil {
					return s, err
				}
				entry.RefPicDiff[j] = uint8(diff)
			}
			s.TemporalGroup[i] = entry
		}
	}
	return s, nil
}

// parseMetadataTimecode parses metadata_timecode() per spec section 6.7.6
// followed by trailing-bit validation.
func parseMetadataTimecode(src []byte) (MetadataTimecode, error) {
	var tc MetadataTimecode
	r := bitstream.NewReader(src)
	v, err := r.ReadBits(5)
	if err != nil {
		return tc, err
	}
	tc.CountingType = uint8(v)
	if b, err := r.ReadBool(); err != nil {
		return tc, err
	} else {
		tc.FullTimestamp = b
	}
	if b, err := r.ReadBool(); err != nil {
		return tc, err
	} else {
		tc.Discontinuity = b
	}
	if b, err := r.ReadBool(); err != nil {
		return tc, err
	} else {
		tc.CountDropped = b
	}
	frames, err := r.ReadBits(9)
	if err != nil {
		return tc, err
	}
	tc.NFrames = uint16(frames)

	if tc.FullTimestamp {
		s, err := r.ReadBits(6)
		if err != nil {
			return tc, err
		}
		tc.Seconds = uint8(s)
		m, err := r.ReadBits(6)
		if err != nil {
			return tc, err
		}
		tc.Minutes = uint8(m)
		h, err := r.ReadBits(5)
		if err != nil {
			return tc, err
		}
		tc.Hours = uint8(h)
		tc.HasSeconds = true
		tc.HasMinutes = true
		tc.HasHours = true
	} else {
		secondsFlag, err := r.ReadBool()
		if err != nil {
			return tc, err
		}
		tc.SecondsFlag = secondsFlag
		if secondsFlag {
			s, err := r.ReadBits(6)
			if err != nil {
				return tc, err
			}
			tc.Seconds = uint8(s)
			tc.HasSeconds = true
			minutesFlag, err := r.ReadBool()
			if err != nil {
				return tc, err
			}
			if minutesFlag {
				m, err := r.ReadBits(6)
				if err != nil {
					return tc, err
				}
				tc.Minutes = uint8(m)
				tc.HasMinutes = true
				hoursFlag, err := r.ReadBool()
				if err != nil {
					return tc, err
				}
				if hoursFlag {
					h, err := r.ReadBits(5)
					if err != nil {
						return tc, err
					}
					tc.Hours = uint8(h)
					tc.HasHours = true
				}
			}
		}
	}

	offLen, err := r.ReadBits(5)
	if err != nil {
		return tc, err
	}
	tc.TimeOffsetLength = uint8(offLen)
	if tc.TimeOffsetLength > 0 {
		v, err := r.ReadBits(tc.TimeOffsetLength)
		if err != nil {
			return tc, err
		}
		tc.TimeOffsetValue = uint32(v)
	}
	if err := r.ReadTrailingBits(); err != nil {
		return tc, err
	}
	return tc, nil
}

// requireTrailing enforces the spec rule that the byte-aligned payload trailing
// region must contain a 0x80 byte followed only by zero padding. Empty trailing
// is rejected because the spec mandates one trailing bit.
func requireTrailing(src []byte) error {
	if len(src) == 0 {
		return ErrMetadataMissingTrailing
	}
	end := lastNonZeroIndex(src)
	if end < 0 {
		return ErrMetadataMissingTrailing
	}
	if src[end] != 0x80 {
		return ErrMetadataInvalidTrailing
	}
	return nil
}

// lastNonZeroIndex returns the index of the last nonzero byte in src, or -1 if
// src is all zero.
func lastNonZeroIndex(src []byte) int {
	for i := len(src) - 1; i >= 0; i-- {
		if src[i] != 0 {
			return i
		}
	}
	return -1
}

// lastNonZero returns the last nonzero byte in src, or zero if src is all zero.
func lastNonZero(src []byte) byte {
	i := lastNonZeroIndex(src)
	if i < 0 {
		return 0
	}
	return src[i]
}
