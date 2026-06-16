// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package rtp

import (
	"errors"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
)

const DependencyDescriptorMandatorySize = 3

const (
	DependencyDescriptorMaxSpatialLayers  = 4
	DependencyDescriptorMaxTemporalLayers = 8
	DependencyDescriptorMaxDecodeTargets  = 32
	DependencyDescriptorMaxTemplates      = 64
	DependencyDescriptorMaxFrameDiffs     = 32
)

// DependencyDescriptorDecodeTargetIndication is the 2-bit relation between a
// frame and a decode target in WebRTC's dependency descriptor extension.
type DependencyDescriptorDecodeTargetIndication uint8

const (
	DependencyDescriptorDecodeTargetNotPresent DependencyDescriptorDecodeTargetIndication = iota
	DependencyDescriptorDecodeTargetDiscardable
	DependencyDescriptorDecodeTargetSwitch
	DependencyDescriptorDecodeTargetRequired
)

func (d DependencyDescriptorDecodeTargetIndication) Valid() bool {
	return d <= DependencyDescriptorDecodeTargetRequired
}

// DependencyDescriptorMandatory is the fixed 24-bit prefix of WebRTC's RTP
// dependency descriptor header extension.
type DependencyDescriptorMandatory struct {
	FirstPacketInFrame bool
	LastPacketInFrame  bool
	TemplateID         uint8
	FrameNumber        uint16
}

// DependencyDescriptorResolution is one render-resolution entry from an
// attached dependency structure.
type DependencyDescriptorResolution struct {
	Width  uint16
	Height uint16
}

// DependencyDescriptorFrameDependencies describes the template-derived or
// custom dependencies for one frame.
type DependencyDescriptorFrameDependencies struct {
	SpatialID  uint8
	TemporalID uint8

	DTIs   [DependencyDescriptorMaxDecodeTargets]DependencyDescriptorDecodeTargetIndication
	DTINum uint8

	FrameDiffs   [DependencyDescriptorMaxFrameDiffs]uint16
	FrameDiffNum uint8

	ChainDiffs   [DependencyDescriptorMaxDecodeTargets]uint8
	ChainDiffNum uint8
}

// DependencyDescriptorStructure is the parsed template dependency structure.
// It is fixed-size so callers can keep receive-side structure state without
// allocations.
type DependencyDescriptorStructure struct {
	StructureID      uint8
	NumDecodeTargets uint8
	NumChains        uint8

	DecodeTargetProtectedByChain [DependencyDescriptorMaxDecodeTargets]uint8
	DecodeTargetSpatialID        [DependencyDescriptorMaxDecodeTargets]uint8
	DecodeTargetTemporalID       [DependencyDescriptorMaxDecodeTargets]uint8

	Templates   [DependencyDescriptorMaxTemplates]DependencyDescriptorFrameDependencies
	TemplateNum uint8

	Resolutions   [DependencyDescriptorMaxSpatialLayers]DependencyDescriptorResolution
	ResolutionNum uint8
}

// DependencyDescriptor is the full parsed WebRTC dependency descriptor RTP
// header extension.
type DependencyDescriptor struct {
	Mandatory DependencyDescriptorMandatory

	HasExtendedFields       bool
	HasAttachedStructure    bool
	HasCustomDTIs           bool
	HasCustomFrameDiffs     bool
	HasCustomChainDiffs     bool
	HasActiveDecodeTargets  bool
	ActiveDecodeTargetsMask uint32
	FrameDependencies       DependencyDescriptorFrameDependencies
	HasResolution           bool
	Resolution              DependencyDescriptorResolution
	AttachedStructure       DependencyDescriptorStructure
}

func PutDependencyDescriptorMandatory(dst []byte, descriptor DependencyDescriptorMandatory) (int, error) {
	if len(dst) < DependencyDescriptorMandatorySize {
		return 0, ErrShortBuffer
	}
	if descriptor.TemplateID >= DependencyDescriptorMaxTemplates {
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

// ParseDependencyDescriptor parses a full WebRTC dependency descriptor RTP
// header extension value. structure may be nil only when src carries an
// attached dependency structure; pass the last attached structure for compact
// descriptors that refer to prior templates.
func ParseDependencyDescriptor(src []byte, structure *DependencyDescriptorStructure) (DependencyDescriptor, int, error) {
	mandatory, consumed, err := ParseDependencyDescriptorMandatory(src)
	if err != nil {
		return DependencyDescriptor{}, 0, err
	}
	descriptor := DependencyDescriptor{
		Mandatory: mandatory,
	}
	if len(src) == DependencyDescriptorMandatorySize {
		if structure == nil {
			return DependencyDescriptor{}, 0, ErrInvalidDependencyDescriptor
		}
		if err := descriptor.fillFrameDependencyDefinition(structure, false, false, false, nil); err != nil {
			return DependencyDescriptor{}, 0, err
		}
		return descriptor, consumed, nil
	}

	r := bitstream.NewReader(src)
	if err := r.SkipBits(DependencyDescriptorMandatorySize * 8); err != nil {
		return DependencyDescriptor{}, 0, mapDependencyDescriptorBitstreamError(err)
	}

	attachedStructurePresent, err := readDependencyDescriptorBool(&r)
	if err != nil {
		return DependencyDescriptor{}, 0, err
	}
	activeDecodeTargetsPresent, err := readDependencyDescriptorBool(&r)
	if err != nil {
		return DependencyDescriptor{}, 0, err
	}
	customDTIs, err := readDependencyDescriptorBool(&r)
	if err != nil {
		return DependencyDescriptor{}, 0, err
	}
	customFrameDiffs, err := readDependencyDescriptorBool(&r)
	if err != nil {
		return DependencyDescriptor{}, 0, err
	}
	customChainDiffs, err := readDependencyDescriptorBool(&r)
	if err != nil {
		return DependencyDescriptor{}, 0, err
	}

	descriptor.HasExtendedFields = true
	descriptor.HasAttachedStructure = attachedStructurePresent
	descriptor.HasCustomDTIs = customDTIs
	descriptor.HasCustomFrameDiffs = customFrameDiffs
	descriptor.HasCustomChainDiffs = customChainDiffs

	activeStructure := structure
	if attachedStructurePresent {
		if err := parseDependencyDescriptorStructure(&r, &descriptor.AttachedStructure); err != nil {
			return DependencyDescriptor{}, 0, err
		}
		descriptor.HasActiveDecodeTargets = true
		descriptor.ActiveDecodeTargetsMask = dependencyDescriptorAllDecodeTargetsMask(descriptor.AttachedStructure.NumDecodeTargets)
		activeStructure = &descriptor.AttachedStructure
	}
	if activeStructure == nil {
		return DependencyDescriptor{}, 0, ErrInvalidDependencyDescriptor
	}
	if err := validateDependencyDescriptorStructure(*activeStructure); err != nil {
		return DependencyDescriptor{}, 0, err
	}
	if activeDecodeTargetsPresent {
		mask, err := readDependencyDescriptorBits(&r, activeStructure.NumDecodeTargets)
		if err != nil {
			return DependencyDescriptor{}, 0, err
		}
		descriptor.HasActiveDecodeTargets = true
		descriptor.ActiveDecodeTargetsMask = uint32(mask)
	}
	if err := descriptor.fillFrameDependencyDefinition(activeStructure, customDTIs, customFrameDiffs, customChainDiffs, &r); err != nil {
		return DependencyDescriptor{}, 0, err
	}
	consumed = (r.BitsRead() + 7) >> 3
	if consumed > len(src) {
		return DependencyDescriptor{}, 0, ErrShortPayload
	}
	return descriptor, consumed, nil
}

func parseDependencyDescriptorStructure(r *bitstream.Reader, structure *DependencyDescriptorStructure) error {
	structureID, err := readDependencyDescriptorBits(r, 6)
	if err != nil {
		return err
	}
	dtCntMinusOne, err := readDependencyDescriptorBits(r, 5)
	if err != nil {
		return err
	}
	structure.StructureID = uint8(structureID)
	structure.NumDecodeTargets = uint8(dtCntMinusOne + 1)
	if structure.NumDecodeTargets == 0 || structure.NumDecodeTargets > DependencyDescriptorMaxDecodeTargets {
		return ErrInvalidDependencyDescriptor
	}
	if err := parseDependencyDescriptorTemplateLayers(r, structure); err != nil {
		return err
	}
	if err := parseDependencyDescriptorTemplateDTIs(r, structure); err != nil {
		return err
	}
	if err := parseDependencyDescriptorTemplateFrameDiffs(r, structure); err != nil {
		return err
	}
	if err := parseDependencyDescriptorTemplateChains(r, structure); err != nil {
		return err
	}
	fillDependencyDescriptorDecodeTargetLayers(structure)

	resolutionsPresent, err := readDependencyDescriptorBool(r)
	if err != nil {
		return err
	}
	if resolutionsPresent {
		if err := parseDependencyDescriptorResolutions(r, structure); err != nil {
			return err
		}
	}
	return validateDependencyDescriptorStructure(*structure)
}

func parseDependencyDescriptorTemplateLayers(r *bitstream.Reader, structure *DependencyDescriptorStructure) error {
	var spatialID uint8
	var temporalID uint8
	for {
		if structure.TemplateNum >= DependencyDescriptorMaxTemplates {
			return ErrInvalidDependencyDescriptor
		}
		template := &structure.Templates[structure.TemplateNum]
		template.SpatialID = spatialID
		template.TemporalID = temporalID
		structure.TemplateNum++

		nextLayerIDC, err := readDependencyDescriptorBits(r, 2)
		if err != nil {
			return err
		}
		switch nextLayerIDC {
		case 0:
		case 1:
			temporalID++
			if temporalID >= DependencyDescriptorMaxTemporalLayers {
				return ErrInvalidDependencyDescriptor
			}
		case 2:
			temporalID = 0
			spatialID++
			if spatialID >= DependencyDescriptorMaxSpatialLayers {
				return ErrInvalidDependencyDescriptor
			}
		case 3:
			return nil
		default:
			return ErrInvalidDependencyDescriptor
		}
	}
}

func parseDependencyDescriptorTemplateDTIs(r *bitstream.Reader, structure *DependencyDescriptorStructure) error {
	for i := uint8(0); i < structure.TemplateNum; i++ {
		template := &structure.Templates[i]
		template.DTINum = structure.NumDecodeTargets
		for j := uint8(0); j < structure.NumDecodeTargets; j++ {
			dti, err := readDependencyDescriptorBits(r, 2)
			if err != nil {
				return err
			}
			template.DTIs[j] = DependencyDescriptorDecodeTargetIndication(dti)
		}
	}
	return nil
}

func parseDependencyDescriptorTemplateFrameDiffs(r *bitstream.Reader, structure *DependencyDescriptorStructure) error {
	for i := uint8(0); i < structure.TemplateNum; i++ {
		template := &structure.Templates[i]
		for {
			follows, err := readDependencyDescriptorBool(r)
			if err != nil {
				return err
			}
			if !follows {
				break
			}
			if template.FrameDiffNum >= DependencyDescriptorMaxFrameDiffs {
				return ErrInvalidDependencyDescriptor
			}
			diffMinusOne, err := readDependencyDescriptorBits(r, 4)
			if err != nil {
				return err
			}
			template.FrameDiffs[template.FrameDiffNum] = uint16(diffMinusOne + 1)
			template.FrameDiffNum++
		}
	}
	return nil
}

func parseDependencyDescriptorTemplateChains(r *bitstream.Reader, structure *DependencyDescriptorStructure) error {
	chains, err := readDependencyDescriptorNonSymmetric(r, uint32(structure.NumDecodeTargets)+1)
	if err != nil {
		return err
	}
	if chains > uint32(structure.NumDecodeTargets) || chains > DependencyDescriptorMaxDecodeTargets {
		return ErrInvalidDependencyDescriptor
	}
	structure.NumChains = uint8(chains)
	if structure.NumChains == 0 {
		return nil
	}
	for i := uint8(0); i < structure.NumDecodeTargets; i++ {
		chain, err := readDependencyDescriptorNonSymmetric(r, uint32(structure.NumChains))
		if err != nil {
			return err
		}
		structure.DecodeTargetProtectedByChain[i] = uint8(chain)
	}
	for i := uint8(0); i < structure.TemplateNum; i++ {
		template := &structure.Templates[i]
		template.ChainDiffNum = structure.NumChains
		for j := uint8(0); j < structure.NumChains; j++ {
			diff, err := readDependencyDescriptorBits(r, 4)
			if err != nil {
				return err
			}
			template.ChainDiffs[j] = uint8(diff)
		}
	}
	return nil
}

func fillDependencyDescriptorDecodeTargetLayers(structure *DependencyDescriptorStructure) {
	for target := uint8(0); target < structure.NumDecodeTargets; target++ {
		var spatialID uint8
		var temporalID uint8
		for templateIndex := uint8(0); templateIndex < structure.TemplateNum; templateIndex++ {
			template := structure.Templates[templateIndex]
			if template.DTIs[target] == DependencyDescriptorDecodeTargetNotPresent {
				continue
			}
			if template.SpatialID > spatialID {
				spatialID = template.SpatialID
			}
			if template.TemporalID > temporalID {
				temporalID = template.TemporalID
			}
		}
		structure.DecodeTargetSpatialID[target] = spatialID
		structure.DecodeTargetTemporalID[target] = temporalID
	}
}

func parseDependencyDescriptorResolutions(r *bitstream.Reader, structure *DependencyDescriptorStructure) error {
	maxSpatialID := uint8(0)
	for i := uint8(0); i < structure.TemplateNum; i++ {
		if structure.Templates[i].SpatialID > maxSpatialID {
			maxSpatialID = structure.Templates[i].SpatialID
		}
	}
	resolutionNum := maxSpatialID + 1
	if resolutionNum > DependencyDescriptorMaxSpatialLayers {
		return ErrInvalidDependencyDescriptor
	}
	for i := uint8(0); i < resolutionNum; i++ {
		widthMinusOne, err := readDependencyDescriptorBits(r, 16)
		if err != nil {
			return err
		}
		heightMinusOne, err := readDependencyDescriptorBits(r, 16)
		if err != nil {
			return err
		}
		structure.Resolutions[i] = DependencyDescriptorResolution{
			Width:  uint16(widthMinusOne + 1),
			Height: uint16(heightMinusOne + 1),
		}
	}
	structure.ResolutionNum = resolutionNum
	return nil
}

func (descriptor *DependencyDescriptor) fillFrameDependencyDefinition(structure *DependencyDescriptorStructure, customDTIs bool, customFrameDiffs bool, customChainDiffs bool, r *bitstream.Reader) error {
	if err := validateDependencyDescriptorStructure(*structure); err != nil {
		return err
	}
	templateIndex := (uint16(descriptor.Mandatory.TemplateID) + DependencyDescriptorMaxTemplates - uint16(structure.StructureID)) % DependencyDescriptorMaxTemplates
	if templateIndex >= uint16(structure.TemplateNum) {
		return ErrInvalidDependencyDescriptor
	}
	descriptor.FrameDependencies = structure.Templates[templateIndex]
	if customDTIs {
		if r == nil {
			return ErrInvalidDependencyDescriptor
		}
		if err := parseDependencyDescriptorFrameDTIs(r, structure.NumDecodeTargets, &descriptor.FrameDependencies); err != nil {
			return err
		}
	}
	if customFrameDiffs {
		if r == nil {
			return ErrInvalidDependencyDescriptor
		}
		if err := parseDependencyDescriptorFrameDiffs(r, &descriptor.FrameDependencies); err != nil {
			return err
		}
	}
	if customChainDiffs {
		if r == nil {
			return ErrInvalidDependencyDescriptor
		}
		if err := parseDependencyDescriptorFrameChains(r, structure.NumChains, &descriptor.FrameDependencies); err != nil {
			return err
		}
	}
	if structure.ResolutionNum != 0 {
		if descriptor.FrameDependencies.SpatialID >= structure.ResolutionNum {
			return ErrInvalidDependencyDescriptor
		}
		descriptor.HasResolution = true
		descriptor.Resolution = structure.Resolutions[descriptor.FrameDependencies.SpatialID]
	}
	return nil
}

func parseDependencyDescriptorFrameDTIs(r *bitstream.Reader, count uint8, frame *DependencyDescriptorFrameDependencies) error {
	frame.DTINum = count
	for i := uint8(0); i < count; i++ {
		dti, err := readDependencyDescriptorBits(r, 2)
		if err != nil {
			return err
		}
		frame.DTIs[i] = DependencyDescriptorDecodeTargetIndication(dti)
	}
	return nil
}

func parseDependencyDescriptorFrameDiffs(r *bitstream.Reader, frame *DependencyDescriptorFrameDependencies) error {
	frame.FrameDiffNum = 0
	for {
		size, err := readDependencyDescriptorBits(r, 2)
		if err != nil {
			return err
		}
		if size == 0 {
			return nil
		}
		if frame.FrameDiffNum >= DependencyDescriptorMaxFrameDiffs {
			return ErrInvalidDependencyDescriptor
		}
		diffMinusOne, err := readDependencyDescriptorBits(r, uint8(4*size))
		if err != nil {
			return err
		}
		frame.FrameDiffs[frame.FrameDiffNum] = uint16(diffMinusOne + 1)
		frame.FrameDiffNum++
	}
}

func parseDependencyDescriptorFrameChains(r *bitstream.Reader, count uint8, frame *DependencyDescriptorFrameDependencies) error {
	frame.ChainDiffNum = count
	for i := uint8(0); i < count; i++ {
		diff, err := readDependencyDescriptorBits(r, 8)
		if err != nil {
			return err
		}
		frame.ChainDiffs[i] = uint8(diff)
	}
	return nil
}

func validateDependencyDescriptorStructure(structure DependencyDescriptorStructure) error {
	if structure.TemplateNum == 0 ||
		structure.TemplateNum > DependencyDescriptorMaxTemplates ||
		structure.NumDecodeTargets == 0 ||
		structure.NumDecodeTargets > DependencyDescriptorMaxDecodeTargets ||
		structure.NumChains > structure.NumDecodeTargets ||
		structure.StructureID >= DependencyDescriptorMaxTemplates ||
		structure.ResolutionNum > DependencyDescriptorMaxSpatialLayers {
		return ErrInvalidDependencyDescriptor
	}
	for i := uint8(0); i < structure.TemplateNum; i++ {
		template := structure.Templates[i]
		if template.SpatialID >= DependencyDescriptorMaxSpatialLayers ||
			template.TemporalID >= DependencyDescriptorMaxTemporalLayers ||
			template.DTINum != structure.NumDecodeTargets ||
			template.FrameDiffNum > DependencyDescriptorMaxFrameDiffs ||
			template.ChainDiffNum != structure.NumChains {
			return ErrInvalidDependencyDescriptor
		}
		for j := uint8(0); j < template.DTINum; j++ {
			if !template.DTIs[j].Valid() {
				return ErrInvalidDependencyDescriptor
			}
		}
		for j := uint8(0); j < template.FrameDiffNum; j++ {
			if template.FrameDiffs[j] == 0 {
				return ErrInvalidDependencyDescriptor
			}
		}
		if i == 0 {
			if template.SpatialID != 0 || template.TemporalID != 0 {
				return ErrInvalidDependencyDescriptor
			}
			continue
		}
		if !dependencyDescriptorValidNextTemplateLayer(structure.Templates[i-1], template) {
			return ErrInvalidDependencyDescriptor
		}
	}
	for i := uint8(0); i < structure.NumDecodeTargets; i++ {
		if structure.NumChains != 0 && structure.DecodeTargetProtectedByChain[i] >= structure.NumChains {
			return ErrInvalidDependencyDescriptor
		}
		if structure.DecodeTargetSpatialID[i] >= DependencyDescriptorMaxSpatialLayers ||
			structure.DecodeTargetTemporalID[i] >= DependencyDescriptorMaxTemporalLayers {
			return ErrInvalidDependencyDescriptor
		}
	}
	for i := uint8(0); i < structure.ResolutionNum; i++ {
		if structure.Resolutions[i].Width == 0 || structure.Resolutions[i].Height == 0 {
			return ErrInvalidDependencyDescriptor
		}
	}
	return nil
}

func dependencyDescriptorValidNextTemplateLayer(previous DependencyDescriptorFrameDependencies, next DependencyDescriptorFrameDependencies) bool {
	if next.SpatialID == previous.SpatialID && next.TemporalID == previous.TemporalID {
		return true
	}
	if next.SpatialID == previous.SpatialID && next.TemporalID == previous.TemporalID+1 {
		return true
	}
	return next.SpatialID == previous.SpatialID+1 && next.TemporalID == 0
}

func readDependencyDescriptorBool(r *bitstream.Reader) (bool, error) {
	value, err := r.ReadBool()
	if err != nil {
		return false, mapDependencyDescriptorBitstreamError(err)
	}
	return value, nil
}

func readDependencyDescriptorBits(r *bitstream.Reader, n uint8) (uint64, error) {
	value, err := r.ReadBits(n)
	if err != nil {
		return 0, mapDependencyDescriptorBitstreamError(err)
	}
	return value, nil
}

func readDependencyDescriptorNonSymmetric(r *bitstream.Reader, numValues uint32) (uint32, error) {
	if numValues == 0 || numValues > 1<<31 {
		return 0, ErrInvalidDependencyDescriptor
	}
	if numValues == 1 {
		return 0, nil
	}
	width := uint8(32 - dependencyDescriptorLeadingZeros32(numValues-1))
	numMinBitsValues := (uint32(1) << width) - numValues
	value, err := readDependencyDescriptorBits(r, width-1)
	if err != nil {
		return 0, err
	}
	if value < uint64(numMinBitsValues) {
		return uint32(value), nil
	}
	bit, err := readDependencyDescriptorBits(r, 1)
	if err != nil {
		return 0, err
	}
	out := (uint32(value) << 1) + uint32(bit) - numMinBitsValues
	if out >= numValues {
		return 0, ErrInvalidDependencyDescriptor
	}
	return out, nil
}

func dependencyDescriptorLeadingZeros32(value uint32) uint8 {
	var zeros uint8
	for bit := uint32(1 << 31); bit != 0 && value&bit == 0; bit >>= 1 {
		zeros++
	}
	return zeros
}

func dependencyDescriptorAllDecodeTargetsMask(count uint8) uint32 {
	if count >= 32 {
		return ^uint32(0)
	}
	return uint32(1<<count) - 1
}

func mapDependencyDescriptorBitstreamError(err error) error {
	switch {
	case errors.Is(err, bitstream.ErrNotEnoughBits):
		return ErrShortPayload
	case errors.Is(err, bitstream.ErrInvalidBitCount):
		return ErrInvalidDependencyDescriptor
	default:
		return err
	}
}
