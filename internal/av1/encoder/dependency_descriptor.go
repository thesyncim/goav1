package encoder

import "github.com/thesyncim/goav1/internal/av1/bitstream"

type webRTCDependencyDescriptorMatch struct {
	templateIndex   uint8
	needCustomDTIs  bool
	needCustomDiffs bool
	frameDiffs      [WebRTCMaxFrameReferences]uint16
	frameDiffNum    uint8
}

func WebRTCDependencyDescriptorSize(structure WebRTCFrameDependencyStructure, info WebRTCGenericFrameInfo, attachStructure bool) (int, error) {
	match, err := webRTCDependencyDescriptorMatchFrame(structure, info)
	if err != nil {
		return 0, err
	}
	bits := 24
	if attachStructure || match.needCustomDTIs || match.needCustomDiffs {
		bits += 5
		if attachStructure {
			structureBits, err := webRTCDependencyStructureBits(structure)
			if err != nil {
				return 0, err
			}
			bits += structureBits
		}
		if match.needCustomDTIs {
			bits += 2 * int(structure.NumDecodeTargets)
		}
		if match.needCustomDiffs {
			bits += webRTCFrameDiffsBits(match.frameDiffs, match.frameDiffNum)
		}
	}
	return (bits + 7) >> 3, nil
}

func AppendWebRTCDependencyDescriptor(dst []byte, structure WebRTCFrameDependencyStructure, info WebRTCGenericFrameInfo, firstPacketInFrame bool, lastPacketInFrame bool, attachStructure bool) ([]byte, error) {
	size, err := WebRTCDependencyDescriptorSize(structure, info, attachStructure)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < size {
		return dst, bitstream.ErrShortBuffer
	}

	off := len(dst)
	out := dst[:off+size]
	w := newBitWriter(out[off:])
	if err := writeWebRTCDependencyDescriptor(&w, structure, info, firstPacketInFrame, lastPacketInFrame, attachStructure); err != nil {
		return dst, err
	}
	for !w.byteAligned() {
		if err := w.writeBit(0); err != nil {
			return dst, err
		}
	}
	return out, nil
}

func writeWebRTCDependencyDescriptor(w *bitWriter, structure WebRTCFrameDependencyStructure, info WebRTCGenericFrameInfo, firstPacketInFrame bool, lastPacketInFrame bool, attachStructure bool) error {
	match, err := webRTCDependencyDescriptorMatchFrame(structure, info)
	if err != nil {
		return err
	}
	templateID := (structure.StructureID + match.templateIndex) % WebRTCRtpDependencyMaxTemplates
	if err := w.writeBool(firstPacketInFrame); err != nil {
		return err
	}
	if err := w.writeBool(lastPacketInFrame); err != nil {
		return err
	}
	if err := w.writeBits(uint64(templateID), 6); err != nil {
		return err
	}
	if err := w.writeBits(uint64(uint16(info.FrameID)), 16); err != nil {
		return err
	}
	if !attachStructure && !match.needCustomDTIs && !match.needCustomDiffs {
		return nil
	}
	if err := w.writeBool(attachStructure); err != nil {
		return err
	}
	if err := w.writeBool(false); err != nil {
		return err
	}
	if err := w.writeBool(match.needCustomDTIs); err != nil {
		return err
	}
	if err := w.writeBool(match.needCustomDiffs); err != nil {
		return err
	}
	if err := w.writeBool(false); err != nil {
		return err
	}
	if attachStructure {
		if err := writeWebRTCDependencyStructure(w, structure); err != nil {
			return err
		}
	}
	if match.needCustomDTIs {
		for i := uint8(0); i < structure.NumDecodeTargets; i++ {
			if err := w.writeBits(uint64(info.DTIs[i]), 2); err != nil {
				return err
			}
		}
	}
	if match.needCustomDiffs {
		if err := writeWebRTCFrameDiffs(w, match.frameDiffs, match.frameDiffNum); err != nil {
			return err
		}
	}
	return nil
}

func webRTCDependencyDescriptorMatchFrame(structure WebRTCFrameDependencyStructure, info WebRTCGenericFrameInfo) (webRTCDependencyDescriptorMatch, error) {
	if err := validateWebRTCDependencyStructure(structure); err != nil {
		return webRTCDependencyDescriptorMatch{}, err
	}
	if info.DTINum != structure.NumDecodeTargets || info.DependencyNum > WebRTCMaxFrameReferences {
		return webRTCDependencyDescriptorMatch{}, ErrInvalidFrame
	}

	var match webRTCDependencyDescriptorMatch
	found := false
	for i := uint8(0); i < structure.TemplateNum; i++ {
		template := structure.Templates[i]
		if template.SpatialID == info.SpatialID && template.TemporalID == info.TemporalID {
			match.templateIndex = i
			found = true
			break
		}
	}
	if !found {
		return webRTCDependencyDescriptorMatch{}, ErrInvalidFrame
	}

	template := structure.Templates[match.templateIndex]
	for i := uint8(0); i < structure.NumDecodeTargets; i++ {
		if info.DTIs[i] != template.DTIs[i] {
			match.needCustomDTIs = true
			break
		}
	}
	match.frameDiffNum = info.DependencyNum
	for i := uint8(0); i < info.DependencyNum; i++ {
		if info.Dependencies[i] >= info.FrameID {
			return webRTCDependencyDescriptorMatch{}, ErrInvalidFrame
		}
		diff := info.FrameID - info.Dependencies[i]
		if diff == 0 || diff > 1<<12 {
			return webRTCDependencyDescriptorMatch{}, ErrInvalidFrame
		}
		match.frameDiffs[i] = uint16(diff)
	}
	if template.FrameDiffNum != match.frameDiffNum {
		match.needCustomDiffs = true
	} else {
		for i := uint8(0); i < match.frameDiffNum; i++ {
			if template.FrameDiffs[i] != match.frameDiffs[i] {
				match.needCustomDiffs = true
				break
			}
		}
	}
	return match, nil
}

func validateWebRTCDependencyStructure(structure WebRTCFrameDependencyStructure) error {
	if structure.TemplateNum == 0 || structure.TemplateNum > WebRTCRtpDependencyMaxTemplates ||
		structure.NumDecodeTargets == 0 || structure.NumDecodeTargets > WebRTCRtpDependencyMaxDecodeTargets ||
		structure.NumChains > structure.NumDecodeTargets ||
		structure.StructureID >= WebRTCRtpDependencyMaxTemplates ||
		structure.ResolutionNum > WebRTCMaxSpatialLayers {
		return ErrInvalidFrame
	}
	for i := uint8(0); i < structure.TemplateNum; i++ {
		template := structure.Templates[i]
		if template.SpatialID >= WebRTCMaxSpatialLayers || template.TemporalID >= WebRTCMaxTemporalLayers ||
			template.DTINum != structure.NumDecodeTargets ||
			template.FrameDiffNum > WebRTCMaxFrameReferences ||
			template.ChainDiffNum != structure.NumChains {
			return ErrInvalidFrame
		}
		if i == 0 {
			if template.SpatialID != 0 || template.TemporalID != 0 {
				return ErrInvalidFrame
			}
			continue
		}
		if webRTCNextLayerIDC(structure.Templates[i-1], template) > 3 {
			return ErrInvalidFrame
		}
	}
	for i := uint8(0); i < structure.NumDecodeTargets; i++ {
		if structure.NumChains != 0 && structure.DecodeTargetProtectedByChain[i] >= structure.NumChains {
			return ErrInvalidFrame
		}
	}
	for i := uint8(0); i < structure.ResolutionNum; i++ {
		if !structure.Resolutions[i].Valid() || structure.Resolutions[i].Width > 1<<16 || structure.Resolutions[i].Height > 1<<16 {
			return ErrInvalidFrame
		}
	}
	return nil
}

func webRTCDependencyStructureBits(structure WebRTCFrameDependencyStructure) (int, error) {
	if err := validateWebRTCDependencyStructure(structure); err != nil {
		return 0, err
	}
	bits := 11
	bits += 2 * int(structure.TemplateNum)
	bits += 2 * int(structure.TemplateNum) * int(structure.NumDecodeTargets)
	bits += int(structure.TemplateNum)
	for i := uint8(0); i < structure.TemplateNum; i++ {
		bits += 5 * int(structure.Templates[i].FrameDiffNum)
	}
	n, err := webRTCNonSymmetricBits(uint32(structure.NumChains), uint32(structure.NumDecodeTargets)+1)
	if err != nil {
		return 0, err
	}
	bits += n
	if structure.NumChains != 0 {
		for i := uint8(0); i < structure.NumDecodeTargets; i++ {
			n, err = webRTCNonSymmetricBits(uint32(structure.DecodeTargetProtectedByChain[i]), uint32(structure.NumChains))
			if err != nil {
				return 0, err
			}
			bits += n
		}
		bits += 4 * int(structure.TemplateNum) * int(structure.NumChains)
	}
	bits += 1 + 32*int(structure.ResolutionNum)
	return bits, nil
}

func writeWebRTCDependencyStructure(w *bitWriter, structure WebRTCFrameDependencyStructure) error {
	if err := validateWebRTCDependencyStructure(structure); err != nil {
		return err
	}
	if err := w.writeBits(uint64(structure.StructureID), 6); err != nil {
		return err
	}
	if err := w.writeBits(uint64(structure.NumDecodeTargets-1), 5); err != nil {
		return err
	}
	for i := uint8(1); i < structure.TemplateNum; i++ {
		if err := w.writeBits(uint64(webRTCNextLayerIDC(structure.Templates[i-1], structure.Templates[i])), 2); err != nil {
			return err
		}
	}
	if err := w.writeBits(3, 2); err != nil {
		return err
	}
	for i := uint8(0); i < structure.TemplateNum; i++ {
		template := structure.Templates[i]
		for j := uint8(0); j < structure.NumDecodeTargets; j++ {
			if err := w.writeBits(uint64(template.DTIs[j]), 2); err != nil {
				return err
			}
		}
	}
	for i := uint8(0); i < structure.TemplateNum; i++ {
		template := structure.Templates[i]
		for j := uint8(0); j < template.FrameDiffNum; j++ {
			if err := w.writeBits(uint64(0x10|((template.FrameDiffs[j]-1)&0xf)), 5); err != nil {
				return err
			}
		}
		if err := w.writeBit(0); err != nil {
			return err
		}
	}
	if err := writeWebRTCNonSymmetric(w, uint32(structure.NumChains), uint32(structure.NumDecodeTargets)+1); err != nil {
		return err
	}
	if structure.NumChains != 0 {
		for i := uint8(0); i < structure.NumDecodeTargets; i++ {
			if err := writeWebRTCNonSymmetric(w, uint32(structure.DecodeTargetProtectedByChain[i]), uint32(structure.NumChains)); err != nil {
				return err
			}
		}
		for i := uint8(0); i < structure.TemplateNum; i++ {
			template := structure.Templates[i]
			for j := uint8(0); j < structure.NumChains; j++ {
				if err := w.writeBits(uint64(template.ChainDiffs[j]), 4); err != nil {
					return err
				}
			}
		}
	}
	if err := w.writeBool(structure.ResolutionNum != 0); err != nil {
		return err
	}
	for i := uint8(0); i < structure.ResolutionNum; i++ {
		resolution := structure.Resolutions[i]
		if err := w.writeBits(uint64(resolution.Width-1), 16); err != nil {
			return err
		}
		if err := w.writeBits(uint64(resolution.Height-1), 16); err != nil {
			return err
		}
	}
	return nil
}

func webRTCNextLayerIDC(previous WebRTCFrameDependencyTemplate, next WebRTCFrameDependencyTemplate) uint8 {
	if next.SpatialID == previous.SpatialID && next.TemporalID == previous.TemporalID {
		return 0
	}
	if next.SpatialID == previous.SpatialID && next.TemporalID == previous.TemporalID+1 {
		return 1
	}
	if next.SpatialID == previous.SpatialID+1 && next.TemporalID == 0 {
		return 2
	}
	return 4
}

func webRTCFrameDiffsBits(diffs [WebRTCMaxFrameReferences]uint16, count uint8) int {
	bits := 2
	for i := uint8(0); i < count; i++ {
		bits += 2
		switch {
		case diffs[i] <= 1<<4:
			bits += 4
		case diffs[i] <= 1<<8:
			bits += 8
		default:
			bits += 12
		}
	}
	return bits
}

func writeWebRTCFrameDiffs(w *bitWriter, diffs [WebRTCMaxFrameReferences]uint16, count uint8) error {
	for i := uint8(0); i < count; i++ {
		diff := diffs[i]
		switch {
		case diff == 0:
			return ErrInvalidFrame
		case diff <= 1<<4:
			if err := w.writeBits(uint64(0x10|((diff-1)&0xf)), 6); err != nil {
				return err
			}
		case diff <= 1<<8:
			if err := w.writeBits(uint64(0x200|((diff-1)&0xff)), 10); err != nil {
				return err
			}
		case diff <= 1<<12:
			if err := w.writeBits(uint64(0x3000|((diff-1)&0xfff)), 14); err != nil {
				return err
			}
		default:
			return ErrInvalidFrame
		}
	}
	return w.writeBits(0, 2)
}

func webRTCNonSymmetricBits(value uint32, numValues uint32) (int, error) {
	if numValues == 1 {
		if value == 0 {
			return 0, nil
		}
		return 0, ErrInvalidFrame
	}
	if numValues == 0 || value >= numValues {
		return 0, ErrInvalidFrame
	}
	width := uint8(32 - leadingZeros32(numValues-1))
	numMinBitsValues := (uint32(1) << width) - numValues
	if value < numMinBitsValues {
		return int(width - 1), nil
	}
	return int(width), nil
}

func writeWebRTCNonSymmetric(w *bitWriter, value uint32, numValues uint32) error {
	if numValues == 1 {
		if value == 0 {
			return nil
		}
		return ErrInvalidFrame
	}
	if numValues == 0 || value >= numValues {
		return ErrInvalidFrame
	}
	width := uint8(32 - leadingZeros32(numValues-1))
	numMinBitsValues := (uint32(1) << width) - numValues
	if value < numMinBitsValues {
		return w.writeBits(uint64(value), width-1)
	}
	return w.writeBits(uint64(value+numMinBitsValues), width)
}

func leadingZeros32(value uint32) uint8 {
	var zeros uint8
	for bit := uint32(1 << 31); bit != 0 && value&bit == 0; bit >>= 1 {
		zeros++
	}
	return zeros
}
