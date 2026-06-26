package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func defaultVideoColorConfig() SequenceColorConfig {
	return SequenceColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
}

func parserColorConfig(color SequenceColorConfig) parser.ColorConfig {
	bitDepth := color.BitDepth
	if bitDepth == 0 {
		bitDepth = 8
	}
	return parser.ColorConfig{
		BitDepth:                bitDepth,
		MonoChrome:              color.MonoChrome,
		ColorDescriptionPresent: color.ColorDescriptionPresent,
		ColorPrimaries:          color.ColorPrimaries,
		TransferCharacteristics: color.TransferCharacteristics,
		MatrixCoefficients:      color.MatrixCoefficients,
		ColorRange:              color.ColorRange,
		SubsamplingX:            color.SubsamplingX,
		SubsamplingY:            color.SubsamplingY,
		ChromaSamplePosition:    color.ChromaSamplePosition,
		SeparateUVDeltaQ:        color.SeparateUVDeltaQ,
	}
}

func videoColorProfile(color SequenceColorConfig) Profile {
	switch {
	case color.MonoChrome:
		if color.BitDepth == 12 {
			return Profile2
		}
		return Profile0
	case !color.SubsamplingX && !color.SubsamplingY:
		return Profile1
	case color.SubsamplingX && !color.SubsamplingY:
		return Profile2
	default:
		return Profile0
	}
}

func videoColorIs420(color parser.ColorConfig) bool {
	return !color.MonoChrome && color.BitDepth == 8 && color.SubsamplingX && color.SubsamplingY
}

func videoColorIs444(color parser.ColorConfig) bool {
	return !color.MonoChrome && color.BitDepth == 8 && !color.SubsamplingX && !color.SubsamplingY
}

func videoColorIs422(color parser.ColorConfig) bool {
	return !color.MonoChrome && color.BitDepth == 8 && color.SubsamplingX && !color.SubsamplingY
}

func videoColorSupported(color SequenceColorConfig) bool {
	pc := parserColorConfig(color)
	return videoColorIs420(pc) || videoColorIs422(pc) || videoColorIs444(pc)
}

func highBitDepthVideoColorSupported(profile Profile, color SequenceColorConfig) bool {
	pc := parserColorConfig(color)
	if pc.MonoChrome || (pc.BitDepth != 10 && pc.BitDepth != 12) {
		return false
	}
	if err := validateSequenceColorConfig(profile, color); err != nil {
		return false
	}
	return (pc.SubsamplingX && pc.SubsamplingY) ||
		(pc.SubsamplingX && !pc.SubsamplingY) ||
		(!pc.SubsamplingX && !pc.SubsamplingY)
}

func videoColorLabel(color parser.ColorConfig) string {
	switch {
	case color.MonoChrome:
		return "monochrome"
	case color.SubsamplingX && color.SubsamplingY:
		return "4:2:0"
	case color.SubsamplingX && !color.SubsamplingY:
		return "4:2:2"
	default:
		return "4:4:4"
	}
}

func videoColorSupportsInLoopFilters(color parser.ColorConfig) bool {
	return videoColorIs420(color)
}

func chromaShiftX(color parser.ColorConfig) int {
	if color.SubsamplingX {
		return 1
	}
	return 0
}

func chromaShiftY(color parser.ColorConfig) int {
	if color.SubsamplingY {
		return 1
	}
	return 0
}

func chromaWidthForColor(width int, color parser.ColorConfig) int {
	if color.MonoChrome {
		return 0
	}
	return (width + (1 << chromaShiftX(color)) - 1) >> chromaShiftX(color)
}

func chromaHeightForColor(height int, color parser.ColorConfig) int {
	if color.MonoChrome {
		return 0
	}
	return (height + (1 << chromaShiftY(color)) - 1) >> chromaShiftY(color)
}

func chromaXForColor(x int, color parser.ColorConfig) int {
	return x >> chromaShiftX(color)
}

func chromaYForColor(y int, color parser.ColorConfig) int {
	return y >> chromaShiftY(color)
}

func chromaX4ForColor(x4 uint8, color parser.ColorConfig) uint8 {
	return uint8(int(x4) >> chromaShiftX(color))
}

func chromaY4ForColor(y4 uint8, color parser.ColorConfig) uint8 {
	return uint8(int(y4) >> chromaShiftY(color))
}

func planeBlockPixels(size tile.BlockSize) (int, int, error) {
	dims, ok := size.Dimensions()
	if !ok {
		return 0, 0, tile.ErrInvalidDecodeState
	}
	return int(dims.W4) * 4, int(dims.H4) * 4, nil
}
