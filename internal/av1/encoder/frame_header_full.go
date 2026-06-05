package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// IntraFrameHeaderParams is a complete key/intra-only uncompressed_header()
// through film_grain_params(). Tile-group payload syntax is intentionally not
// part of this structure.
type IntraFrameHeaderParams struct {
	Prefix           FrameHeaderPrefix
	Size             IntraFrameSize
	Tile             TileInfo
	Quantization     QuantizationParams
	Segmentation     SegmentationParams
	Delta            DeltaParams
	LoopFilter       LoopFilterParams
	CDEF             CDEFParams
	Restoration      RestorationParams
	TransformRef     TransformReferenceParams
	FrameMode        FrameModeParams
	FilmGrain        FilmGrainParams
	AllLossless      bool
	PreviousLFDeltas *LoopFilterDeltas
}

// InterFrameHeaderParams is a complete inter/switch uncompressed_header()
// through film_grain_params(). Tile-group payload syntax is intentionally not
// part of this structure.
type InterFrameHeaderParams struct {
	Prefix           FrameHeaderPrefix
	Size             InterFrameSize
	Tile             TileInfo
	Quantization     QuantizationParams
	Segmentation     SegmentationParams
	Delta            DeltaParams
	LoopFilter       LoopFilterParams
	CDEF             CDEFParams
	Restoration      RestorationParams
	TransformRef     TransformReferenceParams
	SkipMode         SkipModeParams
	FrameMode        FrameModeParams
	GlobalMotion     GlobalMotionParams
	FilmGrain        FilmGrainParams
	AllLossless      bool
	References       *parser.ReferenceState
	PreviousLFDeltas *LoopFilterDeltas
}

func IntraFrameHeaderPayloadSize(seq SequenceHeader, header IntraFrameHeaderParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeIntraFrameHeaderPayload(&w, seq, header); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendIntraFrameHeaderPayload(dst []byte, seq SequenceHeader, header IntraFrameHeaderParams) ([]byte, error) {
	payloadSize, err := IntraFrameHeaderPayloadSize(seq, header)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeIntraFrameHeaderPayload(&w, seq, header); err != nil {
		return dst, err
	}
	return out, nil
}

func InterFrameHeaderPayloadSize(seq SequenceHeader, header InterFrameHeaderParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeInterFrameHeaderPayload(&w, seq, header); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendInterFrameHeaderPayload(dst []byte, seq SequenceHeader, header InterFrameHeaderParams) ([]byte, error) {
	payloadSize, err := InterFrameHeaderPayloadSize(seq, header)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeInterFrameHeaderPayload(&w, seq, header); err != nil {
		return dst, err
	}
	return out, nil
}

func writeIntraFrameHeaderPayload(w *bitWriter, seq SequenceHeader, header IntraFrameHeaderParams) error {
	if err := writeFrameHeaderPrefixPayload(w, seq, header.Prefix); err != nil {
		return err
	}
	if err := writeIntraFrameSizePayload(w, seq, header.Prefix, header.Size); err != nil {
		return err
	}
	if err := writeTileInfoPayload(w, seq, header.Prefix, codedWidthFromIntraFrameSize(header.Size), header.Size.Height, header.Tile); err != nil {
		return err
	}
	if err := writeQuantizationParamsPayload(w, seq, header.Quantization); err != nil {
		return err
	}
	if err := writeSegmentationParamsPayload(w, header.Prefix, header.Segmentation); err != nil {
		return err
	}
	if err := writeDeltaParamsPayload(w, header.Size.AllowIntrabc, header.Quantization, header.Delta); err != nil {
		return err
	}
	if err := writeLoopFilterParamsPayload(w, seq, header.Prefix, header.Size.AllowIntrabc, header.AllLossless, header.LoopFilter, header.PreviousLFDeltas); err != nil {
		return err
	}
	if err := writeCDEFParamsPayload(w, seq, header.Size.AllowIntrabc, header.AllLossless, header.CDEF); err != nil {
		return err
	}
	if err := writeRestorationParamsPayload(w, seq, header.Size.AllowIntrabc, header.Size.SuperResDenominator != 8, header.AllLossless, header.Restoration); err != nil {
		return err
	}
	if err := writeTransformReferenceParamsPayload(w, header.Prefix, header.AllLossless, header.TransformRef); err != nil {
		return err
	}
	if err := writeFrameModeParamsPayload(w, seq, header.Prefix, header.FrameMode); err != nil {
		return err
	}
	if err := writeGlobalMotionParamsPayload(w, header.Prefix, InterFrameSize{}, TileInfo{}, nil, DefaultGlobalMotionParams()); err != nil {
		return err
	}
	return writeFilmGrainParamsPayload(w, seq, header.Prefix, InterFrameSize{}, nil, header.FilmGrain)
}

func writeInterFrameHeaderPayload(w *bitWriter, seq SequenceHeader, header InterFrameHeaderParams) error {
	if err := writeFrameHeaderPrefixPayload(w, seq, header.Prefix); err != nil {
		return err
	}
	if err := writeInterFrameSizePayload(w, seq, header.Prefix, header.Size); err != nil {
		return err
	}
	if err := writeTileInfoPayload(w, seq, header.Prefix, codedWidthFromInterFrameSize(header.Size), header.Size.Height, header.Tile); err != nil {
		return err
	}
	if err := writeQuantizationParamsPayload(w, seq, header.Quantization); err != nil {
		return err
	}
	if err := writeSegmentationParamsPayload(w, header.Prefix, header.Segmentation); err != nil {
		return err
	}
	if err := writeDeltaParamsPayload(w, false, header.Quantization, header.Delta); err != nil {
		return err
	}
	if err := writeLoopFilterParamsPayload(w, seq, header.Prefix, false, header.AllLossless, header.LoopFilter, header.PreviousLFDeltas); err != nil {
		return err
	}
	if err := writeCDEFParamsPayload(w, seq, false, header.AllLossless, header.CDEF); err != nil {
		return err
	}
	if err := writeRestorationParamsPayload(w, seq, false, header.Size.SuperResDenominator != 8, header.AllLossless, header.Restoration); err != nil {
		return err
	}
	if err := writeTransformReferenceParamsPayload(w, header.Prefix, header.AllLossless, header.TransformRef); err != nil {
		return err
	}
	if err := writeSkipModeParamsPayload(w, seq, header.Prefix, header.Size, header.References, header.TransformRef, header.SkipMode); err != nil {
		return err
	}
	if err := writeFrameModeParamsPayload(w, seq, header.Prefix, header.FrameMode); err != nil {
		return err
	}
	if err := writeGlobalMotionParamsPayload(w, header.Prefix, header.Size, header.Tile, header.References, header.GlobalMotion); err != nil {
		return err
	}
	return writeFilmGrainParamsPayload(w, seq, header.Prefix, header.Size, header.References, header.FilmGrain)
}
