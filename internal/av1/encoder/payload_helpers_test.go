package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

type payloadWrite func(*bitWriter) error

func payloadSize(write payloadWrite) (int, error) {
	w := newSizingBitWriter()
	if err := write(&w); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func appendPayload(dst []byte, write payloadWrite) ([]byte, error) {
	n, err := payloadSize(write)
	if err != nil {
		return dst, err
	}
	out := append(dst, make([]byte, n)...)
	w := newBitWriter(out[len(dst):])
	if err := write(&w); err != nil {
		return dst, err
	}
	if w.bytesWritten() != n {
		return dst, fmt.Errorf("payload wrote %d bytes, sized %d", w.bytesWritten(), n)
	}
	return out, nil
}

func cdefWrite(seq SequenceHeader, allowIntrabc, allLossless bool, cdef CDEFParams) payloadWrite {
	return func(w *bitWriter) error { return writeCDEFParamsPayload(w, seq, allowIntrabc, allLossless, cdef) }
}

func appendCDEFParamsPayload(dst []byte, seq SequenceHeader, size IntraFrameSize, allLossless bool, cdef CDEFParams) ([]byte, error) {
	return appendPayload(dst, cdefWrite(seq, size.AllowIntrabc, allLossless, cdef))
}

func appendCDEFParamsInterPayload(dst []byte, seq SequenceHeader, _ InterFrameSize, allLossless bool, cdef CDEFParams) ([]byte, error) {
	return appendPayload(dst, cdefWrite(seq, false, allLossless, cdef))
}

func deltaWrite(allowIntrabc bool, quant QuantizationParams, delta DeltaParams) payloadWrite {
	return func(w *bitWriter) error { return writeDeltaParamsPayload(w, allowIntrabc, quant, delta) }
}

func appendDeltaParamsPayload(dst []byte, size IntraFrameSize, quant QuantizationParams, delta DeltaParams) ([]byte, error) {
	return appendPayload(dst, deltaWrite(size.AllowIntrabc, quant, delta))
}

func appendDeltaParamsInterPayload(dst []byte, _ InterFrameSize, quant QuantizationParams, delta DeltaParams) ([]byte, error) {
	return appendPayload(dst, deltaWrite(false, quant, delta))
}

func appendFilmGrainParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params FilmGrainParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeFilmGrainParamsPayload(w, seq, prefix, size, refs, params)
	})
}

func appendFrameHeaderPrefixPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeFrameHeaderPrefixPayload(w, seq, prefix) })
}

func appendIntraFrameHeaderPayload(dst []byte, seq SequenceHeader, header IntraFrameHeaderParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeIntraFrameHeaderPayload(w, seq, header) })
}

func appendInterFrameHeaderPayload(dst []byte, seq SequenceHeader, header InterFrameHeaderParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeInterFrameHeaderPayload(w, seq, header) })
}

func appendFrameModeParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, params FrameModeParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeFrameModeParamsPayload(w, seq, prefix, params) })
}

// appendFrameHeaderIntraPayload writes the frame header through frame_size()
// on the intra path.
func appendFrameHeaderIntraPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		if err := writeFrameHeaderPrefixPayload(w, seq, prefix); err != nil {
			return err
		}
		return writeIntraFrameSizePayload(w, seq, prefix, size)
	})
}

// appendFrameHeaderInterPayload writes the frame header through frame_size()
// on the inter/switch path.
func appendFrameHeaderInterPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		if err := writeFrameHeaderPrefixPayload(w, seq, prefix); err != nil {
			return err
		}
		return writeInterFrameSizePayload(w, seq, prefix, size)
	})
}

func appendGlobalMotionParamsPayload(dst []byte, prefix FrameHeaderPrefix, size InterFrameSize, tiles parser.TileInfo, refs *parser.ReferenceState, params GlobalMotionParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeGlobalMotionParamsPayload(w, prefix, size, tiles, refs, params)
	})
}

func loopFilterWrite(seq SequenceHeader, prefix FrameHeaderPrefix, allowIntrabc, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) payloadWrite {
	return func(w *bitWriter) error {
		return writeLoopFilterParamsPayload(w, seq, prefix, allowIntrabc, allLossless, lf, previous)
	}
}

func appendLoopFilterParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size IntraFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) ([]byte, error) {
	return appendPayload(dst, loopFilterWrite(seq, prefix, size.AllowIntrabc, allLossless, lf, previous))
}

func appendLoopFilterParamsInterPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, _ InterFrameSize, allLossless bool, lf LoopFilterParams, previous *LoopFilterDeltas) ([]byte, error) {
	return appendPayload(dst, loopFilterWrite(seq, prefix, false, allLossless, lf, previous))
}

func appendQuantizationParamsPayload(dst []byte, seq SequenceHeader, quant QuantizationParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeQuantizationParamsPayload(w, seq, quant) })
}

func restorationWrite(seq SequenceHeader, allowIntrabc, superRes, allLossless bool, restoration RestorationParams) payloadWrite {
	return func(w *bitWriter) error {
		return writeRestorationParamsPayload(w, seq, allowIntrabc, superRes, allLossless, restoration)
	}
}

func appendRestorationParamsPayload(dst []byte, seq SequenceHeader, size IntraFrameSize, allLossless bool, restoration RestorationParams) ([]byte, error) {
	return appendPayload(dst, restorationWrite(seq, size.AllowIntrabc, size.SuperResDenominator != 8, allLossless, restoration))
}

func appendRestorationParamsInterPayload(dst []byte, seq SequenceHeader, size InterFrameSize, allLossless bool, restoration RestorationParams) ([]byte, error) {
	return appendPayload(dst, restorationWrite(seq, false, size.SuperResDenominator != 8, allLossless, restoration))
}

func appendSegmentationParamsPayload(dst []byte, prefix FrameHeaderPrefix, seg SegmentationParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeSegmentationParamsPayload(w, prefix, seg) })
}

func appendSequenceHeaderPayload(dst []byte, seq SequenceHeader) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error { return writeSequenceHeaderPayload(w, seq) })
}

func appendSkipModeParamsPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, transformRef TransformReferenceParams, params SkipModeParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeSkipModeParamsPayload(w, seq, prefix, size, refs, transformRef, params)
	})
}

func appendTileInfoPayload(dst []byte, seq SequenceHeader, prefix FrameHeaderPrefix, codedWidth uint32, height uint32, tiles TileInfo) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeTileInfoPayload(w, seq, prefix, codedWidth, height, tiles)
	})
}

func appendTransformReferenceParamsPayload(dst []byte, prefix FrameHeaderPrefix, allLossless bool, params TransformReferenceParams) ([]byte, error) {
	return appendPayload(dst, func(w *bitWriter) error {
		return writeTransformReferenceParamsPayload(w, prefix, allLossless, params)
	})
}
