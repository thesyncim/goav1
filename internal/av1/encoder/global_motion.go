package encoder

import (
	"math/bits"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

const (
	warpedModelPrecBits = 16
	subexpFinK          = 3

	gmAbsTransBits      = 12
	gmAbsTransOnlyBits  = gmAbsTransBits - 6 + 3
	gmTransPrecDiff     = warpedModelPrecBits - 6
	gmTransOnlyPrecDiff = warpedModelPrecBits - 3

	gmAlphaMax = 1 << 12
)

type GlobalMotionType uint8

const (
	GlobalMotionIdentity GlobalMotionType = iota
	GlobalMotionTranslation
	GlobalMotionRotZoom
	GlobalMotionAffine
)

type WarpedMotionParams struct {
	Type   GlobalMotionType
	Matrix [6]int32
}

type GlobalMotionParams struct {
	Ref [7]WarpedMotionParams
}

func DefaultWarpedMotionParams() WarpedMotionParams {
	return WarpedMotionParams{
		Type: GlobalMotionIdentity,
		Matrix: [6]int32{
			0, 0,
			1 << warpedModelPrecBits, 0,
			0, 1 << warpedModelPrecBits,
		},
	}
}

func DefaultGlobalMotionParams() GlobalMotionParams {
	var params GlobalMotionParams
	defaultWarp := DefaultWarpedMotionParams()
	for i := range params.Ref {
		params.Ref[i] = defaultWarp
	}
	return params
}

func GlobalMotionParamsPayloadSize(prefix FrameHeaderPrefix, size InterFrameSize, tiles parser.TileInfo, refs *parser.ReferenceState, params GlobalMotionParams) (int, error) {
	w := newSizingBitWriter()
	if err := writeGlobalMotionParamsPayload(&w, prefix, size, tiles, refs, params); err != nil {
		return 0, err
	}
	return w.bytesWritten(), nil
}

func AppendGlobalMotionParamsPayload(dst []byte, prefix FrameHeaderPrefix, size InterFrameSize, tiles parser.TileInfo, refs *parser.ReferenceState, params GlobalMotionParams) ([]byte, error) {
	payloadSize, err := GlobalMotionParamsPayloadSize(prefix, size, tiles, refs, params)
	if err != nil {
		return dst, err
	}
	if cap(dst)-len(dst) < payloadSize {
		return dst, bitstream.ErrShortBuffer
	}
	off := len(dst)
	out := dst[:off+payloadSize]
	w := newBitWriter(out[off:])
	if err := writeGlobalMotionParamsPayload(&w, prefix, size, tiles, refs, params); err != nil {
		return dst, err
	}
	return out, nil
}

func writeGlobalMotionParamsPayload(w *bitWriter, prefix FrameHeaderPrefix, size InterFrameSize, tiles parser.TileInfo, refs *parser.ReferenceState, params GlobalMotionParams) error {
	if err := validateGlobalMotionParams(prefix, size, refs, params); err != nil {
		return err
	}
	if !prefix.FrameType.interOrSwitch() {
		return nil
	}
	for i := uint8(0); i < parser.InterRefsPerFrame; i++ {
		refParams, err := encoderGlobalMotionReferenceParams(prefix, size, refs, int(i))
		if err != nil {
			return err
		}
		if err := writeGlobalMotionRef(w, params.Ref[i], refParams, tiles.AllowHighPrecisionMV); err != nil {
			return err
		}
	}
	return nil
}

func validateGlobalMotionParams(prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, params GlobalMotionParams) error {
	if !prefix.FrameType.valid() || prefix.PrimaryRefFrame > EncoderPrimaryRefNone {
		return ErrInvalidFrame
	}
	if !prefix.FrameType.interOrSwitch() {
		if params != DefaultGlobalMotionParams() {
			return ErrInvalidFrame
		}
		return nil
	}
	if prefix.PrimaryRefFrame != EncoderPrimaryRefNone {
		if prefix.PrimaryRefFrame >= parser.InterRefsPerFrame || refs == nil {
			return ErrInvalidFrame
		}
		slot := size.RefFrameIdx[prefix.PrimaryRefFrame]
		if slot >= parser.RefFrames || !refs.Frames[slot].Valid {
			return ErrInvalidFrame
		}
	}
	for i := range params.Ref {
		if params.Ref[i].Type > GlobalMotionAffine {
			return ErrInvalidFrame
		}
	}
	return nil
}

func encoderGlobalMotionReferenceParams(prefix FrameHeaderPrefix, size InterFrameSize, refs *parser.ReferenceState, ref int) (WarpedMotionParams, error) {
	defaultWarp := DefaultWarpedMotionParams()
	if prefix.PrimaryRefFrame == EncoderPrimaryRefNone {
		return defaultWarp, nil
	}
	slot := size.RefFrameIdx[prefix.PrimaryRefFrame]
	reference := refs.Frames[slot]
	return encoderWarpedMotionParamsFromParser(reference.GlobalMotion.Ref[ref]), nil
}

func encoderWarpedMotionParamsFromParser(params parser.WarpedMotionParams) WarpedMotionParams {
	return WarpedMotionParams{
		Type:   GlobalMotionType(params.Type),
		Matrix: params.Matrix,
	}
}

func writeGlobalMotionRef(w *bitWriter, gm WarpedMotionParams, ref WarpedMotionParams, allowHighPrecisionMV bool) error {
	if gm.Type == GlobalMotionIdentity {
		if gm != DefaultWarpedMotionParams() {
			return ErrInvalidFrame
		}
		return w.writeBool(false)
	}
	if err := w.writeBool(true); err != nil {
		return err
	}
	if err := w.writeBool(gm.Type == GlobalMotionRotZoom); err != nil {
		return err
	}
	if gm.Type == GlobalMotionTranslation {
		if err := w.writeBool(true); err != nil {
			return err
		}
	} else if gm.Type == GlobalMotionAffine {
		if err := w.writeBool(false); err != nil {
			return err
		}
	}

	if gm.Type >= GlobalMotionRotZoom {
		if gm.Matrix[2]&(1) != 0 || gm.Matrix[3]&(1) != 0 {
			return ErrInvalidFrame
		}
		v := (gm.Matrix[2] - (1 << warpedModelPrecBits)) >> 1
		refV := (ref.Matrix[2] - (1 << warpedModelPrecBits)) >> 1
		if err := writeSignedPrimitiveRefSubexpFin(w, gmAlphaMax+1, subexpFinK, refV, v); err != nil {
			return err
		}
		v = gm.Matrix[3] >> 1
		refV = ref.Matrix[3] >> 1
		if err := writeSignedPrimitiveRefSubexpFin(w, gmAlphaMax+1, subexpFinK, refV, v); err != nil {
			return err
		}
	}

	if gm.Type == GlobalMotionAffine {
		if gm.Matrix[4]&(1) != 0 || gm.Matrix[5]&(1) != 0 {
			return ErrInvalidFrame
		}
		v := gm.Matrix[4] >> 1
		refV := ref.Matrix[4] >> 1
		if err := writeSignedPrimitiveRefSubexpFin(w, gmAlphaMax+1, subexpFinK, refV, v); err != nil {
			return err
		}
		v = (gm.Matrix[5] - (1 << warpedModelPrecBits)) >> 1
		refV = (ref.Matrix[5] - (1 << warpedModelPrecBits)) >> 1
		if err := writeSignedPrimitiveRefSubexpFin(w, gmAlphaMax+1, subexpFinK, refV, v); err != nil {
			return err
		}
	} else if gm.Type >= GlobalMotionRotZoom {
		if gm.Matrix[4] != -gm.Matrix[3] || gm.Matrix[5] != gm.Matrix[2] {
			return ErrInvalidFrame
		}
	} else if gm.Matrix[2] != 1<<warpedModelPrecBits || gm.Matrix[3] != 0 || gm.Matrix[4] != 0 || gm.Matrix[5] != 1<<warpedModelPrecBits {
		return ErrInvalidFrame
	}

	if gm.Type >= GlobalMotionTranslation {
		bits := uint8(gmAbsTransBits)
		shift := uint8(gmTransPrecDiff)
		if gm.Type == GlobalMotionTranslation {
			bits = gmAbsTransOnlyBits
			shift = gmTransOnlyPrecDiff
			if !allowHighPrecisionMV {
				bits--
				shift++
			}
		}
		mask := int32((1 << shift) - 1)
		if gm.Matrix[0]&mask != 0 || gm.Matrix[1]&mask != 0 {
			return ErrInvalidFrame
		}
		n := uint32(1<<bits) + 1
		if err := writeSignedPrimitiveRefSubexpFin(w, n, subexpFinK, ref.Matrix[0]>>shift, gm.Matrix[0]>>shift); err != nil {
			return err
		}
		if err := writeSignedPrimitiveRefSubexpFin(w, n, subexpFinK, ref.Matrix[1]>>shift, gm.Matrix[1]>>shift); err != nil {
			return err
		}
	}
	return nil
}

func writeSignedPrimitiveRefSubexpFin(w *bitWriter, n uint32, k uint8, ref int32, value int32) error {
	scaledN := (n << 1) - 1
	scaledRef := int64(ref) + int64(n) - 1
	scaledValue := int64(value) + int64(n) - 1
	if scaledRef < 0 || scaledRef >= int64(scaledN) || scaledValue < 0 || scaledValue >= int64(scaledN) {
		return ErrInvalidFrame
	}
	v := recenterFiniteNonNeg(scaledN, uint32(scaledRef), uint32(scaledValue))
	return writePrimitiveSubexpFin(w, scaledN, k, v)
}

func writePrimitiveSubexpFin(w *bitWriter, n uint32, k uint8, v uint32) error {
	if v >= n {
		return ErrInvalidFrame
	}
	i := 0
	mk := uint32(0)
	for {
		b := k
		if i > 0 {
			b += uint8(i - 1)
		}
		a := uint32(1) << b
		if n <= mk+3*a {
			return writePrimitiveQUniform(w, n-mk, v-mk)
		}
		if v < mk+a {
			if err := w.writeBool(false); err != nil {
				return err
			}
			return w.writeBits(uint64(v-mk), b)
		}
		if err := w.writeBool(true); err != nil {
			return err
		}
		i++
		mk += a
	}
}

func writePrimitiveQUniform(w *bitWriter, n uint32, v uint32) error {
	if v >= n {
		return ErrInvalidFrame
	}
	if n <= 1 {
		return nil
	}
	l := uint8(bits.Len32(n))
	m := (uint32(1) << l) - n
	if v < m {
		return w.writeBits(uint64(v), l-1)
	}
	value := v + m
	if err := w.writeBits(uint64(value>>1), l-1); err != nil {
		return err
	}
	return w.writeBit(uint8(value & 1))
}

func recenterFiniteNonNeg(n uint32, ref uint32, value uint32) uint32 {
	if ref<<1 <= n {
		return recenterNonNeg(ref, value)
	}
	return recenterNonNeg(n-1-ref, n-1-value)
}

func recenterNonNeg(ref uint32, value uint32) uint32 {
	if value > ref<<1 {
		return value
	}
	if value >= ref {
		return (value - ref) << 1
	}
	return ((ref - value) << 1) - 1
}
