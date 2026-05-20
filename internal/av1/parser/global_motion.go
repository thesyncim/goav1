package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

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
	Ref      [InterRefsPerFrame]WarpedMotionParams
	BitsRead int
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
	for i := 0; i < InterRefsPerFrame; i++ {
		params.Ref[i] = defaultWarp
	}
	return params
}

func ParseGlobalMotionParams(payload []byte, prefix FrameHeaderPrefix, size FrameSize, tiles TileInfo, refs *ReferenceState, frameMode FrameModeParams) (GlobalMotionParams, error) {
	r := bitstream.NewReader(payload)
	if err := r.SkipBits(frameMode.BitsRead); err != nil {
		return GlobalMotionParams{}, err
	}

	params := DefaultGlobalMotionParams()
	if !frameTypeIsInterOrSwitch(prefix.FrameType) {
		params.BitsRead = r.BitsRead()
		return params, nil
	}

	for i := 0; i < InterRefsPerFrame; i++ {
		refParams, err := globalMotionReferenceParams(prefix, size, refs, i)
		if err != nil {
			return GlobalMotionParams{}, err
		}
		gm, err := readGlobalMotionRef(&r, refParams, tiles.AllowHighPrecisionMV)
		if err != nil {
			return GlobalMotionParams{}, err
		}
		params.Ref[i] = gm
	}
	params.BitsRead = r.BitsRead()
	return params, nil
}

func globalMotionReferenceParams(prefix FrameHeaderPrefix, size FrameSize, refs *ReferenceState, ref int) (WarpedMotionParams, error) {
	defaultWarp := DefaultWarpedMotionParams()
	if prefix.PrimaryRefFrame == PrimaryRefNone {
		return defaultWarp, nil
	}
	if prefix.PrimaryRefFrame >= InterRefsPerFrame {
		return WarpedMotionParams{}, ErrInvalidFrameHeader
	}
	if refs == nil {
		return WarpedMotionParams{}, ErrReferenceFrameNeeded
	}
	slot := size.RefFrameIdx[prefix.PrimaryRefFrame]
	if slot >= RefFrames {
		return WarpedMotionParams{}, ErrInvalidFrameHeader
	}
	reference := refs.Frames[slot]
	if !reference.Valid {
		return WarpedMotionParams{}, ErrReferenceFrameNeeded
	}
	return reference.GlobalMotion.Ref[ref], nil
}

func readGlobalMotionRef(r *bitstream.Reader, ref WarpedMotionParams, allowHighPrecisionMV bool) (WarpedMotionParams, error) {
	gm := DefaultWarpedMotionParams()
	isGlobal, err := r.ReadBool()
	if err != nil {
		return WarpedMotionParams{}, err
	}
	if !isGlobal {
		return gm, nil
	}

	rotZoom, err := r.ReadBool()
	if err != nil {
		return WarpedMotionParams{}, err
	}
	if rotZoom {
		gm.Type = GlobalMotionRotZoom
	} else {
		translation, err := r.ReadBool()
		if err != nil {
			return WarpedMotionParams{}, err
		}
		if translation {
			gm.Type = GlobalMotionTranslation
		} else {
			gm.Type = GlobalMotionAffine
		}
	}

	if gm.Type >= GlobalMotionRotZoom {
		v, err := readSignedPrimitiveRefSubexpFin(r, gmAlphaMax+1, subexpFinK, (ref.Matrix[2]-(1<<warpedModelPrecBits))>>1)
		if err != nil {
			return WarpedMotionParams{}, err
		}
		gm.Matrix[2] = (1 << warpedModelPrecBits) + 2*v
		v, err = readSignedPrimitiveRefSubexpFin(r, gmAlphaMax+1, subexpFinK, ref.Matrix[3]>>1)
		if err != nil {
			return WarpedMotionParams{}, err
		}
		gm.Matrix[3] = 2 * v
	}

	if gm.Type == GlobalMotionAffine {
		v, err := readSignedPrimitiveRefSubexpFin(r, gmAlphaMax+1, subexpFinK, ref.Matrix[4]>>1)
		if err != nil {
			return WarpedMotionParams{}, err
		}
		gm.Matrix[4] = 2 * v
		v, err = readSignedPrimitiveRefSubexpFin(r, gmAlphaMax+1, subexpFinK, (ref.Matrix[5]-(1<<warpedModelPrecBits))>>1)
		if err != nil {
			return WarpedMotionParams{}, err
		}
		gm.Matrix[5] = (1 << warpedModelPrecBits) + 2*v
	} else {
		gm.Matrix[4] = -gm.Matrix[3]
		gm.Matrix[5] = gm.Matrix[2]
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
		n := uint32(1<<bits) + 1
		v, err := readSignedPrimitiveRefSubexpFin(r, n, subexpFinK, ref.Matrix[0]>>shift)
		if err != nil {
			return WarpedMotionParams{}, err
		}
		gm.Matrix[0] = v * (1 << shift)
		v, err = readSignedPrimitiveRefSubexpFin(r, n, subexpFinK, ref.Matrix[1]>>shift)
		if err != nil {
			return WarpedMotionParams{}, err
		}
		gm.Matrix[1] = v * (1 << shift)
	}
	return gm, nil
}

func readSignedPrimitiveRefSubexpFin(r *bitstream.Reader, n uint32, k uint8, ref int32) (int32, error) {
	scaledN := (n << 1) - 1
	scaledRef := int64(ref) + int64(n) - 1
	if scaledRef < 0 || scaledRef >= int64(scaledN) {
		return 0, ErrInvalidFrameHeader
	}
	v, err := readPrimitiveRefSubexpFin(r, scaledN, k, uint32(scaledRef))
	if err != nil {
		return 0, err
	}
	return int32(v) - int32(n) + 1, nil
}

func readPrimitiveRefSubexpFin(r *bitstream.Reader, n uint32, k uint8, ref uint32) (uint32, error) {
	v, err := readPrimitiveSubexpFin(r, n, k)
	if err != nil {
		return 0, err
	}
	return invRecenterFiniteNonNeg(n, ref, v), nil
}

func readPrimitiveSubexpFin(r *bitstream.Reader, n uint32, k uint8) (uint32, error) {
	i := 0
	mk := uint32(0)
	for {
		b := k
		if i > 0 {
			b += uint8(i - 1)
		}
		a := uint32(1) << b
		if n <= mk+3*a {
			v, err := readPrimitiveQUniform(r, n-mk)
			if err != nil {
				return 0, err
			}
			return v + mk, nil
		}
		more, err := r.ReadBool()
		if err != nil {
			return 0, err
		}
		if !more {
			v, err := r.ReadBits(b)
			if err != nil {
				return 0, err
			}
			return uint32(v) + mk, nil
		}
		i++
		mk += a
	}
}

func readPrimitiveQUniform(r *bitstream.Reader, n uint32) (uint32, error) {
	if n <= 1 {
		return 0, nil
	}
	l := uint8(msb32(n) + 1)
	m := (uint32(1) << l) - n
	v, err := r.ReadBits(l - 1)
	if err != nil {
		return 0, err
	}
	if uint32(v) < m {
		return uint32(v), nil
	}
	bit, err := r.ReadBool()
	if err != nil {
		return 0, err
	}
	return (uint32(v) << 1) - m + boolBit(bit), nil
}

func invRecenterFiniteNonNeg(n uint32, ref uint32, v uint32) uint32 {
	if ref<<1 <= n {
		return invRecenterNonNeg(ref, v)
	}
	return n - 1 - invRecenterNonNeg(n-1-ref, v)
}

func invRecenterNonNeg(ref uint32, v uint32) uint32 {
	if v > ref<<1 {
		return v
	}
	if v&1 == 0 {
		return (v >> 1) + ref
	}
	return ref - ((v + 1) >> 1)
}

func boolBit(v bool) uint32 {
	if v {
		return 1
	}
	return 0
}

func msb32(v uint32) int {
	msb := -1
	for v != 0 {
		v >>= 1
		msb++
	}
	return msb
}
