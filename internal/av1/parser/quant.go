// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// QuantizationParams is quantization_params() from uncompressed_header().
type QuantizationParams struct {
	BaseQIdx uint8

	YDCDelta int16
	UDCDelta int16
	UACDelta int16
	VDCDelta int16
	VACDelta int16

	DiffUVDeltas bool

	UsingQMatrix  bool
	QMatrixLevelY uint8
	QMatrixLevelU uint8
	QMatrixLevelV uint8

	BitsRead uint16
}

func ParseQuantizationParams(payload []byte, seq SequenceHeader, tiles TileInfo) (QuantizationParams, error) {
	r := bitstream.NewReader(payload)
	if err := skipBitsRead(&r, tiles.BitsRead); err != nil {
		return QuantizationParams{}, err
	}

	var quant QuantizationParams
	v, err := r.ReadBits(8)
	if err != nil {
		return QuantizationParams{}, err
	}
	quant.BaseQIdx = uint8(v)

	if quant.YDCDelta, err = readDeltaQ(&r); err != nil {
		return QuantizationParams{}, err
	}
	if !seq.ColorConfig.MonoChrome {
		if seq.ColorConfig.SeparateUVDeltaQ {
			if quant.DiffUVDeltas, err = r.ReadBool(); err != nil {
				return QuantizationParams{}, err
			}
		}
		if quant.UDCDelta, err = readDeltaQ(&r); err != nil {
			return QuantizationParams{}, err
		}
		if quant.UACDelta, err = readDeltaQ(&r); err != nil {
			return QuantizationParams{}, err
		}
		if quant.DiffUVDeltas {
			if quant.VDCDelta, err = readDeltaQ(&r); err != nil {
				return QuantizationParams{}, err
			}
			if quant.VACDelta, err = readDeltaQ(&r); err != nil {
				return QuantizationParams{}, err
			}
		} else {
			quant.VDCDelta = quant.UDCDelta
			quant.VACDelta = quant.UACDelta
		}
	}

	if quant.UsingQMatrix, err = r.ReadBool(); err != nil {
		return QuantizationParams{}, err
	}
	if quant.UsingQMatrix {
		v, err = r.ReadBits(4)
		if err != nil {
			return QuantizationParams{}, err
		}
		quant.QMatrixLevelY = uint8(v)
		v, err = r.ReadBits(4)
		if err != nil {
			return QuantizationParams{}, err
		}
		quant.QMatrixLevelU = uint8(v)
		if seq.ColorConfig.SeparateUVDeltaQ {
			v, err = r.ReadBits(4)
			if err != nil {
				return QuantizationParams{}, err
			}
			quant.QMatrixLevelV = uint8(v)
		} else {
			quant.QMatrixLevelV = quant.QMatrixLevelU
		}
	}

	bitsRead, err := readerBitsRead(&r)
	if err != nil {
		return QuantizationParams{}, err
	}
	quant.BitsRead = bitsRead
	return quant, nil
}

func readDeltaQ(r *bitstream.Reader) (int16, error) {
	present, err := r.ReadBool()
	if err != nil {
		return 0, err
	}
	if !present {
		return 0, nil
	}
	return readSignedBits(r, 7)
}

func readSignedBits(r *bitstream.Reader, n uint8) (int16, error) {
	v, err := r.ReadBits(n)
	if err != nil {
		return 0, err
	}
	shift := 32 - n
	return int16(int32(uint32(v)<<shift) >> shift), nil
}
