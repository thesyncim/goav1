// Ported from libaom: av1/decoder/decodeframe.c
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const (
	RestorationUnitMax = 256
)

type RestorationType uint8

const (
	RestorationNone RestorationType = iota
	RestorationSwitchable
	RestorationWiener
	RestorationSGRProj
)

// RestorationParams is loop_restoration_params() from uncompressed_header().
type RestorationParams struct {
	Type [3]RestorationType

	UnitSizeYLog2  uint8
	UnitSizeUVLog2 uint8
	UnitSizeY      uint16
	UnitSizeUV     uint16

	BitsRead uint16
}

func ParseRestorationParams(payload []byte, seq SequenceHeader, size FrameSize, seg SegmentationParams, cdef CDEFParams) (RestorationParams, error) {
	r := bitstream.NewReader(payload)
	if err := skipBitsRead(&r, cdef.BitsRead); err != nil {
		return RestorationParams{}, err
	}

	var restoration RestorationParams
	if (seg.AllLossless && !size.SuperResEnabled) || !seq.EnableRestoration || size.AllowIntrabc {
		bitsRead, err := readerBitsRead(&r)
		if err != nil {
			return RestorationParams{}, err
		}
		restoration.BitsRead = bitsRead
		return restoration, nil
	}

	v, err := r.ReadBits(2)
	if err != nil {
		return RestorationParams{}, err
	}
	restoration.Type[0] = RestorationType(v)
	if !seq.ColorConfig.MonoChrome {
		v, err = r.ReadBits(2)
		if err != nil {
			return RestorationParams{}, err
		}
		restoration.Type[1] = RestorationType(v)
		v, err = r.ReadBits(2)
		if err != nil {
			return RestorationParams{}, err
		}
		restoration.Type[2] = RestorationType(v)
	}

	if restoration.Type[0] != RestorationNone || restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone {
		log2 := uint8(6)
		if seq.Use128x128Superblock {
			log2 = 7
		}
		increase, err := r.ReadBool()
		if err != nil {
			return RestorationParams{}, err
		}
		if increase {
			log2++
			if !seq.Use128x128Superblock {
				more, err := r.ReadBool()
				if err != nil {
					return RestorationParams{}, err
				}
				if more {
					log2++
				}
			}
		}
		restoration.UnitSizeYLog2 = log2
		restoration.UnitSizeUVLog2 = log2
		if !seq.ColorConfig.MonoChrome &&
			(restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) &&
			seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY {
			smaller, err := r.ReadBool()
			if err != nil {
				return RestorationParams{}, err
			}
			if smaller {
				restoration.UnitSizeUVLog2--
			}
		}
		restoration.UnitSizeY = 1 << restoration.UnitSizeYLog2
		restoration.UnitSizeUV = 1 << restoration.UnitSizeUVLog2
	} else {
		restoration.UnitSizeYLog2 = 8
		restoration.UnitSizeUVLog2 = 8
		restoration.UnitSizeY = RestorationUnitMax
		restoration.UnitSizeUV = RestorationUnitMax
	}

	bitsRead, err := readerBitsRead(&r)
	if err != nil {
		return RestorationParams{}, err
	}
	restoration.BitsRead = bitsRead
	return restoration, nil
}
