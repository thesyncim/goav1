package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

// DeltaParams is delta_q_params() plus delta_lf_params().
type DeltaParams struct {
	DeltaQPresent  bool
	DeltaQResLog2  uint8
	DeltaLFPresent bool
	DeltaLFResLog2 uint8
	DeltaLFMulti   bool
	BitsRead       int
}

func ParseDeltaParams(payload []byte, size FrameSize, quant QuantizationParams, seg SegmentationParams) (DeltaParams, error) {
	r := bitstream.NewReader(payload)
	if err := r.SkipBits(seg.BitsRead); err != nil {
		return DeltaParams{}, err
	}

	var delta DeltaParams
	if quant.BaseQIdx != 0 {
		present, err := r.ReadBool()
		if err != nil {
			return DeltaParams{}, err
		}
		delta.DeltaQPresent = present
		if delta.DeltaQPresent {
			v, err := r.ReadBits(2)
			if err != nil {
				return DeltaParams{}, err
			}
			delta.DeltaQResLog2 = uint8(v)
			if !size.AllowIntrabc {
				if delta.DeltaLFPresent, err = r.ReadBool(); err != nil {
					return DeltaParams{}, err
				}
			}
			if delta.DeltaLFPresent {
				v, err = r.ReadBits(2)
				if err != nil {
					return DeltaParams{}, err
				}
				delta.DeltaLFResLog2 = uint8(v)
				if delta.DeltaLFMulti, err = r.ReadBool(); err != nil {
					return DeltaParams{}, err
				}
			}
		}
	}

	delta.BitsRead = r.BitsRead()
	return delta, nil
}
