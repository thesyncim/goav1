package parser

import "github.com/thesyncim/goav1/internal/av1/bitstream"

const MaxSegments = 8

// SegmentData is one segment's frame-header feature data.
type SegmentData struct {
	DeltaQ    int16
	DeltaLFYV int16
	DeltaLFYH int16
	DeltaLFU  int16
	DeltaLFV  int16
	RefFrame  int8
	Skip      bool
	GlobalMV  bool
}

// SegmentationData is the reference-copyable segmentation feature state.
type SegmentationData struct {
	Segments     [MaxSegments]SegmentData
	LastActiveID int8
	Preskip      bool
}

// SegmentationParams is segmentation_params() plus q-index/lossless derivation.
type SegmentationParams struct {
	Enabled        bool
	UpdateMap      bool
	TemporalUpdate bool
	UpdateData     bool
	Data           SegmentationData
	QIndex         [MaxSegments]uint8
	Lossless       [MaxSegments]bool
	AllLossless    bool
	BitsRead       int
}

func ParseSegmentationParams(payload []byte, prefix FrameHeaderPrefix, quant QuantizationParams, previous *SegmentationData) (SegmentationParams, error) {
	r := bitstream.NewReader(payload)
	if err := r.SkipBits(quant.BitsRead); err != nil {
		return SegmentationParams{}, err
	}

	var seg SegmentationParams
	enabled, err := r.ReadBool()
	if err != nil {
		return SegmentationParams{}, err
	}
	seg.Enabled = enabled
	if seg.Enabled {
		if prefix.PrimaryRefFrame == PrimaryRefNone {
			seg.UpdateMap = true
			seg.UpdateData = true
		} else {
			if seg.UpdateMap, err = r.ReadBool(); err != nil {
				return SegmentationParams{}, err
			}
			if seg.UpdateMap {
				if seg.TemporalUpdate, err = r.ReadBool(); err != nil {
					return SegmentationParams{}, err
				}
			}
			if seg.UpdateData, err = r.ReadBool(); err != nil {
				return SegmentationParams{}, err
			}
		}

		if seg.UpdateData {
			if err := readSegmentationData(&r, &seg.Data); err != nil {
				return SegmentationParams{}, err
			}
		} else {
			if previous == nil {
				return SegmentationParams{}, ErrReferenceFrameNeeded
			}
			seg.Data = *previous
		}
	} else {
		clearSegmentationRefs(&seg.Data)
	}

	deriveSegmentationQIndex(&seg, quant)
	seg.BitsRead = r.BitsRead()
	return seg, nil
}

func readSegmentationData(r *bitstream.Reader, data *SegmentationData) error {
	*data = SegmentationData{LastActiveID: -1}
	for i := range MaxSegments {
		seg := &data.Segments[i]
		seg.RefFrame = -1

		active, err := readSegmentSignedFeature(r, 9, &seg.DeltaQ)
		if err != nil {
			return err
		}
		if active {
			data.LastActiveID = int8(i)
		}
		active, err = readSegmentSignedFeature(r, 7, &seg.DeltaLFYV)
		if err != nil {
			return err
		}
		if active {
			data.LastActiveID = int8(i)
		}
		active, err = readSegmentSignedFeature(r, 7, &seg.DeltaLFYH)
		if err != nil {
			return err
		}
		if active {
			data.LastActiveID = int8(i)
		}
		active, err = readSegmentSignedFeature(r, 7, &seg.DeltaLFU)
		if err != nil {
			return err
		}
		if active {
			data.LastActiveID = int8(i)
		}
		active, err = readSegmentSignedFeature(r, 7, &seg.DeltaLFV)
		if err != nil {
			return err
		}
		if active {
			data.LastActiveID = int8(i)
		}

		refEnabled, err := r.ReadBool()
		if err != nil {
			return err
		}
		if refEnabled {
			v, err := r.ReadBits(3)
			if err != nil {
				return err
			}
			seg.RefFrame = int8(v)
			data.LastActiveID = int8(i)
			data.Preskip = true
		}
		if seg.Skip, err = r.ReadBool(); err != nil {
			return err
		}
		if seg.Skip {
			data.LastActiveID = int8(i)
			data.Preskip = true
		}
		if seg.GlobalMV, err = r.ReadBool(); err != nil {
			return err
		}
		if seg.GlobalMV {
			data.LastActiveID = int8(i)
			data.Preskip = true
		}
	}
	return nil
}

func readSegmentSignedFeature(r *bitstream.Reader, bits uint8, dst *int16) (bool, error) {
	enabled, err := r.ReadBool()
	if err != nil {
		return false, err
	}
	if !enabled {
		*dst = 0
		return false, nil
	}
	v, err := readSignedBits(r, bits)
	if err != nil {
		return false, err
	}
	*dst = v
	return true, nil
}

func clearSegmentationRefs(data *SegmentationData) {
	*data = SegmentationData{LastActiveID: -1}
	for i := range MaxSegments {
		data.Segments[i].RefFrame = -1
	}
}

func deriveSegmentationQIndex(seg *SegmentationParams, quant QuantizationParams) {
	deltaLossless := quant.YDCDelta == 0 && quant.UDCDelta == 0 &&
		quant.UACDelta == 0 && quant.VDCDelta == 0 && quant.VACDelta == 0
	seg.AllLossless = true
	for i := range MaxSegments {
		q := int(quant.BaseQIdx)
		if seg.Enabled {
			q += int(seg.Data.Segments[i].DeltaQ)
		}
		if q < 0 {
			q = 0
		} else if q > 255 {
			q = 255
		}
		seg.QIndex[i] = uint8(q)
		seg.Lossless[i] = q == 0 && deltaLossless
		seg.AllLossless = seg.AllLossless && seg.Lossless[i]
	}
}
