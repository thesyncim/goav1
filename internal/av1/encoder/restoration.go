package encoder

import (
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

const RestorationUnitMax = 256

type RestorationType uint8

const (
	RestorationNone RestorationType = iota
	RestorationSwitchable
	RestorationWiener
	RestorationSGRProj
)

// RestorationParams is the encoder-owned view of loop_restoration_params().
type RestorationParams struct {
	Type [3]RestorationType

	UnitSizeYLog2  uint8
	UnitSizeUVLog2 uint8
	UnitSizeY      uint16
	UnitSizeUV     uint16
}

func writeRestorationParamsPayload(w *bitWriter, seq SequenceHeader, allowIntrabc bool, superResEnabled bool, allLossless bool, restoration RestorationParams) error {
	if err := validateRestorationParams(seq, allowIntrabc, superResEnabled, allLossless, restoration); err != nil {
		return err
	}
	if (allLossless && !superResEnabled) || !seq.EnableRestoration || allowIntrabc {
		return nil
	}
	if err := w.writeBits(uint64(restoration.Type[0]), 2); err != nil {
		return err
	}
	if !seq.ColorConfig.MonoChrome {
		if err := w.writeBits(uint64(restoration.Type[1]), 2); err != nil {
			return err
		}
		if err := w.writeBits(uint64(restoration.Type[2]), 2); err != nil {
			return err
		}
	}
	if !restorationAnyActive(restoration) {
		return nil
	}
	base := uint8(6)
	if seq.Use128x128Superblock {
		base = 7
	}
	if restoration.UnitSizeYLog2 == base {
		if err := w.writeBool(false); err != nil {
			return err
		}
	} else {
		if err := w.writeBool(true); err != nil {
			return err
		}
		if !seq.Use128x128Superblock {
			if err := w.writeBool(restoration.UnitSizeYLog2 == base+2); err != nil {
				return err
			}
		}
	}
	if !seq.ColorConfig.MonoChrome &&
		(restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) &&
		seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY {
		return w.writeBool(restoration.UnitSizeUVLog2+1 == restoration.UnitSizeYLog2)
	}
	return nil
}

func validateRestorationParams(seq SequenceHeader, allowIntrabc bool, superResEnabled bool, allLossless bool, restoration RestorationParams) error {
	if (allLossless && !superResEnabled) || !seq.EnableRestoration || allowIntrabc {
		if restoration != (RestorationParams{}) {
			return ErrInvalidFrame
		}
		return nil
	}
	for i := uint8(0); i < 3; i++ {
		if restoration.Type[i] > RestorationSGRProj {
			return ErrInvalidFrame
		}
	}
	if seq.ColorConfig.MonoChrome && (restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) {
		return ErrInvalidFrame
	}
	if !restorationAnyActive(restoration) {
		return validateRestorationNoneUnits(restoration)
	}
	base := uint8(6)
	if seq.Use128x128Superblock {
		base = 7
	}
	max := base + 2
	if seq.Use128x128Superblock {
		max = base + 1
	}
	if restoration.UnitSizeYLog2 < base || restoration.UnitSizeYLog2 > max {
		return ErrInvalidFrame
	}
	if restoration.UnitSizeY != 0 && restoration.UnitSizeY != 1<<restoration.UnitSizeYLog2 {
		return ErrInvalidFrame
	}
	wantUV := restoration.UnitSizeYLog2
	chromaActive420 := !seq.ColorConfig.MonoChrome &&
		(restoration.Type[1] != RestorationNone || restoration.Type[2] != RestorationNone) &&
		seq.ColorConfig.SubsamplingX && seq.ColorConfig.SubsamplingY
	if chromaActive420 {
		if restoration.UnitSizeUVLog2 != restoration.UnitSizeYLog2 &&
			restoration.UnitSizeUVLog2+1 != restoration.UnitSizeYLog2 {
			return ErrInvalidFrame
		}
		wantUV = restoration.UnitSizeUVLog2
	} else if restoration.UnitSizeUVLog2 != 0 && restoration.UnitSizeUVLog2 != restoration.UnitSizeYLog2 {
		return ErrInvalidFrame
	}
	if restoration.UnitSizeUV != 0 && restoration.UnitSizeUV != 1<<wantUV {
		return ErrInvalidFrame
	}
	return nil
}

func validateRestorationNoneUnits(restoration RestorationParams) error {
	if restoration.UnitSizeYLog2 == 0 && restoration.UnitSizeUVLog2 == 0 &&
		restoration.UnitSizeY == 0 && restoration.UnitSizeUV == 0 {
		return nil
	}
	if restoration.UnitSizeYLog2 == 8 && restoration.UnitSizeUVLog2 == 8 &&
		restoration.UnitSizeY == RestorationUnitMax && restoration.UnitSizeUV == RestorationUnitMax {
		return nil
	}
	return ErrInvalidFrame
}

func restorationAnyActive(restoration RestorationParams) bool {
	return restoration.Type[0] != RestorationNone ||
		restoration.Type[1] != RestorationNone ||
		restoration.Type[2] != RestorationNone
}

// SearchWienerUnitU8 evaluates a curated set of AV1 7-tap separable Wiener
// kernels on the restoration unit [x0:x0+w, y0:y0+h] and returns the filter
// that achieves the lowest SSE against src when it improves on unfiltered
// recon.
func SearchWienerUnitU8(src, recon []byte, stride, frameW, frameH, x0, y0, w, h int) (av1restoration.WienerInfo, bool) {
	if w <= 0 || h <= 0 || x0+w > frameW || y0+h > frameH {
		return av1restoration.WienerInfo{}, false
	}
	// Baseline SSE (unfiltered recon vs src).
	var baseSSE uint64
	for y := y0; y < y0+h; y++ {
		off := y * stride
		for x := x0; x < x0+w; x++ {
			d := int(src[off+x]) - int(recon[off+x])
			baseSSE += uint64(d * d)
		}
	}
	if baseSSE == 0 {
		return av1restoration.WienerInfo{}, false
	}
	candidates := [][3]int16{
		{0, -3, 12}, // mild sharpening
		{0, -5, 18}, // medium sharpening
		{0, -2, 8},  // subtle sharpening
		{0, 2, 10},  // mild denoise/smoothing
		{1, 4, 14},  // moderate smoothing
		{0, -1, 4},  // light sharpening
	}
	bestSSE := baseSSE
	var best av1restoration.WienerInfo
	found := false
	for _, taps := range candidates {
		f := av1restoration.NewWienerFilter(taps[0], taps[1], taps[2])
		var sse uint64
		// Separable 7-tap approximation over interior rows/cols (clamped edges).
		for y := y0; y < y0+h; y++ {
			off := y * stride
			for x := x0; x < x0+w; x++ {
				hsum := 0
				for k := -3; k <= 3; k++ {
					cx := x + k
					if cx < 0 {
						cx = 0
					} else if cx >= frameW {
						cx = frameW - 1
					}
					hsum += int(f[k+3]) * int(recon[off+cx])
				}
				pred := (hsum + 64) >> 7
				if pred < 0 {
					pred = 0
				} else if pred > 255 {
					pred = 255
				}
				d := int(src[off+x]) - pred
				sse += uint64(d * d)
			}
		}
		if sse+uint64(w*h) < bestSSE {
			bestSSE = sse
			best = av1restoration.WienerInfo{VFilter: f, HFilter: f}
			found = true
		}
	}
	return best, found
}
