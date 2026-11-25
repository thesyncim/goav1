package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

const (
	wienerTap0SubexpK = 1
	wienerTap1SubexpK = 2
	wienerTap2SubexpK = 3
)

// SGRProjInfo is the entropy-decoded per-restoration-unit SGR projection info.
type SGRProjInfo struct {
	ParamsIndex uint8
	XQD         [2]int8
}

// RestorationUnit is one decoded loop-restoration unit's syntax.
type RestorationUnit struct {
	Type    parser.RestorationType
	Wiener  av1restoration.WienerInfo
	SGRProj SGRProjInfo
}

// RestorationReferences are the previous Wiener/SGR parameters used by AV1's
// reference-subexponential coding. Callers keep one reference per plane.
type RestorationReferences struct {
	Wiener  av1restoration.WienerInfo
	SGRProj SGRProjInfo
}

// RestorationCDFs contains the caller-owned CDFs used by restoration unit
// syntax in tile data.
type RestorationCDFs struct {
	Switchable *entropy.CDF
	Wiener     *entropy.CDF
	SGRProj    *entropy.CDF
}

func DefaultRestorationReferences() RestorationReferences {
	return RestorationReferences{
		Wiener: av1restoration.DefaultWienerInfo(),
		SGRProj: SGRProjInfo{
			XQD: [2]int8{
				(av1restoration.SGRProjPrjMin0 + av1restoration.SGRProjPrjMax0) / 2,
				(av1restoration.SGRProjPrjMin1 + av1restoration.SGRProjPrjMax1) / 2,
			},
		},
	}
}

func InitDefaultRestorationCDFs(cdfs RestorationCDFs) error {
	if cdfs.Switchable == nil || cdfs.Wiener == nil || cdfs.SGRProj == nil {
		return entropy.ErrInvalidCDF
	}
	if err := cdfs.Switchable.Init([]uint16{9413, 22581}); err != nil {
		return err
	}
	if err := cdfs.Wiener.Init([]uint16{11570}); err != nil {
		return err
	}
	return cdfs.SGRProj.Init([]uint16{16855})
}

// ReadRestorationUnit decodes the loop_restoration_read_sb_coeffs() syntax for
// one restoration unit and updates refs when AV1's reference state advances.
func (s *DecodeState) ReadRestorationUnit(frameType parser.RestorationType, plane int, refs *RestorationReferences, cdfs RestorationCDFs) (RestorationUnit, error) {
	if s == nil {
		return RestorationUnit{}, ErrInvalidDecodeState
	}
	if frameType == parser.RestorationNone {
		return RestorationUnit{Type: parser.RestorationNone}, nil
	}
	if plane < 0 || plane > 2 || refs == nil || !validRestorationReferences(*refs) {
		return RestorationUnit{}, ErrInvalidDecodeState
	}

	switch frameType {
	case parser.RestorationSwitchable:
		symbol, err := s.Reader.ReadCDF(cdfs.Switchable)
		if err != nil {
			return RestorationUnit{}, err
		}
		switch symbol {
		case 0:
			return RestorationUnit{Type: parser.RestorationNone}, nil
		case 1:
			info, err := readWienerFilter(&s.Reader, plane > 0, refs)
			if err != nil {
				return RestorationUnit{}, err
			}
			return RestorationUnit{Type: parser.RestorationWiener, Wiener: info}, nil
		case 2:
			info, err := readSGRProjFilter(&s.Reader, refs)
			if err != nil {
				return RestorationUnit{}, err
			}
			return RestorationUnit{Type: parser.RestorationSGRProj, SGRProj: info}, nil
		default:
			return RestorationUnit{}, ErrInvalidDecodeState
		}
	case parser.RestorationWiener:
		symbol, err := s.Reader.ReadCDF(cdfs.Wiener)
		if err != nil {
			return RestorationUnit{}, err
		}
		if symbol == 0 {
			return RestorationUnit{Type: parser.RestorationNone}, nil
		}
		info, err := readWienerFilter(&s.Reader, plane > 0, refs)
		if err != nil {
			return RestorationUnit{}, err
		}
		return RestorationUnit{Type: parser.RestorationWiener, Wiener: info}, nil
	case parser.RestorationSGRProj:
		symbol, err := s.Reader.ReadCDF(cdfs.SGRProj)
		if err != nil {
			return RestorationUnit{}, err
		}
		if symbol == 0 {
			return RestorationUnit{Type: parser.RestorationNone}, nil
		}
		info, err := readSGRProjFilter(&s.Reader, refs)
		if err != nil {
			return RestorationUnit{}, err
		}
		return RestorationUnit{Type: parser.RestorationSGRProj, SGRProj: info}, nil
	default:
		return RestorationUnit{}, ErrInvalidDecodeState
	}
}

// WriteRestorationUnit encodes one loop-restoration unit (exact inverse of
// ReadRestorationUnit) and advances refs when a filter is coded.
func WriteRestorationUnit(w *entropy.Writer, frameType parser.RestorationType, plane int, refs *RestorationReferences, cdfs RestorationCDFs, unit RestorationUnit) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	if frameType == parser.RestorationNone {
		return nil
	}
	if plane < 0 || plane > 2 || refs == nil || !validRestorationReferences(*refs) {
		return ErrInvalidDecodeState
	}
	switch frameType {
	case parser.RestorationSwitchable:
		if cdfs.Switchable == nil {
			return ErrInvalidDecodeState
		}
		switch unit.Type {
		case parser.RestorationNone:
			w.WriteCDF(cdfs.Switchable, 0)
			return nil
		case parser.RestorationWiener:
			w.WriteCDF(cdfs.Switchable, 1)
			return writeWienerFilter(w, plane > 0, refs, unit.Wiener)
		case parser.RestorationSGRProj:
			w.WriteCDF(cdfs.Switchable, 2)
			return writeSGRProjFilter(w, refs, unit.SGRProj)
		default:
			return ErrInvalidDecodeState
		}
	case parser.RestorationWiener:
		if cdfs.Wiener == nil {
			return ErrInvalidDecodeState
		}
		if unit.Type == parser.RestorationNone {
			w.WriteCDF(cdfs.Wiener, 0)
			return nil
		}
		if unit.Type != parser.RestorationWiener {
			return ErrInvalidDecodeState
		}
		w.WriteCDF(cdfs.Wiener, 1)
		return writeWienerFilter(w, plane > 0, refs, unit.Wiener)
	case parser.RestorationSGRProj:
		if cdfs.SGRProj == nil {
			return ErrInvalidDecodeState
		}
		if unit.Type == parser.RestorationNone {
			w.WriteCDF(cdfs.SGRProj, 0)
			return nil
		}
		if unit.Type != parser.RestorationSGRProj {
			return ErrInvalidDecodeState
		}
		w.WriteCDF(cdfs.SGRProj, 1)
		return writeSGRProjFilter(w, refs, unit.SGRProj)
	default:
		return ErrInvalidDecodeState
	}
}

func writeWienerFilter(w *entropy.Writer, chroma bool, refs *RestorationReferences, info av1restoration.WienerInfo) error {
	if !chroma {
		if err := writeWienerTap(w, av1restoration.WienerTap0Min, av1restoration.WienerTap0Max, wienerTap0SubexpK, refs.Wiener.VFilter[0], info.VFilter[0]); err != nil {
			return err
		}
	}
	if err := writeWienerTap(w, av1restoration.WienerTap1Min, av1restoration.WienerTap1Max, wienerTap1SubexpK, refs.Wiener.VFilter[1], info.VFilter[1]); err != nil {
		return err
	}
	if err := writeWienerTap(w, av1restoration.WienerTap2Min, av1restoration.WienerTap2Max, wienerTap2SubexpK, refs.Wiener.VFilter[2], info.VFilter[2]); err != nil {
		return err
	}
	if !chroma {
		if err := writeWienerTap(w, av1restoration.WienerTap0Min, av1restoration.WienerTap0Max, wienerTap0SubexpK, refs.Wiener.HFilter[0], info.HFilter[0]); err != nil {
			return err
		}
	}
	if err := writeWienerTap(w, av1restoration.WienerTap1Min, av1restoration.WienerTap1Max, wienerTap1SubexpK, refs.Wiener.HFilter[1], info.HFilter[1]); err != nil {
		return err
	}
	if err := writeWienerTap(w, av1restoration.WienerTap2Min, av1restoration.WienerTap2Max, wienerTap2SubexpK, refs.Wiener.HFilter[2], info.HFilter[2]); err != nil {
		return err
	}
	v0, h0 := info.VFilter[0], info.HFilter[0]
	if chroma {
		v0, h0 = 0, 0
	}
	refs.Wiener = av1restoration.WienerInfo{
		VFilter: av1restoration.NewWienerFilter(v0, info.VFilter[1], info.VFilter[2]),
		HFilter: av1restoration.NewWienerFilter(h0, info.HFilter[1], info.HFilter[2]),
	}
	return nil
}

func writeWienerTap(w *entropy.Writer, minVal, maxVal int, k uint8, ref, val int16) error {
	n := uint32(maxVal - minVal + 1)
	refOff := int(ref) - minVal
	valOff := int(val) - minVal
	if refOff < 0 || refOff >= int(n) || valOff < 0 || valOff >= int(n) {
		return ErrInvalidDecodeState
	}
	return w.WriteRefSubexp(n, k, uint32(refOff), uint32(valOff))
}

func writeSGRProjFilter(w *entropy.Writer, refs *RestorationReferences, info SGRProjInfo) error {
	if info.ParamsIndex >= av1restoration.SGRProjParams {
		return ErrInvalidDecodeState
	}
	w.WriteLiteral(uint32(info.ParamsIndex), 4)
	params := av1restoration.SGRParameterTable[info.ParamsIndex]
	stored := SGRProjInfo{ParamsIndex: info.ParamsIndex}
	switch {
	case params.Radius[0] == 0:
		stored.XQD[0] = 0
		if err := writeSGRProjXQD(w, av1restoration.SGRProjPrjMin1, av1restoration.SGRProjPrjMax1, int(refs.SGRProj.XQD[1]), int(info.XQD[1])); err != nil {
			return err
		}
		stored.XQD[1] = info.XQD[1]
	case params.Radius[1] == 0:
		if err := writeSGRProjXQD(w, av1restoration.SGRProjPrjMin0, av1restoration.SGRProjPrjMax0, int(refs.SGRProj.XQD[0]), int(info.XQD[0])); err != nil {
			return err
		}
		stored.XQD[0] = info.XQD[0]
		stored.XQD[1] = int8(clampInt((1<<av1restoration.SGRProjPrjBits)-int(info.XQD[0]), av1restoration.SGRProjPrjMin1, av1restoration.SGRProjPrjMax1))
	default:
		if err := writeSGRProjXQD(w, av1restoration.SGRProjPrjMin0, av1restoration.SGRProjPrjMax0, int(refs.SGRProj.XQD[0]), int(info.XQD[0])); err != nil {
			return err
		}
		if err := writeSGRProjXQD(w, av1restoration.SGRProjPrjMin1, av1restoration.SGRProjPrjMax1, int(refs.SGRProj.XQD[1]), int(info.XQD[1])); err != nil {
			return err
		}
		stored.XQD = info.XQD
	}
	refs.SGRProj = stored
	return nil
}

func writeSGRProjXQD(w *entropy.Writer, minVal, maxVal, ref, val int) error {
	n := uint32(maxVal - minVal + 1)
	refOff := ref - minVal
	valOff := val - minVal
	if refOff < 0 || refOff >= int(n) || valOff < 0 || valOff >= int(n) {
		return ErrInvalidDecodeState
	}
	return w.WriteRefSubexp(n, av1restoration.SGRProjPrjSubexpK, uint32(refOff), uint32(valOff))
}

func readWienerFilter(r *entropy.Reader, chroma bool, refs *RestorationReferences) (av1restoration.WienerInfo, error) {
	var v0 int16
	var err error
	if chroma {
		v0 = 0
	} else {
		v0, err = readWienerTap(r, av1restoration.WienerTap0Min, av1restoration.WienerTap0Max, wienerTap0SubexpK, refs.Wiener.VFilter[0])
		if err != nil {
			return av1restoration.WienerInfo{}, err
		}
	}
	v1, err := readWienerTap(r, av1restoration.WienerTap1Min, av1restoration.WienerTap1Max, wienerTap1SubexpK, refs.Wiener.VFilter[1])
	if err != nil {
		return av1restoration.WienerInfo{}, err
	}
	v2, err := readWienerTap(r, av1restoration.WienerTap2Min, av1restoration.WienerTap2Max, wienerTap2SubexpK, refs.Wiener.VFilter[2])
	if err != nil {
		return av1restoration.WienerInfo{}, err
	}

	var h0 int16
	if chroma {
		h0 = 0
	} else {
		h0, err = readWienerTap(r, av1restoration.WienerTap0Min, av1restoration.WienerTap0Max, wienerTap0SubexpK, refs.Wiener.HFilter[0])
		if err != nil {
			return av1restoration.WienerInfo{}, err
		}
	}
	h1, err := readWienerTap(r, av1restoration.WienerTap1Min, av1restoration.WienerTap1Max, wienerTap1SubexpK, refs.Wiener.HFilter[1])
	if err != nil {
		return av1restoration.WienerInfo{}, err
	}
	h2, err := readWienerTap(r, av1restoration.WienerTap2Min, av1restoration.WienerTap2Max, wienerTap2SubexpK, refs.Wiener.HFilter[2])
	if err != nil {
		return av1restoration.WienerInfo{}, err
	}

	info := av1restoration.WienerInfo{
		VFilter: av1restoration.NewWienerFilter(v0, v1, v2),
		HFilter: av1restoration.NewWienerFilter(h0, h1, h2),
	}
	refs.Wiener = info
	return info, nil
}

func readWienerTap(r *entropy.Reader, min int, max int, k uint8, ref int16) (int16, error) {
	n := uint32(max - min + 1)
	refOffset := int(ref) - min
	if refOffset < 0 || refOffset >= int(n) {
		return 0, ErrInvalidDecodeState
	}
	v, err := r.ReadRefSubexp(n, k, uint32(refOffset))
	if err != nil {
		return 0, err
	}
	return int16(int(v) + min), nil
}

func readSGRProjFilter(r *entropy.Reader, refs *RestorationReferences) (SGRProjInfo, error) {
	ep, err := r.ReadBits(4)
	if err != nil {
		return SGRProjInfo{}, err
	}
	if ep >= av1restoration.SGRProjParams {
		return SGRProjInfo{}, ErrInvalidDecodeState
	}

	info := SGRProjInfo{ParamsIndex: uint8(ep)}
	params := av1restoration.SGRParameterTable[ep]
	switch {
	case params.Radius[0] == 0:
		info.XQD[0] = 0
		x1, err := readSGRProjXQD(r, av1restoration.SGRProjPrjMin1, av1restoration.SGRProjPrjMax1, int(refs.SGRProj.XQD[1]))
		if err != nil {
			return SGRProjInfo{}, err
		}
		x1Stored, ok := sgrProjXQD(x1)
		if !ok {
			return SGRProjInfo{}, ErrInvalidDecodeState
		}
		info.XQD[1] = x1Stored
	case params.Radius[1] == 0:
		x0, err := readSGRProjXQD(r, av1restoration.SGRProjPrjMin0, av1restoration.SGRProjPrjMax0, int(refs.SGRProj.XQD[0]))
		if err != nil {
			return SGRProjInfo{}, err
		}
		x1 := clampInt((1<<av1restoration.SGRProjPrjBits)-x0, av1restoration.SGRProjPrjMin1, av1restoration.SGRProjPrjMax1)
		x0Stored, ok := sgrProjXQD(x0)
		if !ok {
			return SGRProjInfo{}, ErrInvalidDecodeState
		}
		x1Stored, ok := sgrProjXQD(x1)
		if !ok {
			return SGRProjInfo{}, ErrInvalidDecodeState
		}
		info.XQD[0] = x0Stored
		info.XQD[1] = x1Stored
	default:
		x0, err := readSGRProjXQD(r, av1restoration.SGRProjPrjMin0, av1restoration.SGRProjPrjMax0, int(refs.SGRProj.XQD[0]))
		if err != nil {
			return SGRProjInfo{}, err
		}
		x1, err := readSGRProjXQD(r, av1restoration.SGRProjPrjMin1, av1restoration.SGRProjPrjMax1, int(refs.SGRProj.XQD[1]))
		if err != nil {
			return SGRProjInfo{}, err
		}
		x0Stored, ok := sgrProjXQD(x0)
		if !ok {
			return SGRProjInfo{}, ErrInvalidDecodeState
		}
		x1Stored, ok := sgrProjXQD(x1)
		if !ok {
			return SGRProjInfo{}, ErrInvalidDecodeState
		}
		info.XQD[0] = x0Stored
		info.XQD[1] = x1Stored
	}

	refs.SGRProj = info
	return info, nil
}

func readSGRProjXQD(r *entropy.Reader, min int, max int, ref int) (int, error) {
	n := uint32(max - min + 1)
	refOffset := ref - min
	if refOffset < 0 || refOffset >= int(n) {
		return 0, ErrInvalidDecodeState
	}
	v, err := r.ReadRefSubexp(n, av1restoration.SGRProjPrjSubexpK, uint32(refOffset))
	if err != nil {
		return 0, err
	}
	return int(v) + min, nil
}

func sgrProjXQD(v int) (int8, bool) {
	const (
		min = -1 << 7
		max = 1<<7 - 1
	)
	if v < min || v > max {
		return 0, false
	}
	return int8(v), true
}

func validRestorationReferences(refs RestorationReferences) bool {
	if !validDecodedWienerInfo(refs.Wiener) {
		return false
	}
	return refs.SGRProj.ParamsIndex < av1restoration.SGRProjParams &&
		refs.SGRProj.XQD[0] >= av1restoration.SGRProjPrjMin0 &&
		refs.SGRProj.XQD[0] <= av1restoration.SGRProjPrjMax0 &&
		refs.SGRProj.XQD[1] >= av1restoration.SGRProjPrjMin1 &&
		refs.SGRProj.XQD[1] <= av1restoration.SGRProjPrjMax1
}

func validDecodedWienerInfo(info av1restoration.WienerInfo) bool {
	return validDecodedWienerFilter(info.VFilter) && validDecodedWienerFilter(info.HFilter)
}

func validDecodedWienerFilter(filter av1restoration.WienerFilter) bool {
	sum := int32(0)
	for i := range av1restoration.WienerWin {
		sum += int32(filter[i])
	}
	return filter[7] == 0 &&
		filter[0] == filter[6] &&
		filter[1] == filter[5] &&
		filter[2] == filter[4] &&
		int(filter[0]) >= av1restoration.WienerTap0Min &&
		int(filter[0]) <= av1restoration.WienerTap0Max &&
		int(filter[1]) >= av1restoration.WienerTap1Min &&
		int(filter[1]) <= av1restoration.WienerTap1Max &&
		int(filter[2]) >= av1restoration.WienerTap2Min &&
		int(filter[2]) <= av1restoration.WienerTap2Max &&
		sum == 0
}

func clampInt(v int, lo int, hi int) int {
	return min(max(v, lo), hi)
}
