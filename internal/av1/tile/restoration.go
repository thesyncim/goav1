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

func (info SGRProjInfo) XQDInt() [2]int {
	return [2]int{int(info.XQD[0]), int(info.XQD[1])}
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
