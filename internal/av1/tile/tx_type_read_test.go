package tile

import "github.com/thesyncim/goav1/internal/av1/transform"

// ReadIntraTransformType and ReadInterTransformType are the libaom-shaped luma
// tx_type readers the writer round-trip tests decode with; the decoder reads
// tx_type through CoefBlock.readTxtp.
func (s *DecodeState) ReadIntraTransformType(cdfs *TransformTypeCDFs, req IntraTransformTypeRequest) (transform.Type, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	if !req.Size.Valid() || !req.Mode.Valid() {
		return 0, ErrInvalidDecodeState
	}
	if req.SkipTransform || req.Lossless || (req.QIndexKnown && req.QIndex == 0) {
		return transform.TypeDCTDCT, nil
	}
	p := &txTypeReadPlans[0][txTypePlanReduced(req.ReducedTXSet)][req.Size]
	if !p.ok {
		return 0, ErrInvalidDecodeState
	}
	if p.symbols <= 1 {
		return transform.TypeDCTDCT, nil
	}
	cdf, err := cdfs.IntraCDF(p.index, p.square, req.Mode, p.symbols)
	if err != nil {
		return 0, err
	}
	symbol, err := s.ReadSymbol(cdf)
	if err != nil {
		return 0, err
	}
	return p.typeFromSymbol(symbol)
}

func (s *DecodeState) ReadInterTransformType(cdfs *TransformTypeCDFs, req InterTransformTypeRequest) (transform.Type, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	if !req.Size.Valid() {
		return 0, ErrInvalidDecodeState
	}
	if req.SkipTransform || req.Lossless || (req.QIndexKnown && req.QIndex == 0) {
		return transform.TypeDCTDCT, nil
	}
	p := &txTypeReadPlans[1][txTypePlanReduced(req.ReducedTXSet)][req.Size]
	if !p.ok {
		return 0, ErrInvalidDecodeState
	}
	if p.symbols <= 1 {
		return transform.TypeDCTDCT, nil
	}
	cdf, err := cdfs.InterCDF(p.index, p.square, p.symbols)
	if err != nil {
		return 0, err
	}
	symbol, err := s.ReadSymbol(cdf)
	if err != nil {
		return 0, err
	}
	return p.typeFromSymbol(symbol)
}

// txTypeReadPlan is everything the luma tx_type read derives from the
// transform size alone: the allowed set, its CDF index and symbol count and
// the square size that selects the CDF. dav1d hard-codes the same choices in
// decode_coefs; these plans are built once from the set functions so they
// match them exactly.
type txTypeReadPlan struct {
	set     ExtTXSetType
	index   int
	square  TransformSize
	symbols int
	ok      bool
}

// txTypeReadPlans is indexed by [inter][reduced][size].
var txTypeReadPlans = func() (plans [2][2][transformSizeCount]txTypeReadPlan) {
	for inter := range 2 {
		for reduced := range 2 {
			for size := TransformSize(0); size < transformSizeCount; size++ {
				p := &plans[inter][reduced][size]
				set, err := ExtTXSetTypeFor(size, inter == 1, reduced == 1)
				if err != nil {
					continue
				}
				symbols, err := ExtTXTypeCount(set)
				if err != nil {
					continue
				}
				square, err := TransformSizeSquare(size)
				if err != nil {
					continue
				}
				p.set, p.symbols, p.square, p.ok = set, symbols, square, true
				if symbols > 1 {
					index, err := ExtTXSetIndex(size, inter == 1, reduced == 1)
					if err != nil {
						p.ok = false
						continue
					}
					p.index = index
				}
			}
		}
	}
	return plans
}()

func txTypePlanReduced(reduced bool) int {
	if reduced {
		return 1
	}
	return 0
}

func (p *txTypeReadPlan) typeFromSymbol(symbol int) (transform.Type, error) {
	if symbol < 0 || symbol >= p.symbols {
		return 0, ErrInvalidDecodeState
	}
	return extTXTypeInv[p.set][symbol], nil
}
