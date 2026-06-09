package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// WriteIntraTransformType codes one luma intra tx_type, the forward of
// ReadIntraTransformType. The no-symbol cases mirror the reader exactly:
// skip/lossless/qindex-0 blocks and single-type ext-tx sets code nothing and
// require DCT_DCT. The symbol index comes from the shared ExtTXSymbolForType
// inverse of the reader's ExtTXTypeFromSymbol.
func WriteIntraTransformType(w *entropy.Writer, cdfs *TransformTypeCDFs, req IntraTransformTypeRequest, typ transform.Type) error {
	if w == nil {
		return ErrInvalidDecodeState
	}
	if !req.Size.Valid() || !req.Mode.Valid() {
		return ErrInvalidDecodeState
	}
	if req.SkipTransform || req.Lossless || (req.QIndexKnown && req.QIndex == 0) {
		if typ != transform.TypeDCTDCT {
			return ErrInvalidDecodeState
		}
		return nil
	}
	set, err := ExtTXSetTypeFor(req.Size, false, req.ReducedTXSet)
	if err != nil {
		return err
	}
	symbols, err := ExtTXTypeCount(set)
	if err != nil {
		return err
	}
	if symbols <= 1 {
		if typ != transform.TypeDCTDCT {
			return ErrInvalidDecodeState
		}
		return nil
	}
	index, err := ExtTXSetIndex(req.Size, false, req.ReducedTXSet)
	if err != nil {
		return err
	}
	square, err := TransformSizeSquare(req.Size)
	if err != nil {
		return err
	}
	cdf, err := cdfs.IntraCDF(index, square, req.Mode, symbols)
	if err != nil {
		return err
	}
	symbol, err := ExtTXSymbolForType(set, typ)
	if err != nil {
		return err
	}
	w.WriteCDF(cdf, symbol)
	return nil
}
