package encoder

import (
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// pframe_delta_q.go carries the per-superblock quantizer of cyclic refresh
// through the inter block coder: the quantizers and lambda follow the
// superblock's qindex (SVT-AV1 derives the full lambda from the SB qindex,
// md_process.c av1_lambda_assign_md), and the first coded block of every
// superblock signals the change as delta_qindex (libaom bitstream.c
// write_delta_qindex; read side tile.DecodeState.ReadBlockDeltas).

// setInterLambda derives the inter RD multiplier and the full-pel search's
// SAD-per-bit from the current luma quantizer:
// av1_compute_rd_mult_based_on_qindex at 8-bit (the DC step squared scaled
// by the inter rate multiplier) and av1's sad-per-bit formula.
func (st *lossyEncodeState) setInterLambda() {
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	st.sadPerBit = realtimeSADPerBit(st.yQuant, st.color.BitDepth)
}

// useSBQuant switches the quantizer and lambda state to the qindex of the
// superblock containing luma position (px, py). It is a pure function of the
// position, so serial, split and wavefront walks see identical state.
func (st *lossyEncodeState) useSBQuant(px, py int) error {
	if st.sbQIndex == nil {
		return nil
	}
	q := st.sbQIndex[(py/64)*st.sbQCols+px/64]
	if q == st.quantQ {
		return nil
	}
	for plane, dst := range []*quantize.Quantizer{&st.yQuant, &st.uQuant, &st.vQuant} {
		if st.color.MonoChrome && plane != 0 {
			continue
		}
		pq, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, q, st.color.BitDepth, quantize.Plane(plane))
		if err != nil {
			return err
		}
		*dst = pq
	}
	st.setInterLambda()
	st.quantQ = q
	return nil
}

// writeSBDeltaQ codes delta_qindex after the block's skip flag when the
// decoder reads one: the block opens its superblock and is not a whole-SB
// skip (read_delta_qindex).
func (st *lossyEncodeState) writeSBDeltaQ(block tile.BlockVisit, skip bool) error {
	if st.sbQIndex == nil {
		return nil
	}
	if block.MICol%16 != 0 || block.MIRow%16 != 0 {
		return nil
	}
	if block.Size == tile.BlockSize64x64 && skip {
		return nil
	}
	q := st.sbQIndex[(int(block.MIRow)/16)*st.sbQCols+int(block.MICol)/16]
	if err := tile.WriteSignedDelta(st.w, &st.deltaQCDF, int(q)-int(st.deltaQCur)); err != nil {
		return err
	}
	st.deltaQCur = q
	return nil
}
