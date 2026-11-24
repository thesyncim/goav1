package entropy

import "encoding/binary"

// TxDim is the part of dav1d's TxfmInfo (src/tables.h) read_tx_tree uses:
// the size in 4x4 units and its log2, t_dim->max and t_dim->sub.
type TxDim struct {
	W4, H4, Log2W, Log2H, Max, Sub uint8
}

// TxDims is dav1d_txfm_dimensions in RectTxfmSize order (TX_4X4 ..
// RTX_64X16); the tile package checks it against its own table.
var TxDims = [19]TxDim{
	{1, 1, 0, 0, 0, 0},   // TX_4X4
	{2, 2, 1, 1, 1, 0},   // TX_8X8
	{4, 4, 2, 2, 2, 1},   // TX_16X16
	{8, 8, 3, 3, 3, 2},   // TX_32X32
	{16, 16, 4, 4, 4, 3}, // TX_64X64
	{1, 2, 0, 1, 1, 0},   // RTX_4X8
	{2, 1, 1, 0, 1, 0},   // RTX_8X4
	{2, 4, 1, 2, 2, 1},   // RTX_8X16
	{4, 2, 2, 1, 2, 1},   // RTX_16X8
	{4, 8, 2, 3, 3, 2},   // RTX_16X32
	{8, 4, 3, 2, 3, 2},   // RTX_32X16
	{8, 16, 3, 4, 4, 3},  // RTX_32X64
	{16, 8, 4, 3, 4, 3},  // RTX_64X32
	{1, 4, 0, 2, 2, 5},   // RTX_4X16
	{4, 1, 2, 0, 2, 6},   // RTX_16X4
	{2, 8, 1, 3, 3, 7},   // RTX_8X32
	{8, 2, 3, 1, 3, 8},   // RTX_32X8
	{4, 16, 2, 4, 4, 9},  // RTX_16X64
	{16, 4, 4, 2, 4, 10}, // RTX_64X16
}

// TxTree is the block-invariant state of dav1d's read_tx_tree recursion:
// the txpart CDFs, the above/left transform contexts (superblock-local),
// the split masks being built (b->tx_split0/1) and the block's visible
// right and bottom edges in superblock-local 4x4 units.
type TxTree struct {
	CDF           *[7][3]CDF
	A, L          *[32]uint8
	Right, Bottom int
	Split         [2]uint16
}

// DecodeVarTxTree runs dav1d's read_vartx_tree over all max_ytx units of a
// block while keeping the range coder state (dif, rng, cnt) in registers.
func (c *Cursor) DecodeVarTxTree(t *TxTree, from uint, bx4, by4, vw4, vh4 int, haveTop, haveLeft bool) {
	d := &TxDims[from]
	tw4, th4 := int(d.W4), int(d.H4)
	update := c.allowCDFUpdate
	dif, rng, cnt := c.dif, uint32(c.rng), int32(c.cnt)
	if vw4 <= tw4 && vh4 <= th4 {
		dif, rng, cnt = c.txTree(t, update, from, 0, 0, 0, bx4, by4, haveTop, haveLeft, dif, rng, cnt)
	} else {
		for y, yOff := 0, 0; y < vh4; y, yOff = y+th4, yOff+1 {
			ht := haveTop || y > 0
			for x, xOff := 0, 0; x < vw4; x, xOff = x+tw4, xOff+1 {
				dif, rng, cnt = c.txTree(t, update, from, 0, xOff, yOff, bx4+x, by4+y, ht, haveLeft || x > 0, dif, rng, cnt)
			}
		}
	}
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
}

// DecodeTxTree is one read_tx_tree call of dav1d's read_vartx_tree for the
// max_ytx unit at (x4, y4) with offsets (xOff, yOff), the range state in
// registers through the recursion. haveTop and haveLeft gate the split
// context the way libaom's edge-seeded transform contexts do.
func (c *Cursor) DecodeTxTree(t *TxTree, from uint, xOff, yOff, x4, y4 int, haveTop, haveLeft bool) {
	dif, rng, cnt := c.txTree(t, c.allowCDFUpdate, from, 0, xOff, yOff, x4, y4, haveTop, haveLeft, c.dif, uint32(c.rng), int32(c.cnt))
	c.dif, c.rng, c.cnt = dif, uint16(rng), int16(cnt)
}

func (c *Cursor) txTree(t *TxTree, update bool, from, depth uint, xOff, yOff, x4, y4 int, haveTop, haveLeft bool, dif uint64, rng uint32, cnt int32) (uint64, uint32, int32) {
	d := &TxDims[from]
	txw, txh := d.Log2W, d.Log2H
	split := uint(0)
	if depth < 2 && from > 0 {
		cat := 2*(4-uint(d.Max)) - depth
		a, l := uint(0), uint(0)
		if haveTop && t.A[x4&31] < txw {
			a = 1
		}
		if haveLeft && t.L[y4&31] < txh {
			l = 1
		}
		v := &t.CDF[cat][a+l].values
		if traceEntropyReads {
			traceCDFRead(v[0], 2, dif, rng, c.coefTell(cnt))
		}
		split, dif, rng = readBool(v, dif, rng)
		dif, rng, cnt = coefNorm(dif, rng, cnt)
		if cnt < 0 {
			dif, cnt = c.coefRefill(dif, cnt)
		}
		if update {
			updateCDF2(v, int(split))
		}
		t.Split[depth&1] |= uint16(split) << ((yOff*4 + xOff) & 15)
	}
	if split != 0 && d.Max > 1 {
		sub := uint(d.Sub)
		sd := &TxDims[sub]
		txsw, txsh := int(sd.W4), int(sd.H4)
		depth++
		dif, rng, cnt = c.txTree(t, update, sub, depth, xOff*2, yOff*2, x4, y4, haveTop, haveLeft, dif, rng, cnt)
		if txw >= txh && x4+txsw < t.Right {
			dif, rng, cnt = c.txTree(t, update, sub, depth, xOff*2+1, yOff*2, x4+txsw, y4, haveTop, true, dif, rng, cnt)
		}
		if txh >= txw && y4+txsh < t.Bottom {
			dif, rng, cnt = c.txTree(t, update, sub, depth, xOff*2, yOff*2+1, x4, y4+txsh, true, haveLeft, dif, rng, cnt)
			if txw >= txh && x4+txsw < t.Right {
				dif, rng, cnt = c.txTree(t, update, sub, depth, xOff*2+1, yOff*2+1, x4+txsw, y4+txsh, true, true, dif, rng, cnt)
			}
		}
		return dif, rng, cnt
	}
	markW, markH := txw, txh
	if split != 0 {
		markW, markH = 0, 0
	}
	txCtxSet(t.A[x4&31:], txw, markW)
	txCtxSet(t.L[y4&31:], txh, markH)
	return dif, rng, cnt
}

// txCtxSet is dav1d_memset_pow2: dst[:1<<log2n] = v with one store per up
// to eight bytes; log2n is at most 4 here (a 64-wide transform).
func txCtxSet(dst []uint8, log2n uint8, v uint8) {
	x := uint64(v) * 0x0101010101010101
	switch log2n {
	case 0:
		dst[0] = v
	case 1:
		binary.LittleEndian.PutUint16(dst, uint16(x))
	case 2:
		binary.LittleEndian.PutUint32(dst, uint32(x))
	case 3:
		binary.LittleEndian.PutUint64(dst, x)
	default:
		binary.LittleEndian.PutUint64(dst, x)
		binary.LittleEndian.PutUint64(dst[8:], x)
	}
}
