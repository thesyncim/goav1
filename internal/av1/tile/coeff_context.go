package tile

import (
	"encoding/binary"
	"unsafe"
)

type TXBContext struct {
	TXBSkipContext uint8
	DCSignContext  uint8
}

type CoeffContextRequest struct {
	Plane      uint8
	PlaneBlock BlockSize
	Size       TransformSize

	X4        uint8
	Y4        uint8
	VisibleW4 uint8
	VisibleH4 uint8
}

type CoeffEntropyContext struct {
	Above [3][MaxBlockModeSlots]uint8
	Left  [3][MaxBlockModeSlots]uint8
}

var coeffContextSignDelta = [32]int8{1: -1, 2: 1}

var coeffSkipContexts = [5][5]uint8{
	{1, 2, 2, 2, 3},
	{2, 4, 4, 4, 5},
	{2, 4, 4, 4, 5},
	{2, 4, 4, 4, 5},
	{3, 5, 5, 5, 6},
}

func (c *CoeffEntropyContext) TXBContext(req CoeffContextRequest) (TXBContext, error) {
	if c == nil {
		return TXBContext{}, ErrInvalidDecodeState
	}
	txDims, blockDims, _, _, err := validateCoeffContextRequestWithSpan(req)
	if err != nil {
		return TXBContext{}, err
	}
	return c.txbContextKnown(req, txDims, blockDims)
}

// TXBContextTrusted derives coefficient contexts for callers that have already
// validated req, txDims, blockDims, and neighbor state. It is intended for
// internal hot paths that have selected a shape-specific coefficient writer.
func (c *CoeffEntropyContext) TXBContextTrusted(req CoeffContextRequest, txDims TransformDimensions, blockDims BlockDimensions) TXBContext {
	return c.txbContextReq(req, txDims, blockDims)
}

func (c *CoeffEntropyContext) txbContextKnown(req CoeffContextRequest, txDims TransformDimensions, blockDims BlockDimensions) (TXBContext, error) {
	dcSign := int8(0)
	plane := int(req.Plane)
	x4 := int(req.X4)
	y4 := int(req.Y4)
	for k := 0; k < int(txDims.W4); k++ {
		sign := c.Above[plane][x4+k] >> CoeffContextBits
		switch sign {
		case 0:
		case 1:
			dcSign--
		case 2:
			dcSign++
		default:
			return TXBContext{}, ErrInvalidDecodeState
		}
	}
	for k := 0; k < int(txDims.H4); k++ {
		sign := c.Left[plane][y4+k] >> CoeffContextBits
		switch sign {
		case 0:
		case 1:
			dcSign--
		case 2:
			dcSign++
		default:
			return TXBContext{}, ErrInvalidDecodeState
		}
	}

	txbSkipCtx := uint8(0)
	if req.Plane == 0 {
		if blockDims.W4 != txDims.W4 || blockDims.H4 != txDims.H4 {
			top := coeffContextMagnitude(c.Above[plane][x4 : x4+int(txDims.W4)])
			left := coeffContextMagnitude(c.Left[plane][y4 : y4+int(txDims.H4)])
			txbSkipCtx = coeffSkipContexts[top][left]
		}
	} else {
		ctxBase := coeffEntropyBase(c.Above[plane][x4:x4+int(txDims.W4)],
			c.Left[plane][y4:y4+int(txDims.H4)])
		ctxOffset := uint8(7)
		if int(blockDims.W4)*int(blockDims.H4) > int(txDims.W4)*int(txDims.H4) {
			ctxOffset = 10
		}
		txbSkipCtx = ctxBase + ctxOffset
	}

	return TXBContext{
		TXBSkipContext: txbSkipCtx,
		DCSignContext:  coeffDCSignContext(dcSign),
	}, nil
}

func (c *CoeffEntropyContext) txbContextReq(req CoeffContextRequest, txDims TransformDimensions, blockDims BlockDimensions) TXBContext {
	return c.txbContextTrusted(req.Plane, int(req.X4), int(req.Y4), int(txDims.W4), int(txDims.H4), int(blockDims.W4), int(blockDims.H4))
}

// txbContextTrusted is dav1d's get_skip_ctx and get_dc_sign_ctx
// (src/recon_tmpl.c) for the plane TXB at a/l offsets x4, y4: the context
// bytes are read as one or two words keyed on the transform side, the way
// dav1d switches on t_dim->lw/lh.
func (c *CoeffEntropyContext) txbContextTrusted(plane uint8, x4, y4, w4, h4, blockW4, blockH4 int) TXBContext {
	dcSign := int8(0)
	var aboveRow, leftRow *[MaxBlockModeSlots]uint8
	switch plane {
	case 0:
		aboveRow = &c.Above[0]
		leftRow = &c.Left[0]
	case 1:
		aboveRow = &c.Above[1]
		leftRow = &c.Left[1]
	default:
		aboveRow = &c.Above[2]
		leftRow = &c.Left[2]
	}
	if w4 > 0 {
		_ = aboveRow[x4+w4-1]
	}
	if h4 > 0 {
		_ = leftRow[y4+h4-1]
	}

	txbSkipCtx := uint8(0)
	if w4 == 1 && h4 == 1 {
		top := aboveRow[x4]
		left := leftRow[y4]
		// dav1d get_dc_sign_ctx TX_4X4: the sign flags summed without a table.
		dcSign = int8((top>>(CoeffContextBits+1))&1+(left>>(CoeffContextBits+1))&1) - int8((top>>CoeffContextBits)&1+(left>>CoeffContextBits)&1)
		if plane == 0 {
			if blockW4 != w4 || blockH4 != h4 {
				top &= CoeffContextMask
				top = min(top, 4)
				left &= CoeffContextMask
				left = min(left, 4)
				txbSkipCtx = coeffSkipContexts[top][left]
			}
		} else {
			if top != 0 {
				txbSkipCtx++
			}
			if left != 0 {
				txbSkipCtx++
			}
			txbSkipCtx += 7
			if blockW4*blockH4 > w4*h4 {
				txbSkipCtx += 3
			}
		}
		return TXBContext{
			TXBSkipContext: txbSkipCtx,
			DCSignContext:  coeffDCSignContext(dcSign),
		}
	}
	topMag, topSign := coeffContextWords(aboveRow, x4, w4)
	leftMag, leftSign := coeffContextWords(leftRow, y4, h4)
	dcSign = topSign + leftSign
	if plane == 0 {
		if blockW4 != w4 || blockH4 != h4 {
			top := topMag & CoeffContextMask
			top = min(top, 4)
			left := leftMag & CoeffContextMask
			left = min(left, 4)
			txbSkipCtx = coeffSkipContexts[top][left]
		}
	} else {
		topNonZero := topMag != 0
		leftNonZero := leftMag != 0
		ctxBase := uint8(0)
		if topNonZero {
			ctxBase++
		}
		if leftNonZero {
			ctxBase++
		}
		ctxOffset := uint8(7)
		if blockW4*blockH4 > w4*h4 {
			ctxOffset = 10
		}
		txbSkipCtx = ctxBase + ctxOffset
	}

	return TXBContext{
		TXBSkipContext: txbSkipCtx,
		DCSignContext:  coeffDCSignContext(dcSign),
	}
}

func (c *CoeffEntropyContext) MarkTXB(req CoeffContextRequest, result TXBDecodeResult) error {
	value := uint8(0)
	if !result.AllZero && result.EOB > 0 {
		if !validCoeffEntropyValue(result.CulLevel) {
			return ErrInvalidDecodeState
		}
		value = result.CulLevel
	}
	return c.setTXBContext(req, value)
}

func (c *CoeffEntropyContext) markTXBKnown(req CoeffContextRequest, txDims TransformDimensions, visibleW int, visibleH int, result TXBDecodeResult) {
	value := uint8(0)
	if result.EOB > 0 {
		value = result.CulLevel
	}
	c.setTXBContextReq(req, txDims, visibleW, visibleH, value)
}

func (c *CoeffEntropyContext) ResetBlock(plane int, block BlockSize, x4 int, y4 int) error {
	if c == nil || plane < 0 || plane >= 3 || x4 < 0 || y4 < 0 {
		return ErrInvalidDecodeState
	}
	dims, ok := block.Dimensions()
	if !ok ||
		x4+int(dims.W4) > MaxBlockModeSlots ||
		y4+int(dims.H4) > MaxBlockModeSlots {
		return ErrInvalidDecodeState
	}
	clear(c.Above[plane][x4 : x4+int(dims.W4)])
	clear(c.Left[plane][y4 : y4+int(dims.H4)])
	return nil
}

func (s *DecodeState) ReadCoefficientsTXBWithContext(cdfs *CoeffCDFs, ctx *CoeffEntropyContext, ctxReq CoeffContextRequest, req TXBDecodeRequest, coeffs []int16, scan []int16, levelsScratch []uint8) (TXBDecodeResult, error) {
	req, allZero, err := s.ReadTXBSkipWithContext(cdfs, ctx, ctxReq, req)
	if err != nil {
		return TXBDecodeResult{}, err
	}
	req.TXBSkipKnown = true
	req.TXBSkip = allZero
	result, err := s.ReadCoefficientsTXB(cdfs, req, coeffs, scan, levelsScratch)
	if err != nil {
		return TXBDecodeResult{}, err
	}
	if err := ctx.MarkTXB(ctxReq, result); err != nil {
		return TXBDecodeResult{}, err
	}
	return result, nil
}

func (s *DecodeState) ReadTXBSkipWithContext(cdfs *CoeffCDFs, ctx *CoeffEntropyContext, ctxReq CoeffContextRequest, req TXBDecodeRequest) (TXBDecodeRequest, bool, error) {
	if ctx == nil {
		return TXBDecodeRequest{}, false, ErrInvalidDecodeState
	}
	plane, err := CoeffPlaneTypeForPlane(int(ctxReq.Plane))
	if err != nil {
		return TXBDecodeRequest{}, false, err
	}
	txbCtx, err := ctx.TXBContext(ctxReq)
	if err != nil {
		return TXBDecodeRequest{}, false, err
	}
	req.Size = ctxReq.Size
	req.Plane = plane
	req.TXBSkipContext = txbCtx.TXBSkipContext
	req.DCSignContext = txbCtx.DCSignContext
	allZero, err := s.ReadTXBSkip(cdfs, TXBSkipRequest{Size: req.Size, Context: req.TXBSkipContext})
	if err != nil {
		return TXBDecodeRequest{}, false, err
	}
	return req, allZero, nil
}

func (c *CoeffEntropyContext) setTXBContext(req CoeffContextRequest, value uint8) error {
	if c == nil || !validCoeffEntropyValue(value) {
		return ErrInvalidDecodeState
	}
	txDims, _, visibleW, visibleH, err := validateCoeffContextRequestWithSpan(req)
	if err != nil {
		return err
	}
	c.setTXBContextReq(req, txDims, visibleW, visibleH, value)
	return nil
}

func (c *CoeffEntropyContext) setTXBContextReq(req CoeffContextRequest, txDims TransformDimensions, visibleW int, visibleH int, value uint8) {
	c.setTXBContextKnown(req.Plane, int(req.X4), int(req.Y4), int(txDims.W4), int(txDims.H4), visibleW, visibleH, value)
}

// setTXBContextKnown is the a/l coefficient-context write after the plane TXB
// at x4, y4: dav1d's dav1d_memset_likely_pow2 of the cul_level byte.
func (c *CoeffEntropyContext) setTXBContextKnown(plane uint8, x4, y4, w4, h4, visibleW, visibleH int, value uint8) {
	p := int(plane)
	setCoeffContextRun(&c.Above[p], x4, w4, visibleW, value)
	setCoeffContextRun(&c.Left[p], y4, h4, visibleH, value)
}

// setCoeffContextRun is dav1d's dav1d_memset_likely_pow2 for the a/l coefficient
// contexts: a TXB side of 1, 2, 4, 8 or 16 units fully inside the frame is one
// or two word stores (the width the next get_skip_ctx/get_dc_sign_ctx reads
// back), anything else stores the visible bytes and zeroes the rest.
func setCoeffContextRun(row *[MaxBlockModeSlots]uint8, x, n, visible int, value uint8) {
	const ones = 0x0101010101010101
	if n <= 0 {
		return
	}
	_ = row[x+n-1]
	if visible >= n {
		p := unsafe.Pointer(&row[x])
		v := uint64(value) * ones
		switch n {
		case 1:
			row[x] = value
			return
		case 2:
			*(*uint16)(p) = uint16(v)
			return
		case 4:
			*(*uint32)(p) = uint32(v)
			return
		case 8:
			*(*uint64)(p) = v
			return
		case 16:
			*(*uint64)(p) = v
			*(*uint64)(unsafe.Add(p, 8)) = v
			return
		}
		visible = n
	}
	for k := 0; k < visible; k++ {
		row[x+k] = value
	}
	for k := visible; k < n; k++ {
		row[x+k] = 0
	}
}

func validateCoeffContextRequest(req CoeffContextRequest) (TransformDimensions, BlockDimensions, error) {
	txDims, blockDims, _, _, err := validateCoeffContextRequestWithSpan(req)
	return txDims, blockDims, err
}

func validateCoeffContextRequestWithSpan(req CoeffContextRequest) (TransformDimensions, BlockDimensions, int, int, error) {
	if req.Plane >= 3 {
		return TransformDimensions{}, BlockDimensions{}, 0, 0, ErrInvalidDecodeState
	}
	txDims, ok := req.Size.Dimensions()
	if !ok {
		return TransformDimensions{}, BlockDimensions{}, 0, 0, ErrInvalidDecodeState
	}
	blockDims, ok := req.PlaneBlock.Dimensions()
	x4 := int(req.X4)
	y4 := int(req.Y4)
	if !ok ||
		x4+int(txDims.W4) > MaxBlockModeSlots ||
		y4+int(txDims.H4) > MaxBlockModeSlots {
		return TransformDimensions{}, BlockDimensions{}, 0, 0, ErrInvalidDecodeState
	}
	visibleW, visibleH, err := coeffVisibleSpan(req, txDims)
	if err != nil {
		return TransformDimensions{}, BlockDimensions{}, 0, 0, err
	}
	return txDims, blockDims, visibleW, visibleH, nil
}

func coeffVisibleSpan(req CoeffContextRequest, txDims TransformDimensions) (int, int, error) {
	visibleW := int(txDims.W4)
	if req.VisibleW4 != 0 {
		visibleW = int(req.VisibleW4)
	}
	visibleH := int(txDims.H4)
	if req.VisibleH4 != 0 {
		visibleH = int(req.VisibleH4)
	}
	if visibleW > int(txDims.W4) || visibleH > int(txDims.H4) {
		return 0, 0, ErrInvalidDecodeState
	}
	return visibleW, visibleH, nil
}

func coeffContextMagnitude(values []uint8) uint8 {
	out := uint8(0)
	for _, value := range values {
		out |= value
	}
	out &= CoeffContextMask
	if out > 4 {
		out = 4
	}
	return out
}

func coeffEntropyBase(above []uint8, left []uint8) uint8 {
	base := uint8(0)
	for _, value := range above {
		if value != 0 {
			base++
			break
		}
	}
	for _, value := range left {
		if value != 0 {
			base++
			break
		}
	}
	return base
}

func coeffDCSignContext(dcSign int8) uint8 {
	// dav1d get_dc_sign_ctx: (s != 0) + (s > 0), branch-free.
	return b2u8(dcSign != 0) + b2u8(dcSign > 0)
}

func b2u8(v bool) uint8 {
	var r uint8
	if v {
		r = 1
	}
	return r
}

func validCoeffEntropyValue(value uint8) bool {
	if value > CoeffContextMask+(2<<CoeffContextBits) {
		return false
	}
	return value>>CoeffContextBits <= 2
}

// coeffContextWords returns the OR of the n context bytes at row[x:] and their
// DC-sign sum (positive minus negative entries), reading them as one or two
// words the way dav1d's get_skip_ctx and get_dc_sign_ctx do. A context byte
// holds the magnitude in its low CoeffContextBits bits, then a negative flag
// (bit 3) and a positive flag (bit 4). Transform sides are 1, 2, 4, 8 or 16
// units; anything else takes the per-byte loop.
func coeffContextWords(row *[MaxBlockModeSlots]uint8, x, n int) (uint8, int8) {
	const ones = 0x0101010101010101
	var lo, hi uint64
	switch n {
	case 1:
		lo = uint64(row[x])
	case 2:
		lo = uint64(binary.LittleEndian.Uint16(row[x : x+2]))
	case 4:
		lo = uint64(binary.LittleEndian.Uint32(row[x : x+4]))
	case 8:
		lo = binary.LittleEndian.Uint64(row[x : x+8])
	case 16:
		lo = binary.LittleEndian.Uint64(row[x : x+8])
		hi = binary.LittleEndian.Uint64(row[x+8 : x+16])
	default:
		or := uint8(0)
		sign := int8(0)
		for k := 0; k < n; k++ {
			v := row[x+k]
			or |= v
			sign += coeffContextSignDelta[v>>CoeffContextBits]
		}
		return or, sign
	}
	neg := (((lo >> CoeffContextBits) & ones) + ((hi >> CoeffContextBits) & ones)) * ones >> 56
	pos := (((lo >> (CoeffContextBits + 1)) & ones) + ((hi >> (CoeffContextBits + 1)) & ones)) * ones >> 56
	v := lo | hi
	v |= v >> 32
	v |= v >> 16
	v |= v >> 8
	return uint8(v), int8(pos) - int8(neg)
}
