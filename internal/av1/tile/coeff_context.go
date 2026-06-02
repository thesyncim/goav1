package tile

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

func (c *CoeffEntropyContext) txbContextKnown(req CoeffContextRequest, txDims TransformDimensions, blockDims BlockDimensions) (TXBContext, error) {
	dcSign := 0
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

	txbSkipCtx := 0
	if req.Plane == 0 {
		if blockDims.W4 != txDims.W4 || blockDims.H4 != txDims.H4 {
			top := coeffContextMagnitude(c.Above[plane][x4 : x4+int(txDims.W4)])
			left := coeffContextMagnitude(c.Left[plane][y4 : y4+int(txDims.H4)])
			txbSkipCtx = int(coeffSkipContexts[top][left])
		}
	} else {
		ctxBase := coeffEntropyBase(c.Above[plane][x4:x4+int(txDims.W4)],
			c.Left[plane][y4:y4+int(txDims.H4)])
		ctxOffset := 7
		if int(blockDims.W4)*int(blockDims.H4) > int(txDims.W4)*int(txDims.H4) {
			ctxOffset = 10
		}
		txbSkipCtx = ctxBase + ctxOffset
	}

	return TXBContext{
		TXBSkipContext: uint8(txbSkipCtx),
		DCSignContext:  uint8(coeffDCSignContext(dcSign)),
	}, nil
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

func (c *CoeffEntropyContext) markTXBKnown(req CoeffContextRequest, txDims TransformDimensions, visibleW int, visibleH int, result TXBDecodeResult) error {
	value := uint8(0)
	if !result.AllZero && result.EOB > 0 {
		if !validCoeffEntropyValue(result.CulLevel) {
			return ErrInvalidDecodeState
		}
		value = result.CulLevel
	}
	c.setTXBContextKnown(req, txDims, visibleW, visibleH, value)
	return nil
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
	for k := 0; k < int(dims.W4); k++ {
		c.Above[plane][x4+k] = 0
	}
	for k := 0; k < int(dims.H4); k++ {
		c.Left[plane][y4+k] = 0
	}
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
	allZero, err := s.ReadTXBSkip(cdfs, TXBSkipRequest{Size: req.Size, Context: int(req.TXBSkipContext)})
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
	c.setTXBContextKnown(req, txDims, visibleW, visibleH, value)
	return nil
}

func (c *CoeffEntropyContext) setTXBContextKnown(req CoeffContextRequest, txDims TransformDimensions, visibleW int, visibleH int, value uint8) {
	plane := int(req.Plane)
	x4 := int(req.X4)
	y4 := int(req.Y4)
	for k := 0; k < int(txDims.W4); k++ {
		next := uint8(0)
		if k < visibleW {
			next = value
		}
		c.Above[plane][x4+k] = next
	}
	for k := 0; k < int(txDims.H4); k++ {
		next := uint8(0)
		if k < visibleH {
			next = value
		}
		c.Left[plane][y4+k] = next
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
	if visibleW < 0 || visibleH < 0 || visibleW > int(txDims.W4) || visibleH > int(txDims.H4) {
		return 0, 0, ErrInvalidDecodeState
	}
	return visibleW, visibleH, nil
}

func coeffContextMagnitude(values []uint8) int {
	out := 0
	for _, value := range values {
		out |= int(value)
	}
	out &= CoeffContextMask
	if out > 4 {
		out = 4
	}
	return out
}

func coeffEntropyBase(above []uint8, left []uint8) int {
	base := 0
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

func coeffDCSignContext(dcSign int) int {
	switch {
	case dcSign < 0:
		return 1
	case dcSign > 0:
		return 2
	default:
		return 0
	}
}

func validCoeffEntropyValue(value uint8) bool {
	if value > CoeffContextMask+(2<<CoeffContextBits) {
		return false
	}
	return value>>CoeffContextBits <= 2
}
