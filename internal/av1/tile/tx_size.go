package tile

import (
	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

const (
	TransformSizeContexts      = 3
	TransformSizeCategories    = 4
	TransformPartitionContexts = 3
	TransformPartitionCats     = 7
)

// TransformSize identifies an AV1 rectangular transform size in dav1d/libaom
// TX_* then RTX_* order.
type TransformSize uint8

const (
	TransformSize4x4 TransformSize = iota
	TransformSize8x8
	TransformSize16x16
	TransformSize32x32
	TransformSize64x64
	TransformSize4x8
	TransformSize8x4
	TransformSize8x16
	TransformSize16x8
	TransformSize16x32
	TransformSize32x16
	TransformSize32x64
	TransformSize64x32
	TransformSize4x16
	TransformSize16x4
	TransformSize8x32
	TransformSize32x8
	TransformSize16x64
	TransformSize64x16
	transformSizeCount
)

// TransformDimensions reports a transform's dimensions in 4x4 units plus the
// dav1d/libaom log2 width/height classes and recursive sub-transform.
type TransformDimensions struct {
	W4    uint8
	H4    uint8
	Log2W uint8
	Log2H uint8
	Min   uint8
	Max   uint8
	Sub   TransformSize
	Ctx   uint8
}

// TransformCDFs contains the caller-owned CDFs for selected transform size and
// inter transform partition syntax.
type TransformCDFs struct {
	TxSize      [TransformSizeCategories][TransformSizeContexts]entropy.CDF
	TxPartition [TransformPartitionCats][TransformPartitionContexts]entropy.CDF
}

// SelectedTransformRequest describes the block/frame conditions for selected
// intra transform-size syntax.
type SelectedTransformRequest struct {
	Size          BlockSize
	TransformMode parser.TransformMode
	Lossless      bool
	X4            int
	Y4            int
	HaveTop       bool
	HaveLeft      bool
}

// TransformPartitionRequest describes one inter transform partition split
// decision.
type TransformPartitionRequest struct {
	Size     BlockSize
	From     TransformSize
	Depth    int
	X4       int
	Y4       int
	HaveTop  bool
	HaveLeft bool
}

var transformDimensions = [transformSizeCount]TransformDimensions{
	TransformSize4x4:   {W4: 1, H4: 1, Log2W: 0, Log2H: 0, Min: 0, Max: 0, Ctx: 0},
	TransformSize8x8:   {W4: 2, H4: 2, Log2W: 1, Log2H: 1, Min: 1, Max: 1, Sub: TransformSize4x4, Ctx: 1},
	TransformSize16x16: {W4: 4, H4: 4, Log2W: 2, Log2H: 2, Min: 2, Max: 2, Sub: TransformSize8x8, Ctx: 2},
	TransformSize32x32: {W4: 8, H4: 8, Log2W: 3, Log2H: 3, Min: 3, Max: 3, Sub: TransformSize16x16, Ctx: 3},
	TransformSize64x64: {W4: 16, H4: 16, Log2W: 4, Log2H: 4, Min: 4, Max: 4, Sub: TransformSize32x32, Ctx: 4},
	TransformSize4x8:   {W4: 1, H4: 2, Log2W: 0, Log2H: 1, Min: 0, Max: 1, Sub: TransformSize4x4, Ctx: 1},
	TransformSize8x4:   {W4: 2, H4: 1, Log2W: 1, Log2H: 0, Min: 0, Max: 1, Sub: TransformSize4x4, Ctx: 1},
	TransformSize8x16:  {W4: 2, H4: 4, Log2W: 1, Log2H: 2, Min: 1, Max: 2, Sub: TransformSize8x8, Ctx: 2},
	TransformSize16x8:  {W4: 4, H4: 2, Log2W: 2, Log2H: 1, Min: 1, Max: 2, Sub: TransformSize8x8, Ctx: 2},
	TransformSize16x32: {W4: 4, H4: 8, Log2W: 2, Log2H: 3, Min: 2, Max: 3, Sub: TransformSize16x16, Ctx: 3},
	TransformSize32x16: {W4: 8, H4: 4, Log2W: 3, Log2H: 2, Min: 2, Max: 3, Sub: TransformSize16x16, Ctx: 3},
	TransformSize32x64: {W4: 8, H4: 16, Log2W: 3, Log2H: 4, Min: 3, Max: 4, Sub: TransformSize32x32, Ctx: 4},
	TransformSize64x32: {W4: 16, H4: 8, Log2W: 4, Log2H: 3, Min: 3, Max: 4, Sub: TransformSize32x32, Ctx: 4},
	TransformSize4x16:  {W4: 1, H4: 4, Log2W: 0, Log2H: 2, Min: 0, Max: 2, Sub: TransformSize4x8, Ctx: 1},
	TransformSize16x4:  {W4: 4, H4: 1, Log2W: 2, Log2H: 0, Min: 0, Max: 2, Sub: TransformSize8x4, Ctx: 1},
	TransformSize8x32:  {W4: 2, H4: 8, Log2W: 1, Log2H: 3, Min: 1, Max: 3, Sub: TransformSize8x16, Ctx: 2},
	TransformSize32x8:  {W4: 8, H4: 2, Log2W: 3, Log2H: 1, Min: 1, Max: 3, Sub: TransformSize16x8, Ctx: 2},
	TransformSize16x64: {W4: 4, H4: 16, Log2W: 2, Log2H: 4, Min: 2, Max: 4, Sub: TransformSize16x32, Ctx: 3},
	TransformSize64x16: {W4: 16, H4: 4, Log2W: 4, Log2H: 2, Min: 2, Max: 4, Sub: TransformSize32x16, Ctx: 3},
}

var maxTransformSizeForBlock = [blockSizeCount][4]TransformSize{
	BlockSize128x128: {TransformSize64x64, TransformSize32x32, TransformSize32x32, TransformSize32x32},
	BlockSize128x64:  {TransformSize64x64, TransformSize32x32, TransformSize32x32, TransformSize32x32},
	BlockSize64x128:  {TransformSize64x64, TransformSize32x32, TransformSize4x4, TransformSize32x32},
	BlockSize64x64:   {TransformSize64x64, TransformSize32x32, TransformSize32x32, TransformSize32x32},
	BlockSize64x32:   {TransformSize64x32, TransformSize32x16, TransformSize32x32, TransformSize32x32},
	BlockSize64x16:   {TransformSize64x16, TransformSize32x8, TransformSize32x16, TransformSize32x16},
	BlockSize32x64:   {TransformSize32x64, TransformSize16x32, TransformSize4x4, TransformSize32x32},
	BlockSize32x32:   {TransformSize32x32, TransformSize16x16, TransformSize16x32, TransformSize32x32},
	BlockSize32x16:   {TransformSize32x16, TransformSize16x8, TransformSize16x16, TransformSize32x16},
	BlockSize32x8:    {TransformSize32x8, TransformSize16x4, TransformSize16x8, TransformSize32x8},
	BlockSize16x64:   {TransformSize16x64, TransformSize8x32, TransformSize4x4, TransformSize16x32},
	BlockSize16x32:   {TransformSize16x32, TransformSize8x16, TransformSize4x4, TransformSize16x32},
	BlockSize16x16:   {TransformSize16x16, TransformSize8x8, TransformSize8x16, TransformSize16x16},
	BlockSize16x8:    {TransformSize16x8, TransformSize8x4, TransformSize8x8, TransformSize16x8},
	BlockSize16x4:    {TransformSize16x4, TransformSize8x4, TransformSize8x4, TransformSize16x4},
	BlockSize8x32:    {TransformSize8x32, TransformSize4x16, TransformSize4x4, TransformSize8x32},
	BlockSize8x16:    {TransformSize8x16, TransformSize4x8, TransformSize4x4, TransformSize8x16},
	BlockSize8x8:     {TransformSize8x8, TransformSize4x4, TransformSize4x8, TransformSize8x8},
	BlockSize8x4:     {TransformSize8x4, TransformSize4x4, TransformSize4x4, TransformSize8x4},
	BlockSize4x16:    {TransformSize4x16, TransformSize4x8, TransformSize4x4, TransformSize4x16},
	BlockSize4x8:     {TransformSize4x8, TransformSize4x4, TransformSize4x4, TransformSize4x8},
	BlockSize4x4:     {TransformSize4x4, TransformSize4x4, TransformSize4x4, TransformSize4x4},
}

// Valid reports whether size names an AV1 transform size.
func (size TransformSize) Valid() bool {
	return size < transformSizeCount
}

// Dimensions returns dav1d/libaom's transform dimension record for size.
func (size TransformSize) Dimensions() (TransformDimensions, bool) {
	if !size.Valid() {
		return TransformDimensions{}, false
	}
	return transformDimensions[size], true
}

// TransformSize converts size to the transform package's pixel dimensions.
func (size TransformSize) TransformSize() (transform.Size, error) {
	dims, ok := size.Dimensions()
	if !ok {
		return transform.Size{}, ErrInvalidDecodeState
	}
	return transform.SizeFromDimensions(int(dims.W4)*4, int(dims.H4)*4)
}

// MaxTransformSize returns dav1d's max transform-size table entry for block,
// plane, and color format. Plane 0 is luma; chroma planes use the subsampling
// layout to choose the 4:2:0 / 4:2:2 / 4:4:4 column.
func MaxTransformSize(block BlockSize, color parser.ColorConfig, plane int) (TransformSize, error) {
	if block >= blockSizeCount || plane < 0 || plane > 2 {
		return 0, ErrInvalidDecodeState
	}
	column := 0
	if plane > 0 {
		switch {
		case color.MonoChrome:
			return 0, ErrInvalidDecodeState
		case color.SubsamplingX && color.SubsamplingY:
			column = 1
		case color.SubsamplingX && !color.SubsamplingY:
			column = 2
		default:
			column = 3
		}
	}
	size := maxTransformSizeForBlock[block][column]
	if !size.Valid() {
		return 0, ErrInvalidDecodeState
	}
	return size, nil
}

// InitDefault seeds c with dav1d/libaom's default transform CDFs.
func (c *TransformCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next TransformCDFs
	txSizeDefaults := [...][TransformSizeContexts][]uint16{
		{
			{19968}, {19968}, {24320},
		},
		{
			{12272, 30172}, {12272, 30172}, {18677, 30848},
		},
		{
			{12986, 15180}, {12986, 15180}, {24302, 25602},
		},
		{
			{5782, 11475}, {5782, 11475}, {16803, 22759},
		},
	}
	for category := range txSizeDefaults {
		for ctx := range txSizeDefaults[category] {
			if err := next.TxSize[category][ctx].Init(txSizeDefaults[category][ctx]); err != nil {
				return err
			}
		}
	}
	txPartDefaults := [...][TransformPartitionContexts]uint16{
		{28581, 23846, 20847},
		{24315, 18196, 12133},
		{18791, 10887, 11005},
		{27179, 20004, 11281},
		{26549, 19308, 14224},
		{28015, 21546, 14400},
		{28165, 22401, 16088},
	}
	for category := range txPartDefaults {
		for ctx := range txPartDefaults[category] {
			if err := next.TxPartition[category][ctx].Init([]uint16{txPartDefaults[category][ctx]}); err != nil {
				return err
			}
		}
	}
	*c = next
	return nil
}

// TxSizeCDF returns the selected transform-size CDF for category/context.
func (c *TransformCDFs) TxSizeCDF(category int, ctx int) (*entropy.CDF, error) {
	if c == nil || category < 0 || category >= TransformSizeCategories ||
		ctx < 0 || ctx >= TransformSizeContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.TxSize[category][ctx]
	wantSymbols := 3
	if category == 0 {
		wantSymbols = 2
	}
	if cdf.Symbols() != wantSymbols {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// TxPartitionCDF returns the inter transform partition CDF for category/context.
func (c *TransformCDFs) TxPartitionCDF(category int, ctx int) (*entropy.CDF, error) {
	if c == nil || category < 0 || category >= TransformPartitionCats ||
		ctx < 0 || ctx >= TransformPartitionContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.TxPartition[category][ctx]
	if cdf.Symbols() != 2 {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// SelectedTransformContext returns dav1d's get_tx_ctx() value for intra
// selected transform-size syntax.
func (c *BlockModeContext) SelectedTransformContext(max TransformSize, x4 int, y4 int) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	dims, ok := max.Dimensions()
	if !ok {
		return 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, err
	}
	left := 0
	if c.LeftTxIntra[y4] >= dims.Log2H {
		left = 1
	}
	above := 0
	if c.AboveTxIntra[x4] >= dims.Log2W {
		above = 1
	}
	return left + above, nil
}

// SelectedTransformContextWithAvailability returns libaom's get_tx_size_context()
// result for the selected intra transform-size syntax.
func (c *BlockModeContext) SelectedTransformContextWithAvailability(max TransformSize, x4 int, y4 int, haveTop bool, haveLeft bool) (int, error) {
	if c == nil {
		return 0, ErrInvalidDecodeState
	}
	dims, ok := max.Dimensions()
	if !ok {
		return 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(x4, y4); err != nil {
		return 0, err
	}
	// Use the pre-decode neighbor snapshot when available. The shared
	// AboveIntra/AboveBlockSize/LeftIntra/LeftBlockSize slots at (x4, y4)
	// are overwritten by the current block's MarkIntra / MarkIntrabcMotion
	// before DecodeBlockCoefficients reads tx_size context, so without the
	// snapshot the inter-block branch below would incorrectly read the
	// current block's flags instead of the neighbor's.
	aboveIntra := c.AboveIntra[x4]
	aboveBlockSize := c.AboveBlockSize[x4]
	leftIntra := c.LeftIntra[y4]
	leftBlockSize := c.LeftBlockSize[y4]
	if c.TxNeighborValid {
		aboveIntra = c.TxAboveNeighborIntra
		aboveBlockSize = c.TxAboveNeighborBlockSize
		leftIntra = c.TxLeftNeighborIntra
		leftBlockSize = c.TxLeftNeighborBlockSize
	}
	above := 0
	if c.AboveTx[x4] >= dims.Log2W {
		above = 1
	}
	if haveTop && aboveIntra == 0 {
		above = 0
		if aboveDims, ok := aboveBlockSize.Dimensions(); ok && aboveDims.W4 >= dims.W4 {
			above = 1
		}
	}
	left := 0
	if c.LeftTx[y4] >= dims.Log2H {
		left = 1
	}
	if haveLeft && leftIntra == 0 {
		left = 0
		if leftDims, ok := leftBlockSize.Dimensions(); ok && leftDims.H4 >= dims.H4 {
			left = 1
		}
	}
	switch {
	case haveTop && haveLeft:
		return above + left, nil
	case haveTop:
		return above, nil
	case haveLeft:
		return left, nil
	default:
		return 0, nil
	}
}

// TransformPartitionContext returns dav1d/libaom's inter transform partition
// category/context pair for one split decision.
func (c *BlockModeContext) TransformPartitionContext(req TransformPartitionRequest) (int, int, error) {
	if c == nil || req.Depth < 0 || req.Depth > 2 {
		return 0, 0, ErrInvalidDecodeState
	}
	dims, ok := req.From.Dimensions()
	if !ok || req.From <= TransformSize4x4 {
		return 0, 0, ErrInvalidDecodeState
	}
	if _, ok := req.Size.Dimensions(); !ok {
		return 0, 0, ErrInvalidDecodeState
	}
	if err := validateBlockModeSlot(req.X4, req.Y4); err != nil {
		return 0, 0, err
	}
	category := 2*(int(TransformSize64x64)-int(dims.Max)) - req.Depth
	if category < 0 || category >= TransformPartitionCats {
		return 0, 0, ErrInvalidDecodeState
	}
	// libaom seeds the above/left txfm context arrays with
	// tx_size_wide[TX_SIZES_LARGEST] at frame/tile edges, so the
	// txfm_partition_context() bare dereference reports above=0 (and left=0)
	// when no real neighbor exists. goav1's BlockModeContext leaves those
	// slots zero across blocks, so the "<txw" check would erroneously fire at
	// edges; gate the comparisons on neighbor availability to match libaom.
	above := 0
	if req.HaveTop && c.AboveTx[req.X4] < dims.Log2W {
		above = 1
	}
	left := 0
	if req.HaveLeft && c.LeftTx[req.Y4] < dims.Log2H {
		left = 1
	}
	return category, above + left, nil
}

// MarkTransform updates the top/left transform contexts after a decoded
// transform size.
func (c *BlockModeContext) MarkTransform(block BlockSize, x4 int, y4 int, size TransformSize, intra bool) error {
	if c == nil {
		return ErrInvalidDecodeState
	}
	blockDims, ok := block.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	txDims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	return c.MarkTransformArea(x4, y4, int(blockDims.W4), int(blockDims.H4),
		txDims.Log2W, txDims.Log2H, intra)
}

// MarkTransformArea updates transform contexts for a decoded transform block
// area. spanW4/spanH4 describe the coded txb area in 4x4 units, while
// ctxLog2W/ctxLog2H describe the transform context value written to that area.
// AV1 var-tx split-to-4x4 uses this distinction: the coded 8x8 txb area is
// marked with 4x4 context values.
func (c *BlockModeContext) MarkTransformArea(x4 int, y4 int, spanW4 int, spanH4 int, ctxLog2W uint8, ctxLog2H uint8, intra bool) error {
	if c == nil || spanW4 <= 0 || spanH4 <= 0 ||
		x4 < 0 || y4 < 0 ||
		x4+spanW4 > MaxBlockModeSlots ||
		y4+spanH4 > MaxBlockModeSlots ||
		ctxLog2W > 5 || ctxLog2H > 5 {
		return ErrInvalidDecodeState
	}
	for i := range spanW4 {
		c.AboveTx[x4+i] = ctxLog2W
		if intra {
			c.AboveTxIntra[x4+i] = ctxLog2W
		}
	}
	for i := range spanH4 {
		c.LeftTx[y4+i] = ctxLog2H
		if intra {
			c.LeftTxIntra[y4+i] = ctxLog2H
		}
	}
	return nil
}

// ReadSelectedTransformSize decodes intra selected transform-size syntax.
func (s *DecodeState) ReadSelectedTransformSize(cdfs *TransformCDFs, ctx *BlockModeContext, req SelectedTransformRequest) (TransformSize, error) {
	if s == nil {
		return 0, ErrInvalidDecodeState
	}
	max, err := MaxTransformSize(req.Size, parser.ColorConfig{}, 0)
	if err != nil {
		return 0, err
	}
	if req.Lossless || req.TransformMode == parser.TransformMode4x4Only || max == TransformSize4x4 {
		return TransformSize4x4, nil
	}
	if req.TransformMode == parser.TransformModeLargest {
		return max, nil
	}
	if req.TransformMode != parser.TransformModeSwitchable {
		return 0, ErrInvalidDecodeState
	}

	dims, ok := max.Dimensions()
	if !ok || dims.Max == 0 {
		return 0, ErrInvalidDecodeState
	}
	context, err := ctx.SelectedTransformContextWithAvailability(max, req.X4, req.Y4, req.HaveTop, req.HaveLeft)
	if err != nil {
		return 0, err
	}
	cdf, err := cdfs.TxSizeCDF(int(dims.Max)-1, context)
	if err != nil {
		return 0, err
	}
	depth, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	limit := min(int(dims.Max), 2)
	if depth < 0 || depth > limit {
		return 0, ErrInvalidDecodeState
	}
	size := max
	for depth > 0 {
		dims, ok := size.Dimensions()
		if !ok || dims.Sub == size {
			return 0, ErrInvalidDecodeState
		}
		size = dims.Sub
		depth--
	}
	return size, nil
}

// ReadTransformPartitionSplit decodes one inter transform partition split bit.
func (s *DecodeState) ReadTransformPartitionSplit(cdfs *TransformCDFs, ctx *BlockModeContext, req TransformPartitionRequest) (bool, error) {
	if s == nil {
		return false, ErrInvalidDecodeState
	}
	if req.Depth >= 2 || req.From <= TransformSize4x4 {
		return false, nil
	}
	category, context, err := ctx.TransformPartitionContext(req)
	if err != nil {
		return false, err
	}
	cdf, err := cdfs.TxPartitionCDF(category, context)
	if err != nil {
		return false, err
	}
	symbol, err := s.Reader.ReadCDF(cdf)
	if err != nil {
		return false, err
	}
	return symbol != 0, nil
}
