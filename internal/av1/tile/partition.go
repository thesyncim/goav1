package tile

import "github.com/thesyncim/goav1/internal/av1/entropy"

const (
	PartitionContexts = 4
	MaxPartitionSlots = 16
)

// BlockLevel identifies the recursive AV1 partition level. Values follow
// dav1d's BL_128X128 through BL_8X8 order.
type BlockLevel uint8

const (
	BlockLevel128x128 BlockLevel = iota
	BlockLevel64x64
	BlockLevel32x32
	BlockLevel16x16
	BlockLevel8x8
	blockLevelCount
)

// Partition identifies one AV1 block partition symbol. Values follow dav1d's
// PARTITION_NONE through PARTITION_V4 order.
type Partition uint8

const (
	PartitionNone Partition = iota
	PartitionH
	PartitionV
	PartitionSplit
	PartitionTTopSplit
	PartitionTBottomSplit
	PartitionTLeftSplit
	PartitionTRightSplit
	PartitionH4
	PartitionV4
	partitionCount
)

// BlockSize identifies one AV1 block size in dav1d's BS_* order.
type BlockSize uint8

const (
	BlockSize128x128 BlockSize = iota
	BlockSize128x64
	BlockSize64x128
	BlockSize64x64
	BlockSize64x32
	BlockSize64x16
	BlockSize32x64
	BlockSize32x32
	BlockSize32x16
	BlockSize32x8
	BlockSize16x64
	BlockSize16x32
	BlockSize16x16
	BlockSize16x8
	BlockSize16x4
	BlockSize8x32
	BlockSize8x16
	BlockSize8x8
	BlockSize8x4
	BlockSize4x16
	BlockSize4x8
	BlockSize4x4
	blockSizeCount
)

// BlockDimensions reports a block's dimensions in 4x4 units plus libaom's
// log2 width and height classes.
type BlockDimensions struct {
	W4    uint8
	H4    uint8
	Log2W uint8
	Log2H uint8
}

// PartitionCDFs is the caller-owned mode CDF subset required to read AV1
// partition syntax.
type PartitionCDFs struct {
	Partition [blockLevelCount][PartitionContexts]entropy.CDF
}

// PartitionContext is the caller-owned top/left partition context for one
// superblock. Slots are addressed in 8x8 units, matching dav1d's bx8/by8.
type PartitionContext struct {
	Above [MaxPartitionSlots]uint8
	Left  [MaxPartitionSlots]uint8
}

var partitionSymbolCount = [blockLevelCount]int{
	BlockLevel128x128: int(PartitionTRightSplit) + 1,
	BlockLevel64x64:   int(PartitionV4) + 1,
	BlockLevel32x32:   int(PartitionV4) + 1,
	BlockLevel16x16:   int(PartitionV4) + 1,
	BlockLevel8x8:     int(PartitionSplit) + 1,
}

var blockDimensions = [blockSizeCount]BlockDimensions{
	BlockSize128x128: {W4: 32, H4: 32, Log2W: 5, Log2H: 5},
	BlockSize128x64:  {W4: 32, H4: 16, Log2W: 5, Log2H: 4},
	BlockSize64x128:  {W4: 16, H4: 32, Log2W: 4, Log2H: 5},
	BlockSize64x64:   {W4: 16, H4: 16, Log2W: 4, Log2H: 4},
	BlockSize64x32:   {W4: 16, H4: 8, Log2W: 4, Log2H: 3},
	BlockSize64x16:   {W4: 16, H4: 4, Log2W: 4, Log2H: 2},
	BlockSize32x64:   {W4: 8, H4: 16, Log2W: 3, Log2H: 4},
	BlockSize32x32:   {W4: 8, H4: 8, Log2W: 3, Log2H: 3},
	BlockSize32x16:   {W4: 8, H4: 4, Log2W: 3, Log2H: 2},
	BlockSize32x8:    {W4: 8, H4: 2, Log2W: 3, Log2H: 1},
	BlockSize16x64:   {W4: 4, H4: 16, Log2W: 2, Log2H: 4},
	BlockSize16x32:   {W4: 4, H4: 8, Log2W: 2, Log2H: 3},
	BlockSize16x16:   {W4: 4, H4: 4, Log2W: 2, Log2H: 2},
	BlockSize16x8:    {W4: 4, H4: 2, Log2W: 2, Log2H: 1},
	BlockSize16x4:    {W4: 4, H4: 1, Log2W: 2, Log2H: 0},
	BlockSize8x32:    {W4: 2, H4: 8, Log2W: 1, Log2H: 3},
	BlockSize8x16:    {W4: 2, H4: 4, Log2W: 1, Log2H: 2},
	BlockSize8x8:     {W4: 2, H4: 2, Log2W: 1, Log2H: 1},
	BlockSize8x4:     {W4: 2, H4: 1, Log2W: 1, Log2H: 0},
	BlockSize4x16:    {W4: 1, H4: 4, Log2W: 0, Log2H: 2},
	BlockSize4x8:     {W4: 1, H4: 2, Log2W: 0, Log2H: 1},
	BlockSize4x4:     {W4: 1, H4: 1, Log2W: 0, Log2H: 0},
}

var partitionBlockSizes = [blockLevelCount][partitionCount][2]BlockSize{
	BlockLevel128x128: {
		PartitionNone:         {BlockSize128x128},
		PartitionH:            {BlockSize128x64},
		PartitionV:            {BlockSize64x128},
		PartitionTTopSplit:    {BlockSize64x64, BlockSize128x64},
		PartitionTBottomSplit: {BlockSize128x64, BlockSize64x64},
		PartitionTLeftSplit:   {BlockSize64x64, BlockSize64x128},
		PartitionTRightSplit:  {BlockSize64x128, BlockSize64x64},
	},
	BlockLevel64x64: {
		PartitionNone:         {BlockSize64x64},
		PartitionH:            {BlockSize64x32},
		PartitionV:            {BlockSize32x64},
		PartitionTTopSplit:    {BlockSize32x32, BlockSize64x32},
		PartitionTBottomSplit: {BlockSize64x32, BlockSize32x32},
		PartitionTLeftSplit:   {BlockSize32x32, BlockSize32x64},
		PartitionTRightSplit:  {BlockSize32x64, BlockSize32x32},
		PartitionH4:           {BlockSize64x16},
		PartitionV4:           {BlockSize16x64},
	},
	BlockLevel32x32: {
		PartitionNone:         {BlockSize32x32},
		PartitionH:            {BlockSize32x16},
		PartitionV:            {BlockSize16x32},
		PartitionTTopSplit:    {BlockSize16x16, BlockSize32x16},
		PartitionTBottomSplit: {BlockSize32x16, BlockSize16x16},
		PartitionTLeftSplit:   {BlockSize16x16, BlockSize16x32},
		PartitionTRightSplit:  {BlockSize16x32, BlockSize16x16},
		PartitionH4:           {BlockSize32x8},
		PartitionV4:           {BlockSize8x32},
	},
	BlockLevel16x16: {
		PartitionNone:         {BlockSize16x16},
		PartitionH:            {BlockSize16x8},
		PartitionV:            {BlockSize8x16},
		PartitionTTopSplit:    {BlockSize8x8, BlockSize16x8},
		PartitionTBottomSplit: {BlockSize16x8, BlockSize8x8},
		PartitionTLeftSplit:   {BlockSize8x8, BlockSize8x16},
		PartitionTRightSplit:  {BlockSize8x16, BlockSize8x8},
		PartitionH4:           {BlockSize16x4},
		PartitionV4:           {BlockSize4x16},
	},
	BlockLevel8x8: {
		PartitionNone:  {BlockSize8x8},
		PartitionH:     {BlockSize8x4},
		PartitionV:     {BlockSize4x8},
		PartitionSplit: {BlockSize4x4},
	},
}

var partitionContextValues = [2][blockLevelCount][partitionCount]int8{
	{
		{0x00, 0x00, 0x10, -1, 0x00, 0x10, 0x10, 0x10, -1, -1},
		{0x10, 0x10, 0x18, -1, 0x10, 0x18, 0x18, 0x18, 0x10, 0x1c},
		{0x18, 0x18, 0x1c, -1, 0x18, 0x1c, 0x1c, 0x1c, 0x18, 0x1e},
		{0x1c, 0x1c, 0x1e, -1, 0x1c, 0x1e, 0x1e, 0x1e, 0x1c, 0x1f},
		{0x1e, 0x1e, 0x1f, 0x1f, -1, -1, -1, -1, -1, -1},
	},
	{
		{0x00, 0x10, 0x00, -1, 0x10, 0x10, 0x00, 0x10, -1, -1},
		{0x10, 0x18, 0x10, -1, 0x18, 0x18, 0x10, 0x18, 0x1c, 0x10},
		{0x18, 0x1c, 0x18, -1, 0x1c, 0x1c, 0x18, 0x1c, 0x1e, 0x18},
		{0x1c, 0x1e, 0x1c, -1, 0x1e, 0x1e, 0x1c, 0x1e, 0x1f, 0x1c},
		{0x1e, 0x1f, 0x1e, 0x1f, -1, -1, -1, -1, -1, -1},
	},
}

// Valid reports whether level names an AV1 partition level.
func (level BlockLevel) Valid() bool {
	return level < blockLevelCount
}

// RootBlockLevel returns the superblock partition root for the sequence.
func RootBlockLevel(use128x128Superblock bool) BlockLevel {
	if use128x128Superblock {
		return BlockLevel128x128
	}
	return BlockLevel64x64
}

// HalfSize4x4 returns dav1d's hsz value for level, in 4x4 units.
func (level BlockLevel) HalfSize4x4() uint8 {
	if !level.Valid() {
		return 0
	}
	return 16 >> level
}

// Size4x4 returns the square block width/height represented by level, in 4x4
// units.
func (level BlockLevel) Size4x4() uint8 {
	if !level.Valid() {
		return 0
	}
	return level.HalfSize4x4() << 1
}

// Dimensions returns the 4x4-unit dimensions for size.
func (size BlockSize) Dimensions() (BlockDimensions, bool) {
	if size >= blockSizeCount {
		return BlockDimensions{}, false
	}
	return blockDimensions[size], true
}

// ValidForLevel reports whether partition can be coded at level.
func (partition Partition) ValidForLevel(level BlockLevel) bool {
	if !level.Valid() {
		return false
	}
	return int(partition) < partitionSymbolCount[level]
}

// BlockSizes returns the block-size entries used by non-recursive partitions.
// Split partitions are decoded by recursing to the next block level.
func (partition Partition) BlockSizes(level BlockLevel) (BlockSize, BlockSize, bool) {
	if !partition.ValidForLevel(level) || partition == PartitionSplit && level != BlockLevel8x8 {
		return 0, 0, false
	}
	sizes := partitionBlockSizes[level][partition]
	return sizes[0], sizes[1], true
}

// InitDefault seeds c with dav1d/libaom's default partition CDFs.
func (c *PartitionCDFs) InitDefault() error {
	if c == nil {
		return entropy.ErrInvalidCDF
	}
	var next PartitionCDFs
	defaults := [...][PartitionContexts][]uint16{
		{
			{27899, 28219, 28529, 32484, 32539, 32619, 32639},
			{6607, 6990, 8268, 32060, 32219, 32338, 32371},
			{5429, 6676, 7122, 32027, 32227, 32531, 32582},
			{711, 966, 1172, 32448, 32538, 32617, 32664},
		},
		{
			{20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724},
			{6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630, 30104},
			{5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298, 32238},
			{870, 1212, 1487, 31198, 31394, 31574, 31743, 31881, 32332},
		},
		{
			{18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178, 31544},
			{7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041, 29052},
			{6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526, 32047},
			{1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185, 31899},
		},
		{
			{15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902},
			{7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701, 29834},
			{5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336, 32117},
			{2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829, 31171},
		},
		{
			{19132, 25510, 30392},
			{13928, 19855, 28540},
			{12522, 23679, 28629},
			{9896, 18783, 25853},
		},
	}
	for level := range defaults {
		for ctx := range defaults[level] {
			if err := next.Partition[level][ctx].Init(defaults[level][ctx]); err != nil {
				return err
			}
		}
	}
	*c = next
	return nil
}

// CDF returns the initialized CDF for level/context.
func (c *PartitionCDFs) CDF(level BlockLevel, ctx int) (*entropy.CDF, error) {
	if c == nil || !level.Valid() || ctx < 0 || ctx >= PartitionContexts {
		return nil, entropy.ErrInvalidCDF
	}
	cdf := &c.Partition[level][ctx]
	if cdf.Symbols() != partitionSymbolCount[level] {
		return nil, entropy.ErrInvalidCDF
	}
	return cdf, cdf.Validate()
}

// ReadPartition decodes one AV1 partition decision for level. Boundary cases
// match dav1d's decode_sb(): when only one split direction is available, a
// binary non-adaptive partition probability is gathered from the full CDF.
func (s *DecodeState) ReadPartition(cdfs *PartitionCDFs, level BlockLevel, ctx int, haveHSplit bool, haveVSplit bool) (Partition, error) {
	if s == nil || !level.Valid() {
		return 0, ErrInvalidDecodeState
	}
	if !haveHSplit && !haveVSplit {
		if level >= BlockLevel8x8 {
			return 0, ErrInvalidDecodeState
		}
		return PartitionSplit, nil
	}
	if level >= BlockLevel8x8 && (!haveHSplit || !haveVSplit) {
		return 0, ErrInvalidDecodeState
	}

	cdf, err := cdfs.CDF(level, ctx)
	if err != nil {
		return 0, err
	}
	if haveHSplit && haveVSplit {
		symbol, err := s.ReadSymbol(cdf)
		if err != nil {
			return 0, err
		}
		partition := Partition(symbol)
		if !partition.ValidForLevel(level) {
			return 0, ErrInvalidDecodeState
		}
		return partition, nil
	}

	values := cdf.Values()
	prob := 0
	nonSplit := PartitionH
	if haveHSplit {
		prob = gatherTopPartitionProb(values, level)
	} else {
		prob = gatherLeftPartitionProb(values, level)
		nonSplit = PartitionV
	}
	bit, err := s.Reader.ReadBoolQ15(uint16(prob))
	if err != nil {
		return 0, err
	}
	if bit != 0 {
		return PartitionSplit, nil
	}
	return nonSplit, nil
}

// Context returns dav1d's get_partition_ctx() value.
func (c *PartitionContext) Context(level BlockLevel, yb8 int, xb8 int) (int, error) {
	if c == nil || !level.Valid() || xb8 < 0 || yb8 < 0 ||
		xb8 >= MaxPartitionSlots || yb8 >= MaxPartitionSlots {
		return 0, ErrInvalidDecodeState
	}
	shift := 4 - uint(level)
	return int((c.Above[xb8]>>shift)&1) + int((c.Left[yb8]>>shift)&1)<<1, nil
}

// Mark updates the top/left partition context after a decoded partition that
// writes context in dav1d's decode_sb().
func (c *PartitionContext) Mark(level BlockLevel, partition Partition, xb8 int, yb8 int) error {
	if c == nil || !partition.ValidForLevel(level) || xb8 < 0 || yb8 < 0 {
		return ErrInvalidDecodeState
	}
	if partition == PartitionSplit && level != BlockLevel8x8 {
		return nil
	}
	top := partitionContextValues[0][level][partition]
	left := partitionContextValues[1][level][partition]
	if top < 0 || left < 0 {
		return ErrInvalidDecodeState
	}
	span := int(level.HalfSize4x4())
	if xb8+span > MaxPartitionSlots || yb8+span > MaxPartitionSlots {
		return ErrInvalidDecodeState
	}
	for i := range span {
		c.Above[xb8+i] = uint8(top)
		c.Left[yb8+i] = uint8(left)
	}
	return nil
}

func gatherLeftPartitionProb(cdf []uint16, level BlockLevel) int {
	out := int(cdf[PartitionH-1]) - int(cdf[PartitionH])
	out += int(cdf[PartitionSplit-1]) - int(cdf[PartitionTLeftSplit])
	if level != BlockLevel128x128 {
		out += int(cdf[PartitionH4-1]) - int(cdf[PartitionH4])
	}
	return out
}

func gatherTopPartitionProb(cdf []uint16, level BlockLevel) int {
	out := int(cdf[PartitionV-1]) - int(cdf[PartitionTTopSplit])
	out += int(cdf[PartitionTLeftSplit-1])
	if level != BlockLevel128x128 {
		out += int(cdf[PartitionV4-1]) - int(cdf[PartitionTRightSplit])
	}
	return out
}
