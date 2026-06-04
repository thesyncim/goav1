package tile

// BlockWalkRequest describes the absolute MI region to traverse. MI
// coordinates are AV1 4x4 units; Root is normally RootBlockLevel(sequence).
type BlockWalkRequest struct {
	Root BlockLevel

	MIColStart uint32
	MIRowStart uint32
	MIColEnd   uint32
	MIRowEnd   uint32

	// NeighborMI*Start names the tile/job boundary used for HaveTop and
	// HaveLeft when UseNeighborBounds is set. Block-loop callers that walk one
	// root at a time set these to the full job origin so block mode syntax
	// still sees neighbors across root boundaries.
	UseNeighborBounds  bool
	NeighborMIColStart uint32
	NeighborMIRowStart uint32
}

// BlockVisit describes one decoded leaf block from AV1's partition tree.
type BlockVisit struct {
	MICol    uint16
	MIRow    uint16
	MIColEnd uint16
	MIRowEnd uint16
	X4       uint8
	Y4       uint8

	Level     BlockLevel
	Partition Partition
	Size      BlockSize
	VisibleW4 uint8
	VisibleH4 uint8

	HaveTop  bool
	HaveLeft bool
}

// BlockWalkStats reports how much partition syntax was consumed.
type BlockWalkStats struct {
	PartitionReads int
	Blocks         int
}

// BlockVisitor is called in bitstream traversal order for every leaf block.
type BlockVisitor func(BlockVisit) error

type partitionReadFunc func(level BlockLevel, ctx int, haveRight bool, haveBottom bool) (Partition, error)

// WalkBlocks decodes AV1 partition syntax over req and calls visit for each
// leaf block. cdfs and ctx are caller-owned so adapted state is retained across
// the walk.
func (s *DecodeState) WalkBlocks(cdfs *PartitionCDFs, ctx *PartitionContext, req BlockWalkRequest, visit BlockVisitor) (BlockWalkStats, error) {
	if s == nil {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	return walkBlocks(ctx, req, func(level BlockLevel, context int, haveRight bool, haveBottom bool) (Partition, error) {
		return s.ReadPartition(cdfs, level, context, haveRight, haveBottom)
	}, visit)
}

func walkBlocks(ctx *PartitionContext, req BlockWalkRequest, read partitionReadFunc, visit BlockVisitor) (BlockWalkStats, error) {
	if ctx == nil || read == nil || visit == nil || !req.Root.Valid() ||
		req.MIColEnd <= req.MIColStart || req.MIRowEnd <= req.MIRowStart {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	if req.MIColEnd > uint32(^uint16(0)) || req.MIRowEnd > uint32(^uint16(0)) {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	rootSize := uint32(req.Root.Size4x4())
	if rootSize == 0 || req.MIColStart%rootSize != 0 || req.MIRowStart%rootSize != 0 {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}

	w := partitionWalker{
		ctx:   ctx,
		req:   req,
		read:  read,
		visit: visit,
	}
	for miRow := req.MIRowStart; miRow < req.MIRowEnd; miRow += rootSize {
		for miCol := req.MIColStart; miCol < req.MIColEnd; miCol += rootSize {
			if err := w.walkNode(req.Root, miCol, miRow, miCol, miRow); err != nil {
				return w.stats, err
			}
		}
	}
	return w.stats, nil
}

type partitionWalker struct {
	ctx   *PartitionContext
	req   BlockWalkRequest
	read  partitionReadFunc
	visit BlockVisitor

	stats BlockWalkStats
}

func (w *partitionWalker) walkNode(level BlockLevel, rootCol uint32, rootRow uint32, miCol uint32, miRow uint32) error {
	if miCol >= w.req.MIColEnd || miRow >= w.req.MIRowEnd {
		return nil
	}
	half := uint32(level.HalfSize4x4())
	haveRight := miCol+half < w.req.MIColEnd
	haveBottom := miRow+half < w.req.MIRowEnd
	xb8, yb8, err := partitionContextSlot(rootCol, rootRow, miCol, miRow)
	if err != nil {
		return err
	}
	context, err := w.ctx.Context(level, yb8, xb8)
	if err != nil {
		return err
	}
	partition, err := w.read(level, context, haveRight, haveBottom)
	w.stats.PartitionReads++
	if err != nil {
		return err
	}
	if !partition.ValidForLevel(level) {
		return ErrInvalidDecodeState
	}

	if partition == PartitionSplit && level != BlockLevel8x8 {
		next := level + 1
		if err := w.walkNode(next, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if haveRight {
			if err := w.walkNode(next, rootCol, rootRow, miCol+half, miRow); err != nil {
				return err
			}
		}
		if haveBottom {
			if err := w.walkNode(next, rootCol, rootRow, miCol, miRow+half); err != nil {
				return err
			}
			if haveRight {
				if err := w.walkNode(next, rootCol, rootRow, miCol+half, miRow+half); err != nil {
					return err
				}
			}
		}
		return nil
	}

	if err := w.walkPartitionLeaves(level, partition, rootCol, rootRow, miCol, miRow, haveRight, haveBottom); err != nil {
		return err
	}
	return w.ctx.Mark(level, partition, xb8, yb8)
}

func (w *partitionWalker) walkPartitionLeaves(level BlockLevel, partition Partition, rootCol uint32, rootRow uint32, miCol uint32, miRow uint32, haveRight bool, haveBottom bool) error {
	half := uint32(level.HalfSize4x4())
	switch partition {
	case PartitionNone:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		return w.visitLeaf(level, partition, size, rootCol, rootRow, miCol, miRow)
	case PartitionH:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, size, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if haveBottom {
			return w.visitLeaf(level, partition, size, rootCol, rootRow, miCol, miRow+half)
		}
		return nil
	case PartitionV:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, size, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if haveRight {
			return w.visitLeaf(level, partition, size, rootCol, rootRow, miCol+half, miRow)
		}
		return nil
	case PartitionSplit:
		size, _, ok := partition.BlockSizes(level)
		if !ok || level != BlockLevel8x8 {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, size, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if haveRight {
			if err := w.visitLeaf(level, partition, size, rootCol, rootRow, miCol+half, miRow); err != nil {
				return err
			}
		}
		if haveBottom {
			if err := w.visitLeaf(level, partition, size, rootCol, rootRow, miCol, miRow+half); err != nil {
				return err
			}
			if haveRight {
				if err := w.visitLeaf(level, partition, size, rootCol, rootRow, miCol+half, miRow+half); err != nil {
					return err
				}
			}
		}
		return nil
	case PartitionTTopSplit:
		small, large, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, small, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if err := w.visitLeaf(level, partition, small, rootCol, rootRow, miCol+half, miRow); err != nil {
			return err
		}
		return w.visitLeaf(level, partition, large, rootCol, rootRow, miCol, miRow+half)
	case PartitionTBottomSplit:
		large, small, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, large, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if err := w.visitLeaf(level, partition, small, rootCol, rootRow, miCol, miRow+half); err != nil {
			return err
		}
		return w.visitLeaf(level, partition, small, rootCol, rootRow, miCol+half, miRow+half)
	case PartitionTLeftSplit:
		small, large, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, small, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if err := w.visitLeaf(level, partition, small, rootCol, rootRow, miCol, miRow+half); err != nil {
			return err
		}
		return w.visitLeaf(level, partition, large, rootCol, rootRow, miCol+half, miRow)
	case PartitionTRightSplit:
		large, small, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, large, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if err := w.visitLeaf(level, partition, small, rootCol, rootRow, miCol+half, miRow); err != nil {
			return err
		}
		return w.visitLeaf(level, partition, small, rootCol, rootRow, miCol+half, miRow+half)
	case PartitionH4:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		step := half >> 1
		for i := range uint32(4) {
			y := miRow + i*step
			if i > 0 && y >= w.req.MIRowEnd {
				break
			}
			if err := w.visitLeaf(level, partition, size, rootCol, rootRow, miCol, y); err != nil {
				return err
			}
		}
		return nil
	case PartitionV4:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		step := half >> 1
		for i := range uint32(4) {
			x := miCol + i*step
			if i > 0 && x >= w.req.MIColEnd {
				break
			}
			if err := w.visitLeaf(level, partition, size, rootCol, rootRow, x, miRow); err != nil {
				return err
			}
		}
		return nil
	default:
		return ErrInvalidDecodeState
	}
}

func (w *partitionWalker) visitLeaf(level BlockLevel, partition Partition, size BlockSize, rootCol uint32, rootRow uint32, miCol uint32, miRow uint32) error {
	if miCol >= w.req.MIColEnd || miRow >= w.req.MIRowEnd {
		return nil
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	miEndCol := miCol + uint32(dims.W4)
	miEndRow := miRow + uint32(dims.H4)
	if miEndCol > w.req.MIColEnd {
		miEndCol = w.req.MIColEnd
	}
	if miEndRow > w.req.MIRowEnd {
		miEndRow = w.req.MIRowEnd
	}
	if miEndCol <= miCol || miEndRow <= miRow {
		return nil
	}
	x4 := uint8(miCol - rootCol)
	y4 := uint8(miRow - rootRow)
	w.stats.Blocks++
	return w.visit(BlockVisit{
		MICol:     uint16(miCol),
		MIRow:     uint16(miRow),
		MIColEnd:  uint16(miEndCol),
		MIRowEnd:  uint16(miEndRow),
		X4:        x4,
		Y4:        y4,
		Level:     level,
		Partition: partition,
		Size:      size,
		VisibleW4: uint8(miEndCol - miCol),
		VisibleH4: uint8(miEndRow - miRow),
		HaveTop:   miRow > w.req.neighborMIRowStart(),
		HaveLeft:  miCol > w.req.neighborMIColStart(),
	})
}

func (req BlockWalkRequest) neighborMIColStart() uint32 {
	if req.UseNeighborBounds {
		return req.NeighborMIColStart
	}
	return req.MIColStart
}

func (req BlockWalkRequest) neighborMIRowStart() uint32 {
	if req.UseNeighborBounds {
		return req.NeighborMIRowStart
	}
	return req.MIRowStart
}

func partitionContextSlot(rootCol uint32, rootRow uint32, miCol uint32, miRow uint32) (int, int, error) {
	if miCol < rootCol || miRow < rootRow {
		return 0, 0, ErrInvalidDecodeState
	}
	x := miCol - rootCol
	y := miRow - rootRow
	if x&1 != 0 || y&1 != 0 {
		return 0, 0, ErrInvalidDecodeState
	}
	return int(x >> 1), int(y >> 1), nil
}
