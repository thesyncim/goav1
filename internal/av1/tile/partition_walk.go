package tile

// BlockWalkRequest describes the absolute MI region to traverse. MI
// coordinates are AV1 4x4 units; Root is normally RootBlockLevel(sequence).
type BlockWalkRequest struct {
	Root BlockLevel

	MIColStart uint16
	MIRowStart uint16
	MIColEnd   uint16
	MIRowEnd   uint16

	// NeighborMI*Start names the tile/job boundary used for HaveTop and
	// HaveLeft when UseNeighborBounds is set. Block-loop callers that walk one
	// root at a time set these to the full job origin so block mode syntax
	// still sees neighbors across root boundaries.
	UseNeighborBounds  bool
	NeighborMIColStart uint16
	NeighborMIRowStart uint16
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

	// Edge is dav1d's intra_edge_flags for the block (EDGE_* flags from the
	// intra edge tree), set by the partition walk.
	Edge uint8
}

// BlockWalkStats reports how much partition syntax was consumed.
type BlockWalkStats struct {
	PartitionReads int32
	Blocks         int32
}

// BlockVisitor is called in bitstream traversal order for every leaf block.
type BlockVisitor func(BlockVisit) error

type partitionReadFunc func(level BlockLevel, ctx int, miCol uint32, miRow uint32, haveRight bool, haveBottom bool) (Partition, error)

// WalkBlocks decodes AV1 partition syntax over req and calls visit for each
// leaf block. cdfs and ctx are caller-owned so adapted state is retained across
// the walk.
func (s *DecodeState) WalkBlocks(cdfs *PartitionCDFs, ctx *PartitionContext, req BlockWalkRequest, visit BlockVisitor) (BlockWalkStats, error) {
	if s == nil {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	return walkBlocks(ctx, req, func(level BlockLevel, context int, _ uint32, _ uint32, haveRight bool, haveBottom bool) (Partition, error) {
		return s.ReadPartition(cdfs, level, context, haveRight, haveBottom)
	}, visit)
}

func walkBlocks(ctx *PartitionContext, req BlockWalkRequest, read partitionReadFunc, visit BlockVisitor) (BlockWalkStats, error) {
	if ctx == nil || read == nil || visit == nil || !req.Root.Valid() ||
		req.MIColEnd <= req.MIColStart || req.MIRowEnd <= req.MIRowStart {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	rootSize := uint32(req.Root.Size4x4())
	miColStart := uint32(req.MIColStart)
	miRowStart := uint32(req.MIRowStart)
	miColEnd := uint32(req.MIColEnd)
	miRowEnd := uint32(req.MIRowEnd)
	if rootSize == 0 || miColStart%rootSize != 0 || miRowStart%rootSize != 0 {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}

	w := partitionWalker{
		ctx:   ctx,
		req:   req,
		read:  read,
		visit: visit,
	}
	return w.run(miColStart, miRowStart, miColEnd, miRowEnd, rootSize)
}

// walkBlocksDecode is the decoder's partition walk, dav1d's decode_sb: the
// partition symbols are read straight from s, and every leaf is written in
// place at leaf before visit runs. visit takes no argument so that passing the
// leaf through an unknown function value does not move the walk to the heap.
func walkBlocksDecode(s *DecodeState, cdfs *PartitionCDFs, ctx *PartitionContext, req BlockWalkRequest, leaf *BlockVisit, visit func() error) (BlockWalkStats, error) {
	if s == nil || cdfs == nil || ctx == nil || leaf == nil || visit == nil || !req.Root.Valid() ||
		req.MIColEnd <= req.MIColStart || req.MIRowEnd <= req.MIRowStart {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	rootSize := uint32(req.Root.Size4x4())
	miColStart := uint32(req.MIColStart)
	miRowStart := uint32(req.MIRowStart)
	if rootSize == 0 || miColStart%rootSize != 0 || miRowStart%rootSize != 0 {
		return BlockWalkStats{}, ErrInvalidDecodeState
	}
	w := partitionWalker{
		ctx:          ctx,
		req:          req,
		dec:          s,
		partCDFs:     cdfs,
		leaf:         leaf,
		visitLeafPtr: visit,
	}
	return w.run(miColStart, miRowStart, uint32(req.MIColEnd), uint32(req.MIRowEnd), rootSize)
}

func (w *partitionWalker) run(miColStart, miRowStart, miColEnd, miRowEnd, rootSize uint32) (BlockWalkStats, error) {
	req := &w.req
	for miRow := miRowStart; miRow < miRowEnd; miRow += rootSize {
		for miCol := miColStart; miCol < miColEnd; miCol += rootSize {
			if err := w.walkNode(req.Root, intraEdgeRoot(req.Root), miCol, miRow, miCol, miRow); err != nil {
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

	// The decoder's walk (dav1d decode_sb) reads partition symbols directly
	// from dec/partCDFs and fills each leaf in place at leaf, handing it to
	// visitLeafPtr (which reads it there) instead of building and copying a
	// BlockVisit.
	dec          *DecodeState
	partCDFs     *PartitionCDFs
	leaf         *BlockVisit
	visitLeafPtr func() error

	stats BlockWalkStats
}

func (w *partitionWalker) walkNode(level BlockLevel, node uint16, rootCol uint32, rootRow uint32, miCol uint32, miRow uint32) error {
	miColEnd := uint32(w.req.MIColEnd)
	miRowEnd := uint32(w.req.MIRowEnd)
	if miCol >= miColEnd || miRow >= miRowEnd {
		return nil
	}
	half := uint32(level.HalfSize4x4())
	haveRight := miCol+half < miColEnd
	haveBottom := miRow+half < miRowEnd
	xb8, yb8, err := partitionContextSlot(rootCol, rootRow, miCol, miRow)
	if err != nil {
		return err
	}
	context, err := w.ctx.Context(level, yb8, xb8)
	if err != nil {
		return err
	}
	partition, err := w.readPartition(level, context, miCol, miRow, haveRight, haveBottom)
	w.stats.PartitionReads++
	if err != nil {
		return err
	}
	if !partition.ValidForLevel(level) {
		return ErrInvalidDecodeState
	}

	if partition == PartitionSplit && level != BlockLevel8x8 {
		next := level + 1
		split := &intraEdgeTree.nodes[node].split
		if err := w.walkNode(next, split[0], rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if haveRight {
			if err := w.walkNode(next, split[1], rootCol, rootRow, miCol+half, miRow); err != nil {
				return err
			}
		}
		if haveBottom {
			if err := w.walkNode(next, split[2], rootCol, rootRow, miCol, miRow+half); err != nil {
				return err
			}
			if haveRight {
				if err := w.walkNode(next, split[3], rootCol, rootRow, miCol+half, miRow+half); err != nil {
					return err
				}
			}
		}
		return nil
	}

	if err := w.walkPartitionLeaves(level, partition, &intraEdgeTree.nodes[node], rootCol, rootRow, miCol, miRow, haveRight, haveBottom); err != nil {
		return err
	}
	return w.ctx.Mark(level, partition, xb8, yb8)
}

// readPartition reads one partition symbol: directly from the decoder state on
// the decoder's walk, through the caller's reader otherwise.
func (w *partitionWalker) readPartition(level BlockLevel, context int, miCol uint32, miRow uint32, haveRight bool, haveBottom bool) (Partition, error) {
	if w.partCDFs != nil {
		return w.dec.ReadPartition(w.partCDFs, level, context, haveRight, haveBottom)
	}
	return w.read(level, context, miCol, miRow, haveRight, haveBottom)
}

func (w *partitionWalker) walkPartitionLeaves(level BlockLevel, partition Partition, node *intraEdgeNode, rootCol uint32, rootRow uint32, miCol uint32, miRow uint32, haveRight bool, haveBottom bool) error {
	half := uint32(level.HalfSize4x4())
	switch partition {
	case PartitionNone:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		return w.visitLeaf(level, partition, size, node.o, rootCol, rootRow, miCol, miRow)
	case PartitionH:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, size, node.h[0], rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if haveBottom {
			return w.visitLeaf(level, partition, size, node.h[1], rootCol, rootRow, miCol, miRow+half)
		}
		return nil
	case PartitionV:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, size, node.v[0], rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if haveRight {
			return w.visitLeaf(level, partition, size, node.v[1], rootCol, rootRow, miCol+half, miRow)
		}
		return nil
	case PartitionSplit:
		size, _, ok := partition.BlockSizes(level)
		if !ok || level != BlockLevel8x8 {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, size, edgeAllTRAndBL, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if haveRight {
			if err := w.visitLeaf(level, partition, size, node.tip[0], rootCol, rootRow, miCol+half, miRow); err != nil {
				return err
			}
		}
		if haveBottom {
			if err := w.visitLeaf(level, partition, size, node.tip[1], rootCol, rootRow, miCol, miRow+half); err != nil {
				return err
			}
			if haveRight {
				if err := w.visitLeaf(level, partition, size, node.tip[2], rootCol, rootRow, miCol+half, miRow+half); err != nil {
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
		if err := w.visitLeaf(level, partition, small, edgeAllTRAndBL, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if err := w.visitLeaf(level, partition, small, node.v[1], rootCol, rootRow, miCol+half, miRow); err != nil {
			return err
		}
		return w.visitLeaf(level, partition, large, node.h[1], rootCol, rootRow, miCol, miRow+half)
	case PartitionTBottomSplit:
		large, small, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, large, node.h[0], rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if err := w.visitLeaf(level, partition, small, node.v[0], rootCol, rootRow, miCol, miRow+half); err != nil {
			return err
		}
		return w.visitLeaf(level, partition, small, 0, rootCol, rootRow, miCol+half, miRow+half)
	case PartitionTLeftSplit:
		small, large, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, small, edgeAllTRAndBL, rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if err := w.visitLeaf(level, partition, small, node.h[1], rootCol, rootRow, miCol, miRow+half); err != nil {
			return err
		}
		return w.visitLeaf(level, partition, large, node.v[1], rootCol, rootRow, miCol+half, miRow)
	case PartitionTRightSplit:
		large, small, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		if err := w.visitLeaf(level, partition, large, node.v[0], rootCol, rootRow, miCol, miRow); err != nil {
			return err
		}
		if err := w.visitLeaf(level, partition, small, node.h[0], rootCol, rootRow, miCol+half, miRow); err != nil {
			return err
		}
		return w.visitLeaf(level, partition, small, 0, rootCol, rootRow, miCol+half, miRow+half)
	case PartitionH4:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		h4Edges := [4]uint8{node.h[0], node.h4, edgeAllLeftHasBottom, node.h[1]}
		step := half >> 1
		for i := range uint32(4) {
			y := miRow + i*step
			if i > 0 && y >= uint32(w.req.MIRowEnd) {
				break
			}
			if err := w.visitLeaf(level, partition, size, h4Edges[i], rootCol, rootRow, miCol, y); err != nil {
				return err
			}
		}
		return nil
	case PartitionV4:
		size, _, ok := partition.BlockSizes(level)
		if !ok {
			return ErrInvalidDecodeState
		}
		v4Edges := [4]uint8{node.v[0], node.v4, edgeAllTopHasRight, node.v[1]}
		step := half >> 1
		for i := range uint32(4) {
			x := miCol + i*step
			if i > 0 && x >= uint32(w.req.MIColEnd) {
				break
			}
			if err := w.visitLeaf(level, partition, size, v4Edges[i], rootCol, rootRow, x, miRow); err != nil {
				return err
			}
		}
		return nil
	default:
		return ErrInvalidDecodeState
	}
}

func (w *partitionWalker) visitLeaf(level BlockLevel, partition Partition, size BlockSize, edge uint8, rootCol uint32, rootRow uint32, miCol uint32, miRow uint32) error {
	miColEndLimit := uint32(w.req.MIColEnd)
	miRowEndLimit := uint32(w.req.MIRowEnd)
	if miCol >= miColEndLimit || miRow >= miRowEndLimit {
		return nil
	}
	dims, ok := size.Dimensions()
	if !ok {
		return ErrInvalidDecodeState
	}
	miEndCol := miCol + uint32(dims.W4)
	miEndRow := miRow + uint32(dims.H4)
	if miEndCol > miColEndLimit {
		miEndCol = miColEndLimit
	}
	if miEndRow > miRowEndLimit {
		miEndRow = miRowEndLimit
	}
	if miEndCol <= miCol || miEndRow <= miRow {
		return nil
	}
	x4 := uint8(miCol - rootCol)
	y4 := uint8(miRow - rootRow)
	w.stats.Blocks++
	if b := w.leaf; b != nil {
		b.MICol = uint16(miCol)
		b.MIRow = uint16(miRow)
		b.MIColEnd = uint16(miEndCol)
		b.MIRowEnd = uint16(miEndRow)
		b.X4 = x4
		b.Y4 = y4
		b.Level = level
		b.Partition = partition
		b.Size = size
		b.VisibleW4 = uint8(miEndCol - miCol)
		b.VisibleH4 = uint8(miEndRow - miRow)
		b.HaveTop = miRow > w.req.neighborMIRowStart()
		b.HaveLeft = miCol > w.req.neighborMIColStart()
		b.Edge = edge
		return w.visitLeafPtr()
	}
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
		Edge:      edge,
	})
}

func (req BlockWalkRequest) neighborMIColStart() uint32 {
	if req.UseNeighborBounds {
		return uint32(req.NeighborMIColStart)
	}
	return uint32(req.MIColStart)
}

func (req BlockWalkRequest) neighborMIRowStart() uint32 {
	if req.UseNeighborBounds {
		return uint32(req.NeighborMIRowStart)
	}
	return uint32(req.MIRowStart)
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
