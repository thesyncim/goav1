package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// FrameWorkLoopFilterMap is the decoded frame-level loop-filter metadata map.
type FrameWorkLoopFilterMap = threading.FrameWorkLoopFilterMap

// FrameWorkLoopFilterPostFilterRequest carries decoded loop-filter side data
// and optional caller-owned edge storage for postfilter planning.
type FrameWorkLoopFilterPostFilterRequest struct {
	Map   FrameWorkLoopFilterMap
	Edges []FrameWorkLoopFilterPostFilterEdge
}

// FrameWorkLoopFilterPostFilterEdge describes one luma deblocking edge
// candidate in frame 4x4 coordinates. Final filter-width selection still needs
// the full frame edge scheduler.
type FrameWorkLoopFilterPostFilterEdge struct {
	Plane loopfilter.Plane
	Edge  loopfilter.Edge

	X4      int
	Y4      int
	Length4 int

	Level     uint8
	Transform tile.TransformSize
	Width     int

	BlockMICol uint32
	BlockMIRow uint32
}

// FrameWorkLoopFilterPostFilterScratchSize reports caller-owned scratch needed
// for loop-filter planning and eventual application.
type FrameWorkLoopFilterPostFilterScratchSize struct {
	Edges int
}

// FrameWorkLoopFilterPostFilterLevelStats summarizes resolved levels for one
// plane/edge class.
type FrameWorkLoopFilterPostFilterLevelStats struct {
	Blocks   int
	NonZero  int
	MaxLevel uint8
}

// FrameWorkLoopFilterPostFilterPlan summarizes loop-filter metadata needed by
// the eventual frame-edge scheduler.
type FrameWorkLoopFilterPostFilterPlan struct {
	Active bool

	MICols int
	MIRows int

	Cells   int
	Blocks  int
	Missing int

	TransformReadyBlocks int
	SkipTransformBlocks  int
	LumaTXBs             int
	EdgeCandidates       int
	StoredEdges          int
	DroppedEdges         int

	Levels [3][2]FrameWorkLoopFilterPostFilterLevelStats
}

// LoopFilterPostFilterScratchLen reports scratch lengths needed to collect
// luma edge candidates for the current decoded loop-filter map.
func (ctx FrameWorkPostFilterContext) LoopFilterPostFilterScratchLen(req FrameWorkLoopFilterPostFilterRequest) (FrameWorkLoopFilterPostFilterScratchSize, error) {
	req.Edges = nil
	plan, err := ctx.LoopFilterPostFilterPlan(req)
	if err != nil {
		return FrameWorkLoopFilterPostFilterScratchSize{}, err
	}
	return FrameWorkLoopFilterPostFilterScratchSize{
		Edges: plan.EdgeCandidates,
	}, nil
}

// LoopFilterPostFilterPlan validates decoded loop-filter side data and resolves
// the block-local filter levels that will feed edge-mask construction. It does
// not mutate ctx.Output; full frame edge scheduling remains unsupported.
func (ctx FrameWorkPostFilterContext) LoopFilterPostFilterPlan(req FrameWorkLoopFilterPostFilterRequest) (FrameWorkLoopFilterPostFilterPlan, error) {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkLoopFilterPostFilterPlan{}, nil
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkLoopFilterPostFilterPlan{}, err
	}
	plan := FrameWorkLoopFilterPostFilterPlan{
		Active: true,
		MICols: cols,
		MIRows: rows,
	}

	filterMap := req.Map
	if frameWorkLoopFilterMapEmpty(filterMap) && ctx.LoopFilterMap != nil {
		filterMap = *ctx.LoopFilterMap
	}
	if frameWorkLoopFilterMapEmpty(filterMap) {
		return plan, threading.ErrInvalidBatch
	}
	if err := frameWorkValidateLoopFilterMap(filterMap, cols, rows); err != nil {
		return plan, err
	}

	for row := 0; row < rows; row++ {
		base := row * filterMap.Stride
		for col := 0; col < cols; col++ {
			record := filterMap.Records[base+col]
			if !record.Valid {
				plan.Missing++
				continue
			}
			if err := frameWorkValidateLoopFilterRecord(record, col, row, cols, rows); err != nil {
				return plan, err
			}
			plan.Cells++
			if record.Block.MIRow != uint32(row) || record.Block.MICol != uint32(col) {
				continue
			}
			plan.Blocks++
			if err := frameWorkResolveLoopFilterLevels(ctx, record, &plan); err != nil {
				return plan, err
			}
			if err := frameWorkAccumulateLoopFilterTransformStats(ctx, record, &plan); err != nil {
				return plan, err
			}
			if err := frameWorkAppendLoopFilterLumaEdges(ctx, record, &plan, req.Edges); err != nil {
				return plan, err
			}
		}
	}
	if plan.Missing != 0 {
		return plan, threading.ErrInvalidBatch
	}
	return plan, nil
}

func frameWorkLoopFilterMapEmpty(filterMap FrameWorkLoopFilterMap) bool {
	return filterMap.Stride == 0 && filterMap.Rows == 0 && len(filterMap.Records) == 0
}

func frameWorkLoopFilterMapGrid(size parser.FrameSize) (int, int, error) {
	if size.CodedWidth == 0 || size.Height == 0 {
		return 0, 0, threading.ErrInvalidBatch
	}
	miCols, ok := frameWorkLoopFilterMIExtent(size.CodedWidth)
	if !ok {
		return 0, 0, threading.ErrInvalidBatch
	}
	miRows, ok := frameWorkLoopFilterMIExtent(size.Height)
	if !ok {
		return 0, 0, threading.ErrInvalidBatch
	}
	limit := uint64(^uint(0) >> 1)
	if miCols == 0 || miRows == 0 || uint64(miCols)*uint64(miRows) > limit {
		return 0, 0, threading.ErrInvalidBatch
	}
	return int(miCols), int(miRows), nil
}

func frameWorkLoopFilterMIExtent(pixels uint32) (uint32, bool) {
	if pixels > ^uint32(0)-7 {
		return 0, false
	}
	return ((pixels + 7) >> 3) << 1, true
}

func frameWorkValidateLoopFilterMap(filterMap FrameWorkLoopFilterMap, cols int, rows int) error {
	if filterMap.Stride < cols || filterMap.Rows < rows || filterMap.Stride <= 0 || filterMap.Rows <= 0 {
		return threading.ErrInvalidBatch
	}
	limit := int(^uint(0) >> 1)
	if filterMap.Stride > limit/filterMap.Rows {
		return threading.ErrInvalidBatch
	}
	length := filterMap.Stride * filterMap.Rows
	if len(filterMap.Records) < length {
		return threading.ErrInvalidBatch
	}
	return nil
}

func frameWorkAccumulateLoopFilterTransformStats(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan) error {
	req, err := frameWorkLoopFilterTransformTreeRequest(ctx, record)
	if err != nil {
		return err
	}
	plan.TransformReadyBlocks++
	if record.SkipTransform {
		plan.SkipTransformBlocks++
		return nil
	}
	return record.TransformTree.ForEachLumaTXB(req, func(tile.TransformBlock) error {
		plan.LumaTXBs++
		return nil
	})
}

func frameWorkAppendLoopFilterLumaEdges(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge) error {
	verticalLevel, err := frameWorkResolveLoopFilterLevel(ctx, record, loopfilter.PlaneY, loopfilter.EdgeVertical)
	if err != nil {
		return err
	}
	horizontalLevel, err := frameWorkResolveLoopFilterLevel(ctx, record, loopfilter.PlaneY, loopfilter.EdgeHorizontal)
	if err != nil {
		return err
	}
	if verticalLevel == 0 && horizontalLevel == 0 {
		return nil
	}
	if record.SkipTransform {
		block := record.Block
		if verticalLevel != 0 && block.MICol > 0 {
			width, ok, err := frameWorkLoopFilterScheduledLumaWidth(ctx, loopfilter.EdgeVertical, int(block.MICol), int(block.MIRow), int(block.VisibleH4), record.TransformTree.Y)
			if err != nil {
				return err
			}
			if ok {
				frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
					Plane:      loopfilter.PlaneY,
					Edge:       loopfilter.EdgeVertical,
					X4:         int(block.MICol),
					Y4:         int(block.MIRow),
					Length4:    int(block.VisibleH4),
					Level:      verticalLevel,
					Transform:  record.TransformTree.Y,
					Width:      width,
					BlockMICol: block.MICol,
					BlockMIRow: block.MIRow,
				})
			}
		}
		if horizontalLevel != 0 && block.MIRow > 0 {
			width, ok, err := frameWorkLoopFilterScheduledLumaWidth(ctx, loopfilter.EdgeHorizontal, int(block.MICol), int(block.MIRow), int(block.VisibleW4), record.TransformTree.Y)
			if err != nil {
				return err
			}
			if ok {
				frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
					Plane:      loopfilter.PlaneY,
					Edge:       loopfilter.EdgeHorizontal,
					X4:         int(block.MICol),
					Y4:         int(block.MIRow),
					Length4:    int(block.VisibleW4),
					Level:      horizontalLevel,
					Transform:  record.TransformTree.Y,
					Width:      width,
					BlockMICol: block.MICol,
					BlockMIRow: block.MIRow,
				})
			}
		}
		return nil
	}

	req, err := frameWorkLoopFilterTransformTreeRequest(ctx, record)
	if err != nil {
		return err
	}
	block := record.Block
	return record.TransformTree.ForEachLumaTXB(req, func(tx tile.TransformBlock) error {
		frameX4 := int(block.MICol) + tx.X4 - block.X4
		frameY4 := int(block.MIRow) + tx.Y4 - block.Y4
		if verticalLevel != 0 && frameX4 > 0 {
			width, ok, err := frameWorkLoopFilterScheduledLumaWidth(ctx, loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size)
			if err != nil {
				return err
			}
			if ok {
				frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
					Plane:      loopfilter.PlaneY,
					Edge:       loopfilter.EdgeVertical,
					X4:         frameX4,
					Y4:         frameY4,
					Length4:    int(tx.VisibleH4),
					Level:      verticalLevel,
					Transform:  tx.Size,
					Width:      width,
					BlockMICol: block.MICol,
					BlockMIRow: block.MIRow,
				})
			}
		}
		if horizontalLevel != 0 && frameY4 > 0 {
			width, ok, err := frameWorkLoopFilterScheduledLumaWidth(ctx, loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size)
			if err != nil {
				return err
			}
			if ok {
				frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
					Plane:      loopfilter.PlaneY,
					Edge:       loopfilter.EdgeHorizontal,
					X4:         frameX4,
					Y4:         frameY4,
					Length4:    int(tx.VisibleW4),
					Level:      horizontalLevel,
					Transform:  tx.Size,
					Width:      width,
					BlockMICol: block.MICol,
					BlockMIRow: block.MIRow,
				})
			}
		}
		return nil
	})
}

func frameWorkLoopFilterScheduledLumaWidth(ctx FrameWorkPostFilterContext, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize) (int, bool, error) {
	width, err := frameWorkLoopFilterLumaWidth(edge, tx)
	if err != nil {
		return 0, false, err
	}
	x := x4 * 4
	y := y4 * 4
	length := length4 * 4
	frameWidth := int(ctx.Event.FrameSize.CodedWidth)
	frameHeight := int(ctx.Event.FrameSize.Height)
	for width != 0 {
		if frameWorkLoopFilterEdgeFits(width, edge, x, y, length, frameWidth, frameHeight) {
			return width, true, nil
		}
		width = frameWorkLoopFilterDowngradeWidth(width)
	}
	return 0, false, nil
}

func frameWorkLoopFilterLumaWidth(edge loopfilter.Edge, tx tile.TransformSize) (int, error) {
	dims, ok := tx.Dimensions()
	if !ok {
		return 0, threading.ErrInvalidBatch
	}
	span4 := dims.W4
	if edge == loopfilter.EdgeHorizontal {
		span4 = dims.H4
	}
	switch {
	case span4 <= 1:
		return 4, nil
	case span4 == 2:
		return 8, nil
	default:
		return 14, nil
	}
}

func frameWorkLoopFilterDowngradeWidth(width int) int {
	switch width {
	case 14:
		return 8
	case 8:
		return 4
	default:
		return 0
	}
}

func frameWorkLoopFilterEdgeFits(width int, edge loopfilter.Edge, x int, y int, length int, frameWidth int, frameHeight int) bool {
	if length <= 0 || frameWidth <= 0 || frameHeight <= 0 {
		return false
	}
	radius := width / 2
	switch edge {
	case loopfilter.EdgeHorizontal:
		return y >= radius && y+radius-1 < frameHeight && x >= 0 && x <= frameWidth-length
	case loopfilter.EdgeVertical:
		return x >= radius && x+radius-1 < frameWidth && y >= 0 && y <= frameHeight-length
	default:
		return false
	}
}

func frameWorkStoreLoopFilterEdge(plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, edge FrameWorkLoopFilterPostFilterEdge) {
	plan.EdgeCandidates++
	if plan.StoredEdges < len(edges) {
		edges[plan.StoredEdges] = edge
		plan.StoredEdges++
		return
	}
	plan.DroppedEdges++
}

func frameWorkLoopFilterTransformTreeRequest(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord) (tile.TransformTreeRequest, error) {
	block := record.Block
	dims, ok := block.Size.Dimensions()
	if !ok || block.VisibleW4 == 0 || block.VisibleH4 == 0 ||
		block.VisibleW4 > dims.W4 || block.VisibleH4 > dims.H4 ||
		block.X4 < 0 || block.Y4 < 0 {
		return tile.TransformTreeRequest{}, threading.ErrInvalidBatch
	}
	return tile.TransformTreeRequest{
		Size:          block.Size,
		X4:            block.X4,
		Y4:            block.Y4,
		VisibleW4:     block.VisibleW4,
		VisibleH4:     block.VisibleH4,
		Color:         ctx.Event.SequenceHeader.ColorConfig,
		Inter:         !record.Intra,
		SkipTransform: record.SkipTransform,
	}, nil
}

func frameWorkValidateLoopFilterRecord(record threading.FrameWorkLoopFilterBlockRecord, col int, row int, cols int, rows int) error {
	block := record.Block
	if record.SegmentID >= parser.MaxSegments ||
		block.MIColEnd <= block.MICol ||
		block.MIRowEnd <= block.MIRow ||
		block.MIColEnd > uint32(cols) ||
		block.MIRowEnd > uint32(rows) ||
		uint32(col) < block.MICol ||
		uint32(col) >= block.MIColEnd ||
		uint32(row) < block.MIRow ||
		uint32(row) >= block.MIRowEnd {
		return threading.ErrInvalidBatch
	}
	return nil
}

func frameWorkLoopFilterDeltaState(record threading.FrameWorkLoopFilterBlockRecord) loopfilter.DeltaState {
	state := loopfilter.DeltaState{FromBase: record.DeltaLFFromBase}
	for i := range state.Multi {
		state.Multi[i] = record.DeltaLF[i]
	}
	return state
}

func frameWorkResolveLoopFilterLevel(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, plane loopfilter.Plane, edge loopfilter.Edge) (uint8, error) {
	return loopfilter.ResolveBlockLevel(
		ctx.Event.LoopFilter,
		ctx.Event.Segmentation,
		ctx.Event.Delta,
		frameWorkLoopFilterDeltaState(record),
		loopfilter.LevelRequest{
			Plane:     plane,
			Edge:      edge,
			SegmentID: int(record.SegmentID),
			RefFrame:  record.RefFrame,
			Mode:      record.Mode,
		},
	)
}

func frameWorkResolveLoopFilterLevels(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan) error {
	planeCount := 3
	if ctx.Event.SequenceHeader.ColorConfig.MonoChrome {
		planeCount = 1
	}
	for plane := 0; plane < planeCount; plane++ {
		for edge := 0; edge < 2; edge++ {
			level, err := frameWorkResolveLoopFilterLevel(ctx, record, loopfilter.Plane(plane), loopfilter.Edge(edge))
			if err != nil {
				return err
			}
			stats := &plan.Levels[plane][edge]
			stats.Blocks++
			if level == 0 {
				continue
			}
			stats.NonZero++
			if level > stats.MaxLevel {
				stats.MaxLevel = level
			}
		}
	}
	return nil
}
