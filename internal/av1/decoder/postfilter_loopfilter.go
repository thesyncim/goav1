package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
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

// FrameWorkLoopFilterPostFilterEdge describes one deblocking edge candidate in
// plane-local 4x4 coordinates. Final frame-order scheduling remains separate
// from this block-local candidate collection.
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
	ChromaTXBs           int
	EdgeCandidates       int
	PlaneEdgeCandidates  [3]int
	StoredEdges          int
	DroppedEdges         int

	Levels [3][2]FrameWorkLoopFilterPostFilterLevelStats
}

// FrameWorkLoopFilterPostFilterApplyResult summarizes application of stored
// luma edge candidates through the pure-Go loop-filter kernels.
type FrameWorkLoopFilterPostFilterApplyResult struct {
	Plan FrameWorkLoopFilterPostFilterPlan

	Active   bool
	Edges    int
	Applied  int
	MaxLevel uint8
}

// LoopFilterPostFilterScratchLen reports scratch lengths needed to collect edge
// candidates for the current decoded loop-filter map.
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
// not mutate ctx.Output; full frame-order edge scheduling remains unsupported.
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
			if err := frameWorkAppendLoopFilterChromaEdges(ctx, record, &plan, req.Edges); err != nil {
				return plan, err
			}
		}
	}
	if plan.Missing != 0 {
		return plan, threading.ErrInvalidBatch
	}
	return plan, nil
}

// ApplyLoopFilterLumaEdges validates the decoded loop-filter map, stores luma
// edge candidates in req.Edges, and applies every stored candidate to ctx.Output.
// It is a decoder bridge for the current luma-only edge scheduler; chroma and
// full frame-order integration remain separate work.
func (ctx FrameWorkPostFilterContext) ApplyLoopFilterLumaEdges(req FrameWorkLoopFilterPostFilterRequest) (FrameWorkLoopFilterPostFilterApplyResult, error) {
	plan, err := ctx.LoopFilterPostFilterPlan(req)
	result := FrameWorkLoopFilterPostFilterApplyResult{Plan: plan}
	if err != nil {
		return result, err
	}
	if !plan.Active {
		return FrameWorkLoopFilterPostFilterApplyResult{}, nil
	}
	result.Active = true
	if ctx.Output == nil {
		return result, frame.ErrInvalidSlot
	}
	if plan.DroppedEdges != 0 || plan.StoredEdges != plan.EdgeCandidates {
		return result, frame.ErrShortBuffer
	}
	if plan.PlaneEdgeCandidates[loopfilter.PlaneU] != 0 || plan.PlaneEdgeCandidates[loopfilter.PlaneV] != 0 {
		return result, ErrUnsupportedPostFilter
	}
	edges := req.Edges[:plan.StoredEdges]
	for i := range edges {
		if err := ctx.applyLoopFilterLumaEdge(edges[i]); err != nil {
			return result, err
		}
		result.Edges++
		if edges[i].Level != 0 {
			result.Applied++
			if edges[i].Level > result.MaxLevel {
				result.MaxLevel = edges[i].Level
			}
		}
	}
	return result, nil
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterLumaEdge(edge FrameWorkLoopFilterPostFilterEdge) error {
	if edge.Plane != loopfilter.PlaneY ||
		edge.Length4 <= 0 ||
		edge.Level == 0 ||
		ctx.Output == nil {
		return loopfilter.ErrInvalidFilter
	}
	thresholds, err := loopfilter.ThresholdsForLevel(edge.Level, ctx.Event.LoopFilter.Sharpness)
	if err != nil {
		return err
	}
	return loopfilter.FilterEdgeByWidth(
		edge.Width,
		ctx.Output.Y,
		ctx.Output.Layout.BytesPerSample,
		ctx.Output.Format.BitDepth,
		edge.Edge,
		edge.X4*4,
		edge.Y4*4,
		edge.Length4*4,
		thresholds,
	)
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
	if err := record.TransformTree.ForEachLumaTXB(req, func(tile.TransformBlock) error {
		plan.LumaTXBs++
		return nil
	}); err != nil {
		return err
	}
	if !record.TransformTree.HasUV {
		return nil
	}
	return frameWorkLoopFilterForEachChromaTXB(ctx, record, func(tile.TransformBlock) error {
		plan.ChromaTXBs++
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

func frameWorkAppendLoopFilterChromaEdges(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge) error {
	if ctx.Event.SequenceHeader.ColorConfig.MonoChrome {
		return nil
	}
	for plane := loopfilter.PlaneU; plane <= loopfilter.PlaneV; plane++ {
		verticalLevel, err := frameWorkResolveLoopFilterLevel(ctx, record, plane, loopfilter.EdgeVertical)
		if err != nil {
			return err
		}
		horizontalLevel, err := frameWorkResolveLoopFilterLevel(ctx, record, plane, loopfilter.EdgeHorizontal)
		if err != nil {
			return err
		}
		if verticalLevel == 0 && horizontalLevel == 0 {
			continue
		}
		if record.SkipTransform {
			tx, ok, err := frameWorkLoopFilterChromaBlock(ctx, record)
			if err != nil || !ok {
				return err
			}
			if err := frameWorkAppendLoopFilterChromaTXBEdges(ctx, record, plan, edges, plane, verticalLevel, horizontalLevel, tx); err != nil {
				return err
			}
			continue
		}
		if err := frameWorkLoopFilterForEachChromaTXB(ctx, record, func(tx tile.TransformBlock) error {
			return frameWorkAppendLoopFilterChromaTXBEdges(ctx, record, plan, edges, plane, verticalLevel, horizontalLevel, tx)
		}); err != nil {
			return err
		}
	}
	return nil
}

func frameWorkAppendLoopFilterChromaTXBEdges(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, plane loopfilter.Plane, verticalLevel uint8, horizontalLevel uint8, tx tile.TransformBlock) error {
	frameX4, frameY4 := frameWorkLoopFilterChromaFrame4(ctx, record, tx.X4, tx.Y4)
	if verticalLevel != 0 && frameX4 > 0 {
		width, ok, err := frameWorkLoopFilterScheduledChromaWidth(ctx, plane, loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size)
		if err != nil {
			return err
		}
		if ok {
			frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
				Plane:      plane,
				Edge:       loopfilter.EdgeVertical,
				X4:         frameX4,
				Y4:         frameY4,
				Length4:    int(tx.VisibleH4),
				Level:      verticalLevel,
				Transform:  tx.Size,
				Width:      width,
				BlockMICol: record.Block.MICol,
				BlockMIRow: record.Block.MIRow,
			})
		}
	}
	if horizontalLevel != 0 && frameY4 > 0 {
		width, ok, err := frameWorkLoopFilterScheduledChromaWidth(ctx, plane, loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size)
		if err != nil {
			return err
		}
		if ok {
			frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
				Plane:      plane,
				Edge:       loopfilter.EdgeHorizontal,
				X4:         frameX4,
				Y4:         frameY4,
				Length4:    int(tx.VisibleW4),
				Level:      horizontalLevel,
				Transform:  tx.Size,
				Width:      width,
				BlockMICol: record.Block.MICol,
				BlockMIRow: record.Block.MIRow,
			})
		}
	}
	return nil
}

func frameWorkLoopFilterScheduledLumaWidth(ctx FrameWorkPostFilterContext, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize) (int, bool, error) {
	return frameWorkLoopFilterScheduledWidth(ctx, loopfilter.PlaneY, edge, x4, y4, length4, tx)
}

func frameWorkLoopFilterScheduledChromaWidth(ctx FrameWorkPostFilterContext, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize) (int, bool, error) {
	return frameWorkLoopFilterScheduledWidth(ctx, plane, edge, x4, y4, length4, tx)
}

func frameWorkLoopFilterScheduledWidth(ctx FrameWorkPostFilterContext, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize) (int, bool, error) {
	width, err := frameWorkLoopFilterWidth(plane, edge, tx)
	if err != nil {
		return 0, false, err
	}
	x := x4 * 4
	y := y4 * 4
	length := length4 * 4
	frameWidth, frameHeight, err := frameWorkLoopFilterPlaneSize(ctx, plane)
	if err != nil {
		return 0, false, err
	}
	for width != 0 {
		if frameWorkLoopFilterEdgeFits(width, edge, x, y, length, frameWidth, frameHeight) {
			return width, true, nil
		}
		width = frameWorkLoopFilterDowngradeWidth(width)
	}
	return 0, false, nil
}

func frameWorkLoopFilterLumaWidth(edge loopfilter.Edge, tx tile.TransformSize) (int, error) {
	return frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx)
}

func frameWorkLoopFilterWidth(plane loopfilter.Plane, edge loopfilter.Edge, tx tile.TransformSize) (int, error) {
	dims, ok := tx.Dimensions()
	if !ok {
		return 0, threading.ErrInvalidBatch
	}
	span4 := dims.W4
	if edge == loopfilter.EdgeHorizontal {
		span4 = dims.H4
	}
	if plane == loopfilter.PlaneU || plane == loopfilter.PlaneV {
		if span4 <= 1 {
			return 4, nil
		}
		return 8, nil
	}
	if plane != loopfilter.PlaneY {
		return 0, threading.ErrInvalidBatch
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

func frameWorkLoopFilterPlaneSize(ctx FrameWorkPostFilterContext, plane loopfilter.Plane) (int, int, error) {
	width := int(ctx.Event.FrameSize.CodedWidth)
	height := int(ctx.Event.FrameSize.Height)
	if width <= 0 || height <= 0 {
		return 0, 0, threading.ErrInvalidBatch
	}
	switch plane {
	case loopfilter.PlaneY:
		return width, height, nil
	case loopfilter.PlaneU, loopfilter.PlaneV:
		if ctx.Event.SequenceHeader.ColorConfig.MonoChrome {
			return 0, 0, threading.ErrInvalidBatch
		}
		color := ctx.Event.SequenceHeader.ColorConfig
		return frameWorkLoopFilterSubsampledLength(width, color.SubsamplingX),
			frameWorkLoopFilterSubsampledLength(height, color.SubsamplingY), nil
	default:
		return 0, 0, threading.ErrInvalidBatch
	}
}

func frameWorkLoopFilterSubsampledLength(length int, subsampled bool) int {
	if subsampled {
		return (length + 1) >> 1
	}
	return length
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
	if edge.Plane <= loopfilter.PlaneV {
		plan.PlaneEdgeCandidates[int(edge.Plane)]++
	}
	if plan.StoredEdges < len(edges) {
		edges[plan.StoredEdges] = edge
		plan.StoredEdges++
		return
	}
	plan.DroppedEdges++
}

func frameWorkLoopFilterForEachChromaTXB(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, visit func(tile.TransformBlock) error) error {
	block, ok, err := frameWorkLoopFilterChromaBlock(ctx, record)
	if err != nil || !ok {
		return err
	}
	if visit == nil {
		return threading.ErrInvalidBatch
	}
	uvDims, ok := record.TransformTree.UV.Dimensions()
	if !ok {
		return threading.ErrInvalidBatch
	}
	for y := 0; y < int(block.VisibleH4); y += int(uvDims.H4) {
		for x := 0; x < int(block.VisibleW4); x += int(uvDims.W4) {
			if err := visit(tile.TransformBlock{
				X4:        block.X4 + x,
				Y4:        block.Y4 + y,
				Size:      record.TransformTree.UV,
				VisibleW4: uint8(minInt(int(uvDims.W4), int(block.VisibleW4)-x)),
				VisibleH4: uint8(minInt(int(uvDims.H4), int(block.VisibleH4)-y)),
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

func frameWorkLoopFilterChromaBlock(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord) (tile.TransformBlock, bool, error) {
	req, err := frameWorkLoopFilterTransformTreeRequest(ctx, record)
	if err != nil {
		return tile.TransformBlock{}, false, err
	}
	color := ctx.Event.SequenceHeader.ColorConfig
	if color.MonoChrome || !tile.HasChromaBlock(req, color) {
		return tile.TransformBlock{}, false, nil
	}
	if !record.TransformTree.HasUV || !record.TransformTree.UV.Valid() {
		return tile.TransformBlock{}, false, threading.ErrInvalidBatch
	}
	ssX := frameWorkLoopFilterSubsamplingShift(color.SubsamplingX)
	ssY := frameWorkLoopFilterSubsamplingShift(color.SubsamplingY)
	x4 := req.X4 >> ssX
	y4 := req.Y4 >> ssY
	visibleW4 := ((req.X4 + int(req.VisibleW4) + ssX) >> ssX) - x4
	visibleH4 := ((req.Y4 + int(req.VisibleH4) + ssY) >> ssY) - y4
	if visibleW4 <= 0 || visibleH4 <= 0 ||
		x4+visibleW4 > tile.MaxBlockModeSlots ||
		y4+visibleH4 > tile.MaxBlockModeSlots {
		return tile.TransformBlock{}, false, threading.ErrInvalidBatch
	}
	return tile.TransformBlock{
		X4:        x4,
		Y4:        y4,
		Size:      record.TransformTree.UV,
		VisibleW4: uint8(visibleW4),
		VisibleH4: uint8(visibleH4),
	}, true, nil
}

func frameWorkLoopFilterChromaFrame4(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, x4 int, y4 int) (int, int) {
	color := ctx.Event.SequenceHeader.ColorConfig
	ssX := frameWorkLoopFilterSubsamplingShift(color.SubsamplingX)
	ssY := frameWorkLoopFilterSubsamplingShift(color.SubsamplingY)
	return (int(record.Block.MICol) >> ssX) + x4 - (record.Block.X4 >> ssX),
		(int(record.Block.MIRow) >> ssY) + y4 - (record.Block.Y4 >> ssY)
}

func frameWorkLoopFilterSubsamplingShift(subsampled bool) int {
	if subsampled {
		return 1
	}
	return 0
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
