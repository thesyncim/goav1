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
	// LevelFromPrevious reports that AV1's fallback to the block on the other
	// side of the boundary supplied Level because the current side resolved to 0.
	LevelFromPrevious bool

	BlockMICol uint32
	BlockMIRow uint32
}

// FrameWorkLoopFilterPostFilterScratchSize reports caller-owned scratch needed
// for loop-filter planning and eventual application.
type FrameWorkLoopFilterPostFilterScratchSize struct {
	Edges int
}

// BindEdges validates and slices caller-owned edge storage for loop-filter
// planning/application.
func (s FrameWorkLoopFilterPostFilterScratchSize) BindEdges(edges []FrameWorkLoopFilterPostFilterEdge) ([]FrameWorkLoopFilterPostFilterEdge, error) {
	if s.Edges < 0 || len(edges) < s.Edges {
		return nil, frame.ErrShortBuffer
	}
	return edges[:s.Edges], nil
}

// Max returns the per-field maximum loop-filter scratch size.
func (s FrameWorkLoopFilterPostFilterScratchSize) Max(other FrameWorkLoopFilterPostFilterScratchSize) FrameWorkLoopFilterPostFilterScratchSize {
	return FrameWorkLoopFilterPostFilterScratchSize{Edges: maxInt(s.Edges, other.Edges)}
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
	PreviousLevelEdges   int
	StoredEdges          int
	DroppedEdges         int

	Levels [3][2]FrameWorkLoopFilterPostFilterLevelStats
}

// FrameWorkLoopFilterPostFilterApplyResult summarizes application of stored
// luma edge candidates through the pure-Go loop-filter kernels.
type FrameWorkLoopFilterPostFilterApplyResult struct {
	Plan FrameWorkLoopFilterPostFilterPlan

	Active bool

	Edges   int
	Applied int

	PlaneEdges    [3]int
	PlaneApplied  [3]int
	PlaneMaxLevel [3]uint8

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
	stats, err := filterMap.CoverageStats(cols, rows)
	if err != nil {
		return plan, err
	}
	plan.Cells = stats.Cells
	plan.Blocks = stats.Blocks
	plan.Missing = stats.Missing
	if plan.Missing != 0 {
		return plan, threading.ErrInvalidBatch
	}
	if err := filterMap.ForEachBlockInGrid(cols, rows, func(record threading.FrameWorkLoopFilterBlockRecord) error {
		if err := frameWorkValidateLoopFilterRecord(record, int(record.Block.MICol), int(record.Block.MIRow), cols, rows); err != nil {
			return err
		}
		if err := frameWorkResolveLoopFilterLevels(ctx, record, &plan); err != nil {
			return err
		}
		if err := frameWorkAccumulateLoopFilterTransformStats(ctx, record, &plan); err != nil {
			return err
		}
		if err := frameWorkAppendLoopFilterLumaEdges(ctx, filterMap, record, &plan, req.Edges); err != nil {
			return err
		}
		return frameWorkAppendLoopFilterChromaEdges(ctx, filterMap, record, &plan, req.Edges)
	}); err != nil {
		return plan, err
	}
	return plan, nil
}

// ApplyLoopFilterEdges validates the decoded loop-filter map, stores edge
// candidates in req.Edges, and applies every stored candidate to ctx.Output. It
// is a decoder bridge for the current stored-candidate scheduler; full
// frame-order integration remains separate work.
func (ctx FrameWorkPostFilterContext) ApplyLoopFilterEdges(req FrameWorkLoopFilterPostFilterRequest) (FrameWorkLoopFilterPostFilterApplyResult, error) {
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
	edges := req.Edges[:plan.StoredEdges]
	if err := ctx.applyLoopFilterEdgesInPlanePassOrder(&result, edges, loopfilter.PlaneV); err != nil {
		return result, err
	}
	return result, nil
}

// ApplyLoopFilterLumaEdges validates the decoded loop-filter map, stores luma
// edge candidates in req.Edges, and applies every stored luma candidate to
// ctx.Output. It remains a luma-only bridge for callers that have not opted into
// chroma edge application.
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
	if err := ctx.applyLoopFilterEdgesInPlanePassOrder(&result, edges, loopfilter.PlaneY); err != nil {
		return result, err
	}
	return result, nil
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterEdgesInPlanePassOrder(result *FrameWorkLoopFilterPostFilterApplyResult, edges []FrameWorkLoopFilterPostFilterEdge, maxPlane loopfilter.Plane) error {
	before := result.Edges
	for plane := loopfilter.PlaneY; plane <= maxPlane; plane++ {
		for edgeKind := loopfilter.EdgeVertical; edgeKind <= loopfilter.EdgeHorizontal; edgeKind++ {
			for i := range edges {
				edge := edges[i]
				if edge.Plane != plane || edge.Edge != edgeKind {
					continue
				}
				if err := ctx.applyLoopFilterEdge(edge); err != nil {
					return err
				}
				frameWorkCountAppliedLoopFilterEdge(result, edge)
			}
		}
	}
	if result.Edges-before != len(edges) {
		return loopfilter.ErrInvalidFilter
	}
	return nil
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterLumaEdge(edge FrameWorkLoopFilterPostFilterEdge) error {
	if edge.Plane != loopfilter.PlaneY {
		return loopfilter.ErrInvalidFilter
	}
	return ctx.applyLoopFilterEdge(edge)
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterEdge(edge FrameWorkLoopFilterPostFilterEdge) error {
	if edge.Length4 <= 0 || edge.Level == 0 || ctx.Output == nil {
		return loopfilter.ErrInvalidFilter
	}
	dst, ok := frameWorkLoopFilterOutputPlane(*ctx.Output, edge.Plane)
	if !ok {
		return loopfilter.ErrInvalidFilter
	}
	thresholds, err := loopfilter.ThresholdsForLevel(edge.Level, ctx.Event.LoopFilter.Sharpness)
	if err != nil {
		return err
	}
	return loopfilter.FilterEdgeByWidth(
		edge.Width,
		dst,
		ctx.Output.Layout.BytesPerSample,
		ctx.Output.Format.BitDepth,
		edge.Edge,
		edge.X4*4,
		edge.Y4*4,
		edge.Length4*4,
		thresholds,
	)
}

func frameWorkLoopFilterOutputPlane(output frame.Frame, plane loopfilter.Plane) (frame.Plane, bool) {
	switch plane {
	case loopfilter.PlaneY:
		return output.Y, true
	case loopfilter.PlaneU:
		return output.U, !output.Format.MonoChrome
	case loopfilter.PlaneV:
		return output.V, !output.Format.MonoChrome
	default:
		return frame.Plane{}, false
	}
}

func frameWorkCountAppliedLoopFilterEdge(result *FrameWorkLoopFilterPostFilterApplyResult, edge FrameWorkLoopFilterPostFilterEdge) {
	result.Edges++
	if edge.Plane <= loopfilter.PlaneV {
		result.PlaneEdges[int(edge.Plane)]++
	}
	if edge.Level == 0 {
		return
	}
	result.Applied++
	if edge.Plane <= loopfilter.PlaneV {
		plane := int(edge.Plane)
		result.PlaneApplied[plane]++
		if edge.Level > result.PlaneMaxLevel[plane] {
			result.PlaneMaxLevel[plane] = edge.Level
		}
	}
	if edge.Level > result.MaxLevel {
		result.MaxLevel = edge.Level
	}
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

func frameWorkAppendLoopFilterLumaEdges(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge) error {
	if record.SkipTransform {
		block := record.Block
		if block.MICol > 0 {
			verticalLevel, fromPrevious, err := frameWorkResolveLoopFilterLumaEdgeLevel(ctx, filterMap, record, loopfilter.EdgeVertical, int(block.MICol), int(block.MIRow), plan.MICols, plan.MIRows)
			if err != nil {
				return err
			}
			if verticalLevel != 0 {
				if err := frameWorkAppendLoopFilterLumaEdgeSegments(ctx, filterMap, record, plan, edges, loopfilter.EdgeVertical, int(block.MICol), int(block.MIRow), int(block.VisibleH4), record.TransformTree.Y, verticalLevel, fromPrevious); err != nil {
					return err
				}
			}
		}
		if block.MIRow > 0 {
			horizontalLevel, fromPrevious, err := frameWorkResolveLoopFilterLumaEdgeLevel(ctx, filterMap, record, loopfilter.EdgeHorizontal, int(block.MICol), int(block.MIRow), plan.MICols, plan.MIRows)
			if err != nil {
				return err
			}
			if horizontalLevel != 0 {
				if err := frameWorkAppendLoopFilterLumaEdgeSegments(ctx, filterMap, record, plan, edges, loopfilter.EdgeHorizontal, int(block.MICol), int(block.MIRow), int(block.VisibleW4), record.TransformTree.Y, horizontalLevel, fromPrevious); err != nil {
					return err
				}
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
		if frameX4 > 0 {
			verticalLevel, fromPrevious, err := frameWorkResolveLoopFilterLumaEdgeLevel(ctx, filterMap, record, loopfilter.EdgeVertical, frameX4, frameY4, plan.MICols, plan.MIRows)
			if err != nil {
				return err
			}
			if verticalLevel != 0 {
				if err := frameWorkAppendLoopFilterLumaEdgeSegments(ctx, filterMap, record, plan, edges, loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, verticalLevel, fromPrevious); err != nil {
					return err
				}
			}
		}
		if frameY4 > 0 {
			horizontalLevel, fromPrevious, err := frameWorkResolveLoopFilterLumaEdgeLevel(ctx, filterMap, record, loopfilter.EdgeHorizontal, frameX4, frameY4, plan.MICols, plan.MIRows)
			if err != nil {
				return err
			}
			if horizontalLevel != 0 {
				if err := frameWorkAppendLoopFilterLumaEdgeSegments(ctx, filterMap, record, plan, edges, loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, horizontalLevel, fromPrevious); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

// frameWorkAppendLoopFilterLumaEdgeSegments splits one TX-block-aligned luma
// edge into per-MI-cell segments and emits one edge per maximal contiguous run
// of MI cells that share the same resolved filter width. libaom's deblocker
// resolves filter_length per (mi_col, mi_row) using the local current and
// previous transform sizes (set_one_param_for_line_luma), so a single TX edge
// can need different filter widths along its length when the previous-side
// TX subdivision varies. Emitting one edge with min(width) (the old behaviour)
// applied the narrower filter beyond the cells that actually saw the narrower
// previous TX, e.g. the 10-bit quantizer 32 horizontal edge at MI(40, 2) where
// libaom uses flen=8 at col 40 (prev TX_4X8) and flen=4 at col 41 (prev
// TX_4X4); our previous code took min(8, 4) = 4 across the whole TX edge and
// applied filter4 at col 40 too, producing the q32 LF divergence.
func frameWorkAppendLoopFilterLumaEdgeSegments(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, level uint8, levelFromPrevious bool) error {
	if length4 <= 0 {
		return nil
	}
	currentWidth, err := frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx)
	if err != nil {
		return err
	}
	segStart := 0
	segWidth := 0
	emit := func(start, end int) error {
		if segWidth == 0 || end <= start {
			return nil
		}
		segX4 := x4
		segY4 := y4
		if edge == loopfilter.EdgeHorizontal {
			segX4 = x4 + start
		} else {
			segY4 = y4 + start
		}
		w, ok, err := frameWorkLoopFilterScheduledWidth(ctx, loopfilter.PlaneY, edge, segX4, segY4, end-start, segWidth)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
			Plane:             loopfilter.PlaneY,
			Edge:              edge,
			X4:                segX4,
			Y4:                segY4,
			Length4:           end - start,
			Level:             level,
			Transform:         tx,
			Width:             w,
			LevelFromPrevious: levelFromPrevious,
			BlockMICol:        record.Block.MICol,
			BlockMIRow:        record.Block.MIRow,
		})
		return nil
	}
	for offset := 0; offset < length4; offset++ {
		previousWidth, err := frameWorkLoopFilterPreviousLumaCellWidth(ctx, filterMap, edge, x4, y4, offset, plan.MICols, plan.MIRows)
		if err != nil {
			return err
		}
		width := currentWidth
		if previousWidth < width {
			width = previousWidth
		}
		if width != segWidth {
			if err := emit(segStart, offset); err != nil {
				return err
			}
			segStart = offset
			segWidth = width
		}
	}
	return emit(segStart, length4)
}

// frameWorkLoopFilterPreviousLumaCellWidth returns the previous-side luma
// filter width for a single MI cell on the edge.
func frameWorkLoopFilterPreviousLumaCellWidth(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int) (int, error) {
	boundaryX4, boundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, offset)
	targetX4, targetY4, err := frameWorkLoopFilterPreviousTarget4(edge, boundaryX4, boundaryY4)
	if err != nil {
		return 0, err
	}
	previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, boundaryX4, boundaryY4, cols, rows)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, loopfilter.ErrInvalidFilter
	}
	tx, ok, err := frameWorkLoopFilterLumaTransformAt(ctx, previous, targetX4, targetY4)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, threading.ErrInvalidBatch
	}
	return frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx)
}

func frameWorkResolveLoopFilterLumaEdgeLevel(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, record threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge, x4 int, y4 int, cols int, rows int) (uint8, bool, error) {
	level, err := frameWorkResolveLoopFilterLevel(ctx, record, loopfilter.PlaneY, edge)
	if err != nil {
		return 0, false, err
	}
	if level != 0 {
		return level, false, nil
	}
	previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, x4, y4, cols, rows)
	if err != nil || !ok {
		return 0, false, err
	}
	level, err = frameWorkResolveLoopFilterLevel(ctx, previous, loopfilter.PlaneY, edge)
	if err != nil {
		return 0, false, err
	}
	return level, level != 0, nil
}

func frameWorkLoopFilterPreviousRecord(filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, x4 int, y4 int, cols int, rows int) (threading.FrameWorkLoopFilterBlockRecord, bool, error) {
	prevCol := x4
	prevRow := y4
	switch edge {
	case loopfilter.EdgeVertical:
		prevCol--
	case loopfilter.EdgeHorizontal:
		prevRow--
	default:
		return threading.FrameWorkLoopFilterBlockRecord{}, false, loopfilter.ErrInvalidFilter
	}
	if prevCol < 0 || prevRow < 0 {
		return threading.FrameWorkLoopFilterBlockRecord{}, false, nil
	}
	if prevCol >= cols || prevRow >= rows || prevRow >= filterMap.Rows || prevCol >= filterMap.Stride {
		return threading.FrameWorkLoopFilterBlockRecord{}, false, threading.ErrInvalidBatch
	}
	record, ok, err := filterMap.RecordAt(uint32(prevCol), uint32(prevRow))
	if err != nil {
		return threading.FrameWorkLoopFilterBlockRecord{}, false, err
	}
	if !ok {
		return threading.FrameWorkLoopFilterBlockRecord{}, false, threading.ErrInvalidBatch
	}
	if err := frameWorkValidateLoopFilterRecord(record, prevCol, prevRow, cols, rows); err != nil {
		return threading.FrameWorkLoopFilterBlockRecord{}, false, err
	}
	return record, true, nil
}

func frameWorkAppendLoopFilterChromaEdges(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge) error {
	if ctx.Event.SequenceHeader.ColorConfig.MonoChrome {
		return nil
	}
	if ctx.Event.LoopFilter.LevelY[0] == 0 && ctx.Event.LoopFilter.LevelY[1] == 0 {
		return nil
	}
	for plane := loopfilter.PlaneU; plane <= loopfilter.PlaneV; plane++ {
		base, err := loopfilter.BaseLevel(ctx.Event.LoopFilter, plane, loopfilter.EdgeVertical)
		if err != nil {
			return err
		}
		if base == 0 {
			continue
		}
		if record.SkipTransform {
			tx, ok, err := frameWorkLoopFilterChromaBlock(ctx, record)
			if err != nil || !ok {
				return err
			}
			if err := frameWorkAppendLoopFilterChromaTXBEdges(ctx, filterMap, record, plan, edges, plane, tx); err != nil {
				return err
			}
			continue
		}
		if err := frameWorkLoopFilterForEachChromaTXB(ctx, record, func(tx tile.TransformBlock) error {
			return frameWorkAppendLoopFilterChromaTXBEdges(ctx, filterMap, record, plan, edges, plane, tx)
		}); err != nil {
			return err
		}
	}
	return nil
}

func frameWorkAppendLoopFilterChromaTXBEdges(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, plane loopfilter.Plane, tx tile.TransformBlock) error {
	frameX4, frameY4 := frameWorkLoopFilterChromaFrame4(ctx, record, tx.X4, tx.Y4)
	if frameX4 > 0 {
		verticalLevel, fromPrevious, err := frameWorkResolveLoopFilterChromaEdgeLevel(ctx, filterMap, record, plane, loopfilter.EdgeVertical, frameX4, frameY4, plan.MICols, plan.MIRows)
		if err != nil {
			return err
		}
		if verticalLevel != 0 {
			if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, filterMap, record, plan, edges, plane, loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, verticalLevel, fromPrevious); err != nil {
				return err
			}
		}
	}
	if frameY4 > 0 {
		horizontalLevel, fromPrevious, err := frameWorkResolveLoopFilterChromaEdgeLevel(ctx, filterMap, record, plane, loopfilter.EdgeHorizontal, frameX4, frameY4, plan.MICols, plan.MIRows)
		if err != nil {
			return err
		}
		if horizontalLevel != 0 {
			if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, filterMap, record, plan, edges, plane, loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, horizontalLevel, fromPrevious); err != nil {
				return err
			}
		}
	}
	return nil
}

// frameWorkAppendLoopFilterChromaEdgeSegments mirrors
// frameWorkAppendLoopFilterLumaEdgeSegments for chroma. libaom's chroma
// deblock equally resolves filter_length per chroma cell from the local
// current/previous transform sizes (set_one_param_for_line_chroma); without
// per-cell splitting the chroma plane reproduced the same q32 10-bit
// divergence the luma fix already covered.
func frameWorkAppendLoopFilterChromaEdgeSegments(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, record threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, level uint8, levelFromPrevious bool) error {
	if length4 <= 0 {
		return nil
	}
	currentWidth, err := frameWorkLoopFilterWidth(plane, edge, tx)
	if err != nil {
		return err
	}
	segStart := 0
	segWidth := 0
	emit := func(start, end int) error {
		if segWidth == 0 || end <= start {
			return nil
		}
		segX4 := x4
		segY4 := y4
		if edge == loopfilter.EdgeHorizontal {
			segX4 = x4 + start
		} else {
			segY4 = y4 + start
		}
		w, ok, err := frameWorkLoopFilterScheduledWidth(ctx, plane, edge, segX4, segY4, end-start, segWidth)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
			Plane:             plane,
			Edge:              edge,
			X4:                segX4,
			Y4:                segY4,
			Length4:           end - start,
			Level:             level,
			Transform:         tx,
			Width:             w,
			LevelFromPrevious: levelFromPrevious,
			BlockMICol:        record.Block.MICol,
			BlockMIRow:        record.Block.MIRow,
		})
		return nil
	}
	// Like the luma path we want to emit one edge per maximal run of MI
	// cells that share the same filter width. Chroma adds a wrinkle: at some
	// offsets the previous-side luma record co-locates with the current
	// chroma block, so there is no chroma boundary there (the old behaviour
	// returned hasChroma=false for the entire TX edge in that case). libaom
	// processes the chroma edge per cell regardless, so when the per-cell
	// lookup returns "no chroma at this offset" we treat that as a no-op
	// gap and continue with the prevailing width from the surrounding cells.
	hadAny := false
	lastWidth := 0
	for offset := 0; offset < length4; offset++ {
		previousWidth, hasChroma, err := frameWorkLoopFilterPreviousChromaCellWidth(ctx, filterMap, plane, edge, x4, y4, offset, plan.MICols, plan.MIRows)
		if err != nil {
			return err
		}
		var width int
		if hasChroma {
			width = currentWidth
			if previousWidth < width {
				width = previousWidth
			}
			hadAny = true
			lastWidth = width
		} else {
			// Carry the prevailing width so the per-offset filter still runs
			// at this cell (matches libaom's per-MI-cell processing) when
			// surrounding cells share the same width. Cells that never see a
			// chroma neighbour stay no-ops once segWidth=0 falls through.
			width = lastWidth
		}
		if width != segWidth {
			if err := emit(segStart, offset); err != nil {
				return err
			}
			segStart = offset
			segWidth = width
		}
	}
	if !hadAny {
		return nil
	}
	return emit(segStart, length4)
}

// frameWorkLoopFilterPreviousChromaCellWidth resolves one MI cell's
// previous-side chroma filter width, returning hasChroma=false when the
// previous-side luma record co-locates with the current chroma block (so no
// chroma boundary exists there).
func frameWorkLoopFilterPreviousChromaCellWidth(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int) (int, bool, error) {
	boundaryX4, boundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, offset)
	targetX4, targetY4, err := frameWorkLoopFilterPreviousTarget4(edge, boundaryX4, boundaryY4)
	if err != nil {
		return 0, false, err
	}
	lookupX4, lookupY4 := frameWorkLoopFilterChromaPreviousLookup4(ctx, edge, boundaryX4, boundaryY4)
	previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, lookupX4, lookupY4, cols, rows)
	if err != nil {
		return 0, false, err
	}
	if !ok {
		return 0, false, loopfilter.ErrInvalidFilter
	}
	tx, ok, err := frameWorkLoopFilterChromaTransformAt(ctx, previous, targetX4, targetY4)
	if err != nil {
		return 0, false, err
	}
	if !ok {
		return 0, false, nil
	}
	w, err := frameWorkLoopFilterWidth(plane, edge, tx)
	if err != nil {
		return 0, false, err
	}
	return w, true, nil
}

func frameWorkResolveLoopFilterChromaEdgeLevel(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, record threading.FrameWorkLoopFilterBlockRecord, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, cols int, rows int) (uint8, bool, error) {
	level, err := frameWorkResolveLoopFilterLevel(ctx, record, plane, edge)
	if err != nil {
		return 0, false, err
	}
	if level != 0 {
		return level, false, nil
	}
	prevX4, prevY4 := frameWorkLoopFilterChromaPreviousLookup4(ctx, edge, x4, y4)
	previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, prevX4, prevY4, cols, rows)
	if err != nil || !ok {
		return 0, false, err
	}
	level, err = frameWorkResolveLoopFilterLevel(ctx, previous, plane, edge)
	if err != nil {
		return 0, false, err
	}
	return level, level != 0, nil
}

func frameWorkLoopFilterChromaPreviousLookup4(ctx FrameWorkPostFilterContext, edge loopfilter.Edge, x4 int, y4 int) (int, int) {
	color := ctx.Event.SequenceHeader.ColorConfig
	ssX := frameWorkLoopFilterSubsamplingShift(color.SubsamplingX)
	ssY := frameWorkLoopFilterSubsamplingShift(color.SubsamplingY)
	switch edge {
	case loopfilter.EdgeVertical:
		return x4 << ssX, y4 << ssY
	case loopfilter.EdgeHorizontal:
		return x4 << ssX, y4 << ssY
	default:
		return x4, y4
	}
}

func frameWorkLoopFilterScheduledLumaWidth(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, cols int, rows int) (int, bool, error) {
	width, err := frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx)
	if err != nil {
		return 0, false, err
	}
	previousWidth, err := frameWorkLoopFilterPreviousLumaMinWidth(ctx, filterMap, edge, x4, y4, length4, cols, rows)
	if err != nil {
		return 0, false, err
	}
	if previousWidth < width {
		width = previousWidth
	}
	return frameWorkLoopFilterScheduledWidth(ctx, loopfilter.PlaneY, edge, x4, y4, length4, width)
}

func frameWorkLoopFilterScheduledChromaWidth(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, cols int, rows int) (int, bool, error) {
	width, err := frameWorkLoopFilterWidth(plane, edge, tx)
	if err != nil {
		return 0, false, err
	}
	previousWidth, hasChroma, err := frameWorkLoopFilterPreviousChromaMinWidth(ctx, filterMap, plane, edge, x4, y4, length4, cols, rows)
	if err != nil {
		return 0, false, err
	}
	if !hasChroma {
		return 0, false, nil
	}
	if previousWidth < width {
		width = previousWidth
	}
	return frameWorkLoopFilterScheduledWidth(ctx, plane, edge, x4, y4, length4, width)
}

func frameWorkLoopFilterScheduledWidth(ctx FrameWorkPostFilterContext, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, width int) (int, bool, error) {
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

func frameWorkLoopFilterPreviousLumaMinWidth(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, x4 int, y4 int, length4 int, cols int, rows int) (int, error) {
	minWidth := 0
	for offset := 0; offset < length4; offset++ {
		boundaryX4, boundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, offset)
		targetX4, targetY4, err := frameWorkLoopFilterPreviousTarget4(edge, boundaryX4, boundaryY4)
		if err != nil {
			return 0, err
		}
		previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, boundaryX4, boundaryY4, cols, rows)
		if err != nil {
			return 0, err
		}
		if !ok {
			return 0, loopfilter.ErrInvalidFilter
		}
		tx, ok, err := frameWorkLoopFilterLumaTransformAt(ctx, previous, targetX4, targetY4)
		if err != nil {
			return 0, err
		}
		if !ok {
			return 0, threading.ErrInvalidBatch
		}
		width, err := frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx)
		if err != nil {
			return 0, err
		}
		if minWidth == 0 || width < minWidth {
			minWidth = width
		}
	}
	if minWidth == 0 {
		return 0, loopfilter.ErrInvalidFilter
	}
	return minWidth, nil
}

// frameWorkLoopFilterPreviousChromaMinWidth returns the minimum chroma filter
// width across the edge offsets. If every previous-side luma record shares its
// chroma with the current block (e.g. small blocks below the chroma min-block
// size), there is no chroma edge to schedule at this position and the function
// returns hasChroma=false.
func frameWorkLoopFilterPreviousChromaMinWidth(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, cols int, rows int) (int, bool, error) {
	minWidth := 0
	hasChroma := false
	for offset := 0; offset < length4; offset++ {
		boundaryX4, boundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, offset)
		targetX4, targetY4, err := frameWorkLoopFilterPreviousTarget4(edge, boundaryX4, boundaryY4)
		if err != nil {
			return 0, false, err
		}
		lookupX4, lookupY4 := frameWorkLoopFilterChromaPreviousLookup4(ctx, edge, boundaryX4, boundaryY4)
		previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, lookupX4, lookupY4, cols, rows)
		if err != nil {
			return 0, false, err
		}
		if !ok {
			return 0, false, loopfilter.ErrInvalidFilter
		}
		tx, ok, err := frameWorkLoopFilterChromaTransformAt(ctx, previous, targetX4, targetY4)
		if err != nil {
			return 0, false, err
		}
		if !ok {
			continue
		}
		hasChroma = true
		width, err := frameWorkLoopFilterWidth(plane, edge, tx)
		if err != nil {
			return 0, false, err
		}
		if minWidth == 0 || width < minWidth {
			minWidth = width
		}
	}
	if !hasChroma {
		return 0, false, nil
	}
	if minWidth == 0 {
		return 0, false, loopfilter.ErrInvalidFilter
	}
	return minWidth, true, nil
}

func frameWorkLoopFilterBoundaryOffset(edge loopfilter.Edge, x4 int, y4 int, offset int) (int, int) {
	switch edge {
	case loopfilter.EdgeVertical:
		return x4, y4 + offset
	case loopfilter.EdgeHorizontal:
		return x4 + offset, y4
	default:
		return x4, y4
	}
}

func frameWorkLoopFilterPreviousTarget4(edge loopfilter.Edge, x4 int, y4 int) (int, int, error) {
	switch edge {
	case loopfilter.EdgeVertical:
		return x4 - 1, y4, nil
	case loopfilter.EdgeHorizontal:
		return x4, y4 - 1, nil
	default:
		return 0, 0, loopfilter.ErrInvalidFilter
	}
}

func frameWorkLoopFilterLumaTransformAt(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, frameX4 int, frameY4 int) (tile.TransformSize, bool, error) {
	req, err := frameWorkLoopFilterTransformTreeRequest(ctx, record)
	if err != nil {
		return 0, false, err
	}
	localX4 := record.Block.X4 + frameX4 - int(record.Block.MICol)
	localY4 := record.Block.Y4 + frameY4 - int(record.Block.MIRow)
	if localX4 < req.X4 || localY4 < req.Y4 ||
		localX4 >= req.X4+int(req.VisibleW4) ||
		localY4 >= req.Y4+int(req.VisibleH4) {
		return 0, false, nil
	}
	if record.SkipTransform {
		if !record.TransformTree.Y.Valid() {
			return 0, false, threading.ErrInvalidBatch
		}
		return record.TransformTree.Y, true, nil
	}
	found := tile.TransformSize(0)
	ok := false
	if err := record.TransformTree.ForEachLumaTXB(req, func(tx tile.TransformBlock) error {
		if frameWorkLoopFilterTransformBlockContains(tx, localX4, localY4) {
			found = tx.Size
			ok = true
		}
		return nil
	}); err != nil {
		return 0, false, err
	}
	return found, ok, nil
}

func frameWorkLoopFilterChromaTransformAt(ctx FrameWorkPostFilterContext, record threading.FrameWorkLoopFilterBlockRecord, frameX4 int, frameY4 int) (tile.TransformSize, bool, error) {
	block, ok, err := frameWorkLoopFilterChromaBlock(ctx, record)
	if err != nil || !ok {
		return 0, ok, err
	}
	color := ctx.Event.SequenceHeader.ColorConfig
	ssX := frameWorkLoopFilterSubsamplingShift(color.SubsamplingX)
	ssY := frameWorkLoopFilterSubsamplingShift(color.SubsamplingY)
	localX4 := (record.Block.X4 >> ssX) + frameX4 - (int(record.Block.MICol) >> ssX)
	localY4 := (record.Block.Y4 >> ssY) + frameY4 - (int(record.Block.MIRow) >> ssY)
	if !frameWorkLoopFilterTransformBlockContains(block, localX4, localY4) {
		return 0, false, nil
	}
	if record.SkipTransform {
		return block.Size, true, nil
	}
	found := tile.TransformSize(0)
	foundBlock := false
	if err := frameWorkLoopFilterForEachChromaTXB(ctx, record, func(tx tile.TransformBlock) error {
		if frameWorkLoopFilterTransformBlockContains(tx, localX4, localY4) {
			found = tx.Size
			foundBlock = true
		}
		return nil
	}); err != nil {
		return 0, false, err
	}
	return found, foundBlock, nil
}

func frameWorkLoopFilterTransformBlockContains(tx tile.TransformBlock, x4 int, y4 int) bool {
	return x4 >= tx.X4 && y4 >= tx.Y4 &&
		x4 < tx.X4+int(tx.VisibleW4) &&
		y4 < tx.Y4+int(tx.VisibleH4)
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
		// libaom av1_loopfilter.c: chroma uses 4-tap when dim==0 (span4<=1)
		// and 6-tap otherwise. The 8-tap and 14-tap filters are luma-only.
		if span4 <= 1 {
			return 4, nil
		}
		return 6, nil
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
	case 6:
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
	if edge.LevelFromPrevious {
		plan.PreviousLevelEdges++
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
