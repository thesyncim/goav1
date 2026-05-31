package decoder

import (
	"errors"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

var errFrameWorkLoopFilterTransformFound = errors.New("loop filter transform found")

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

type frameWorkLoopFilterPlanningContext struct {
	bounds [3]frameWorkLoopFilterBounds
	color  parser.ColorConfig
	ssX    int
	ssY    int
}

type frameWorkLoopFilterLevelContext struct {
	loopFilter   *parser.LoopFilterParams
	segmentation *parser.SegmentationParams
	delta        *parser.DeltaParams
	base         [3][2]uint8
	lumaZero     bool
	monoChrome   bool
}

func frameWorkLoopFilterLevelContextFor(event *Event) frameWorkLoopFilterLevelContext {
	loopFilter := &event.LoopFilter
	return frameWorkLoopFilterLevelContext{
		loopFilter:   loopFilter,
		segmentation: &event.Segmentation,
		delta:        &event.Delta,
		base: [3][2]uint8{
			{loopfilter.ClampLevel(int(loopFilter.LevelY[0])), loopfilter.ClampLevel(int(loopFilter.LevelY[1]))},
			{loopfilter.ClampLevel(int(loopFilter.LevelU)), loopfilter.ClampLevel(int(loopFilter.LevelU))},
			{loopfilter.ClampLevel(int(loopFilter.LevelV)), loopfilter.ClampLevel(int(loopFilter.LevelV))},
		},
		lumaZero:   loopFilter.LevelY[0] == 0 && loopFilter.LevelY[1] == 0,
		monoChrome: event.SequenceHeader.ColorConfig.MonoChrome,
	}
}

func frameWorkLoopFilterPlanningContextFor(ctx FrameWorkPostFilterContext) (frameWorkLoopFilterPlanningContext, error) {
	color := ctx.Event.SequenceHeader.ColorConfig
	planning := frameWorkLoopFilterPlanningContext{
		color: color,
		ssX:   frameWorkLoopFilterSubsamplingShift(color.SubsamplingX),
		ssY:   frameWorkLoopFilterSubsamplingShift(color.SubsamplingY),
	}
	bounds, err := frameWorkLoopFilterPlaneBounds(ctx, loopfilter.PlaneY)
	if err != nil {
		return frameWorkLoopFilterPlanningContext{}, err
	}
	planning.bounds[loopfilter.PlaneY] = bounds
	if color.MonoChrome {
		return planning, nil
	}
	for plane := loopfilter.PlaneU; plane <= loopfilter.PlaneV; plane++ {
		bounds, err := frameWorkLoopFilterPlaneBounds(ctx, plane)
		if err != nil {
			return frameWorkLoopFilterPlanningContext{}, err
		}
		planning.bounds[plane] = bounds
	}
	return planning, nil
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
	planning, err := frameWorkLoopFilterPlanningContextFor(ctx)
	if err != nil {
		return plan, err
	}
	levelCtx := frameWorkLoopFilterLevelContextFor(&ctx.Event)
	if err := frameWorkLoopFilterForEachValidatedBlock(filterMap, cols, rows, &plan, func(record *threading.FrameWorkLoopFilterBlockRecord) error {
		levels, err := frameWorkResolveLoopFilterRecordLevels(levelCtx, record)
		if err != nil {
			return err
		}
		frameWorkAccumulateLoopFilterLevelStats(levelCtx, levels, &plan)
		if err := frameWorkAccumulateLoopFilterTransformStats(ctx, record, &plan); err != nil {
			return err
		}
		if err := frameWorkAppendLoopFilterLumaEdges(ctx, levelCtx, filterMap, record, &plan, req.Edges, planning.bounds[loopfilter.PlaneY], levels); err != nil {
			return err
		}
		return frameWorkAppendLoopFilterChromaEdges(ctx, levelCtx, filterMap, record, &plan, req.Edges, planning, levels)
	}); err != nil {
		return plan, err
	}
	return plan, nil
}

func frameWorkLoopFilterForEachValidatedBlock(filterMap FrameWorkLoopFilterMap, cols int, rows int, plan *FrameWorkLoopFilterPostFilterPlan, visit func(*threading.FrameWorkLoopFilterBlockRecord) error) error {
	if plan == nil || visit == nil {
		return threading.ErrInvalidBatch
	}
	for row := 0; row < rows; row++ {
		base := row * filterMap.Stride
		for col := 0; col < cols; col++ {
			record := &filterMap.Records[base+col]
			if !record.Valid {
				plan.Missing++
				continue
			}
			if err := frameWorkValidateLoopFilterRecord(record, col, row, cols, rows); err != nil {
				return err
			}
			plan.Cells++
			if record.Block.MICol != uint32(col) || record.Block.MIRow != uint32(row) {
				continue
			}
			plan.Blocks++
			if err := visit(record); err != nil {
				return err
			}
		}
	}
	if plan.Missing != 0 {
		return threading.ErrInvalidBatch
	}
	return nil
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
	var planes [3]frame.Plane
	var planeReady [3]bool
	var thresholds [loopfilter.MaxLevel + 1]loopfilter.Thresholds
	var thresholdReady [loopfilter.MaxLevel + 1]bool
	bytesPerSample := ctx.Output.Layout.BytesPerSample
	bitDepth := ctx.Output.Format.BitDepth
	for plane := loopfilter.PlaneY; plane <= maxPlane; plane++ {
		for edgeKind := loopfilter.EdgeVertical; edgeKind <= loopfilter.EdgeHorizontal; edgeKind++ {
			for i := range edges {
				edge := edges[i]
				if edge.Plane != plane || edge.Edge != edgeKind {
					continue
				}
				if edge.Length4 <= 0 || edge.Level == 0 {
					return loopfilter.ErrInvalidFilter
				}
				if edge.Level > loopfilter.MaxLevel {
					return loopfilter.ErrInvalidFilter
				}
				if !planeReady[plane] {
					dst, ok := frameWorkLoopFilterOutputPlane(*ctx.Output, plane)
					if !ok {
						return loopfilter.ErrInvalidFilter
					}
					planeW, planeH, err := frameWorkLoopFilterBufferSize(ctx, plane)
					if err != nil {
						return err
					}
					planes[plane] = frameWorkLoopFilterAlignedPlane(dst, planeW, planeH, bytesPerSample)
					planeReady[plane] = true
				}
				if !thresholdReady[edge.Level] {
					th, err := loopfilter.ThresholdsForLevel(edge.Level, ctx.Event.LoopFilter.Sharpness)
					if err != nil {
						return err
					}
					thresholds[edge.Level] = th
					thresholdReady[edge.Level] = true
				}
				if err := loopfilter.FilterEdgeByWidth(
					edge.Width,
					planes[plane],
					bytesPerSample,
					bitDepth,
					edge.Edge,
					edge.X4*4,
					edge.Y4*4,
					edge.Length4*4,
					thresholds[edge.Level],
				); err != nil {
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

func (ctx FrameWorkPostFilterContext) applyLoopFilterEdge(edge FrameWorkLoopFilterPostFilterEdge) error {
	if edge.Length4 <= 0 || edge.Level == 0 || ctx.Output == nil {
		return loopfilter.ErrInvalidFilter
	}
	dst, ok := frameWorkLoopFilterOutputPlane(*ctx.Output, edge.Plane)
	if !ok {
		return loopfilter.ErrInvalidFilter
	}
	// libaom's loop filter taps read/write the bottom/right partial-superblock
	// padding the reconstruction stage populated. The byte buffer is allocated at
	// the 8-aligned MI grid (frame.RequiredSize / frameWorkLoopFilterBufferSize),
	// so expand the cropped plane view to that extent before filtering; otherwise
	// near-boundary edges whose taps reach into the padding (e.g. a width-14
	// filter at the 226x226 right edge) would exceed the cropped Width/Height and
	// be rejected.
	planeW, planeH, err := frameWorkLoopFilterBufferSize(ctx, edge.Plane)
	if err != nil {
		return err
	}
	dst = frameWorkLoopFilterAlignedPlane(dst, planeW, planeH, ctx.Output.Layout.BytesPerSample)
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

// frameWorkLoopFilterAlignedPlane expands a cropped plane view to the MI-aligned
// deblock extent (planeW x planeH from frameWorkLoopFilterPlaneSize). The byte
// buffer was allocated at the MI-aligned dimensions by frame.RequiredSize, so
// the expanded view shares Pix and Stride; only Width/Height grow, capped to the
// row stride and the allocated row count. This mirrors frameWorkCDEFAlignedPlane
// for the loop-filter stage.
func frameWorkLoopFilterAlignedPlane(plane frame.Plane, planeW int, planeH int, bytesPerSample int) frame.Plane {
	if bytesPerSample <= 0 || plane.Stride <= 0 {
		return plane
	}
	if planeW > plane.Width {
		if w := plane.Stride / bytesPerSample; planeW > w {
			planeW = w
		}
		if planeW > plane.Width {
			plane.Width = planeW
		}
	}
	if planeH > plane.Height {
		if h := len(plane.Pix) / plane.Stride; planeH > h {
			planeH = h
		}
		if planeH > plane.Height {
			plane.Height = planeH
		}
	}
	return plane
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

func frameWorkAccumulateLoopFilterTransformStats(ctx FrameWorkPostFilterContext, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan) error {
	req, err := frameWorkLoopFilterTransformTreeRequest(ctx.Event.SequenceHeader.ColorConfig, record)
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
	color := ctx.Event.SequenceHeader.ColorConfig
	count, err := frameWorkLoopFilterCountChromaTXBsWithShifts(color, record, frameWorkLoopFilterSubsamplingShift(color.SubsamplingX), frameWorkLoopFilterSubsamplingShift(color.SubsamplingY))
	if err != nil {
		return err
	}
	plan.ChromaTXBs += count
	return nil
}

func frameWorkAppendLoopFilterLumaEdges(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, levels [3][2]uint8) error {
	currentVertical := levels[loopfilter.PlaneY][loopfilter.EdgeVertical]
	currentHorizontal := levels[loopfilter.PlaneY][loopfilter.EdgeHorizontal]
	if record.SkipTransform {
		block := record.Block
		if block.MICol > 0 {
			if err := frameWorkAppendLoopFilterLumaEdgeSegments(ctx, levelCtx, filterMap, record, plan, edges, bounds, loopfilter.EdgeVertical, int(block.MICol), int(block.MIRow), int(block.VisibleH4), record.TransformTree.Y, currentVertical); err != nil {
				return err
			}
		}
		if block.MIRow > 0 {
			if err := frameWorkAppendLoopFilterLumaEdgeSegments(ctx, levelCtx, filterMap, record, plan, edges, bounds, loopfilter.EdgeHorizontal, int(block.MICol), int(block.MIRow), int(block.VisibleW4), record.TransformTree.Y, currentHorizontal); err != nil {
				return err
			}
		}
		return nil
	}

	req, err := frameWorkLoopFilterTransformTreeRequest(ctx.Event.SequenceHeader.ColorConfig, record)
	if err != nil {
		return err
	}
	block := record.Block
	return record.TransformTree.ForEachLumaTXB(req, func(tx tile.TransformBlock) error {
		frameX4 := int(block.MICol) + tx.X4 - block.X4
		frameY4 := int(block.MIRow) + tx.Y4 - block.Y4
		if frameX4 > 0 {
			if err := frameWorkAppendLoopFilterLumaEdgeSegments(ctx, levelCtx, filterMap, record, plan, edges, bounds, loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, currentVertical); err != nil {
				return err
			}
		}
		if frameY4 > 0 {
			if err := frameWorkAppendLoopFilterLumaEdgeSegments(ctx, levelCtx, filterMap, record, plan, edges, bounds, loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, currentHorizontal); err != nil {
				return err
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
func frameWorkAppendLoopFilterLumaEdgeSegments(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevel uint8) error {
	if length4 <= 0 {
		return nil
	}
	length4, err := frameWorkLoopFilterClampEdgeLengthInBounds(bounds, edge, x4, y4, length4)
	if err != nil {
		return err
	}
	if length4 <= 0 {
		return nil
	}
	currentWidth, err := frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx)
	if err != nil {
		return err
	}
	segStart := 0
	segWidth := 0
	var segLevel uint8
	var segFromPrevious bool
	emit := func(start, end int) error {
		if segWidth == 0 || segLevel == 0 || end <= start {
			return nil
		}
		segX4 := x4
		segY4 := y4
		if edge == loopfilter.EdgeHorizontal {
			segX4 = x4 + start
		} else {
			segY4 = y4 + start
		}
		w, ok, err := frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, segX4, segY4, end-start, segWidth)
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
			Level:             segLevel,
			Transform:         tx,
			Width:             w,
			LevelFromPrevious: segFromPrevious,
			BlockMICol:        record.Block.MICol,
			BlockMIRow:        record.Block.MIRow,
		})
		return nil
	}
	var previousCache frameWorkLoopFilterLumaPreviousCache
	needPreviousLevel := currentLevel == 0
	if handled, err := frameWorkTryAppendLoopFilterFixedLumaEdge(levelCtx, filterMap, ctx.Event.SequenceHeader.ColorConfig, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentWidth, currentLevel, needPreviousLevel); handled || err != nil {
		return err
	}
	for offset := range length4 {
		// libaom resolves filter level and width per MI cell along the edge
		// (set_lpf_parameters): the current block's level is constant, but the
		// previous-side block (and its transform size) can vary cell-by-cell,
		// so a single TX edge can span cells with different levels/widths. When
		// the current level is zero, the edge still filters with the previous
		// block's level (level_from_previous); a TX edge bordering an
		// intra block at one cell and an inter (level-0) block at another must
		// therefore split on both width and level.
		previousWidth, previousLevel, err := previousCache.lookup(levelCtx, filterMap, ctx.Event.SequenceHeader.ColorConfig, edge, x4, y4, offset, plan.MICols, plan.MIRows, needPreviousLevel)
		if err != nil {
			return err
		}
		level := currentLevel
		fromPrevious := false
		if level == 0 {
			level = previousLevel
			fromPrevious = previousLevel != 0
		}
		width := min(previousWidth, currentWidth)
		if level == 0 {
			width = 0
		}
		if width != segWidth || level != segLevel || fromPrevious != segFromPrevious {
			if err := emit(segStart, offset); err != nil {
				return err
			}
			segStart = offset
			segWidth = width
			segLevel = level
			segFromPrevious = fromPrevious
		}
	}
	return emit(segStart, length4)
}

func frameWorkTryAppendLoopFilterFixedLumaEdge(levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentWidth int, currentLevel uint8, needPreviousLevel bool) (bool, error) {
	previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, x4, y4, plan.MICols, plan.MIRows)
	if err != nil {
		return true, err
	}
	if !ok {
		return true, loopfilter.ErrInvalidFilter
	}
	if !frameWorkLoopFilterPreviousRecordCoversRun(previous, edge, x4, y4, length4) {
		return false, nil
	}
	if !previous.SkipTransform && previous.TransformTree.Variable {
		return false, nil
	}
	if !previous.TransformTree.Y.Valid() {
		return true, threading.ErrInvalidBatch
	}
	req, err := frameWorkLoopFilterTransformTreeRequest(color, previous)
	if err != nil {
		return true, err
	}
	if !frameWorkLoopFilterPreviousLumaRunInRequest(previous, req, edge, x4, y4, length4) {
		return false, nil
	}
	previousWidth, err := frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, previous.TransformTree.Y)
	if err != nil {
		return true, err
	}
	level := currentLevel
	fromPrevious := false
	if level == 0 && needPreviousLevel {
		previousLevel, err := frameWorkResolveLoopFilterLevel(levelCtx, previous, loopfilter.PlaneY, edge)
		if err != nil {
			return true, err
		}
		level = previousLevel
		fromPrevious = previousLevel != 0
	}
	if level == 0 {
		return true, nil
	}
	width := min(previousWidth, currentWidth)
	if width == 0 {
		return true, nil
	}
	w, ok, err := frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, x4, y4, length4, width)
	if err != nil {
		return true, err
	}
	if !ok {
		return true, nil
	}
	frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
		Plane:             loopfilter.PlaneY,
		Edge:              edge,
		X4:                x4,
		Y4:                y4,
		Length4:           length4,
		Level:             level,
		Transform:         tx,
		Width:             w,
		LevelFromPrevious: fromPrevious,
		BlockMICol:        record.Block.MICol,
		BlockMIRow:        record.Block.MIRow,
	})
	return true, nil
}

func frameWorkLoopFilterPreviousRecordCoversRun(record *threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge, x4 int, y4 int, length4 int) bool {
	if record == nil || length4 <= 0 {
		return false
	}
	block := record.Block
	switch edge {
	case loopfilter.EdgeVertical:
		prevCol := x4 - 1
		return prevCol >= int(block.MICol) && prevCol < int(block.MIColEnd) &&
			y4 >= int(block.MIRow) && y4+length4 <= int(block.MIRowEnd)
	case loopfilter.EdgeHorizontal:
		prevRow := y4 - 1
		return prevRow >= int(block.MIRow) && prevRow < int(block.MIRowEnd) &&
			x4 >= int(block.MICol) && x4+length4 <= int(block.MIColEnd)
	default:
		return false
	}
}

func frameWorkLoopFilterPreviousLumaRunInRequest(record *threading.FrameWorkLoopFilterBlockRecord, req tile.TransformTreeRequest, edge loopfilter.Edge, x4 int, y4 int, length4 int) bool {
	startX4, startY4, err := frameWorkLoopFilterPreviousTarget4(edge, x4, y4)
	if err != nil {
		return false
	}
	endBoundaryX4, endBoundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, length4-1)
	endX4, endY4, err := frameWorkLoopFilterPreviousTarget4(edge, endBoundaryX4, endBoundaryY4)
	if err != nil {
		return false
	}
	startLocalX4 := record.Block.X4 + startX4 - int(record.Block.MICol)
	startLocalY4 := record.Block.Y4 + startY4 - int(record.Block.MIRow)
	endLocalX4 := record.Block.X4 + endX4 - int(record.Block.MICol)
	endLocalY4 := record.Block.Y4 + endY4 - int(record.Block.MIRow)
	return startLocalX4 >= req.X4 && startLocalY4 >= req.Y4 &&
		startLocalX4 < req.X4+int(req.VisibleW4) &&
		startLocalY4 < req.Y4+int(req.VisibleH4) &&
		endLocalX4 >= req.X4 && endLocalY4 >= req.Y4 &&
		endLocalX4 < req.X4+int(req.VisibleW4) &&
		endLocalY4 < req.Y4+int(req.VisibleH4)
}

type frameWorkLoopFilterLumaPreviousCache struct {
	valid      bool
	miCol      uint32
	miRow      uint32
	record     *threading.FrameWorkLoopFilterBlockRecord
	req        tile.TransformTreeRequest
	fixed      bool
	width      int
	tx         tile.TransformBlock
	txWidth    int
	txValid    bool
	level      uint8
	levelValid bool
}

func (c *frameWorkLoopFilterLumaPreviousCache) lookup(levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int, needLevel bool) (int, uint8, error) {
	boundaryX4, boundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, offset)
	targetX4, targetY4, err := frameWorkLoopFilterPreviousTarget4(edge, boundaryX4, boundaryY4)
	if err != nil {
		return 0, 0, err
	}
	previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, boundaryX4, boundaryY4, cols, rows)
	if err != nil {
		return 0, 0, err
	}
	if !ok {
		return 0, 0, loopfilter.ErrInvalidFilter
	}
	if !c.valid || c.miCol != previous.Block.MICol || c.miRow != previous.Block.MIRow {
		req, err := frameWorkLoopFilterTransformTreeRequest(color, previous)
		if err != nil {
			return 0, 0, err
		}
		fixed := previous.SkipTransform || !previous.TransformTree.Variable
		width := 0
		if fixed {
			if !previous.TransformTree.Y.Valid() {
				return 0, 0, threading.ErrInvalidBatch
			}
			width, err = frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, previous.TransformTree.Y)
			if err != nil {
				return 0, 0, err
			}
		}
		*c = frameWorkLoopFilterLumaPreviousCache{
			valid:  true,
			miCol:  previous.Block.MICol,
			miRow:  previous.Block.MIRow,
			record: previous,
			req:    req,
			fixed:  fixed,
			width:  width,
		}
	}
	localX4 := c.record.Block.X4 + targetX4 - int(c.record.Block.MICol)
	localY4 := c.record.Block.Y4 + targetY4 - int(c.record.Block.MIRow)
	if localX4 < c.req.X4 || localY4 < c.req.Y4 ||
		localX4 >= c.req.X4+int(c.req.VisibleW4) ||
		localY4 >= c.req.Y4+int(c.req.VisibleH4) {
		return 0, 0, threading.ErrInvalidBatch
	}
	width := c.width
	if !c.fixed {
		if !c.txValid || !frameWorkLoopFilterTransformBlockContains(c.tx, localX4, localY4) {
			tx, ok, err := frameWorkLoopFilterFindLumaTransform(c.record.TransformTree, c.req, localX4, localY4)
			if err != nil {
				return 0, 0, err
			}
			if !ok {
				return 0, 0, threading.ErrInvalidBatch
			}
			width, err = frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx.Size)
			if err != nil {
				return 0, 0, err
			}
			c.tx = tx
			c.txWidth = width
			c.txValid = true
		} else {
			width = c.txWidth
		}
	}
	if needLevel && !c.levelValid {
		level, err := frameWorkResolveLoopFilterLevel(levelCtx, c.record, loopfilter.PlaneY, edge)
		if err != nil {
			return 0, 0, err
		}
		c.level = level
		c.levelValid = true
	}
	return width, c.level, nil
}

// frameWorkLoopFilterPreviousLumaCellLevel resolves the loop-filter level of the
// block on the previous side of the edge for a single MI cell. libaom uses this
// previous-block level when the current block's resolved level is zero
// (get_filter_level fallback in set_lpf_parameters).
func frameWorkLoopFilterPreviousLumaCellLevel(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int) (uint8, error) {
	boundaryX4, boundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, offset)
	previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, boundaryX4, boundaryY4, cols, rows)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, nil
	}
	return frameWorkResolveLoopFilterLevel(frameWorkLoopFilterLevelContextFor(&ctx.Event), previous, loopfilter.PlaneY, edge)
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

func frameWorkLoopFilterPreviousRecord(filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, x4 int, y4 int, cols int, rows int) (*threading.FrameWorkLoopFilterBlockRecord, bool, error) {
	prevCol := x4
	prevRow := y4
	switch edge {
	case loopfilter.EdgeVertical:
		prevCol--
	case loopfilter.EdgeHorizontal:
		prevRow--
	default:
		return nil, false, loopfilter.ErrInvalidFilter
	}
	if prevCol < 0 || prevRow < 0 {
		return nil, false, nil
	}
	if prevCol >= cols || prevRow >= rows || prevRow >= filterMap.Rows || prevCol >= filterMap.Stride {
		return nil, false, threading.ErrInvalidBatch
	}
	record := &filterMap.Records[prevRow*filterMap.Stride+prevCol]
	if !record.Valid {
		return nil, false, threading.ErrInvalidBatch
	}
	return record, true, nil
}

func frameWorkAppendLoopFilterChromaEdges(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, planning frameWorkLoopFilterPlanningContext, levels [3][2]uint8) error {
	if ctx.Event.SequenceHeader.ColorConfig.MonoChrome {
		return nil
	}
	if ctx.Event.LoopFilter.LevelY[0] == 0 && ctx.Event.LoopFilter.LevelY[1] == 0 {
		return nil
	}
	var planes [2]loopfilter.Plane
	var bounds [2]frameWorkLoopFilterBounds
	var vertical [2]uint8
	var horizontal [2]uint8
	active := 0
	for plane := loopfilter.PlaneU; plane <= loopfilter.PlaneV; plane++ {
		base, err := loopfilter.BaseLevel(ctx.Event.LoopFilter, plane, loopfilter.EdgeVertical)
		if err != nil {
			return err
		}
		if base == 0 {
			continue
		}
		planes[active] = plane
		bounds[active] = planning.bounds[plane]
		vertical[active] = levels[plane][loopfilter.EdgeVertical]
		horizontal[active] = levels[plane][loopfilter.EdgeHorizontal]
		active++
	}
	if active == 0 {
		return nil
	}
	fuseUV := active == 2 && planes[0] == loopfilter.PlaneU && planes[1] == loopfilter.PlaneV && bounds[0] == bounds[1]
	if record.SkipTransform {
		tx, ok, err := frameWorkLoopFilterChromaBlockWithShifts(planning.color, record, planning.ssX, planning.ssY)
		if err != nil || !ok {
			return err
		}
		if fuseUV {
			return frameWorkAppendLoopFilterChromaTXBEdgesUV(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], tx, vertical[0], vertical[1], horizontal[0], horizontal[1], planning.ssX, planning.ssY)
		}
		for i := 0; i < active; i++ {
			if err := frameWorkAppendLoopFilterChromaTXBEdges(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[i], planes[i], tx, vertical[i], horizontal[i], planning.ssX, planning.ssY); err != nil {
				return err
			}
		}
		return nil
	}
	return frameWorkAppendLoopFilterChromaTXBs(ctx, levelCtx, filterMap, record, plan, edges, planning, bounds, planes, vertical, horizontal, active, fuseUV)
}

func frameWorkAppendLoopFilterChromaTXBs(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, planning frameWorkLoopFilterPlanningContext, bounds [2]frameWorkLoopFilterBounds, planes [2]loopfilter.Plane, vertical [2]uint8, horizontal [2]uint8, active int, fuseUV bool) error {
	block, ok, err := frameWorkLoopFilterChromaBlockWithShifts(planning.color, record, planning.ssX, planning.ssY)
	if err != nil || !ok {
		return err
	}
	uvDims, ok := record.TransformTree.UV.Dimensions()
	if !ok {
		return threading.ErrInvalidBatch
	}
	for y := 0; y < int(block.VisibleH4); y += int(uvDims.H4) {
		for x := 0; x < int(block.VisibleW4); x += int(uvDims.W4) {
			tx := tile.TransformBlock{
				X4:        block.X4 + x,
				Y4:        block.Y4 + y,
				Size:      record.TransformTree.UV,
				VisibleW4: uint8(minInt(int(uvDims.W4), int(block.VisibleW4)-x)),
				VisibleH4: uint8(minInt(int(uvDims.H4), int(block.VisibleH4)-y)),
			}
			frameX4, frameY4 := frameWorkLoopFilterChromaFrame4WithShifts(record, tx.X4, tx.Y4, planning.ssX, planning.ssY)
			if fuseUV {
				if frameX4 > 0 {
					if err := frameWorkAppendLoopFilterChromaEdgeSegmentsUV(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, vertical[0], vertical[1], planning.ssX, planning.ssY); err != nil {
						return err
					}
				}
				if frameY4 > 0 {
					if err := frameWorkAppendLoopFilterChromaEdgeSegmentsUV(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, horizontal[0], horizontal[1], planning.ssX, planning.ssY); err != nil {
						return err
					}
				}
				continue
			}
			for i := 0; i < active; i++ {
				if frameX4 > 0 {
					if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[i], planes[i], loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, vertical[i], planning.ssX, planning.ssY); err != nil {
						return err
					}
				}
				if frameY4 > 0 {
					if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[i], planes[i], loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, horizontal[i], planning.ssX, planning.ssY); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func frameWorkAppendLoopFilterChromaTXBEdges(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, plane loopfilter.Plane, tx tile.TransformBlock, currentVertical uint8, currentHorizontal uint8, ssX int, ssY int) error {
	frameX4, frameY4 := frameWorkLoopFilterChromaFrame4WithShifts(record, tx.X4, tx.Y4, ssX, ssY)
	if frameX4 > 0 {
		if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, plane, loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, currentVertical, ssX, ssY); err != nil {
			return err
		}
	}
	if frameY4 > 0 {
		if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, plane, loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, currentHorizontal, ssX, ssY); err != nil {
			return err
		}
	}
	return nil
}

func frameWorkAppendLoopFilterChromaTXBEdgesUV(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, tx tile.TransformBlock, currentVerticalU uint8, currentVerticalV uint8, currentHorizontalU uint8, currentHorizontalV uint8, ssX int, ssY int) error {
	frameX4, frameY4 := frameWorkLoopFilterChromaFrame4WithShifts(record, tx.X4, tx.Y4, ssX, ssY)
	if frameX4 > 0 {
		if err := frameWorkAppendLoopFilterChromaEdgeSegmentsUV(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, currentVerticalU, currentVerticalV, ssX, ssY); err != nil {
			return err
		}
	}
	if frameY4 > 0 {
		if err := frameWorkAppendLoopFilterChromaEdgeSegmentsUV(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, currentHorizontalU, currentHorizontalV, ssX, ssY); err != nil {
			return err
		}
	}
	return nil
}

func frameWorkAppendLoopFilterChromaEdgeSegmentsUV(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevelU uint8, currentLevelV uint8, ssX int, ssY int) error {
	if currentLevelU != 0 && currentLevelU == currentLevelV {
		return frameWorkAppendLoopFilterChromaEdgeSegmentsAndDuplicateUV(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentLevelU, ssX, ssY)
	}
	if currentLevelU != 0 && currentLevelV != 0 {
		return frameWorkAppendLoopFilterChromaEdgeSegmentsUnequalUV(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentLevelU, currentLevelV, ssX, ssY)
	}
	if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.PlaneU, edge, x4, y4, length4, tx, currentLevelU, ssX, ssY); err != nil {
		return err
	}
	return frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.PlaneV, edge, x4, y4, length4, tx, currentLevelV, ssX, ssY)
}

func frameWorkAppendLoopFilterChromaEdgeSegmentsUnequalUV(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevelU uint8, currentLevelV uint8, ssX int, ssY int) error {
	if length4 <= 0 {
		return nil
	}
	length4, err := frameWorkLoopFilterClampEdgeLengthInBounds(bounds, edge, x4, y4, length4)
	if err != nil {
		return err
	}
	if length4 <= 0 {
		return nil
	}
	currentWidth, err := frameWorkLoopFilterWidth(loopfilter.PlaneU, edge, tx)
	if err != nil {
		return err
	}
	beforeCandidates := plan.EdgeCandidates
	beforeStored := plan.StoredEdges
	if handled, err := frameWorkTryAppendLoopFilterFixedChromaEdge(levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.PlaneU, edge, x4, y4, length4, tx, currentWidth, currentLevelU, false, ssX, ssY); handled || err != nil {
		if err != nil {
			return err
		}
		candidates := plan.EdgeCandidates - beforeCandidates
		stored := plan.StoredEdges - beforeStored
		for i := 0; i < stored; i++ {
			edgeCopy := edges[beforeStored+i]
			edgeCopy.Plane = loopfilter.PlaneV
			edgeCopy.Level = currentLevelV
			frameWorkStoreLoopFilterEdge(plan, edges, edgeCopy)
		}
		if dropped := candidates - stored; dropped > 0 {
			plan.EdgeCandidates += dropped
			plan.PlaneEdgeCandidates[loopfilter.PlaneV] += dropped
			plan.DroppedEdges += dropped
		}
		return nil
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
		w, ok, err := frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, segX4, segY4, end-start, segWidth)
		if err != nil {
			return err
		}
		if !ok {
			return nil
		}
		edgeRecord := FrameWorkLoopFilterPostFilterEdge{
			Plane:      loopfilter.PlaneU,
			Edge:       edge,
			X4:         segX4,
			Y4:         segY4,
			Length4:    end - start,
			Level:      currentLevelU,
			Transform:  tx,
			Width:      w,
			BlockMICol: record.Block.MICol,
			BlockMIRow: record.Block.MIRow,
		}
		frameWorkStoreLoopFilterEdge(plan, edges, edgeRecord)
		edgeRecord.Plane = loopfilter.PlaneV
		edgeRecord.Level = currentLevelV
		frameWorkStoreLoopFilterEdge(plan, edges, edgeRecord)
		return nil
	}
	hadAny := false
	var previousCache frameWorkLoopFilterChromaPreviousCache
	for offset := range length4 {
		previousWidth, hasChroma, _, err := previousCache.lookup(levelCtx, filterMap, color, loopfilter.PlaneU, edge, x4, y4, offset, plan.MICols, plan.MIRows, false, ssX, ssY)
		if err != nil {
			return err
		}
		width := 0
		if hasChroma {
			width = min(previousWidth, currentWidth)
			hadAny = true
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

func frameWorkAppendLoopFilterChromaEdgeSegmentsAndDuplicateUV(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevel uint8, ssX int, ssY int) error {
	beforeCandidates := plan.EdgeCandidates
	beforeStored := plan.StoredEdges
	if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.PlaneU, edge, x4, y4, length4, tx, currentLevel, ssX, ssY); err != nil {
		return err
	}
	candidates := plan.EdgeCandidates - beforeCandidates
	stored := plan.StoredEdges - beforeStored
	for i := 0; i < stored; i++ {
		edgeCopy := edges[beforeStored+i]
		edgeCopy.Plane = loopfilter.PlaneV
		frameWorkStoreLoopFilterEdge(plan, edges, edgeCopy)
	}
	if dropped := candidates - stored; dropped > 0 {
		plan.EdgeCandidates += dropped
		plan.PlaneEdgeCandidates[loopfilter.PlaneV] += dropped
		plan.DroppedEdges += dropped
	}
	return nil
}

// frameWorkAppendLoopFilterChromaEdgeSegments mirrors
// frameWorkAppendLoopFilterLumaEdgeSegments for chroma. libaom's chroma
// deblock equally resolves filter_length per chroma cell from the local
// current/previous transform sizes (set_one_param_for_line_chroma); without
// per-cell splitting the chroma plane reproduced the same q32 10-bit
// divergence the luma fix already covered.
func frameWorkAppendLoopFilterChromaEdgeSegments(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevel uint8, ssX int, ssY int) error {
	if length4 <= 0 {
		return nil
	}
	length4, err := frameWorkLoopFilterClampEdgeLengthInBounds(bounds, edge, x4, y4, length4)
	if err != nil {
		return err
	}
	if length4 <= 0 {
		return nil
	}
	currentWidth, err := frameWorkLoopFilterWidth(plane, edge, tx)
	if err != nil {
		return err
	}
	segStart := 0
	segWidth := 0
	var segLevel uint8
	var segFromPrevious bool
	emit := func(start, end int) error {
		if segWidth == 0 || segLevel == 0 || end <= start {
			return nil
		}
		segX4 := x4
		segY4 := y4
		if edge == loopfilter.EdgeHorizontal {
			segX4 = x4 + start
		} else {
			segY4 = y4 + start
		}
		w, ok, err := frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, segX4, segY4, end-start, segWidth)
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
			Level:             segLevel,
			Transform:         tx,
			Width:             w,
			LevelFromPrevious: segFromPrevious,
			BlockMICol:        record.Block.MICol,
			BlockMIRow:        record.Block.MIRow,
		})
		return nil
	}
	// Like the luma path we want to emit one edge per maximal run of MI
	// cells that share the same filter width and level. Chroma adds a wrinkle:
	// at some offsets the previous-side luma record co-locates with the current
	// chroma block, so there is no chroma boundary there (the old behaviour
	// returned hasChroma=false for the entire TX edge in that case). libaom
	// processes the chroma edge per cell regardless, so when the per-cell
	// lookup returns "no chroma at this offset" we treat that as a no-op
	// gap and continue with the prevailing width from the surrounding cells.
	// The level is resolved per cell too: when the current block's level is
	// zero the edge filters with the previous block's level, which can vary
	// cell-by-cell along the same TX edge.
	hadAny := false
	lastWidth := 0
	var previousCache frameWorkLoopFilterChromaPreviousCache
	needPreviousLevel := currentLevel == 0
	if handled, err := frameWorkTryAppendLoopFilterFixedChromaEdge(levelCtx, filterMap, color, record, plan, edges, bounds, plane, edge, x4, y4, length4, tx, currentWidth, currentLevel, needPreviousLevel, ssX, ssY); handled || err != nil {
		return err
	}
	for offset := range length4 {
		previousWidth, hasChroma, previousLevel, err := previousCache.lookup(levelCtx, filterMap, color, plane, edge, x4, y4, offset, plan.MICols, plan.MIRows, needPreviousLevel, ssX, ssY)
		if err != nil {
			return err
		}
		var width int
		level := currentLevel
		fromPrevious := false
		if hasChroma {
			width = min(previousWidth, currentWidth)
			if level == 0 {
				level = previousLevel
				fromPrevious = previousLevel != 0
			}
			hadAny = true
			lastWidth = width
		} else {
			// Carry the prevailing width so the per-offset filter still runs
			// at this cell (matches libaom's per-MI-cell processing) when
			// surrounding cells share the same width. Cells that never see a
			// chroma neighbour stay no-ops once segWidth=0 falls through.
			width = lastWidth
			level = 0
		}
		if level == 0 {
			width = 0
		}
		if width != segWidth || level != segLevel || fromPrevious != segFromPrevious {
			if err := emit(segStart, offset); err != nil {
				return err
			}
			segStart = offset
			segWidth = width
			segLevel = level
			segFromPrevious = fromPrevious
		}
	}
	if !hadAny {
		return nil
	}
	return emit(segStart, length4)
}

func frameWorkTryAppendLoopFilterFixedChromaEdge(levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentWidth int, currentLevel uint8, needPreviousLevel bool, ssX int, ssY int) (bool, error) {
	previous, ok, err := frameWorkLoopFilterPreviousChromaRecordWithShifts(filterMap, edge, x4, y4, plan.MICols, plan.MIRows, ssX, ssY)
	if err != nil {
		return true, err
	}
	if !ok {
		return true, loopfilter.ErrInvalidFilter
	}
	if !frameWorkLoopFilterPreviousChromaRecordCoversRun(previous, edge, x4, y4, length4, ssX, ssY) {
		return false, nil
	}
	block, blockOK, err := frameWorkLoopFilterChromaBlockWithShifts(color, previous, ssX, ssY)
	if err != nil {
		return true, err
	}
	if !blockOK || !frameWorkLoopFilterPreviousChromaRunInBlock(previous, block, edge, x4, y4, length4, ssX, ssY) {
		return false, nil
	}
	previousWidth, err := frameWorkLoopFilterWidth(plane, edge, block.Size)
	if err != nil {
		return true, err
	}
	level := currentLevel
	fromPrevious := false
	if level == 0 && needPreviousLevel {
		previousLevel, err := frameWorkResolveLoopFilterLevel(levelCtx, previous, plane, edge)
		if err != nil {
			return true, err
		}
		level = previousLevel
		fromPrevious = previousLevel != 0
	}
	if level == 0 {
		return true, nil
	}
	width := min(previousWidth, currentWidth)
	if width == 0 {
		return true, nil
	}
	w, ok, err := frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, x4, y4, length4, width)
	if err != nil {
		return true, err
	}
	if !ok {
		return true, nil
	}
	frameWorkStoreLoopFilterEdge(plan, edges, FrameWorkLoopFilterPostFilterEdge{
		Plane:             plane,
		Edge:              edge,
		X4:                x4,
		Y4:                y4,
		Length4:           length4,
		Level:             level,
		Transform:         tx,
		Width:             w,
		LevelFromPrevious: fromPrevious,
		BlockMICol:        record.Block.MICol,
		BlockMIRow:        record.Block.MIRow,
	})
	return true, nil
}

func frameWorkLoopFilterPreviousChromaRecordCoversRun(record *threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge, x4 int, y4 int, length4 int, ssX int, ssY int) bool {
	if record == nil || length4 <= 0 {
		return false
	}
	block := record.Block
	switch edge {
	case loopfilter.EdgeVertical:
		prevCol := ((x4 - 1) << ssX) | ssX
		startRow := (y4 << ssY) | ssY
		endRow := ((y4 + length4 - 1) << ssY) | ssY
		return prevCol >= int(block.MICol) && prevCol < int(block.MIColEnd) &&
			startRow >= int(block.MIRow) && endRow < int(block.MIRowEnd)
	case loopfilter.EdgeHorizontal:
		startCol := (x4 << ssX) | ssX
		endCol := ((x4 + length4 - 1) << ssX) | ssX
		prevRow := ((y4 - 1) << ssY) | ssY
		return prevRow >= int(block.MIRow) && prevRow < int(block.MIRowEnd) &&
			startCol >= int(block.MICol) && endCol < int(block.MIColEnd)
	default:
		return false
	}
}

func frameWorkLoopFilterPreviousChromaRunInBlock(record *threading.FrameWorkLoopFilterBlockRecord, block tile.TransformBlock, edge loopfilter.Edge, x4 int, y4 int, length4 int, ssX int, ssY int) bool {
	startX4, startY4, err := frameWorkLoopFilterPreviousTarget4(edge, x4, y4)
	if err != nil {
		return false
	}
	endBoundaryX4, endBoundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, length4-1)
	endX4, endY4, err := frameWorkLoopFilterPreviousTarget4(edge, endBoundaryX4, endBoundaryY4)
	if err != nil {
		return false
	}
	startLocalX4 := (record.Block.X4 >> ssX) + startX4 - (int(record.Block.MICol) >> ssX)
	startLocalY4 := (record.Block.Y4 >> ssY) + startY4 - (int(record.Block.MIRow) >> ssY)
	endLocalX4 := (record.Block.X4 >> ssX) + endX4 - (int(record.Block.MICol) >> ssX)
	endLocalY4 := (record.Block.Y4 >> ssY) + endY4 - (int(record.Block.MIRow) >> ssY)
	return frameWorkLoopFilterTransformBlockContains(block, startLocalX4, startLocalY4) &&
		frameWorkLoopFilterTransformBlockContains(block, endLocalX4, endLocalY4)
}

type frameWorkLoopFilterChromaPreviousCache struct {
	valid      bool
	miCol      uint32
	miRow      uint32
	record     *threading.FrameWorkLoopFilterBlockRecord
	block      tile.TransformBlock
	blockOK    bool
	width      int
	level      uint8
	levelValid bool
}

func (c *frameWorkLoopFilterChromaPreviousCache) lookup(levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int, needLevel bool, ssX int, ssY int) (int, bool, uint8, error) {
	boundaryX4, boundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, offset)
	targetX4, targetY4, err := frameWorkLoopFilterPreviousTarget4(edge, boundaryX4, boundaryY4)
	if err != nil {
		return 0, false, 0, err
	}
	previous, ok, err := frameWorkLoopFilterPreviousChromaRecordWithShifts(filterMap, edge, boundaryX4, boundaryY4, cols, rows, ssX, ssY)
	if err != nil {
		return 0, false, 0, err
	}
	if !ok {
		return 0, false, 0, loopfilter.ErrInvalidFilter
	}
	if !c.valid || c.miCol != previous.Block.MICol || c.miRow != previous.Block.MIRow {
		block, blockOK, err := frameWorkLoopFilterChromaBlockWithShifts(color, previous, ssX, ssY)
		if err != nil {
			return 0, false, 0, err
		}
		width := 0
		if blockOK {
			width, err = frameWorkLoopFilterWidth(plane, edge, block.Size)
			if err != nil {
				return 0, false, 0, err
			}
		}
		*c = frameWorkLoopFilterChromaPreviousCache{
			valid:   true,
			miCol:   previous.Block.MICol,
			miRow:   previous.Block.MIRow,
			record:  previous,
			block:   block,
			blockOK: blockOK,
			width:   width,
		}
	}
	if !c.blockOK {
		return 0, false, 0, nil
	}
	localX4 := (c.record.Block.X4 >> ssX) + targetX4 - (int(c.record.Block.MICol) >> ssX)
	localY4 := (c.record.Block.Y4 >> ssY) + targetY4 - (int(c.record.Block.MIRow) >> ssY)
	if !frameWorkLoopFilterTransformBlockContains(c.block, localX4, localY4) {
		return 0, false, 0, nil
	}
	if needLevel && !c.levelValid {
		level, err := frameWorkResolveLoopFilterLevel(levelCtx, c.record, plane, edge)
		if err != nil {
			return 0, false, 0, err
		}
		c.level = level
		c.levelValid = true
	}
	return c.width, true, c.level, nil
}

// frameWorkLoopFilterPreviousChromaCellLevel resolves the previous-side chroma
// block's loop-filter level for one MI cell, used as libaom's fallback when the
// current chroma block's level is zero.
func frameWorkLoopFilterPreviousChromaCellLevel(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int) (uint8, error) {
	boundaryX4, boundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, offset)
	previous, ok, err := frameWorkLoopFilterPreviousChromaRecord(ctx, filterMap, edge, boundaryX4, boundaryY4, cols, rows)
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, nil
	}
	return frameWorkResolveLoopFilterLevel(frameWorkLoopFilterLevelContextFor(&ctx.Event), previous, plane, edge)
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
	previous, ok, err := frameWorkLoopFilterPreviousChromaRecord(ctx, filterMap, edge, boundaryX4, boundaryY4, cols, rows)
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

// frameWorkLoopFilterPreviousChromaRecord resolves the loop-filter record on
// the previous side of a chroma edge boundary. boundaryX4/boundaryY4 are
// expressed in chroma 4x4 frame coordinates and identify the leading edge of
// the current chroma cell.
//
// libaom (av1/common/av1_loopfilter.c set_one_param_for_line_chroma) maps each
// chroma cell to the bottom-right luma MI of the corresponding 2x2 luma
// cluster (mi_row|=scale_vert, mi_col|=scale_horz) and reads the previous
// neighbour via mi - mode_step where mode_step is one chroma cell expressed in
// luma MI units (1 << scale). The OR-anchor matters when chroma is subsampled
// and the previous-side luma cluster contains sub8x8 luma blocks whose
// records differ across the even/odd rows or columns: looking up the previous
// neighbour at the cluster's top-left luma MI returns a different record
// (and hence a different chroma TX size) than libaom's bottom-right anchor.
func frameWorkLoopFilterPreviousChromaRecord(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, boundaryX4 int, boundaryY4 int, cols int, rows int) (*threading.FrameWorkLoopFilterBlockRecord, bool, error) {
	color := ctx.Event.SequenceHeader.ColorConfig
	ssX := frameWorkLoopFilterSubsamplingShift(color.SubsamplingX)
	ssY := frameWorkLoopFilterSubsamplingShift(color.SubsamplingY)
	return frameWorkLoopFilterPreviousChromaRecordWithShifts(filterMap, edge, boundaryX4, boundaryY4, cols, rows, ssX, ssY)
}

func frameWorkLoopFilterPreviousChromaRecordWithShifts(filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, boundaryX4 int, boundaryY4 int, cols int, rows int, ssX int, ssY int) (*threading.FrameWorkLoopFilterBlockRecord, bool, error) {
	var prevCol, prevRow int
	switch edge {
	case loopfilter.EdgeVertical:
		prevCol = ((boundaryX4 - 1) << ssX) | ssX
		prevRow = (boundaryY4 << ssY) | ssY
	case loopfilter.EdgeHorizontal:
		prevCol = (boundaryX4 << ssX) | ssX
		prevRow = ((boundaryY4 - 1) << ssY) | ssY
	default:
		return nil, false, loopfilter.ErrInvalidFilter
	}
	if prevCol < 0 || prevRow < 0 {
		return nil, false, nil
	}
	if prevCol >= cols || prevRow >= rows || prevRow >= filterMap.Rows || prevCol >= filterMap.Stride {
		return nil, false, threading.ErrInvalidBatch
	}
	record := &filterMap.Records[prevRow*filterMap.Stride+prevCol]
	if !record.Valid {
		return nil, false, threading.ErrInvalidBatch
	}
	return record, true, nil
}

// frameWorkLoopFilterClampEdgeLength clips a deblocking edge's run length to the
// visible plane extent. libaom's deblocker (av1_filter_block_plane_vert/_horz
// and the _opt variants) bounds the per-line filter loop by
// y_range/x_range = AOMMIN(plane_mi_dim - mi_coord, MAX_MIB_SIZE) where
// plane_mi_dim = CEIL_POWER_OF_TWO(dst.{height,width}, MI_SIZE_LOG2) is derived
// from the CROPPED frame dimension, not the (8-aligned) MI grid. When a frame's
// height is not a multiple of 8 the MI grid is one MI row taller than the
// cropped plane (e.g. 180px -> 45 cropped MI rows but a 46-row grid), so a
// transform block in the bottom partial superblock yields a vertical edge whose
// TX-derived length spills one MI row past the frame. Without this clamp that
// vertical edge fails frameWorkLoopFilterEdgeFits (y > frameHeight-length) and
// is dropped entirely, leaving the bottom superblock undeblocked — the
// 8-bit-monochrome frame-9 divergence. We clamp the run direction (rows for a
// vertical edge, cols for a horizontal edge) to the cropped plane MI extent so
// the edge spans only the visible samples, matching libaom's y_range/x_range.
func frameWorkLoopFilterClampEdgeLength(ctx FrameWorkPostFilterContext, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int) (int, error) {
	bounds, err := frameWorkLoopFilterPlaneBounds(ctx, plane)
	if err != nil {
		return 0, err
	}
	return frameWorkLoopFilterClampEdgeLengthInBounds(bounds, edge, x4, y4, length4)
}

func frameWorkLoopFilterClampEdgeLengthInBounds(bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int) (int, error) {
	var limit int
	switch edge {
	case loopfilter.EdgeVertical:
		limit = (bounds.posHeight+3)>>2 - y4
	case loopfilter.EdgeHorizontal:
		limit = (bounds.posWidth+3)>>2 - x4
	default:
		return 0, loopfilter.ErrInvalidFilter
	}
	if limit < 0 {
		limit = 0
	}
	if length4 > limit {
		length4 = limit
	}
	return length4, nil
}

func frameWorkLoopFilterScheduledWidth(ctx FrameWorkPostFilterContext, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, width int) (int, bool, error) {
	bounds, err := frameWorkLoopFilterPlaneBounds(ctx, plane)
	if err != nil {
		return 0, false, err
	}
	return frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, x4, y4, length4, width)
}

type frameWorkLoopFilterBounds struct {
	posWidth  int
	posHeight int
	bufWidth  int
	bufHeight int
}

func frameWorkLoopFilterPlaneBounds(ctx FrameWorkPostFilterContext, plane loopfilter.Plane) (frameWorkLoopFilterBounds, error) {
	posWidth, posHeight, err := frameWorkLoopFilterPlaneSize(ctx, plane)
	if err != nil {
		return frameWorkLoopFilterBounds{}, err
	}
	bufWidth, bufHeight, err := frameWorkLoopFilterBufferSize(ctx, plane)
	if err != nil {
		return frameWorkLoopFilterBounds{}, err
	}
	return frameWorkLoopFilterBounds{
		posWidth:  posWidth,
		posHeight: posHeight,
		bufWidth:  bufWidth,
		bufHeight: bufHeight,
	}, nil
}

func frameWorkLoopFilterScheduledWidthInBounds(bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, width int) (int, bool, error) {
	x := x4 * 4
	y := y4 * 4
	length := length4 * 4
	// Edge position bound (MI(4)-aligned crop): libaom emits no internal deblock
	// edge inside the single-transform partial-superblock padding, i.e. past the
	// crop dimension rounded up to MI_SIZE. Suppress any edge whose perpendicular
	// position lands in that padding (e.g. a phantom vertical edge at x=36 for a
	// 34x34 frame, or a horizontal edge at y in the same gap), which goav1's
	// TX-edge enumeration can produce but libaom never filters.
	switch edge {
	case loopfilter.EdgeVertical:
		if x >= bounds.posWidth {
			return 0, false, nil
		}
	case loopfilter.EdgeHorizontal:
		if y >= bounds.posHeight {
			return 0, false, nil
		}
	default:
		return 0, false, loopfilter.ErrInvalidFilter
	}
	// Filter-tap fit bound (MI(8)-aligned = the allocated buffer extent): libaom
	// applies the TX-derived filter width without clamping the taps to the crop;
	// the taps read/write the partial-superblock padding the reconstruction stage
	// populated. The byte buffer spans the 8-aligned extent (frame.RequiredSize),
	// so a wide filter (e.g. width-14) on a near-boundary edge such as the
	// 226x226 vertical edge at x=224 fits and must not be downgraded.
	for width != 0 {
		if frameWorkLoopFilterEdgeFits(width, edge, x, y, length, bounds.bufWidth, bounds.bufHeight) {
			return width, true, nil
		}
		width = frameWorkLoopFilterDowngradeWidth(width)
	}
	return 0, false, nil
}

// frameWorkLoopFilterBufferSize returns the per-plane allocated buffer extent in
// pixels (frame dimensions rounded up to a whole 8-luma-pixel MI grid, derived
// per plane via the subsampling, matching frame.RequiredSize). The deblock
// filter taps may read/write up to this extent (the partial-superblock padding),
// so frameWorkLoopFilterEdgeFits must not downgrade filters whose taps stay
// within it even when they pass the cropped surface.
func frameWorkLoopFilterBufferSize(ctx FrameWorkPostFilterContext, plane loopfilter.Plane) (int, int, error) {
	width := int(ctx.Event.FrameSize.CodedWidth)
	height := int(ctx.Event.FrameSize.Height)
	if width <= 0 || height <= 0 {
		return 0, 0, threading.ErrInvalidBatch
	}
	alignedW := (width + 7) &^ 7
	alignedH := (height + 7) &^ 7
	switch plane {
	case loopfilter.PlaneY:
		return alignedW, alignedH, nil
	case loopfilter.PlaneU, loopfilter.PlaneV:
		if ctx.Event.SequenceHeader.ColorConfig.MonoChrome {
			return 0, 0, threading.ErrInvalidBatch
		}
		color := ctx.Event.SequenceHeader.ColorConfig
		if color.SubsamplingX {
			alignedW >>= 1
		}
		if color.SubsamplingY {
			alignedH >>= 1
		}
		return alignedW, alignedH, nil
	default:
		return 0, 0, threading.ErrInvalidBatch
	}
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

func frameWorkLoopFilterLumaTransformAt(ctx FrameWorkPostFilterContext, record *threading.FrameWorkLoopFilterBlockRecord, frameX4 int, frameY4 int) (tile.TransformSize, bool, error) {
	req, err := frameWorkLoopFilterTransformTreeRequest(ctx.Event.SequenceHeader.ColorConfig, record)
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
	if record.SkipTransform || !record.TransformTree.Variable {
		if !record.TransformTree.Y.Valid() {
			return 0, false, threading.ErrInvalidBatch
		}
		return record.TransformTree.Y, true, nil
	}
	found := tile.TransformSize(0)
	ok := false
	tx, ok, err := frameWorkLoopFilterFindLumaTransform(record.TransformTree, req, localX4, localY4)
	if err != nil {
		return 0, false, err
	}
	found = tx.Size
	return found, ok, nil
}

func frameWorkLoopFilterFindLumaTransform(tree tile.TransformTreeResult, req tile.TransformTreeRequest, localX4 int, localY4 int) (tile.TransformBlock, bool, error) {
	var found tile.TransformBlock
	ok := false
	err := tree.ForEachLumaTXB(req, func(tx tile.TransformBlock) error {
		if frameWorkLoopFilterTransformBlockContains(tx, localX4, localY4) {
			found = tx
			ok = true
			return errFrameWorkLoopFilterTransformFound
		}
		return nil
	})
	if err == errFrameWorkLoopFilterTransformFound {
		return found, true, nil
	}
	if err != nil {
		return tile.TransformBlock{}, false, err
	}
	return found, ok, nil
}

func frameWorkLoopFilterChromaTransformAt(ctx FrameWorkPostFilterContext, record *threading.FrameWorkLoopFilterBlockRecord, frameX4 int, frameY4 int) (tile.TransformSize, bool, error) {
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
	return block.Size, true, nil
}

func frameWorkLoopFilterTransformBlockContains(tx tile.TransformBlock, x4 int, y4 int) bool {
	return x4 >= tx.X4 && y4 >= tx.Y4 &&
		x4 < tx.X4+int(tx.VisibleW4) &&
		y4 < tx.Y4+int(tx.VisibleH4)
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

// frameWorkLoopFilterPlaneSize returns the per-plane deblock extent in pixels.
//
// libaom's decoder loop filter iterates the MI grid up to plane_mi_{cols,rows}
// (= ROUND_POWER_OF_TWO(mi_cols/mi_rows, subsampling); mi_cols = ROUND_UP(width,
// 8) / MI_SIZE), so the bottom/right partial-superblock padding the
// reconstruction stage populated participates in filtering. However, the
// rightmost/bottommost transform block of the partial superblock extends across
// the whole [crop .. MI-aligned] padding as a single transform, so libaom emits
// no internal deblock edge inside that padding past the crop dimension rounded
// up to a whole MI_SIZE (4-pixel) cell: every real edge it filters lands at or
// before CEIL(crop, MI_SIZE). The visible right/bottom edges whose taps reach
// just into that 4-pixel-aligned padding (e.g. the 34x34 vertical edge at x=32,
// the 226x226 right edges) must still be applied with the full filter width.
//
// We therefore size the deblock extent at the crop dimension rounded up to
// MI_SIZE: large enough that frameWorkLoopFilterEdgeFits / ScheduledWidth no
// longer downgrade those near-boundary edges (the padding the taps read is real
// reconstruction output, not the cropped surface), yet not so large that the
// per-cell edge enumeration would emit a spurious edge in the single-transform
// padding past the MI cell, which libaom never filters. The byte buffer is
// allocated at the MI(8)-aligned size, so the taps always have backing samples.
func frameWorkLoopFilterPlaneSize(ctx FrameWorkPostFilterContext, plane loopfilter.Plane) (int, int, error) {
	width := int(ctx.Event.FrameSize.CodedWidth)
	height := int(ctx.Event.FrameSize.Height)
	if width <= 0 || height <= 0 {
		return 0, 0, threading.ErrInvalidBatch
	}
	switch plane {
	case loopfilter.PlaneY:
		return frameWorkLoopFilterMIAlignedLength(width), frameWorkLoopFilterMIAlignedLength(height), nil
	case loopfilter.PlaneU, loopfilter.PlaneV:
		if ctx.Event.SequenceHeader.ColorConfig.MonoChrome {
			return 0, 0, threading.ErrInvalidBatch
		}
		color := ctx.Event.SequenceHeader.ColorConfig
		return frameWorkLoopFilterMIAlignedLength(frameWorkLoopFilterSubsampledLength(width, color.SubsamplingX)),
			frameWorkLoopFilterMIAlignedLength(frameWorkLoopFilterSubsampledLength(height, color.SubsamplingY)), nil
	default:
		return 0, 0, threading.ErrInvalidBatch
	}
}

// frameWorkLoopFilterMIAlignedLength rounds a plane crop dimension up to a whole
// MI_SIZE (4-pixel) cell.
func frameWorkLoopFilterMIAlignedLength(length int) int {
	return (length + 3) &^ 3
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

func frameWorkLoopFilterCountChromaTXBsWithShifts(color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, ssX int, ssY int) (int, error) {
	block, ok, err := frameWorkLoopFilterChromaBlockWithShifts(color, record, ssX, ssY)
	if err != nil || !ok {
		return 0, err
	}
	uvDims, ok := record.TransformTree.UV.Dimensions()
	if !ok {
		return 0, threading.ErrInvalidBatch
	}
	w := int(block.VisibleW4)
	h := int(block.VisibleH4)
	txW := int(uvDims.W4)
	txH := int(uvDims.H4)
	if w <= 0 || h <= 0 || txW <= 0 || txH <= 0 {
		return 0, threading.ErrInvalidBatch
	}
	return ((w + txW - 1) / txW) * ((h + txH - 1) / txH), nil
}

func frameWorkLoopFilterChromaBlock(ctx FrameWorkPostFilterContext, record *threading.FrameWorkLoopFilterBlockRecord) (tile.TransformBlock, bool, error) {
	color := ctx.Event.SequenceHeader.ColorConfig
	return frameWorkLoopFilterChromaBlockWithShifts(color, record, frameWorkLoopFilterSubsamplingShift(color.SubsamplingX), frameWorkLoopFilterSubsamplingShift(color.SubsamplingY))
}

func frameWorkLoopFilterChromaBlockWithShifts(color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, ssX int, ssY int) (tile.TransformBlock, bool, error) {
	req, err := frameWorkLoopFilterTransformTreeRequest(color, record)
	if err != nil {
		return tile.TransformBlock{}, false, err
	}
	if color.MonoChrome || !tile.HasChromaBlock(req, color) {
		return tile.TransformBlock{}, false, nil
	}
	if !record.TransformTree.HasUV || !record.TransformTree.UV.Valid() {
		return tile.TransformBlock{}, false, threading.ErrInvalidBatch
	}
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

func frameWorkLoopFilterChromaFrame4(ctx FrameWorkPostFilterContext, record *threading.FrameWorkLoopFilterBlockRecord, x4 int, y4 int) (int, int) {
	color := ctx.Event.SequenceHeader.ColorConfig
	ssX := frameWorkLoopFilterSubsamplingShift(color.SubsamplingX)
	ssY := frameWorkLoopFilterSubsamplingShift(color.SubsamplingY)
	return frameWorkLoopFilterChromaFrame4WithShifts(record, x4, y4, ssX, ssY)
}

func frameWorkLoopFilterChromaFrame4WithShifts(record *threading.FrameWorkLoopFilterBlockRecord, x4 int, y4 int, ssX int, ssY int) (int, int) {
	return (int(record.Block.MICol) >> ssX) + x4 - (record.Block.X4 >> ssX),
		(int(record.Block.MIRow) >> ssY) + y4 - (record.Block.Y4 >> ssY)
}

func frameWorkLoopFilterSubsamplingShift(subsampled bool) int {
	if subsampled {
		return 1
	}
	return 0
}

func frameWorkLoopFilterTransformTreeRequest(color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord) (tile.TransformTreeRequest, error) {
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
		Color:         color,
		Inter:         !record.Intra,
		SkipTransform: record.SkipTransform,
	}, nil
}

func frameWorkValidateLoopFilterRecord(record *threading.FrameWorkLoopFilterBlockRecord, col int, row int, cols int, rows int) error {
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

func frameWorkLoopFilterDeltaState(record *threading.FrameWorkLoopFilterBlockRecord) loopfilter.DeltaState {
	state := loopfilter.DeltaState{FromBase: record.DeltaLFFromBase}
	for i := range state.Multi {
		state.Multi[i] = record.DeltaLF[i]
	}
	return state
}

func frameWorkResolveLoopFilterLevel(levelCtx frameWorkLoopFilterLevelContext, record *threading.FrameWorkLoopFilterBlockRecord, plane loopfilter.Plane, edge loopfilter.Edge) (uint8, error) {
	if record == nil || levelCtx.loopFilter == nil || levelCtx.segmentation == nil || levelCtx.delta == nil {
		return 0, threading.ErrInvalidBatch
	}
	if plane > loopfilter.PlaneV || edge > loopfilter.EdgeHorizontal ||
		record.RefFrame < 0 || record.RefFrame >= parser.RefFrames ||
		(record.Mode != loopfilter.ModeDeltaClassZero && record.Mode != loopfilter.ModeDeltaClassMotion) {
		return 0, loopfilter.ErrInvalidFilter
	}
	base := levelCtx.base[plane][edge]
	if levelCtx.lumaZero || (plane != loopfilter.PlaneY && base == 0) {
		return 0, nil
	}
	deltaLF, err := frameWorkLoopFilterSelectedDelta(levelCtx.delta, record, plane, edge)
	if err != nil {
		return 0, err
	}
	segDelta, err := frameWorkLoopFilterSegmentDelta(levelCtx.segmentation, int(record.SegmentID), plane, edge)
	if err != nil {
		return 0, err
	}
	level := int(loopfilter.ClampLevel(int(base) + int(deltaLF)))
	level = int(loopfilter.ClampLevel(level + int(segDelta)))
	if levelCtx.loopFilter.ModeRefDeltaEnabled {
		scale := 1 << (level >> 5)
		delta := int(levelCtx.loopFilter.Deltas.Ref[record.RefFrame])
		if record.RefFrame > 0 {
			delta += int(levelCtx.loopFilter.Deltas.Mode[record.Mode])
		}
		level = int(loopfilter.ClampLevel(level + delta*scale))
	}
	return uint8(level), nil
}

func frameWorkLoopFilterSelectedDelta(delta *parser.DeltaParams, record *threading.FrameWorkLoopFilterBlockRecord, plane loopfilter.Plane, edge loopfilter.Edge) (int8, error) {
	if !delta.DeltaLFPresent {
		return 0, nil
	}
	if !delta.DeltaLFMulti {
		return record.DeltaLFFromBase, nil
	}
	switch plane {
	case loopfilter.PlaneY:
		if edge == loopfilter.EdgeVertical {
			return record.DeltaLF[0], nil
		}
		if edge == loopfilter.EdgeHorizontal {
			return record.DeltaLF[1], nil
		}
	case loopfilter.PlaneU:
		if edge <= loopfilter.EdgeHorizontal {
			return record.DeltaLF[2], nil
		}
	case loopfilter.PlaneV:
		if edge <= loopfilter.EdgeHorizontal {
			return record.DeltaLF[3], nil
		}
	}
	return 0, loopfilter.ErrInvalidFilter
}

func frameWorkLoopFilterSegmentDelta(seg *parser.SegmentationParams, segmentID int, plane loopfilter.Plane, edge loopfilter.Edge) (int16, error) {
	if segmentID < 0 || segmentID >= parser.MaxSegments || edge > loopfilter.EdgeHorizontal {
		return 0, loopfilter.ErrInvalidFilter
	}
	if !seg.Enabled {
		return 0, nil
	}
	data := seg.Data.Segments[segmentID]
	switch plane {
	case loopfilter.PlaneY:
		if edge == loopfilter.EdgeVertical {
			return data.DeltaLFYV, nil
		}
		return data.DeltaLFYH, nil
	case loopfilter.PlaneU:
		return data.DeltaLFU, nil
	case loopfilter.PlaneV:
		return data.DeltaLFV, nil
	default:
		return 0, loopfilter.ErrInvalidFilter
	}
}

func frameWorkResolveLoopFilterLevels(levelCtx frameWorkLoopFilterLevelContext, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan) error {
	levels, err := frameWorkResolveLoopFilterRecordLevels(levelCtx, record)
	if err != nil {
		return err
	}
	frameWorkAccumulateLoopFilterLevelStats(levelCtx, levels, plan)
	return nil
}

func frameWorkResolveLoopFilterRecordLevels(levelCtx frameWorkLoopFilterLevelContext, record *threading.FrameWorkLoopFilterBlockRecord) ([3][2]uint8, error) {
	var levels [3][2]uint8
	planeCount := 3
	if levelCtx.monoChrome {
		planeCount = 1
	}
	for plane := 0; plane < planeCount; plane++ {
		for edge := range 2 {
			level, err := frameWorkResolveLoopFilterLevel(levelCtx, record, loopfilter.Plane(plane), loopfilter.Edge(edge))
			if err != nil {
				return [3][2]uint8{}, err
			}
			levels[plane][edge] = level
		}
	}
	return levels, nil
}

func frameWorkAccumulateLoopFilterLevelStats(levelCtx frameWorkLoopFilterLevelContext, levels [3][2]uint8, plan *FrameWorkLoopFilterPostFilterPlan) {
	planeCount := 3
	if levelCtx.monoChrome {
		planeCount = 1
	}
	for plane := 0; plane < planeCount; plane++ {
		for edge := range 2 {
			level := levels[plane][edge]
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
}
