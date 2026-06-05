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

const frameWorkLoopFilterApplyScheduleCap = 4096

// FrameWorkLoopFilterMap is the decoded frame-level loop-filter metadata map.
type FrameWorkLoopFilterMap = threading.FrameWorkLoopFilterMap

// FrameWorkLoopFilterPostFilterRequest carries decoded loop-filter side data
// and optional caller-owned edge storage for postfilter planning.
type FrameWorkLoopFilterPostFilterRequest struct {
	Map             FrameWorkLoopFilterMap
	Edges           []FrameWorkLoopFilterPostFilterEdge
	Schedule        []uint32
	TrustedCoverage bool
}

// FrameWorkLoopFilterPostFilterEdge describes one deblocking edge candidate in
// plane-local 4x4 coordinates. Final frame-order scheduling remains separate
// from this block-local candidate collection.
type FrameWorkLoopFilterPostFilterEdge struct {
	// AV1 frame dimensions are capped at 65536 samples, so 4x4-unit frame
	// coordinates and run lengths fit comfortably in 16 bits.
	X4         uint16
	Y4         uint16
	Length4    uint16
	BlockMICol uint16
	BlockMIRow uint16

	Plane     loopfilter.Plane
	Edge      loopfilter.Edge
	Level     uint8
	Width     uint8
	Transform tile.TransformSize
	// LevelFromPrevious reports that AV1's fallback to the block on the other
	// side of the boundary supplied Level because the current side resolved to 0.
	LevelFromPrevious bool
}

// FrameWorkLoopFilterPostFilterScratchSize reports caller-owned scratch needed
// for loop-filter planning and eventual application.
type FrameWorkLoopFilterPostFilterScratchSize struct {
	Edges    int
	Schedule int
}

// BindEdges validates and slices caller-owned edge storage for loop-filter
// planning/application.
func (s FrameWorkLoopFilterPostFilterScratchSize) BindEdges(edges []FrameWorkLoopFilterPostFilterEdge) ([]FrameWorkLoopFilterPostFilterEdge, error) {
	if s.Edges < 0 || len(edges) < s.Edges {
		return nil, frame.ErrShortBuffer
	}
	return edges[:s.Edges], nil
}

// BindSchedule validates and slices caller-owned schedule storage for
// loop-filter application.
func (s FrameWorkLoopFilterPostFilterScratchSize) BindSchedule(schedule []uint32) ([]uint32, error) {
	if s.Schedule < 0 || len(schedule) < s.Schedule {
		return nil, frame.ErrShortBuffer
	}
	return schedule[:s.Schedule], nil
}

// Max returns the per-field maximum loop-filter scratch size.
func (s FrameWorkLoopFilterPostFilterScratchSize) Max(other FrameWorkLoopFilterPostFilterScratchSize) FrameWorkLoopFilterPostFilterScratchSize {
	return FrameWorkLoopFilterPostFilterScratchSize{
		Edges:    maxInt(s.Edges, other.Edges),
		Schedule: maxInt(s.Schedule, other.Schedule),
	}
}

// FrameWorkLoopFilterPostFilterLevelStats summarizes resolved levels for one
// plane/edge class.
type FrameWorkLoopFilterPostFilterLevelStats struct {
	Blocks   int32
	NonZero  int32
	MaxLevel uint8
}

// FrameWorkLoopFilterPostFilterPlan summarizes loop-filter metadata needed by
// the eventual frame-edge scheduler.
type FrameWorkLoopFilterPostFilterPlan struct {
	StoredEdges uint32

	Levels [3][2]FrameWorkLoopFilterPostFilterLevelStats

	Cells   int32
	Blocks  int32
	Missing int32

	TransformReadyBlocks int32
	SkipTransformBlocks  int32
	LumaTXBs             int32
	ChromaTXBs           int32

	EdgeCandidates      uint32
	PlaneEdgeCandidates [3]uint32
	PreviousLevelEdges  uint32
	DroppedEdges        uint32

	MICols uint16
	MIRows uint16

	Active bool
}

type frameWorkLoopFilterPlanningContext struct {
	bounds [3]frameWorkLoopFilterBounds
	color  parser.ColorConfig
	ssX    uint8
	ssY    uint8
}

type frameWorkLoopFilterLevelContext struct {
	loopFilter   *parser.LoopFilterParams
	segmentation *parser.SegmentationParams
	delta        *parser.DeltaParams
	base         [3][2]uint8
	lumaZero     bool
	monoChrome   bool
}

var frameWorkLoopFilterWidthByTX = [...][2][2]uint8{
	tile.TransformSize4x4:   {{4, 4}, {4, 4}},
	tile.TransformSize8x8:   {{8, 8}, {6, 6}},
	tile.TransformSize16x16: {{14, 14}, {6, 6}},
	tile.TransformSize32x32: {{14, 14}, {6, 6}},
	tile.TransformSize64x64: {{14, 14}, {6, 6}},
	tile.TransformSize4x8:   {{4, 8}, {4, 6}},
	tile.TransformSize8x4:   {{8, 4}, {6, 4}},
	tile.TransformSize8x16:  {{8, 14}, {6, 6}},
	tile.TransformSize16x8:  {{14, 8}, {6, 6}},
	tile.TransformSize16x32: {{14, 14}, {6, 6}},
	tile.TransformSize32x16: {{14, 14}, {6, 6}},
	tile.TransformSize32x64: {{14, 14}, {6, 6}},
	tile.TransformSize64x32: {{14, 14}, {6, 6}},
	tile.TransformSize4x16:  {{4, 14}, {4, 6}},
	tile.TransformSize16x4:  {{14, 4}, {6, 4}},
	tile.TransformSize8x32:  {{8, 14}, {6, 6}},
	tile.TransformSize32x8:  {{14, 8}, {6, 6}},
	tile.TransformSize16x64: {{14, 14}, {6, 6}},
	tile.TransformSize64x16: {{14, 14}, {6, 6}},
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
	bounds, err := frameWorkLoopFilterPlanningBoundsFor(ctx, color)
	if err != nil {
		return frameWorkLoopFilterPlanningContext{}, err
	}
	planning.bounds = bounds
	return planning, nil
}

func frameWorkLoopFilterPlanningBoundsFor(ctx FrameWorkPostFilterContext, color parser.ColorConfig) ([3]frameWorkLoopFilterBounds, error) {
	width, ok := frameWorkLoopFilterFrameDimension32(ctx.Event.FrameSize.CodedWidth)
	if !ok {
		return [3]frameWorkLoopFilterBounds{}, threading.ErrInvalidBatch
	}
	height, ok := frameWorkLoopFilterFrameDimension32(ctx.Event.FrameSize.Height)
	if !ok {
		return [3]frameWorkLoopFilterBounds{}, threading.ErrInvalidBatch
	}
	alignedW, ok := frameWorkLoopFilterAligned8Dimension(ctx.Event.FrameSize.CodedWidth)
	if !ok {
		return [3]frameWorkLoopFilterBounds{}, threading.ErrInvalidBatch
	}
	alignedH, ok := frameWorkLoopFilterAligned8Dimension(ctx.Event.FrameSize.Height)
	if !ok {
		return [3]frameWorkLoopFilterBounds{}, threading.ErrInvalidBatch
	}
	var bounds [3]frameWorkLoopFilterBounds
	bounds[loopfilter.PlaneY] = frameWorkLoopFilterBounds{
		posWidth:  frameWorkLoopFilterMIAlignedLength(width),
		posHeight: frameWorkLoopFilterMIAlignedLength(height),
		bufWidth:  alignedW,
		bufHeight: alignedH,
	}
	if color.MonoChrome {
		return bounds, nil
	}
	chromaPosW := frameWorkLoopFilterMIAlignedLength(frameWorkLoopFilterSubsampledLength(width, color.SubsamplingX))
	chromaPosH := frameWorkLoopFilterMIAlignedLength(frameWorkLoopFilterSubsampledLength(height, color.SubsamplingY))
	chromaBufW := alignedW
	chromaBufH := alignedH
	if color.SubsamplingX {
		chromaBufW >>= 1
	}
	if color.SubsamplingY {
		chromaBufH >>= 1
	}
	chromaBounds := frameWorkLoopFilterBounds{
		posWidth:  chromaPosW,
		posHeight: chromaPosH,
		bufWidth:  chromaBufW,
		bufHeight: chromaBufH,
	}
	bounds[loopfilter.PlaneU] = chromaBounds
	bounds[loopfilter.PlaneV] = chromaBounds
	return bounds, nil
}

// FrameWorkLoopFilterPostFilterApplyResult summarizes application of stored
// luma edge candidates through the pure-Go loop-filter kernels.
type FrameWorkLoopFilterPostFilterApplyResult struct {
	Plan FrameWorkLoopFilterPostFilterPlan

	Active bool

	Edges   uint32
	Applied uint32

	PlaneEdges    [3]uint32
	PlaneApplied  [3]uint32
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
	edgeCandidates, ok := frameWorkLoopFilterCounterLen(plan.EdgeCandidates)
	if !ok {
		return FrameWorkLoopFilterPostFilterScratchSize{}, frame.ErrInvalidFormat
	}
	return FrameWorkLoopFilterPostFilterScratchSize{
		Edges:    edgeCandidates,
		Schedule: edgeCandidates,
	}, nil
}

// LoopFilterPostFilterScratchUpperBound reports a sufficient edge buffer size
// without walking decoded loop-filter side data. ApplyLoopFilterEdges still
// computes the exact edge count before mutating output and returns short-buffer
// errors if a caller supplies less than the exact count.
func (ctx FrameWorkPostFilterContext) LoopFilterPostFilterScratchUpperBound() (FrameWorkLoopFilterPostFilterScratchSize, error) {
	if !ctx.RemainingPostFilters().Has(FrameWorkPostFilterLoopFilter) {
		return FrameWorkLoopFilterPostFilterScratchSize{}, nil
	}
	cols, rows, err := frameWorkLoopFilterMapGrid(ctx.Event.FrameSize)
	if err != nil {
		return FrameWorkLoopFilterPostFilterScratchSize{}, err
	}
	maxInt := int(^uint(0) >> 1)
	if rows != 0 && cols > maxInt/rows {
		return FrameWorkLoopFilterPostFilterScratchSize{}, frame.ErrInvalidFormat
	}
	cells := cols * rows
	if cells > maxInt/8 {
		return FrameWorkLoopFilterPostFilterScratchSize{}, frame.ErrInvalidFormat
	}
	candidates := cells * 8
	return FrameWorkLoopFilterPostFilterScratchSize{Edges: candidates, Schedule: candidates}, nil
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
		MICols: uint16(cols),
		MIRows: uint16(rows),
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
	if req.TrustedCoverage {
		return ctx.loopFilterPostFilterPlanTrusted(filterMap, planning, levelCtx, plan, req.Edges)
	}
	stride := int(filterMap.Stride)
	for row := 0; row < rows; row++ {
		base := row * stride
		for col := 0; col < cols; col++ {
			record := &filterMap.Records[base+col]
			if !record.Valid {
				plan.Missing++
				continue
			}
			if err := frameWorkValidateLoopFilterRecord(record, col, row, cols, rows); err != nil {
				return plan, err
			}
			plan.Cells++
			if int(record.Block.MICol) != col || int(record.Block.MIRow) != row {
				continue
			}
			plan.Blocks++
			levels, err := frameWorkResolveLoopFilterRecordLevels(levelCtx, record)
			if err != nil {
				return plan, err
			}
			frameWorkAccumulateLoopFilterLevelStats(levelCtx, levels, &plan)
			plan.TransformReadyBlocks++
			if record.SkipTransform {
				plan.SkipTransformBlocks++
			}
			if err := frameWorkAppendLoopFilterLumaEdges(ctx, levelCtx, filterMap, record, &plan, req.Edges, planning.bounds[loopfilter.PlaneY], levels); err != nil {
				return plan, err
			}
			if err := frameWorkAppendLoopFilterChromaEdges(ctx, levelCtx, filterMap, record, &plan, req.Edges, planning, levels); err != nil {
				return plan, err
			}
		}
	}
	if plan.Missing != 0 {
		return plan, threading.ErrInvalidBatch
	}
	return plan, nil
}

func (ctx FrameWorkPostFilterContext) loopFilterPostFilterPlanTrusted(filterMap FrameWorkLoopFilterMap, planning frameWorkLoopFilterPlanningContext, levelCtx frameWorkLoopFilterLevelContext, plan FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge) (FrameWorkLoopFilterPostFilterPlan, error) {
	cols := int(plan.MICols)
	rows := int(plan.MIRows)
	stride := int(filterMap.Stride)
	for row := 0; row < rows; row++ {
		base := row * stride
		for col := 0; col < cols; {
			record := &filterMap.Records[base+col]
			if !record.Valid {
				plan.Missing++
				col++
				continue
			}
			block := record.Block
			miCol := int(block.MICol)
			miRow := int(block.MIRow)
			miColEnd := int(block.MIColEnd)
			miRowEnd := int(block.MIRowEnd)
			if miColEnd <= miCol || miRowEnd <= miRow ||
				miColEnd > cols || miRowEnd > rows ||
				col < miCol || col >= miColEnd || row < miRow || row >= miRowEnd {
				return plan, threading.ErrInvalidBatch
			}
			nextCol := miColEnd
			if nextCol <= col {
				return plan, threading.ErrInvalidBatch
			}
			if miCol != col || miRow != row {
				col = nextCol
				continue
			}
			plan.Cells += int32((miColEnd - miCol) * (miRowEnd - miRow))
			plan.Blocks++
			levels, err := frameWorkResolveLoopFilterRecordLevels(levelCtx, record)
			if err != nil {
				return plan, err
			}
			frameWorkAccumulateLoopFilterLevelStats(levelCtx, levels, &plan)
			plan.TransformReadyBlocks++
			if record.SkipTransform {
				plan.SkipTransformBlocks++
			}
			if err := frameWorkAppendLoopFilterLumaEdges(ctx, levelCtx, filterMap, record, &plan, edges, planning.bounds[loopfilter.PlaneY], levels); err != nil {
				return plan, err
			}
			if err := frameWorkAppendLoopFilterChromaEdges(ctx, levelCtx, filterMap, record, &plan, edges, planning, levels); err != nil {
				return plan, err
			}
			col = nextCol
		}
	}
	if plan.Missing != 0 {
		return plan, threading.ErrInvalidBatch
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
	storedEdges, ok := frameWorkLoopFilterCounterLen(plan.StoredEdges)
	if !ok {
		return result, frame.ErrInvalidFormat
	}
	edges := req.Edges[:storedEdges]
	if err := ctx.applyLoopFilterEdgesInPlanePassOrder(&result, edges, req.Schedule, loopfilter.PlaneV); err != nil {
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
	storedEdges, ok := frameWorkLoopFilterCounterLen(plan.StoredEdges)
	if !ok {
		return result, frame.ErrInvalidFormat
	}
	edges := req.Edges[:storedEdges]
	if err := ctx.applyLoopFilterEdgesInPlanePassOrder(&result, edges, req.Schedule, loopfilter.PlaneY); err != nil {
		return result, err
	}
	return result, nil
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterEdgesInPlanePassOrder(result *FrameWorkLoopFilterPostFilterApplyResult, edges []FrameWorkLoopFilterPostFilterEdge, scheduleScratch []uint32, maxPlane loopfilter.Plane) error {
	before := result.Edges
	expected, ok := frameWorkLoopFilterCounter(len(edges))
	if !ok {
		return loopfilter.ErrInvalidFilter
	}
	if len(edges) <= frameWorkLoopFilterApplyScheduleCap {
		var schedule [frameWorkLoopFilterApplyScheduleCap]uint32
		return ctx.applyLoopFilterEdgesInPlanePassOrderSchedule(result, edges, schedule[:len(edges)], maxPlane, before, expected)
	}
	if len(scheduleScratch) >= len(edges) {
		return ctx.applyLoopFilterEdgesInPlanePassOrderSchedule(result, edges, scheduleScratch[:len(edges)], maxPlane, before, expected)
	}
	return ctx.applyLoopFilterEdgesInPlanePassOrderScan(result, edges, maxPlane, before, expected)
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterEdgesInPlanePassOrderSchedule(result *FrameWorkLoopFilterPostFilterApplyResult, edges []FrameWorkLoopFilterPostFilterEdge, schedule []uint32, maxPlane loopfilter.Plane, before uint32, expected uint32) error {
	var counts [3][2]uint32
	var levelUsed [loopfilter.MaxLevel + 1]bool
	for i := range edges {
		edge := &edges[i]
		if edge.Plane > maxPlane || edge.Edge > loopfilter.EdgeHorizontal ||
			edge.Length4 == 0 || edge.Level == 0 || edge.Level > loopfilter.MaxLevel {
			return loopfilter.ErrInvalidFilter
		}
		counts[edge.Plane][edge.Edge]++
		levelUsed[edge.Level] = true
	}
	var thresholds [loopfilter.MaxLevel + 1]loopfilter.Thresholds
	for level := uint8(1); level <= loopfilter.MaxLevel; level++ {
		if !levelUsed[level] {
			continue
		}
		th, err := loopfilter.ThresholdsForLevel(level, ctx.Event.LoopFilter.Sharpness)
		if err != nil {
			return err
		}
		thresholds[level] = th
	}
	var starts [3][2]uint32
	next := uint32(0)
	for plane := loopfilter.PlaneY; plane <= maxPlane; plane++ {
		for edgeKind := loopfilter.EdgeVertical; edgeKind <= loopfilter.EdgeHorizontal; edgeKind++ {
			starts[plane][edgeKind] = next
			next += counts[plane][edgeKind]
		}
	}
	positions := starts
	for i := range edges {
		edge := &edges[i]
		pos := positions[edge.Plane][edge.Edge]
		schedule[pos] = uint32(i)
		positions[edge.Plane][edge.Edge] = pos + 1
	}
	if err := ctx.applyLoopFilterScheduledEdges(result, edges, schedule, starts, counts, maxPlane, &thresholds); err != nil {
		return err
	}
	if result.Edges-before != expected {
		return loopfilter.ErrInvalidFilter
	}
	return nil
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterEdgesInPlanePassOrderScan(result *FrameWorkLoopFilterPostFilterApplyResult, edges []FrameWorkLoopFilterPostFilterEdge, maxPlane loopfilter.Plane, before uint32, expected uint32) error {
	var planes [3]frame.Plane
	var planeReady [3]bool
	var thresholds [loopfilter.MaxLevel + 1]loopfilter.Thresholds
	var thresholdReady [loopfilter.MaxLevel + 1]bool
	bytesPerSample := ctx.Output.Layout.BytesPerSample
	bitDepth := ctx.Output.Format.BitDepth
	for plane := loopfilter.PlaneY; plane <= maxPlane; plane++ {
		for edgeKind := loopfilter.EdgeVertical; edgeKind <= loopfilter.EdgeHorizontal; edgeKind++ {
			for i := range edges {
				edge := &edges[i]
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
				x := int32(edge.X4) * 4
				y := int32(edge.Y4) * 4
				length := int32(edge.Length4) * 4
				switch edge.Width {
				case 4:
					if err := loopfilter.Filter4Edge(planes[plane], bytesPerSample, bitDepth, edge.Edge, x, y, length, thresholds[edge.Level]); err != nil {
						return err
					}
				case 6:
					if err := loopfilter.Filter6Edge(planes[plane], bytesPerSample, bitDepth, edge.Edge, x, y, length, thresholds[edge.Level]); err != nil {
						return err
					}
				case 8:
					if err := loopfilter.Filter8Edge(planes[plane], bytesPerSample, bitDepth, edge.Edge, x, y, length, thresholds[edge.Level]); err != nil {
						return err
					}
				case 14:
					if err := loopfilter.Filter14Edge(planes[plane], bytesPerSample, bitDepth, edge.Edge, x, y, length, thresholds[edge.Level]); err != nil {
						return err
					}
				default:
					return loopfilter.ErrInvalidFilter
				}
				frameWorkCountAppliedLoopFilterEdge(result, edge)
			}
		}
	}
	if result.Edges-before != expected {
		return loopfilter.ErrInvalidFilter
	}
	return nil
}

func (ctx FrameWorkPostFilterContext) applyLoopFilterScheduledEdges(result *FrameWorkLoopFilterPostFilterApplyResult, edges []FrameWorkLoopFilterPostFilterEdge, schedule []uint32, starts [3][2]uint32, counts [3][2]uint32, maxPlane loopfilter.Plane, thresholds *[loopfilter.MaxLevel + 1]loopfilter.Thresholds) error {
	bytesPerSample := ctx.Output.Layout.BytesPerSample
	bitDepth := ctx.Output.Format.BitDepth
	for plane := loopfilter.PlaneY; plane <= maxPlane; plane++ {
		if counts[plane][loopfilter.EdgeVertical]+counts[plane][loopfilter.EdgeHorizontal] == 0 {
			continue
		}
		dst, ok := frameWorkLoopFilterOutputPlane(*ctx.Output, plane)
		if !ok {
			return loopfilter.ErrInvalidFilter
		}
		planeW, planeH, err := frameWorkLoopFilterBufferSize(ctx, plane)
		if err != nil {
			return err
		}
		dst = frameWorkLoopFilterAlignedPlane(dst, planeW, planeH, bytesPerSample)
		for edgeKind := loopfilter.EdgeVertical; edgeKind <= loopfilter.EdgeHorizontal; edgeKind++ {
			start := int(starts[plane][edgeKind])
			end := start + int(counts[plane][edgeKind])
			for _, edgeIndex := range schedule[start:end] {
				edge := &edges[edgeIndex]
				x := int32(edge.X4) * 4
				y := int32(edge.Y4) * 4
				length := int32(edge.Length4) * 4
				switch edge.Width {
				case 4:
					if err := loopfilter.Filter4Edge(dst, bytesPerSample, bitDepth, edge.Edge, x, y, length, thresholds[edge.Level]); err != nil {
						return err
					}
				case 6:
					if err := loopfilter.Filter6Edge(dst, bytesPerSample, bitDepth, edge.Edge, x, y, length, thresholds[edge.Level]); err != nil {
						return err
					}
				case 8:
					if err := loopfilter.Filter8Edge(dst, bytesPerSample, bitDepth, edge.Edge, x, y, length, thresholds[edge.Level]); err != nil {
						return err
					}
				case 14:
					if err := loopfilter.Filter14Edge(dst, bytesPerSample, bitDepth, edge.Edge, x, y, length, thresholds[edge.Level]); err != nil {
						return err
					}
				default:
					return loopfilter.ErrInvalidFilter
				}
				frameWorkCountAppliedLoopFilterEdge(result, edge)
			}
		}
	}
	return nil
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
func frameWorkLoopFilterAlignedPlane(plane frame.Plane, planeW int32, planeH int32, bytesPerSample int) frame.Plane {
	if bytesPerSample <= 0 || plane.Stride <= 0 {
		return plane
	}
	if planeW > int32(plane.Width) {
		if w := int32(plane.Stride / bytesPerSample); planeW > w {
			planeW = w
		}
		if planeW > int32(plane.Width) {
			plane.Width = int(planeW)
		}
	}
	if planeH > int32(plane.Height) {
		if h := int32(len(plane.Pix) / plane.Stride); planeH > h {
			planeH = h
		}
		if planeH > int32(plane.Height) {
			plane.Height = int(planeH)
		}
	}
	return plane
}

func frameWorkCountAppliedLoopFilterEdge(result *FrameWorkLoopFilterPostFilterApplyResult, edge *FrameWorkLoopFilterPostFilterEdge) {
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
	if miCols > uint32(^uint16(0)) || miRows > uint32(^uint16(0)) {
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
	if cols <= 0 || rows <= 0 ||
		filterMap.Stride <= 0 || filterMap.Rows <= 0 ||
		cols > int(filterMap.Stride) || rows > int(filterMap.Rows) {
		return threading.ErrInvalidBatch
	}
	limit := int(^uint(0) >> 1)
	if uint64(filterMap.Stride)*uint64(filterMap.Rows) > uint64(limit) {
		return threading.ErrInvalidBatch
	}
	length := int(filterMap.Stride) * int(filterMap.Rows)
	if len(filterMap.Records) < length {
		return threading.ErrInvalidBatch
	}
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
	if !record.TransformTree.Variable {
		return frameWorkAppendLoopFilterFixedLumaTXBs(ctx, levelCtx, filterMap, record, plan, edges, bounds, req, currentVertical, currentHorizontal)
	}
	block := record.Block
	blockX4 := int(block.X4)
	blockY4 := int(block.Y4)
	return record.TransformTree.ForEachLumaTXB(req, func(tx tile.TransformBlock) error {
		plan.LumaTXBs++
		frameX4 := int(block.MICol) + int(tx.X4) - blockX4
		frameY4 := int(block.MIRow) + int(tx.Y4) - blockY4
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

func frameWorkAppendLoopFilterFixedLumaTXBs(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, req tile.TransformTreeRequest, currentVertical uint8, currentHorizontal uint8) error {
	if !record.TransformTree.Y.Valid() {
		return threading.ErrInvalidBatch
	}
	dims, ok := record.TransformTree.Y.Dimensions()
	if !ok {
		return threading.ErrInvalidBatch
	}
	var verticalWidth uint8
	var horizontalWidth uint8
	var err error
	haveVerticalWidth := false
	haveHorizontalWidth := false
	block := record.Block
	blockX4 := int(block.X4)
	blockY4 := int(block.Y4)
	reqX4 := int(req.X4)
	reqY4 := int(req.Y4)
	xEnd := reqX4 + int(req.VisibleW4)
	yEnd := reqY4 + int(req.VisibleH4)
	frameBaseX4 := int(block.MICol) + reqX4 - blockX4
	frameY4 := int(block.MIRow) + reqY4 - blockY4
	for y4 := reqY4; y4 < yEnd; y4 += int(dims.H4) {
		visibleH4 := minInt(int(dims.H4), yEnd-y4)
		if visibleH4 <= 0 {
			return threading.ErrInvalidBatch
		}
		frameX4 := frameBaseX4
		for x4 := reqX4; x4 < xEnd; x4 += int(dims.W4) {
			visibleW4 := minInt(int(dims.W4), xEnd-x4)
			if visibleW4 <= 0 {
				return threading.ErrInvalidBatch
			}
			plan.LumaTXBs++
			if frameX4 > 0 {
				if !haveVerticalWidth {
					verticalWidth, err = frameWorkLoopFilterWidth(loopfilter.PlaneY, loopfilter.EdgeVertical, record.TransformTree.Y)
					if err != nil {
						return err
					}
					haveVerticalWidth = true
				}
				if err := frameWorkAppendLoopFilterLumaEdgeSegmentsWithWidth(ctx, levelCtx, filterMap, record, plan, edges, bounds, loopfilter.EdgeVertical, frameX4, frameY4, visibleH4, record.TransformTree.Y, verticalWidth, currentVertical); err != nil {
					return err
				}
			}
			if frameY4 > 0 {
				if !haveHorizontalWidth {
					horizontalWidth, err = frameWorkLoopFilterWidth(loopfilter.PlaneY, loopfilter.EdgeHorizontal, record.TransformTree.Y)
					if err != nil {
						return err
					}
					haveHorizontalWidth = true
				}
				if err := frameWorkAppendLoopFilterLumaEdgeSegmentsWithWidth(ctx, levelCtx, filterMap, record, plan, edges, bounds, loopfilter.EdgeHorizontal, frameX4, frameY4, visibleW4, record.TransformTree.Y, horizontalWidth, currentHorizontal); err != nil {
					return err
				}
			}
			frameX4 += int(dims.W4)
		}
		frameY4 += int(dims.H4)
	}
	return nil
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
	currentWidth, err := frameWorkLoopFilterWidth(loopfilter.PlaneY, edge, tx)
	if err != nil {
		return err
	}
	return frameWorkAppendLoopFilterLumaEdgeSegmentsWithWidth(ctx, levelCtx, filterMap, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentWidth, currentLevel)
}

func frameWorkAppendLoopFilterLumaEdgeSegmentsWithWidth(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentWidth uint8, currentLevel uint8) error {
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
	segStart := 0
	var segWidth uint8
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
			X4:                uint16(segX4),
			Y4:                uint16(segY4),
			Length4:           uint16(end - start),
			Level:             segLevel,
			Transform:         tx,
			Width:             uint8(w),
			LevelFromPrevious: segFromPrevious,
			BlockMICol:        uint16(record.Block.MICol),
			BlockMIRow:        uint16(record.Block.MIRow),
		})
		return nil
	}
	var previousCache frameWorkLoopFilterLumaPreviousCache
	needPreviousLevel := currentLevel == 0
	if handled, err := frameWorkTryAppendLoopFilterFixedLumaEdge(levelCtx, filterMap, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentWidth, currentLevel, needPreviousLevel); handled || err != nil {
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
		previousWidth, previousLevel, err := previousCache.lookup(levelCtx, filterMap, ctx.Event.SequenceHeader.ColorConfig, edge, x4, y4, offset, int(plan.MICols), int(plan.MIRows), needPreviousLevel)
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

func frameWorkTryAppendLoopFilterFixedLumaEdge(levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentWidth uint8, currentLevel uint8, needPreviousLevel bool) (bool, error) {
	previous, ok, err := frameWorkLoopFilterPreviousRecord(filterMap, edge, x4, y4, int(plan.MICols), int(plan.MIRows))
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
	if err := frameWorkLoopFilterValidateBlockVisible(previous); err != nil {
		return true, err
	}
	if !frameWorkLoopFilterPreviousLumaRunInBlock(previous, edge, x4, y4, length4) {
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
		X4:                uint16(x4),
		Y4:                uint16(y4),
		Length4:           uint16(length4),
		Level:             level,
		Transform:         tx,
		Width:             uint8(w),
		LevelFromPrevious: fromPrevious,
		BlockMICol:        uint16(record.Block.MICol),
		BlockMIRow:        uint16(record.Block.MIRow),
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
	startLocalX4 := int(record.Block.X4) + startX4 - int(record.Block.MICol)
	startLocalY4 := int(record.Block.Y4) + startY4 - int(record.Block.MIRow)
	endLocalX4 := int(record.Block.X4) + endX4 - int(record.Block.MICol)
	endLocalY4 := int(record.Block.Y4) + endY4 - int(record.Block.MIRow)
	reqX4 := int(req.X4)
	reqY4 := int(req.Y4)
	return startLocalX4 >= reqX4 && startLocalY4 >= reqY4 &&
		startLocalX4 < reqX4+int(req.VisibleW4) &&
		startLocalY4 < reqY4+int(req.VisibleH4) &&
		endLocalX4 >= reqX4 && endLocalY4 >= reqY4 &&
		endLocalX4 < reqX4+int(req.VisibleW4) &&
		endLocalY4 < reqY4+int(req.VisibleH4)
}

func frameWorkLoopFilterValidateBlockVisible(record *threading.FrameWorkLoopFilterBlockRecord) error {
	block := record.Block
	dims, ok := block.Size.Dimensions()
	if !ok || block.VisibleW4 == 0 || block.VisibleH4 == 0 ||
		block.VisibleW4 > dims.W4 || block.VisibleH4 > dims.H4 {
		return threading.ErrInvalidBatch
	}
	return nil
}

func frameWorkLoopFilterPreviousLumaRunInBlock(record *threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge, x4 int, y4 int, length4 int) bool {
	startX4, startY4, err := frameWorkLoopFilterPreviousTarget4(edge, x4, y4)
	if err != nil {
		return false
	}
	endBoundaryX4, endBoundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, length4-1)
	endX4, endY4, err := frameWorkLoopFilterPreviousTarget4(edge, endBoundaryX4, endBoundaryY4)
	if err != nil {
		return false
	}
	block := record.Block
	startLocalX4 := int(block.X4) + startX4 - int(block.MICol)
	startLocalY4 := int(block.Y4) + startY4 - int(block.MIRow)
	endLocalX4 := int(block.X4) + endX4 - int(block.MICol)
	endLocalY4 := int(block.Y4) + endY4 - int(block.MIRow)
	reqX4 := int(block.X4)
	reqY4 := int(block.Y4)
	return startLocalX4 >= reqX4 && startLocalY4 >= reqY4 &&
		startLocalX4 < reqX4+int(block.VisibleW4) &&
		startLocalY4 < reqY4+int(block.VisibleH4) &&
		endLocalX4 >= reqX4 && endLocalY4 >= reqY4 &&
		endLocalX4 < reqX4+int(block.VisibleW4) &&
		endLocalY4 < reqY4+int(block.VisibleH4)
}

type frameWorkLoopFilterLumaPreviousCache struct {
	valid      bool
	miCol      uint16
	miRow      uint16
	record     *threading.FrameWorkLoopFilterBlockRecord
	req        tile.TransformTreeRequest
	fixed      bool
	width      uint8
	tx         tile.TransformBlock
	txWidth    uint8
	txValid    bool
	level      uint8
	levelValid bool
}

func (c *frameWorkLoopFilterLumaPreviousCache) lookup(levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int, needLevel bool) (uint8, uint8, error) {
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
		var width uint8
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
	localX4 := int(c.record.Block.X4) + targetX4 - int(c.record.Block.MICol)
	localY4 := int(c.record.Block.Y4) + targetY4 - int(c.record.Block.MIRow)
	reqX4 := int(c.req.X4)
	reqY4 := int(c.req.Y4)
	if localX4 < reqX4 || localY4 < reqY4 ||
		localX4 >= reqX4+int(c.req.VisibleW4) ||
		localY4 >= reqY4+int(c.req.VisibleH4) {
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
func frameWorkLoopFilterPreviousLumaCellWidth(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int) (uint8, error) {
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
	if prevCol >= cols || prevRow >= rows ||
		prevRow >= int(filterMap.Rows) || prevCol >= int(filterMap.Stride) {
		return nil, false, threading.ErrInvalidBatch
	}
	record := &filterMap.Records[prevRow*int(filterMap.Stride)+prevCol]
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
		base := levelCtx.base[plane][loopfilter.EdgeVertical]
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
		if !record.SkipTransform && record.TransformTree.HasUV {
			count, err := frameWorkLoopFilterCountChromaTXBsWithShifts(planning.color, record, planning.ssX, planning.ssY)
			if err != nil {
				return err
			}
			plan.ChromaTXBs += int32(count)
		}
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
	verticalFusedUV := fuseUV && vertical[0] != 0 && vertical[1] != 0
	horizontalFusedUV := fuseUV && horizontal[0] != 0 && horizontal[1] != 0
	verticalEqualUV := verticalFusedUV && vertical[0] == vertical[1]
	horizontalEqualUV := horizontalFusedUV && horizontal[0] == horizontal[1]
	verticalUnequalUV := verticalFusedUV && vertical[0] != vertical[1]
	horizontalUnequalUV := horizontalFusedUV && horizontal[0] != horizontal[1]
	var verticalWidth uint8
	var horizontalWidth uint8
	if verticalFusedUV {
		verticalWidth, err = frameWorkLoopFilterWidth(loopfilter.PlaneU, loopfilter.EdgeVertical, record.TransformTree.UV)
		if err != nil {
			return err
		}
	}
	if horizontalFusedUV {
		horizontalWidth, err = frameWorkLoopFilterWidth(loopfilter.PlaneU, loopfilter.EdgeHorizontal, record.TransformTree.UV)
		if err != nil {
			return err
		}
	}
	for y := 0; y < int(block.VisibleH4); y += int(uvDims.H4) {
		for x := 0; x < int(block.VisibleW4); x += int(uvDims.W4) {
			plan.ChromaTXBs++
			tx := tile.TransformBlock{
				X4:        uint8(int(block.X4) + x),
				Y4:        uint8(int(block.Y4) + y),
				Size:      record.TransformTree.UV,
				VisibleW4: uint8(minInt(int(uvDims.W4), int(block.VisibleW4)-x)),
				VisibleH4: uint8(minInt(int(uvDims.H4), int(block.VisibleH4)-y)),
			}
			frameX4, frameY4 := frameWorkLoopFilterChromaFrame4WithShifts(record, int(tx.X4), int(tx.Y4), planning.ssX, planning.ssY)
			if fuseUV {
				if frameX4 > 0 {
					if verticalEqualUV {
						if err := frameWorkAppendLoopFilterChromaEdgeSegmentsEqualUVWithWidth(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, verticalWidth, vertical[0], planning.ssX, planning.ssY); err != nil {
							return err
						}
					} else if verticalUnequalUV {
						if err := frameWorkAppendLoopFilterChromaEdgeSegmentsUnequalUVWithWidth(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, verticalWidth, vertical[0], vertical[1], planning.ssX, planning.ssY); err != nil {
							return err
						}
					} else {
						if err := frameWorkAppendLoopFilterChromaEdgeSegmentsUV(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], loopfilter.EdgeVertical, frameX4, frameY4, int(tx.VisibleH4), tx.Size, vertical[0], vertical[1], planning.ssX, planning.ssY); err != nil {
							return err
						}
					}
				}
				if frameY4 > 0 {
					if horizontalEqualUV {
						if err := frameWorkAppendLoopFilterChromaEdgeSegmentsEqualUVWithWidth(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, horizontalWidth, horizontal[0], planning.ssX, planning.ssY); err != nil {
							return err
						}
					} else if horizontalUnequalUV {
						if err := frameWorkAppendLoopFilterChromaEdgeSegmentsUnequalUVWithWidth(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, horizontalWidth, horizontal[0], horizontal[1], planning.ssX, planning.ssY); err != nil {
							return err
						}
					} else {
						if err := frameWorkAppendLoopFilterChromaEdgeSegmentsUV(ctx, levelCtx, filterMap, planning.color, record, plan, edges, bounds[0], loopfilter.EdgeHorizontal, frameX4, frameY4, int(tx.VisibleW4), tx.Size, horizontal[0], horizontal[1], planning.ssX, planning.ssY); err != nil {
							return err
						}
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

func frameWorkAppendLoopFilterChromaTXBEdges(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, plane loopfilter.Plane, tx tile.TransformBlock, currentVertical uint8, currentHorizontal uint8, ssX uint8, ssY uint8) error {
	frameX4, frameY4 := frameWorkLoopFilterChromaFrame4WithShifts(record, int(tx.X4), int(tx.Y4), ssX, ssY)
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

func frameWorkAppendLoopFilterChromaTXBEdgesUV(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, tx tile.TransformBlock, currentVerticalU uint8, currentVerticalV uint8, currentHorizontalU uint8, currentHorizontalV uint8, ssX uint8, ssY uint8) error {
	frameX4, frameY4 := frameWorkLoopFilterChromaFrame4WithShifts(record, int(tx.X4), int(tx.Y4), ssX, ssY)
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

func frameWorkAppendLoopFilterChromaEdgeSegmentsUV(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevelU uint8, currentLevelV uint8, ssX uint8, ssY uint8) error {
	if currentLevelU != 0 && currentLevelU == currentLevelV {
		return frameWorkAppendLoopFilterChromaEdgeSegmentsEqualUV(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentLevelU, ssX, ssY)
	}
	if currentLevelU != 0 && currentLevelV != 0 {
		return frameWorkAppendLoopFilterChromaEdgeSegmentsUnequalUV(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentLevelU, currentLevelV, ssX, ssY)
	}
	if err := frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.PlaneU, edge, x4, y4, length4, tx, currentLevelU, ssX, ssY); err != nil {
		return err
	}
	return frameWorkAppendLoopFilterChromaEdgeSegments(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.PlaneV, edge, x4, y4, length4, tx, currentLevelV, ssX, ssY)
}

func frameWorkAppendLoopFilterChromaEdgeSegmentsUnequalUV(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevelU uint8, currentLevelV uint8, ssX uint8, ssY uint8) error {
	if length4 <= 0 {
		return nil
	}
	currentWidth, err := frameWorkLoopFilterWidth(loopfilter.PlaneU, edge, tx)
	if err != nil {
		return err
	}
	return frameWorkAppendLoopFilterChromaEdgeSegmentsUnequalUVWithWidth(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentWidth, currentLevelU, currentLevelV, ssX, ssY)
}

func frameWorkAppendLoopFilterChromaEdgeSegmentsUnequalUVWithWidth(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentWidth uint8, currentLevelU uint8, currentLevelV uint8, ssX uint8, ssY uint8) error {
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
	beforeCandidates := plan.EdgeCandidates
	beforeStored := plan.StoredEdges
	if handled, err := frameWorkTryAppendLoopFilterFixedChromaEdge(levelCtx, filterMap, color, record, plan, edges, bounds, loopfilter.PlaneU, edge, x4, y4, length4, tx, currentWidth, currentLevelU, false, ssX, ssY); handled || err != nil {
		if err != nil {
			return err
		}
		beforeStoredLen, ok := frameWorkLoopFilterCounterLen(beforeStored)
		if !ok {
			return frame.ErrInvalidFormat
		}
		candidates := plan.EdgeCandidates - beforeCandidates
		stored := plan.StoredEdges - beforeStored
		storedCount, ok := frameWorkLoopFilterCounterLen(stored)
		if !ok {
			return frame.ErrInvalidFormat
		}
		for i := 0; i < storedCount; i++ {
			edgeCopy := edges[beforeStoredLen+i]
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
	var segWidth uint8
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
			X4:         uint16(segX4),
			Y4:         uint16(segY4),
			Length4:    uint16(end - start),
			Level:      currentLevelU,
			Transform:  tx,
			Width:      uint8(w),
			BlockMICol: uint16(record.Block.MICol),
			BlockMIRow: uint16(record.Block.MIRow),
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
		previousWidth, hasChroma, _, err := previousCache.lookup(levelCtx, filterMap, color, loopfilter.PlaneU, edge, x4, y4, offset, int(plan.MICols), int(plan.MIRows), false, ssX, ssY)
		if err != nil {
			return err
		}
		var width uint8
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

func frameWorkAppendLoopFilterChromaEdgeSegmentsEqualUV(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevel uint8, ssX uint8, ssY uint8) error {
	if length4 <= 0 || currentLevel == 0 {
		return nil
	}
	currentWidth, err := frameWorkLoopFilterWidth(loopfilter.PlaneU, edge, tx)
	if err != nil {
		return err
	}
	return frameWorkAppendLoopFilterChromaEdgeSegmentsEqualUVWithWidth(ctx, levelCtx, filterMap, color, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentWidth, currentLevel, ssX, ssY)
}

func frameWorkAppendLoopFilterChromaEdgeSegmentsEqualUVWithWidth(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentWidth uint8, currentLevel uint8, ssX uint8, ssY uint8) error {
	if length4 <= 0 || currentLevel == 0 {
		return nil
	}
	length4, err := frameWorkLoopFilterClampEdgeLengthInBounds(bounds, edge, x4, y4, length4)
	if err != nil {
		return err
	}
	if length4 <= 0 {
		return nil
	}
	if handled, err := frameWorkTryAppendLoopFilterFixedChromaEdgeUV(filterMap, color, record, plan, edges, bounds, edge, x4, y4, length4, tx, currentWidth, currentLevel, ssX, ssY); handled || err != nil {
		return err
	}

	segStart := 0
	var segWidth uint8
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
			X4:         uint16(segX4),
			Y4:         uint16(segY4),
			Length4:    uint16(end - start),
			Level:      currentLevel,
			Transform:  tx,
			Width:      uint8(w),
			BlockMICol: uint16(record.Block.MICol),
			BlockMIRow: uint16(record.Block.MIRow),
		}
		frameWorkStoreLoopFilterEdge(plan, edges, edgeRecord)
		edgeRecord.Plane = loopfilter.PlaneV
		frameWorkStoreLoopFilterEdge(plan, edges, edgeRecord)
		return nil
	}
	hadAny := false
	var lastWidth uint8
	var previousCache frameWorkLoopFilterChromaPreviousCache
	for offset := range length4 {
		previousWidth, hasChroma, _, err := previousCache.lookup(levelCtx, filterMap, color, loopfilter.PlaneU, edge, x4, y4, offset, int(plan.MICols), int(plan.MIRows), false, ssX, ssY)
		if err != nil {
			return err
		}
		var width uint8
		if hasChroma {
			width = min(previousWidth, currentWidth)
			hadAny = true
			lastWidth = width
		} else {
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

// frameWorkAppendLoopFilterChromaEdgeSegments mirrors
// frameWorkAppendLoopFilterLumaEdgeSegments for chroma. libaom's chroma
// deblock equally resolves filter_length per chroma cell from the local
// current/previous transform sizes (set_one_param_for_line_chroma); without
// per-cell splitting the chroma plane reproduced the same q32 10-bit
// divergence the luma fix already covered.
func frameWorkAppendLoopFilterChromaEdgeSegments(ctx FrameWorkPostFilterContext, levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentLevel uint8, ssX uint8, ssY uint8) error {
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
	var segWidth uint8
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
			X4:                uint16(segX4),
			Y4:                uint16(segY4),
			Length4:           uint16(end - start),
			Level:             segLevel,
			Transform:         tx,
			Width:             uint8(w),
			LevelFromPrevious: segFromPrevious,
			BlockMICol:        uint16(record.Block.MICol),
			BlockMIRow:        uint16(record.Block.MIRow),
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
	var lastWidth uint8
	var previousCache frameWorkLoopFilterChromaPreviousCache
	needPreviousLevel := currentLevel == 0
	if handled, err := frameWorkTryAppendLoopFilterFixedChromaEdge(levelCtx, filterMap, color, record, plan, edges, bounds, plane, edge, x4, y4, length4, tx, currentWidth, currentLevel, needPreviousLevel, ssX, ssY); handled || err != nil {
		return err
	}
	for offset := range length4 {
		previousWidth, hasChroma, previousLevel, err := previousCache.lookup(levelCtx, filterMap, color, plane, edge, x4, y4, offset, int(plan.MICols), int(plan.MIRows), needPreviousLevel, ssX, ssY)
		if err != nil {
			return err
		}
		var width uint8
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

func frameWorkTryAppendLoopFilterFixedChromaEdge(levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentWidth uint8, currentLevel uint8, needPreviousLevel bool, ssX uint8, ssY uint8) (bool, error) {
	previous, ok, err := frameWorkLoopFilterPreviousChromaRecordWithShifts(filterMap, edge, x4, y4, int(plan.MICols), int(plan.MIRows), ssX, ssY)
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
		X4:                uint16(x4),
		Y4:                uint16(y4),
		Length4:           uint16(length4),
		Level:             level,
		Transform:         tx,
		Width:             uint8(w),
		LevelFromPrevious: fromPrevious,
		BlockMICol:        uint16(record.Block.MICol),
		BlockMIRow:        uint16(record.Block.MIRow),
	})
	return true, nil
}

func frameWorkTryAppendLoopFilterFixedChromaEdgeUV(filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, plan *FrameWorkLoopFilterPostFilterPlan, edges []FrameWorkLoopFilterPostFilterEdge, bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, tx tile.TransformSize, currentWidth uint8, currentLevel uint8, ssX uint8, ssY uint8) (bool, error) {
	previous, ok, err := frameWorkLoopFilterPreviousChromaRecordWithShifts(filterMap, edge, x4, y4, int(plan.MICols), int(plan.MIRows), ssX, ssY)
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
	previousWidth, err := frameWorkLoopFilterWidth(loopfilter.PlaneU, edge, block.Size)
	if err != nil {
		return true, err
	}
	width := min(previousWidth, currentWidth)
	if width == 0 || currentLevel == 0 {
		return true, nil
	}
	w, ok, err := frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, x4, y4, length4, width)
	if err != nil {
		return true, err
	}
	if !ok {
		return true, nil
	}
	edgeRecord := FrameWorkLoopFilterPostFilterEdge{
		Plane:      loopfilter.PlaneU,
		Edge:       edge,
		X4:         uint16(x4),
		Y4:         uint16(y4),
		Length4:    uint16(length4),
		Level:      currentLevel,
		Transform:  tx,
		Width:      uint8(w),
		BlockMICol: uint16(record.Block.MICol),
		BlockMIRow: uint16(record.Block.MIRow),
	}
	frameWorkStoreLoopFilterEdge(plan, edges, edgeRecord)
	edgeRecord.Plane = loopfilter.PlaneV
	frameWorkStoreLoopFilterEdge(plan, edges, edgeRecord)
	return true, nil
}

func frameWorkLoopFilterPreviousChromaRecordCoversRun(record *threading.FrameWorkLoopFilterBlockRecord, edge loopfilter.Edge, x4 int, y4 int, length4 int, ssX uint8, ssY uint8) bool {
	if record == nil || length4 <= 0 {
		return false
	}
	ssXi := int(ssX)
	ssYi := int(ssY)
	block := record.Block
	switch edge {
	case loopfilter.EdgeVertical:
		prevCol := ((x4 - 1) << ssXi) | ssXi
		startRow := (y4 << ssYi) | ssYi
		endRow := ((y4 + length4 - 1) << ssYi) | ssYi
		return prevCol >= int(block.MICol) && prevCol < int(block.MIColEnd) &&
			startRow >= int(block.MIRow) && endRow < int(block.MIRowEnd)
	case loopfilter.EdgeHorizontal:
		startCol := (x4 << ssXi) | ssXi
		endCol := ((x4 + length4 - 1) << ssXi) | ssXi
		prevRow := ((y4 - 1) << ssYi) | ssYi
		return prevRow >= int(block.MIRow) && prevRow < int(block.MIRowEnd) &&
			startCol >= int(block.MICol) && endCol < int(block.MIColEnd)
	default:
		return false
	}
}

func frameWorkLoopFilterPreviousChromaRunInBlock(record *threading.FrameWorkLoopFilterBlockRecord, block tile.TransformBlock, edge loopfilter.Edge, x4 int, y4 int, length4 int, ssX uint8, ssY uint8) bool {
	startX4, startY4, err := frameWorkLoopFilterPreviousTarget4(edge, x4, y4)
	if err != nil {
		return false
	}
	endBoundaryX4, endBoundaryY4 := frameWorkLoopFilterBoundaryOffset(edge, x4, y4, length4-1)
	endX4, endY4, err := frameWorkLoopFilterPreviousTarget4(edge, endBoundaryX4, endBoundaryY4)
	if err != nil {
		return false
	}
	startLocalX4 := (int(record.Block.X4) >> ssX) + startX4 - (int(record.Block.MICol) >> ssX)
	startLocalY4 := (int(record.Block.Y4) >> ssY) + startY4 - (int(record.Block.MIRow) >> ssY)
	endLocalX4 := (int(record.Block.X4) >> ssX) + endX4 - (int(record.Block.MICol) >> ssX)
	endLocalY4 := (int(record.Block.Y4) >> ssY) + endY4 - (int(record.Block.MIRow) >> ssY)
	return frameWorkLoopFilterTransformBlockContains(block, startLocalX4, startLocalY4) &&
		frameWorkLoopFilterTransformBlockContains(block, endLocalX4, endLocalY4)
}

type frameWorkLoopFilterChromaPreviousCache struct {
	valid      bool
	miCol      uint16
	miRow      uint16
	record     *threading.FrameWorkLoopFilterBlockRecord
	block      tile.TransformBlock
	blockOK    bool
	width      uint8
	level      uint8
	levelValid bool
}

func (c *frameWorkLoopFilterChromaPreviousCache) lookup(levelCtx frameWorkLoopFilterLevelContext, filterMap FrameWorkLoopFilterMap, color parser.ColorConfig, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int, needLevel bool, ssX uint8, ssY uint8) (uint8, bool, uint8, error) {
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
		var width uint8
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
	localX4 := (int(c.record.Block.X4) >> ssX) + targetX4 - (int(c.record.Block.MICol) >> ssX)
	localY4 := (int(c.record.Block.Y4) >> ssY) + targetY4 - (int(c.record.Block.MIRow) >> ssY)
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
func frameWorkLoopFilterPreviousChromaCellWidth(ctx FrameWorkPostFilterContext, filterMap FrameWorkLoopFilterMap, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, offset int, cols int, rows int) (uint8, bool, error) {
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

func frameWorkLoopFilterPreviousChromaRecordWithShifts(filterMap FrameWorkLoopFilterMap, edge loopfilter.Edge, boundaryX4 int, boundaryY4 int, cols int, rows int, ssX uint8, ssY uint8) (*threading.FrameWorkLoopFilterBlockRecord, bool, error) {
	ssXi := int(ssX)
	ssYi := int(ssY)
	var prevCol, prevRow int
	switch edge {
	case loopfilter.EdgeVertical:
		prevCol = ((boundaryX4 - 1) << ssXi) | ssXi
		prevRow = (boundaryY4 << ssYi) | ssYi
	case loopfilter.EdgeHorizontal:
		prevCol = (boundaryX4 << ssXi) | ssXi
		prevRow = ((boundaryY4 - 1) << ssYi) | ssYi
	default:
		return nil, false, loopfilter.ErrInvalidFilter
	}
	if prevCol < 0 || prevRow < 0 {
		return nil, false, nil
	}
	if prevCol >= cols || prevRow >= rows ||
		prevRow >= int(filterMap.Rows) || prevCol >= int(filterMap.Stride) {
		return nil, false, threading.ErrInvalidBatch
	}
	record := &filterMap.Records[prevRow*int(filterMap.Stride)+prevCol]
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
		limit = int((bounds.posHeight+3)>>2) - y4
	case loopfilter.EdgeHorizontal:
		limit = int((bounds.posWidth+3)>>2) - x4
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

func frameWorkLoopFilterScheduledWidth(ctx FrameWorkPostFilterContext, plane loopfilter.Plane, edge loopfilter.Edge, x4 int, y4 int, length4 int, width uint8) (uint8, bool, error) {
	bounds, err := frameWorkLoopFilterPlaneBounds(ctx, plane)
	if err != nil {
		return 0, false, err
	}
	return frameWorkLoopFilterScheduledWidthInBounds(bounds, edge, x4, y4, length4, width)
}

type frameWorkLoopFilterBounds struct {
	posWidth  int32
	posHeight int32
	bufWidth  int32
	bufHeight int32
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

func frameWorkLoopFilterScheduledWidthInBounds(bounds frameWorkLoopFilterBounds, edge loopfilter.Edge, x4 int, y4 int, length4 int, width uint8) (uint8, bool, error) {
	x, ok := frameWorkLoopFilterScale4(x4)
	if !ok {
		return 0, false, loopfilter.ErrInvalidFilter
	}
	y, ok := frameWorkLoopFilterScale4(y4)
	if !ok {
		return 0, false, loopfilter.ErrInvalidFilter
	}
	length, ok := frameWorkLoopFilterScale4(length4)
	if !ok {
		return 0, false, loopfilter.ErrInvalidFilter
	}
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
func frameWorkLoopFilterBufferSize(ctx FrameWorkPostFilterContext, plane loopfilter.Plane) (int32, int32, error) {
	alignedW, ok := frameWorkLoopFilterAligned8Dimension(ctx.Event.FrameSize.CodedWidth)
	if !ok {
		return 0, 0, threading.ErrInvalidBatch
	}
	alignedH, ok := frameWorkLoopFilterAligned8Dimension(ctx.Event.FrameSize.Height)
	if !ok {
		return 0, 0, threading.ErrInvalidBatch
	}
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
	localX4 := int(record.Block.X4) + frameX4 - int(record.Block.MICol)
	localY4 := int(record.Block.Y4) + frameY4 - int(record.Block.MIRow)
	reqX4 := int(req.X4)
	reqY4 := int(req.Y4)
	if localX4 < reqX4 || localY4 < reqY4 ||
		localX4 >= reqX4+int(req.VisibleW4) ||
		localY4 >= reqY4+int(req.VisibleH4) {
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
	localX4 := (int(record.Block.X4) >> ssX) + frameX4 - (int(record.Block.MICol) >> ssX)
	localY4 := (int(record.Block.Y4) >> ssY) + frameY4 - (int(record.Block.MIRow) >> ssY)
	if !frameWorkLoopFilterTransformBlockContains(block, localX4, localY4) {
		return 0, false, nil
	}
	return block.Size, true, nil
}

func frameWorkLoopFilterTransformBlockContains(tx tile.TransformBlock, x4 int, y4 int) bool {
	txX4 := int(tx.X4)
	txY4 := int(tx.Y4)
	return x4 >= txX4 && y4 >= txY4 &&
		x4 < txX4+int(tx.VisibleW4) &&
		y4 < txY4+int(tx.VisibleH4)
}

func frameWorkLoopFilterWidth(plane loopfilter.Plane, edge loopfilter.Edge, tx tile.TransformSize) (uint8, error) {
	if edge > loopfilter.EdgeHorizontal || !tx.Valid() || int(tx) >= len(frameWorkLoopFilterWidthByTX) {
		return 0, threading.ErrInvalidBatch
	}
	if plane == loopfilter.PlaneU || plane == loopfilter.PlaneV {
		return frameWorkLoopFilterWidthByTX[tx][1][edge], nil
	}
	if plane != loopfilter.PlaneY {
		return 0, threading.ErrInvalidBatch
	}
	return frameWorkLoopFilterWidthByTX[tx][0][edge], nil
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
func frameWorkLoopFilterPlaneSize(ctx FrameWorkPostFilterContext, plane loopfilter.Plane) (int32, int32, error) {
	width, ok := frameWorkLoopFilterFrameDimension32(ctx.Event.FrameSize.CodedWidth)
	if !ok {
		return 0, 0, threading.ErrInvalidBatch
	}
	height, ok := frameWorkLoopFilterFrameDimension32(ctx.Event.FrameSize.Height)
	if !ok {
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
func frameWorkLoopFilterMIAlignedLength(length int32) int32 {
	return (length + 3) &^ 3
}

func frameWorkLoopFilterSubsampledLength(length int32, subsampled bool) int32 {
	if subsampled {
		return (length + 1) >> 1
	}
	return length
}

func frameWorkLoopFilterDowngradeWidth(width uint8) uint8 {
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

const frameWorkLoopFilterMaxInt32 = uint32(1<<31 - 1)

func frameWorkLoopFilterFrameDimension32(v uint32) (int32, bool) {
	if v == 0 || v > frameWorkLoopFilterMaxInt32-7 {
		return 0, false
	}
	return int32(v), true
}

func frameWorkLoopFilterAligned8Dimension(v uint32) (int32, bool) {
	if v == 0 || v > frameWorkLoopFilterMaxInt32-7 {
		return 0, false
	}
	return int32((v + 7) &^ 7), true
}

func frameWorkLoopFilterScale4(v int) (int32, bool) {
	const max = int64(1<<31 - 1)
	const min = -1 << 31
	scaled := int64(v) * 4
	if scaled < min || scaled > max {
		return 0, false
	}
	return int32(scaled), true
}

func frameWorkLoopFilterEdgeFits(width uint8, edge loopfilter.Edge, x int32, y int32, length int32, frameWidth int32, frameHeight int32) bool {
	if length <= 0 || frameWidth <= 0 || frameHeight <= 0 {
		return false
	}
	radius := int32(width) / 2
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
	if plan.StoredEdges < uint32(len(edges)) {
		storedEdges := int(plan.StoredEdges)
		edges[storedEdges] = edge
		plan.StoredEdges++
		return
	}
	plan.DroppedEdges++
}

func frameWorkLoopFilterCounterLen(count uint32) (int, bool) {
	if uint64(count) > uint64(^uint(0)>>1) {
		return 0, false
	}
	return int(count), true
}

func frameWorkLoopFilterCounter(count int) (uint32, bool) {
	if count < 0 || uint64(count) > uint64(^uint32(0)) {
		return 0, false
	}
	return uint32(count), true
}

func frameWorkLoopFilterCountChromaTXBsWithShifts(color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, ssX uint8, ssY uint8) (int, error) {
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

func frameWorkLoopFilterChromaBlockWithShifts(color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord, ssX uint8, ssY uint8) (tile.TransformBlock, bool, error) {
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
	reqX4 := int(req.X4)
	reqY4 := int(req.Y4)
	ssXi := int(ssX)
	ssYi := int(ssY)
	x4 := reqX4 >> ssXi
	y4 := reqY4 >> ssYi
	visibleW4 := ((reqX4 + int(req.VisibleW4) + ssXi) >> ssXi) - x4
	visibleH4 := ((reqY4 + int(req.VisibleH4) + ssYi) >> ssYi) - y4
	if visibleW4 <= 0 || visibleH4 <= 0 ||
		x4+visibleW4 > tile.MaxBlockModeSlots ||
		y4+visibleH4 > tile.MaxBlockModeSlots {
		return tile.TransformBlock{}, false, threading.ErrInvalidBatch
	}
	return tile.TransformBlock{
		X4:        uint8(x4),
		Y4:        uint8(y4),
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

func frameWorkLoopFilterChromaFrame4WithShifts(record *threading.FrameWorkLoopFilterBlockRecord, x4 int, y4 int, ssX uint8, ssY uint8) (int, int) {
	return (int(record.Block.MICol) >> ssX) + x4 - (int(record.Block.X4) >> ssX),
		(int(record.Block.MIRow) >> ssY) + y4 - (int(record.Block.Y4) >> ssY)
}

func frameWorkLoopFilterSubsamplingShift(subsampled bool) uint8 {
	if subsampled {
		return 1
	}
	return 0
}

func frameWorkLoopFilterTransformTreeRequest(color parser.ColorConfig, record *threading.FrameWorkLoopFilterBlockRecord) (tile.TransformTreeRequest, error) {
	block := record.Block
	dims, ok := block.Size.Dimensions()
	if !ok || block.VisibleW4 == 0 || block.VisibleH4 == 0 ||
		block.VisibleW4 > dims.W4 || block.VisibleH4 > dims.H4 {
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
	miCol := int(block.MICol)
	miRow := int(block.MIRow)
	miColEnd := int(block.MIColEnd)
	miRowEnd := int(block.MIRowEnd)
	if record.SegmentID >= parser.MaxSegments ||
		miColEnd <= miCol ||
		miRowEnd <= miRow ||
		miColEnd > cols ||
		miRowEnd > rows ||
		col < miCol ||
		col >= miColEnd ||
		row < miRow ||
		row >= miRowEnd {
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
		record.RefFrame >= parser.RefFrames ||
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
	if record == nil || levelCtx.loopFilter == nil || levelCtx.segmentation == nil || levelCtx.delta == nil {
		return levels, threading.ErrInvalidBatch
	}
	if record.RefFrame >= parser.RefFrames ||
		(record.Mode != loopfilter.ModeDeltaClassZero && record.Mode != loopfilter.ModeDeltaClassMotion) {
		return levels, loopfilter.ErrInvalidFilter
	}
	if levelCtx.lumaZero {
		return levels, nil
	}
	planeCount := 3
	if levelCtx.monoChrome {
		planeCount = 1
	}
	for plane := 0; plane < planeCount; plane++ {
		for edge := range 2 {
			base := levelCtx.base[plane][edge]
			if plane != int(loopfilter.PlaneY) && base == 0 {
				continue
			}
			deltaLF := int8(0)
			if levelCtx.delta.DeltaLFPresent {
				if !levelCtx.delta.DeltaLFMulti {
					deltaLF = record.DeltaLFFromBase
				} else {
					switch plane {
					case int(loopfilter.PlaneY):
						deltaLF = record.DeltaLF[edge]
					case int(loopfilter.PlaneU):
						deltaLF = record.DeltaLF[2]
					case int(loopfilter.PlaneV):
						deltaLF = record.DeltaLF[3]
					}
				}
			}
			segDelta := int16(0)
			if levelCtx.segmentation.Enabled {
				if record.SegmentID >= parser.MaxSegments {
					return levels, loopfilter.ErrInvalidFilter
				}
				data := levelCtx.segmentation.Data.Segments[record.SegmentID]
				switch plane {
				case int(loopfilter.PlaneY):
					if edge == int(loopfilter.EdgeVertical) {
						segDelta = data.DeltaLFYV
					} else {
						segDelta = data.DeltaLFYH
					}
				case int(loopfilter.PlaneU):
					segDelta = data.DeltaLFU
				case int(loopfilter.PlaneV):
					segDelta = data.DeltaLFV
				}
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
			levels[plane][edge] = uint8(level)
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
