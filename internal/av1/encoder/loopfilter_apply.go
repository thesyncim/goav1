// Ported from libaom:
//   av1/encoder/picklpf.c (av1_pick_filter_level, LPF_PICK_FROM_Q: the
//   8-bit linear fits filt = q*17563 - 421574 (key) and
//   filt = q*12034 + 650707 (inter, the realtime multiplier) in Q18)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

// loopFilterMaxArea bounds the frame sizes that run the in-loop deblocking
// pass. The edge planner walks every MI cell per frame, which costs ~12ms at
// 720p (within a 60fps budget alongside encoding) but ~50ms at 1080p; larger
// frames keep the filter off until the planner is fast enough.
const loopFilterMaxArea = 1920 * 1080

// filterLevelFromQIndex is av1_pick_filter_level's LPF_PICK_FROM_Q estimate
// at 8-bit depth: a linear fit from the AC quantizer step to the searched
// filter level, with the realtime inter multiplier (12034, the >352x288
// nonrd shape). Levels clamp to the syntax range [0, 63].
func filterLevelFromQIndex(qIndex uint8, key bool) uint8 {
	q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.PlaneY)
	if err != nil {
		return 0
	}
	ac := int(q.AC)
	var filt int
	if key {
		filt = (ac*17563 - 421574 + (1 << 17)) >> 18
	} else {
		filt = (ac*12034 + 650707 + (1 << 17)) >> 18
	}
	if filt < 0 {
		filt = 0
	} else if filt > 63 {
		filt = 63
	}
	return uint8(filt)
}

// markLoopFilterBlock fills the decoder's per-MI loop-filter records for one
// coded block - the same data MarkBlockPtr derives from a decoded visit, so
// the encoder-side deblocking pass plans exactly the edges the decoder will.
func markLoopFilterBlock(m *threading.FrameWorkLoopFilterMap, block tile.BlockVisit, tree tile.TransformTreeResult,
	skip bool, intra bool, refFrame uint8, mode loopfilter.ModeDeltaClass) error {

	lfBlock, err := threading.FrameWorkLoopFilterBlockFromVisit(block)
	if err != nil {
		return err
	}
	record := threading.FrameWorkLoopFilterBlockRecord{
		Valid:         true,
		Block:         lfBlock,
		TransformTree: tree,
		SkipTransform: skip,
		Intra:         intra,
		RefFrame:      refFrame,
		Mode:          mode,
	}
	stride := int(m.Stride)
	if int(lfBlock.MIColEnd) > stride || int(lfBlock.MIRowEnd) > int(m.Rows) {
		return fmt.Errorf("encoder: loop-filter block %d,%d..%d,%d outside %dx%d map",
			lfBlock.MICol, lfBlock.MIRow, lfBlock.MIColEnd, lfBlock.MIRowEnd, m.Stride, m.Rows)
	}
	for miRow := int(lfBlock.MIRow); miRow < int(lfBlock.MIRowEnd); miRow++ {
		row := miRow * stride
		cells := m.Records[row+int(lfBlock.MICol) : row+int(lfBlock.MIColEnd)]
		for i := range cells {
			cells[i] = record
		}
	}
	return nil
}

// loopFilterApplier owns the reusable frame-level deblocking state: the
// decoder's loop-filter map over caller-owned records and the edge scratch
// for its planner. One applier serves a stream of same-sized frames.
type loopFilterApplier struct {
	records    []threading.FrameWorkLoopFilterBlockRecord
	edges      []decoder.FrameWorkLoopFilterPostFilterEdge
	bandBufs   [][]decoder.FrameWorkLoopFilterPostFilterEdge
	planeEdges [3][]decoder.FrameWorkLoopFilterPostFilterEdge
	schedule   []uint32
	filtMap    threading.FrameWorkLoopFilterMap
	event      decoder.Event
	bound      bool

	// Persistent band workers: per-frame goroutine spawns allocate their
	// closures, so the workers park on a job channel for the applier's
	// lifetime and read their per-frame inputs from these fields.
	work    chan lfJob
	done    chan struct{}
	started bool
	jobCtx  decoder.FrameWorkPostFilterContext
	output  frame.Frame
	counts  [lfPlanBands]uint32
	errs    [lfPlanBands]error
	// dirEdges partitions the planned edges by plane and edge direction for
	// the two-phase banded apply; applyErrs holds one slot per apply job.
	dirEdges  [3][2][]decoder.FrameWorkLoopFilterPostFilterEdge
	applyErrs [lfApplyMaxJobs]error
}

// lfApplyLumaBands is the fan-out of the per-direction luma apply. Bands cut
// at superblock-row boundaries; same-direction deblock edges never overlap in
// read or written pixels (the AV1 constraint libaom's row-mt loop filter
// relies on: filter taps are bounded by the adjacent transform sizes), so the
// banded application is bit-exact against the sequential pass, with the
// vertical phase completing frame-wide before the horizontal phase starts.
const lfApplyLumaBands = 4

// lfApplyMaxJobs bounds one phase's job count: luma bands plus one job per
// chroma plane.
const lfApplyMaxJobs = lfApplyLumaBands + 2

// lfJob is one parked-worker task: an edge-planning band or an edge-set apply.
type lfJob struct {
	plan   bool
	band   int
	r0, r1 int
	plane  int
	slot   int
	edges  []decoder.FrameWorkLoopFilterPostFilterEdge
}

// startWorkers launches the persistent workers once.
func (a *loopFilterApplier) startWorkers() {
	if a.started {
		return
	}
	a.work = make(chan lfJob, lfPlanBands)
	a.done = make(chan struct{}, lfPlanBands)
	for range lfPlanBands {
		go func() {
			for j := range a.work {
				if j.plan {
					plan, err := a.jobCtx.LoopFilterPostFilterPlan(decoder.FrameWorkLoopFilterPostFilterRequest{
						Map:             a.filtMap,
						Edges:           a.bandBufs[j.band],
						TrustedCoverage: true,
						MIRowStart:      j.r0,
						MIRowEnd:        j.r1,
					})
					switch {
					case err != nil:
						a.errs[j.band] = err
					case plan.DroppedEdges != 0:
						a.errs[j.band] = fmt.Errorf("encoder: loop-filter band %d dropped %d edges", j.band, plan.DroppedEdges)
					default:
						a.counts[j.band] = plan.StoredEdges
					}
				} else {
					_, a.applyErrs[j.slot] = a.jobCtx.ApplyPlannedLoopFilterPlaneEdges(j.edges, nil, loopfilter.Plane(j.plane))
				}
				a.done <- struct{}{}
			}
		}()
	}
	a.started = true
}

// lfPlanBands is the fan-out width of the banded edge planning pass. The
// planner sweep is read-only over the shared map, so bands plan
// concurrently; concatenating their edges in band order reproduces the
// whole-frame plan exactly (pinned by the decoder's banded parity test).
const lfPlanBands = 8

// init binds the map and scratch for the stream's frame geometry.
func (a *loopFilterApplier) init(width, height int) error {
	seq := parser.SequenceHeader{
		ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
	}
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:  threading.FrameWorkSequenceContextFromHeader(seq),
			FrameSize: parser.FrameSize{CodedWidth: uint32(width), Height: uint32(height)},
		},
	}
	_, _, length, err := batch.LoopFilterMapShape()
	if err != nil {
		return err
	}
	if len(a.records) < length {
		a.records = make([]threading.FrameWorkLoopFilterBlockRecord, length)
	}
	a.filtMap, err = batch.BindLoopFilterMap(a.records)
	if err != nil {
		return err
	}
	a.event = decoder.Event{
		SequenceHeader: seq,
		FrameSize:      parser.FrameSize{CodedWidth: uint32(width), Height: uint32(height)},
	}
	// Scratch sizing requires an event that still owes a loop-filter pass;
	// the per-frame apply overrides the levels.
	sizingEvent := a.event
	sizingEvent.LoopFilter.LevelY = [2]uint8{1, 1}
	ctx := decoder.FrameWorkPostFilterContext{Event: sizingEvent, LoopFilterMap: &a.filtMap}
	size, err := ctx.LoopFilterPostFilterScratchUpperBound()
	if err != nil {
		return err
	}
	a.edges, err = size.BindEdges(make([]decoder.FrameWorkLoopFilterPostFilterEdge, size.Edges))
	if err != nil {
		return err
	}
	// Pre-size the per-plane/direction partition buckets to the frame's edge
	// upper bound so the per-frame append that fills them never grows on the
	// first detailed frame (a black Prewarm frame plans almost no edges and
	// would otherwise leave these empty, allocating on the first live frame).
	for p := range a.dirEdges {
		for d := range a.dirEdges[p] {
			if cap(a.dirEdges[p][d]) < size.Edges {
				a.dirEdges[p][d] = make([]decoder.FrameWorkLoopFilterPostFilterEdge, 0, size.Edges)
			}
		}
	}
	a.schedule, err = size.BindSchedule(make([]uint32, size.Schedule))
	if err != nil {
		return err
	}
	// Per-band scratch: a band plans the blocks originating in its rows,
	// whose cells extend at most 8 MI rows (a 32px block) past the band end;
	// 8 edge candidates per cell bounds each band's storage.
	rows := int(a.filtMap.Rows)
	bands := lfPlanBands
	if rows < bands*4 {
		bands = 1
	}
	rowsPerBand := (rows + bands - 1) / bands
	bandBound := (rowsPerBand + 8) * int(a.filtMap.Stride) * 8
	a.bandBufs = make([][]decoder.FrameWorkLoopFilterPostFilterEdge, bands)
	for i := range a.bandBufs {
		a.bandBufs[i] = make([]decoder.FrameWorkLoopFilterPostFilterEdge, bandBound)
	}
	if cap(a.planeEdges[0]) < len(a.edges) {
		for p := range a.planeEdges {
			a.planeEdges[p] = make([]decoder.FrameWorkLoopFilterPostFilterEdge, 0, len(a.edges))
		}
	}
	a.bound = true
	return nil
}

// reset clears the per-frame records before a frame's tiles fill them.
func (a *loopFilterApplier) reset() error {
	return a.filtMap.Reset()
}

func (a *loopFilterApplier) close() {
	if a.work != nil {
		close(a.work)
		a.work = nil
	}
	a.done = nil
	a.started = false
}

func (a *loopFilterApplier) bindApplyContext(recon *SourceFrame420, lf parser.LoopFilterParams) (bool, error) {
	if !a.bound {
		return false, fmt.Errorf("encoder: loop-filter applier not initialized")
	}
	if lf.LevelY[0] == 0 && lf.LevelY[1] == 0 && lf.LevelU == 0 && lf.LevelV == 0 {
		return false, nil
	}
	event := a.event
	event.LoopFilter = lf
	a.output = frame.Frame{
		Format: frame.Format{
			Width: recon.Width, Height: recon.Height,
			BitDepth: 8, SubsamplingX: true, SubsamplingY: true,
		},
		Layout: frame.Layout{BytesPerSample: 1},
		Y:      frame.Plane{Pix: recon.Y, Stride: recon.YStride, Width: recon.Width, Height: recon.Height},
		U:      frame.Plane{Pix: recon.U, Stride: recon.ChromaStride, Width: recon.Width / 2, Height: recon.Height / 2},
		V:      frame.Plane{Pix: recon.V, Stride: recon.ChromaStride, Width: recon.Width / 2, Height: recon.Height / 2},
	}
	a.jobCtx = decoder.FrameWorkPostFilterContext{
		Event:         event,
		Output:        &a.output,
		LoopFilterMap: &a.filtMap,
	}
	return true, nil
}

// apply runs the decoder's deblocking planner and kernels over the encoder
// reconstruction with the frame's signaled filter levels, leaving recon equal
// to the decoder's post-loop-filter output.
func (a *loopFilterApplier) apply(recon *SourceFrame420, lf parser.LoopFilterParams) error {
	active, err := a.bindApplyContext(recon, lf)
	if err != nil || !active {
		return err
	}
	a.startWorkers()

	// Banded planning: partition block-origin rows across workers, then
	// concatenate the per-band edges in band order (which reproduces the
	// whole-frame plan) and run the sequential kernel passes once.
	rows := int(a.filtMap.Rows)
	bands := len(a.bandBufs)
	rowsPerBand := (rows + bands - 1) / bands
	counts := a.counts[:bands]
	errs := a.errs[:bands]
	jobs := 0
	for b := range bands {
		r0 := b * rowsPerBand
		r1 := min(r0+rowsPerBand, rows)
		counts[b], errs[b] = 0, nil
		if r0 >= r1 {
			continue
		}
		a.work <- lfJob{plan: true, band: b, r0: r0, r1: r1}
		jobs++
	}
	for range jobs {
		<-a.done
	}
	// Partition the per-band edges by plane and direction, preserving band
	// (row-major) order within each list. Planes touch disjoint surfaces, and
	// within one plane the sequential kernel semantics are "every vertical
	// edge, then every horizontal edge": the two directions run as separate
	// phases with a frame-wide barrier, and each phase fans out in
	// superblock-row bands (bit-exact; see lfApplyLumaBands).
	for p := range a.dirEdges {
		for d := range a.dirEdges[p] {
			a.dirEdges[p][d] = a.dirEdges[p][d][:0]
		}
	}
	for b := range bands {
		if errs[b] != nil {
			return errs[b]
		}
		for _, e := range a.bandBufs[b][:counts[b]] {
			a.dirEdges[e.Plane][e.Edge] = append(a.dirEdges[e.Plane][e.Edge], e)
		}
	}
	for dir := range 2 {
		jobs = 0
		slot := 0
		luma := a.dirEdges[0][dir]
		if len(luma) > 0 {
			target := (len(luma) + lfApplyLumaBands - 1) / lfApplyLumaBands
			start := 0
			for start < len(luma) {
				end := start + target
				if end >= len(luma) {
					end = len(luma)
				} else {
					// Extend to the next superblock-row boundary so a band
					// owns whole SB rows (edge records never span SB rows).
					cutRow := luma[end-1].Y4 / 16
					for end < len(luma) && luma[end].Y4/16 == cutRow {
						end++
					}
				}
				a.applyErrs[slot] = nil
				a.work <- lfJob{edges: luma[start:end], plane: 0, slot: slot}
				slot++
				jobs++
				start = end
			}
		}
		for p := 1; p <= 2; p++ {
			if len(a.dirEdges[p][dir]) == 0 {
				continue
			}
			a.applyErrs[slot] = nil
			a.work <- lfJob{edges: a.dirEdges[p][dir], plane: p, slot: slot}
			slot++
			jobs++
		}
		for range jobs {
			<-a.done
		}
		for s := 0; s < slot; s++ {
			if a.applyErrs[s] != nil {
				return a.applyErrs[s]
			}
		}
	}
	return nil
}

func (a *loopFilterApplier) applySerial(recon *SourceFrame420, lf parser.LoopFilterParams) error {
	active, err := a.bindApplyContext(recon, lf)
	if err != nil || !active {
		return err
	}
	plan, err := a.jobCtx.LoopFilterPostFilterPlan(decoder.FrameWorkLoopFilterPostFilterRequest{
		Map:             a.filtMap,
		Edges:           a.edges,
		TrustedCoverage: true,
	})
	switch {
	case err != nil:
		return err
	case plan.DroppedEdges != 0:
		return fmt.Errorf("encoder: loop-filter dropped %d edges", plan.DroppedEdges)
	}
	edges := a.edges[:plan.StoredEdges]
	_, err = a.jobCtx.ApplyPlannedLoopFilterEdges(edges, a.schedule[:plan.StoredEdges])
	return err
}
