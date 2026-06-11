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
	"sync"

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
	a.bound = true
	return nil
}

// reset clears the per-frame records before a frame's tiles fill them.
func (a *loopFilterApplier) reset() error {
	return a.filtMap.Reset()
}

// apply runs the decoder's deblocking planner and kernels over the encoder
// reconstruction with the frame's signaled filter levels, leaving recon equal
// to the decoder's post-loop-filter output.
func (a *loopFilterApplier) apply(recon *SourceFrame420, lf parser.LoopFilterParams) error {
	if !a.bound {
		return fmt.Errorf("encoder: loop-filter applier not initialized")
	}
	if lf.LevelY[0] == 0 && lf.LevelY[1] == 0 && lf.LevelU == 0 && lf.LevelV == 0 {
		return nil
	}
	event := a.event
	event.LoopFilter = lf
	output := frame.Frame{
		Format: frame.Format{
			Width: recon.Width, Height: recon.Height,
			BitDepth: 8, SubsamplingX: true, SubsamplingY: true,
		},
		Layout: frame.Layout{BytesPerSample: 1},
		Y:      frame.Plane{Pix: recon.Y, Stride: recon.YStride, Width: recon.Width, Height: recon.Height},
		U:      frame.Plane{Pix: recon.U, Stride: recon.ChromaStride, Width: recon.Width / 2, Height: recon.Height / 2},
		V:      frame.Plane{Pix: recon.V, Stride: recon.ChromaStride, Width: recon.Width / 2, Height: recon.Height / 2},
	}
	ctx := decoder.FrameWorkPostFilterContext{
		Event:         event,
		Output:        &output,
		LoopFilterMap: &a.filtMap,
	}

	// Banded planning: partition block-origin rows across workers, then
	// concatenate the per-band edges in band order (which reproduces the
	// whole-frame plan) and run the sequential kernel passes once.
	rows := int(a.filtMap.Rows)
	bands := len(a.bandBufs)
	rowsPerBand := (rows + bands - 1) / bands
	var wg sync.WaitGroup
	counts := make([]uint32, bands)
	errs := make([]error, bands)
	for b := range bands {
		r0 := b * rowsPerBand
		r1 := min(r0+rowsPerBand, rows)
		if r0 >= r1 {
			continue
		}
		wg.Add(1)
		go func(b, r0, r1 int) {
			defer wg.Done()
			plan, err := ctx.LoopFilterPostFilterPlan(decoder.FrameWorkLoopFilterPostFilterRequest{
				Map:             a.filtMap,
				Edges:           a.bandBufs[b],
				TrustedCoverage: true,
				MIRowStart:      r0,
				MIRowEnd:        r1,
			})
			if err != nil {
				errs[b] = err
				return
			}
			if plan.DroppedEdges != 0 {
				errs[b] = fmt.Errorf("encoder: loop-filter band %d dropped %d edges", b, plan.DroppedEdges)
				return
			}
			counts[b] = plan.StoredEdges
		}(b, r0, r1)
	}
	wg.Wait()
	// Partition the per-band edges by plane, preserving band (row-major)
	// order within each plane; the three planes touch disjoint surfaces, so
	// their sequential kernel passes run concurrently.
	if a.planeEdges[0] == nil {
		for p := range a.planeEdges {
			a.planeEdges[p] = make([]decoder.FrameWorkLoopFilterPostFilterEdge, 0, len(a.edges))
		}
	}
	for p := range a.planeEdges {
		a.planeEdges[p] = a.planeEdges[p][:0]
	}
	for b := range bands {
		if errs[b] != nil {
			return errs[b]
		}
		for _, e := range a.bandBufs[b][:counts[b]] {
			a.planeEdges[e.Plane] = append(a.planeEdges[e.Plane], e)
		}
	}
	var applyErrs [3]error
	for p := range a.planeEdges {
		if len(a.planeEdges[p]) == 0 {
			continue
		}
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			_, applyErrs[p] = ctx.ApplyPlannedLoopFilterPlaneEdges(a.planeEdges[p], nil, loopfilter.Plane(p))
		}(p)
	}
	wg.Wait()
	for p := range applyErrs {
		if applyErrs[p] != nil {
			return applyErrs[p]
		}
	}
	return nil
}
