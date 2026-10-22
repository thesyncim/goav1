// The CDEF applier the encoder runs over its reconstruction; the strengths
// come from the SVT-AV1 search in cdef_search.go.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package encoder

import (
	"fmt"
	"sync"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// cdefParserParams is the parser-typed mirror the filtering context consumes.
func cdefParserParams(p CDEFParams) parser.CDEFParams {
	out := parser.CDEFParams{
		Damping:       p.Damping,
		Bits:          p.Bits,
		StrengthCount: 1 << p.Bits,
	}
	copy(out.YStrength[:], p.YStrength[:])
	copy(out.UVStrength[:], p.UVStrength[:])
	return out
}

// cdefApplier owns the reusable frame-level CDEF state: the index map
// (derived from the loop-filter records), the serial/snapshot sample scratch,
// and per-band line/boundary scratch for the parallel unit-row application.
type cdefApplier struct {
	idxValues []uint8
	idxRead   []bool
	idxMap    threading.FrameWorkCDEFIndexMap
	idxCols   int
	idxRows   int

	samples   [3][]uint16
	dirGrid   []cdefDirectionGrid
	varGrid   []cdefVarianceGrid
	bandIn    [][]uint16
	bandUnit  [][]uint16
	bandLine  [cdefApplyBands][3][]uint16
	bandBound [cdefApplyBands]decoder.FrameWorkCDEFPostFilterU8BandBoundary
	unitRows  int
	seqHeader parser.SequenceHeader
	frameSize parser.FrameSize
	bound     bool

	// Persistent band workers reading their per-frame inputs from these
	// fields, so the apply path spawns no goroutines and allocates nothing.
	work    chan cdefApplyJob
	done    chan struct{}
	started bool
	wg      sync.WaitGroup
	jobCtx  decoder.FrameWorkPostFilterContext
	jobReq  decoder.FrameWorkCDEFPostFilterRequest
	output  frame.Frame
	errs    [cdefApplyBands]error
}

// startWorkers launches the persistent unit-row band workers once.
func (a *cdefApplier) startWorkers() {
	if a.started {
		return
	}
	a.work = make(chan cdefApplyJob, cdefApplyBands)
	a.done = make(chan struct{}, cdefApplyBands)
	a.wg.Add(cdefApplyBands)
	for b := range cdefApplyBands {
		go func(b int) {
			defer a.wg.Done()
			for j := range a.work {
				band := a.jobReq
				band.SampleScratch = a.bandLine[b]
				band.InputScratch = a.bandIn[b]
				band.UnitDstScratch = a.bandUnit[b]
				_, a.errs[j.slot] = a.jobCtx.ApplyCDEFPostFilterUnitRowsU8(band, a.bandBound[j.slot], j.rowStart, j.rowEnd)
				a.done <- struct{}{}
			}
		}(b)
	}
	a.started = true
}

type cdefDirectionGrid = cdef.DirectionGrid
type cdefVarianceGrid = cdef.VarianceGrid

const cdefApplyBands = 8

type cdefApplyJob struct {
	rowStart int
	rowEnd   int
	slot     int
}

// init binds the index map and scratch for the stream's frame geometry.
func (a *cdefApplier) init(width, height int, cdefParams parser.CDEFParams) error {
	a.seqHeader = parser.SequenceHeader{
		EnableCDEF:  true,
		ColorConfig: parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true},
	}
	a.frameSize = parser.FrameSize{CodedWidth: uint32(width), Height: uint32(height)}
	batch := threading.FrameWorkBatch{
		FrameWorkFrameContext: threading.FrameWorkFrameContext{
			Sequence:  threading.FrameWorkSequenceContextFromHeader(a.seqHeader),
			FrameSize: a.frameSize,
			CDEF:      cdefParams,
		},
	}
	cols, rows, length, err := batch.CDEFIndexMapShape()
	if err != nil {
		return err
	}
	a.idxCols, a.idxRows = cols, rows
	if len(a.idxValues) < length {
		a.idxValues = make([]uint8, length)
		a.idxRead = make([]bool, length)
	}
	a.idxMap, err = batch.BindCDEFIndexMap(a.idxValues, a.idxRead)
	if err != nil {
		return err
	}
	// Scratch sizing reads output plane geometry only; nil planes with the
	// right shape avoid allocating a dummy 1080p frame just to size scratch.
	shape := frame.Frame{
		Format: frame.Format{
			Width: width, Height: height,
			BitDepth: 8, SubsamplingX: true, SubsamplingY: true,
		},
		Layout: frame.Layout{BytesPerSample: 1},
		Y:      frame.Plane{Stride: width, Width: width, Height: height},
		U:      frame.Plane{Stride: width / 2, Width: width / 2, Height: height / 2},
		V:      frame.Plane{Stride: width / 2, Width: width / 2, Height: height / 2},
	}
	// Scratch sizing requires parameters that still owe a CDEF pass; the
	// per-frame apply carries the real strengths.
	sizingParams := cdefParams
	// The nominal strength needs a nonzero PRIMARY component (strength
	// packs primary*4 + secondary): secondary-only filtering skips the
	// direction search, so a secondary-only nominal sizes the direction
	// and variance grids to zero and the first frame with a primary
	// strength overruns them.
	sizingParams.YStrength[0] = 4
	sizingParams.UVStrength[0] = 4
	sizingParams.StrengthCount = 1
	ctx := decoder.FrameWorkPostFilterContext{
		Event:  decoder.Event{SequenceHeader: a.seqHeader, FrameSize: a.frameSize, CDEF: sizingParams},
		Output: &shape,
	}
	size, err := ctx.CDEFPostFilterScratchLen()
	if err != nil {
		return err
	}
	for p := range a.samples {
		if len(a.samples[p]) < size.Samples[p] {
			a.samples[p] = make([]uint16, size.Samples[p])
		}
	}
	if len(a.dirGrid) < size.DirectionGrid {
		a.dirGrid = make([]cdefDirectionGrid, size.DirectionGrid)
		a.varGrid = make([]cdefVarianceGrid, size.VarianceGrid)
	}
	if a.bandIn == nil {
		a.bandIn = make([][]uint16, cdefApplyBands)
		a.bandUnit = make([][]uint16, cdefApplyBands)
	}
	planeWidths := [3]int{width, width / 2, width / 2}
	for b := range a.bandIn {
		if len(a.bandIn[b]) < size.Input {
			a.bandIn[b] = make([]uint16, size.Input)
		}
		if len(a.bandUnit[b]) < size.UnitDst {
			a.bandUnit[b] = make([]uint16, size.UnitDst)
		}
		for plane, planeWidth := range planeWidths {
			lineNeed := 2 * planeWidth
			if len(a.bandLine[b][plane]) < lineNeed {
				a.bandLine[b][plane] = make([]uint16, lineNeed)
			}
			boundaryNeed := cdef.VerticalBorder * planeWidth
			if len(a.bandBound[b].Top[plane]) < boundaryNeed {
				a.bandBound[b].Top[plane] = make([]byte, boundaryNeed)
			}
			if len(a.bandBound[b].Bottom[plane]) < boundaryNeed {
				a.bandBound[b].Bottom[plane] = make([]byte, boundaryNeed)
			}
		}
	}
	a.unitRows = rows
	a.bound = true
	return nil
}

// markFromLoopFilterMap derives the per-unit index-coded flags the decoder
// reconstructs while parsing: a 64x64 unit codes (and therefore filters with)
// strength set zero exactly when at least one of its blocks is not skipped.
func (a *cdefApplier) markFromLoopFilterMap(m *threading.FrameWorkLoopFilterMap, fbIdx []uint8) {
	for i := range a.idxRead[:a.idxCols*a.idxRows] {
		a.idxRead[i] = false
		a.idxValues[i] = 0
	}
	stride := int(m.Stride)
	rows := int(m.Rows)
	for miRow := 0; miRow < rows; miRow++ {
		unitRow := miRow >> 4
		base := miRow * stride
		for miCol := 0; miCol < stride; miCol++ {
			rec := &m.Records[base+miCol]
			if rec.Valid && !rec.SkipTransform {
				u := unitRow*a.idxCols + miCol>>4
				a.idxRead[u] = true
				if fbIdx != nil {
					a.idxValues[u] = fbIdx[u]
				}
			}
		}
	}
}

func (a *cdefApplier) close() {
	if a.work != nil {
		close(a.work)
		a.wg.Wait()
		a.work = nil
	}
	a.done = nil
	a.started = false
}

func (a *cdefApplier) bindApplyContext(recon *SourceFrame420, cdefParams parser.CDEFParams, lfMap *threading.FrameWorkLoopFilterMap, fbIdx []uint8, loadSnapshot bool) (bool, error) {
	if !a.bound {
		return false, fmt.Errorf("encoder: cdef applier not initialized")
	}
	zero := true
	for i := 0; i < int(cdefParams.StrengthCount); i++ {
		if cdefParams.YStrength[i] != 0 || cdefParams.UVStrength[i] != 0 {
			zero = false
			break
		}
	}
	if zero {
		return false, nil
	}
	a.markFromLoopFilterMap(lfMap, fbIdx)
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
		Event: decoder.Event{
			SequenceHeader: a.seqHeader,
			FrameSize:      a.frameSize,
			CDEF:           cdefParams,
		},
		Output:        &a.output,
		LoopFilterMap: lfMap,
	}
	a.jobCtx = a.jobCtx.WithCompletedPostFilters(decoder.FrameWorkPostFilterLoopFilter)
	a.jobReq = decoder.FrameWorkCDEFPostFilterRequest{
		IndexMap:      a.idxMap,
		SampleScratch: a.samples,
		DirectionGrid: a.dirGrid,
		VarianceGrid:  a.varGrid,
	}
	if loadSnapshot {
		if err := a.jobCtx.LoadCDEFPostFilterSamples(a.jobReq); err != nil {
			return false, err
		}
	}
	return true, nil
}

// apply runs the decoder's banded CDEF over the reconstruction with the
// frame's signaled parameters, leaving recon equal to the decoder's
// post-CDEF output.
func (a *cdefApplier) apply(recon *SourceFrame420, cdefParams parser.CDEFParams, lfMap *threading.FrameWorkLoopFilterMap, fbIdx []uint8) error {
	active, err := a.bindApplyContext(recon, cdefParams, lfMap, fbIdx, false)
	if err != nil || !active {
		return err
	}
	bands := cdefApplyBands
	if a.unitRows < bands {
		bands = 1
	}
	for b := range a.errs {
		a.errs[b] = nil
	}
	jobs := 0
	for b := 0; b < bands; b++ {
		r0 := b * a.unitRows / bands
		r1 := (b + 1) * a.unitRows / bands
		if r0 >= r1 {
			continue
		}
		if err := a.jobCtx.LoadCDEFPostFilterU8BandBoundary(&a.bandBound[b], r0, r1); err != nil {
			return err
		}
	}
	a.startWorkers()
	for b := 0; b < bands; b++ {
		r0 := b * a.unitRows / bands
		r1 := (b + 1) * a.unitRows / bands
		if r0 >= r1 {
			continue
		}
		a.work <- cdefApplyJob{rowStart: r0, rowEnd: r1, slot: b}
		jobs++
	}
	for range jobs {
		<-a.done
	}
	for b := range a.errs {
		if a.errs[b] != nil {
			return a.errs[b]
		}
	}
	return nil
}

func (a *cdefApplier) applySerial(recon *SourceFrame420, cdefParams parser.CDEFParams, lfMap *threading.FrameWorkLoopFilterMap, fbIdx []uint8) error {
	active, err := a.bindApplyContext(recon, cdefParams, lfMap, fbIdx, false)
	if err != nil || !active {
		return err
	}
	req := a.jobReq
	req.InputScratch = a.bandIn[0]
	req.UnitDstScratch = a.bandUnit[0]
	_, err = a.jobCtx.ApplyCDEFPostFilter(req)
	return err
}
