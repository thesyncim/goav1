// Ported from libaom:
//   av1/encoder/pickcdef.c (av1_pick_cdef_from_qp: the CDEF_PICK_FROM_Q
//   quadratic fits from the AC quantizer to the primary/secondary strengths
//   and the qindex-derived damping)
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package encoder

import (
	"fmt"
	"math"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// cdefStrengthsFromQIndex is av1_pick_cdef_from_qp at 8-bit depth without
// the screen-content model: one strength set (cdef_bits = 0), the quadratic
// strength fits, and damping 3 + (qindex >> 6). The returned strengths pack
// primary*4 + secondary like the syntax expects.
func cdefStrengthsFromQIndex(qIndex uint8, intra bool) (y uint8, uv uint8, damping uint8) {
	quant, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.PlaneY)
	if err != nil {
		return 0, 0, 3
	}
	q := float64(quant.AC)
	clamp := func(v float64, hi int) int {
		i := int(math.Round(v))
		if i < 0 {
			return 0
		}
		if i > hi {
			return hi
		}
		return i
	}
	var yF1, yF2, uvF1, uvF2 int
	if intra {
		yF1 = clamp(q*q*0.0000033731974+q*0.008070594+0.0187634, 15)
		yF2 = clamp(q*q*0.0000029167343+q*0.0027798624+0.0079405, 3)
		uvF1 = clamp(q*q*-0.0000130790995+q*0.012892405-0.00748388, 15)
		uvF2 = clamp(q*q*0.0000032651783+q*0.00035520183+0.00228092, 3)
	} else {
		yF1 = clamp(q*q*-0.0000023593946+q*0.0068615186+0.02709886, 15)
		yF2 = clamp(q*q*-0.00000057629734+q*0.0013993345+0.03831067, 3)
		uvF1 = clamp(q*q*-0.0000007095069+q*0.0034628846+0.00887099, 15)
		uvF2 = clamp(q*q*0.00000023874085+q*0.00028223585+0.05576307, 3)
	}
	return uint8(yF1*4 + yF2), uint8(uvF1*4 + uvF2), 3 + (qIndex >> 6)
}

// cdefHeaderParams builds the frame-header view of the q-derived strengths.
// Below the quantizer threshold the strengths zero out: fine-quantized
// frames carry detail the deringing filter would smooth away, and the
// damage compounds through the reference chain at high rates.
const cdefMinQIndex = 96

func cdefHeaderParams(qIndex uint8, intra bool) CDEFParams {
	y, uv, damping := cdefStrengthsFromQIndex(qIndex, intra)
	p := CDEFParams{Damping: damping, Bits: 0}
	if qIndex >= cdefMinQIndex {
		p.YStrength[0] = y
		p.UVStrength[0] = uv
	}
	return p
}

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
// (derived from the loop-filter records), the pre-filter sample snapshot,
// and per-band scratch for the parallel unit-row application.
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
	unitRows  int
	seqHeader parser.SequenceHeader
	frameSize parser.FrameSize
	bound     bool

	// Persistent band workers reading their per-frame inputs from these
	// fields, so the apply path spawns no goroutines and allocates nothing.
	work    chan [2]int
	done    chan struct{}
	started bool
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
	a.work = make(chan [2]int, cdefApplyBands)
	a.done = make(chan struct{}, cdefApplyBands)
	for b := range cdefApplyBands {
		go func(b int) {
			for j := range a.work {
				band := a.jobReq
				band.InputScratch = a.bandIn[b]
				band.UnitDstScratch = a.bandUnit[b]
				_, a.errs[b] = a.jobCtx.ApplyCDEFPostFilterUnitRows(band, j[0], j[1])
				a.done <- struct{}{}
			}
		}(b)
	}
	a.started = true
}

type cdefDirectionGrid = cdef.DirectionGrid
type cdefVarianceGrid = cdef.VarianceGrid

const cdefApplyBands = 8

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
	for b := range a.bandIn {
		if len(a.bandIn[b]) < size.Input {
			a.bandIn[b] = make([]uint16, size.Input)
		}
		if len(a.bandUnit[b]) < size.UnitDst {
			a.bandUnit[b] = make([]uint16, size.UnitDst)
		}
	}
	a.unitRows = rows
	a.bound = true
	a.startWorkers()
	return nil
}

// markFromLoopFilterMap derives the per-unit index-coded flags the decoder
// reconstructs while parsing: a 64x64 unit codes (and therefore filters with)
// strength set zero exactly when at least one of its blocks is not skipped.
func (a *cdefApplier) markFromLoopFilterMap(m *threading.FrameWorkLoopFilterMap) {
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
				a.idxRead[unitRow*a.idxCols+miCol>>4] = true
			}
		}
	}
}

// apply runs the decoder's banded CDEF over the reconstruction with the
// frame's signaled parameters, leaving recon equal to the decoder's
// post-CDEF output.
func (a *cdefApplier) apply(recon *SourceFrame420, cdefParams parser.CDEFParams, lfMap *threading.FrameWorkLoopFilterMap) error {
	if !a.bound {
		return fmt.Errorf("encoder: cdef applier not initialized")
	}
	zero := true
	for i := 0; i < int(cdefParams.StrengthCount); i++ {
		if cdefParams.YStrength[i] != 0 || cdefParams.UVStrength[i] != 0 {
			zero = false
			break
		}
	}
	if zero {
		return nil
	}
	a.markFromLoopFilterMap(lfMap)
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
	if err := a.jobCtx.LoadCDEFPostFilterSamples(a.jobReq); err != nil {
		return err
	}
	bands := cdefApplyBands
	if a.unitRows < bands {
		bands = 1
	}
	a.startWorkers()
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
		a.work <- [2]int{r0, r1}
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
