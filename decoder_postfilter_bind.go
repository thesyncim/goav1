package goav1

import internalthreading "github.com/thesyncim/goav1/internal/av1/threading"

func DecoderFrameWorkCDEFIndexMapShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkCDEFIndexBatch(sequence, size, CDEFParams{})
	return batch.CDEFIndexMapShape()
}

func DecoderFrameWorkLoopFilterMapShape(sequence SequenceHeader, size FrameSize) (cols int, rows int, length int, err error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.LoopFilterMapShape()
}

func BindDecoderFrameWorkCDEFIndexMap(sequence SequenceHeader, size FrameSize, cdef CDEFParams, index []uint8, read []bool) (DecoderFrameWorkCDEFIndexMap, error) {
	batch := decoderFrameWorkCDEFIndexBatch(sequence, size, cdef)
	return batch.BindCDEFIndexMap(index, read)
}

func BindDecoderFrameWorkLoopFilterMap(sequence SequenceHeader, size FrameSize, records []DecoderFrameWorkLoopFilterBlockRecord) (DecoderFrameWorkLoopFilterMap, error) {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	return batch.BindLoopFilterMap(records)
}

func ResetDecoderFrameWorkCDEFIndexMap(indexMap DecoderFrameWorkCDEFIndexMap) error {
	return indexMap.Reset()
}

func MarkDecoderFrameWorkCDEFIndexMapBlock(indexMap DecoderFrameWorkCDEFIndexMap, cdef CDEFParams, visit TileBlockLoopVisit) error {
	return indexMap.MarkBlock(cdef, visit)
}

func ResetDecoderFrameWorkLoopFilterMap(filterMap DecoderFrameWorkLoopFilterMap) error {
	return filterMap.Reset()
}

func MarkDecoderFrameWorkLoopFilterMapBlock(filterMap DecoderFrameWorkLoopFilterMap, visit TileBlockLoopVisit, state *TileDecodeState) error {
	return filterMap.MarkBlock(visit, state)
}

func BindDecoderFrameWorkLoopFilterPostFilterRequest(size DecoderFrameWorkLoopFilterPostFilterScratchSize, filterMap DecoderFrameWorkLoopFilterMap, edges []DecoderFrameWorkLoopFilterPostFilterEdge) (DecoderFrameWorkLoopFilterPostFilterRequest, error) {
	if len(edges) < size.Edges {
		return DecoderFrameWorkLoopFilterPostFilterRequest{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkLoopFilterPostFilterRequest{
		Map:   filterMap,
		Edges: edges[:size.Edges],
	}, nil
}

func DecoderFrameWorkRestorationFramePlan(sequence SequenceHeader, size FrameSize, restoration RestorationParams) (TileRestorationFramePlan, error) {
	batch := decoderFrameWorkRestorationBatch(sequence, size, restoration)
	return batch.RestorationFramePlan()
}

func BindDecoderFrameWorkRestorationFrameBuffers(sequence SequenceHeader, size FrameSize, restoration RestorationParams, records []TileRestorationUnitRecord, above []uint16, below []uint16) (DecoderFrameWorkRestorationFrameBuffers, error) {
	batch := decoderFrameWorkRestorationBatch(sequence, size, restoration)
	return batch.BindRestorationFrameBuffers(records, above, below)
}

func BindDecoderFrameWorkCDEFPostFilterRequest(size DecoderFrameWorkCDEFPostFilterScratchSize, indexMap DecoderFrameWorkCDEFIndexMap, sampleScratch [3][]uint16, dstScratch [3][]uint16, directionGrid []CDEFDirectionGrid, varianceGrid []CDEFVarianceGrid, inputScratch []uint16, unitDstScratch []uint16) (DecoderFrameWorkCDEFPostFilterRequest, error) {
	if len(directionGrid) < size.DirectionGrid ||
		len(varianceGrid) < size.VarianceGrid ||
		len(inputScratch) < size.Input ||
		len(unitDstScratch) < size.UnitDst {
		return DecoderFrameWorkCDEFPostFilterRequest{}, ErrFrameShortBuffer
	}
	req := DecoderFrameWorkCDEFPostFilterRequest{
		IndexMap:       indexMap,
		DirectionGrid:  directionGrid[:size.DirectionGrid],
		VarianceGrid:   varianceGrid[:size.VarianceGrid],
		InputScratch:   inputScratch[:size.Input],
		UnitDstScratch: unitDstScratch[:size.UnitDst],
	}
	for plane := 0; plane < 3; plane++ {
		if len(sampleScratch[plane]) < size.Samples[plane] ||
			len(dstScratch[plane]) < size.Dst[plane] {
			return DecoderFrameWorkCDEFPostFilterRequest{}, ErrFrameShortBuffer
		}
		req.SampleScratch[plane] = sampleScratch[plane][:size.Samples[plane]]
		req.DstScratch[plane] = dstScratch[plane][:size.Dst[plane]]
	}
	return req, nil
}

func BindDecoderFrameWorkSuperResPostFilterRequest(size DecoderFrameWorkSuperResPostFilterScratchSize, outputFrame []byte, codedScratch [3][]uint16, outputScratch [3][]uint16) (DecoderFrameWorkSuperResPostFilterRequest, error) {
	if len(outputFrame) < size.OutputFrame {
		return DecoderFrameWorkSuperResPostFilterRequest{}, ErrFrameShortBuffer
	}
	req := DecoderFrameWorkSuperResPostFilterRequest{
		OutputFrame: outputFrame[:size.OutputFrame],
	}
	for plane := 0; plane < 3; plane++ {
		if len(codedScratch[plane]) < size.CodedSamples[plane] ||
			len(outputScratch[plane]) < size.OutputSamples[plane] {
			return DecoderFrameWorkSuperResPostFilterRequest{}, ErrFrameShortBuffer
		}
		req.CodedScratch[plane] = codedScratch[plane][:size.CodedSamples[plane]]
		req.OutputScratch[plane] = outputScratch[plane][:size.OutputSamples[plane]]
	}
	return req, nil
}

func BindDecoderFrameWorkRestorationPostFilterRequest(size DecoderFrameWorkRestorationPostFilterScratchSize, records [3][]TileRestorationUnitRecord, boundaries [3]TileRestorationStripeBoundaries, dataScratch []uint16, dstScratch []uint16, wienerScratch []uint16, sgrProjScratch []int32, boundaryAboveScratch []uint16, boundaryBelowScratch []uint16, optimized bool) (DecoderFrameWorkRestorationPostFilterRequest, error) {
	if len(dataScratch) < size.Samples.DataLen ||
		len(dstScratch) < size.Samples.DstLen ||
		len(wienerScratch) < size.Apply.Unit.Wiener ||
		len(sgrProjScratch) < size.Apply.Unit.SGRProj ||
		len(boundaryAboveScratch) < size.Apply.Boundary.Above ||
		len(boundaryBelowScratch) < size.Apply.Boundary.Below {
		return DecoderFrameWorkRestorationPostFilterRequest{}, ErrFrameShortBuffer
	}
	return DecoderFrameWorkRestorationPostFilterRequest{
		Records:     records,
		Boundaries:  boundaries,
		DataScratch: dataScratch[:size.Samples.DataLen],
		DstScratch:  dstScratch[:size.Samples.DstLen],
		Scratch: TileRestorationUnitRecordBoundaryScratch{
			Unit: TileRestorationUnitScratch{
				Wiener:  wienerScratch[:size.Apply.Unit.Wiener],
				SGRProj: sgrProjScratch[:size.Apply.Unit.SGRProj],
			},
			Boundary: TileRestorationStripeBoundaryScratch{
				Above: boundaryAboveScratch[:size.Apply.Boundary.Above],
				Below: boundaryBelowScratch[:size.Apply.Boundary.Below],
			},
		},
		Optimized: optimized,
	}, nil
}

func BindDecoderFrameWorkFilmGrainPostFilterRequest(size DecoderFrameWorkFilmGrainPostFilterScratchSize, lumaGrain []int16, chromaGrain [2][]int16, lumaSamples []uint16, chromaSamples [2][]uint16) (DecoderFrameWorkFilmGrainPostFilterRequest, error) {
	if len(lumaGrain) < size.LumaGrain ||
		len(lumaSamples) < size.LumaSamples {
		return DecoderFrameWorkFilmGrainPostFilterRequest{}, ErrFrameShortBuffer
	}
	req := DecoderFrameWorkFilmGrainPostFilterRequest{
		LumaGrain:   lumaGrain[:size.LumaGrain],
		LumaSamples: lumaSamples[:size.LumaSamples],
	}
	for plane := 0; plane < 2; plane++ {
		if len(chromaGrain[plane]) < size.ChromaGrain[plane] ||
			len(chromaSamples[plane]) < size.ChromaSamples[plane] {
			return DecoderFrameWorkFilmGrainPostFilterRequest{}, ErrFrameShortBuffer
		}
		req.ChromaGrain[plane] = chromaGrain[plane][:size.ChromaGrain[plane]]
		req.ChromaSamples[plane] = chromaSamples[plane][:size.ChromaSamples[plane]]
	}
	return req, nil
}

func decoderFrameWorkCDEFIndexBatch(sequence SequenceHeader, size FrameSize, cdef CDEFParams) internalthreading.FrameWorkBatch {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	batch.CDEF = cdef
	return batch
}

func decoderFrameWorkRestorationBatch(sequence SequenceHeader, size FrameSize, restoration RestorationParams) internalthreading.FrameWorkBatch {
	batch := decoderFrameWorkFrameBatch(sequence, size)
	batch.Restoration = restoration
	return batch
}

func decoderFrameWorkFrameBatch(sequence SequenceHeader, size FrameSize) internalthreading.FrameWorkBatch {
	return internalthreading.FrameWorkBatch{
		FrameWorkFrameContext: internalthreading.FrameWorkFrameContext{
			Sequence:  internalthreading.FrameWorkSequenceContextFromHeader(sequence),
			FrameSize: size,
		},
	}
}
