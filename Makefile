.PHONY: test bench fuzz-smoke testvectors testvectors-fast testvectors-full test-motion-conformance test-transform-conformance alloc

FUZZTIME ?= 250000x
FUZZPARALLEL ?= 8
FUZZFLAGS = -run '^$$' -fuzztime=$(FUZZTIME) -parallel=$(FUZZPARALLEL)

test:
	go test ./...

bench:
	go test -run '^$$' -bench=. -benchmem ./...

fuzz-smoke:
	go test ./internal/av1/bitstream $(FUZZFLAGS) -fuzz=FuzzReadLEB128
	go test ./internal/av1/ivf $(FUZZFLAGS) -fuzz=FuzzIterator
	go test ./internal/av1/obu $(FUZZFLAGS) -fuzz=FuzzParseHeader
	go test ./internal/av1/obu $(FUZZFLAGS) -fuzz=FuzzTemporalUnitIterator
	go test ./internal/av1/obu $(FUZZFLAGS) -fuzz=FuzzAnnexBIterator
	go test ./internal/av1/rtp $(FUZZFLAGS) -fuzz=FuzzPayloadIterator
	go test ./internal/av1/rtp $(FUZZFLAGS) -fuzz=FuzzPacketizer
	go test ./internal/av1/rtp $(FUZZFLAGS) -fuzz=FuzzAssembleFrame
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzUpdateCDF
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzCDFState
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzReaderBinarySymbol
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzReaderSignedDelta
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzReaderUniformSubexp
	go test ./internal/av1/frame $(FUZZFLAGS) -fuzz=FuzzSamplePlaneRoundTrip
	go test ./internal/av1/frame $(FUZZFLAGS) -fuzz=FuzzBorderedSamplePlaneRoundTrip
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseSequenceHeader
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseFrameHeaderPrefix
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseIntraFrameSize
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseFrameSize
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseTileInfo
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseTileGroupHeader
	go test ./internal/av1/quantize $(FUZZFLAGS) -fuzz=FuzzDequantizeBlock
	go test ./internal/av1/quantize $(FUZZFLAGS) -fuzz=FuzzQuantizeFPNoQMatrix
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzBuildJobs
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzJobPayload
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzNewEntropyReader
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeStateReset
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeStateBlockDeltas
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadPartition
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzWalkBlocksScripted
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeBlockLoop
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadBlockModePrefix
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadIntraEntry
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadSelectedTransformSize
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadTransformPartitionSplit
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeTransformTree
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadInterReferences
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadInterMode
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadMotionMode
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadInterIntra
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadCompoundBlend
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadCoeffPrimitives
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadCoefficientsTXB
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzCoeffEntropyContext
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeLumaCoefficients
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeChromaCoefficients
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeBlockCoefficients
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeStateRestorationUnit
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationUnitNone
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationUnitRecordNone
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationUnitRecordWithBoundaries
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationPlaneRecords
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationFramePlane
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationFrame
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationFrameToFrame
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzRestorationUnitSchedule
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzBuildRestorationFramePlan
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzBindRestorationFramePlanBuffers
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzRestorationPlaneRecordBuffer
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzRestorationUnitGeometry
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzRestorationProcessingUnit
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzExtendRestorationFrame
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzRestorationStripeBoundary
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzSaveRestorationBoundaryLines
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzSaveRestorationFrameBoundaryLines
	go test ./internal/av1/dsp $(FUZZFLAGS) -fuzz=FuzzPlaneBlockOps
	go test ./internal/av1/dsp $(FUZZFLAGS) -fuzz=FuzzMinMaxAbsDiff8x8
	go test ./internal/av1/dsp $(FUZZFLAGS) -fuzz=FuzzBlendA64Mask
	go test ./internal/av1/cdef $(FUZZFLAGS) -fuzz=FuzzFindDirection
	go test ./internal/av1/cdef $(FUZZFLAGS) -fuzz=FuzzFilterBlock
	go test ./internal/av1/cdef $(FUZZFLAGS) -fuzz=FuzzFilterFrameBlocks
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzPredictIntraPlaneBlock
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzDCPredictor
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzStaticIntraPredictors
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzFilterIntraPredictor
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzPredictDirectionalIntraPlaneBlock
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzFilterIntraEdge
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzUpsampleIntraEdge
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzIntraEdgeDecisions
	go test ./internal/av1/prediction $(FUZZFLAGS) -fuzz=FuzzCFLSubsampleAndPredict
	go test ./internal/av1/motion $(FUZZFLAGS) -fuzz=FuzzPredictInterPlaneBlock
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzBuildBatches
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchJobPayload
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchJobEntropyReader
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchJobRegion
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchLoopRestorationPlan
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchReconstructBlockCoeff
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchDecodeAndReconstructJobResiduals
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseQuantizationParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseSegmentationParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseDeltaParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseLoopFilterParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseCDEFParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseRestorationParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseTransformReferenceParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseSkipModeParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseFrameModeParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseGlobalMotionParams
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseFilmGrainParams
	go test ./internal/av1/transform $(FUZZFLAGS) -fuzz=FuzzInverseIdentityBlock
	go test ./internal/av1/transform $(FUZZFLAGS) -fuzz=FuzzInverseDCTBlock
	go test ./internal/av1/transform $(FUZZFLAGS) -fuzz=FuzzInverseBlock
	go test ./internal/av1/transform $(FUZZFLAGS) -fuzz=FuzzFillDefaultScan
	go test ./internal/av1/transform $(FUZZFLAGS) -fuzz=FuzzTXBHelpers
	go test ./internal/av1/transform $(FUZZFLAGS) -fuzz=FuzzTypeClass
	go test ./internal/av1/reconstruct $(FUZZFLAGS) -fuzz=FuzzReconstructPlaneBlock
	go test ./internal/av1/loopfilter $(FUZZFLAGS) -fuzz=FuzzResolveLevel
	go test ./internal/av1/loopfilter $(FUZZFLAGS) -fuzz=FuzzThresholdsForLevel
	go test ./internal/av1/loopfilter $(FUZZFLAGS) -fuzz=FuzzFilter4Edge
	go test ./internal/av1/loopfilter $(FUZZFLAGS) -fuzz=FuzzFilter4BlockEdge
	go test ./internal/av1/loopfilter $(FUZZFLAGS) -fuzz=FuzzFilter6Edge
	go test ./internal/av1/loopfilter $(FUZZFLAGS) -fuzz=FuzzFilter8Edge
	go test ./internal/av1/loopfilter $(FUZZFLAGS) -fuzz=FuzzFilter14Edge
	go test ./internal/av1/restoration $(FUZZFLAGS) -fuzz=FuzzApplySelfguidedRestoration
	go test ./internal/av1/restoration $(FUZZFLAGS) -fuzz=FuzzApplyWienerRestoration

testvectors:
	go test ./internal/av1/ivf ./internal/av1/obu ./internal/av1/parser ./internal/av1/rtp ./internal/av1/testvector -count=1
	go test -tags goav1_oracle ./internal/av1/ivf ./internal/av1/obu ./internal/av1/parser ./internal/av1/rtp ./internal/av1/testvector -count=1

testvectors-fast:
	go test ./internal/av1/testvector -count=1
	go test -tags goav1_oracle ./internal/av1/testvector -run 'TestLibaomQuantizer00|TestFrameMD5|TestOracleEnabled|TestLibaomRemoteManifest' -count=1

testvectors-full: testvectors
	GOAV1_FULL_LIBAOM_VECTORS=1 go test -tags goav1_oracle ./internal/av1/testvector -run TestLibaomRemoteSuiteFullDownloads -count=1

test-motion-conformance:
	GOAV1_FULL_LIBAOM_CONVOLVE=1 go test ./internal/av1/motion -run 'TestLibaomConvolve' -count=1

test-transform-conformance:
	go test ./internal/av1/transform -run 'TestInverseDCT1DMatchesLibaomReference|TestInverseADST1DRoundTripMatchesLibaomShape|TestInverseFlipADST1DReversesADST|TestInverseDCTBlockSupportedSizes|TestInverseBlockHybridTransforms|TestInverseWHT4x4Block' -count=1

alloc:
	./scripts/check_allocs.sh

sync-upstreams:
	./scripts/sync_upstreams.sh

verify-upstreams:
	./scripts/verify_upstreams.sh
