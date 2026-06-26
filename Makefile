.PHONY: test bench bench-all bench-public bench-cross gc-metrics compiler-reports fuzz-smoke testvectors testvectors-fast testvectors-full test-motion-conformance test-transform-conformance alloc trace-zero vet fmt-check fmt-check-strict tidy-check webrtc-reference webrtc-browser webrtc-production dryrun-fast dryrun-relevant-supported dryrun-full dryrun-extended dryrun-profiles dryrun-corpus dryrun-external-corpus ci-local help

FUZZTIME ?= 250000x
FUZZPARALLEL ?= 8
FUZZFLAGS = -run '^$$' -fuzztime=$(FUZZTIME) -parallel=$(FUZZPARALLEL)
BENCHTIME ?= 3s
GCMETRICS_COUNT ?= 5
WEBRTC_PRODUCTION_TESTS = Test(AV1SDP|AV1RTCP|EncoderWebRTC|NewDecoderFromRTPPayloads|ParseRTPPacketDependencyDescriptor|PublicDecoderFrameWorkResidual(EventRunner.*TileList|StreamRunnerRTP)|PublicDecoderRTP(Packet|PayloadRunner)|PublicEncodeI(400|420)|PublicEncoderWebRTC|PublicLayeredDecoderRTP|PublicParseTileListOBU|PublicPlanDecoderTileList|PublicResolveDecoderTileList|PublicRTC|PublicRTP|PublicTileList|PublicWebRTCEncoder|RTCP|SimpleDecoderTileListIVFPlayback)

test:
	go test ./...

# bench runs the end-to-end decoder throughput benchmarks at the repo root:
# frames/sec, MB/sec, ns/op, and the steady-state allocation guardrail.
# Pass BENCHTIME=10s (or larger) for stable numbers; the default 3s is
# enough for smoke testing.
bench:
	go test -run '^$$' -bench='^BenchmarkDecode' -benchmem -benchtime=$(BENCHTIME) .

# bench-all runs every Go testing benchmark in the repository: the
# end-to-end decoder benchmarks above plus all per-stage micro-benchmarks
# (transform, cdef, restoration, motion, prediction, public API, etc.).
bench-all:
	go test -run '^$$' -bench=. -benchmem -benchtime=$(BENCHTIME) ./...

bench-public:
	go test -run '^$$' -bench='BenchmarkPublic' -benchmem -benchtime=$(BENCHTIME) .

# bench-cross is a PERF-TRACKING tool (not a conformance gate). It times
# goav1's full decode + post-filter chain (reusing the MD5-verifying oracle
# harness) against the reference C decoders found on PATH: libaom's aomdec,
# dav1d, and SVT-AV1's SvtAv1DecApp if present. Missing decoders are skipped,
# never a hard failure. goav1 is timed in-process while the C decoders run as
# subprocesses, so on the tiny bundled clips the raw numbers are startup-
# dominated; the report prints both raw and startup-adjusted estimates with
# the caveats made impossible to miss. Needs the goav1_oracle build tag so it
# can reach the oracle decode helper.
bench-cross:
	GOAV1_CROSS_BENCH=1 go test -tags goav1_oracle -run TestCrossDecoderThroughput ./internal/av1/testvector -v -count=1 -timeout 600s

gc-metrics:
	GODEBUG=gctrace=1 go test -run '^$$' -bench='BenchmarkDecode.*GCMetrics|BenchmarkDecodeFullVectorAllocs' -benchmem -count=$(GCMETRICS_COUNT) .

compiler-reports:
	./scripts/check_compiler_reports.sh

fuzz-smoke:
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicSimpleDecoderIVF
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicDecodeTileBlockCoefficients
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicParseTileListOBU
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicDecodeAndReconstructDecoderFrameWorkBlockCoefficients
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicDecodeAndReconstructDecoderFrameWorkJobResiduals
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicDecodeAndRetainDecoderFrameWorkBatchResiduals
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicDecoderFrameWorkBatchResidualRunnerSideData
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicRunDecoderFrameWorkEventWithResidualRunner
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicDecoderFrameWorkSideDataBinding
	go test . $(FUZZFLAGS) -fuzz=FuzzPublicDecoderFrameWorkPostFilterScratchContext
	go test ./internal/av1/bitstream $(FUZZFLAGS) -fuzz=FuzzReadLEB128
	go test ./internal/av1/decoder $(FUZZFLAGS) -fuzz=FuzzStreamPush
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
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseTileListOBU
	go test ./internal/av1/quantize $(FUZZFLAGS) -fuzz='^FuzzDequantizeBlock$$'
	go test ./internal/av1/quantize $(FUZZFLAGS) -fuzz=FuzzDequantizeBlockScaledQMatrix
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
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadMotionVector
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReferenceMVStack
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzBuildReferenceMVStack
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadMotionMode
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadInterIntra
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadCompoundBlend
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadCoeffPrimitives
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadCoefficientsTXB
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzCoeffEntropyContext
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeLumaCoefficients
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeChromaCoefficients
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeBlockCoefficients
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzReadPaletteMode
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzPaletteColorIndexContext
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzPaletteCache
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeStateRestorationUnit
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationUnitNone
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationUnitRecordNone
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationUnitRecordWithBoundaries
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationPlaneRecords
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzApplyRestorationFramePlane
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz='^FuzzApplyRestorationFrame$$'
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
	go test ./internal/av1/motion $(FUZZFLAGS) -fuzz='^FuzzPredictInterPlaneBlock$$'
	go test ./internal/av1/motion $(FUZZFLAGS) -fuzz=FuzzPredictInterPlaneBlockFromOriginFullpel
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzBuildBatches
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchJobPayload
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchJobEntropyReader
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchJobRegion
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchLoopRestorationPlan
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchReconstructBlockCoeff
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchDecodeAndReconstructJobResiduals
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkDistanceWeightedCompoundOffsets
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchPredictBlockLumaIntra
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

trace-zero:
	./scripts/check_trace_zero.sh

vet:
	go vet ./...

fmt-check:
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt: the following files would be reformatted:"; \
		echo "$$unformatted"; \
		gofmt -d $$unformatted || true; \
	else \
		echo "gofmt: clean"; \
	fi

fmt-check-strict:
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt: the following files need formatting:" >&2; \
		echo "$$unformatted" >&2; \
		gofmt -d $$unformatted >&2 || true; \
		exit 1; \
	fi

tidy-check:
	@cp go.mod go.mod.orig; \
	[ -f go.sum ] && cp go.sum go.sum.orig || true; \
	go mod tidy; \
	rc=0; \
	diff -u go.mod.orig go.mod || rc=1; \
	if [ -f go.sum.orig ]; then diff -u go.sum.orig go.sum || rc=1; fi; \
	mv go.mod.orig go.mod; \
	[ -f go.sum.orig ] && mv go.sum.orig go.sum || true; \
	if [ $$rc -ne 0 ]; then echo "go.mod/go.sum is not tidy; run 'go mod tidy'" >&2; fi; \
	exit $$rc

webrtc-reference:
	GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS=1 go test . -run 'TestPublicRTCEncoder.*ReferenceDecoders$$' -count=1 -timeout 900s -v

webrtc-browser:
	GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS=1 go -C examples/browser-push test . -run TestEndToEndAV1OverRTP -count=1 -timeout 180s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveAV1PlaybackStats -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPPlaybackStats -count=1 -timeout 240s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPRepeatedPlaybackSoak -count=1 -timeout 180s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPControlChurnPlayback -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPReceiverEstimatedMaximumBitrateControl -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPTransportWideCCFeedback -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPImpairmentFeedback -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPNACKRetransmission -count=1 -timeout 120s -v

webrtc-production:
	GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS=1 go test . -run '$(WEBRTC_PRODUCTION_TESTS)' -count=1 -timeout 1200s -v
	GOAV1_REQUIRE_WEBRTC_REFERENCE_DECODERS=1 go -C examples/browser-push test . -run TestEndToEndAV1OverRTP -count=1 -timeout 180s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveAV1PlaybackStats -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPPlaybackStats -count=1 -timeout 240s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPRepeatedPlaybackSoak -count=1 -timeout 180s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPControlChurnPlayback -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPReceiverEstimatedMaximumBitrateControl -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPTransportWideCCFeedback -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPImpairmentFeedback -count=1 -timeout 120s -v
	GOAV1_REQUIRE_WEBRTC_BROWSER=1 go -C examples/browser-push test . -run TestBrowserLiveRTCEncoderDirectRTPNACKRetransmission -count=1 -timeout 120s -v

dryrun-fast:
	GOAV1_FAST_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 go test -tags goav1_oracle ./internal/av1/testvector -run 'TestLibaomFastFrameWorkDryRun' -count=1 -timeout 600s -v

# dryrun-relevant-supported runs the strict-MD5 SuiteLevelRelevant framework
# dry-run cohort.
dryrun-relevant-supported:
	GOAV1_RELEVANT_SUPPORTED_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 go test -tags goav1_oracle ./internal/av1/testvector -run 'TestLibaomRelevantSupportedFrameWorkDryRun' -count=1 -timeout 900s -v

# dryrun-full runs every checksum-pinned vector in the committed libaom
# manifest through the strict-MD5 framework dry-run. It is heavier than
# dryrun-extended because it also includes the relevant-supported cohort.
dryrun-full:
	GOAV1_FULL_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 go test -tags goav1_oracle ./internal/av1/testvector -run 'TestLibaomFullFrameWorkDryRun' -count=1 -timeout 2400s -v

# dryrun-profiles decodes the vendored profile-conformance clips (profile-0
# 4:2:0 S_FRAME, profile-1 4:4:4 8/10-bit, profile-1 4:4:4 palette/CDEF/restoration/edge/tile/super-res,
# profile-1 10-bit inter tile and super-res+restoration, profile-2 4:2:2
# 8/10/12-bit plus 4:2:0/4:4:4 12-bit edge/super-res clips) and
# asserts byte-exact per-frame MD5 against their libaom goldens.
# The test lives in its own package (internal/av1/testvector/profiles) so it
# compiles into a separate test binary from the oracle suite and cannot share
# process state with the fast/extended dry-runs.
dryrun-profiles:
	go test -tags goav1_oracle ./internal/av1/testvector/profiles -run 'TestProfileVendoredClips' -count=1 -timeout 600s -v

# dryrun-corpus runs the generated real-content corpus through a single
# byte-exact stream-MD5 verification pass. Generate the local ignored corpus
# first with scripts/gen_bench_corpus.sh; this target skips cleanly when no
# corpus is present.
dryrun-corpus:
	GOAV1_CORPUS_CONFORMANCE=1 go test -tags goav1_oracle ./internal/av1/testvector -run 'TestGeneratedCorpusConformance' -count=1 -timeout 900s -v

# dryrun-external-corpus recursively scans GOAV1_EXTERNAL_CORPUS_DIR or
# GOAV1_EXTERNAL_CORPUS_DIRS for local third-party IVF corpora. Each .ivf must
# have a supported stream-MD5 or per-frame MD5 sidecar.
dryrun-external-corpus:
	GOAV1_EXTERNAL_CORPUS=1 go test -tags goav1_oracle ./internal/av1/testvector -run 'TestExternalCorpusConformance' -count=1 -timeout 2400s -v

# dryrun-extended runs the opt-in SuiteLevelExtended framework dry-run cohort
# under strict per-frame MD5. It downloads multi-quantizer 10-bit, additional
# SVC, and larger-size libaom vectors. This target is never part of CI or the
# default test gates; it is a local diagnostic for surfacing latent decoder gaps.
dryrun-extended:
	GOAV1_EXTENDED_LIBAOM_FRAMEWORK_DRYRUN=1 GOAV1_STRICT_MD5=1 go test -tags goav1_oracle ./internal/av1/testvector -run 'TestLibaomExtendedFrameWorkDryRun' -count=1 -timeout 1800s -v

ci-local: fmt-check vet test alloc compiler-reports trace-zero

# CMDDIR is the source directory of the aom-go-dec CLI. CMDBIN is the path
# `build-cmd` writes the binary to (bin/aom-go-dec by default). Override
# either on the command line, e.g. `make build-cmd CMDBIN=/tmp/aom-go-dec`.
CMDDIR ?= ./cmd/aom-go-dec
CMDBIN ?= bin/aom-go-dec

# build-cmd compiles the aom-go-dec CLI into ./bin so it is easy to run
# against arbitrary IVF files without polluting the module cache.
build-cmd:
	@mkdir -p $(dir $(CMDBIN))
	go build -o $(CMDBIN) $(CMDDIR)
	@echo "built $(CMDBIN)"

# install-cmd installs the aom-go-dec CLI into $GOBIN (or $GOPATH/bin) using
# the standard `go install` flow so it lands on $PATH for users who have
# their Go bin directory configured.
install-cmd:
	go install $(CMDDIR)

help:
	@echo "Available targets:"
	@echo "  test                       go test ./..."
	@echo "  bench                      end-to-end frames/sec + MB/sec on bundled libaom IVF"
	@echo "  bench-all                  full microbenchmark sweep across every package"
	@echo "  bench-public               run public benchmarks"
	@echo "  bench-cross                goav1 vs aomdec/dav1d/SVT throughput (perf tool, startup-aware)"
	@echo "  gc-metrics                 decode GC scan/object-count benchmarks"
	@echo "  compiler-reports           fail on new hot-package heap escapes; report BCE sites"
	@echo "  alloc                      run allocation regression checks"
	@echo "  trace-zero                 prove entropy tracing compiles out of release hot paths"
	@echo "  vet                        go vet ./..."
	@echo "  fmt-check                  report any files gofmt would reformat (non-blocking)"
	@echo "  fmt-check-strict           fail if gofmt would reformat anything"
	@echo "  tidy-check                 fail if go.mod/go.sum is not tidy"
	@echo "  webrtc-reference           require aomdec+dav1d for WebRTC encoder reference-decode matrix"
	@echo "  webrtc-browser             require aomdec+dav1d for the browser-push WebRTC example"
	@echo "  webrtc-production          strict WebRTC encoder/decoder/RTP/RTCP/browser gate with reference decoders"
	@echo "  fuzz-smoke                 short fuzz sweep across packages"
	@echo "  testvectors                committed test-vector suite (with oracle)"
	@echo "  testvectors-fast           fast slice of the test-vector suite"
	@echo "  testvectors-full           full libaom remote suite (downloads vectors)"
	@echo "  dryrun-fast                strict-MD5 fast libaom framework dry-run"
	@echo "  dryrun-relevant-supported  strict-MD5 relevant dry-run"
	@echo "  dryrun-full                strict-MD5 all committed libaom framework vectors"
	@echo "  dryrun-extended            strict-MD5 opt-in extended dry-run (10-bit q-sweep, larger sizes, extra SVC)"
	@echo "  dryrun-profiles            byte-exact profile clips (profile-0 S_FRAME, profile-1 4:4:4 incl. palette/filters/super-res, profile-2 4:2:2/12-bit)"
	@echo "  dryrun-corpus              byte-exact generated real-content corpus (requires local generated corpus)"
	@echo "  dryrun-external-corpus     byte-exact local external IVF corpus (requires GOAV1_EXTERNAL_CORPUS_DIR(S))"
	@echo "  test-motion-conformance    libaom convolve conformance"
	@echo "  test-transform-conformance libaom transform conformance"
	@echo "  ci-local                   run fmt-check + vet + test + alloc + compiler-reports + trace-zero"
	@echo "  build-cmd                  build the aom-go-dec CLI into ./bin"
	@echo "  install-cmd                go install the aom-go-dec CLI"

sync-upstreams:
	./scripts/sync_upstreams.sh

verify-upstreams:
	./scripts/verify_upstreams.sh
