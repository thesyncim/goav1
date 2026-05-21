.PHONY: test bench fuzz-smoke alloc

FUZZTIME ?= 250000x
FUZZPARALLEL ?= 8
FUZZFLAGS = -run '^$$' -fuzztime=$(FUZZTIME) -parallel=$(FUZZPARALLEL)

test:
	go test ./...

bench:
	go test -run '^$$' -bench=. -benchmem ./...

fuzz-smoke:
	go test ./internal/av1/bitstream $(FUZZFLAGS) -fuzz=FuzzReadLEB128
	go test ./internal/av1/obu $(FUZZFLAGS) -fuzz=FuzzParseHeader
	go test ./internal/av1/rtp $(FUZZFLAGS) -fuzz=FuzzPayloadIterator
	go test ./internal/av1/rtp $(FUZZFLAGS) -fuzz=FuzzPacketizer
	go test ./internal/av1/rtp $(FUZZFLAGS) -fuzz=FuzzAssembleFrame
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzUpdateCDF
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzCDFState
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzReaderBinarySymbol
	go test ./internal/av1/entropy $(FUZZFLAGS) -fuzz=FuzzReaderUniformSubexp
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseSequenceHeader
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseFrameHeaderPrefix
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseIntraFrameSize
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseFrameSize
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseTileInfo
	go test ./internal/av1/parser $(FUZZFLAGS) -fuzz=FuzzParseTileGroupHeader
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzBuildJobs
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzJobPayload
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzNewEntropyReader
	go test ./internal/av1/tile $(FUZZFLAGS) -fuzz=FuzzDecodeStateReset
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzBuildBatches
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchJobPayload
	go test ./internal/av1/threading $(FUZZFLAGS) -fuzz=FuzzFrameWorkBatchJobEntropyReader
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

alloc:
	./scripts/check_allocs.sh

sync-upstreams:
	./scripts/sync_upstreams.sh

verify-upstreams:
	./scripts/verify_upstreams.sh
