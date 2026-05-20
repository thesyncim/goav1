.PHONY: test bench fuzz-smoke alloc

test:
	go test ./...

bench:
	go test -run '^$$' -bench=. -benchmem ./...

fuzz-smoke:
	go test ./internal/av1/bitstream -run '^$$' -fuzz=FuzzReadLEB128 -fuzztime=20s
	go test ./internal/av1/obu -run '^$$' -fuzz=FuzzParseHeader -fuzztime=20s
	go test ./internal/av1/rtp -run '^$$' -fuzz=FuzzPayloadIterator -fuzztime=20s
	go test ./internal/av1/rtp -run '^$$' -fuzz=FuzzPacketizer -fuzztime=20s
	go test ./internal/av1/rtp -run '^$$' -fuzz=FuzzAssembleFrame -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseSequenceHeader -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseFrameHeaderPrefix -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseIntraFrameSize -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseFrameSize -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseTileInfo -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseTileGroupHeader -fuzztime=20s
	go test ./internal/av1/tile -run '^$$' -fuzz=FuzzBuildJobs -fuzztime=20s
	go test ./internal/av1/threading -run '^$$' -fuzz=FuzzBuildBatches -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseQuantizationParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseSegmentationParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseDeltaParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseLoopFilterParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseCDEFParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseRestorationParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseTransformReferenceParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseSkipModeParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseFrameModeParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseGlobalMotionParams -fuzztime=20s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseFilmGrainParams -fuzztime=20s

alloc:
	./scripts/check_allocs.sh

sync-upstreams:
	./scripts/sync_upstreams.sh

verify-upstreams:
	./scripts/verify_upstreams.sh
