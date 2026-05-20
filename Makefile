.PHONY: test bench fuzz-smoke alloc

test:
	go test ./...

bench:
	go test -run '^$$' -bench=. -benchmem ./...

fuzz-smoke:
	go test ./internal/av1/bitstream -run '^$$' -fuzz=FuzzReadLEB128 -fuzztime=10s
	go test ./internal/av1/obu -run '^$$' -fuzz=FuzzParseHeader -fuzztime=10s
	go test ./internal/av1/rtp -run '^$$' -fuzz=FuzzPayloadIterator -fuzztime=10s
	go test ./internal/av1/rtp -run '^$$' -fuzz=FuzzPacketizer -fuzztime=10s
	go test ./internal/av1/rtp -run '^$$' -fuzz=FuzzAssembleFrame -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseSequenceHeader -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseFrameHeaderPrefix -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseIntraFrameSize -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseFrameSize -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseTileInfo -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseQuantizationParams -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseSegmentationParams -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseDeltaParams -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseLoopFilterParams -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseCDEFParams -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseRestorationParams -fuzztime=10s

alloc:
	./scripts/check_allocs.sh

sync-upstreams:
	./scripts/sync_upstreams.sh

verify-upstreams:
	./scripts/verify_upstreams.sh
