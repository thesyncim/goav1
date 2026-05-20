.PHONY: test bench fuzz-smoke alloc

test:
	go test ./...

bench:
	go test -run '^$$' -bench=. -benchmem ./...

fuzz-smoke:
	go test ./internal/av1/bitstream -run '^$$' -fuzz=FuzzReadLEB128 -fuzztime=10s
	go test ./internal/av1/obu -run '^$$' -fuzz=FuzzParseHeader -fuzztime=10s
	go test ./internal/av1/rtp -run '^$$' -fuzz=FuzzPayloadIterator -fuzztime=10s
	go test ./internal/av1/parser -run '^$$' -fuzz=FuzzParseSequenceHeader -fuzztime=10s

alloc:
	./scripts/check_allocs.sh
