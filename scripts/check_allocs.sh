#!/usr/bin/env sh
set -eu

go test ./internal/av1/bitstream ./internal/av1/obu ./internal/av1/rtp ./internal/av1/memory
