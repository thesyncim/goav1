#!/usr/bin/env sh
set -eu

go test ./internal/av1/bitstream ./internal/av1/obu ./internal/av1/rtp ./internal/av1/entropy ./internal/av1/parser ./internal/av1/decoder ./internal/av1/frame ./internal/av1/memory ./internal/av1/dsp ./internal/av1/tile ./internal/av1/threading
