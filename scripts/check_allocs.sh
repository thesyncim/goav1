#!/usr/bin/env sh
set -eu

go test ./internal/av1/bitstream ./internal/av1/obu ./internal/av1/rtp ./internal/av1/entropy ./internal/av1/parser ./internal/av1/decoder ./internal/av1/frame ./internal/av1/memory ./internal/av1/dsp ./internal/av1/prediction ./internal/av1/motion ./internal/av1/quantize ./internal/av1/transform ./internal/av1/reconstruct ./internal/av1/tile ./internal/av1/threading
