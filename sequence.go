package goav1

import internalparser "github.com/thesyncim/goav1/internal/av1/parser"

// SequenceHeader is the parsed AV1 sequence_header_obu(): profile, level,
// color config, operating points, decoder model, and feature enable bits.
// It is the long-lived per-coded-video-sequence parameter set.
type SequenceHeader = internalparser.SequenceHeader

// FrameType is the AV1 frame_type code (key, inter, intra-only, switch).
type FrameType = internalparser.FrameType

// FrameSize holds the AV1 frame_size() result: coded, upscaled, and render
// dimensions in samples.
type FrameSize = internalparser.FrameSize
