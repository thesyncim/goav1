// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package parser implements the higher-level AV1 syntax parsers that sit on
// top of the OBU framing primitives in package obu.
//
// It covers the uncompressed-header and metadata syntax of the AV1
// specification: sequence headers (spec §5.5), frame and frame-size headers
// (spec §5.9), tile information and tile-group headers (spec §5.11.1, §5.11.3),
// and the per-frame configuration syntax for quantization, segmentation, delta
// quantizer/loop-filter, CDEF, loop restoration, loop filter, reference frames,
// skip mode, global motion, and film grain (spec §5.9.x). The element-level
// reads mirror libaom's av1/decoder/obu.c and decodeframe.c read order so the
// bit positions stay byte-exact against the reference decoder.
//
// Parsers in this package are stateless with respect to reconstruction: they
// decode syntax into plain value structs and leave prediction, transform, and
// post-filter stages to the decoder packages above them.
package parser
