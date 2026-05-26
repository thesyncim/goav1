// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package bitstream contains byte-level AV1 bitstream primitives.
//
// The package is deliberately small and allocation-free. Higher-level parser
// packages own interpretation and state; this package only handles primitive
// encodings that must be shared across OBU, RTP, entropy, and container code.
package bitstream
