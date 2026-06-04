// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package rtp

import "errors"

var (
	ErrShortPayload                = errors.New("rtpav1: short payload")
	ErrReservedBit                 = errors.New("rtpav1: reserved bit set")
	ErrInvalidAggregationHeader    = errors.New("rtpav1: invalid aggregation header")
	ErrInvalidElementCount         = errors.New("rtpav1: invalid element count")
	ErrShortBuffer                 = errors.New("rtpav1: short buffer")
	ErrZeroLengthElement           = errors.New("rtpav1: zero-length OBU element")
	ErrUnexpectedContinuation      = errors.New("rtpav1: unexpected OBU continuation")
	ErrFragmentInterrupted         = errors.New("rtpav1: fragmented OBU interrupted")
	ErrEmptyFrame                  = errors.New("rtpav1: empty frame")
	ErrMTUTooSmall                 = errors.New("rtpav1: MTU too small")
	ErrInvalidPayloadLimits        = errors.New("rtpav1: invalid payload limits")
	ErrOBUBufferTooSmall           = errors.New("rtpav1: OBU scratch buffer too small")
	ErrPacketPlanTooSmall          = errors.New("rtpav1: packet plan scratch buffer too small")
	ErrInvalidDependencyDescriptor = errors.New("rtpav1: invalid dependency descriptor")
)
