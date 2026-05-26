// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

// Package rtp implements the AV1 RTP payload format.
//
// The code here operates on RTP payload bytes only. RTP packet headers,
// retransmission, jitter buffering, and SRTP live at integration boundaries.
package rtp
