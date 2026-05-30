// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build arm64 && !purego

package superres

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the NEON super-res row kernel on arm64 when NEON is available
// (always, on the targets Go's arm64 backend runs on). The pure-Go reference
// remains the fallback for edge columns and degenerate shapes.
func init() {
	if cpu.Detected.NEON {
		upscaleRowImpl = upscaleRowNEON
	} else {
		upscaleRowImpl = upscaleRowPureGo
	}
}
