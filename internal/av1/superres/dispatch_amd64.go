// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && !purego

package superres

import "github.com/thesyncim/goav1/internal/av1/dsp/cpu"

// init binds the architecture-best super-res row kernel on amd64. When the CPU
// advertises AVX2 (and the toolchain emits valid AVX2 — see cpu.Detected) the
// AVX2 polyphase kernel handles the in-bounds central span; otherwise the
// pure-Go reference is used. Both paths are bit-identical to upscaleRowPureGo;
// the AVX2 path falls back to the pure-Go reference for boundary-clamped edge
// pixels and degenerate shapes.
func init() {
	if cpu.Detected.AVX2 {
		upscaleRowImpl = upscaleRowAVX2
	} else {
		upscaleRowImpl = upscaleRowPureGo
	}
}
