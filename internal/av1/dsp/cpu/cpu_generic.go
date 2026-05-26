// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build !amd64 && !arm64

package cpu

// init is a no-op on architectures the DSP dispatcher does not yet know
// how to target. Detected remains zero-valued, so every kernel falls back
// to its pure-Go path. This is the correct default: bit-exact behaviour
// across every GOOS/GOARCH the toolchain supports.
func init() {}
