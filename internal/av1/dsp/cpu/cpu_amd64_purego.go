// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64 && purego

package cpu

// init populates Detected on the amd64 purego build, which excludes the CPUID
// assembly helper. Only the GOAMD64=v1 SSE2 baseline is reported; the purego
// build runs pure-Go kernels regardless, so no higher feature level is probed.
func init() {
	Detected.SSE2 = true
}
