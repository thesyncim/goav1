// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build amd64

package cpu

// init populates Detected on amd64 builds.
//
// Today this is a conservative stub: the Go toolchain guarantees a working
// SSE2 unit on every amd64 target (GOAMD64=v1 baseline), so we report SSE2
// unconditionally. Detecting higher feature levels (SSE4.1/4.2, AVX2,
// AVX512) requires CPUID, which means either a Go assembly stub or pulling
// in golang.org/x/sys/cpu. The dispatch layer is wired so that adding those
// detections is a one-spot change here — the kernel variants and the
// dispatcher already check the corresponding Features fields.
//
// TODO: replace this stub with proper CPUID-based detection. A small
// assembly helper (cpuid_amd64.s) is sufficient and keeps us inside the
// stdlib.
func init() {
	Detected.SSE2 = true
}
