// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

//go:build arm64 && !purego

package transform

// These kernels transform four adjacent columns in parallel. Their adapters
// enforce the buffer and stage-range preconditions documented in the assembly.
//
//go:noescape
func inverseDCT8Col4NEON(base *int32, rowStrideBytes, min, max int64)

//go:noescape
func inverseDCT16Col4NEON(base *int32, rowStrideBytes, min, max int64)

//go:noescape
func inverseADST8Col4NEON(base *int32, rowStrideBytes, min, max int64)
