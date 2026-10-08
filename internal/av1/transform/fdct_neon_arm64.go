//go:build arm64 && !purego && !goexperiment.simd

package transform

// The default arm64 build has no Go SIMD backend; use the scalar reference.
var forwardDCT4x4Impl = forwardDCT4x4PureGo
var forwardDCT4x4Trusted8BitImpl = forwardDCT4x4PureGo
var forwardDCT8x8Impl = forwardDCT8x8PureGo
var forwardDCT8x8Trusted8BitImpl = forwardDCT8x8PureGo
