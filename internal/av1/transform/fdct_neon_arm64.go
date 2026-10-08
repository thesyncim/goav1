//go:build arm64 && !purego && !goexperiment.simd

package transform

// The default arm64 build has no Go SIMD backend; use the scalar reference.
var forwardDCT4x4Impl = forwardDCT4x4PureGo
var forwardDCT4x4Trusted8BitImpl = forwardDCT4x4PureGo
var forwardDCT8x8Impl = forwardDCT8x8PureGo
var forwardDCT8x8Trusted8BitImpl = forwardDCT8x8PureGo
var forwardDCT16x16Impl = forwardDCT16x16PureGo
var forwardDCT16x16Trusted8BitImpl = forwardDCT16x16PureGo
var forwardDCT32x32Impl = forwardDCT32x32PureGo
var forwardDCT32x32Trusted8BitImpl = forwardDCT32x32PureGo
