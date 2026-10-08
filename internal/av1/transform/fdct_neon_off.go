//go:build !arm64 || purego

package transform

// forwardDCT8x8Impl is the 8x8 forward DCT kernel dispatch; without a vector
// kernel it runs the portable pass structure.
var forwardDCT8x8Impl = forwardDCT8x8PureGo

// These kernels are selected by ForwardBlock8BitResidualTrusted. Builds
// without a narrow vector backend use the scalar implementation directly.
var forwardDCT8x8Trusted8BitImpl = forwardDCT8x8PureGo

// forwardDCT4x4Impl is the 4x4 forward DCT kernel dispatch.
var forwardDCT4x4Impl = forwardDCT4x4PureGo
var forwardDCT4x4Trusted8BitImpl = forwardDCT4x4PureGo

// forwardDCT16x16Impl is the 16x16 forward DCT kernel dispatch.
var forwardDCT16x16Impl = forwardDCT16x16PureGo
var forwardDCT16x16Trusted8BitImpl = forwardDCT16x16PureGo

// forwardDCT32x32Impl is the 32x32 forward DCT kernel dispatch.
var forwardDCT32x32Impl = forwardDCT32x32PureGo
var forwardDCT32x32Trusted8BitImpl = forwardDCT32x32PureGo
