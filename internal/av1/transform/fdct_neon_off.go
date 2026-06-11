//go:build !arm64 || purego

package transform

// forwardDCT8x8Impl is the 8x8 forward DCT kernel dispatch; without a vector
// kernel it runs the portable pass structure.
var forwardDCT8x8Impl = forwardDCT8x8PureGo

// forwardDCT4x4Impl is the 4x4 forward DCT kernel dispatch.
var forwardDCT4x4Impl = forwardDCT4x4PureGo

// forwardDCT16x16Impl is the 16x16 forward DCT kernel dispatch.
var forwardDCT16x16Impl = forwardDCT16x16PureGo
