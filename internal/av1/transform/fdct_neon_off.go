//go:build !arm64 || purego

package transform

// forwardDCT8x8Impl is the 8x8 forward DCT kernel dispatch; without a vector
// kernel it runs the portable pass structure.
var forwardDCT8x8Impl = forwardDCT8x8PureGo
