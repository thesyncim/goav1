//go:build !darwin && !linux

package goav1

// threadCPU is unavailable here; CPU-time checks skip.
func threadCPU() (uint64, bool) { return 0, false }
