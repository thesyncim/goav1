//go:build 386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm

package lfmask

import "unsafe"

// storeLevelPair writes the level pair v (low byte first) at p and p+1 as one
// 16-bit store (little-endian byte order), the store dav1d's adjacent uint8
// writes compile to.
func storeLevelPair(p *uint8, v uint16) {
	*(*uint16)(unsafe.Pointer(p)) = v
}
