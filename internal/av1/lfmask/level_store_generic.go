//go:build !(386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm)

package lfmask

import "unsafe"

// storeLevelPair writes the level pair v (low byte first) at p and p+1.
func storeLevelPair(p *uint8, v uint16) {
	*p = uint8(v)
	*(*uint8)(unsafe.Add(unsafe.Pointer(p), 1)) = uint8(v >> 8)
}
