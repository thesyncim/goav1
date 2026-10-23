//go:build 386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm

package lfmask

import "unsafe"

// storeLevelPair writes the level bytes l0, l1 at p and p+1 as one 16-bit
// store (little-endian byte order), the store dav1d's adjacent uint8 writes
// compile to.
func storeLevelPair(p *uint8, l0, l1 uint8) {
	*(*uint16)(unsafe.Pointer(p)) = uint16(l0) | uint16(l1)<<8
}
