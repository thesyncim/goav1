//go:build !(386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm)

package lfmask

import "unsafe"

// storeLevelPair writes the level bytes l0, l1 at p and p+1.
func storeLevelPair(p *uint8, l0, l1 uint8) {
	*p = l0
	*(*uint8)(unsafe.Add(unsafe.Pointer(p), 1)) = l1
}
