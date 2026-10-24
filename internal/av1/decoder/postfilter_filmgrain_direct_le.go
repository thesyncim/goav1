//go:build 386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm

package decoder

import (
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// frameWorkFilmGrainPlaneU16 returns a uint16 sample view of a little-endian
// high-bit-depth plane, as dav1d's pixel pointers. The view is used
// synchronously and never stored.
func frameWorkFilmGrainPlaneU16(plane frame.Plane) ([]uint16, bool) {
	if len(plane.Pix)%2 != 0 || plane.Stride%2 != 0 {
		return nil, false
	}
	if len(plane.Pix) == 0 {
		return nil, true
	}
	base := unsafe.SliceData(plane.Pix)
	if uintptr(unsafe.Pointer(base))%unsafe.Alignof(uint16(0)) != 0 {
		return nil, false
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(base)), len(plane.Pix)/2), true
}
