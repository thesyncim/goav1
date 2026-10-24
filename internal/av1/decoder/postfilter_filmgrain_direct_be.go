//go:build !(386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm)

package decoder

import "github.com/thesyncim/goav1/internal/av1/frame"

// frameWorkFilmGrainPlaneU16 has no in-place sample view on big-endian hosts;
// high bit depth keeps the staged film-grain path there.
func frameWorkFilmGrainPlaneU16(frame.Plane) ([]uint16, bool) { return nil, false }
