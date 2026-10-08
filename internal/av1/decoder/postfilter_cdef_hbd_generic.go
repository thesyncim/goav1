//go:build !(386 || amd64 || arm || arm64 || loong64 || mips64le || mipsle || ppc64le || riscv64 || wasm)

package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// frameWorkCDEFHBDInPlaceAvailable keeps the portable byte-order-independent
// snapshot path active where a byte plane cannot be safely viewed as native
// uint16 samples.
func frameWorkCDEFHBDInPlaceAvailable() bool { return false }

func frameWorkCDEFHBDPlaneViewAvailable(frame.Plane) bool { return false }

func frameWorkApplyCDEFPlaneRowsHBD(params parser.CDEFParams, indexMap FrameWorkCDEFIndexMap, skipMap *FrameWorkLoopFilterMap, cols, rows, rowStart, rowEnd int, dst frame.Plane, lineScratch []uint16, boundary FrameWorkCDEFPostFilterHBD16BandBoundary, input []uint16, blockStorage []cdef.BlockPosition, directions *cdef.DirectionGrid, variances *cdef.VarianceGrid, directionGrid []cdef.DirectionGrid, varianceGrid []cdef.VarianceGrid, plane, xDec, yDec, coeffShift int, forceLumaDirections bool) (uint32, uint32, error) {
	return 0, 0, frame.ErrInvalidFormat
}
