//go:build !goav1_debug_cdef

package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func cdefDebugUnit(plane int, unitRow int, unitCol int) bool {
	return false
}

func cdefDebugLogUnit(plane int, unitRow int, unitCol int, packed uint8, params cdef.FrameFilterParams, dirs cdef.DirectionGrid, vars cdef.VarianceGrid, input []uint16, nBlocks int) {
}

func cdefDebugLogUnitDst(plane int, unitRow int, unitCol int, unitDst []uint16) {
}

func cdefDebugLogChromaPreLR(output *frame.Frame) {
}

func cdefDebugLogLRRecords(records [3][]tile.RestorationUnitRecord) {
}
