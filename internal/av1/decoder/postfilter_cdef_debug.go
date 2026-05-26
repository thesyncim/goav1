//go:build goav1_debug_cdef

package decoder

import (
	"fmt"
	"os"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func cdefDebugLogLRRecords(records [3][]tile.RestorationUnitRecord) {
	if os.Getenv("GOAV1_DEBUG_CDEF_UNIT") == "" {
		return
	}
	for plane := 0; plane < 3; plane++ {
		for i, rec := range records[plane] {
			fmt.Fprintf(os.Stderr, "[LR_REC] plane=%d unit=%d rect=%+v type=%d wiener_v=%v wiener_h=%v\n",
				plane, i, rec.Rect, rec.Unit.Type, rec.Unit.Wiener.VFilter, rec.Unit.Wiener.HFilter)
		}
	}
}

func cdefDebugLogChromaPreLR(output *frame.Frame) {
	if output == nil {
		return
	}
	if os.Getenv("GOAV1_DEBUG_CDEF_UNIT") == "" {
		return
	}
	u := output.U
	if len(u.Pix) == 0 {
		return
	}
	for r := 20; r < 36; r++ {
		vals := make([]byte, 8)
		for c := 0; c < 8 && c < u.Width; c++ {
			vals[c] = u.Pix[r*u.Stride+c]
		}
		fmt.Fprintf(os.Stderr, "[LR_PRE] U r=%d cols 0-7: %v\n", r, vals)
	}
}

func cdefDebugUnit(plane int, unitRow int, unitCol int) bool {
	return os.Getenv("GOAV1_DEBUG_CDEF_UNIT") != "" && unitRow == 0 && unitCol == 0
}

func cdefDebugLogUnitDst(plane int, unitRow int, unitCol int, unitDst []uint16) {
	if plane != 1 && plane != 2 {
		return
	}
	if os.Getenv("GOAV1_DEBUG_CDEF_UNIT") == "" || unitRow != 0 || unitCol != 0 {
		return
	}
	for r := 28; r < 32; r++ {
		vals := make([]uint16, 12)
		for c := 0; c < 12; c++ {
			vals[c] = unitDst[r*cdef.BStride+c]
		}
		fmt.Fprintf(os.Stderr, "[CDEF] plane=%d unitDst r=%d cols 0-11: %v\n", plane, r, vals)
	}
}

func cdefDebugLogUnit(plane int, unitRow int, unitCol int, packed uint8, params cdef.FrameFilterParams, dirs cdef.DirectionGrid, vars cdef.VarianceGrid, input []uint16, nBlocks int) {
	fmt.Fprintf(os.Stderr, "[CDEF] plane=%d unit=(%d,%d) packed=%d level=%d sec=%d damping=%d coeffShift=%d xDec=%d yDec=%d blocks=%d\n",
		plane, unitRow, unitCol, packed, params.Level, params.SecondaryStrength, params.Damping, params.CoeffShift, params.XDec, params.YDec, nBlocks)
	// dump directions for plane=Y unit
	if plane == 0 {
		fmt.Fprintf(os.Stderr, "[CDEF] plane=0 directions row 6: %v\n", dirs[6])
		fmt.Fprintf(os.Stderr, "[CDEF] plane=0 directions row 7: %v\n", dirs[7])
		fmt.Fprintf(os.Stderr, "[CDEF] plane=0 variances row 6: %v\n", vars[6])
	}
	if plane == 1 || plane == 2 {
		// Print input values at block by=7 (rows 28-31), col 0..7
		fmt.Fprintf(os.Stderr, "[CDEF] plane=%d directions used row 6: %v\n", plane, dirs[6])
		fmt.Fprintf(os.Stderr, "[CDEF] plane=%d directions used row 7: %v\n", plane, dirs[7])
		origin := cdef.VerticalBorder*cdef.BStride + cdef.HorizontalBorder
		for r := 28; r < 34; r++ {
			vals := make([]uint16, 12)
			for c := 0; c < 12; c++ {
				vals[c] = input[origin+r*cdef.BStride+c]
			}
			fmt.Fprintf(os.Stderr, "[CDEF] plane=%d input r=%d cols 0-11: %v\n", plane, r, vals)
		}
	}
}
