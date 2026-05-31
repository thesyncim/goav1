//go:build goav1_debug_refmv

package tile

import (
	"fmt"
	"os"
)

var debugRefMVEnabled = os.Getenv("GOAV1_DEBUG_REFMV") != ""

func debugRefMVPrintf(req ReferenceMVStackRequest, format string, args ...any) {
	if req.MIRow != 34 || req.MICol != 4 {
		return
	}
	fmt.Fprintf(os.Stderr, "REFMV mi=(%d,%d) X4=%d Y4=%d size=%d: ", req.MICol, req.MIRow, req.X4, req.Y4, req.Size)
	fmt.Fprintf(os.Stderr, format, args...)
	fmt.Fprintln(os.Stderr)
}

func debugReferenceMVStack(req ReferenceMVStackRequest, result *ReferenceMVStackResult) {
	debugRefMVPrintf(req, "STACK count=%d row=%d col=%d ctx=%d nearest=%d", result.Stack.Count, result.RowMatches, result.ColumnMatches, result.ModeContext, result.NearestCount)
	for i := 0; i < result.Stack.Count; i++ {
		debugRefMVPrintf(req, "  stack[%d] this=(%d,%d) w=%d", i, result.Stack.Candidates[i].This.Row, result.Stack.Candidates[i].This.Col, result.Stack.Candidates[i].Weight)
	}
}
