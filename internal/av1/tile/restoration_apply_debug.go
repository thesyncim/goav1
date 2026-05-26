//go:build goav1_debug_wiener

package tile

import (
	"fmt"
	"os"

	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

var wienerDebugCount int

func wienerDebugPreApply(src []uint16, srcStride int, srcOrigin int, width int, height int, info av1restoration.WienerInfo) {
	if os.Getenv("GOAV1_DEBUG_WIENER") == "" {
		return
	}
	wienerDebugCount++
	// Only log first few calls
	if wienerDebugCount > 12 {
		return
	}
	fmt.Fprintf(os.Stderr, "[WIENER_PRE] call=%d w=%d h=%d vfilter=%v hfilter=%v\n",
		wienerDebugCount, width, height, info.VFilter, info.HFilter)
	// Print first 4 cols of rows -3..height+2 around the origin
	for r := -3; r < height+3; r++ {
		var vals [8]uint16
		for c := 0; c < 8; c++ {
			vals[c] = src[srcOrigin+r*srcStride+c]
		}
		fmt.Fprintf(os.Stderr, "[WIENER_PRE] call=%d r=%d cols 0-7: %v\n", wienerDebugCount, r, vals)
	}
}

func wienerDebugPostApply(dst []uint16, dstStride int, width int, height int) {
	if os.Getenv("GOAV1_DEBUG_WIENER") == "" {
		return
	}
	if wienerDebugCount > 12 {
		return
	}
	for r := 0; r < height; r++ {
		var vals [8]uint16
		for c := 0; c < 8; c++ {
			vals[c] = dst[r*dstStride+c]
		}
		fmt.Fprintf(os.Stderr, "[WIENER_POST] call=%d r=%d cols 0-7: %v\n", wienerDebugCount, r, vals)
	}
}
