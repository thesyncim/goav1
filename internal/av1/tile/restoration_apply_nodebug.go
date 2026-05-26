//go:build !goav1_debug_wiener

package tile

import av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"

func wienerDebugPreApply(src []uint16, srcStride int, srcOrigin int, width int, height int, info av1restoration.WienerInfo) {
}

func wienerDebugPostApply(dst []uint16, dstStride int, width int, height int) {
}
