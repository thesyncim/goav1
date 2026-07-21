//go:build !arm64 || purego

package decoder

import (
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
)

func filterMaskLine8Available() bool { return false }

func filterLumaMaskLine8Trusted(frame.Plane, int, *[3]uint32, [][4]uint8, int, int, int, loopfilter.Edge, *lfmask.FilterLUT) {
}

func filterChromaMaskLine8Trusted(frame.Plane, int, *[2]uint32, [][4]uint8, int, int, int, loopfilter.Edge, *lfmask.FilterLUT) {
}
