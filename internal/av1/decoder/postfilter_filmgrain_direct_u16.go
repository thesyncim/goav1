package decoder

import "github.com/thesyncim/goav1/internal/av1/frame"

// frameWorkFilmGrainPlaneU16 reports that a 10/12-bit plane has no direct
// []uint16 view: the frame stores those samples as little-endian bytes, and
// viewing them as uint16s needs unsafe. High bit depth takes the staged
// film-grain path.
func frameWorkFilmGrainPlaneU16(frame.Plane) ([]uint16, bool) { return nil, false }
