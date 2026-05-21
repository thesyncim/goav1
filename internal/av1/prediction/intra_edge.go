package prediction

const (
	intraEdgeMaxSize = 129
	intraEdgeTaps    = 5
)

var intraEdgeFilterKernels = [3][intraEdgeTaps]int{
	{0, 4, 8, 4, 0},
	{0, 5, 6, 5, 0},
	{2, 4, 4, 4, 2},
}

// FilterIntraEdge applies libaom's av1_filter_intra_edge_c smoothing kernel to
// edge in-place. scratch is caller-owned temporary storage and must be at least
// len(edge) samples when strength is non-zero.
func FilterIntraEdge(edge []uint16, scratch []uint16, strength uint8, bitDepth uint8) error {
	if len(edge) == 0 || len(edge) > intraEdgeMaxSize || strength > 3 {
		return ErrInvalidPrediction
	}
	if strength == 0 {
		return nil
	}
	max, err := intraEdgeSampleMax(bitDepth)
	if err != nil {
		return err
	}
	if err := validateSamples(edge, max); err != nil {
		return err
	}
	if len(scratch) < len(edge) {
		return ErrInvalidPrediction
	}

	copy(scratch, edge)
	kernel := intraEdgeFilterKernels[strength-1]
	for i := 1; i < len(edge); i++ {
		sum := 0
		for j := 0; j < intraEdgeTaps; j++ {
			k := i - 2 + j
			if k < 0 {
				k = 0
			} else if k >= len(edge) {
				k = len(edge) - 1
			}
			sum += int(scratch[k]) * kernel[j]
		}
		edge[i] = uint16((sum + 8) >> 4)
	}
	return nil
}

func intraEdgeSampleMax(bitDepth uint8) (uint16, error) {
	switch bitDepth {
	case 8:
		return 0xff, nil
	case 10, 12:
		return uint16((1 << bitDepth) - 1), nil
	default:
		return 0, ErrInvalidPrediction
	}
}
