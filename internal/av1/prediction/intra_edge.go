package prediction

const (
	intraEdgeMaxSize         = 129
	intraEdgeMaxUpsampleSize = 16
	intraEdgeTaps            = 5
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

// UpsampleIntraEdge applies libaom's av1_upsample_intra_edge_c interpolation
// to edge in-place. origin identifies C's p[0] inside edge; edge[origin-1] is
// p[-1], and the function writes p[-2] through p[2*size-2].
func UpsampleIntraEdge(edge []uint16, origin int, size int, scratch []uint16, bitDepth uint8) error {
	if size <= 0 || size > intraEdgeMaxUpsampleSize || origin < 2 {
		return ErrInvalidPrediction
	}
	if origin > len(edge)-size || origin > len(edge)-(2*size-1) {
		return ErrInvalidPrediction
	}
	if len(scratch) < size+3 {
		return ErrInvalidPrediction
	}
	max, err := intraEdgeSampleMax(bitDepth)
	if err != nil {
		return err
	}
	if err := validateSamples(edge[origin-1:origin+size], max); err != nil {
		return err
	}

	scratch[0] = edge[origin-1]
	scratch[1] = edge[origin-1]
	copy(scratch[2:2+size], edge[origin:origin+size])
	lastSource := origin + size - 1
	scratch[size+2] = edge[lastSource]

	edge[origin-2] = scratch[0]
	for i := 0; i < size; i++ {
		sum := -int(scratch[i]) + 9*int(scratch[i+1]) + 9*int(scratch[i+2]) - int(scratch[i+3])
		sample := (sum + 8) >> 4
		if sample < 0 {
			sample = 0
		} else if sample > int(max) {
			sample = int(max)
		}
		edge[origin+2*i-1] = uint16(sample)
		edge[origin+2*i] = scratch[i+2]
	}
	return nil
}

// IntraEdgeFilterStrength matches libaom's intra_edge_filter_strength helper.
func IntraEdgeFilterStrength(blockSize0 int, blockSize1 int, delta int, smoothNeighbor bool) uint8 {
	d := absInt(delta)
	blockWH := blockSize0 + blockSize1
	strength := uint8(0)
	if !smoothNeighbor {
		if blockWH <= 8 {
			if d >= 56 {
				strength = 1
			}
		} else if blockWH <= 12 {
			if d >= 40 {
				strength = 1
			}
		} else if blockWH <= 16 {
			if d >= 40 {
				strength = 1
			}
		} else if blockWH <= 24 {
			if d >= 8 {
				strength = 1
			}
			if d >= 16 {
				strength = 2
			}
			if d >= 32 {
				strength = 3
			}
		} else if blockWH <= 32 {
			if d >= 1 {
				strength = 1
			}
			if d >= 4 {
				strength = 2
			}
			if d >= 32 {
				strength = 3
			}
		} else if d >= 1 {
			strength = 3
		}
		return strength
	}

	if blockWH <= 8 {
		if d >= 40 {
			strength = 1
		}
		if d >= 64 {
			strength = 2
		}
	} else if blockWH <= 16 {
		if d >= 20 {
			strength = 1
		}
		if d >= 48 {
			strength = 2
		}
	} else if blockWH <= 24 {
		if d >= 4 {
			strength = 3
		}
	} else if d >= 1 {
		strength = 3
	}
	return strength
}

// UseIntraEdgeUpsample matches libaom's av1_use_intra_edge_upsample helper.
func UseIntraEdgeUpsample(blockSize0 int, blockSize1 int, delta int, smoothNeighbor bool) bool {
	d := absInt(delta)
	blockWH := blockSize0 + blockSize1
	if d == 0 || d >= 40 {
		return false
	}
	if smoothNeighbor {
		return blockWH <= 8
	}
	return blockWH <= 16
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

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
