package loopfilter

// Thresholds contains the scalar limits consumed by AV1 deblocking filters.
type Thresholds struct {
	Limit            uint8
	BlockLimit       uint8
	HighEdgeVariance uint8
}

// ThresholdsForLevel derives the deblocking thresholds for a loop-filter level
// and sharpness value.
func ThresholdsForLevel(level uint8, sharpness uint8) (Thresholds, error) {
	if level > MaxLevel || sharpness > MaxSharpness {
		return Thresholds{}, ErrInvalidFilter
	}

	shift := 0
	if sharpness > 0 {
		shift++
	}
	if sharpness > 4 {
		shift++
	}

	limit := int(level) >> shift
	if sharpness > 0 {
		maxLimit := 9 - int(sharpness)
		if limit > maxLimit {
			limit = maxLimit
		}
	}
	if limit < 1 {
		limit = 1
	}

	return Thresholds{
		Limit:            uint8(limit),
		BlockLimit:       uint8(2*(int(level)+2) + limit),
		HighEdgeVariance: level >> 4,
	}, nil
}
