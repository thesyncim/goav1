package quantize

const (
	QMLevelBits = 4
	QMLevels    = 1 << QMLevelBits

	DefaultQMFirst         = 5
	DefaultQMLast          = 9
	DefaultQMFirstAllIntra = 4
	DefaultQMLastAllIntra  = 10
)

// QMLevel maps qindex to the inter quantization-matrix level range. This is
// the same increasing formula used by libaom's aom_get_qmlevel().
func QMLevel(qIndex int, first int, last int) int {
	return first + (qIndex*(last+1-first))/QIndexRange
}

// QMLevelAllIntra maps qindex to the all-intra quantization-matrix level range.
func QMLevelAllIntra(qIndex int, first int, last int) int {
	level := 4
	switch {
	case qIndex <= 40:
		level = 10
	case qIndex <= 100:
		level = 9
	case qIndex <= 160:
		level = 8
	case qIndex <= 200:
		level = 7
	case qIndex <= 220:
		level = 6
	case qIndex <= 240:
		level = 5
	}
	return clampInt(level, first, last)
}

func clampInt(v int, lo int, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
