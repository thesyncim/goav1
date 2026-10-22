package encoder

// SetCDEFDecisionTestHook installs (nil removes) the searched-picture CDEF
// decision rewrite for external tests.
func SetCDEFDecisionTestHook(h func(bits *uint8, y, uv *[MaxCDEFStrengths]uint8, fbIdx []uint8)) {
	if h == nil {
		cdefDecisionTestHook = nil
		return
	}
	cdefDecisionTestHook = func(p *CDEFParams, fbIdx []uint8) {
		h(&p.Bits, &p.YStrength, &p.UVStrength, fbIdx)
	}
}
