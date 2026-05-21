package testvector

var coreVectors = [...]Vector{
	{
		Tag:   TagOBULowOverheadTemporalDelimiter,
		Kind:  KindOBU,
		Name:  "low-overhead temporal delimiter obu",
		Input: []byte{0x12, 0x00},
		Want:  nil,
	},
	{
		Tag:   TagRTPPayloadSingleOBU,
		Kind:  KindRTP,
		Name:  "rtp single inferred-length frame-header obu",
		Input: []byte{0x10, 0x18, 0xaa},
		Want:  []byte{0x18, 0xaa},
	},
	{
		Tag:   TagRTPPayloadFragmentedOBU,
		Kind:  KindRTP,
		Name:  "rtp starts fragmented frame obu",
		Input: []byte{0x50, 0x30, 0xaa, 0xbb},
		Want:  []byte{0x30, 0xaa, 0xbb},
	},
}

// CoreSuite returns the minimal byte-level vectors that cover the current
// transport and OBU foundations.
func CoreSuite() Suite {
	return Suite{
		Name: "goav1-core",
		Manifest: Manifest{
			Vectors: coreVectors[:],
		},
	}
}
