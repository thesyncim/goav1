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
		Tag:   TagOBUAnnexBTemporalUnit,
		Kind:  KindAnnexB,
		Name:  "annex b temporal unit with two frame units",
		Input: []byte{0x0a, 0x05, 0x01, 0x10, 0x02, 0x08, 0xaa, 0x03, 0x02, 0x18, 0xbb},
		Want:  []byte{0x10, 0x08, 0xaa, 0x18, 0xbb},
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
	{
		Tag:  TagIVFSingleFrameAV1,
		Kind: KindIVF,
		Name: "ivf single av1 frame",
		Input: []byte{
			'D', 'K', 'I', 'F',
			0x00, 0x00,
			0x20, 0x00,
			'A', 'V', '0', '1',
			0x10, 0x00,
			0x10, 0x00,
			0x1e, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00,
			0x01, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00,
			0x02, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00,
			0x12, 0x00,
		},
		Want: []byte{0x12, 0x00},
	},
	{
		Tag:   TagParserSequenceFullLowOverhead,
		Kind:  KindParser,
		Name:  "libaom full sequence header low-overhead obu",
		Input: []byte{0x0a, 0x0b, 0x00, 0x00, 0x00, 0x04, 0x45, 0x7e, 0x3e, 0xff, 0xfc, 0xc0, 0x20},
		Want:  []byte{0x00, 0x00, 0x00, 0x04, 0x45, 0x7e, 0x3e, 0xff, 0xfc, 0xc0, 0x20},
	},
	{
		Tag:   TagParserSequenceReducedLowOverhead,
		Kind:  KindParser,
		Name:  "libaom reduced still sequence header low-overhead obu",
		Input: []byte{0x0a, 0x07, 0x18, 0x22, 0x2b, 0xf1, 0xfe, 0xc0, 0x20},
		Want:  []byte{0x18, 0x22, 0x2b, 0xf1, 0xfe, 0xc0, 0x20},
	},
	{
		Tag:   TagParserSequenceFullAnnexB,
		Kind:  KindParser,
		Name:  "libaom full sequence header annex b obu",
		Input: []byte{0x0c, 0x08, 0x00, 0x00, 0x00, 0x04, 0x45, 0x7e, 0x3e, 0xff, 0xfc, 0xc0, 0x20},
		Want:  []byte{0x00, 0x00, 0x00, 0x04, 0x45, 0x7e, 0x3e, 0xff, 0xfc, 0xc0, 0x20},
	},
	{
		Tag:   TagParserSequenceReducedAnnexB,
		Kind:  KindParser,
		Name:  "libaom reduced still sequence header annex b obu",
		Input: []byte{0x08, 0x08, 0x18, 0x22, 0x2b, 0xf1, 0xfe, 0xc0, 0x20},
		Want:  []byte{0x18, 0x22, 0x2b, 0xf1, 0xfe, 0xc0, 0x20},
	},
	{
		Tag:   TagParserSequenceBuganizer502133197,
		Kind:  KindParser,
		Name:  "libaom buganizer 502133197 sequence header low-overhead obu",
		Input: []byte{0x0a, 0x10, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		Want:  []byte{0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
	},
}

var coreIVFSingleFrameMD5 = MD5{0x5a, 0x68, 0xde, 0x99, 0x7d, 0x60, 0xaf, 0xa9, 0x08, 0x3b, 0x17, 0xfe, 0x00, 0xf7, 0xcd, 0xf2}

var coreDigests = [...]FrameDigest{
	{Tag: TagIVFSingleFrameAV1, FrameIndex: 0, MD5: coreIVFSingleFrameMD5},
}

const (
	coreIndexOBULowOverheadTemporalDelimiter = iota
	coreIndexOBUAnnexBTemporalUnit
	coreIndexRTPPayloadSingleOBU
	coreIndexRTPPayloadFragmentedOBU
	coreIndexIVFSingleFrameAV1
	coreIndexParserSequenceFullLowOverhead
	coreIndexParserSequenceReducedLowOverhead
	coreIndexParserSequenceFullAnnexB
	coreIndexParserSequenceReducedAnnexB
	coreIndexParserSequenceBuganizer502133197
)

// CoreVector returns one core vector by stable numeric tag without scanning
// names or manifests.
func CoreVector(tag Tag) (Vector, bool) {
	switch tag {
	case TagOBULowOverheadTemporalDelimiter:
		return coreVectors[coreIndexOBULowOverheadTemporalDelimiter], true
	case TagOBUAnnexBTemporalUnit:
		return coreVectors[coreIndexOBUAnnexBTemporalUnit], true
	case TagRTPPayloadSingleOBU:
		return coreVectors[coreIndexRTPPayloadSingleOBU], true
	case TagRTPPayloadFragmentedOBU:
		return coreVectors[coreIndexRTPPayloadFragmentedOBU], true
	case TagIVFSingleFrameAV1:
		return coreVectors[coreIndexIVFSingleFrameAV1], true
	case TagParserSequenceFullLowOverhead:
		return coreVectors[coreIndexParserSequenceFullLowOverhead], true
	case TagParserSequenceReducedLowOverhead:
		return coreVectors[coreIndexParserSequenceReducedLowOverhead], true
	case TagParserSequenceFullAnnexB:
		return coreVectors[coreIndexParserSequenceFullAnnexB], true
	case TagParserSequenceReducedAnnexB:
		return coreVectors[coreIndexParserSequenceReducedAnnexB], true
	case TagParserSequenceBuganizer502133197:
		return coreVectors[coreIndexParserSequenceBuganizer502133197], true
	default:
		return Vector{}, false
	}
}

// CoreFrameDigest returns one core frame digest by stable numeric tag and frame
// index without scanning names or manifests.
func CoreFrameDigest(tag Tag, frameIndex uint32) (FrameDigest, bool) {
	switch tag {
	case TagIVFSingleFrameAV1:
		if frameIndex == 0 {
			return coreDigests[0], true
		}
	}
	return FrameDigest{}, false
}

// CoreSuite returns the minimal byte-level vectors that cover the current
// transport and OBU foundations.
func CoreSuite() Suite {
	return Suite{
		Name: "goav1-core",
		Manifest: Manifest{
			Vectors: coreVectors[:],
			Digests: coreDigests[:],
		},
	}
}
