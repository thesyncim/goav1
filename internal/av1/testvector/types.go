package testvector

import "errors"

var (
	ErrInvalidTag      = errors.New("testvector: invalid tag")
	ErrDuplicateTag    = errors.New("testvector: duplicate tag")
	ErrMissingVector   = errors.New("testvector: missing vector")
	ErrMissingDigest   = errors.New("testvector: missing digest")
	ErrMismatchedBytes = errors.New("testvector: mismatched bytes")
	ErrMismatchedMD5   = errors.New("testvector: mismatched md5")
	ErrInvalidMD5      = errors.New("testvector: invalid md5")
	ErrInvalidRemote   = errors.New("testvector: invalid remote vector")
	ErrChecksum        = errors.New("testvector: checksum mismatch")
)

// Tag identifies a vector or oracle output without string matching in hot
// loops. Tags are stable across manifests; zero is reserved for invalid data.
type Tag uint32

// Kind groups vectors by the stage they exercise.
type Kind uint8

const (
	KindOBU Kind = iota + 1
	KindRTP
	KindIVF
	KindParser
	KindDecoder
	KindDSP
	KindAnnexB
)

const (
	TagOBULowOverheadTemporalDelimiter  Tag = 0x0001_0001
	TagOBUAnnexBTemporalUnit            Tag = 0x0001_0002
	TagRTPPayloadSingleOBU              Tag = 0x0002_0001
	TagRTPPayloadFragmentedOBU          Tag = 0x0002_0002
	TagIVFSingleFrameAV1                Tag = 0x0003_0001
	TagParserSequenceFullLowOverhead    Tag = 0x0004_0001
	TagParserSequenceReducedLowOverhead Tag = 0x0004_0002
	TagParserSequenceFullAnnexB         Tag = 0x0004_0003
	TagParserSequenceReducedAnnexB      Tag = 0x0004_0004
	TagParserSequenceBuganizer502133197 Tag = 0x0004_0005
	TagDecoderLibaomQuantizer00         Tag = 0x0005_0001
	TagDecoderLibaomQuantizer01         Tag = 0x0005_0002
	TagDecoderLibaomSize16x16           Tag = 0x0005_0003
	TagDecoderLibaomQuantizer10Bit00    Tag = 0x0005_0004
	TagDecoderLibaomAllIntra            Tag = 0x0005_0005
	TagDecoderLibaomCDFUpdate           Tag = 0x0005_0006
	TagDecoderLibaomMV                  Tag = 0x0005_0007
	TagDecoderLibaomMFMV                Tag = 0x0005_0008
	TagDecoderLibaomIntrabc             Tag = 0x0005_0009
	TagDecoderLibaomSVC                 Tag = 0x0005_000a
	TagDecoderLibaomFilmGrain           Tag = 0x0005_000b
	TagDecoderLibaomMonochrome          Tag = 0x0005_000c
	TagDecoderLibaomFilmGrain10Bit      Tag = 0x0005_000d
	TagDecoderLibaomMonochrome10Bit     Tag = 0x0005_000e
)

// Vector describes one byte-level input and its expected oracle output. Input
// and Want are caller-owned; Vector never copies them.
type Vector struct {
	Tag   Tag
	Kind  Kind
	Name  string
	Input []byte
	Want  []byte
}

type MD5 [16]byte

// FrameDigest describes the expected decoded-frame digest for one vector.
type FrameDigest struct {
	Tag        Tag
	FrameIndex uint32
	MD5        MD5
}

// Manifest is a caller-owned view of a vector suite.
type Manifest struct {
	Vectors []Vector
	Digests []FrameDigest
}

// Suite names a manifest so test runners can compose transport, syntax, and
// decoder-vector sets without reflection.
type Suite struct {
	Name     string
	Manifest Manifest
}

func (m Manifest) Find(tag Tag) (Vector, bool) {
	for _, vector := range m.Vectors {
		if vector.Tag == tag {
			return vector, true
		}
	}
	return Vector{}, false
}

func (m Manifest) FindDigest(tag Tag, frameIndex uint32) (FrameDigest, bool) {
	for _, digest := range m.Digests {
		if digest.Tag == tag && digest.FrameIndex == frameIndex {
			return digest, true
		}
	}
	return FrameDigest{}, false
}

func (m Manifest) Validate() error {
	for i, vector := range m.Vectors {
		if vector.Tag == 0 {
			return ErrInvalidTag
		}
		for j := 0; j < i; j++ {
			if m.Vectors[j].Tag == vector.Tag {
				return ErrDuplicateTag
			}
		}
	}
	for i, digest := range m.Digests {
		if digest.Tag == 0 {
			return ErrInvalidTag
		}
		for j := 0; j < i; j++ {
			if m.Digests[j].Tag == digest.Tag && m.Digests[j].FrameIndex == digest.FrameIndex {
				return ErrDuplicateTag
			}
		}
	}
	return nil
}

func ParseMD5Hex(src []byte) (MD5, error) {
	if len(src) < 32 {
		return MD5{}, ErrInvalidMD5
	}
	var md5 MD5
	for i := 0; i < len(md5); i++ {
		hi, ok := hexNibble(src[i*2])
		if !ok {
			return MD5{}, ErrInvalidMD5
		}
		lo, ok := hexNibble(src[i*2+1])
		if !ok {
			return MD5{}, ErrInvalidMD5
		}
		md5[i] = hi<<4 | lo
	}
	return md5, nil
}

func hexNibble(b byte) (byte, bool) {
	switch {
	case b >= '0' && b <= '9':
		return b - '0', true
	case b >= 'a' && b <= 'f':
		return b - 'a' + 10, true
	case b >= 'A' && b <= 'F':
		return b - 'A' + 10, true
	default:
		return 0, false
	}
}
