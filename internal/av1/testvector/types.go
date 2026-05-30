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

	// Extended cohort: opt-in vectors that probe multi-quantizer 10-bit,
	// larger resolutions, and additional SVC permutations. They live in
	// SuiteLevelExtended | SuiteLevelFull and are not part of the default
	// fast gate.
	TagDecoderLibaomQuantizer10Bit32 Tag = 0x0005_000f
	TagDecoderLibaomQuantizer10Bit63 Tag = 0x0005_0010
	TagDecoderLibaomSize34x34        Tag = 0x0005_0011
	TagDecoderLibaomSize64x64        Tag = 0x0005_0012
	TagDecoderLibaomSize66x66        Tag = 0x0005_0013
	TagDecoderLibaomSize208x208      Tag = 0x0005_0014
	TagDecoderLibaomSize226x226      Tag = 0x0005_0015
	TagDecoderLibaomSVCL2T1          Tag = 0x0005_0016
	TagDecoderLibaomSVCL2T2          Tag = 0x0005_0017

	// Full quantizer + size sweep cohort. These mirror the remaining
	// in-scope 8/10-bit 4:2:0 libaom decode vectors (av1-1-b8-00-quantizer
	// 02..63, av1-1-b10-00-quantizer 01..62 minus 32, and the
	// av1-1-b8-01-size WxH sweep). All carry SuiteLevelExtended |
	// SuiteLevelFull so they ride the opt-in extended dry-run only.
	TagDecoderLibaomQuantizer02      Tag = 0x00050018
	TagDecoderLibaomQuantizer03      Tag = 0x00050019
	TagDecoderLibaomQuantizer04      Tag = 0x0005001a
	TagDecoderLibaomQuantizer05      Tag = 0x0005001b
	TagDecoderLibaomQuantizer06      Tag = 0x0005001c
	TagDecoderLibaomQuantizer07      Tag = 0x0005001d
	TagDecoderLibaomQuantizer08      Tag = 0x0005001e
	TagDecoderLibaomQuantizer09      Tag = 0x0005001f
	TagDecoderLibaomQuantizer10      Tag = 0x00050020
	TagDecoderLibaomQuantizer11      Tag = 0x00050021
	TagDecoderLibaomQuantizer12      Tag = 0x00050022
	TagDecoderLibaomQuantizer13      Tag = 0x00050023
	TagDecoderLibaomQuantizer14      Tag = 0x00050024
	TagDecoderLibaomQuantizer15      Tag = 0x00050025
	TagDecoderLibaomQuantizer16      Tag = 0x00050026
	TagDecoderLibaomQuantizer17      Tag = 0x00050027
	TagDecoderLibaomQuantizer18      Tag = 0x00050028
	TagDecoderLibaomQuantizer19      Tag = 0x00050029
	TagDecoderLibaomQuantizer20      Tag = 0x0005002a
	TagDecoderLibaomQuantizer21      Tag = 0x0005002b
	TagDecoderLibaomQuantizer22      Tag = 0x0005002c
	TagDecoderLibaomQuantizer23      Tag = 0x0005002d
	TagDecoderLibaomQuantizer24      Tag = 0x0005002e
	TagDecoderLibaomQuantizer25      Tag = 0x0005002f
	TagDecoderLibaomQuantizer26      Tag = 0x00050030
	TagDecoderLibaomQuantizer27      Tag = 0x00050031
	TagDecoderLibaomQuantizer28      Tag = 0x00050032
	TagDecoderLibaomQuantizer29      Tag = 0x00050033
	TagDecoderLibaomQuantizer30      Tag = 0x00050034
	TagDecoderLibaomQuantizer31      Tag = 0x00050035
	TagDecoderLibaomQuantizer32      Tag = 0x00050036
	TagDecoderLibaomQuantizer33      Tag = 0x00050037
	TagDecoderLibaomQuantizer34      Tag = 0x00050038
	TagDecoderLibaomQuantizer35      Tag = 0x00050039
	TagDecoderLibaomQuantizer36      Tag = 0x0005003a
	TagDecoderLibaomQuantizer37      Tag = 0x0005003b
	TagDecoderLibaomQuantizer38      Tag = 0x0005003c
	TagDecoderLibaomQuantizer39      Tag = 0x0005003d
	TagDecoderLibaomQuantizer40      Tag = 0x0005003e
	TagDecoderLibaomQuantizer41      Tag = 0x0005003f
	TagDecoderLibaomQuantizer42      Tag = 0x00050040
	TagDecoderLibaomQuantizer43      Tag = 0x00050041
	TagDecoderLibaomQuantizer44      Tag = 0x00050042
	TagDecoderLibaomQuantizer45      Tag = 0x00050043
	TagDecoderLibaomQuantizer46      Tag = 0x00050044
	TagDecoderLibaomQuantizer47      Tag = 0x00050045
	TagDecoderLibaomQuantizer48      Tag = 0x00050046
	TagDecoderLibaomQuantizer49      Tag = 0x00050047
	TagDecoderLibaomQuantizer50      Tag = 0x00050048
	TagDecoderLibaomQuantizer51      Tag = 0x00050049
	TagDecoderLibaomQuantizer52      Tag = 0x0005004a
	TagDecoderLibaomQuantizer53      Tag = 0x0005004b
	TagDecoderLibaomQuantizer54      Tag = 0x0005004c
	TagDecoderLibaomQuantizer55      Tag = 0x0005004d
	TagDecoderLibaomQuantizer56      Tag = 0x0005004e
	TagDecoderLibaomQuantizer57      Tag = 0x0005004f
	TagDecoderLibaomQuantizer58      Tag = 0x00050050
	TagDecoderLibaomQuantizer59      Tag = 0x00050051
	TagDecoderLibaomQuantizer60      Tag = 0x00050052
	TagDecoderLibaomQuantizer61      Tag = 0x00050053
	TagDecoderLibaomQuantizer62      Tag = 0x00050054
	TagDecoderLibaomQuantizer63      Tag = 0x00050055
	TagDecoderLibaomQuantizer10Bit01 Tag = 0x00050056
	TagDecoderLibaomQuantizer10Bit02 Tag = 0x00050057
	TagDecoderLibaomQuantizer10Bit03 Tag = 0x00050058
	TagDecoderLibaomQuantizer10Bit04 Tag = 0x00050059
	TagDecoderLibaomQuantizer10Bit05 Tag = 0x0005005a
	TagDecoderLibaomQuantizer10Bit06 Tag = 0x0005005b
	TagDecoderLibaomQuantizer10Bit07 Tag = 0x0005005c
	TagDecoderLibaomQuantizer10Bit08 Tag = 0x0005005d
	TagDecoderLibaomQuantizer10Bit09 Tag = 0x0005005e
	TagDecoderLibaomQuantizer10Bit10 Tag = 0x0005005f
	TagDecoderLibaomQuantizer10Bit11 Tag = 0x00050060
	TagDecoderLibaomQuantizer10Bit12 Tag = 0x00050061
	TagDecoderLibaomQuantizer10Bit13 Tag = 0x00050062
	TagDecoderLibaomQuantizer10Bit14 Tag = 0x00050063
	TagDecoderLibaomQuantizer10Bit15 Tag = 0x00050064
	TagDecoderLibaomQuantizer10Bit16 Tag = 0x00050065
	TagDecoderLibaomQuantizer10Bit17 Tag = 0x00050066
	TagDecoderLibaomQuantizer10Bit18 Tag = 0x00050067
	TagDecoderLibaomQuantizer10Bit19 Tag = 0x00050068
	TagDecoderLibaomQuantizer10Bit20 Tag = 0x00050069
	TagDecoderLibaomQuantizer10Bit21 Tag = 0x0005006a
	TagDecoderLibaomQuantizer10Bit22 Tag = 0x0005006b
	TagDecoderLibaomQuantizer10Bit23 Tag = 0x0005006c
	TagDecoderLibaomQuantizer10Bit24 Tag = 0x0005006d
	TagDecoderLibaomQuantizer10Bit25 Tag = 0x0005006e
	TagDecoderLibaomQuantizer10Bit26 Tag = 0x0005006f
	TagDecoderLibaomQuantizer10Bit27 Tag = 0x00050070
	TagDecoderLibaomQuantizer10Bit28 Tag = 0x00050071
	TagDecoderLibaomQuantizer10Bit29 Tag = 0x00050072
	TagDecoderLibaomQuantizer10Bit30 Tag = 0x00050073
	TagDecoderLibaomQuantizer10Bit31 Tag = 0x00050074
	TagDecoderLibaomQuantizer10Bit33 Tag = 0x00050075
	TagDecoderLibaomQuantizer10Bit34 Tag = 0x00050076
	TagDecoderLibaomQuantizer10Bit35 Tag = 0x00050077
	TagDecoderLibaomQuantizer10Bit36 Tag = 0x00050078
	TagDecoderLibaomQuantizer10Bit37 Tag = 0x00050079
	TagDecoderLibaomQuantizer10Bit38 Tag = 0x0005007a
	TagDecoderLibaomQuantizer10Bit39 Tag = 0x0005007b
	TagDecoderLibaomQuantizer10Bit40 Tag = 0x0005007c
	TagDecoderLibaomQuantizer10Bit41 Tag = 0x0005007d
	TagDecoderLibaomQuantizer10Bit42 Tag = 0x0005007e
	TagDecoderLibaomQuantizer10Bit43 Tag = 0x0005007f
	TagDecoderLibaomQuantizer10Bit44 Tag = 0x00050080
	TagDecoderLibaomQuantizer10Bit45 Tag = 0x00050081
	TagDecoderLibaomQuantizer10Bit46 Tag = 0x00050082
	TagDecoderLibaomQuantizer10Bit47 Tag = 0x00050083
	TagDecoderLibaomQuantizer10Bit48 Tag = 0x00050084
	TagDecoderLibaomQuantizer10Bit49 Tag = 0x00050085
	TagDecoderLibaomQuantizer10Bit50 Tag = 0x00050086
	TagDecoderLibaomQuantizer10Bit51 Tag = 0x00050087
	TagDecoderLibaomQuantizer10Bit52 Tag = 0x00050088
	TagDecoderLibaomQuantizer10Bit53 Tag = 0x00050089
	TagDecoderLibaomQuantizer10Bit54 Tag = 0x0005008a
	TagDecoderLibaomQuantizer10Bit55 Tag = 0x0005008b
	TagDecoderLibaomQuantizer10Bit56 Tag = 0x0005008c
	TagDecoderLibaomQuantizer10Bit57 Tag = 0x0005008d
	TagDecoderLibaomQuantizer10Bit58 Tag = 0x0005008e
	TagDecoderLibaomQuantizer10Bit59 Tag = 0x0005008f
	TagDecoderLibaomQuantizer10Bit60 Tag = 0x00050090
	TagDecoderLibaomQuantizer10Bit61 Tag = 0x00050091
	TagDecoderLibaomQuantizer10Bit62 Tag = 0x00050092
	TagDecoderLibaomSize16x18        Tag = 0x00050093
	TagDecoderLibaomSize16x32        Tag = 0x00050094
	TagDecoderLibaomSize16x34        Tag = 0x00050095
	TagDecoderLibaomSize16x64        Tag = 0x00050096
	TagDecoderLibaomSize16x66        Tag = 0x00050097
	TagDecoderLibaomSize18x16        Tag = 0x00050098
	TagDecoderLibaomSize18x18        Tag = 0x00050099
	TagDecoderLibaomSize18x32        Tag = 0x0005009a
	TagDecoderLibaomSize18x34        Tag = 0x0005009b
	TagDecoderLibaomSize18x64        Tag = 0x0005009c
	TagDecoderLibaomSize18x66        Tag = 0x0005009d
	TagDecoderLibaomSize196x196      Tag = 0x0005009e
	TagDecoderLibaomSize196x198      Tag = 0x0005009f
	TagDecoderLibaomSize196x200      Tag = 0x000500a0
	TagDecoderLibaomSize196x202      Tag = 0x000500a1
	TagDecoderLibaomSize196x208      Tag = 0x000500a2
	TagDecoderLibaomSize196x210      Tag = 0x000500a3
	TagDecoderLibaomSize196x224      Tag = 0x000500a4
	TagDecoderLibaomSize196x226      Tag = 0x000500a5
	TagDecoderLibaomSize198x196      Tag = 0x000500a6
	TagDecoderLibaomSize198x198      Tag = 0x000500a7
	TagDecoderLibaomSize198x200      Tag = 0x000500a8
	TagDecoderLibaomSize198x202      Tag = 0x000500a9
	TagDecoderLibaomSize198x208      Tag = 0x000500aa
	TagDecoderLibaomSize198x210      Tag = 0x000500ab
	TagDecoderLibaomSize198x224      Tag = 0x000500ac
	TagDecoderLibaomSize198x226      Tag = 0x000500ad
	TagDecoderLibaomSize200x196      Tag = 0x000500ae
	TagDecoderLibaomSize200x198      Tag = 0x000500af
	TagDecoderLibaomSize200x200      Tag = 0x000500b0
	TagDecoderLibaomSize200x202      Tag = 0x000500b1
	TagDecoderLibaomSize200x208      Tag = 0x000500b2
	TagDecoderLibaomSize200x210      Tag = 0x000500b3
	TagDecoderLibaomSize200x224      Tag = 0x000500b4
	TagDecoderLibaomSize200x226      Tag = 0x000500b5
	TagDecoderLibaomSize202x196      Tag = 0x000500b6
	TagDecoderLibaomSize202x198      Tag = 0x000500b7
	TagDecoderLibaomSize202x200      Tag = 0x000500b8
	TagDecoderLibaomSize202x202      Tag = 0x000500b9
	TagDecoderLibaomSize202x208      Tag = 0x000500ba
	TagDecoderLibaomSize202x210      Tag = 0x000500bb
	TagDecoderLibaomSize202x224      Tag = 0x000500bc
	TagDecoderLibaomSize202x226      Tag = 0x000500bd
	TagDecoderLibaomSize208x196      Tag = 0x000500be
	TagDecoderLibaomSize208x198      Tag = 0x000500bf
	TagDecoderLibaomSize208x200      Tag = 0x000500c0
	TagDecoderLibaomSize208x202      Tag = 0x000500c1
	TagDecoderLibaomSize208x210      Tag = 0x000500c2
	TagDecoderLibaomSize208x224      Tag = 0x000500c3
	TagDecoderLibaomSize208x226      Tag = 0x000500c4
	TagDecoderLibaomSize210x196      Tag = 0x000500c5
	TagDecoderLibaomSize210x198      Tag = 0x000500c6
	TagDecoderLibaomSize210x200      Tag = 0x000500c7
	TagDecoderLibaomSize210x202      Tag = 0x000500c8
	TagDecoderLibaomSize210x208      Tag = 0x000500c9
	TagDecoderLibaomSize210x210      Tag = 0x000500ca
	TagDecoderLibaomSize210x224      Tag = 0x000500cb
	TagDecoderLibaomSize210x226      Tag = 0x000500cc
	TagDecoderLibaomSize224x196      Tag = 0x000500cd
	TagDecoderLibaomSize224x198      Tag = 0x000500ce
	TagDecoderLibaomSize224x200      Tag = 0x000500cf
	TagDecoderLibaomSize224x202      Tag = 0x000500d0
	TagDecoderLibaomSize224x208      Tag = 0x000500d1
	TagDecoderLibaomSize224x210      Tag = 0x000500d2
	TagDecoderLibaomSize224x224      Tag = 0x000500d3
	TagDecoderLibaomSize224x226      Tag = 0x000500d4
	TagDecoderLibaomSize226x196      Tag = 0x000500d5
	TagDecoderLibaomSize226x198      Tag = 0x000500d6
	TagDecoderLibaomSize226x200      Tag = 0x000500d7
	TagDecoderLibaomSize226x202      Tag = 0x000500d8
	TagDecoderLibaomSize226x208      Tag = 0x000500d9
	TagDecoderLibaomSize226x210      Tag = 0x000500da
	TagDecoderLibaomSize226x224      Tag = 0x000500db
	TagDecoderLibaomSize32x16        Tag = 0x000500dc
	TagDecoderLibaomSize32x18        Tag = 0x000500dd
	TagDecoderLibaomSize32x32        Tag = 0x000500de
	TagDecoderLibaomSize32x34        Tag = 0x000500df
	TagDecoderLibaomSize32x64        Tag = 0x000500e0
	TagDecoderLibaomSize32x66        Tag = 0x000500e1
	TagDecoderLibaomSize34x16        Tag = 0x000500e2
	TagDecoderLibaomSize34x18        Tag = 0x000500e3
	TagDecoderLibaomSize34x32        Tag = 0x000500e4
	TagDecoderLibaomSize34x64        Tag = 0x000500e5
	TagDecoderLibaomSize34x66        Tag = 0x000500e6
	TagDecoderLibaomSize64x16        Tag = 0x000500e7
	TagDecoderLibaomSize64x18        Tag = 0x000500e8
	TagDecoderLibaomSize64x32        Tag = 0x000500e9
	TagDecoderLibaomSize64x34        Tag = 0x000500ea
	TagDecoderLibaomSize64x66        Tag = 0x000500eb
	TagDecoderLibaomSize66x16        Tag = 0x000500ec
	TagDecoderLibaomSize66x18        Tag = 0x000500ed
	TagDecoderLibaomSize66x32        Tag = 0x000500ee
	TagDecoderLibaomSize66x34        Tag = 0x000500ef
	TagDecoderLibaomSize66x64        Tag = 0x000500f0
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
		for j := range i {
			if m.Vectors[j].Tag == vector.Tag {
				return ErrDuplicateTag
			}
		}
	}
	for i, digest := range m.Digests {
		if digest.Tag == 0 {
			return ErrInvalidTag
		}
		for j := range i {
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
	for i := range len(md5) {
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
