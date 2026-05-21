package testvector

import "errors"

var (
	ErrInvalidTag      = errors.New("testvector: invalid tag")
	ErrDuplicateTag    = errors.New("testvector: duplicate tag")
	ErrMissingVector   = errors.New("testvector: missing vector")
	ErrMismatchedBytes = errors.New("testvector: mismatched bytes")
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
)

const (
	TagOBULowOverheadTemporalDelimiter Tag = 0x0001_0001
	TagRTPPayloadSingleOBU             Tag = 0x0002_0001
	TagRTPPayloadFragmentedOBU         Tag = 0x0002_0002
	TagIVFSingleFrameAV1               Tag = 0x0003_0001
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

// Manifest is a caller-owned view of a vector suite.
type Manifest struct {
	Vectors []Vector
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
	return nil
}
