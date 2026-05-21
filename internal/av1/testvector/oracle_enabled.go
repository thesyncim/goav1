//go:build goav1_oracle

package testvector

import "bytes"

const OracleEnabled = true

// Oracle compares outputs against a caller-owned manifest in explicit
// goav1_oracle test runs.
type Oracle struct {
	manifest Manifest
}

func NewOracle(manifest Manifest) Oracle {
	return Oracle{manifest: manifest}
}

func (o Oracle) CheckBytes(tag Tag, got []byte) error {
	vector, ok := o.manifest.Find(tag)
	if !ok {
		return ErrMissingVector
	}
	if !bytes.Equal(got, vector.Want) {
		return ErrMismatchedBytes
	}
	return nil
}

func (o Oracle) CheckMD5(tag Tag, frameIndex uint32, got MD5) error {
	digest, ok := o.manifest.FindDigest(tag, frameIndex)
	if !ok {
		return ErrMissingDigest
	}
	if got != digest.MD5 {
		return ErrMismatchedMD5
	}
	return nil
}
