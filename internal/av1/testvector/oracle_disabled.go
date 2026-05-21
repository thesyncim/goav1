//go:build !goav1_oracle

package testvector

const OracleEnabled = false

// Oracle is zero-size when oracle checks are disabled. Keep oracle call sites
// guarded by OracleEnabled so production and ordinary unit-test builds retain
// no manifest state or comparison work.
type Oracle struct{}

func NewOracle(Manifest) Oracle {
	return Oracle{}
}

func (Oracle) CheckBytes(Tag, []byte) error {
	return nil
}
