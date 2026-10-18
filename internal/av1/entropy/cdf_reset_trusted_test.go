package entropy

import (
	"errors"
	"testing"
)

func TestCDFResetCountTrusted(t *testing.T) {
	var unused CDF
	ResetCDFCountTrusted(&unused)
	if unused != (CDF{}) {
		t.Fatal("unused CDF changed")
	}
	for symbols := 2; symbols <= MaxSymbols; symbols++ {
		var cdf CDF
		if err := cdf.InitUniform(symbols); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 63; i++ {
			if err := cdf.Update((i*7 + 1) % symbols); err != nil {
				t.Fatal(err)
			}
		}
		for i := symbols + 1; i < len(cdf.values); i++ {
			cdf.values[i] = 0xbeef
		}
		want := cdf
		if err := want.ResetCount(); err != nil {
			t.Fatal(err)
		}
		ResetCDFCountTrusted(&cdf)
		if cdf != want {
			t.Fatalf("symbols=%d: trusted reset changed probabilities or inactive storage", symbols)
		}
	}
}

func TestCDFResetCountValidatesWritableValues(t *testing.T) {
	for _, mutate := range []func([]uint16){
		func(v []uint16) { v[4] = MaxCDFCount + 1 },
		func(v []uint16) { v[3] = 1 },
		func(v []uint16) { v[0] = 0 },
	} {
		var cdf CDF
		if err := cdf.InitUniform(4); err != nil {
			t.Fatal(err)
		}
		if err := cdf.Update(2); err != nil {
			t.Fatal(err)
		}
		mutate(cdf.Values())
		before := cdf
		if err := cdf.ResetCount(); !errors.Is(err, ErrInvalidCDF) {
			t.Fatalf("invalid writable values: got %v", err)
		}
		if cdf != before {
			t.Fatal("invalid CDF was changed before reporting its error")
		}
	}
}
