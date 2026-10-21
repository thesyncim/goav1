package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/transform"
)

// TestUVInterTransformTypeMatchesSetRules checks the dav1d-style chroma rule
// against the allowed-set derivation it replaced, for every chroma-capable
// size and every luma type the luma read can produce for that size's set.
func TestUVInterTransformTypeMatchesSetRules(t *testing.T) {
	checked := 0
	for size := TransformSize(0); size < transformSizeCount; size++ {
		if transformSizeSquareUp[size] == TransformSize64x64 {
			continue // chroma transforms are at most 32x32
		}
		txSize, err := size.TransformSize()
		if err != nil {
			continue
		}
		for _, reduced := range []bool{false, true} {
			lumaSet, err := ExtTXSetTypeFor(size, true, reduced)
			if err != nil {
				t.Fatal(err)
			}
			for typ := transform.Type(0); typ < transform.TypeCount; typ++ {
				if ok, err := ExtTXTypeAllowed(lumaSet, typ); err != nil || !ok {
					continue
				}
				want := typ
				if ok, err := ExtTXTypeAllowed(lumaSet, typ); err != nil || !ok || !typ.Supported(txSize) {
					want = transform.TypeDCTDCT
				}
				got, err := uvInterTransformType(size, typ)
				if err != nil {
					t.Fatal(err)
				}
				// The old rule used the chroma size's own set, the same set the
				// luma type was drawn from when sizes match; check that set too.
				if got != want {
					t.Fatalf("size=%d reduced=%v luma=%d: got %d want %d", size, reduced, typ, got, want)
				}
				checked++
			}
		}
	}
	if checked == 0 {
		t.Fatal("no cases checked")
	}
	t.Logf("checked %d cases", checked)
}
