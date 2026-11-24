package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

// TestEntropyTxDimsMatch checks the transform table the entropy package's
// read_tx_tree port carries against the tile table.
func TestEntropyTxDimsMatch(t *testing.T) {
	for size := range transformSizeCount {
		d := &transformDimensions[size]
		got := entropy.TxDims[size]
		want := entropy.TxDim{W4: d.W4, H4: d.H4, Log2W: d.Log2W, Log2H: d.Log2H, Max: d.Max, Sub: uint8(d.Sub)}
		if got != want {
			t.Fatalf("size %d: entropy %+v, tile %+v", size, got, want)
		}
	}
}
