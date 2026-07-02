package entropy

import "testing"

// TestCostSymbolUniformBinary checks the av1_cost_symbol port at the anchor
// point libaom defines exactly: a 50% binary symbol costs one bit, and
// av1_prob_cost[0] is 512 (one bit in AV1_PROB_COST_SHIFT units).
func TestCostSymbolUniformBinary(t *testing.T) {
	var cdf CDF
	if err := cdf.InitUniform(2); err != nil {
		t.Fatal(err)
	}
	for sym := range 2 {
		if got := CostSymbol(&cdf, sym); got != 512 {
			t.Fatalf("uniform binary symbol %d cost = %d, want 512", sym, got)
		}
	}
}

// TestCostSymbolSkewOrdering checks that the likely symbol prices below one
// bit and the unlikely one above, and that a more extreme skew is priced
// more extremely.
func TestCostSymbolSkewOrdering(t *testing.T) {
	var mild, strong CDF
	// 75/25 split.
	if err := mild.Init([]uint16{24576}); err != nil {
		t.Fatal(err)
	}
	// 31/32 split.
	if err := strong.Init([]uint16{31744}); err != nil {
		t.Fatal(err)
	}
	if l, u := CostSymbol(&mild, 0), CostSymbol(&mild, 1); l >= 512 || u <= 512 {
		t.Fatalf("75/25 costs = %d/%d, want likely < 512 < unlikely", l, u)
	}
	if CostSymbol(&strong, 0) >= CostSymbol(&mild, 0) {
		t.Fatalf("stronger skew must price the likely symbol cheaper: %d vs %d",
			CostSymbol(&strong, 0), CostSymbol(&mild, 0))
	}
	if CostSymbol(&strong, 1) <= CostSymbol(&mild, 1) {
		t.Fatalf("stronger skew must price the unlikely symbol dearer: %d vs %d",
			CostSymbol(&strong, 1), CostSymbol(&mild, 1))
	}
	// ~5 bits for a 1/32 symbol (av1_cost_symbol is exact at powers of two).
	if got := CostSymbol(&strong, 1); got != 5*512 {
		t.Fatalf("1/32 symbol cost = %d, want %d", got, 5*512)
	}
}

// TestCostSymbolExtremesStayInTable exercises the clamping paths: a
// probability at the CDF floor and one at the ceiling must not index outside
// av1_prob_cost.
func TestCostSymbolExtremesStayInTable(t *testing.T) {
	var cdf CDF
	if err := cdf.Init([]uint16{32767}); err != nil {
		t.Fatal(err)
	}
	if got := CostSymbol(&cdf, 0); got != 0 {
		// p15 = 32767 clamps to prob 255 whose av1_prob_cost entry is 3.
		if got != 3 {
			t.Fatalf("near-certain symbol cost = %d, want 3", got)
		}
	}
	if got := CostSymbol(&cdf, 1); got == 0 || got > 15*512 {
		t.Fatalf("near-impossible symbol cost = %d, want within (0, %d]", got, 15*512)
	}
}
