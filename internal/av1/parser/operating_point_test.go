package parser

import "testing"

// Layer-scenario fixtures derived from the example operating-point IDCs in
// libaom/test/decode_scalability_test.cc. The 12-bit operating_point_idc
// has its low 8 bits selecting temporal layers (bit j -> Tj) and its high
// 4 bits selecting spatial layers (bit 8+j -> Sj).
//
//	L2T1 (2 spatial, 1 temporal):
//	    operating_point_idc[0] = 0x301  (bits {0,8,9}) -> T0 in S0..S1
//	    operating_point_idc[1] = 0x101  (bits {0,8})   -> T0 in S0 only
//
//	L2T2 (2 spatial, 2 temporal):
//	    operating_point_idc[0] = 0x303  (bits {0,1,8,9}) -> T0..T1 in S0..S1
//	    operating_point_idc[1] = 0x301  (bits {0,8,9})   -> T0    in S0..S1
//	    operating_point_idc[2] = 0x103  (bits {0,1,8})   -> T0..T1 in S0
//	    operating_point_idc[3] = 0x101  (bits {0,8})     -> T0    in S0

func TestOperatingPointIDCMatchesBaseLayerOnly(t *testing.T) {
	// operating_point_idc == 0: any layer matches.
	cases := []struct {
		t, s uint8
	}{
		{0, 0},
		{0, 1},
		{1, 0},
		{7, 3},
	}
	for _, c := range cases {
		if !OperatingPointIDCMatches(OperatingPointIDCAnyLayer, c.t, c.s) {
			t.Fatalf("idc=0 t=%d s=%d: expected match", c.t, c.s)
		}
	}
}

func TestOperatingPointIDCMatchesTwoSpatial(t *testing.T) {
	// L2T1: op0 = 0x301 (T0 in S0..S1), op1 = 0x101 (T0 in S0).
	op0 := uint16(0x301)
	op1 := uint16(0x101)

	if !OperatingPointIDCMatches(op0, 0, 0) {
		t.Fatal("op0 should include (T0,S0)")
	}
	if !OperatingPointIDCMatches(op0, 0, 1) {
		t.Fatal("op0 should include (T0,S1)")
	}
	if OperatingPointIDCMatches(op0, 1, 0) {
		t.Fatal("op0 must not include (T1,S0)")
	}
	if OperatingPointIDCMatches(op0, 0, 2) {
		t.Fatal("op0 must not include (T0,S2)")
	}

	if !OperatingPointIDCMatches(op1, 0, 0) {
		t.Fatal("op1 should include (T0,S0)")
	}
	if OperatingPointIDCMatches(op1, 0, 1) {
		t.Fatal("op1 must not include (T0,S1)")
	}
	if OperatingPointIDCMatches(op1, 1, 0) {
		t.Fatal("op1 must not include (T1,S0)")
	}
}

func TestOperatingPointIDCMatchesTwoSpatialTwoTemporal(t *testing.T) {
	// L2T2: op0 = 0x303, op1 = 0x301, op2 = 0x103, op3 = 0x101.
	op0 := uint16(0x303)
	op1 := uint16(0x301)
	op2 := uint16(0x103)
	op3 := uint16(0x101)

	matchPairs := [][2]uint8{
		{0, 0}, {1, 0}, {0, 1}, {1, 1},
	}
	for _, p := range matchPairs {
		if !OperatingPointIDCMatches(op0, p[0], p[1]) {
			t.Fatalf("op0 should include (T%d,S%d)", p[0], p[1])
		}
	}

	// op1 = 0x301 -> T0 only in S0..S1.
	if !OperatingPointIDCMatches(op1, 0, 0) || !OperatingPointIDCMatches(op1, 0, 1) {
		t.Fatal("op1 should include T0 in both spatial layers")
	}
	if OperatingPointIDCMatches(op1, 1, 0) || OperatingPointIDCMatches(op1, 1, 1) {
		t.Fatal("op1 must not include T1 in either spatial layer")
	}

	// op2 = 0x103 -> T0..T1 in S0 only.
	if !OperatingPointIDCMatches(op2, 0, 0) || !OperatingPointIDCMatches(op2, 1, 0) {
		t.Fatal("op2 should include T0..T1 in S0")
	}
	if OperatingPointIDCMatches(op2, 0, 1) || OperatingPointIDCMatches(op2, 1, 1) {
		t.Fatal("op2 must not include S1")
	}

	// op3 = 0x101 -> base layer only.
	if !OperatingPointIDCMatches(op3, 0, 0) {
		t.Fatal("op3 should include base layer")
	}
	if OperatingPointIDCMatches(op3, 1, 0) || OperatingPointIDCMatches(op3, 0, 1) || OperatingPointIDCMatches(op3, 1, 1) {
		t.Fatal("op3 must include only (T0,S0)")
	}
}

func TestOperatingPointIDCMatchesRejectsOutOfRangeLayers(t *testing.T) {
	idc := uint16(0x303)
	if OperatingPointIDCMatches(idc, MaxOperatingPointTemporalLayers, 0) {
		t.Fatal("temporal_id beyond MAX must not match a non-zero idc")
	}
	if OperatingPointIDCMatches(idc, 0, MaxOperatingPointSpatialLayers) {
		t.Fatal("spatial_id beyond MAX must not match a non-zero idc")
	}
	// any-layer sentinel still matches even out-of-range IDs.
	if !OperatingPointIDCMatches(OperatingPointIDCAnyLayer, MaxOperatingPointTemporalLayers, MaxOperatingPointSpatialLayers) {
		t.Fatal("any-layer sentinel should always match")
	}
}

func TestOperatingPointTemporalAndSpatialCounts(t *testing.T) {
	// 0x303 -> T0..T1 in S0..S1 => 2 temporal, 2 spatial.
	if got := OperatingPointTemporalLayerCount(0x303); got != 2 {
		t.Fatalf("temporal count(0x303)=%d want 2", got)
	}
	if got := OperatingPointSpatialLayerCount(0x303); got != 2 {
		t.Fatalf("spatial count(0x303)=%d want 2", got)
	}
	// 0x101 -> base only.
	if got := OperatingPointTemporalLayerCount(0x101); got != 1 {
		t.Fatalf("temporal count(0x101)=%d want 1", got)
	}
	if got := OperatingPointSpatialLayerCount(0x101); got != 1 {
		t.Fatalf("spatial count(0x101)=%d want 1", got)
	}
	// idc==0 -> single-layer stream.
	if got := OperatingPointTemporalLayerCount(0); got != 1 {
		t.Fatalf("temporal count(0)=%d want 1", got)
	}
	if got := OperatingPointSpatialLayerCount(0); got != 1 {
		t.Fatalf("spatial count(0)=%d want 1", got)
	}
}

func TestSelectOperatingPointFirstMatchWins(t *testing.T) {
	var sh SequenceHeader
	sh.OperatingPointsCount = 4
	sh.OperatingPoints[0].IDC = 0x303
	sh.OperatingPoints[1].IDC = 0x301
	sh.OperatingPoints[2].IDC = 0x103
	sh.OperatingPoints[3].IDC = 0x101

	// (T0,S0) is covered by op0 first.
	if idx, ok := SelectOperatingPoint(sh, 0, 0); !ok || idx != 0 {
		t.Fatalf("select(T0,S0) idx=%d ok=%v want 0,true", idx, ok)
	}
	// (T1,S1) is only covered by op0.
	if idx, ok := SelectOperatingPoint(sh, 1, 1); !ok || idx != 0 {
		t.Fatalf("select(T1,S1) idx=%d ok=%v want 0,true", idx, ok)
	}
	// (T1,S2) is covered by no op.
	if _, ok := SelectOperatingPoint(sh, 1, 2); ok {
		t.Fatal("select(T1,S2) should not match any op")
	}
}

func TestSelectOperatingPointAnyLayer(t *testing.T) {
	var sh SequenceHeader
	sh.OperatingPointsCount = 1
	sh.OperatingPoints[0].IDC = OperatingPointIDCAnyLayer
	if idx, ok := SelectOperatingPoint(sh, 4, 3); !ok || idx != 0 {
		t.Fatalf("any-layer select idx=%d ok=%v want 0,true", idx, ok)
	}
}

func TestSelectOperatingPointEmpty(t *testing.T) {
	var sh SequenceHeader
	if _, ok := SelectOperatingPoint(sh, 0, 0); ok {
		t.Fatal("empty sequence header should not select")
	}
}
