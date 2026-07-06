package encoder

import (
	"math/rand"
	"testing"
)

// bruteSAD is the scalar reference for an 8x8 SAD between arbitrary plane
// positions.
func bruteSAD(src []byte, srcStride, sx, sy int, ref []byte, refStride, rx, ry, n int) uint32 {
	var total uint32
	for r := 0; r < n; r++ {
		so := (sy+r)*srcStride + sx
		ro := (ry+r)*refStride + rx
		for c := 0; c < n; c++ {
			d := int(src[so+c]) - int(ref[ro+c])
			if d < 0 {
				d = -d
			}
			total += uint32(d)
		}
	}
	return total
}

// sweepPUGeometry returns the SB-relative (x, y, size) of tier-zero PU index
// pu, mirroring the realtimeSBMultiSizeME layout.
func sweepPUGeometry(pu int) (x, y, n int) {
	switch {
	case pu == 0:
		return 0, 0, 64
	case pu < 5:
		q := pu - 1
		return (q & 1) * 32, (q >> 1) * 32, 32
	case pu < 21:
		i := pu - 5
		q, w := i>>2, i&3
		return (q&1)*32 + (w&1)*16, (q>>1)*32 + (w>>1)*16, 16
	default:
		i := pu - 21
		n16, c := i>>2, i&3
		q, w := n16>>2, n16&3
		return (q&1)*32 + (w&1)*16 + (c&1)*8, (q>>1)*32 + (w>>1)*16 + (c>>1)*8, 8
	}
}

// TestRealtimeSBMultiSizeMESweepMatchesBruteForce checks the hierarchical
// per-PU argmins of the sweep against an independent brute-force search over
// the same window: every PU must report the minimum full SAD over the swept
// positions with the earliest position (scan order) winning ties.
func TestRealtimeSBMultiSizeMESweepMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	const w, h = 192, 160
	mk := func() SourceFrame420 {
		y := make([]byte, w*h)
		for i := range y {
			y[i] = byte(rng.Intn(256))
		}
		return SourceFrame420{Y: y, YStride: w, Width: w, Height: h}
	}
	src, ref := mk(), mk()

	var st lossyEncodeState
	st.depthRemovalIsRef = true

	for _, sb := range [][2]int{{0, 0}, {64, 64}, {128, 96}, {64, 0}} {
		px, py := sb[0], sb[1]
		var me realtimeSBMultiSizeME
		st.realtimeSBMultiSizeMESweep(&me, src, ref, px, py)
		if !me.valid {
			t.Fatalf("sweep at (%d,%d) not valid", px, py)
		}

		// Recompute the swept window exactly as the sweep clamps it (hme is
		// nil so the center is (0,0)).
		nx, ny := realtimeSweepSearchW, realtimeSweepSearchH
		x0 := minInt(maxInt(-(realtimeSweepSearchW>>1), -px), w-64-px-(nx-1))
		y0 := minInt(maxInt(-(realtimeSweepSearchH>>1), -py), h-64-py-(ny-1))

		for pu := 0; pu < 85; pu++ {
			bx, by, n := sweepPUGeometry(pu)
			bestSAD := ^uint32(0)
			bestMV := [2]int16{}
			for yi := 0; yi < ny; yi++ {
				for xi := 0; xi < nx; xi++ {
					mvX, mvY := x0+xi, y0+yi
					s := bruteSAD(src.Y, w, px+bx, py+by, ref.Y, w, px+bx+mvX, py+by+mvY, n)
					if s < bestSAD {
						bestSAD = s
						bestMV = [2]int16{int16(mvX), int16(mvY)}
					}
				}
			}
			if me.bestSAD[pu] != bestSAD || me.bestMV[pu] != bestMV {
				t.Fatalf("SB (%d,%d) pu %d (%dx%d at +%d+%d): sweep sad=%d mv=%v, brute sad=%d mv=%v",
					px, py, pu, n, n, bx, by, me.bestSAD[pu], me.bestMV[pu], bestSAD, bestMV)
			}
		}

		// compute_distortion sums.
		var d32, d16, d8 uint32
		for i := 1; i < 5; i++ {
			d32 += me.bestSAD[i]
		}
		for i := 5; i < 21; i++ {
			d16 += me.bestSAD[i]
		}
		for i := 21; i < 85; i++ {
			d8 += me.bestSAD[i]
		}
		if me.dist64 != me.bestSAD[0] || me.dist32 != d32 || me.dist16 != d16 || me.dist8 != d8 {
			t.Fatalf("SB (%d,%d) distortion sums mismatch", px, py)
		}
		if me.dist16 < me.dist8 {
			t.Fatalf("SB (%d,%d): dist16 %d < dist8 %d (independent argmins cannot make the parent cheaper)", px, py, me.dist16, me.dist8)
		}
	}
}

// TestRealtimeDepthRemovalBelow16Arms exercises the ported threshold algebra
// on synthetic distortion profiles.
func TestRealtimeDepthRemovalBelow16Arms(t *testing.T) {
	me := realtimeSBMultiSizeME{valid: true}
	// Near-zero 16x16 cost: the absolute cost arm must fire at level 9. Keep
	// dist32 large so the Phase-2 arms stay out of the picture.
	me.dist32 = 1 << 24
	me.dist16, me.dist8 = 100, 90
	me.me8x8CostVariance = 1 << 30 // noise guard zeroes the dev threshold
	dec := realtimeDepthRemovalDecide(9, 120, &me)
	if !dec.cost16Arm || !dec.below16 {
		t.Fatalf("cost arm should fire on tiny dist16: %+v", dec)
	}
	if dec.dev16Arm {
		t.Fatalf("dev arm must be silenced by the high-variance noise guard: %+v", dec)
	}

	// Large distortion with tight 16->8 deviation: only the dev arm fires.
	me.dist16, me.dist8 = 900000, 880000 // dev16 = 22
	me.me8x8CostVariance = 0             // LOW guard: dev th <<= 2
	dec = realtimeDepthRemovalDecide(9, 120, &me)
	if dec.cost16Arm {
		t.Fatalf("cost arm must not fire on large dist16: %+v", dec)
	}
	if !dec.dev16Arm || !dec.below16 {
		t.Fatalf("dev arm should fire on tight deviation: %+v", dec)
	}
	if dec.dev16 != 22 {
		t.Fatalf("dev16 = %d, want 22", dec.dev16)
	}

	// Wide deviation: nothing fires.
	me.dist16, me.dist8 = 900000, 200000
	dec = realtimeDepthRemovalDecide(9, 120, &me)
	if dec.below16 {
		t.Fatalf("no arm should fire on wide deviation: %+v", dec)
	}

	// Level 0 disables the gate.
	me.dist16, me.dist8 = 100, 90
	dec = realtimeDepthRemovalDecide(0, 120, &me)
	if dec.below16 || dec.below32 || dec.below64 {
		t.Fatalf("level 0 must disable the gate: %+v", dec)
	}
}

// TestRealtimeDepthRemovalBelow32And64Arms exercises the Phase-2 threshold
// algebra (enc_mode_config.c:3196-3232) on synthetic distortion profiles.
// At level 9 / qIndex 120: fastLambda = 2664, costTh32 = costTh64 =
// RDCOST(2664, 8192, 512*8) = 566912, so the cost arms fire below
// dist ~4429; picture_qp = 30 gives qpFactor = 3, so dev32To16Th = 50*3 =
// 150 (LOW noise band) and dev32To8Th = 150*5/4 = 187.
func TestRealtimeDepthRemovalBelow32And64Arms(t *testing.T) {
	me := realtimeSBMultiSizeME{valid: true}

	// Tiny 32x32 cost: the 32 cost arm fires; wide 32->16 deviation keeps the
	// dev arm silent (dist16 tiny makes dev32To16 huge).
	me.dist64 = 1 << 24
	me.dist32, me.dist16, me.dist8 = 4000, 100, 90
	me.me8x8CostVariance = 1 << 30 // HIGH noise band zeroes the dev thresholds
	dec := realtimeDepthRemovalDecide(9, 120, &me)
	if !dec.cost32Arm || !dec.below32 {
		t.Fatalf("32 cost arm should fire on tiny dist32: %+v", dec)
	}
	if dec.dev32Arm {
		t.Fatalf("32 dev arm must be silenced by the high-variance noise guard: %+v", dec)
	}
	if dec.below64 {
		t.Fatalf("64 arm must not fire on large dist64: %+v", dec)
	}

	// Large distortions with tight 32->16 and 32->8 deviations: only the dual
	// dev arm fires. dev32To16 = 22 < 150, dev32To8 = 46 < 187.
	me.dist32, me.dist16, me.dist8 = 900000, 880000, 860000
	me.me8x8CostVariance = 0 // LOW band leaves dev32To16Th untouched
	dec = realtimeDepthRemovalDecide(9, 120, &me)
	if dec.cost32Arm {
		t.Fatalf("32 cost arm must not fire on large dist32: %+v", dec)
	}
	if !dec.dev32Arm || !dec.below32 {
		t.Fatalf("32 dev arm should fire on tight dual deviation: %+v", dec)
	}
	if dec.dev32To16 != 22 || dec.dev32To8 != 46 {
		t.Fatalf("dev32To16 = %d dev32To8 = %d, want 22/46", dec.dev32To16, dec.dev32To8)
	}

	// The middle noise band halves dev32To16Th (75, 32->8 th 93): still fires
	// at 22/46. picture_qp 30 divides the raw variance by 23.
	me.me8x8CostVariance = 30000 * 23 // 30000 after the qp division: middle band
	dec = realtimeDepthRemovalDecide(9, 120, &me)
	if !dec.dev32Arm {
		t.Fatalf("32 dev arm should survive the middle noise band: %+v", dec)
	}

	// The dual condition: tight 32->16 but wide 32->8 must NOT fire.
	me.dist32, me.dist16, me.dist8 = 900000, 880000, 200000
	me.me8x8CostVariance = 0
	dec = realtimeDepthRemovalDecide(9, 120, &me)
	if dec.dev32Arm || dec.below32 {
		t.Fatalf("32 dev arm requires BOTH deviations tight: %+v", dec)
	}

	// Tiny 64x64 cost: the (cost-only) 64 arm fires.
	me.dist64 = 4000
	dec = realtimeDepthRemovalDecide(9, 120, &me)
	if !dec.below64 {
		t.Fatalf("64 cost arm should fire on tiny dist64: %+v", dec)
	}

	// Level 14 (leaf ladder) raises the 32/64 multipliers to 16: a dist that
	// misses at level 9 (>= 4429) fires at level 14 (th 1091200 ->
	// dist < 8525).
	me = realtimeSBMultiSizeME{valid: true}
	me.dist64 = 1 << 24
	me.dist32, me.dist16, me.dist8 = 6000, 100, 90
	me.me8x8CostVariance = 1 << 30
	if dec = realtimeDepthRemovalDecide(9, 120, &me); dec.cost32Arm {
		t.Fatalf("32 cost arm must miss at level 9 with dist32=6000: %+v", dec)
	}
	if dec = realtimeDepthRemovalDecide(14, 120, &me); !dec.cost32Arm {
		t.Fatalf("32 cost arm should fire at level 14 with dist32=6000: %+v", dec)
	}
}
