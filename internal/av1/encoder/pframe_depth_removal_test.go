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
	// Near-zero 16x16 cost: the absolute cost arm must fire at level 9.
	me.dist16, me.dist8 = 100, 90
	me.me8x8CostVariance = 1 << 30 // noise guard zeroes the dev threshold
	dec := realtimeDepthRemovalBelow16(9, 120, &me)
	if !dec.costArm || !dec.disallow {
		t.Fatalf("cost arm should fire on tiny dist16: %+v", dec)
	}
	if dec.devArm {
		t.Fatalf("dev arm must be silenced by the high-variance noise guard: %+v", dec)
	}

	// Large distortion with tight 16->8 deviation: only the dev arm fires.
	me.dist16, me.dist8 = 900000, 880000 // dev16 = 22
	me.me8x8CostVariance = 0             // LOW guard: dev th <<= 2
	dec = realtimeDepthRemovalBelow16(9, 120, &me)
	if dec.costArm {
		t.Fatalf("cost arm must not fire on large dist16: %+v", dec)
	}
	if !dec.devArm || !dec.disallow {
		t.Fatalf("dev arm should fire on tight deviation: %+v", dec)
	}
	if dec.dev16 != 22 {
		t.Fatalf("dev16 = %d, want 22", dec.dev16)
	}

	// Wide deviation: nothing fires.
	me.dist16, me.dist8 = 900000, 200000
	dec = realtimeDepthRemovalBelow16(9, 120, &me)
	if dec.disallow {
		t.Fatalf("no arm should fire on wide deviation: %+v", dec)
	}

	// Level 0 disables the gate.
	me.dist16, me.dist8 = 100, 90
	dec = realtimeDepthRemovalBelow16(0, 120, &me)
	if dec.disallow {
		t.Fatalf("level 0 must disable the gate: %+v", dec)
	}
}
