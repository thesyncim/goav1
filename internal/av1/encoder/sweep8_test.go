package encoder

import (
	"math/rand"
	"testing"
)

// TestSweepSearchPoint8MatchesPoint1 checks the eight-position batch
// (open_loop_me_get_eight_search_point_results_block) against eight
// single-position searches (open_loop_me_get_search_point_results_block).
func TestSweepSearchPoint8MatchesPoint1(t *testing.T) {
	const stride = 160
	rng := rand.New(rand.NewSource(5))
	src := make([]byte, stride*100)
	ref := make([]byte, stride*100)
	for trial := 0; trial < 24; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		if trial%2 == 1 { // structured: ref is src shifted, so argmins move
			for i := range ref {
				ref[i] = src[(i+3)%len(src)] + byte(rng.Intn(5))
			}
		}
		var a, b realtimeSBMultiSizeME
		a.reset()
		b.reset()
		srcBase, refBase := 10*stride+20, 9*stride+14
		for y := 0; y < 3; y++ {
			rb := refBase + y*stride
			if !a.sweepSearchPoint8(src, ref, stride, srcBase, rb, -6, y-1) {
				t.Fatal("batch declined in-bounds reads")
			}
			for k := range 8 {
				b.sweepSearchPoint1(src, stride, ref, stride, srcBase, rb+k, -6+k, y-1)
			}
		}
		if a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: batch differs: %v %v vs %v %v", trial, a.bestSAD[:6], a.bestMV[:6], b.bestSAD[:6], b.bestMV[:6])
		}
	}
}

// TestSVTAllSAD8x8_16x16SubMatchesScalar is the differential test of the
// svt_ext_all_sad_calculation_8x8_16x16 kernel against its scalar
// definition, on exact-length slices (the kernel's last read is the slice
// end), with random running bests and ties.
func TestSVTAllSAD8x8_16x16SubMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const stride = 96
	src := make([]byte, 64*stride)
	ref := make([]byte, 64*stride)
	for trial := 0; trial < 160; trial++ {
		fillMEPattern(rng, src, trial)
		fillMEPattern(rng, ref, trial/4)
		s := src[:63*stride+64]
		ro := rng.Intn(stride - 71)
		r := ref[ro : ro+63*stride+71]
		var a realtimeSBMultiSizeME
		a.reset()
		for i := range a.bestSAD {
			// Random running bests, including ties with the new SADs.
			if trial%3 != 0 {
				a.bestSAD[i] = uint32(rng.Intn(20000))
				a.bestMV[i] = [2]int16{int16(rng.Intn(9) - 4), 1}
			}
		}
		b := a
		mvX := rng.Intn(40) - 20
		if trial%7 == 0 {
			mvX = -3 // positions cross zero: the x half wraps in 16 bits
		}
		mvY := rng.Intn(9) - 4
		var gA, gB [16][8]uint32
		svtAllSAD8x8_16x16Sub(&a, s, r, stride, mvX, mvY, &gA)
		svtAllSAD8x8_16x16SubScalar(&b, s, r, stride, mvX, mvY, &gB)
		if gA != gB || a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: kernel and scalar differ", trial)
		}
		a.eightSAD32x32_64x64(&gA, mvX, mvY)
		b.eightSAD32x32_64x64Scalar(&gB, mvX, mvY)
		if a.bestSAD != b.bestSAD || a.bestMV != b.bestMV {
			t.Fatalf("trial %d: 32x32/64x64 differ", trial)
		}
	}
}

// eightSAD32x32_64x64Scalar is svt_ext_eight_sad_calculation_32x32_64x64_c.
func (me *realtimeSBMultiSizeME) eightSAD32x32_64x64Scalar(sad16 *[16][8]uint32, mvX, mvY int) {
	for k := range 8 {
		var s64 uint32
		for q := range 4 {
			s := sad16[4*q][k] + sad16[4*q+1][k] + sad16[4*q+2][k] + sad16[4*q+3][k]
			me.update(1+q, s, mvX+k, mvY)
			s64 += s
		}
		me.update(0, s64, mvX+k, mvY)
	}
}

func BenchmarkSweepSearchPoint8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	const stride = 1440
	src := make([]byte, stride*80)
	ref := make([]byte, stride*80)
	fillMEPattern(rng, src, 0)
	fillMEPattern(rng, ref, 0)
	var me realtimeSBMultiSizeME
	for b.Loop() {
		me.reset()
		me.sweepSearchPoint8(src, ref, stride, 4*stride+100, 3*stride+96, -4, -1)
	}
}
