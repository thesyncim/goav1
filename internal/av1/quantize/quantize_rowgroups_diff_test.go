package quantize

import (
	"math/bits"
	"math/rand"
	"reflect"
	"testing"
	"unsafe"
)

// Independent derivation of the old quantize_b scalar rule. In particular it
// does not use invertQuant, quantizeScalarB, or either vector implementation.
func originalRowGroupLevel(c int32, step, dcStep int32, ts uint8) int16 {
	factor := int64(80)
	if dcStep < 148 {
		factor = 84
	}
	roundShift := func(v int64, shift uint8) int64 {
		if shift == 0 {
			return v
		}
		return (v + (int64(1) << (shift - 1))) >> shift
	}
	zbin := roundShift((factor*int64(step)+64)>>7, ts)
	v := int64(c)
	if v < 0 {
		v = -v
	}
	if v < zbin {
		return 0
	}
	v += roundShift((48*int64(step))>>7, ts)
	if v > 32767 {
		v = 32767
	}
	lg := 0
	for x := step; x > 1; x >>= 1 {
		lg++
	}
	multiplier := 1 + (int64(1)<<uint(16+lg))/int64(step) - 65536
	v = (((v*multiplier)>>16 + v) * (int64(1) << uint(16-lg))) >> (16 - ts)
	if v > 32767 {
		v = 32767
	}
	if c < 0 {
		v = -v
	}
	return int16(v)
}

// Models the frozen original two-pass result, including the SIMD int32
// intermediate before the int64 widening products and its later DC correction.
// The separate loop deliberately does not share the fused operation order.
func originalRowGroupDiff(coeff []int32, n int, live uint8, q Quantizer, ts uint8) ([]int16, int64, int64) {
	levels := make([]int16, n*n)
	for i, c := range coeff[:n*n] {
		if live&(1<<uint((i%n)/4)) == 0 {
			continue
		}
		step := q.AC
		if i == 0 {
			step = q.DC
		}
		levels[i] = originalRowGroupLevel(c, step, q.DC, ts)
	}
	var diff, rate int64
	for i, v := range levels {
		if v == 0 {
			continue
		}
		dq := (int32(v) * q.AC) >> ts
		delta := int32(2*coeff[i] - dq)
		diff += int64(dq) * int64(delta)
		mag := int64(v)
		if mag < 0 {
			mag = -mag
		}
		length := int64(0)
		for ; mag != 0; mag >>= 1 {
			length++
		}
		rate += (2 + length) << 9
	}
	if v := levels[0]; v != 0 {
		c := int64(coeff[0])
		a := c - ((int64(v) * int64(q.AC)) >> ts)
		d := c - ((int64(v) * int64(q.DC)) >> ts)
		diff += a*a - d*d
	}
	return levels, diff, rate
}

func TestQuantizeBRowGroupsDiffOriginalNumeric(t *testing.T) {
	rng := rand.New(rand.NewSource(1054))
	quants := []Quantizer{{4, 4}, {1, 1}, {147, 148}, {148, 147}, {120, 140}, {65535, 65535}, {1828, 4}, {4, 1828}}
	extremes := []int32{-2147483648, 2147483647, -32768, 32767, -1, 0, 1, -100000, 100000}
	handledCount := 0
	for _, n := range []int{8, 16, 32} {
		for mask := 0; mask < 1<<(n/4); mask++ {
			for qi, q := range quants {
				ts := uint8((mask + qi) % 3)
				coeff := make([]int32, n*n+8)
				for i := range coeff {
					switch (mask + qi) % 4 {
					case 0:
						coeff[i] = extremes[i%len(extremes)]
					case 1:
						coeff[i] = rng.Int31n(65536) - 32768
					case 2:
						coeff[i] = int32((i%9)-4) * q.AC / 4
					}
				}
				before := append([]int32(nil), coeff...)
				want, wantDiff, wantRate := originalRowGroupDiff(coeff, n, uint8(mask), q, ts)
				old := make([]int16, n*n)
				oldNZ := QuantizeBRowGroupsTrusted(old, coeff, n, uint8(mask), q, ts)
				if !reflect.DeepEqual(old, want) || oldNZ != (wantRate != 0) {
					t.Fatalf("original quantizer mismatch n=%d mask=%x q=%+v ts=%d", n, mask, q, ts)
				}
				got := make([]int16, n*n+8)
				for i := range got {
					got[i] = 0x5a5a
				}
				diff, rate, handled := TryQuantizeBRowGroupsDiffTrusted(got, coeff, n, uint8(mask), q, ts)
				wantHandled := quantizeBRowGroupsDiffImpl != nil && bits.Len32(uint32(q.AC))-1 >= int(ts)
				if handled != wantHandled {
					t.Fatalf("handled=%v want %v n=%d q=%+v ts=%d", handled, wantHandled, n, q, ts)
				}
				if handled {
					handledCount++
					if !reflect.DeepEqual(got[:n*n], want) || diff != wantDiff || rate != wantRate {
						t.Fatalf("n=%d mask=%x q=%+v ts=%d: diff/rate=%d/%d want=%d/%d levelsEqual=%v", n, mask, q, ts, diff, rate, wantDiff, wantRate, reflect.DeepEqual(got[:n*n], want))
					}
				} else {
					for _, v := range got[:n*n] {
						if v != 0x5a5a {
							t.Fatal("declined input was modified")
						}
					}
				}
				for _, v := range got[n*n:] {
					if v != 0x5a5a {
						t.Fatal("output tail changed")
					}
				}
				if !reflect.DeepEqual(coeff, before) {
					t.Fatal("input changed")
				}
			}
		}
	}
	// AC-only nonzero DC becomes zero after its separate quantizer, and the
	// converse. These exercise the old full-block nonzero-rescan corner.
	for _, q := range []Quantizer{{1828, 4}, {4, 1828}} {
		for _, dc := range []int32{-4, 4} {
			var coeff [64]int32
			var got [64]int16
			coeff[0] = dc
			want, wantDiff, wantRate := originalRowGroupDiff(coeff[:], 8, 3, q, 0)
			diff, rate, handled := TryQuantizeBRowGroupsDiffTrusted(got[:], coeff[:], 8, 3, q, 0)
			if handled {
				handledCount++
				if !reflect.DeepEqual(got[:], want) || diff != wantDiff || rate != wantRate {
					t.Fatalf("DC override q=%+v dc=%d: diff/rate=%d/%d want=%d/%d", q, dc, diff, rate, wantDiff, wantRate)
				}
			}
		}
	}
	if quantizeBRowGroupsDiffImpl != nil && handledCount == 0 {
		t.Fatal("fused backend did not execute")
	}
	t.Logf("fused cases executed: %d", handledCount)
}

func TestQuantizeBRowGroupsDiffDeclinesOverlapAndUnsupported(t *testing.T) {
	const n = 8
	for _, offset := range []int{0, 1, 4, 31, 63, 127} {
		backing := make([]int32, 256)
		for i := range backing {
			backing[i] = int32(i*137 + 50)
		}
		levels := unsafe.Slice((*int16)(unsafe.Pointer(&backing[0])), 2*len(backing))
		before := append([]int32(nil), backing...)
		if _, _, handled := TryQuantizeBRowGroupsDiffTrusted(levels[offset:offset+64], backing[:64], n, 3, Quantizer{120, 140}, 0); handled {
			t.Fatalf("overlap offset=%d accepted", offset)
		}
		if !reflect.DeepEqual(backing, before) {
			t.Fatalf("overlap offset=%d changed storage", offset)
		}
	}
	for _, tc := range []struct {
		n, qlen, clen int
		q             Quantizer
		ts            uint8
	}{
		{4, 64, 64, Quantizer{120, 140}, 0},
		{8, 63, 64, Quantizer{120, 140}, 0},
		{8, 64, 63, Quantizer{120, 140}, 0},
		{8, 64, 64, Quantizer{0, 140}, 0},
		{8, 64, 64, Quantizer{120, 65536}, 0},
		{8, 64, 64, Quantizer{120, 140}, 3},
		{8, 64, 64, Quantizer{1, 1}, 2},
	} {
		qcoeff, coeff := make([]int16, tc.qlen), make([]int32, tc.clen)
		for i := range qcoeff {
			qcoeff[i] = 123
		}
		before := append([]int16(nil), qcoeff...)
		if _, _, handled := TryQuantizeBRowGroupsDiffTrusted(qcoeff, coeff, tc.n, 0xff, tc.q, tc.ts); handled {
			t.Fatalf("unsupported input accepted: %+v", tc)
		}
		if !reflect.DeepEqual(qcoeff, before) {
			t.Fatal("declined output changed")
		}
	}
}

func TestQuantizeBRowGroupsDiffNoAllocation(t *testing.T) {
	var coeff [1024]int32
	var levels [1024]int16
	for i := range coeff {
		coeff[i] = int32(i*17 - 5000)
	}
	if got := testing.AllocsPerRun(10, func() {
		TryQuantizeBRowGroupsDiffTrusted(levels[:], coeff[:], 32, 0x55, Quantizer{120, 140}, 1)
	}); got != 0 {
		t.Fatalf("allocations=%v", got)
	}
}
