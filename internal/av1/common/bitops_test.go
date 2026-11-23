package common

import (
	"math"
	"testing"
)

func TestMSB32MatchesLibaomLog2Test(t *testing.T) {
	for n := uint32(1); n < 10000; n++ {
		want := int(math.Floor(math.Log2(float64(n))))
		if got := MSB32(n); got != want {
			t.Fatalf("MSB32(%d)=%d want %d", n, got, want)
		}
	}
	for exponent := 2; exponent < 32; exponent++ {
		power := uint32(1) << exponent
		if got := MSB32(power - 1); got != exponent-1 {
			t.Fatalf("MSB32(%d)=%d want %d", power-1, got, exponent-1)
		}
		if got := MSB32(power); got != exponent {
			t.Fatalf("MSB32(%d)=%d want %d", power, got, exponent)
		}
		if got := MSB32(power + 1); got != exponent {
			t.Fatalf("MSB32(%d)=%d want %d", power+1, got, exponent)
		}
	}
	if got := MSB32(0); got != -1 {
		t.Fatalf("MSB32(0)=%d want -1", got)
	}
}

func TestBitOpsAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		if MSB32(0x80000000) != 31 {
			t.Fatal("bad bit op")
		}
	})
	if allocs != 0 {
		t.Fatalf("bit ops allocated: %f", allocs)
	}
}
