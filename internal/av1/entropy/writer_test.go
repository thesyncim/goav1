package entropy

import (
	"math/rand"
	"testing"
)

// These tests are the oracle round-trip gate for the range Writer: every stream
// it produces must decode, symbol-for-symbol, in Reader (the byte-exact od_ec_dec
// port). CDF adaptation is disabled on the decoder for the fixed-CDF tests so the
// icdf used to encode is reused verbatim to decode.

// buildICDF returns an AV1 inverse CDF (icdf[i] = 32768 - cumulative, monotone
// decreasing, icdf[nsyms-1] == 0) of length nsyms+1 with a zeroed count slot.
func buildICDF(t *testing.T, weights []int) []uint16 {
	t.Helper()
	total := 0
	for _, wgt := range weights {
		if wgt <= 0 {
			t.Fatalf("weight must be > 0, got %d", wgt)
		}
		total += wgt
	}
	n := len(weights)
	icdf := make([]uint16, n+1)
	cum := 0
	for i, wgt := range weights {
		cum += wgt
		c := (cum * CDFProbTop) / total
		if i == n-1 {
			c = CDFProbTop
		}
		icdf[i] = uint16(CDFProbTop - c)
	}
	for i := 1; i < n; i++ {
		if icdf[i] >= icdf[i-1] {
			t.Fatalf("icdf not strictly decreasing at %d: %v", i, icdf)
		}
	}
	if err := ValidateCDF(icdf, n); err != nil {
		t.Fatalf("ValidateCDF: %v", err)
	}
	return icdf
}

func TestWriterRoundTripBools(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const count = 5000
	vals := make([]int, count)
	probs := make([]uint16, count)
	w := NewWriter(make([]byte, 0, 1<<16))
	for i := range count {
		probs[i] = uint16(1 + rng.Intn(CDFProbTop-1))
		vals[i] = rng.Intn(2)
		w.WriteBoolQ15(vals[i], uint32(probs[i]))
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := NewReaderWithCDFUpdate(buf, false)
	for i := range count {
		got, err := r.ReadBoolQ15(probs[i])
		if err != nil {
			t.Fatalf("ReadBoolQ15[%d]: %v", i, err)
		}
		if int(got) != vals[i] {
			t.Fatalf("bool[%d] = %d, want %d (prob %d)", i, got, vals[i], probs[i])
		}
	}
}

func TestWriterRoundTripLiterals(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const count = 4000
	vals := make([]uint32, count)
	nbits := make([]int, count)
	w := NewWriter(make([]byte, 0, 1<<16))
	for i := range count {
		nbits[i] = 1 + rng.Intn(16)
		vals[i] = rng.Uint32() & ((1 << uint(nbits[i])) - 1)
		w.WriteLiteral(vals[i], nbits[i])
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := NewReaderWithCDFUpdate(buf, false)
	for i := range count {
		got, err := r.ReadBits(uint8(nbits[i]))
		if err != nil {
			t.Fatalf("ReadBits[%d]: %v", i, err)
		}
		if got != vals[i] {
			t.Fatalf("literal[%d] = %d, want %d (%d bits)", i, got, vals[i], nbits[i])
		}
	}
}

func TestWriterRoundTripSymbols(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	icdfs := make(map[int][]uint16)
	for nsyms := 2; nsyms <= MaxSymbols; nsyms++ {
		weights := make([]int, nsyms)
		for i := range weights {
			weights[i] = i + 1
		}
		icdfs[nsyms] = buildICDF(t, weights)
	}
	const count = 6000
	type rec struct{ nsyms, sym int }
	recs := make([]rec, count)
	w := NewWriter(make([]byte, 0, 1<<16))
	for i := range count {
		nsyms := 2 + rng.Intn(MaxSymbols-1)
		sym := rng.Intn(nsyms)
		recs[i] = rec{nsyms, sym}
		w.WriteSymbol(sym, icdfs[nsyms], nsyms)
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := NewReaderWithCDFUpdate(buf, false)
	for i := range count {
		cdf := append([]uint16(nil), icdfs[recs[i].nsyms]...)
		got, err := r.ReadSymbol(cdf, recs[i].nsyms)
		if err != nil {
			t.Fatalf("ReadSymbol[%d]: %v", i, err)
		}
		if got != recs[i].sym {
			t.Fatalf("symbol[%d] = %d, want %d (nsyms %d)", i, got, recs[i].sym, recs[i].nsyms)
		}
	}
}

func TestWriterRoundTripMixed(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	icdf := buildICDF(t, []int{3, 1, 4, 1, 5, 9})
	const count = 8000
	type op struct {
		kind  int
		a, b  int
		prob  uint16
		value uint32
	}
	ops := make([]op, count)
	w := NewWriter(make([]byte, 0, 1<<17))
	for i := range count {
		switch rng.Intn(3) {
		case 0:
			o := op{kind: 0, prob: uint16(1 + rng.Intn(CDFProbTop-1)), a: rng.Intn(2)}
			ops[i] = o
			w.WriteBoolQ15(o.a, uint32(o.prob))
		case 1:
			o := op{kind: 1, a: rng.Intn(6)}
			ops[i] = o
			w.WriteSymbol(o.a, icdf, 6)
		case 2:
			n := 1 + rng.Intn(16)
			o := op{kind: 2, b: n, value: rng.Uint32() & ((1 << uint(n)) - 1)}
			ops[i] = o
			w.WriteLiteral(o.value, n)
		}
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := NewReaderWithCDFUpdate(buf, false)
	for i := range count {
		o := ops[i]
		switch o.kind {
		case 0:
			got, err := r.ReadBoolQ15(o.prob)
			if err != nil || int(got) != o.a {
				t.Fatalf("mixed bool[%d] = %d err %v, want %d", i, got, err, o.a)
			}
		case 1:
			cdf := append([]uint16(nil), icdf...)
			got, err := r.ReadSymbol(cdf, 6)
			if err != nil || got != o.a {
				t.Fatalf("mixed symbol[%d] = %d err %v, want %d", i, got, err, o.a)
			}
		case 2:
			got, err := r.ReadBits(uint8(o.b))
			if err != nil || got != o.value {
				t.Fatalf("mixed literal[%d] = %d err %v, want %d", i, got, err, o.value)
			}
		}
	}
}

// TestWriterAdaptiveRoundTrip encodes with on-the-fly CDF adaptation and decodes
// with the decoder's adaptation enabled. Starting from the same default CDFs both
// sides evolve identically (WriteSymbolAdaptive and ReadSymbol share
// updateCDFWindow), so they agree on every symbol.
func TestWriterAdaptiveRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	const count = 8000
	encCDF := make(map[int][]uint16)
	decCDF := make(map[int][]uint16)
	for nsyms := 2; nsyms <= MaxSymbols; nsyms++ {
		c := make([]uint16, nsyms+1)
		if err := InitUniformCDF(c, nsyms); err != nil {
			t.Fatalf("InitUniformCDF: %v", err)
		}
		encCDF[nsyms] = c
		decCDF[nsyms] = append([]uint16(nil), c...)
	}
	type rec struct{ nsyms, sym int }
	recs := make([]rec, count)
	w := NewWriter(make([]byte, 0, 1<<16))
	for i := range count {
		nsyms := 2 + rng.Intn(MaxSymbols-1)
		sym := rng.Intn(nsyms)
		recs[i] = rec{nsyms, sym}
		w.WriteSymbolAdaptive(sym, encCDF[nsyms])
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := NewReaderWithCDFUpdate(buf, true)
	for i := range count {
		got, err := r.ReadSymbol(decCDF[recs[i].nsyms], recs[i].nsyms)
		if err != nil {
			t.Fatalf("ReadSymbol[%d]: %v", i, err)
		}
		if got != recs[i].sym {
			t.Fatalf("adaptive symbol[%d] = %d, want %d (nsyms %d)", i, got, recs[i].sym, recs[i].nsyms)
		}
	}
}

func TestWriterZeroAlloc(t *testing.T) {
	icdf := buildICDF(t, []int{2, 3, 5, 7})
	dst := make([]byte, 0, 1<<16)
	allocs := testing.AllocsPerRun(50, func() {
		w := NewWriter(dst)
		for i := range 2000 {
			w.WriteBoolQ15(i&1, 12000)
			w.WriteSymbol(i&3, icdf, 4)
			w.WriteLiteral(uint32(i), 8)
		}
		if _, err := w.Finish(); err != nil {
			t.Fatalf("finish: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("encode path allocated %v objects/run, want 0", allocs)
	}
}
