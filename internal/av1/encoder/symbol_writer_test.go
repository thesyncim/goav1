package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

// These tests are the oracle round-trip gate for the range encoder: every stream
// produced by symbolWriter must decode, symbol-for-symbol, in the decoder's
// entropy.Reader (a faithful od_ec_dec port that is byte-exact against libaom).
// CDF adaptation is disabled on the decoder so the fixed icdf used to encode is
// also used verbatim to decode (the encoder's adaptation pass is a later step).

// buildICDF returns an AV1 inverse CDF (icdf[i] = 32768 - cumulative_freq, monotone
// decreasing, icdf[nsyms-1] == 0) of length nsyms+1 with the trailing adaptation
// count slot zeroed, from per-symbol weights (all > 0).
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
		c := (cum * cdfProbTopEnc) / total
		if i == n-1 {
			c = cdfProbTopEnc
		}
		icdf[i] = uint16(cdfProbTopEnc - c)
	}
	// Validate strictly-decreasing with a nonzero gap so every symbol has mass.
	for i := 1; i < n; i++ {
		if icdf[i] >= icdf[i-1] {
			t.Fatalf("icdf not strictly decreasing at %d: %v", i, icdf)
		}
	}
	if icdf[n-1] != 0 {
		t.Fatalf("icdf terminator != 0: %v", icdf)
	}
	if err := entropy.ValidateCDF(icdf, n); err != nil {
		t.Fatalf("ValidateCDF: %v", err)
	}
	return icdf
}

func TestSymbolWriterRoundTripBools(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const count = 5000
	vals := make([]int, count)
	probs := make([]uint16, count)
	w := newSymbolWriter(make([]byte, 0, 1<<16))
	for i := range count {
		probs[i] = uint16(1 + rng.Intn(cdfProbTopEnc-1)) // [1, 32767]
		vals[i] = rng.Intn(2)
		w.writeBoolQ15(vals[i], uint32(probs[i]))
	}
	buf, err := w.finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := entropy.NewReaderWithCDFUpdate(buf, false)
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

func TestSymbolWriterRoundTripLiterals(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const count = 4000
	vals := make([]uint32, count)
	nbits := make([]int, count)
	w := newSymbolWriter(make([]byte, 0, 1<<16))
	for i := range count {
		nbits[i] = 1 + rng.Intn(16) // 1..16 bits
		vals[i] = rng.Uint32() & ((1 << uint(nbits[i])) - 1)
		w.writeLiteral(vals[i], nbits[i])
	}
	buf, err := w.finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := entropy.NewReaderWithCDFUpdate(buf, false)
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

func TestSymbolWriterRoundTripSymbols(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	// One representative icdf per alphabet size 2..16, both uniform-ish and skewed.
	icdfs := make(map[int][]uint16)
	for nsyms := 2; nsyms <= entropyMaxSymbols(); nsyms++ {
		weights := make([]int, nsyms)
		for i := range weights {
			weights[i] = i + 1 // skewed: increasing probability
		}
		icdfs[nsyms] = buildICDF(t, weights)
	}

	const count = 6000
	type rec struct {
		nsyms int
		sym   int
	}
	recs := make([]rec, count)
	w := newSymbolWriter(make([]byte, 0, 1<<16))
	for i := range count {
		nsyms := 2 + rng.Intn(entropyMaxSymbols()-1)
		sym := rng.Intn(nsyms)
		recs[i] = rec{nsyms: nsyms, sym: sym}
		w.writeSymbol(sym, icdfs[nsyms], nsyms)
	}
	buf, err := w.finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := entropy.NewReaderWithCDFUpdate(buf, false)
	for i := range count {
		cdf := append([]uint16(nil), icdfs[recs[i].nsyms]...) // ReadSymbol may touch count slot
		got, err := r.ReadSymbol(cdf, recs[i].nsyms)
		if err != nil {
			t.Fatalf("ReadSymbol[%d]: %v", i, err)
		}
		if got != recs[i].sym {
			t.Fatalf("symbol[%d] = %d, want %d (nsyms %d)", i, got, recs[i].sym, recs[i].nsyms)
		}
	}
}

// TestSymbolWriterRoundTripMixed interleaves bools, symbols, and literals in a
// single stream and decodes them in the same order — the realistic tile-coding
// scenario where the range coder state is shared across symbol kinds.
func TestSymbolWriterRoundTripMixed(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	icdf := buildICDF(t, []int{3, 1, 4, 1, 5, 9}) // 6-symbol skewed alphabet

	const count = 8000
	type op struct {
		kind  int // 0 bool, 1 symbol, 2 literal
		a, b  int
		prob  uint16
		value uint32
	}
	ops := make([]op, count)
	w := newSymbolWriter(make([]byte, 0, 1<<17))
	for i := range count {
		switch k := rng.Intn(3); k {
		case 0:
			o := op{kind: 0, prob: uint16(1 + rng.Intn(cdfProbTopEnc-1)), a: rng.Intn(2)}
			ops[i] = o
			w.writeBoolQ15(o.a, uint32(o.prob))
		case 1:
			o := op{kind: 1, a: rng.Intn(6)}
			ops[i] = o
			w.writeSymbol(o.a, icdf, 6)
		case 2:
			n := 1 + rng.Intn(16)
			o := op{kind: 2, b: n, value: rng.Uint32() & ((1 << uint(n)) - 1)}
			ops[i] = o
			w.writeLiteral(o.value, n)
		}
	}
	buf, err := w.finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	r := entropy.NewReaderWithCDFUpdate(buf, false)
	for i := range count {
		o := ops[i]
		switch o.kind {
		case 0:
			got, err := r.ReadBoolQ15(o.prob)
			if err != nil {
				t.Fatalf("mixed bool[%d]: %v", i, err)
			}
			if int(got) != o.a {
				t.Fatalf("mixed bool[%d] = %d, want %d", i, got, o.a)
			}
		case 1:
			cdf := append([]uint16(nil), icdf...)
			got, err := r.ReadSymbol(cdf, 6)
			if err != nil {
				t.Fatalf("mixed symbol[%d]: %v", i, err)
			}
			if got != o.a {
				t.Fatalf("mixed symbol[%d] = %d, want %d", i, got, o.a)
			}
		case 2:
			got, err := r.ReadBits(uint8(o.b))
			if err != nil {
				t.Fatalf("mixed literal[%d]: %v", i, err)
			}
			if got != o.value {
				t.Fatalf("mixed literal[%d] = %d, want %d", i, got, o.value)
			}
		}
	}
}

// TestSymbolWriterZeroAlloc proves the hot encode path is allocation-free when the
// caller-owned buffer is large enough (no growth). finish reslices in place.
func TestSymbolWriterZeroAlloc(t *testing.T) {
	icdf := buildICDF(t, []int{2, 3, 5, 7})
	dst := make([]byte, 0, 1<<16)
	allocs := testing.AllocsPerRun(50, func() {
		w := newSymbolWriter(dst)
		for i := range 2000 {
			w.writeBoolQ15(i&1, 12000)
			w.writeSymbol(i&3, icdf, 4)
			w.writeLiteral(uint32(i), 8)
		}
		if _, err := w.finish(); err != nil {
			t.Fatalf("finish: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("encode path allocated %v objects/run, want 0", allocs)
	}
}

// entropyMaxSymbols mirrors entropy.MaxSymbols without importing the constant name
// transitively; the alphabet cap is 16.
func entropyMaxSymbols() int { return 16 }
