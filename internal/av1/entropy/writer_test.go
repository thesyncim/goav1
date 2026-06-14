package entropy

import (
	"errors"
	"math/rand"
	"testing"
	"unsafe"
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

func TestWriterHotStructSize(t *testing.T) {
	if size := unsafe.Sizeof(Writer{}); size > 64 {
		t.Fatalf("Writer size=%d max=64", size)
	}
}

func TestWriterNilDestinationFinishes(t *testing.T) {
	w := NewWriter(nil)
	for i := range 32 {
		w.WriteBit(i & 1)
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}
	if len(buf) == 0 {
		t.Fatal("nil destination produced no bytes")
	}
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

func TestWriteBitMatchesBoolQ15(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	generic := NewWriter(make([]byte, 0, 1<<16))
	specialized := NewWriter(make([]byte, 0, 1<<16))
	for i := range 5000 {
		bit := rng.Intn(2)
		generic.WriteBoolQ15(bit, 1<<14)
		specialized.WriteBit(bit)
		if generic.Tell() != specialized.Tell() {
			t.Fatalf("tell diverged at %d: generic=%d specialized=%d", i, generic.Tell(), specialized.Tell())
		}
	}
	genericBytes, err := generic.Finish()
	if err != nil {
		t.Fatalf("generic finish: %v", err)
	}
	specializedBytes, err := specialized.Finish()
	if err != nil {
		t.Fatalf("specialized finish: %v", err)
	}
	if string(genericBytes) != string(specializedBytes) {
		t.Fatalf("encoded bytes diverged: generic=%x specialized=%x", genericBytes, specializedBytes)
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

func TestWriteBinaryCDFTrustedMatchesWriteCDF(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	var genericCDF, specializedCDF CDF
	if err := genericCDF.Init([]uint16{18000}); err != nil {
		t.Fatal(err)
	}
	specializedCDF = genericCDF

	generic := NewWriter(make([]byte, 0, 1<<16))
	specialized := NewWriter(make([]byte, 0, 1<<16))
	for i := range 4000 {
		sym := rng.Intn(2)
		generic.WriteCDF(&genericCDF, sym)
		specialized.WriteBinaryCDFTrusted(&specializedCDF, sym)
		if genericCDF != specializedCDF {
			t.Fatalf("cdf diverged after symbol %d: generic=%v specialized=%v", i, genericCDF.Values(), specializedCDF.Values())
		}
	}
	genericBytes, err := generic.Finish()
	if err != nil {
		t.Fatalf("generic finish: %v", err)
	}
	specializedBytes, err := specialized.Finish()
	if err != nil {
		t.Fatalf("specialized finish: %v", err)
	}
	if string(genericBytes) != string(specializedBytes) {
		t.Fatalf("encoded bytes diverged: generic=%x specialized=%x", genericBytes, specializedBytes)
	}
}

func TestWriteCDF3MatchesWriteCDF(t *testing.T) {
	rng := rand.New(rand.NewSource(39))
	var genericCDF, specializedCDF CDF
	if err := genericCDF.Init([]uint16{9000, 22000}); err != nil {
		t.Fatal(err)
	}
	specializedCDF = genericCDF

	generic := NewWriter(make([]byte, 0, 1<<16))
	specialized := NewWriter(make([]byte, 0, 1<<16))
	for i := range 4000 {
		sym := rng.Intn(3)
		generic.WriteCDF(&genericCDF, sym)
		specialized.WriteCDF3(&specializedCDF, sym)
		if genericCDF != specializedCDF {
			t.Fatalf("cdf diverged after symbol %d: generic=%v specialized=%v", i, genericCDF.Values(), specializedCDF.Values())
		}
	}
	genericBytes, err := generic.Finish()
	if err != nil {
		t.Fatalf("generic finish: %v", err)
	}
	specializedBytes, err := specialized.Finish()
	if err != nil {
		t.Fatalf("specialized finish: %v", err)
	}
	if string(genericBytes) != string(specializedBytes) {
		t.Fatalf("encoded bytes diverged: generic=%x specialized=%x", genericBytes, specializedBytes)
	}
}

func TestWriteCDF4MatchesWriteCDF(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	var genericCDF, specializedCDF CDF
	if err := genericCDF.Init([]uint16{4096, 15000, 28000}); err != nil {
		t.Fatal(err)
	}
	specializedCDF = genericCDF

	generic := NewWriter(make([]byte, 0, 1<<16))
	specialized := NewWriter(make([]byte, 0, 1<<16))
	for i := range 4000 {
		sym := rng.Intn(4)
		generic.WriteCDF(&genericCDF, sym)
		specialized.WriteCDF4(&specializedCDF, sym)
		if genericCDF != specializedCDF {
			t.Fatalf("cdf diverged after symbol %d: generic=%v specialized=%v", i, genericCDF.Values(), specializedCDF.Values())
		}
	}
	genericBytes, err := generic.Finish()
	if err != nil {
		t.Fatalf("generic finish: %v", err)
	}
	specializedBytes, err := specialized.Finish()
	if err != nil {
		t.Fatalf("specialized finish: %v", err)
	}
	if string(genericBytes) != string(specializedBytes) {
		t.Fatalf("encoded bytes diverged: generic=%x specialized=%x", genericBytes, specializedBytes)
	}
}

func TestWriteCDF5MatchesWriteCDF(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	var genericCDF, specializedCDF CDF
	if err := genericCDF.Init([]uint16{3000, 9000, 17000, 26000}); err != nil {
		t.Fatal(err)
	}
	specializedCDF = genericCDF

	generic := NewWriter(make([]byte, 0, 1<<16))
	specialized := NewWriter(make([]byte, 0, 1<<16))
	for i := range 4000 {
		sym := rng.Intn(5)
		generic.WriteCDF(&genericCDF, sym)
		specialized.WriteCDF5(&specializedCDF, sym)
		if genericCDF != specializedCDF {
			t.Fatalf("cdf diverged after symbol %d: generic=%v specialized=%v", i, genericCDF.Values(), specializedCDF.Values())
		}
	}
	genericBytes, err := generic.Finish()
	if err != nil {
		t.Fatalf("generic finish: %v", err)
	}
	specializedBytes, err := specialized.Finish()
	if err != nil {
		t.Fatalf("specialized finish: %v", err)
	}
	if string(genericBytes) != string(specializedBytes) {
		t.Fatalf("encoded bytes diverged: generic=%x specialized=%x", genericBytes, specializedBytes)
	}
}

func TestWriteCDF7MatchesWriteCDF(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	var genericCDF, specializedCDF CDF
	if err := genericCDF.Init([]uint16{2000, 6000, 12000, 18000, 25000, 30000}); err != nil {
		t.Fatal(err)
	}
	specializedCDF = genericCDF

	generic := NewWriter(make([]byte, 0, 1<<16))
	specialized := NewWriter(make([]byte, 0, 1<<16))
	for i := range 4000 {
		sym := rng.Intn(7)
		generic.WriteCDF(&genericCDF, sym)
		specialized.WriteCDF7(&specializedCDF, sym)
		if genericCDF != specializedCDF {
			t.Fatalf("cdf diverged after symbol %d: generic=%v specialized=%v", i, genericCDF.Values(), specializedCDF.Values())
		}
	}
	genericBytes, err := generic.Finish()
	if err != nil {
		t.Fatalf("generic finish: %v", err)
	}
	specializedBytes, err := specialized.Finish()
	if err != nil {
		t.Fatalf("specialized finish: %v", err)
	}
	if string(genericBytes) != string(specializedBytes) {
		t.Fatalf("encoded bytes diverged: generic=%x specialized=%x", genericBytes, specializedBytes)
	}
}

func TestCountingWriterTellMatchesWriter(t *testing.T) {
	rng := rand.New(rand.NewSource(83))
	icdf := buildICDF(t, []int{3, 1, 4, 1})
	var normalBinaryCDF, countBinaryCDF, bitCounterBinaryCDF CDF
	if err := normalBinaryCDF.Init([]uint16{18000}); err != nil {
		t.Fatal(err)
	}
	countBinaryCDF = normalBinaryCDF
	bitCounterBinaryCDF = normalBinaryCDF
	var normalCDF, countCDF, bitCounterCDF CDF
	if err := normalCDF.Init([]uint16{4096, 15000, 28000}); err != nil {
		t.Fatal(err)
	}
	countCDF = normalCDF
	bitCounterCDF = normalCDF
	var normalCDF3, countCDF3, bitCounterCDF3 CDF
	if err := normalCDF3.Init([]uint16{9000, 22000}); err != nil {
		t.Fatal(err)
	}
	countCDF3 = normalCDF3
	bitCounterCDF3 = normalCDF3
	var normalCDF5, countCDF5, bitCounterCDF5 CDF
	if err := normalCDF5.Init([]uint16{3000, 9000, 17000, 26000}); err != nil {
		t.Fatal(err)
	}
	countCDF5 = normalCDF5
	bitCounterCDF5 = normalCDF5
	var normalCDF7, countCDF7, bitCounterCDF7 CDF
	if err := normalCDF7.Init([]uint16{2000, 6000, 12000, 18000, 25000, 30000}); err != nil {
		t.Fatal(err)
	}
	countCDF7 = normalCDF7
	bitCounterCDF7 = normalCDF7

	normal := NewWriter(make([]byte, 0, 1<<16))
	count := NewCountingWriter()
	bitCounter := NewBitCounter()
	for i := range 5000 {
		switch rng.Intn(8) {
		case 0:
			bit := rng.Intn(2)
			p := uint32(1 + rng.Intn(CDFProbTop-1))
			normal.WriteBoolQ15(bit, p)
			count.WriteBoolQ15(bit, p)
			bitCounter.WriteBoolQ15(bit, p)
		case 1:
			bit := rng.Intn(2)
			normal.WriteBit(bit)
			count.WriteBit(bit)
			bitCounter.WriteBit(bit)
		case 2:
			sym := rng.Intn(4)
			normal.WriteCDF4(&normalCDF, sym)
			count.WriteCDF4(&countCDF, sym)
			bitCounter.WriteCDF4(&bitCounterCDF, sym)
			if normalCDF != countCDF || normalCDF != bitCounterCDF {
				t.Fatalf("cdf diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalCDF.Values(), countCDF.Values(), bitCounterCDF.Values())
			}
		case 3:
			sym := rng.Intn(3)
			normal.WriteCDF3(&normalCDF3, sym)
			count.WriteCDF3(&countCDF3, sym)
			bitCounter.WriteCDF3(&bitCounterCDF3, sym)
			if normalCDF3 != countCDF3 || normalCDF3 != bitCounterCDF3 {
				t.Fatalf("cdf3 diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalCDF3.Values(), countCDF3.Values(), bitCounterCDF3.Values())
			}
		case 4:
			sym := rng.Intn(2)
			normal.WriteBinaryCDFTrusted(&normalBinaryCDF, sym)
			count.WriteBinaryCDFTrusted(&countBinaryCDF, sym)
			bitCounter.WriteBinaryCDFTrusted(&bitCounterBinaryCDF, sym)
			if normalBinaryCDF != countBinaryCDF || normalBinaryCDF != bitCounterBinaryCDF {
				t.Fatalf("binary cdf diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalBinaryCDF.Values(), countBinaryCDF.Values(), bitCounterBinaryCDF.Values())
			}
		case 5:
			sym := rng.Intn(5)
			normal.WriteCDF5(&normalCDF5, sym)
			count.WriteCDF5(&countCDF5, sym)
			bitCounter.WriteCDF5(&bitCounterCDF5, sym)
			if normalCDF5 != countCDF5 || normalCDF5 != bitCounterCDF5 {
				t.Fatalf("cdf5 diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalCDF5.Values(), countCDF5.Values(), bitCounterCDF5.Values())
			}
		case 6:
			sym := rng.Intn(7)
			normal.WriteCDF7(&normalCDF7, sym)
			count.WriteCDF7(&countCDF7, sym)
			bitCounter.WriteCDF7(&bitCounterCDF7, sym)
			if normalCDF7 != countCDF7 || normalCDF7 != bitCounterCDF7 {
				t.Fatalf("cdf7 diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalCDF7.Values(), countCDF7.Values(), bitCounterCDF7.Values())
			}
		default:
			sym := rng.Intn(4)
			normal.WriteSymbol(sym, icdf, 4)
			count.WriteSymbol(sym, icdf, 4)
			bitCounter.WriteSymbol(sym, icdf, 4)
		}
		if normal.Tell() != count.Tell() || normal.Tell() != bitCounter.Tell() {
			t.Fatalf("tell diverged after op %d: normal=%d count=%d bitCounter=%d", i, normal.Tell(), count.Tell(), bitCounter.Tell())
		}
	}
	if _, err := count.Finish(); !errors.Is(err, ErrCountOnlyFinish) {
		t.Fatalf("count-only finish err = %v, want %v", err, ErrCountOnlyFinish)
	}
	if _, err := normal.Finish(); err != nil {
		t.Fatalf("normal finish: %v", err)
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

func BenchmarkWriterBinaryCDFStream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(69))
	for i := range syms {
		if rng.Intn(8) == 0 {
			syms[i] = 1
		}
	}
	var base CDF
	if err := base.Init([]uint16{18000}); err != nil {
		b.Fatal(err)
	}
	buf := make([]byte, 0, 1<<16)
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("trusted", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteBinaryCDFTrusted(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkWriterCDF3Stream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(70))
	for i := range syms {
		switch r := rng.Intn(16); {
		case r < 10:
			syms[i] = 0
		case r < 14:
			syms[i] = 1
		default:
			syms[i] = 2
		}
	}
	var base CDF
	if err := base.Init([]uint16{9000, 22000}); err != nil {
		b.Fatal(err)
	}
	buf := make([]byte, 0, 1<<16)
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("specialized", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF3(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkWriterCDF4Stream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(71))
	for i := range syms {
		switch r := rng.Intn(16); {
		case r < 9:
			syms[i] = 0
		case r < 13:
			syms[i] = 1
		case r < 15:
			syms[i] = 2
		default:
			syms[i] = 3
		}
	}
	var base CDF
	if err := base.Init([]uint16{4096, 15000, 28000}); err != nil {
		b.Fatal(err)
	}
	buf := make([]byte, 0, 1<<16)
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("specialized", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF4(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkWriterCDF5Stream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(72))
	for i := range syms {
		switch r := rng.Intn(24); {
		case r < 9:
			syms[i] = 0
		case r < 15:
			syms[i] = 1
		case r < 19:
			syms[i] = 2
		case r < 22:
			syms[i] = 3
		default:
			syms[i] = 4
		}
	}
	var base CDF
	if err := base.Init([]uint16{3000, 9000, 17000, 26000}); err != nil {
		b.Fatal(err)
	}
	buf := make([]byte, 0, 1<<16)
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("specialized", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF5(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

var bitCounterCDF4BenchSink int

func BenchmarkWriterCDF7Stream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(72))
	for i := range syms {
		switch r := rng.Intn(32); {
		case r < 11:
			syms[i] = 0
		case r < 18:
			syms[i] = 1
		case r < 23:
			syms[i] = 2
		case r < 27:
			syms[i] = 3
		case r < 30:
			syms[i] = 4
		case r < 31:
			syms[i] = 5
		default:
			syms[i] = 6
		}
	}
	var base CDF
	if err := base.Init([]uint16{2000, 6000, 12000, 18000, 25000, 30000}); err != nil {
		b.Fatal(err)
	}
	buf := make([]byte, 0, 1<<16)
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("specialized", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			cdf := base
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteCDF7(&cdf, sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkWriterBitStream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(73))
	for i := range syms {
		syms[i] = rng.Intn(2)
	}
	buf := make([]byte, 0, 1<<16)
	b.SetBytes(streamLen)
	b.Run("writer", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			w := NewWriter(buf[:0])
			for _, sym := range syms {
				w.WriteBit(sym)
			}
			if _, err := w.Finish(); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("counter", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteBit(sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
}

func BenchmarkBitCounterCDF7Stream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(72))
	for i := range syms {
		switch r := rng.Intn(32); {
		case r < 11:
			syms[i] = 0
		case r < 18:
			syms[i] = 1
		case r < 23:
			syms[i] = 2
		case r < 27:
			syms[i] = 3
		case r < 30:
			syms[i] = 4
		case r < 31:
			syms[i] = 5
		default:
			syms[i] = 6
		}
	}
	var base CDF
	if err := base.Init([]uint16{2000, 6000, 12000, 18000, 25000, 30000}); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			cdf := base
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
	b.Run("specialized", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			cdf := base
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteCDF7(&cdf, sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
}

func BenchmarkBitCounterCDF5Stream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(72))
	for i := range syms {
		switch r := rng.Intn(24); {
		case r < 9:
			syms[i] = 0
		case r < 15:
			syms[i] = 1
		case r < 19:
			syms[i] = 2
		case r < 22:
			syms[i] = 3
		default:
			syms[i] = 4
		}
	}
	var base CDF
	if err := base.Init([]uint16{3000, 9000, 17000, 26000}); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			cdf := base
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
	b.Run("specialized", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			cdf := base
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteCDF5(&cdf, sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
}

func BenchmarkBitCounterCDF3Stream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(70))
	for i := range syms {
		switch r := rng.Intn(16); {
		case r < 10:
			syms[i] = 0
		case r < 14:
			syms[i] = 1
		default:
			syms[i] = 2
		}
	}
	var base CDF
	if err := base.Init([]uint16{9000, 22000}); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			cdf := base
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
	b.Run("specialized", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			cdf := base
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteCDF3(&cdf, sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
}

func BenchmarkBitCounterCDF4Stream(b *testing.B) {
	const streamLen = 4096
	syms := make([]int, streamLen)
	rng := rand.New(rand.NewSource(71))
	for i := range syms {
		switch r := rng.Intn(16); {
		case r < 9:
			syms[i] = 0
		case r < 13:
			syms[i] = 1
		case r < 15:
			syms[i] = 2
		default:
			syms[i] = 3
		}
	}
	var base CDF
	if err := base.Init([]uint16{4096, 15000, 28000}); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(streamLen)
	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			cdf := base
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteCDF(&cdf, sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
	b.Run("specialized", func(b *testing.B) {
		b.ReportAllocs()
		sum := 0
		for b.Loop() {
			cdf := base
			w := NewBitCounter()
			for _, sym := range syms {
				w.WriteCDF4(&cdf, sym)
			}
			sum += w.Tell()
		}
		bitCounterCDF4BenchSink = sum
	})
}
