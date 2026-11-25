package entropy

import (
	"bytes"
	"errors"
	"math/bits"
	"math/rand"
	randv2 "math/rand/v2"
	"sort"
	"strconv"
	"testing"
	"unsafe"
)

func TestCDFResetCountTrusted(t *testing.T) {
	var unused CDF
	ResetCDFCountTrusted(&unused)
	if unused != (CDF{}) {
		t.Fatal("unused CDF changed")
	}
	for symbols := 2; symbols <= MaxSymbols; symbols++ {
		var cdf CDF
		if err := cdf.InitUniform(symbols); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 63; i++ {
			if err := cdf.Update((i*7 + 1) % symbols); err != nil {
				t.Fatal(err)
			}
		}
		for i := symbols + 1; i < len(cdf.values); i++ {
			cdf.values[i] = 0xbeef
		}
		want := cdf
		if err := want.ResetCount(); err != nil {
			t.Fatal(err)
		}
		ResetCDFCountTrusted(&cdf)
		if cdf != want {
			t.Fatalf("symbols=%d: trusted reset changed probabilities or inactive storage", symbols)
		}
	}
}

func TestCDFResetCountValidatesWritableValues(t *testing.T) {
	for _, mutate := range []func([]uint16){
		func(v []uint16) { v[4] = MaxCDFCount + 1 },
		func(v []uint16) { v[3] = 1 },
		func(v []uint16) { v[0] = 0 },
	} {
		var cdf CDF
		if err := cdf.InitUniform(4); err != nil {
			t.Fatal(err)
		}
		if err := cdf.Update(2); err != nil {
			t.Fatal(err)
		}
		mutate(cdf.Values())
		before := cdf
		if err := cdf.ResetCount(); !errors.Is(err, ErrInvalidCDF) {
			t.Fatalf("invalid writable values: got %v", err)
		}
		if cdf != before {
			t.Fatal("invalid CDF was changed before reporting its error")
		}
	}
}

func TestInitCDF(t *testing.T) {
	var cdf [CDFSize4]uint16
	if err := InitCDF(cdf[:], []uint16{8192, 16384, 24576}); err != nil {
		t.Fatal(err)
	}
	want := [CDFSize4]uint16{24576, 16384, 8192, 0, 0}
	if cdf != want {
		t.Fatalf("cdf=%v want %v", cdf, want)
	}
	if err := ValidateCDF(cdf[:], 4); err != nil {
		t.Fatal(err)
	}
}

func TestInitUniformCDF(t *testing.T) {
	var cdf [CDFSize4]uint16
	if err := InitUniformCDF(cdf[:], 4); err != nil {
		t.Fatal(err)
	}
	want := [CDFSize4]uint16{24576, 16384, 8192, 0, 0}
	if cdf != want {
		t.Fatalf("cdf=%v want %v", cdf, want)
	}
}

func TestCDFStateInit(t *testing.T) {
	var cdf CDF
	if err := cdf.Init([]uint16{8192, 16384, 24576}); err != nil {
		t.Fatal(err)
	}
	want := []uint16{24576, 16384, 8192, 0, 0}
	assertCDFValues(t, cdf.Values(), want)
	if cdf.Symbols() != 4 {
		t.Fatalf("symbols=%d want 4", cdf.Symbols())
	}
	if err := cdf.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCDFStateInitUniform(t *testing.T) {
	var cdf CDF
	if err := cdf.InitUniform(4); err != nil {
		t.Fatal(err)
	}
	want := []uint16{24576, 16384, 8192, 0, 0}
	assertCDFValues(t, cdf.Values(), want)
	if cdf.Symbols() != 4 {
		t.Fatalf("symbols=%d want 4", cdf.Symbols())
	}
	if err := cdf.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCDFStateInitDefaultDelta(t *testing.T) {
	var cdf CDF
	if err := cdf.InitDefaultDelta(); err != nil {
		t.Fatal(err)
	}
	assertCDFValues(t, cdf.Values(), []uint16{4608, 648, 91, 0, 0})
	if cdf.Symbols() != DeltaSmall+1 {
		t.Fatalf("symbols=%d want %d", cdf.Symbols(), DeltaSmall+1)
	}
	if err := cdf.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCDFStateCopyUpdateReset(t *testing.T) {
	var src CDF
	if err := src.InitUniform(2); err != nil {
		t.Fatal(err)
	}
	var dst CDF
	if err := dst.CopyFrom(&src); err != nil {
		t.Fatal(err)
	}
	if err := dst.Update(1); err != nil {
		t.Fatal(err)
	}
	assertCDFValues(t, src.Values(), []uint16{16384, 0, 0})
	assertCDFValues(t, dst.Values(), []uint16{17408, 0, 1})

	if err := dst.ResetCount(); err != nil {
		t.Fatal(err)
	}
	assertCDFValues(t, dst.Values(), []uint16{17408, 0, 0})

	dst.Reset()
	if err := dst.Validate(); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("reset Validate err=%v want %v", err, ErrInvalidCDF)
	}
}

func TestCDFStateRejectsInvalidInputs(t *testing.T) {
	var nilCDF *CDF
	if err := nilCDF.Init([]uint16{16384}); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("nil Init err=%v want %v", err, ErrInvalidCDF)
	}
	if err := nilCDF.InitUniform(2); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("nil InitUniform err=%v want %v", err, ErrInvalidCDF)
	}
	if err := nilCDF.InitDefaultDelta(); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("nil InitDefaultDelta err=%v want %v", err, ErrInvalidCDF)
	}
	if err := nilCDF.Validate(); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("nil Validate err=%v want %v", err, ErrInvalidCDF)
	}
	if err := nilCDF.Update(0); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("nil Update err=%v want %v", err, ErrInvalidCDF)
	}
	var dst CDF
	if err := dst.CopyFrom(nil); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("nil CopyFrom err=%v want %v", err, ErrInvalidCDF)
	}
	if err := nilCDF.CopyFrom(&dst); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("nil receiver CopyFrom err=%v want %v", err, ErrInvalidCDF)
	}
	if err := dst.InitUniform(1); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("small InitUniform err=%v want %v", err, ErrInvalidCDF)
	}
	if err := dst.InitUniform(MaxSymbols + 1); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("large InitUniform err=%v want %v", err, ErrInvalidCDF)
	}
	var invalid CDF
	if err := dst.CopyFrom(&invalid); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("invalid CopyFrom err=%v want %v", err, ErrInvalidCDF)
	}
}

func TestUpdateCDF(t *testing.T) {
	tests := []struct {
		name    string
		cdf     []uint16
		symbols int
		symbol  int
		want    []uint16
	}{
		{
			name:    "binary zero",
			cdf:     []uint16{16384, 0, 0},
			symbols: 2,
			symbol:  0,
			want:    []uint16{15360, 0, 1},
		},
		{
			name:    "binary one",
			cdf:     []uint16{16384, 0, 0},
			symbols: 2,
			symbol:  1,
			want:    []uint16{17408, 0, 1},
		},
		{
			name:    "quaternary middle",
			cdf:     []uint16{24576, 16384, 8192, 0, 0},
			symbols: 4,
			symbol:  2,
			want:    []uint16{24832, 16896, 7936, 0, 1},
		},
		{
			name:    "count saturated",
			cdf:     []uint16{16384, 0, MaxCDFCount},
			symbols: 2,
			symbol:  1,
			want:    []uint16{16640, 0, MaxCDFCount},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := UpdateCDF(tt.cdf, tt.symbols, tt.symbol); err != nil {
				t.Fatal(err)
			}
			if len(tt.cdf) != len(tt.want) {
				t.Fatalf("len=%d want %d", len(tt.cdf), len(tt.want))
			}
			for i := 0; i < len(tt.want); i++ {
				if tt.cdf[i] != tt.want[i] {
					t.Fatalf("cdf=%v want %v", tt.cdf, tt.want)
				}
			}
		})
	}
}

func TestUpdateCDFValuesMatchesWindow(t *testing.T) {
	for symbols := 2; symbols <= MaxSymbols; symbols++ {
		for _, count := range []uint16{0, 1, 15, 16, 31, MaxCDFCount} {
			for symbol := 0; symbol < symbols; symbol++ {
				var fixed [MaxSymbols + 1]uint16
				if err := InitUniformCDF(fixed[:], symbols); err != nil {
					t.Fatal(err)
				}
				fixed[symbols] = count
				window := append([]uint16(nil), fixed[:symbols+1]...)

				updateCDFValues(&fixed, symbols, symbol)
				updateCDFWindow(window, symbol)

				for i, want := range window {
					if fixed[i] != want {
						t.Fatalf("symbols=%d count=%d symbol=%d cdf=%v want %v", symbols, count, symbol, fixed[:symbols+1], window)
					}
				}
			}
		}
	}
}

func TestRejectsInvalidCDF(t *testing.T) {
	tests := []struct {
		name    string
		cdf     []uint16
		symbols int
	}{
		{name: "short", cdf: []uint16{0, 0}, symbols: 2},
		{name: "too few symbols", cdf: []uint16{0, 0}, symbols: 1},
		{name: "too many symbols", cdf: make([]uint16, MaxSymbols+2), symbols: MaxSymbols + 1},
		{name: "count too large", cdf: []uint16{16384, 0, MaxCDFCount + 1}, symbols: 2},
		{name: "missing terminal zero", cdf: []uint16{16384, 1, 0}, symbols: 2},
		{name: "increasing inverse cdf", cdf: []uint16{100, 200, 0, 0}, symbols: 3},
		{name: "top inverse cdf", cdf: []uint16{CDFProbTop, 0, 0}, symbols: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateCDF(tt.cdf, tt.symbols); !errors.Is(err, ErrInvalidCDF) {
				t.Fatalf("ValidateCDF err=%v want %v", err, ErrInvalidCDF)
			}
		})
	}
}

func TestRejectsInvalidInitCDF(t *testing.T) {
	var cdf [CDFSize4]uint16
	tests := [][]uint16{
		{0, 1, 2},
		{1, 1, 2},
		{1, 2, CDFProbTop + 1},
	}
	for _, cumulative := range tests {
		if err := InitCDF(cdf[:], cumulative); !errors.Is(err, ErrInvalidCDF) {
			t.Fatalf("InitCDF(%v) err=%v want %v", cumulative, err, ErrInvalidCDF)
		}
	}
}

func TestRejectsInvalidSymbol(t *testing.T) {
	cdf := []uint16{16384, 0, 0}
	if err := UpdateCDF(cdf, 2, -1); !errors.Is(err, ErrInvalidSymbol) {
		t.Fatalf("negative symbol err=%v want %v", err, ErrInvalidSymbol)
	}
	if err := UpdateCDF(cdf, 2, 2); !errors.Is(err, ErrInvalidSymbol) {
		t.Fatalf("large symbol err=%v want %v", err, ErrInvalidSymbol)
	}
}

func TestUpdateCDFAllocs(t *testing.T) {
	cdf := []uint16{24576, 16384, 8192, 0, 0}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := UpdateCDF(cdf, 4, 2); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("UpdateCDF allocated: %f", allocs)
	}
}

func TestCDFStateAllocs(t *testing.T) {
	var src CDF
	var dst CDF
	allocs := testing.AllocsPerRun(1000, func() {
		if err := src.InitUniform(4); err != nil {
			t.Fatal(err)
		}
		if err := src.InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		if err := dst.CopyFrom(&src); err != nil {
			t.Fatal(err)
		}
		if err := dst.Update(2); err != nil {
			t.Fatal(err)
		}
		if err := dst.Validate(); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("CDF state allocated: %f", allocs)
	}
}

func FuzzUpdateCDF(f *testing.F) {
	f.Add(uint8(2), []byte{0, 1, 1, 0})
	f.Add(uint8(4), []byte{2, 3, 0, 1})
	f.Add(uint8(MaxSymbols), []byte{15, 0, 14, 7})

	f.Fuzz(func(t *testing.T, symbolCount uint8, symbols []byte) {
		n := max(int(symbolCount%MaxSymbols)+1, 2)
		var cdf [MaxSymbols + 1]uint16
		if err := InitUniformCDF(cdf[:], n); err != nil {
			t.Fatal(err)
		}
		for i := range symbols {
			symbol := int(symbols[i]) % n
			if err := UpdateCDF(cdf[:], n, symbol); err != nil {
				t.Fatalf("UpdateCDF err=%v n=%d symbol=%d cdf=%v", err, n, symbol, cdf[:n+1])
			}
			if err := ValidateCDF(cdf[:], n); err != nil {
				t.Fatalf("ValidateCDF err=%v n=%d symbol=%d cdf=%v", err, n, symbol, cdf[:n+1])
			}
		}
	})
}

func FuzzCDFState(f *testing.F) {
	f.Add(uint8(2), []byte{0, 1, 1, 0})
	f.Add(uint8(4), []byte{2, 3, 0, 1})
	f.Add(uint8(MaxSymbols), []byte{15, 0, 14, 7})

	f.Fuzz(func(t *testing.T, symbolCount uint8, symbols []byte) {
		n := max(int(symbolCount%MaxSymbols)+1, 2)
		var cdf CDF
		if err := cdf.InitUniform(n); err != nil {
			t.Fatal(err)
		}
		for i := range symbols {
			symbol := int(symbols[i]) % n
			var copy CDF
			if err := copy.CopyFrom(&cdf); err != nil {
				t.Fatalf("CopyFrom err=%v n=%d cdf=%v", err, n, cdf.Values())
			}
			if err := copy.Update(symbol); err != nil {
				t.Fatalf("Update err=%v n=%d symbol=%d cdf=%v", err, n, symbol, copy.Values())
			}
			if err := copy.Validate(); err != nil {
				t.Fatalf("Validate err=%v n=%d symbol=%d cdf=%v", err, n, symbol, copy.Values())
			}
			cdf = copy
		}
	})
}

func BenchmarkUpdateCDF(b *testing.B) {
	cdf := []uint16{24576, 16384, 8192, 0, 0}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = UpdateCDF(cdf, 4, i&3)
	}
}

func BenchmarkInitUniformCDF(b *testing.B) {
	var cdf [MaxSymbols + 1]uint16

	b.ReportAllocs()
	for b.Loop() {
		_ = InitUniformCDF(cdf[:], MaxSymbols)
	}
}

func BenchmarkCDFStateInitUniform(b *testing.B) {
	var cdf CDF

	b.ReportAllocs()
	for b.Loop() {
		_ = cdf.InitUniform(MaxSymbols)
	}
}

func BenchmarkCDFStateUpdate(b *testing.B) {
	var cdf CDF
	if err := cdf.InitUniform(4); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = cdf.Update(i & 3)
	}
}

func BenchmarkCDFStateCopyFrom(b *testing.B) {
	var src CDF
	if err := src.InitUniform(MaxSymbols); err != nil {
		b.Fatal(err)
	}
	var dst CDF

	b.ReportAllocs()
	for b.Loop() {
		_ = dst.CopyFrom(&src)
	}
}

func assertCDFValues(t *testing.T, got []uint16, want []uint16) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("cdf=%v want %v", got, want)
		}
	}
}

const CDFSize4 = 5

func TestCoeffScanContextLayout(t *testing.T) {
	var p CoeffScanContext
	if unsafe.Sizeof(p) != 8 || unsafe.Offsetof(p.Pos) != 0 || unsafe.Offsetof(p.Padded) != 2 || unsafe.Offsetof(p.Lower2DOffset) != 4 || unsafe.Offsetof(p.BR2DOffset) != 5 || unsafe.Offsetof(p.LowerEOBCtx) != 6 || unsafe.Offsetof(p.BREOBCtx) != 7 {
		t.Fatal("coefficient scan descriptor layout changed")
	}
}

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

func TestCursorReadCDFSymbolsMatchesReader(t *testing.T) {
	rnd := randv2.New(randv2.NewPCG(5, 6))
	src := make([]byte, 4096)
	for i := range src {
		src[i] = byte(rnd.Uint32())
	}
	for n := 2; n <= 16; n++ {
		r1 := NewReaderWithCDFUpdate(src, true)
		r2 := NewReaderWithCDFUpdate(src, true)
		var a, b [5]CDF
		for k := range a {
			a[k].InitUniform(n)
			b[k].InitUniform(n)
		}
		for it := 0; it < 3000; it++ {
			k := rnd.IntN(5)
			x, _ := r1.ReadCDF(&a[k])
			c := r2.Cursor()
			y := c.ReadCDFSymbolsUnchecked(&b[k], n)
			c.CommitStateTo(&r2)
			if x != y || a[k] != b[k] {
				t.Fatalf("n=%d it=%d: reader %d cursor %d\n%v\n%v", n, it, x, y, a[k], b[k])
			}
		}
	}
}

func TestReaderHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "Reader", size: unsafe.Sizeof(Reader{}), max: 48},
		{name: "Cursor", size: unsafe.Sizeof(Cursor{}), max: 48},
	}
	for _, tt := range tests {
		if tt.size > tt.max {
			t.Fatalf("%s size=%d max=%d", tt.name, tt.size, tt.max)
		}
	}
}

func TestReaderReadBitKnownValues(t *testing.T) {
	r := NewReaderWithCDFUpdate([]byte{0x00}, true)
	bit, err := r.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 0 {
		t.Fatalf("zero stream bit=%d want 0", bit)
	}

	r = NewReaderWithCDFUpdate([]byte{0xff}, true)
	bit, err = r.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("one stream bit=%d want 1", bit)
	}
}

func TestReaderReadBool(t *testing.T) {
	r := NewReaderWithCDFUpdate([]byte{0xff}, true)
	bit, err := r.ReadBool(128)
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("bit=%d want 1", bit)
	}

	r = NewReaderWithCDFUpdate([]byte{0x00}, true)
	bit, err = r.ReadBool(128)
	if err != nil {
		t.Fatal(err)
	}
	if bit != 0 {
		t.Fatalf("bit=%d want 0", bit)
	}
}

func TestReaderReadBits(t *testing.T) {
	r := NewReaderWithCDFUpdate([]byte{0xff, 0x00}, true)
	got, err := r.ReadBits(4)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0xf {
		t.Fatalf("bits=%#x want 0xf", got)
	}
	if r.BitsRead() <= 0 {
		t.Fatalf("BitsRead=%d", r.BitsRead())
	}
}

func TestReaderReadBitsMatchesRepeatedReadBit(t *testing.T) {
	src := []byte{0xa5, 0x5a, 0xc3, 0x3c, 0xf0, 0x0f, 0x99, 0x66}
	for n := uint8(0); n <= 32; n++ {
		bulk := NewReaderWithCDFUpdate(src, true)
		step := NewReaderWithCDFUpdate(src, true)
		got, err := bulk.ReadBits(n)
		if err != nil {
			t.Fatalf("ReadBits(%d): %v", n, err)
		}
		var want uint32
		for i := uint8(0); i < n; i++ {
			bit, err := step.ReadBit()
			if err != nil {
				t.Fatalf("ReadBit[%d/%d]: %v", i, n, err)
			}
			want = (want << 1) | uint32(bit)
		}
		if got != want {
			t.Fatalf("ReadBits(%d)=%#x want %#x", n, got, want)
		}
		if bulk.BitsRead() != step.BitsRead() {
			t.Fatalf("ReadBits(%d) BitsRead=%d want %d", n, bulk.BitsRead(), step.BitsRead())
		}
		nextBulk, bulkErr := bulk.ReadBit()
		nextStep, stepErr := step.ReadBit()
		if bulkErr != stepErr || nextBulk != nextStep {
			t.Fatalf("ReadBits(%d) next bit=%d/%v want %d/%v", n, nextBulk, bulkErr, nextStep, stepErr)
		}
	}
}

func TestReaderReadUniform(t *testing.T) {
	r := NewReaderWithCDFUpdate([]byte{0x00}, true)
	got, err := r.ReadUniform(10)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("zero uniform=%d want 0", got)
	}

	r = NewReaderWithCDFUpdate([]byte{0xff}, true)
	got, err = r.ReadUniform(10)
	if err != nil {
		t.Fatal(err)
	}
	if got != 9 {
		t.Fatalf("one uniform=%d want 9", got)
	}

	r = NewReaderWithCDFUpdate(nil, true)
	got, err = r.ReadUniform(1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("single uniform=%d want 0", got)
	}
}

func TestReaderReadSubexp(t *testing.T) {
	r := NewReaderWithCDFUpdate([]byte{0x00}, true)
	got, err := r.ReadSubexp(16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("zero subexp=%d want 0", got)
	}

	r = NewReaderWithCDFUpdate([]byte{0xff, 0xff}, true)
	got, err = r.ReadSubexp(16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 15 {
		t.Fatalf("one subexp=%d want 15", got)
	}
}

func TestReaderReadRefSubexp(t *testing.T) {
	r := NewReaderWithCDFUpdate([]byte{0x00}, true)
	got, err := r.ReadRefSubexp(10, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got != 4 {
		t.Fatalf("ref subexp=%d want 4", got)
	}
}

func TestReaderReadSignedRefSubexp(t *testing.T) {
	r := NewReaderWithCDFUpdate([]byte{0x00}, true)
	got, err := r.ReadSignedRefSubexp(5, 1, -1)
	if err != nil {
		t.Fatal(err)
	}
	if got != -1 {
		t.Fatalf("signed ref subexp=%d want -1", got)
	}
}

func TestReaderReadSymbol(t *testing.T) {
	cdf := []uint16{16384, 0, 0}
	r := NewReaderWithCDFUpdate([]byte{0x00}, true)
	symbol, err := r.ReadSymbol(cdf, 2)
	if err != nil {
		t.Fatal(err)
	}
	if symbol != 0 {
		t.Fatalf("symbol=%d want 0", symbol)
	}
	want := []uint16{15360, 0, 1}
	for i := range want {
		if cdf[i] != want[i] {
			t.Fatalf("cdf=%v want %v", cdf, want)
		}
	}
}

func TestReaderReadCDF(t *testing.T) {
	var cdf CDF
	if err := cdf.InitUniform(2); err != nil {
		t.Fatal(err)
	}
	r := NewReaderWithCDFUpdate([]byte{0x00}, true)
	symbol, err := r.ReadCDF(&cdf)
	if err != nil {
		t.Fatal(err)
	}
	if symbol != 0 {
		t.Fatalf("symbol=%d want 0", symbol)
	}
	want := []uint16{15360, 0, 1}
	for i := range want {
		if cdf.Values()[i] != want[i] {
			t.Fatalf("cdf=%v want %v", cdf.Values(), want)
		}
	}
}

func TestReaderReadBinaryCDFTrustedMatchesReadCDF(t *testing.T) {
	src := []byte{0xa5, 0x5a, 0xc3}
	var genericCDF CDF
	if err := genericCDF.InitUniform(2); err != nil {
		t.Fatal(err)
	}
	trustedCDF := genericCDF
	generic := NewReaderWithCDFUpdate(src, true)
	trusted := NewReaderWithCDFUpdate(src, true)

	for i := 0; i < 8; i++ {
		want, err := generic.ReadSymbol(genericCDF.Values(), genericCDF.Symbols())
		if err != nil {
			t.Fatal(err)
		}
		got, err := trusted.ReadBinaryCDFTrusted(&trustedCDF)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("symbol[%d]=%d want %d", i, got, want)
		}
		if trustedCDF != genericCDF {
			t.Fatalf("cdf[%d]=%v want %v", i, trustedCDF.Values(), genericCDF.Values())
		}
	}
}

func TestReaderReadSmallCDFTrustedMatchesReadCDF(t *testing.T) {
	src := []byte{0xa5, 0x5a, 0xc3, 0x3c, 0xf0}
	for _, symbols := range []int{3, 4} {
		t.Run("symbols", func(t *testing.T) {
			var genericCDF CDF
			if err := genericCDF.InitUniform(symbols); err != nil {
				t.Fatal(err)
			}
			trustedCDF := genericCDF
			generic := NewReaderWithCDFUpdate(src, true)
			trusted := NewReaderWithCDFUpdate(src, true)

			for i := 0; i < 16; i++ {
				want, err := generic.ReadSymbol(genericCDF.Values(), genericCDF.Symbols())
				if err != nil {
					t.Fatal(err)
				}
				var got int
				switch symbols {
				case 3:
					got, err = trusted.ReadCDF3Trusted(&trustedCDF)
				case 4:
					got, err = trusted.ReadCDF4Trusted(&trustedCDF)
				}
				if err != nil {
					t.Fatal(err)
				}
				if got != want {
					t.Fatalf("symbol[%d]=%d want %d", i, got, want)
				}
				if trustedCDF != genericCDF {
					t.Fatalf("cdf[%d]=%v want %v", i, trustedCDF.Values(), genericCDF.Values())
				}
			}
		})
	}
}

func TestCursorTrustedReadsMatchReader(t *testing.T) {
	src := []byte{0xa5, 0x5a, 0xc3, 0x3c, 0xf0, 0x0f, 0x99}
	for _, allowUpdate := range []bool{true, false} {
		t.Run("allow_update", func(t *testing.T) {
			ref := NewReaderWithCDFUpdate(src, allowUpdate)
			gotReader := NewReaderWithCDFUpdate(src, allowUpdate)
			got := gotReader.Cursor()

			var ref2, got2 CDF
			if err := ref2.InitUniform(2); err != nil {
				t.Fatal(err)
			}
			got2 = ref2
			var ref3, got3 CDF
			if err := ref3.InitUniform(3); err != nil {
				t.Fatal(err)
			}
			got3 = ref3
			var ref4, got4 CDF
			if err := ref4.InitUniform(4); err != nil {
				t.Fatal(err)
			}
			got4 = ref4
			var refHi, gotHi CDF
			if err := refHi.InitUniform(4); err != nil {
				t.Fatal(err)
			}
			gotHi = refHi
			var ref7, got7 CDF
			if err := ref7.InitUniform(7); err != nil {
				t.Fatal(err)
			}
			got7 = ref7

			for i := 0; i < 16; i++ {
				want7, err := ref.ReadCDF(&ref7)
				if err != nil {
					t.Fatal(err)
				}
				if got7Symbol := got.ReadCDFUnchecked(&got7); got7Symbol != want7 {
					t.Fatalf("cdf7[%d]=%d want %d", i, got7Symbol, want7)
				}
				want4 := ref.ReadCDF4Unchecked(&ref4)
				if got4Symbol := got.ReadCDF4Unchecked(&got4); got4Symbol != want4 {
					t.Fatalf("cdf4[%d]=%d want %d", i, got4Symbol, want4)
				}
				wantHi := uint8(0)
				for j := 0; j < 4; j++ {
					sym := ref.ReadCDF4Unchecked(&refHi)
					wantHi += uint8(sym)
					if sym < 3 {
						break
					}
				}
				if gotHiSymbol := got.ReadCDF4HighTokenUnchecked(&gotHi); gotHiSymbol != wantHi {
					t.Fatalf("cdf4hi[%d]=%d want %d", i, gotHiSymbol, wantHi)
				}
				want3 := ref.ReadCDF3Unchecked(&ref3)
				if got3Symbol := got.ReadCDF3Unchecked(&got3); got3Symbol != want3 {
					t.Fatalf("cdf3[%d]=%d want %d", i, got3Symbol, want3)
				}
				want2 := ref.ReadBinaryCDFUnchecked(&ref2)
				if got2Symbol := got.ReadBinaryCDFUnchecked(&got2); got2Symbol != want2 {
					t.Fatalf("cdf2[%d]=%d want %d", i, got2Symbol, want2)
				}
				wantBit := ref.ReadBitTrusted()
				if gotBit := got.ReadBitTrustedInline(); gotBit != wantBit {
					t.Fatalf("inline bit[%d]=%d want %d", i, gotBit, wantBit)
				}
				wantBits := ref.ReadBitsTrusted(3)
				if gotBits := got.ReadBitsTrusted(3); gotBits != wantBits {
					t.Fatalf("bits[%d]=%d want %d", i, gotBits, wantBits)
				}
			}
			got.CommitTo(&gotReader)

			if gotReader.pos != ref.pos || gotReader.dif != ref.dif || gotReader.rng != ref.rng ||
				gotReader.cnt != ref.cnt || gotReader.tellOffs != ref.tellOffs ||
				gotReader.allowCDFUpdate != ref.allowCDFUpdate {
				t.Fatalf("reader state mismatch: got pos=%d dif=%08x rng=%08x cnt=%d tell=%d allow=%v want pos=%d dif=%08x rng=%08x cnt=%d tell=%d allow=%v",
					gotReader.pos, gotReader.dif, gotReader.rng, gotReader.cnt, gotReader.tellOffs, gotReader.allowCDFUpdate,
					ref.pos, ref.dif, ref.rng, ref.cnt, ref.tellOffs, ref.allowCDFUpdate)
			}
			if got2 != ref2 || got3 != ref3 || got4 != ref4 || gotHi != refHi || got7 != ref7 {
				t.Fatalf("cdf mismatch: got %v/%v/%v/%v/%v want %v/%v/%v/%v/%v",
					got2.Values(), got3.Values(), got4.Values(), gotHi.Values(), got7.Values(),
					ref2.Values(), ref3.Values(), ref4.Values(), refHi.Values(), ref7.Values())
			}
		})
	}
}

func TestReaderReadSignedDeltaZero(t *testing.T) {
	var cdf CDF
	if err := cdf.InitDefaultDelta(); err != nil {
		t.Fatal(err)
	}
	r := NewReaderWithCDFUpdate([]byte{0x00}, true)
	delta, err := r.ReadSignedDelta(&cdf, DeltaSmall)
	if err != nil {
		t.Fatal(err)
	}
	if delta != 0 {
		t.Fatalf("delta=%d want 0", delta)
	}
	want := []uint16{4464, 628, 89, 0, 1}
	for i := range want {
		if cdf.Values()[i] != want[i] {
			t.Fatalf("cdf=%v want %v", cdf.Values(), want)
		}
	}
}

func TestReaderReadSignedDeltaExtendedTail(t *testing.T) {
	var cdf CDF
	if err := cdf.InitDefaultDelta(); err != nil {
		t.Fatal(err)
	}
	r := NewReaderWithCDFUpdate([]byte{0xff, 0xff, 0xff}, true)
	delta, err := r.ReadSignedDelta(&cdf, DeltaSmall)
	if err != nil {
		t.Fatal(err)
	}
	if delta > -DeltaSmall && delta < DeltaSmall {
		t.Fatalf("delta=%d did not use extended tail", delta)
	}
	if err := cdf.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestReaderReadSymbolCanDisableCDFUpdate(t *testing.T) {
	cdf := []uint16{16384, 0, 0}
	r := NewReaderWithCDFUpdate([]byte{0xff}, false)
	if r.AllowCDFUpdate() {
		t.Fatal("CDF update enabled")
	}
	symbol, err := r.ReadSymbol(cdf, 2)
	if err != nil {
		t.Fatal(err)
	}
	if symbol != 1 {
		t.Fatalf("symbol=%d want 1", symbol)
	}
	if cdf[0] != 16384 || cdf[2] != 0 {
		t.Fatalf("cdf updated: %v", cdf)
	}

	r.SetCDFUpdate(true)
	if !r.AllowCDFUpdate() {
		t.Fatal("CDF update disabled")
	}
}

func TestReaderReadSymbolMatchesBoolQ15(t *testing.T) {
	src := []byte{0xa5, 0x5a}
	cdf := []uint16{12345, 0, 0}
	symbolReader := NewReaderWithCDFUpdate(src, false)
	boolReader := NewReaderWithCDFUpdate(src, false)

	symbol, err := symbolReader.ReadSymbol(cdf, 2)
	if err != nil {
		t.Fatal(err)
	}
	bit, err := boolReader.ReadBoolQ15(cdf[0])
	if err != nil {
		t.Fatal(err)
	}
	if symbol != int(bit) {
		t.Fatalf("symbol=%d bit=%d", symbol, bit)
	}
}

func TestReaderRejectsInvalidInputs(t *testing.T) {
	r := NewReaderWithCDFUpdate(nil, true)
	if _, err := r.ReadBoolQ15(0); !errors.Is(err, ErrInvalidProbability) {
		t.Fatalf("zero probability err=%v want %v", err, ErrInvalidProbability)
	}
	if _, err := r.ReadBoolQ15(CDFProbTop); !errors.Is(err, ErrInvalidProbability) {
		t.Fatalf("top probability err=%v want %v", err, ErrInvalidProbability)
	}
	if _, err := r.ReadBits(33); !errors.Is(err, ErrInvalidBitCount) {
		t.Fatalf("ReadBits err=%v want %v", err, ErrInvalidBitCount)
	}
	if _, err := r.ReadUniform(0); !errors.Is(err, ErrInvalidRange) {
		t.Fatalf("ReadUniform err=%v want %v", err, ErrInvalidRange)
	}
	if _, err := r.ReadSubexp(0, 1); !errors.Is(err, ErrInvalidRange) {
		t.Fatalf("ReadSubexp n err=%v want %v", err, ErrInvalidRange)
	}
	if _, err := r.ReadSubexp(10, 32); !errors.Is(err, ErrInvalidRange) {
		t.Fatalf("ReadSubexp k err=%v want %v", err, ErrInvalidRange)
	}
	if _, err := r.ReadRefSubexp(3, 1, 3); !errors.Is(err, ErrInvalidRange) {
		t.Fatalf("ReadRefSubexp err=%v want %v", err, ErrInvalidRange)
	}
	if _, err := r.ReadSignedRefSubexp(5, 1, 5); !errors.Is(err, ErrInvalidRange) {
		t.Fatalf("ReadSignedRefSubexp err=%v want %v", err, ErrInvalidRange)
	}
	if _, err := r.ReadSymbol([]uint16{CDFProbTop, 0, 0}, 2); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("ReadSymbol err=%v want %v", err, ErrInvalidCDF)
	}
	if _, err := r.ReadCDF(nil); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("ReadCDF nil err=%v want %v", err, ErrInvalidCDF)
	}
	var cdf CDF
	if _, err := r.ReadCDF(&cdf); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("ReadCDF zero err=%v want %v", err, ErrInvalidCDF)
	}
	if _, err := r.ReadSignedDelta(nil, DeltaSmall); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("ReadSignedDelta nil err=%v want %v", err, ErrInvalidCDF)
	}
	if _, err := r.ReadSignedDelta(&cdf, DeltaSmall); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("ReadSignedDelta zero err=%v want %v", err, ErrInvalidCDF)
	}
	var binary CDF
	if err := binary.InitUniform(2); err != nil {
		t.Fatal(err)
	}
	if _, err := r.ReadSignedDelta(&binary, DeltaSmall); !errors.Is(err, ErrInvalidCDF) {
		t.Fatalf("ReadSignedDelta mismatched CDF err=%v want %v", err, ErrInvalidCDF)
	}
	if _, err := r.ReadSignedDelta(&binary, 0); !errors.Is(err, ErrInvalidRange) {
		t.Fatalf("ReadSignedDelta small err=%v want %v", err, ErrInvalidRange)
	}
	if _, err := r.ReadSignedDelta(&binary, MaxSymbols); !errors.Is(err, ErrInvalidRange) {
		t.Fatalf("ReadSignedDelta large small err=%v want %v", err, ErrInvalidRange)
	}
}

func TestReaderAllocs(t *testing.T) {
	src := []byte{0xff, 0x00, 0xa5}
	cdf := []uint16{24576, 16384, 8192, 0, 0}
	var state CDF
	var delta CDF
	allocs := testing.AllocsPerRun(1000, func() {
		r := NewReaderWithCDFUpdate(src, true)
		if err := state.InitUniform(4); err != nil {
			t.Fatal(err)
		}
		if err := delta.InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		_, err := r.ReadBits(3)
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.ReadSymbol(cdf, 4)
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.ReadCDF(&state)
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.ReadSignedDelta(&delta, DeltaSmall)
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.ReadUniform(257)
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.ReadSubexp(257, 3)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Reader allocated: %f", allocs)
	}
}

func FuzzReaderUniformSubexp(f *testing.F) {
	f.Add([]byte{0x00}, uint16(1), uint8(0), uint16(0))
	f.Add([]byte{0xff}, uint16(16), uint8(1), uint16(4))
	f.Add([]byte{0xa5, 0x5a}, uint16(255), uint8(3), uint16(128))

	f.Fuzz(func(t *testing.T, src []byte, rawN uint16, rawK uint8, rawRef uint16) {
		n := uint32(rawN%4096) + 1
		k := rawK & 7

		r := NewReaderWithCDFUpdate(src, true)
		uniform, err := r.ReadUniform(n)
		if err != nil {
			t.Fatalf("ReadUniform err=%v n=%d", err, n)
		}
		if uniform >= n {
			t.Fatalf("uniform=%d n=%d", uniform, n)
		}

		r = NewReaderWithCDFUpdate(src, true)
		subexp, err := r.ReadSubexp(n, k)
		if err != nil {
			t.Fatalf("ReadSubexp err=%v n=%d k=%d", err, n, k)
		}
		if subexp >= n {
			t.Fatalf("subexp=%d n=%d k=%d", subexp, n, k)
		}

		ref := uint32(rawRef) % n
		r = NewReaderWithCDFUpdate(src, true)
		recentered, err := r.ReadRefSubexp(n, k, ref)
		if err != nil {
			t.Fatalf("ReadRefSubexp err=%v n=%d k=%d ref=%d", err, n, k, ref)
		}
		if recentered >= n {
			t.Fatalf("recentered=%d n=%d k=%d ref=%d", recentered, n, k, ref)
		}
	})
}

func FuzzReaderBinarySymbol(f *testing.F) {
	f.Add([]byte{0x00}, uint16(16384))
	f.Add([]byte{0xff}, uint16(16384))
	f.Add([]byte{0xa5, 0x5a}, uint16(12345))

	f.Fuzz(func(t *testing.T, src []byte, rawProb uint16) {
		prob := (rawProb % (CDFProbTop - 1)) + 1
		cdf := []uint16{prob, 0, 0}
		symbolReader := NewReaderWithCDFUpdate(src, false)
		boolReader := NewReaderWithCDFUpdate(src, false)

		symbol, err := symbolReader.ReadSymbol(cdf, 2)
		if err != nil {
			t.Fatalf("ReadSymbol err=%v prob=%d", err, prob)
		}
		bit, err := boolReader.ReadBoolQ15(prob)
		if err != nil {
			t.Fatalf("ReadBoolQ15 err=%v prob=%d", err, prob)
		}
		if symbol != int(bit) {
			t.Fatalf("symbol=%d bit=%d prob=%d src=%x", symbol, bit, prob, src)
		}
		if cdf[0] != prob || cdf[1] != 0 || cdf[2] != 0 {
			t.Fatalf("cdf mutated: %v", cdf)
		}
	})
}

func FuzzReaderSignedDelta(f *testing.F) {
	f.Add([]byte{0x00})
	f.Add([]byte{0xff, 0xff, 0xff})
	f.Add([]byte{0xa5, 0x5a})

	f.Fuzz(func(t *testing.T, src []byte) {
		var cdf CDF
		if err := cdf.InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		r := NewReaderWithCDFUpdate(src, true)
		delta, err := r.ReadSignedDelta(&cdf, DeltaSmall)
		if err != nil {
			t.Fatalf("ReadSignedDelta err=%v src=%x", err, src)
		}
		if delta < -512 || delta > 512 {
			t.Fatalf("delta=%d out of range src=%x", delta, src)
		}
		if err := cdf.Validate(); err != nil {
			t.Fatalf("Validate err=%v cdf=%v", err, cdf.Values())
		}
	})
}

func BenchmarkReaderReadBit(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		_, _ = r.ReadBit()
	}
}

func BenchmarkCursorReadBitTrustedStream(b *testing.B) {
	src := benchStream()
	sum := 0

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		c := r.Cursor()
		for range benchSymbolsPerOp {
			sum += int(c.ReadBitTrusted())
		}
	}
	benchmarkReaderSink = sum
}

func BenchmarkCursorReadBitsTrustedStream(b *testing.B) {
	src := benchStream()
	var sum uint32

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		c := r.Cursor()
		for range benchSymbolsPerOp {
			sum += c.ReadBitsTrusted(3)
		}
	}
	benchmarkReaderSink = int(sum)
}

func BenchmarkReaderReadBits(b *testing.B) {
	src := benchStream()

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		for range benchSymbolsPerOp {
			_, _ = r.ReadBits(8)
		}
	}
}

func BenchmarkReaderReadUniform(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		_, _ = r.ReadUniform(257)
	}
}

func BenchmarkReaderReadSubexp(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		_, _ = r.ReadSubexp(257, 3)
	}
}

func BenchmarkReaderReadSymbol(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}
	cdf := []uint16{24576, 16384, 8192, 0, 0}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		_, _ = r.ReadSymbol(cdf, 4)
	}
}

func BenchmarkReaderReadCDF(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}
	var cdf CDF
	if err := cdf.InitUniform(4); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		_, _ = r.ReadCDF(&cdf)
	}
}

func BenchmarkReaderReadSignedDelta(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}
	var cdf CDF
	if err := cdf.InitDefaultDelta(); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		_, _ = r.ReadSignedDelta(&cdf, DeltaSmall)
	}
}

// benchStream is a pseudo-random payload large enough to keep one Reader busy
// for many symbols without exhausting the bitstream, so the benchmarks below
// measure steady-state per-symbol cost rather than NewReader/refill overhead.
func benchStream() []byte {
	const n = 1 << 16
	s := make([]byte, n)
	x := uint32(0x12345678)
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = byte(x >> 16)
	}
	return s
}

// benchSymbolsPerOp is the number of symbols decoded per benchmark iteration.
// A large fresh stream supplies them, so NewReader/refill overhead amortizes
// and each reported ns/op approximates one steady-state symbol decode.
const benchSymbolsPerOp = 4096

var benchmarkReaderSink int

// BenchmarkReaderSymbolStream decodes a long run of CDF-adapted symbols from a
// single Reader, isolating ReadSymbol's per-symbol cost (the decoder hot path).
// CDF state is reset each batch so adaptation stays deterministic.
func BenchmarkReaderSymbolStream(b *testing.B) {
	src := benchStream()

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		cdf := []uint16{24576, 16384, 8192, 0, 0}
		for range benchSymbolsPerOp {
			if _, err := r.ReadSymbol(cdf, 4); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkReaderSymbolStreamNoUpdate measures the decode path with CDF
// adaptation disabled (used for some frame headers / non-adaptive contexts).
func BenchmarkReaderSymbolStreamNoUpdate(b *testing.B) {
	src := benchStream()
	cdf := []uint16{24576, 16384, 8192, 0, 0}

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, false)
		for range benchSymbolsPerOp {
			if _, err := r.ReadSymbol(cdf, 4); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkReaderCDFStream decodes a long run of CDF-adapted symbols through the
// *CDF entry point (ReadCDF), the exact path the tile decoder uses for mode, MV,
// and coefficient syntax. It isolates ReadCDF's steady-state per-symbol cost,
// including the routing through the validation-free trusted core. Compare against
// BenchmarkReaderSymbolStream (slice path, full per-call ValidateCDF scan) to see
// the cost of the monotonicity validation that ReadCDF skips.
func BenchmarkReaderCDFStream(b *testing.B) {
	src := benchStream()

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		var cdf CDF
		if err := cdf.Init([]uint16{8192, 16384, 24576}); err != nil {
			b.Fatal(err)
		}
		for range benchSymbolsPerOp {
			if _, err := r.ReadCDF(&cdf); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkReaderCDFStream2(b *testing.B) {
	benchmarkReaderCDFStreamSymbols(b, 2)
}

func BenchmarkReaderCDFStream3(b *testing.B) {
	benchmarkReaderCDFStreamSymbols(b, 3)
}

func BenchmarkReaderCDFStream4(b *testing.B) {
	benchmarkReaderCDFStreamSymbols(b, 4)
}

func BenchmarkCursorCDF4Stream(b *testing.B) {
	benchmarkCursorCDF4Stream(b, true)
}

func BenchmarkCursorCDF4StreamNoUpdate(b *testing.B) {
	benchmarkCursorCDF4Stream(b, false)
}

func BenchmarkCursorCDFStream2(b *testing.B) {
	benchmarkCursorCDFStreamSymbols(b, 2)
}

func BenchmarkCursorCDFSymbolsStream2(b *testing.B) {
	benchmarkCursorCDFSymbolsStream(b, 2)
}

func BenchmarkCursorCDFStream5(b *testing.B) {
	benchmarkCursorCDFStreamSymbols(b, 5)
}

func BenchmarkCursorCDFSymbolsStream5(b *testing.B) {
	benchmarkCursorCDFSymbolsStream(b, 5)
}

func BenchmarkCursorCDFStream6(b *testing.B) {
	benchmarkCursorCDFStreamSymbols(b, 6)
}

func BenchmarkCursorCDFSymbolsStream6(b *testing.B) {
	benchmarkCursorCDFSymbolsStream(b, 6)
}

func BenchmarkCursorCDFStream7(b *testing.B) {
	benchmarkCursorCDFStreamSymbols(b, 7)
}

func BenchmarkCursorCDFSymbolsStream7(b *testing.B) {
	benchmarkCursorCDFSymbolsStream(b, 7)
}

func BenchmarkCursorCDFStream8(b *testing.B) {
	benchmarkCursorCDFStreamSymbols(b, 8)
}

func BenchmarkCursorCDFSymbolsStream8(b *testing.B) {
	benchmarkCursorCDFSymbolsStream(b, 8)
}

func BenchmarkCursorCDFStream9(b *testing.B) {
	benchmarkCursorCDFStreamSymbols(b, 9)
}

func BenchmarkCursorCDFSymbolsStream9(b *testing.B) {
	benchmarkCursorCDFSymbolsStream(b, 9)
}

func benchmarkCursorCDF4Stream(b *testing.B, update bool) {
	b.Helper()
	src := benchStream()
	sum := 0

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, update)
		c := r.Cursor()
		var cdf CDF
		if err := cdf.InitUniform(4); err != nil {
			b.Fatal(err)
		}
		for range benchSymbolsPerOp {
			sum += c.ReadCDF4Unchecked(&cdf)
		}
	}
	benchmarkReaderSink = sum
}

func benchmarkCursorCDFStreamSymbols(b *testing.B, symbols int) {
	b.Helper()
	src := benchStream()
	sum := 0

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		c := r.Cursor()
		var cdf CDF
		if err := cdf.InitUniform(symbols); err != nil {
			b.Fatal(err)
		}
		for range benchSymbolsPerOp {
			sum += c.ReadCDFUnchecked(&cdf)
		}
	}
	benchmarkReaderSink = sum
}

func benchmarkCursorCDFSymbolsStream(b *testing.B, symbols int) {
	b.Helper()
	src := benchStream()
	sum := 0

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		c := r.Cursor()
		var cdf CDF
		if err := cdf.InitUniform(symbols); err != nil {
			b.Fatal(err)
		}
		for range benchSymbolsPerOp {
			sum += c.ReadCDFSymbolsUnchecked(&cdf, symbols)
		}
	}
	benchmarkReaderSink = sum
}

func BenchmarkCursorCDF4HighTokenStream(b *testing.B) {
	benchmarkCursorCDF4HighTokenStream(b, true)
}

func BenchmarkCursorCDF4HighTokenStreamNoUpdate(b *testing.B) {
	benchmarkCursorCDF4HighTokenStream(b, false)
}

func benchmarkCursorCDF4HighTokenStream(b *testing.B, update bool) {
	b.Helper()
	src := benchStream()
	sum := 0

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, update)
		c := r.Cursor()
		var cdf CDF
		if err := cdf.InitUniform(4); err != nil {
			b.Fatal(err)
		}
		for range benchSymbolsPerOp {
			sum += int(c.ReadCDF4HighTokenUnchecked(&cdf))
		}
	}
	benchmarkReaderSink = sum
}

func benchmarkReaderCDFStreamSymbols(b *testing.B, symbols int) {
	b.Helper()
	src := benchStream()

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		var cdf CDF
		if err := cdf.InitUniform(symbols); err != nil {
			b.Fatal(err)
		}
		for range benchSymbolsPerOp {
			if _, err := r.ReadCDF(&cdf); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkReaderCDFStream16 decodes through ReadCDF with a 16-symbol CDF, the
// AV1 maximum. The per-call ValidateCDF monotonicity scan is O(symbols), so this
// is where skipping it (ReadCDF routes to the trusted core) saves the most. The
// 16-entry CDFs are exercised in real streams by the larger mode and partition
// alphabets.
func BenchmarkReaderCDFStream16(b *testing.B) {
	src := benchStream()

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		var cdf CDF
		if err := cdf.InitUniform(16); err != nil {
			b.Fatal(err)
		}
		for range benchSymbolsPerOp {
			if _, err := r.ReadCDF(&cdf); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkReaderSymbolStream16 is the slice-path (full ValidateCDF scan)
// counterpart to BenchmarkReaderCDFStream16, decoding a 16-symbol CDF via
// ReadSymbol. The delta between the two benchmarks at 16 symbols is the
// per-symbol cost of the monotonicity validation that the trusted path elides.
func BenchmarkReaderSymbolStream16(b *testing.B) {
	src := benchStream()

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		var cdf CDF
		if err := cdf.InitUniform(16); err != nil {
			b.Fatal(err)
		}
		values := cdf.Values()
		r := NewReaderWithCDFUpdate(src, true)
		for range benchSymbolsPerOp {
			if _, err := r.ReadSymbol(values, 16); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkReaderBoolStream isolates ReadBoolQ15's per-call cost.
func BenchmarkReaderBoolStream(b *testing.B) {
	src := benchStream()

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReaderWithCDFUpdate(src, true)
		for range benchSymbolsPerOp {
			if _, err := r.ReadBoolQ15(CDFProbTop / 2); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// dav1dMsac is dav1d's MsacContext bookkeeping (src/msac.c dav1d_msac_init,
// ctx_refill, ctx_norm, dav1d_msac_decode_bool_c) over a 64-bit window, the
// reference for TestReaderOverreadMatchesDav1dCnt.
type dav1dMsac struct {
	buf []byte
	pos int
	dif uint64
	rng uint32
	cnt int
}

func newDav1dMsac(buf []byte) *dav1dMsac {
	s := &dav1dMsac{buf: buf, rng: 0x8000, cnt: -15}
	s.refill()
	return s
}

func (s *dav1dMsac) refill() {
	c := 64 - s.cnt - 24
	dif := s.dif
	for {
		if s.pos >= len(s.buf) {
			dif |= ^(^uint64(0xff) << uint(c))
			break
		}
		dif |= uint64(s.buf[s.pos]^0xff) << uint(c)
		s.pos++
		c -= 8
		if c < 0 {
			break
		}
	}
	s.dif = dif
	s.cnt = 64 - c - 24
}

func (s *dav1dMsac) norm(dif uint64, rng uint32) {
	d := 15 ^ (31 ^ bits.LeadingZeros32(rng))
	cnt := s.cnt
	s.dif = dif << uint(d)
	s.rng = rng << uint(d)
	s.cnt = cnt - d
	if uint(cnt) < uint(d) {
		s.refill()
	}
}

func (s *dav1dMsac) decodeBool(f uint32) uint8 {
	r := s.rng
	dif := s.dif
	v := ((r>>8)*(f>>ecProbShift))>>(7-ecProbShift) + ecMinProb
	vw := uint64(v) << 48
	ret := dif >= vw
	if ret {
		dif -= vw
		v = r - v
	}
	s.norm(dif, v)
	if ret {
		return 0
	}
	return 1
}

// overread is dav1d_decode_tile_sbrow's "error out on symbol decoder
// overread" condition.
func (s *dav1dMsac) overread() bool { return s.cnt <= -15 }

// TestReaderOverreadMatchesDav1dCnt drives Reader and dav1d's msac over the
// same symbols and checks that Overread fires exactly when dav1d's
// msac.cnt <= -15 does, including tens of thousands of padding bits past the
// end of the data (tellOffs once wrapped there and hid the overread).
func TestReaderOverreadMatchesDav1dCnt(t *testing.T) {
	rng := uint64(0x9e3779b97f4a7c15)
	next := func() uint64 {
		rng ^= rng << 13
		rng ^= rng >> 7
		rng ^= rng << 17
		return rng
	}
	for _, n := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 500, 4096} {
		for trial := range 8 {
			buf := make([]byte, n)
			for i := range buf {
				buf[i] = byte(next())
			}
			r := NewReaderWithCDFUpdate(buf, true)
			m := newDav1dMsac(buf)
			same := true
			for step := 0; step < 8*n+120000; step++ {
				var prob uint16
				switch next() % 4 {
				case 0:
					prob = CDFProbTop / 2
				case 1:
					prob = uint16(1 + next()%64)
				case 2:
					prob = uint16(CDFProbTop - 1 - next()%64)
				default:
					prob = uint16(1 + next()%(CDFProbTop-1))
				}
				got, err := r.ReadBoolQ15(prob)
				if err != nil {
					t.Fatal(err)
				}
				want := m.decodeBool(uint32(prob))
				// The two windows pad past the end differently once the
				// padding is far past dav1d's threshold; the symbols agree up
				// to there.
				if same && !m.overread() && got != want {
					t.Fatalf("len=%d trial=%d step=%d: bit %d, dav1d %d", n, trial, step, got, want)
				}
				if got != want {
					same = false
				}
				if r.Overread() != m.overread() {
					t.Fatalf("len=%d trial=%d step=%d: Overread=%v, dav1d cnt=%d (BitsRead=%d)", n, trial, step, r.Overread(), m.cnt, r.BitsRead())
				}
			}
		}
	}
}

// logTestOp is one randomized writer call; kind selects the Writer method.
type logTestOp struct {
	kind int
	s    int
	f    uint32
	cdf  int
}

func logTestCDFs(t *testing.T) []CDF {
	t.Helper()
	sizes := []int{2, 3, 4, 4, 5, 7, 8, 16}
	out := make([]CDF, len(sizes))
	for i, n := range sizes {
		if err := out[i].InitUniform(n); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// writeLogTestOps codes ops through every hooked Writer entry point.
func writeLogTestOps(w *Writer, cdfs []CDF, ops []logTestOp, markEvery int) {
	for i, op := range ops {
		if markEvery > 0 && i%markEvery == 0 {
			w.LogSuperblock()
		}
		c := &cdfs[op.cdf]
		switch op.kind {
		case 0:
			w.WriteBoolQ15(op.s&1, op.f)
		case 1:
			w.WriteBit(op.s & 1)
		case 2:
			w.WriteCDF(c, op.s%int(c.symbols))
		case 3:
			w.WriteSymbol(op.s%int(c.symbols), c.values[:], int(c.symbols))
		case 4:
			switch c.symbols {
			case 2:
				w.WriteBinaryCDFTrusted(c, op.s%2)
			case 3:
				w.WriteCDF3(c, op.s%3)
			case 4:
				if op.s%5 == 4 {
					w.WriteCDF4Zero(c)
				} else {
					w.WriteCDF4(c, op.s%4)
				}
			case 5:
				w.WriteCDF5(c, op.s%5)
			case 7:
				w.WriteCDF7(c, op.s%7)
			default:
				w.WriteCDF(c, op.s%int(c.symbols))
			}
		case 5:
			w.WriteLiteral(uint32(op.s), 5)
		}
		if markEvery > 0 && i%markEvery == 1 {
			w.LogCDEFUnits(0, 1)
			w.LogCDEFUnits(0, 1) // later blocks of the unit read nothing
		}
	}
}

func randomLogTestOps(rng *rand.Rand, n int, cdfCount int) []logTestOp {
	ops := make([]logTestOp, n)
	for i := range ops {
		ops[i] = logTestOp{
			kind: rng.Intn(6),
			s:    rng.Intn(64),
			f:    uint32(1 + rng.Intn(CDFProbTop-1)),
			cdf:  rng.Intn(cdfCount),
		}
	}
	return ops
}

// TestWriterLogReplayIdentity: replaying the log with no literals reproduces
// the original writer's bytes exactly, for every hooked symbol writer.
func TestWriterLogReplayIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		ops := randomLogTestOps(rng, 1+rng.Intn(4000), 8)
		orig := NewWriter(nil)
		var log WriterLog
		orig.SetLog(&log)
		writeLogTestOps(&orig, logTestCDFs(t), ops, 37)
		want, err := orig.Finish()
		if err != nil {
			t.Fatal(err)
		}
		got, err := log.Replay(nil, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("trial %d: replay %d bytes != original %d bytes", trial, len(got), len(want))
		}
	}
}

// TestWriterLogReplayLiterals: the spliced cdef_idx literals decode at the
// marked positions (once per superblock unit) and every other symbol still
// decodes to the written value.
func TestWriterLogReplayLiterals(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const markEvery = 29
	for trial := 0; trial < 100; trial++ {
		ops := randomLogTestOps(rng, 1+rng.Intn(3000), 8)
		bits := 1 + rng.Intn(3)
		w := NewWriter(nil)
		var log WriterLog
		w.SetLog(&log)
		writeLogTestOps(&w, logTestCDFs(t), ops, markEvery)
		if _, err := w.Finish(); err != nil {
			t.Fatal(err)
		}
		sbs := (len(ops) + markEvery - 1) / markEvery
		idx := make([]uint8, sbs*4)
		for i := range idx {
			idx[i] = uint8(rng.Intn(1 << bits))
		}
		data, err := log.Replay(nil, idx, bits)
		if err != nil {
			t.Fatal(err)
		}
		r := NewReaderWithCDFUpdate(data, true)
		cdfs := logTestCDFs(t)
		sb := -1
		for i, op := range ops {
			if i%markEvery == 0 {
				sb++
			}
			c := &cdfs[op.cdf]
			var got, want int
			switch op.kind {
			case 0:
				v, _ := r.ReadBoolQ15(uint16(op.f))
				got, want = int(v), op.s&1
			case 1, 5:
				n := uint8(1)
				want = op.s & 1
				if op.kind == 5 {
					n, want = 5, op.s&31
				}
				v, _ := r.ReadBits(n)
				got = int(v)
			case 2, 4:
				want = op.s % int(c.symbols)
				if op.kind == 4 && c.symbols == 2 {
					want = op.s % 2
				}
				if op.kind == 4 && c.symbols == 4 && op.s%5 == 4 {
					want = 0
				}
				got, _ = r.ReadCDF(c)
			case 3:
				want = op.s % int(c.symbols)
				tmp := c.values
				got, _ = r.ReadSymbol(tmp[:], int(c.symbols))
			}
			if got != want {
				t.Fatalf("trial %d op %d kind %d: decoded %d want %d", trial, i, op.kind, got, want)
			}
			if i%markEvery == 1 {
				v, _ := r.ReadBits(uint8(bits))
				if uint8(v) != idx[sb*4] {
					t.Fatalf("trial %d sb %d: cdef_idx %d want %d", trial, sb, v, idx[sb*4])
				}
			}
		}
	}
}

// TestWriterLogDeferredIdentity: a deferred logging writer adapts every CDF
// exactly as a coding writer does, and its log's replay is the coding
// writer's stream, so coding only in Replay changes no byte.
func TestWriterLogDeferredIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 200; trial++ {
		ops := randomLogTestOps(rng, 1+rng.Intn(4000), 8)
		plain := NewWriter(nil)
		plainCDFs := logTestCDFs(t)
		writeLogTestOps(&plain, plainCDFs, ops, 37)
		want, err := plain.Finish()
		if err != nil {
			t.Fatal(err)
		}
		deferred := NewWriter(nil)
		var log WriterLog
		log.SetDeferred(true)
		deferred.SetLog(&log)
		deferredCDFs := logTestCDFs(t)
		writeLogTestOps(&deferred, deferredCDFs, ops, 37)
		for i := range plainCDFs {
			if plainCDFs[i] != deferredCDFs[i] {
				t.Fatalf("trial %d: CDF %d adapted differently while deferred", trial, i)
			}
		}
		got, err := log.Replay(nil, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("trial %d: deferred replay %d bytes != coded %d bytes", trial, len(got), len(want))
		}
	}
}

// TestUpdateCDFWideMatchesValues checks the vector CDF update against
// updateCDFValues for every alphabet from 5 symbols, symbol and count.
func TestUpdateCDFWideMatchesValues(t *testing.T) {
	if !symbol16SIMD {
		t.Skip("no vector update")
	}
	rng := rand.New(rand.NewSource(4))
	for n := 5; n <= MaxSymbols; n++ {
		for trial := 0; trial < 2000; trial++ {
			var c CDF
			if err := c.InitUniform(n); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < trial%40; i++ {
				updateCDFValues(&c.values, n, rng.Intn(n))
			}
			got, want := c.values, c.values
			s := rng.Intn(n)
			updateCDFWide(&got, n, s)
			updateCDFValues(&want, n, s)
			if got != want {
				t.Fatalf("n=%d s=%d: got %v want %v", n, s, got, want)
			}
		}
	}
}

const (
	roundTripRowBinary = iota
	roundTripRowCDF3
	roundTripRowCDF4
	roundTripRowCDF5
	roundTripRowCDF7
	roundTripRowCDF8
	roundTripRowCDF16
	roundTripRows
)

var roundTripRowSymbols = [roundTripRows]int{2, 3, 4, 5, 7, 8, 16}

type roundTripCDFs struct {
	Rows [roundTripRows]CDF
}

type roundTripFixedCDF struct {
	symbols int
	values  []uint16
}

type roundTripRecord struct {
	bytes []byte
	cdfs  roundTripCDFs
	err   error
}

type roundTripOpKind uint8

const (
	roundTripOpBoolQ15 roundTripOpKind = iota
	roundTripOpBit
	roundTripOpLiteral
	roundTripOpFixedSymbol
	roundTripOpAdaptiveSymbol
	roundTripOpCDF
	roundTripOpBinaryCDF
	roundTripOpCDF3
	roundTripOpCDF4
	roundTripOpCDF4Zero
	roundTripOpCDF5
	roundTripOpCDF7
	roundTripOpCoeffRun
)

type roundTripOp struct {
	kind  roundTripOpKind
	row   uint8
	sym   uint8
	prob  uint16
	nbits uint8
	value uint32
	level uint16
	sign  uint8
	dc    bool
}

func roundTripICDFFromWeights(t *testing.T, weights []int) []uint16 {
	t.Helper()
	total := 0
	for _, weight := range weights {
		if weight <= 0 {
			t.Fatalf("invalid fixed CDF weight %d", weight)
		}
		total += weight
	}
	cumulative := make([]uint16, len(weights)-1)
	sum := 0
	for i := range cumulative {
		sum += weights[i]
		v := (sum * CDFProbTop) / total
		if v <= 0 || v >= CDFProbTop {
			t.Fatalf("invalid fixed CDF cumulative %d from weights %v", v, weights)
		}
		cumulative[i] = uint16(v)
	}
	icdf := make([]uint16, len(weights)+1)
	if err := InitCDF(icdf, cumulative); err != nil {
		t.Fatalf("InitCDF fixed %v: %v", weights, err)
	}
	return icdf
}

func roundTripFixedCDFs(t *testing.T) []roundTripFixedCDF {
	t.Helper()
	weights := [][]int{
		{1, 5},
		{3, 1, 4, 1},
		{2, 7, 1, 8, 2, 8, 1, 8},
		{1, 1, 2, 3, 5, 8, 13, 21, 3, 2, 1, 4, 6, 7, 9, 11},
	}
	out := make([]roundTripFixedCDF, len(weights))
	for i, row := range weights {
		out[i] = roundTripFixedCDF{
			symbols: len(row),
			values:  roundTripICDFFromWeights(t, row),
		}
	}
	return out
}

func roundTripSeedRow(t *testing.T, rng *rand.Rand, cdf *CDF, symbols, shape int) {
	t.Helper()
	if symbols < 2 || symbols > MaxSymbols {
		t.Fatalf("invalid symbols=%d", symbols)
	}
	cumulative := make([]uint16, symbols-1)
	switch shape % 4 {
	case 0:
		if err := cdf.InitUniform(symbols); err != nil {
			t.Fatalf("InitUniform(%d): %v", symbols, err)
		}
	case 1:
		v := 0
		for i := range cumulative {
			v += 1 + rng.Intn(3)
			cumulative[i] = uint16(v)
		}
		if err := cdf.Init(cumulative); err != nil {
			t.Fatalf("tail-heavy Init(%d): %v", symbols, err)
		}
	case 2:
		v := CDFProbTop
		for i := len(cumulative) - 1; i >= 0; i-- {
			v -= 1 + rng.Intn(3)
			cumulative[i] = uint16(v)
		}
		if err := cdf.Init(cumulative); err != nil {
			t.Fatalf("head-heavy Init(%d): %v", symbols, err)
		}
	default:
		vals := make([]int, len(cumulative))
		for i := range vals {
			for {
				v := 1 + rng.Intn(CDFProbTop-1)
				used := false
				for j := 0; j < i; j++ {
					if vals[j] == v {
						used = true
						break
					}
				}
				if !used {
					vals[i] = v
					break
				}
			}
		}
		sort.Ints(vals)
		for i, v := range vals {
			cumulative[i] = uint16(v)
		}
		if err := cdf.Init(cumulative); err != nil {
			t.Fatalf("random Init(%d): %v", symbols, err)
		}
	}
	warm := rng.Intn(48)
	for i := 0; i < warm; i++ {
		var sym int
		switch shape % 4 {
		case 1:
			if rng.Intn(8) == 0 {
				sym = rng.Intn(symbols)
			} else {
				sym = symbols - 1
			}
		case 2:
			if rng.Intn(8) == 0 {
				sym = rng.Intn(symbols)
			} else {
				sym = 0
			}
		default:
			sym = rng.Intn(symbols)
		}
		if err := cdf.Update(sym); err != nil {
			t.Fatalf("warm Update(%d/%d): %v", sym, symbols, err)
		}
	}
}

func roundTripSeedCDFs(t *testing.T, rng *rand.Rand, shape int) roundTripCDFs {
	t.Helper()
	var cdfs roundTripCDFs
	for row, symbols := range roundTripRowSymbols {
		roundTripSeedRow(t, rng, &cdfs.Rows[row], symbols, shape+row)
	}
	return cdfs
}

func roundTripRandomOp(rng *rand.Rand, fixed []roundTripFixedCDF) roundTripOp {
	switch rng.Intn(13) {
	case 0:
		return roundTripOp{
			kind: roundTripOpBoolQ15,
			sym:  uint8(rng.Intn(2)),
			prob: uint16(1 + rng.Intn(CDFProbTop-1)),
		}
	case 1:
		return roundTripOp{kind: roundTripOpBit, sym: uint8(rng.Intn(2))}
	case 2:
		n := uint8(1 + rng.Intn(24))
		return roundTripOp{
			kind:  roundTripOpLiteral,
			nbits: n,
			value: rng.Uint32() & ((uint32(1) << n) - 1),
		}
	case 3:
		row := uint8(rng.Intn(len(fixed)))
		return roundTripOp{
			kind: roundTripOpFixedSymbol,
			row:  row,
			sym:  uint8(rng.Intn(fixed[row].symbols)),
		}
	case 4:
		row := uint8(roundTripRowCDF8 + rng.Intn(2))
		return roundTripOp{
			kind: roundTripOpAdaptiveSymbol,
			row:  row,
			sym:  uint8(rng.Intn(roundTripRowSymbols[row])),
		}
	case 5:
		row := uint8(rng.Intn(roundTripRows))
		return roundTripOp{
			kind: roundTripOpCDF,
			row:  row,
			sym:  uint8(rng.Intn(roundTripRowSymbols[row])),
		}
	case 6:
		return roundTripOp{kind: roundTripOpBinaryCDF, sym: uint8(rng.Intn(2))}
	case 7:
		return roundTripOp{kind: roundTripOpCDF3, sym: uint8(rng.Intn(3))}
	case 8:
		if rng.Intn(4) == 0 {
			return roundTripOp{kind: roundTripOpCDF4Zero}
		}
		return roundTripOp{kind: roundTripOpCDF4, sym: uint8(rng.Intn(4))}
	case 9:
		return roundTripOp{kind: roundTripOpCDF5, sym: uint8(rng.Intn(5))}
	case 10:
		return roundTripOp{kind: roundTripOpCDF7, sym: uint8(rng.Intn(7))}
	default:
		level := 1 + rng.Intn(40)
		if rng.Intn(5) == 0 {
			level = 15 + rng.Intn(64)
		}
		return roundTripOp{
			kind:  roundTripOpCoeffRun,
			level: uint16(level),
			sign:  uint8(rng.Intn(2)),
			dc:    rng.Intn(3) == 0,
		}
	}
}

func roundTripBuildOps(rng *rand.Rand, fixed []roundTripFixedCDF, n int) []roundTripOp {
	ops := make([]roundTripOp, n)
	for i := range ops {
		ops[i] = roundTripRandomOp(rng, fixed)
	}
	return ops
}

func roundTripWriteGolomb(w *Writer, level int) {
	x := level + 1
	length := 0
	for i := x; i != 0; i >>= 1 {
		length++
	}
	for range length - 1 {
		w.WriteBit(0)
	}
	for i := length - 1; i >= 0; i-- {
		w.WriteBit((x >> uint(i)) & 1)
	}
}

const (
	roundTripNumBaseLevels  = 2
	roundTripBaseRange      = 12
	roundTripBRCDFSize      = 4
	roundTripMaxBaseBRRange = 15
)

func roundTripWriteBR(w *Writer, cdf *CDF, level int) {
	baseRange := level - 1 - roundTripNumBaseLevels
	for idx := 0; idx < roundTripBaseRange; idx += roundTripBRCDFSize - 1 {
		k := min(baseRange-idx, roundTripBRCDFSize-1)
		w.WriteCDF4(cdf, k)
		if k < roundTripBRCDFSize-1 {
			break
		}
	}
}

func roundTripWriteCoeffRun(w *Writer, cdfs *roundTripCDFs, op roundTripOp) {
	level := int(op.level)
	baseSym := min(level, 3) - 1
	w.WriteCDF3(&cdfs.Rows[roundTripRowCDF3], baseSym)
	if level > roundTripNumBaseLevels {
		roundTripWriteBR(w, &cdfs.Rows[roundTripRowCDF4], level)
	}
	if op.dc {
		w.WriteBinaryCDFTrusted(&cdfs.Rows[roundTripRowBinary], int(op.sign))
	} else {
		w.WriteBit(int(op.sign))
	}
	if level >= roundTripMaxBaseBRRange {
		roundTripWriteGolomb(w, level-roundTripMaxBaseBRRange)
	}
}

func roundTripApplyOp(w *Writer, cdfs *roundTripCDFs, fixed []roundTripFixedCDF, op roundTripOp) {
	switch op.kind {
	case roundTripOpBoolQ15:
		w.WriteBoolQ15(int(op.sym), uint32(op.prob))
	case roundTripOpBit:
		w.WriteBit(int(op.sym))
	case roundTripOpLiteral:
		w.WriteLiteral(op.value, int(op.nbits))
	case roundTripOpFixedSymbol:
		row := fixed[op.row]
		w.WriteSymbol(int(op.sym), row.values, row.symbols)
	case roundTripOpAdaptiveSymbol:
		row := &cdfs.Rows[op.row]
		w.WriteSymbolAdaptive(int(op.sym), row.Values())
	case roundTripOpCDF:
		w.WriteCDF(&cdfs.Rows[op.row], int(op.sym))
	case roundTripOpBinaryCDF:
		w.WriteBinaryCDFTrusted(&cdfs.Rows[roundTripRowBinary], int(op.sym))
	case roundTripOpCDF3:
		w.WriteCDF3(&cdfs.Rows[roundTripRowCDF3], int(op.sym))
	case roundTripOpCDF4:
		w.WriteCDF4(&cdfs.Rows[roundTripRowCDF4], int(op.sym))
	case roundTripOpCDF4Zero:
		w.WriteCDF4Zero(&cdfs.Rows[roundTripRowCDF4])
	case roundTripOpCDF5:
		w.WriteCDF5(&cdfs.Rows[roundTripRowCDF5], int(op.sym))
	case roundTripOpCDF7:
		w.WriteCDF7(&cdfs.Rows[roundTripRowCDF7], int(op.sym))
	case roundTripOpCoeffRun:
		roundTripWriteCoeffRun(w, cdfs, op)
	}
}

func roundTripEncodeStream(ops []roundTripOp, initial roundTripCDFs, fixed []roundTripFixedCDF, dstCap int) roundTripRecord {
	cdfs := initial
	w := NewWriter(make([]byte, 0, dstCap))
	for _, op := range ops {
		roundTripApplyOp(&w, &cdfs, fixed, op)
	}
	buf, err := w.Finish()
	return roundTripRecord{bytes: buf, cdfs: cdfs, err: err}
}

func roundTripReadGolomb(t *testing.T, r *Reader, opIndex int) int {
	t.Helper()
	zeros := 0
	for {
		bit, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d golomb prefix: %v", opIndex, err)
		}
		if bit != 0 {
			break
		}
		zeros++
		if zeros > 32 {
			t.Fatalf("op=%d golomb prefix too long", opIndex)
		}
	}
	value := 1
	for i := 0; i < zeros; i++ {
		bit, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d golomb suffix[%d]: %v", opIndex, i, err)
		}
		value = (value << 1) | int(bit)
	}
	return value - 1
}

func roundTripReadSymbol(t *testing.T, r *Reader, cdf *CDF, want int, opIndex int, label string) {
	t.Helper()
	got, err := r.ReadSymbol(cdf.Values(), cdf.Symbols())
	if err != nil {
		t.Fatalf("op=%d %s ReadSymbol: %v", opIndex, label, err)
	}
	if got != want {
		t.Fatalf("op=%d %s symbol=%d want %d", opIndex, label, got, want)
	}
}

func roundTripReadBR(t *testing.T, r *Reader, cdf *CDF, level, opIndex int) {
	t.Helper()
	baseRange := level - 1 - roundTripNumBaseLevels
	for idx := 0; idx < roundTripBaseRange; idx += roundTripBRCDFSize - 1 {
		want := min(baseRange-idx, roundTripBRCDFSize-1)
		roundTripReadSymbol(t, r, cdf, want, opIndex, "coeff-br")
		if want < roundTripBRCDFSize-1 {
			break
		}
	}
}

func roundTripDecodeCoeffRun(t *testing.T, r *Reader, cdfs *roundTripCDFs, op roundTripOp, opIndex int) {
	t.Helper()
	level := int(op.level)
	roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF3], min(level, 3)-1, opIndex, "coeff-base-eob")
	if level > roundTripNumBaseLevels {
		roundTripReadBR(t, r, &cdfs.Rows[roundTripRowCDF4], level, opIndex)
	}
	if op.dc {
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowBinary], int(op.sign), opIndex, "coeff-dc-sign")
	} else {
		got, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d coeff-sign ReadBit: %v", opIndex, err)
		}
		if got != op.sign {
			t.Fatalf("op=%d coeff-sign=%d want %d", opIndex, got, op.sign)
		}
	}
	if level >= roundTripMaxBaseBRRange {
		got := roundTripReadGolomb(t, r, opIndex)
		want := level - roundTripMaxBaseBRRange
		if got != want {
			t.Fatalf("op=%d coeff-golomb=%d want %d", opIndex, got, want)
		}
	}
}

func roundTripDecodeOp(t *testing.T, r *Reader, cdfs *roundTripCDFs, fixed []roundTripFixedCDF, op roundTripOp, opIndex int) {
	t.Helper()
	switch op.kind {
	case roundTripOpBoolQ15:
		got, err := r.ReadBoolQ15(op.prob)
		if err != nil {
			t.Fatalf("op=%d ReadBoolQ15: %v", opIndex, err)
		}
		if got != op.sym {
			t.Fatalf("op=%d bool=%d want %d", opIndex, got, op.sym)
		}
	case roundTripOpBit:
		got, err := r.ReadBit()
		if err != nil {
			t.Fatalf("op=%d ReadBit: %v", opIndex, err)
		}
		if got != op.sym {
			t.Fatalf("op=%d bit=%d want %d", opIndex, got, op.sym)
		}
	case roundTripOpLiteral:
		got, err := r.ReadBits(op.nbits)
		if err != nil {
			t.Fatalf("op=%d ReadBits(%d): %v", opIndex, op.nbits, err)
		}
		if got != op.value {
			t.Fatalf("op=%d literal=%#x want %#x", opIndex, got, op.value)
		}
	case roundTripOpFixedSymbol:
		row := fixed[op.row]
		cdf := append([]uint16(nil), row.values...)
		got, err := r.ReadSymbol(cdf, row.symbols)
		if err != nil {
			t.Fatalf("op=%d fixed ReadSymbol: %v", opIndex, err)
		}
		if got != int(op.sym) {
			t.Fatalf("op=%d fixed symbol=%d want %d", opIndex, got, op.sym)
		}
	case roundTripOpAdaptiveSymbol:
		roundTripReadSymbol(t, r, &cdfs.Rows[op.row], int(op.sym), opIndex, "adaptive")
	case roundTripOpCDF:
		roundTripReadSymbol(t, r, &cdfs.Rows[op.row], int(op.sym), opIndex, "cdf")
	case roundTripOpBinaryCDF:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowBinary], int(op.sym), opIndex, "binary")
	case roundTripOpCDF3:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF3], int(op.sym), opIndex, "cdf3")
	case roundTripOpCDF4:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF4], int(op.sym), opIndex, "cdf4")
	case roundTripOpCDF4Zero:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF4], 0, opIndex, "cdf4-zero")
	case roundTripOpCDF5:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF5], int(op.sym), opIndex, "cdf5")
	case roundTripOpCDF7:
		roundTripReadSymbol(t, r, &cdfs.Rows[roundTripRowCDF7], int(op.sym), opIndex, "cdf7")
	case roundTripOpCoeffRun:
		roundTripDecodeCoeffRun(t, r, cdfs, op, opIndex)
	}
}

func roundTripRoundTrip(t *testing.T, tag string, rec roundTripRecord, ops []roundTripOp, initial roundTripCDFs, fixed []roundTripFixedCDF) {
	t.Helper()
	if rec.err != nil {
		t.Fatalf("%s: Finish: %v", tag, rec.err)
	}
	cdfs := initial
	r := NewReaderWithCDFUpdate(rec.bytes, true)
	for i, op := range ops {
		roundTripDecodeOp(t, &r, &cdfs, fixed, op, i)
	}
	if cdfs != rec.cdfs {
		t.Fatalf("%s decode CDF image mismatch: got %#v want %#v", tag, cdfs, rec.cdfs)
	}
}

func TestWriterReaderRoundTrip(t *testing.T) {
	streams := 24
	if testing.Short() {
		streams = 4
	}
	fixed := roundTripFixedCDFs(t)
	rng := rand.New(rand.NewSource(0xb0a51d))
	for shape := 0; shape < 4; shape++ {
		for stream := 0; stream < streams; stream++ {
			initial := roundTripSeedCDFs(t, rng, shape+stream)
			ops := roundTripBuildOps(rng, fixed, 96+rng.Intn(192))
			dstCap := rng.Intn(19)
			rec := roundTripEncodeStream(ops, initial, fixed, dstCap)
			tag := "shape=" + strconv.Itoa(shape) + "/stream=" + strconv.Itoa(stream) + "/dstCap=" + strconv.Itoa(dstCap)
			roundTripRoundTrip(t, tag, rec, ops, initial, fixed)
		}
	}
}

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
		switch rng.Intn(6) {
		case 0:
			bit := rng.Intn(2)
			normal.WriteBit(bit)
			count.WriteBit(bit)
			bitCounter.WriteBit(bit)
		case 1:
			sym := rng.Intn(4)
			if sym == 0 {
				normal.WriteCDF4Zero(&normalCDF)
				count.WriteCDF4Zero(&countCDF)
				bitCounter.WriteCDF4Zero(&bitCounterCDF)
			} else {
				normal.WriteCDF4(&normalCDF, sym)
				count.WriteCDF4(&countCDF, sym)
				bitCounter.WriteCDF4(&bitCounterCDF, sym)
			}
			if normalCDF != countCDF || normalCDF != bitCounterCDF {
				t.Fatalf("cdf diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalCDF.Values(), countCDF.Values(), bitCounterCDF.Values())
			}
		case 2:
			sym := rng.Intn(3)
			normal.WriteCDF3(&normalCDF3, sym)
			count.WriteCDF3(&countCDF3, sym)
			bitCounter.WriteCDF3(&bitCounterCDF3, sym)
			if normalCDF3 != countCDF3 || normalCDF3 != bitCounterCDF3 {
				t.Fatalf("cdf3 diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalCDF3.Values(), countCDF3.Values(), bitCounterCDF3.Values())
			}
		case 3:
			sym := rng.Intn(2)
			normal.WriteBinaryCDFTrusted(&normalBinaryCDF, sym)
			count.WriteBinaryCDFTrusted(&countBinaryCDF, sym)
			bitCounter.WriteBinaryCDFTrusted(&bitCounterBinaryCDF, sym)
			if normalBinaryCDF != countBinaryCDF || normalBinaryCDF != bitCounterBinaryCDF {
				t.Fatalf("binary cdf diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalBinaryCDF.Values(), countBinaryCDF.Values(), bitCounterBinaryCDF.Values())
			}
		case 4:
			sym := rng.Intn(5)
			normal.WriteCDF5(&normalCDF5, sym)
			count.WriteCDF5(&countCDF5, sym)
			bitCounter.WriteCDF5(&bitCounterCDF5, sym)
			if normalCDF5 != countCDF5 || normalCDF5 != bitCounterCDF5 {
				t.Fatalf("cdf5 diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalCDF5.Values(), countCDF5.Values(), bitCounterCDF5.Values())
			}
		case 5:
			sym := rng.Intn(7)
			normal.WriteCDF7(&normalCDF7, sym)
			count.WriteCDF7(&countCDF7, sym)
			bitCounter.WriteCDF7(&bitCounterCDF7, sym)
			if normalCDF7 != countCDF7 || normalCDF7 != bitCounterCDF7 {
				t.Fatalf("cdf7 diverged after symbol %d: normal=%v count=%v bitCounter=%v", i, normalCDF7.Values(), countCDF7.Values(), bitCounterCDF7.Values())
			}
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
