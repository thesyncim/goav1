package entropy

import (
	"errors"
	"testing"
)

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
