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

func FuzzUpdateCDF(f *testing.F) {
	f.Add(uint8(2), []byte{0, 1, 1, 0})
	f.Add(uint8(4), []byte{2, 3, 0, 1})
	f.Add(uint8(MaxSymbols), []byte{15, 0, 14, 7})

	f.Fuzz(func(t *testing.T, symbolCount uint8, symbols []byte) {
		n := int(symbolCount%MaxSymbols) + 1
		if n < 2 {
			n = 2
		}
		var cdf [MaxSymbols + 1]uint16
		if err := InitUniformCDF(cdf[:], n); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < len(symbols); i++ {
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
	for i := 0; i < b.N; i++ {
		_ = InitUniformCDF(cdf[:], MaxSymbols)
	}
}

const CDFSize4 = 5
