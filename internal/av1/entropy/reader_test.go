package entropy

import (
	"errors"
	"testing"
)

func TestReaderReadBitKnownValues(t *testing.T) {
	r := NewReader([]byte{0x00})
	bit, err := r.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 0 {
		t.Fatalf("zero stream bit=%d want 0", bit)
	}

	r = NewReader([]byte{0xff})
	bit, err = r.ReadBit()
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("one stream bit=%d want 1", bit)
	}
}

func TestReaderReadBool(t *testing.T) {
	r := NewReader([]byte{0xff})
	bit, err := r.ReadBool(128)
	if err != nil {
		t.Fatal(err)
	}
	if bit != 1 {
		t.Fatalf("bit=%d want 1", bit)
	}

	r = NewReader([]byte{0x00})
	bit, err = r.ReadBool(128)
	if err != nil {
		t.Fatal(err)
	}
	if bit != 0 {
		t.Fatalf("bit=%d want 0", bit)
	}
}

func TestReaderReadBits(t *testing.T) {
	r := NewReader([]byte{0xff, 0x00})
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

func TestReaderReadUniform(t *testing.T) {
	r := NewReader([]byte{0x00})
	got, err := r.ReadUniform(10)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("zero uniform=%d want 0", got)
	}

	r = NewReader([]byte{0xff})
	got, err = r.ReadUniform(10)
	if err != nil {
		t.Fatal(err)
	}
	if got != 9 {
		t.Fatalf("one uniform=%d want 9", got)
	}

	r = NewReader(nil)
	got, err = r.ReadUniform(1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("single uniform=%d want 0", got)
	}
}

func TestReaderReadSubexp(t *testing.T) {
	r := NewReader([]byte{0x00})
	got, err := r.ReadSubexp(16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("zero subexp=%d want 0", got)
	}

	r = NewReader([]byte{0xff, 0xff})
	got, err = r.ReadSubexp(16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got != 15 {
		t.Fatalf("one subexp=%d want 15", got)
	}
}

func TestReaderReadRefSubexp(t *testing.T) {
	r := NewReader([]byte{0x00})
	got, err := r.ReadRefSubexp(10, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got != 4 {
		t.Fatalf("ref subexp=%d want 4", got)
	}
}

func TestReaderReadSignedRefSubexp(t *testing.T) {
	r := NewReader([]byte{0x00})
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
	r := NewReader([]byte{0x00})
	symbol, err := r.ReadSymbol(cdf, 2)
	if err != nil {
		t.Fatal(err)
	}
	if symbol != 0 {
		t.Fatalf("symbol=%d want 0", symbol)
	}
	want := []uint16{15360, 0, 1}
	for i := 0; i < len(want); i++ {
		if cdf[i] != want[i] {
			t.Fatalf("cdf=%v want %v", cdf, want)
		}
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
	r := NewReader(nil)
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
}

func TestReaderAllocs(t *testing.T) {
	src := []byte{0xff, 0x00, 0xa5}
	cdf := []uint16{24576, 16384, 8192, 0, 0}
	allocs := testing.AllocsPerRun(1000, func() {
		r := NewReader(src)
		_, err := r.ReadBits(3)
		if err != nil {
			t.Fatal(err)
		}
		_, err = r.ReadSymbol(cdf, 4)
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

		r := NewReader(src)
		uniform, err := r.ReadUniform(n)
		if err != nil {
			t.Fatalf("ReadUniform err=%v n=%d", err, n)
		}
		if uniform >= n {
			t.Fatalf("uniform=%d n=%d", uniform, n)
		}

		r = NewReader(src)
		subexp, err := r.ReadSubexp(n, k)
		if err != nil {
			t.Fatalf("ReadSubexp err=%v n=%d k=%d", err, n, k)
		}
		if subexp >= n {
			t.Fatalf("subexp=%d n=%d k=%d", subexp, n, k)
		}

		ref := uint32(rawRef) % n
		r = NewReader(src)
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

func BenchmarkReaderReadBit(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := NewReader(src)
		_, _ = r.ReadBit()
	}
}

func BenchmarkReaderReadUniform(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := NewReader(src)
		_, _ = r.ReadUniform(257)
	}
}

func BenchmarkReaderReadSubexp(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := NewReader(src)
		_, _ = r.ReadSubexp(257, 3)
	}
}

func BenchmarkReaderReadSymbol(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}
	cdf := []uint16{24576, 16384, 8192, 0, 0}

	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := NewReader(src)
		_, _ = r.ReadSymbol(cdf, 4)
	}
}
