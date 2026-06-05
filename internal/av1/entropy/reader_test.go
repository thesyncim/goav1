package entropy

import (
	"errors"
	"testing"
	"unsafe"
)

func TestReaderHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		size uintptr
		max  uintptr
	}{
		{name: "Reader", size: unsafe.Sizeof(Reader{}), max: 40},
		{name: "Cursor", size: unsafe.Sizeof(Cursor{}), max: 40},
	}
	for _, tt := range tests {
		if tt.size > tt.max {
			t.Fatalf("%s size=%d max=%d", tt.name, tt.size, tt.max)
		}
	}
}

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

func TestReaderReadBitsMatchesRepeatedReadBit(t *testing.T) {
	src := []byte{0xa5, 0x5a, 0xc3, 0x3c, 0xf0, 0x0f, 0x99, 0x66}
	for n := uint8(0); n <= 32; n++ {
		bulk := NewReader(src)
		step := NewReader(src)
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
	r := NewReader([]byte{0x00})
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
	generic := NewReader(src)
	trusted := NewReader(src)

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
			generic := NewReader(src)
			trusted := NewReader(src)

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
				if gotBit := got.ReadBitTrusted(); gotBit != wantBit {
					t.Fatalf("bit[%d]=%d want %d", i, gotBit, wantBit)
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
	r := NewReader([]byte{0x00})
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
	r := NewReader([]byte{0xff, 0xff, 0xff})
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
		r := NewReader(src)
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

func FuzzReaderSignedDelta(f *testing.F) {
	f.Add([]byte{0x00})
	f.Add([]byte{0xff, 0xff, 0xff})
	f.Add([]byte{0xa5, 0x5a})

	f.Fuzz(func(t *testing.T, src []byte) {
		var cdf CDF
		if err := cdf.InitDefaultDelta(); err != nil {
			t.Fatal(err)
		}
		r := NewReader(src)
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
		r := NewReader(src)
		_, _ = r.ReadBit()
	}
}

func BenchmarkCursorReadBitTrustedStream(b *testing.B) {
	src := benchStream()
	sum := 0

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReader(src)
		c := r.Cursor()
		for range benchSymbolsPerOp {
			sum += int(c.ReadBitTrusted())
		}
	}
	benchmarkReaderSink = sum
}

func BenchmarkReaderReadBits(b *testing.B) {
	src := benchStream()

	b.ReportAllocs()
	b.SetBytes(benchSymbolsPerOp)
	for b.Loop() {
		r := NewReader(src)
		for range benchSymbolsPerOp {
			_, _ = r.ReadBits(8)
		}
	}
}

func BenchmarkReaderReadUniform(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReader(src)
		_, _ = r.ReadUniform(257)
	}
}

func BenchmarkReaderReadSubexp(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReader(src)
		_, _ = r.ReadSubexp(257, 3)
	}
}

func BenchmarkReaderReadSymbol(b *testing.B) {
	src := []byte{0xff, 0x00, 0xa5, 0x5a}
	cdf := []uint16{24576, 16384, 8192, 0, 0}

	b.ReportAllocs()
	for b.Loop() {
		r := NewReader(src)
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
		r := NewReader(src)
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
		r := NewReader(src)
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
		r := NewReader(src)
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
		r := NewReader(src)
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

func BenchmarkCursorCDFStream10(b *testing.B) {
	benchmarkCursorCDFStreamSymbols(b, 10)
}

func BenchmarkCursorCDFSymbolsStream10(b *testing.B) {
	benchmarkCursorCDFSymbolsStream(b, 10)
}

func BenchmarkCursorCDFStream11(b *testing.B) {
	benchmarkCursorCDFStreamSymbols(b, 11)
}

func BenchmarkCursorCDFSymbolsStream11(b *testing.B) {
	benchmarkCursorCDFSymbolsStream(b, 11)
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
		r := NewReader(src)
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
		r := NewReader(src)
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
		r := NewReader(src)
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
		r := NewReader(src)
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
		r := NewReader(src)
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
		r := NewReader(src)
		for range benchSymbolsPerOp {
			if _, err := r.ReadBoolQ15(CDFProbTop / 2); err != nil {
				b.Fatal(err)
			}
		}
	}
}
