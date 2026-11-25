//go:build goexperiment.simd && amd64 && !purego

package transform

import (
	"encoding/binary"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"simd/archsimd"
	"strings"
	"testing"
)

func TestDCT2LaneSIMDDispatchBound(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"DCT8 row", inverseDCT8Row2Impl, inverseDCT8Row2SIMDAdapter},
		{"DCT16 row", inverseDCT16Row2Impl, inverseDCT16Row2SIMDAdapter},
		{"DCT8 column", inverseDCT8Col2Impl, inverseDCT8Col2SIMDAdapter},
		{"DCT16 column", inverseDCT16Col2Impl, inverseDCT16Col2SIMDAdapter},
	}
	for _, check := range checks {
		if reflect.ValueOf(check.got).Pointer() != reflect.ValueOf(check.want).Pointer() {
			t.Errorf("%s dispatch is not bound to Go SIMD", check.name)
		}
	}
}

func TestDCT2LaneSIMDMatchesScalarFullRange(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("AVX2 unavailable")
	}
	rng := rand.New(rand.NewSource(0xDC7216))
	clamps := [][2]int32{
		{-32768, 32767},
		{-(1 << 19), (1 << 19) - 1},
		{math.MinInt32, math.MaxInt32},
		{-1, 1},
	}
	for _, tc := range []struct {
		name string
		n    int
		row  func([]int32, []int32, int32, int32)
		col  func([]int32, int, int32, int32)
		one  func([]int32, int, int32, int32)
	}{
		{"DCT8", dct8Size, inverseDCT8Row2SIMDAdapter, inverseDCT8Col2SIMDAdapter, func(x []int32, stride int, lo, hi int32) { inverseDCT8(x, stride, lo, hi) }},
		{"DCT16", dct16Size, inverseDCT16Row2SIMDAdapter, inverseDCT16Col2SIMDAdapter, func(x []int32, stride int, lo, hi int32) { inverseDCT16(x, stride, lo, hi) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for iter := 0; iter < 80; iter++ {
				lo, hi := clamps[iter%len(clamps)][0], clamps[iter%len(clamps)][1]
				for offset := 0; offset < 4; offset++ {
					got := make([]int32, offset+2*tc.n+5)
					for i := range got {
						switch i % 11 {
						case 0:
							got[i] = math.MinInt32
						case 1:
							got[i] = math.MaxInt32
						case 2:
							got[i] = lo
						case 3:
							got[i] = hi
						default:
							got[i] = int32(rng.Uint32())
						}
					}
					want := append([]int32(nil), got...)
					tc.row(got[offset:offset+tc.n], got[offset+tc.n+1:offset+2*tc.n+1], lo, hi)
					tc.one(want[offset:offset+tc.n], 1, lo, hi)
					tc.one(want[offset+tc.n+1:offset+2*tc.n+1], 1, lo, hi)
					if !reflect.DeepEqual(got, want) {
						for i := range got {
							if got[i] != want[i] {
								t.Fatalf("row mismatch iter=%d offset=%d clamp=[%d,%d] index=%d got=%d want=%d", iter, offset, lo, hi, i, got[i], want[i])
							}
						}
					}
				}

				rowStride := 2 + iter%9
				offset := iter % 4
				length := (tc.n-1)*rowStride + 2
				got := make([]int32, offset+length+5)
				for i := range got {
					switch i % 13 {
					case 0:
						got[i] = math.MinInt32
					case 1:
						got[i] = math.MaxInt32
					case 2:
						got[i] = lo
					case 3:
						got[i] = hi
					default:
						got[i] = int32(rng.Uint32())
					}
				}
				want := append([]int32(nil), got...)
				tc.col(got[offset:offset+length], rowStride, lo, hi)
				tc.one(want[offset:], rowStride, lo, hi)
				tc.one(want[offset+1:], rowStride, lo, hi)
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("column mismatch iter=%d offset=%d stride=%d clamp=[%d,%d]", iter, offset, rowStride, lo, hi)
				}
			}
		})
	}
}

// TestForwardDCT8x8SIMDMatchesPureGo proves the AVX2 8x8 kernel bit-exact with
// the portable reference across random 8-bit residual ranges and strides. It
// calls the kernel directly so hosts whose CPUID hides AVX2 (Rosetta) still
// exercise the VEX path.
func TestForwardDCT8x8SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(67))
	const resStride, coeffStride = 23, 17
	residual := make([]int16, resStride*16)
	for trial := range 3000 {
		for i := range residual {
			residual[i] = int16(rng.Intn(511)) - 255
		}
		want := make([]int32, coeffStride*16)
		got := make([]int32, coeffStride*16)
		forwardDCT8x8PureGo(want, coeffStride, residual, resStride)
		forwardDCT8x8SIMD(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] simd %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

// TestForwardDCT8x8SIMDBinding checks that the dispatcher selects the AVX2
// kernel whenever the host advertises AVX2.
func TestForwardDCTSIMDBindingsAMD64(t *testing.T) {
	if !archsimd.X86.AVX2() {
		t.Skip("host has no AVX2")
	}
	check := func(name string, got any, want string) {
		t.Helper()
		fn := runtime.FuncForPC(reflect.ValueOf(got).Pointer())
		if fn == nil || !strings.Contains(fn.Name(), want) {
			t.Fatalf("%s forward DCT bound to %v, want %s", name, fn, want)
		}
	}
	check("4x4", forwardDCT4x4Impl, "forwardDCT4x4SIMD")
	check("8x8", forwardDCT8x8Impl, "forwardDCT8x8SIMDGuarded")
	check("16x16", forwardDCT16x16Impl, "forwardDCT16x16SIMDGuarded")
	check("32x32", forwardDCT32x32Impl, "forwardDCT32x32SIMDGuarded")
}

// TestForwardDCT4x4SIMDMatchesPureGo proves the AVX2 4x4 kernel bit-exact with
// the portable reference over random residuals across the guard range and
// beyond it (the guarded entry routes out-of-range inputs to the scalar path).
func TestForwardDCT4x4SIMDMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	const resStride, coeffStride = 13, 9
	residual := make([]int16, resStride*4)
	for trial := range 3000 {
		span := 511
		if trial%2 == 1 {
			span = 4097
		}
		for i := range residual {
			residual[i] = int16(rng.Intn(span) - span/2)
		}
		want := make([]int32, coeffStride*4)
		got := make([]int32, coeffStride*4)
		forwardDCT4x4PureGo(want, coeffStride, residual, resStride)
		forwardDCT4x4SIMD(got, coeffStride, residual, resStride)
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("trial %d: coeff[%d] simd %d want %d", trial, i, got[i], want[i])
			}
		}
	}
}

type itxRectFused func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool

// testInverseAddRect checks a fused w x h kernel against the staged
// transform and residual add for every type but IDTX: coefficients inside
// the coded extent, garbage in the rows past it (the staged path never
// reads them), magnitudes past the int16 clamps, and separate pred/dst
// buffers. Realistic magnitudes must never decline.
func testInverseAddRect(t *testing.T, w, h int, fused itxRectFused) {
	if !itxAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(int64(91 + w*h + w)))
	stride := w + 8
	coeff := make([]int32, w*h)
	raw := make([]int32, w*h)
	pred := make([]byte, h*stride)
	got := make([]byte, h*stride)
	want := make([]byte, h*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 100000; trial++ {
		txType := Type(trial % int(TypeCount))
		if txType == TypeIDTX {
			continue
		}
		activeRows, activeCols := 1+rng.Intn(h), 1+rng.Intn(w)
		mode := (trial / int(TypeCount)) % 6
		mag := []int{16, 256, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < w; c++ {
			for r := 0; r < h; r++ {
				v := int32(0)
				switch {
				case r >= activeRows:
					v = int32(rng.Intn(2001) - 1000)
				case c < activeCols && rng.Intn(3) != 0:
					v = int32(rng.Intn(2*mag+1) - mag)
				}
				coeff[c*h+r] = v
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		if err := InverseBlockBitDepthRawBounded(raw, coeff, h, Size{Width: uint8(w), Height: uint8(h)}, txType, 8, activeRows, activeCols); err != nil {
			t.Fatal(err)
		}
		copy(want, pred)
		for r := 0; r < h; r++ {
			for c := 0; c < w; c++ {
				v := int(pred[r*stride+c]) + int((int64(raw[r*w+c])+8)>>4)
				want[r*stride+c] = byte(min(max(v, 0), 255))
			}
		}
		copy(got, pred)
		ok := fused(got, stride, got, stride, coeff, h, txType, activeRows, activeCols)
		if mode <= 2 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 2 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst", trial)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (type %d, mag %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, txType, mag, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = 0xa5
		}
		if !fused(got, stride, pred, stride, coeff, h, txType, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < h; r++ {
			for c := 0; c < stride; c++ {
				wv := want[r*stride+c]
				if c >= w {
					wv = 0xa5
				}
				if got[r*stride+c] != wv {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("%dx%d: accepted %d, declined %d (realistic magnitudes: %d of %d declined)", w, h, accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if fused(got, stride, pred, stride, coeff, h, TypeIDTX, 1, 1) {
		t.Fatal("IDTX accepted")
	}
}

func TestInverseAdd4x8MatchesStaged(t *testing.T) { testInverseAddRect(t, 4, 8, InverseAdd4x8) }
func TestInverseAdd8x4MatchesStaged(t *testing.T) { testInverseAddRect(t, 8, 4, InverseAdd8x4) }

func TestInverseAdd4x16MatchesStaged(t *testing.T) { testInverseAddRect(t, 4, 16, InverseAdd4x16) }
func TestInverseAdd16x4MatchesStaged(t *testing.T) { testInverseAddRect(t, 16, 4, InverseAdd16x4) }
func TestInverseAdd8x16MatchesStaged(t *testing.T) { testInverseAddRect(t, 8, 16, InverseAdd8x16) }
func TestInverseAdd16x8MatchesStaged(t *testing.T) { testInverseAddRect(t, 16, 8, InverseAdd16x8) }

// The square 8x8 kernels load every coefficient column whole, so unlike the
// rect kernels they need the rows past activeRows zeroed, as the decoder's
// coefficient buffers always are: clear the garbage testInverseAddRect plants.
func TestInverseAdd8x8AllTypesMatchesStaged(t *testing.T) {
	var zeroed [64]int32
	testInverseAddRect(t, 8, 8, func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool {
		for c := 0; c < 8; c++ {
			for r := 0; r < 8; r++ {
				v := int32(0)
				if r < activeRows {
					v = coeff[c*coeffStride+r]
				}
				zeroed[c*8+r] = v
			}
		}
		if txType == TypeDCTDCT {
			return InverseAddDCT8x8(dst, dstStride, pred, predStride, zeroed[:], 8, activeRows, activeCols)
		}
		return InverseAdd8x8Hybrid(dst, dstStride, pred, predStride, zeroed[:], 8, txType, activeRows, activeCols)
	})
}

// TestInverseAdd4x4MatchesStaged checks the fused 4x4 kernel against the
// staged transform and residual add for every type but IDTX, every coded
// row extent, magnitudes past the int16 clamps, garbage below the coded
// rows, and separate pred/dst buffers.
func TestInverseAdd4x4MatchesStaged(t *testing.T) {
	if !itxAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewSource(44))
	const n, stride = 4, 4 + 8
	coeff := make([]int32, n*n)
	raw := make([]int32, n*n)
	pred := make([]byte, n*stride)
	got := make([]byte, n*stride)
	want := make([]byte, n*stride)
	for trial := 0; trial < 200000; trial++ {
		txType := Type(trial % int(TypeCount))
		if txType == TypeIDTX {
			continue
		}
		activeRows := 1 + rng.Intn(n)
		mag := []int{16, 256, 4096, 32767, 70000, 1 << 30}[(trial/int(TypeCount))%6]
		for c := 0; c < n; c++ {
			for r := 0; r < n; r++ {
				coeff[c*n+r] = 0
				if rng.Intn(3) != 0 {
					coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
				}
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: n, Height: n}, txType, 8, activeRows, n); err != nil {
			t.Fatal(err)
		}
		copy(want, pred)
		for r := 0; r < n; r++ {
			for c := 0; c < n; c++ {
				v := int(pred[r*stride+c]) + int((int64(raw[r*n+c])+8)>>4)
				want[r*stride+c] = byte(min(max(v, 0), 255))
			}
		}
		copy(got, pred)
		if !InverseAdd4x4(got, stride, got, stride, coeff, n, txType, activeRows) {
			t.Fatalf("trial %d: type %d declined", trial, txType)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (type %d, mag %d, rows %d): mismatch at row %d col %d: fused %d staged %d",
					trial, txType, mag, activeRows, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = 0xa5
		}
		if !InverseAdd4x4(got, stride, pred, stride, coeff, n, txType, activeRows) {
			t.Fatalf("trial %d: separate-pred call declined", trial)
		}
		for r := 0; r < n; r++ {
			for c := 0; c < stride; c++ {
				w := want[r*stride+c]
				if c >= n {
					w = 0xa5
				}
				if got[r*stride+c] != w {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
		}
	}
	if InverseAdd4x4(got, stride, pred, stride, coeff, n, TypeIDTX, 4) {
		t.Fatal("IDTX accepted")
	}
	if InverseAdd4x4(got, stride, pred, stride, coeff, n, TypeDCTDCT, 0) {
		t.Fatal("activeRows 0 accepted")
	}
}

// TestAMD64FusedITXDispatch checks that the AVX2 fused kernels are live on
// this machine: every 8-bit shape accepts a realistic block.
func TestAMD64FusedITXDispatch(t *testing.T) {
	if !itxAVX2 {
		t.Skip("no AVX2")
	}
	for _, c := range amd64ITXBenchCases() {
		dst := make([]byte, c.w*c.h)
		if !c.fused(dst, c.w, c.coeff) {
			t.Errorf("%s: fused kernel declined a realistic block", c.name)
		}
	}
}

type amd64ITXCase struct {
	name       string
	w, h       int
	size       Size
	txType     Type
	rows, cols int
	coeff      []int32
	fused      func(dst []byte, stride int, coeff []int32) bool
}

func amd64ITXBenchCases() []amd64ITXCase {
	rng := rand.New(rand.NewSource(7))
	mk := func(name string, w, h int, txType Type, rows, cols int, fused func(c amd64ITXCase, dst []byte, stride int, coeff []int32) bool) amd64ITXCase {
		scanW, scanH := min(w, 32), min(h, 32)
		coeff := make([]int32, scanW*scanH)
		for c := 0; c < cols; c++ {
			for r := 0; r < rows; r++ {
				coeff[c*scanH+r] = int32(rng.Intn(513) - 256)
			}
		}
		cs := amd64ITXCase{name: name, w: w, h: h, size: Size{Width: uint8(w), Height: uint8(h)}, txType: txType, rows: rows, cols: cols, coeff: coeff}
		cs.fused = func(dst []byte, stride int, coeff []int32) bool { return fused(cs, dst, stride, coeff) }
		return cs
	}
	sq := func(c amd64ITXCase, dst []byte, stride int, coeff []int32) bool {
		scanH := min(c.h, 32)
		switch {
		case c.w == 4 && c.h == 16:
			return InverseAdd4x16(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.w == 16 && c.h == 4:
			return InverseAdd16x4(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.w == 8 && c.h == 16:
			return InverseAdd8x16(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.w == 16 && c.h == 8:
			return InverseAdd16x8(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.w == 4 && c.h == 8:
			return InverseAdd4x8(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.w == 8 && c.h == 4:
			return InverseAdd8x4(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.w == 4:
			return InverseAdd4x4(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows)
		case c.txType != TypeDCTDCT && c.w == 16:
			return InverseAdd16x16Hybrid(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.txType != TypeDCTDCT:
			return InverseAdd8x8Hybrid(dst, stride, dst, stride, coeff, scanH, c.txType, c.rows, c.cols)
		case c.w == 8:
			return InverseAddDCT8x8(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		case c.w == 16 && c.h == 16:
			return InverseAddDCT16x16(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		case c.w == 16:
			return InverseAddDCT16x32(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		case c.w == 32 && c.h == 16:
			return InverseAddDCT32x16(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		case c.w == 32 && c.rows <= 8 && c.cols <= 8:
			return InverseAddDCT32x32Low8(dst, stride, dst, stride, coeff, scanH)
		case c.w == 32 && c.rows <= 16 && c.cols <= 16:
			return InverseAddDCT32x32Low16(dst, stride, dst, stride, coeff, scanH, c.rows)
		case c.w == 32:
			return InverseAddDCT32x32Full(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		default:
			return InverseAddDCT64x64(dst, stride, dst, stride, coeff, scanH, c.rows, c.cols)
		}
	}
	return []amd64ITXCase{
		mk("4x4/dct", 4, 4, TypeDCTDCT, 4, 4, sq),
		mk("4x4/adst_adst", 4, 4, TypeADSTADST, 4, 4, sq),
		mk("4x4/flipadst_dct", 4, 4, TypeFlipADSTDCT, 4, 4, sq),
		mk("4x4/v_dct", 4, 4, TypeVDCT, 4, 4, sq),
		mk("4x8/dct", 4, 8, TypeDCTDCT, 8, 4, sq),
		mk("4x8/adst_adst", 4, 8, TypeADSTADST, 8, 4, sq),
		mk("8x4/dct", 8, 4, TypeDCTDCT, 4, 8, sq),
		mk("8x4/dct_adst/low4", 8, 4, TypeDCTADST, 4, 4, sq),
		mk("4x16/adst_dct", 4, 16, TypeADSTDCT, 16, 4, sq),
		mk("16x4/dct_adst", 16, 4, TypeDCTADST, 4, 16, sq),
		mk("8x16/dct", 8, 16, TypeDCTDCT, 16, 8, sq),
		mk("16x8/dct", 16, 8, TypeDCTDCT, 8, 16, sq),
		mk("8x8/dct/low4", 8, 8, TypeDCTDCT, 4, 4, sq),
		mk("8x8/dct/full", 8, 8, TypeDCTDCT, 8, 8, sq),
		mk("8x8/adst_adst", 8, 8, TypeADSTADST, 8, 8, sq),
		mk("16x16/dct/low4", 16, 16, TypeDCTDCT, 4, 4, sq),
		mk("16x16/dct/full", 16, 16, TypeDCTDCT, 16, 16, sq),
		mk("16x16/adst_dct", 16, 16, TypeADSTDCT, 8, 8, sq),
		mk("32x32/dct/low8", 32, 32, TypeDCTDCT, 8, 8, sq),
		mk("32x32/dct/low16", 32, 32, TypeDCTDCT, 16, 16, sq),
		mk("32x32/dct/full", 32, 32, TypeDCTDCT, 32, 32, sq),
		mk("64x64/dct/low8", 64, 64, TypeDCTDCT, 8, 8, sq),
		mk("64x64/dct/full", 64, 64, TypeDCTDCT, 32, 32, sq),
		mk("16x32/dct", 16, 32, TypeDCTDCT, 16, 16, sq),
		mk("32x16/dct", 32, 16, TypeDCTDCT, 16, 16, sq),
	}
}

// BenchmarkAMD64FusedITX compares the AVX2 fused transform + add with the
// staged path it replaces (generic: the staged inverse transform into int32
// scratch, then the clipped residual add).
func BenchmarkAMD64FusedITX(b *testing.B) {
	for _, c := range amd64ITXBenchCases() {
		dst := make([]byte, c.w*c.h)
		raw := make([]int32, c.w*c.h)
		b.Run("shape="+c.name+"/impl=generic", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, c.coeff, min(c.h, 32), c.size, c.txType, 8, c.rows, c.cols); err != nil {
					b.Fatal(err)
				}
				for j, v := range raw {
					dst[j] = byte(min(max(int32(dst[j])+((v+8)>>4), 0), 255))
				}
			}
		})
		b.Run("shape="+c.name+"/impl=avx2", func(b *testing.B) {
			if !itxAVX2 {
				b.Skip("no AVX2")
			}
			for i := 0; i < b.N; i++ {
				if !c.fused(dst, c.w, c.coeff) {
					b.Fatal("declined")
				}
			}
		})
	}
}

// itxFused16Oracle is the staged path the fused 16x16 kernel replaces.
func itxFused16Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 16*16)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 16, Height: 16}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 16; r++ {
		for c := 0; c < 16; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*16+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT16x16MatchesStagedPath: every (rows, cols) shape up to
// 16x16, every network form pairing, byte-exact against the staged path;
// declined blocks untouched; no declines at +-64 / +-1024.
func TestInverseAddDCT16x16MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	const stride = 16 + 8
	coeff := make([]int32, 16*16)
	pred := make([]byte, 16*stride)
	got := make([]byte, 16*stride)
	want := make([]byte, 16*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 30000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(16), 1+rng.Intn(16)
		switch trial % 4 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
		case 1:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*16+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused16Oracle(t, want, stride, pred, stride, coeff, 16, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT16x16(got, stride, got, stride, coeff, 16, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT16x16(got, stride, pred, stride, coeff, 16, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 16; r++ {
			for c := 0; c < 16; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 16; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT16x16(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	for _, shape := range []struct {
		name       string
		rows, cols int
	}{{"low4", 4, 4}, {"low8", 8, 8}, {"full", 16, 16}} {
		coeff := make([]int32, 16*16)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*16+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, 16*16)
		for i := range dst {
			dst[i] = byte(rng.Intn(256))
		}
		raw := make([]int32, 16*16)
		b.Run(shape.name+"/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT16x16(dst, 16, dst, 16, coeff, 16, shape.rows, shape.cols) {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, 16, Size{Width: 16, Height: 16}, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// itxFused32Oracle is the staged path the fused kernel replaces: the
// transform into the raw int32 scratch, then Round2(., 4) added to the
// prediction and clipped to bytes.
func itxFused32Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 32*32)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 32; r++ {
		for c := 0; c < 32; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*32+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT32x32Low8MatchesStagedPath: every block the fused path
// accepts reproduces the staged transform plus the raw add byte for byte, a
// declined block leaves dst untouched, and the decline rate at the
// magnitudes real content codes is zero.
func TestInverseAddDCT32x32Low8MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(8), 1+rng.Intn(8)
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		if rng.Intn(4) == 0 {
			coeff[0] = int32(rng.Intn(2*mag+1) - mag)
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		// In place (the decoder: the prediction already sits in dst).
		copy(got, pred)
		ok := InverseAddDCT32x32Low8(got, stride, got, stride, coeff, 32)
		if mode <= 2 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 2 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		// Separate prediction buffer (the encoder's reconstruction).
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT32x32Low8(got, stride, pred, stride, coeff, 32) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 32; r++ {
			for c := 0; c < 32; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 32; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT32x32Low8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 32*32)
	for c := 0; c < 8; c++ {
		for r := 0; r < 8; r++ {
			coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 32*32)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !InverseAddDCT32x32Low8(dst, 32, dst, 32, coeff, 32) {
				b.Fatal("declined")
			}
		}
	})
	raw := make([]int32, 32*32)
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, 8, 8); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// TestInverseAddDCT32x32FullMatchesStagedPath: every (rows, cols) shape up
// to 32x32, every network form pairing, byte-exact against the staged path;
// declined blocks untouched; no declines at +-64 / +-1024.
func TestInverseAddDCT32x32FullMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(32), 1+rng.Intn(32)
		switch trial % 4 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		case 1:
			activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT32x32Full(got, stride, got, stride, coeff, 32, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT32x32Full(got, stride, pred, stride, coeff, 32, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 32; r++ {
			for c := 0; c < 32; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 32; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
}

func BenchmarkInverseAddDCT32x32Full(b *testing.B) {
	for _, shape := range []struct {
		name       string
		rows, cols int
	}{{"low16", 16, 16}, {"full", 32, 32}} {
		rng := rand.New(rand.NewSource(5))
		coeff := make([]int32, 32*32)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, 32*32)
		raw := make([]int32, 32*32)
		b.Run(shape.name+"/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT32x32Full(dst, 32, dst, 32, coeff, 32, shape.rows, shape.cols) {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestInverseAddDCT32x32Low16MatchesStagedPath: every block the fused Low16
// path accepts reproduces the staged transform plus the raw add byte for
// byte over the (rows, cols) shapes up to 16x16, including the eight-row
// shapes that skip the second row strip; declined blocks leave dst untouched
// and no block at realistic magnitudes declines.
func TestInverseAddDCT32x32Low16MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	const stride = 32 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 32*stride)
	got := make([]byte, 32*stride)
	want := make([]byte, 32*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 20000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(16), 1+rng.Intn(16)
		if trial%4 == 0 {
			activeRows = 1 + rng.Intn(8)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused32Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT32x32Low16(got, stride, got, stride, coeff, 32, activeRows)
		// Sixteen coefficients of +-4096 in one row exceed the per-lane
		// guard, so only the +-64 and +-1024 classes must never decline.
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT32x32Low16(got, stride, pred, stride, coeff, 32, activeRows) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 32; r++ {
			for c := 0; c < 32; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 32; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT32x32Low16(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 32*32)
	for c := 0; c < 16; c++ {
		for r := 0; r < 16; r++ {
			coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 32*32)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !InverseAddDCT32x32Low16(dst, 32, dst, 32, coeff, 32, 16) {
				b.Fatal("declined")
			}
		}
	})
	raw := make([]int32, 32*32)
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 32, Height: 32}, TypeDCTDCT, 8, 16, 16); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// itxFused64Oracle is the staged path the fused 64x64 kernel replaces.
func itxFused64Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 64*64)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 64, Height: 64}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 64; r++ {
		for c := 0; c < 64; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*64+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT64x64MatchesStagedPath: every (rows, cols) shape within
// the 32x32 corner, every network form pairing, byte-exact against the
// staged path; declined blocks untouched; no declines at +-64 / +-1024.
func TestInverseAddDCT64x64MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 64 + 8
	coeff := make([]int32, 32*32)
	pred := make([]byte, 64*stride)
	got := make([]byte, 64*stride)
	want := make([]byte, 64*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 6000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(32), 1+rng.Intn(32)
		switch trial % 4 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		case 1:
			activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
		}
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused64Oracle(t, want, stride, pred, stride, coeff, 32, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT64x64(got, stride, got, stride, coeff, 32, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT64x64(got, stride, pred, stride, coeff, 32, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 64; r++ {
			for c := 0; c < 64; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 64; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
}

func BenchmarkInverseAddDCT64x64(b *testing.B) {
	for _, shape := range []struct {
		name       string
		rows, cols int
	}{{"low8", 8, 8}, {"low16", 16, 16}, {"low32", 32, 32}} {
		rng := rand.New(rand.NewSource(5))
		coeff := make([]int32, 32*32)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*32+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, 64*64)
		raw := make([]int32, 64*64)
		b.Run(shape.name+"/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if !InverseAddDCT64x64(dst, 64, dst, 64, coeff, 32, shape.rows, shape.cols) {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: 64, Height: 64}, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestInverseAddDCT64x64HBDMatchesStagedPath: every accepted 10- and 12-bit
// 64x64 block (every network form pairing within the 32x32 corner)
// reproduces the staged transform plus the raw add sample for sample, in
// place and with a separate prediction, without writing past the block; a
// declined block is left untouched and low magnitudes never decline.
func TestInverseAddDCT64x64HBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1064))
	const n = 64
	const stride = 2 * (n + 8)
	for _, bd := range []uint8{10, 12} {
		coeff := make([]int32, 32*32)
		raw := make([]int32, n*n)
		pred := make([]byte, n*stride)
		got := make([]byte, n*stride)
		want := make([]byte, n*stride)
		maxV := 1<<bd - 1
		accepted, declined, lowDeclined, low := 0, 0, 0, 0
		for trial := 0; trial < 4000; trial++ {
			clear(coeff)
			activeRows, activeCols := 1+rng.Intn(32), 1+rng.Intn(32)
			switch trial % 4 {
			case 0:
				activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
			case 1:
				activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
			}
			mode := trial % 6
			mag := []int{64, 1024, 4096, 32767, 70000, 300000}[mode]
			for c := 0; c < activeCols; c++ {
				for r := 0; r < activeRows; r++ {
					if rng.Intn(3) == 0 {
						continue
					}
					coeff[c*32+r] = int32(rng.Intn(2*mag+1) - mag)
				}
			}
			for i := 0; i < len(pred); i += 2 {
				v := rng.Intn(maxV + 1)
				if trial%7 == 0 {
					v = maxV * rng.Intn(2)
				}
				binary.LittleEndian.PutUint16(pred[i:], uint16(v))
			}
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 32, Size{Width: n, Height: n}, TypeDCTDCT, bd, activeRows, activeCols); err != nil {
				t.Fatal(err)
			}
			copy(want, pred)
			for r := 0; r < n; r++ {
				for c := 0; c < n; c++ {
					p := int(binary.LittleEndian.Uint16(pred[r*stride+2*c:]))
					v := min(max(p+int((raw[r*n+c]+8)>>4), 0), maxV)
					binary.LittleEndian.PutUint16(want[r*stride+2*c:], uint16(v))
				}
			}
			copy(got, pred)
			ok := InverseAddDCT64x64HBD(got, stride, got, stride, coeff, 32, activeRows, activeCols, bd)
			if mode <= 1 {
				low++
			}
			if !ok {
				declined++
				if mode <= 1 {
					lowDeclined++
				}
				for i := range got {
					if got[i] != pred[i] {
						t.Fatalf("bd %d trial %d: declined block modified dst at %d", bd, trial, i)
					}
				}
				continue
			}
			accepted++
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
				}
			}
			for i := range got {
				got[i] = 0xa5
			}
			if !InverseAddDCT64x64HBD(got, stride, pred, stride, coeff, 32, activeRows, activeCols, bd) {
				t.Fatalf("bd %d trial %d: separate-pred call declined after the in-place call accepted", bd, trial)
			}
			for r := 0; r < n; r++ {
				for c := 0; c < stride; c++ {
					w := want[r*stride+c]
					if c >= 2*n {
						w = 0xa5
					}
					if got[r*stride+c] != w {
						t.Fatalf("bd %d trial %d: separate-pred mismatch at row %d byte %d", bd, trial, r, c)
					}
				}
			}
		}
		t.Logf("bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", bd, accepted, declined, lowDeclined, low)
		if lowDeclined != 0 || accepted == 0 {
			t.Fatalf("bd %d: %d low-magnitude declines, %d accepted", bd, lowDeclined, accepted)
		}
	}
}

// itxFused8Oracle is the staged path the fused 8x8 kernel replaces.
func itxFused8Oracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, 8*8)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: 8, Height: 8}, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < 8; r++ {
		for c := 0; c < 8; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*8+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCT8x8MatchesStagedPath: every (rows, cols) shape, byte-exact
// against the staged path; declined blocks untouched; no declines at
// +-64 / +-1024 / +-4096 (eight coefficients of +-4096 stay under the guard).
func TestInverseAddDCT8x8MatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 8 + 8
	coeff := make([]int32, 8*8)
	pred := make([]byte, 8*stride)
	got := make([]byte, 8*stride)
	want := make([]byte, 8*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < 30000; trial++ {
		clear(coeff)
		activeRows, activeCols := 1+rng.Intn(8), 1+rng.Intn(8)
		mode := trial % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*8+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFused8Oracle(t, want, stride, pred, stride, coeff, 8, activeRows, activeCols)
		copy(got, pred)
		ok := InverseAddDCT8x8(got, stride, got, stride, coeff, 8, activeRows, activeCols)
		if mode <= 2 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 2 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !InverseAddDCT8x8(got, stride, pred, stride, coeff, 8, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := 8; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
	}
}

func BenchmarkInverseAddDCT8x8(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	coeff := make([]int32, 8*8)
	for c := 0; c < 8; c++ {
		for r := 0; r < 8; r++ {
			coeff[c*8+r] = int32(rng.Intn(2049) - 1024)
		}
	}
	dst := make([]byte, 8*8)
	for i := range dst {
		dst[i] = byte(rng.Intn(256))
	}
	raw := make([]int32, 8*8)
	b.Run("fused", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !InverseAddDCT8x8(dst, 8, dst, 8, coeff, 8, 8, 8) {
				b.Fatal("declined")
			}
		}
	})
	b.Run("staged-transform-only", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if err := InverseBlockBitDepthRawBounded(raw, coeff, 8, Size{Width: 8, Height: 8}, TypeDCTDCT, 8, 8, 8); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// itxFusedHBDOracle is the staged transform at bitDepth plus the raw 16bpc
// add: dst = clip(pred + Round2(raw, 4), 0, 2^bd - 1).
func itxFusedHBDOracle(t *testing.T, n int, bitDepth uint8, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, n*n)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, Size{Width: uint8(n), Height: uint8(n)}, TypeDCTDCT, bitDepth, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	maxV := 1<<bitDepth - 1
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			p := int(binary.LittleEndian.Uint16(pred[r*predStride+2*c:]))
			v := min(max(p+int((raw[r*n+c]+8)>>4), 0), maxV)
			binary.LittleEndian.PutUint16(dst[r*dstStride+2*c:], uint16(v))
		}
	}
}

func itxFusedHBD(n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int, bitDepth uint8) bool {
	switch n {
	case 8:
		return InverseAddDCT8x8HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	case 16:
		return InverseAddDCT16x16HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	default:
		return InverseAddDCT32x32HBD(dst, dstStride, pred, predStride, coeff, coeffStride, activeRows, activeCols, bitDepth)
	}
}

// TestInverseAddDCTHBDMatchesStagedPath: every accepted 10- and 12-bit
// block reproduces the staged transform plus the raw add sample for sample,
// in place and with a separate prediction, without writing past the block;
// a declined block is left untouched.
func TestInverseAddDCTHBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1016))
	for _, n := range []int{8, 16, 32} {
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (n + 8)
			coeff := make([]int32, n*n)
			pred := make([]byte, n*stride)
			got := make([]byte, n*stride)
			want := make([]byte, n*stride)
			maxV := 1<<bd - 1
			trials := map[int]int{8: 6000, 16: 8000, 32: 12000}[n]
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < trials; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(min(n, 8)), 1+rng.Intn(min(n, 8))
				case 1:
					activeRows, activeCols = 1+rng.Intn(min(n, 16)), 1+rng.Intn(min(n, 16))
				}
				mode := trial % 6
				mag := []int{64, 256, 1024, 4096, 32767, 300000}[mode]
				for c := 0; c < activeCols; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := 0; i < len(pred); i += 2 {
					v := rng.Intn(maxV + 1)
					if trial%7 == 0 {
						v = maxV * rng.Intn(2)
					}
					binary.LittleEndian.PutUint16(pred[i:], uint16(v))
				}
				copy(want, pred)
				itxFusedHBDOracle(t, n, bd, want, stride, pred, stride, coeff, n, activeRows, activeCols)
				copy(got, pred)
				ok := itxFusedHBD(n, got, stride, got, stride, coeff, n, activeRows, activeCols, bd)
				if mode <= 1 {
					low++
				}
				if !ok {
					declined++
					if mode <= 1 {
						lowDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("n %d bd %d trial %d: declined block modified dst at %d", n, bd, trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("n %d bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", n, bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !itxFusedHBD(n, got, stride, pred, stride, coeff, n, activeRows, activeCols, bd) {
					t.Fatalf("n %d bd %d trial %d: separate-pred call declined after the in-place call accepted", n, bd, trial)
				}
				for r := 0; r < n; r++ {
					for c := 0; c < stride; c++ {
						w := want[r*stride+c]
						if c >= 2*n {
							w = 0xa5
						}
						if got[r*stride+c] != w {
							t.Fatalf("n %d bd %d trial %d: separate-pred mismatch at row %d byte %d", n, bd, trial, r, c)
						}
					}
				}
			}
			t.Logf("n %d bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", n, bd, accepted, declined, lowDeclined, low)
			if lowDeclined != 0 || accepted == 0 {
				t.Fatalf("n %d bd %d: %d low-magnitude declines, %d accepted", n, bd, lowDeclined, accepted)
			}
		}
	}
}

// TestInverseAddHybridHBDMatchesStagedPath: every accepted 10- and 12-bit
// 16x16 and 8x8 DCT/ADST/flipped-ADST block reproduces the staged transform
// plus the raw add sample for sample, in place and with a separate
// prediction, without writing past the block; a declined block is left
// untouched, and the other transform types are declined.
func TestInverseAddHybridHBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1017))
	types := []Type{TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
		TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST}
	fused := func(n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, txType Type, activeRows, activeCols int, bd uint8) bool {
		if n == 8 {
			return InverseAdd8x8HybridHBD(dst, dstStride, pred, predStride, coeff, n, txType, activeRows, activeCols, bd)
		}
		return InverseAdd16x16HybridHBD(dst, dstStride, pred, predStride, coeff, n, txType, activeRows, activeCols, bd)
	}
	for _, n := range []int{8, 16} {
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (n + 8)
			coeff := make([]int32, n*n)
			raw := make([]int32, n*n)
			pred := make([]byte, n*stride)
			got := make([]byte, n*stride)
			want := make([]byte, n*stride)
			maxV := 1<<bd - 1
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < 12000; trial++ {
				txType := types[trial%len(types)]
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
				if trial%3 == 0 {
					activeRows, activeCols = 1+rng.Intn(min(n, 4)), 1+rng.Intn(min(n, 8))
				}
				mode := (trial / len(types)) % 6
				mag := []int{64, 256, 1024, 4096, 32767, 300000}[mode]
				for c := 0; c < activeCols; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := 0; i < len(pred); i += 2 {
					v := rng.Intn(maxV + 1)
					if trial%7 == 0 {
						v = maxV * rng.Intn(2)
					}
					binary.LittleEndian.PutUint16(pred[i:], uint16(v))
				}
				if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, txType, bd, activeRows, activeCols); err != nil {
					t.Fatal(err)
				}
				copy(want, pred)
				for r := 0; r < n; r++ {
					for c := 0; c < n; c++ {
						p := int(binary.LittleEndian.Uint16(pred[r*stride+2*c:]))
						v := min(max(p+int((raw[r*n+c]+8)>>4), 0), maxV)
						binary.LittleEndian.PutUint16(want[r*stride+2*c:], uint16(v))
					}
				}
				copy(got, pred)
				ok := fused(n, got, stride, got, stride, coeff, txType, activeRows, activeCols, bd)
				if mode <= 1 {
					low++
				}
				if !ok {
					declined++
					if mode <= 1 {
						lowDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("n %d bd %d trial %d: declined block modified dst at %d", n, bd, trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("n %d bd %d type %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", n, bd, txType, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !fused(n, got, stride, pred, stride, coeff, txType, activeRows, activeCols, bd) {
					t.Fatalf("n %d bd %d trial %d: separate-pred call declined after the in-place call accepted", n, bd, trial)
				}
				for r := 0; r < n; r++ {
					for c := 0; c < stride; c++ {
						w := want[r*stride+c]
						if c >= 2*n {
							w = 0xa5
						}
						if got[r*stride+c] != w {
							t.Fatalf("n %d bd %d trial %d: separate-pred mismatch at row %d byte %d", n, bd, trial, r, c)
						}
					}
				}
			}
			t.Logf("n %d bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", n, bd, accepted, declined, lowDeclined, low)
			if lowDeclined != 0 || accepted == 0 {
				t.Fatalf("n %d bd %d: %d low-magnitude declines, %d accepted", n, bd, lowDeclined, accepted)
			}
			for _, other := range []Type{TypeDCTDCT, TypeIDTX, TypeVDCT, TypeHDCT} {
				if fused(n, got, stride, pred, stride, coeff, other, 1, 1, bd) {
					t.Fatalf("n %d: type %d accepted", n, other)
				}
			}
		}
	}
}

var itxHybridTypesUnderTest = []Type{
	TypeADSTDCT, TypeDCTADST, TypeADSTADST, TypeFlipADSTDCT, TypeDCTFlipADST,
	TypeFlipADSTFlipADST, TypeADSTFlipADST, TypeFlipADSTADST,
}

// itxFusedHybridOracle is the staged path the fused hybrid kernels replace.
func itxFusedHybridOracle(t *testing.T, n int, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, txType Type, activeRows, activeCols int) {
	t.Helper()
	raw := make([]int32, n*n)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, txType, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < n; r++ {
		for c := 0; c < n; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*n+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

func testInverseAddHybrid(t *testing.T, n int, trials int, fused func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, txType Type, activeRows, activeCols int) bool) {
	rng := rand.New(rand.NewSource(int64(53 + n)))
	stride := n + 8
	coeff := make([]int32, n*n)
	pred := make([]byte, n*stride)
	got := make([]byte, n*stride)
	want := make([]byte, n*stride)
	accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
	for trial := 0; trial < trials; trial++ {
		clear(coeff)
		txType := itxHybridTypesUnderTest[trial%len(itxHybridTypesUnderTest)]
		activeRows, activeCols := 1+rng.Intn(n), 1+rng.Intn(n)
		switch (trial / len(itxHybridTypesUnderTest)) % 3 {
		case 0:
			activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
		case 1:
			activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
		}
		mode := (trial / 7) % 5
		mag := []int{64, 1024, 4096, 32767, 70000}[mode]
		for c := 0; c < activeCols; c++ {
			for r := 0; r < activeRows; r++ {
				if rng.Intn(3) == 0 {
					continue
				}
				coeff[c*n+r] = int32(rng.Intn(2*mag+1) - mag)
			}
		}
		for i := range pred {
			pred[i] = byte(rng.Intn(256))
		}
		copy(want, pred)
		itxFusedHybridOracle(t, n, want, stride, pred, stride, coeff, txType, activeRows, activeCols)
		copy(got, pred)
		ok := fused(got, stride, got, stride, coeff, n, txType, activeRows, activeCols)
		if mode <= 1 {
			realistic++
		}
		if !ok {
			declined++
			if mode <= 1 {
				realisticDeclined++
			}
			for i := range got {
				if got[i] != pred[i] {
					t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
				}
			}
			continue
		}
		accepted++
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d (type %d, mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
					trial, txType, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
			}
		}
		for i := range got {
			got[i] = byte(0xa5)
		}
		if !fused(got, stride, pred, stride, coeff, n, txType, activeRows, activeCols) {
			t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
		}
		for r := 0; r < n; r++ {
			for c := 0; c < n; c++ {
				if got[r*stride+c] != want[r*stride+c] {
					t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
				}
			}
			for c := n; c < stride; c++ {
				if got[r*stride+c] != 0xa5 {
					t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
				}
			}
		}
	}
	t.Logf("%dx%d: accepted %d, declined %d (realistic magnitudes: %d of %d declined)", n, n, accepted, declined, realisticDeclined, realistic)
	if realisticDeclined != 0 {
		t.Fatalf("%d realistic blocks declined", realisticDeclined)
	}
	if accepted == 0 {
		t.Fatal("no block accepted")
	}
	if fused(got, stride, pred, stride, coeff, n, TypeDCTDCT, 1, 1) {
		t.Fatal("DCT_DCT accepted by the hybrid kernel")
	}
	if fused(got, stride, pred, stride, coeff, n, TypeIDTX, 1, 1) {
		t.Fatal("IDTX accepted by the hybrid kernel")
	}
}

// TestInverseAdd16x16HybridMatchesStagedPath: every hybrid type, every
// (rows, cols) shape and form pairing, byte-exact against the staged path.
func TestInverseAdd16x16HybridMatchesStagedPath(t *testing.T) {
	testInverseAddHybrid(t, 16, 40000, InverseAdd16x16Hybrid)
}

// TestInverseAdd8x8HybridMatchesStagedPath is the 8x8 counterpart.
func TestInverseAdd8x8HybridMatchesStagedPath(t *testing.T) {
	testInverseAddHybrid(t, 8, 40000, InverseAdd8x8Hybrid)
}

func BenchmarkInverseAddHybrid(b *testing.B) {
	for _, shape := range []struct {
		name       string
		n          int
		rows, cols int
	}{{"16x16/low4", 16, 4, 4}, {"16x16/full", 16, 16, 16}, {"8x8/full", 8, 8, 8}} {
		rng := rand.New(rand.NewSource(5))
		n := shape.n
		coeff := make([]int32, n*n)
		for c := 0; c < shape.cols; c++ {
			for r := 0; r < shape.rows; r++ {
				coeff[c*n+r] = int32(rng.Intn(2049) - 1024)
			}
		}
		dst := make([]byte, n*n)
		raw := make([]int32, n*n)
		b.Run(shape.name+"/adst-adst/fused", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				var ok bool
				if n == 16 {
					ok = InverseAdd16x16Hybrid(dst, n, dst, n, coeff, n, TypeADSTADST, shape.rows, shape.cols)
				} else {
					ok = InverseAdd8x8Hybrid(dst, n, dst, n, coeff, n, TypeADSTADST, shape.rows, shape.cols)
				}
				if !ok {
					b.Fatal("declined")
				}
			}
		})
		b.Run(shape.name+"/adst-adst/staged-transform-only", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if err := InverseBlockBitDepthRawBounded(raw, coeff, n, Size{Width: uint8(n), Height: uint8(n)}, TypeADSTADST, 8, shape.rows, shape.cols); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// itxFusedRectOracle is the staged path the fused rectangular kernels
// replace: the raw column output, Round2(., 4), added to pred and saturated.
func itxFusedRectOracle(t *testing.T, dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, size Size, activeRows, activeCols int) {
	t.Helper()
	w, h := int(size.Width), int(size.Height)
	raw := make([]int32, w*h)
	if err := InverseBlockBitDepthRawBounded(raw, coeff, coeffStride, size, TypeDCTDCT, 8, activeRows, activeCols); err != nil {
		t.Fatal(err)
	}
	for r := 0; r < h; r++ {
		for c := 0; c < w; c++ {
			v := int(pred[r*predStride+c]) + int((raw[r*w+c]+8)>>4)
			dst[r*dstStride+c] = byte(min(max(v, 0), 255))
		}
	}
}

// TestInverseAddDCTRectMatchesStagedPath: every (rows, cols) shape of the
// 16x32 and 32x16 blocks, every network form pairing, byte-exact against the
// staged path; declined blocks untouched; no declines at +-64 / +-1024;
// magnitudes up to 2^29 exercise the guards and the rect2 pre-clamp.
func TestInverseAddDCTRectMatchesStagedPath(t *testing.T) {
	for _, tc := range []struct {
		name string
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{"16x32", 16, 32, InverseAddDCT16x32},
		{"32x16", 32, 16, InverseAddDCT32x16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(61 + tc.w)))
			size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
			stride := tc.w + 8
			coeff := make([]int32, tc.w*tc.h)
			pred := make([]byte, tc.h*stride)
			got := make([]byte, tc.h*stride)
			want := make([]byte, tc.h*stride)
			accepted, declined, realistic, realisticDeclined := 0, 0, 0, 0
			for trial := 0; trial < 40000; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(tc.h), 1+rng.Intn(tc.w)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(4), 1+rng.Intn(4)
				case 1:
					activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
				}
				mode := trial % 6
				mag := []int{64, 1024, 4096, 32767, 70000, 1 << 29}[mode]
				for c := 0; c < activeCols; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*tc.h+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := range pred {
					pred[i] = byte(rng.Intn(256))
				}
				copy(want, pred)
				itxFusedRectOracle(t, want, stride, pred, stride, coeff, tc.h, size, activeRows, activeCols)
				copy(got, pred)
				ok := tc.fn(got, stride, got, stride, coeff, tc.h, activeRows, activeCols)
				if mode <= 1 {
					realistic++
				}
				if !ok {
					declined++
					if mode <= 1 {
						realisticDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("trial %d: declined block modified dst at %d", trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("trial %d (mode %d, rows %d cols %d): mismatch at row %d col %d: fused %d staged %d",
							trial, mode, activeRows, activeCols, i/stride, i%stride, got[i], want[i])
					}
				}
				for i := range got {
					got[i] = byte(0xa5)
				}
				if !tc.fn(got, stride, pred, stride, coeff, tc.h, activeRows, activeCols) {
					t.Fatalf("trial %d: separate-pred call declined after the in-place call accepted", trial)
				}
				for r := 0; r < tc.h; r++ {
					for c := 0; c < tc.w; c++ {
						if got[r*stride+c] != want[r*stride+c] {
							t.Fatalf("trial %d: separate-pred mismatch at row %d col %d", trial, r, c)
						}
					}
					for c := tc.w; c < stride; c++ {
						if got[r*stride+c] != 0xa5 {
							t.Fatalf("trial %d: wrote past the block at row %d col %d", trial, r, c)
						}
					}
				}
			}
			t.Logf("accepted %d, declined %d (realistic magnitudes: %d of %d declined)", accepted, declined, realisticDeclined, realistic)
			if realisticDeclined != 0 {
				t.Fatalf("%d blocks at realistic magnitudes declined", realisticDeclined)
			}
		})
	}
}

func BenchmarkInverseAddDCTRect(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	for _, tc := range []struct {
		name string
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{"16x32", 16, 32, InverseAddDCT16x32},
		{"32x16", 32, 16, InverseAddDCT32x16},
	} {
		size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
		for _, shape := range []struct {
			name       string
			rows, cols int
		}{{"low4", 4, 4}, {"low8", 8, 8}, {"full", tc.h, tc.w}} {
			coeff := make([]int32, tc.w*tc.h)
			for c := 0; c < shape.cols; c++ {
				for r := 0; r < shape.rows; r++ {
					coeff[c*tc.h+r] = int32(rng.Intn(2049) - 1024)
				}
			}
			dst := make([]byte, tc.w*tc.h)
			for i := range dst {
				dst[i] = byte(rng.Intn(256))
			}
			raw := make([]int32, tc.w*tc.h)
			b.Run(tc.name+"/"+shape.name+"/fused", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if !tc.fn(dst, tc.w, dst, tc.w, coeff, tc.h, shape.rows, shape.cols) {
						b.Fatal("declined")
					}
				}
			})
			b.Run(tc.name+"/"+shape.name+"/staged-transform-only", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					if err := InverseBlockBitDepthRawBounded(raw, coeff, tc.h, size, TypeDCTDCT, 8, shape.rows, shape.cols); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// TestInverseAddDCTRectScaleEdges: a lone coefficient across the rect2 clamp
// boundary (scale(46341) is the first value past the int16 row clamp) and the
// pre-clamp boundary, at several positions, against the staged path.
func TestInverseAddDCTRectScaleEdges(t *testing.T) {
	for _, tc := range []struct {
		w, h int
		fn   func(dst []byte, dstStride int, pred []byte, predStride int, coeff []int32, coeffStride int, activeRows, activeCols int) bool
	}{
		{16, 32, InverseAddDCT16x32},
		{32, 16, InverseAddDCT32x16},
	} {
		size := Size{Width: uint8(tc.w), Height: uint8(tc.h)}
		pred := make([]byte, tc.w*tc.h)
		for i := range pred {
			pred[i] = byte(i * 7)
		}
		for _, v := range []int32{0, 1, 2, 100, 32767, 46339, 46340, 46341, 46342, 65535, 1 << 20, 1<<22 - 1, 1 << 22, 1<<22 + 1, 1 << 30, 1<<31 - 1} {
			for _, sign := range []int32{1, -1} {
				for _, pos := range [][2]int{{0, 0}, {1, 0}, {0, 1}, {3, 5}} {
					coeff := make([]int32, tc.w*tc.h)
					coeff[pos[1]*tc.h+pos[0]] = sign * v
					want := append([]byte(nil), pred...)
					itxFusedRectOracle(t, want, tc.w, pred, tc.w, coeff, tc.h, size, pos[0]+1, pos[1]+1)
					got := append([]byte(nil), pred...)
					if !tc.fn(got, tc.w, got, tc.w, coeff, tc.h, pos[0]+1, pos[1]+1) {
						continue // declined by the guard: the staged path serves it
					}
					for i := range got {
						if got[i] != want[i] {
							t.Fatalf("%dx%d coeff %d at row %d col %d: mismatch at %d: fused %d staged %d", tc.w, tc.h, sign*v, pos[0], pos[1], i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

// TestInverseAddDCTRect2HBDMatchesStagedPath: every accepted 10- and 12-bit
// 16x32 and 32x16 DCT_DCT block reproduces the staged transform plus the raw
// add sample for sample, in place and with a separate prediction, without
// writing past the block; a declined block is left untouched. A block with
// a single coefficient whose 1/sqrt(2)-scaled value leaves int16 but stays
// inside the 10- and 12-bit row clamps (bd + 8 bits) is kept by the staged
// path; the fused load saturates it to +-32767, above the L1 guard, so the
// fused path must decline it.
func TestInverseAddDCTRect2HBDMatchesStagedPath(t *testing.T) {
	rng := rand.New(rand.NewSource(1032))
	for _, shape := range [][2]int{{16, 32}, {32, 16}} {
		w, h := shape[0], shape[1]
		fused := InverseAddDCT16x32HBD
		if w == 32 {
			fused = InverseAddDCT32x16HBD
		}
		cs := min(h, 32)
		for _, bd := range []uint8{10, 12} {
			stride := 2 * (w + 8)
			coeff := make([]int32, w*cs)
			raw := make([]int32, w*h)
			pred := make([]byte, h*stride)
			got := make([]byte, h*stride)
			want := make([]byte, h*stride)
			maxV := 1<<bd - 1
			accepted, declined, lowDeclined, low := 0, 0, 0, 0
			for trial := 0; trial < 8000; trial++ {
				clear(coeff)
				activeRows, activeCols := 1+rng.Intn(h), 1+rng.Intn(w)
				switch trial % 4 {
				case 0:
					activeRows, activeCols = 1+rng.Intn(8), 1+rng.Intn(8)
				case 1:
					activeRows, activeCols = 1+rng.Intn(16), 1+rng.Intn(16)
				}
				mode := trial % 7
				mag := []int{64, 1024, 4096, 32767, 0, 300000, 1 << 23}[mode]
				// Mode 4: one coefficient whose scaled value |c|*181/256 lies
				// between int16 and the row clamp (46341 < |c| < 185000 keeps it
				// under 2^17 - 1, the 10-bit clamp).
				between := mode == 4
				if between {
					v := int32(47000 + rng.Intn(133000))
					if rng.Intn(2) == 0 {
						v = -v
					}
					coeff[rng.Intn(activeCols)*cs+rng.Intn(activeRows)] = v
				}
				for c := 0; c < activeCols && !between; c++ {
					for r := 0; r < activeRows; r++ {
						if rng.Intn(3) == 0 {
							continue
						}
						coeff[c*cs+r] = int32(rng.Intn(2*mag+1) - mag)
					}
				}
				for i := 0; i < len(pred); i += 2 {
					v := rng.Intn(maxV + 1)
					if trial%5 == 0 {
						v = maxV * rng.Intn(2)
					}
					binary.LittleEndian.PutUint16(pred[i:], uint16(v))
				}
				if err := InverseBlockBitDepthRawBounded(raw, coeff, cs, Size{Width: uint8(w), Height: uint8(h)}, TypeDCTDCT, bd, activeRows, activeCols); err != nil {
					t.Fatal(err)
				}
				copy(want, pred)
				for r := 0; r < h; r++ {
					for c := 0; c < w; c++ {
						p := int(binary.LittleEndian.Uint16(pred[r*stride+2*c:]))
						v := min(max(p+int((raw[r*w+c]+8)>>4), 0), maxV)
						binary.LittleEndian.PutUint16(want[r*stride+2*c:], uint16(v))
					}
				}
				copy(got, pred)
				ok := fused(got, stride, got, stride, coeff, cs, activeRows, activeCols, bd)
				if mode <= 1 {
					low++
				}
				if between && ok {
					t.Fatalf("%dx%d bd %d trial %d: a coefficient past the int16 range after the rect2 scale was accepted", w, h, bd, trial)
				}
				if !ok {
					declined++
					if mode <= 1 {
						lowDeclined++
					}
					for i := range got {
						if got[i] != pred[i] {
							t.Fatalf("%dx%d bd %d trial %d: declined block modified dst at %d", w, h, bd, trial, i)
						}
					}
					continue
				}
				accepted++
				for i := range got {
					if got[i] != want[i] {
						t.Fatalf("%dx%d bd %d trial %d (mode %d, rows %d cols %d): mismatch at row %d byte %d", w, h, bd, trial, mode, activeRows, activeCols, i/stride, i%stride)
					}
				}
				for i := range got {
					got[i] = 0xa5
				}
				if !fused(got, stride, pred, stride, coeff, cs, activeRows, activeCols, bd) {
					t.Fatalf("%dx%d bd %d trial %d: separate-pred call declined after the in-place call accepted", w, h, bd, trial)
				}
				for r := 0; r < h; r++ {
					for c := 0; c < stride; c++ {
						wv := want[r*stride+c]
						if c >= 2*w {
							wv = 0xa5
						}
						if got[r*stride+c] != wv {
							t.Fatalf("%dx%d bd %d trial %d: separate-pred mismatch at row %d byte %d", w, h, bd, trial, r, c)
						}
					}
				}
			}
			t.Logf("%dx%d bd %d: accepted %d, declined %d (low magnitudes: %d of %d declined)", w, h, bd, accepted, declined, lowDeclined, low)
			if lowDeclined != 0 || accepted == 0 {
				t.Fatalf("%dx%d bd %d: %d low-magnitude declines, %d accepted", w, h, bd, lowDeclined, accepted)
			}
		}
	}
}
