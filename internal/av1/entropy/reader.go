// Ported from libaom:
//   aom_dsp/entdec.c
//   aom_dsp/bitreader.h
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package entropy

import "math/bits"

const (
	ecMinProb   = 4
	ecProbShift = 6
	ecWindow    = 32
	ecLotsBits  = 0x4000
)

// Reader is an allocation-free AV1 entropy range decoder over one tile payload.
type Reader struct {
	src []byte
	pos int

	dif      uint32
	rng      uint32
	cnt      int
	tellOffs int

	allowCDFUpdate bool
}

// Cursor is a stack-local copy of Reader state for tight decode loops. Callers
// must CommitTo the source Reader before continuing with non-cursor reads.
type Cursor struct {
	src []byte
	pos int

	dif      uint32
	rng      uint32
	cnt      int
	tellOffs int

	allowCDFUpdate bool
}

// Cursor snapshots r for a run of trusted hot-path reads.
func (r *Reader) Cursor() Cursor {
	return Cursor{
		src:            r.src,
		pos:            r.pos,
		dif:            r.dif,
		rng:            r.rng,
		cnt:            r.cnt,
		tellOffs:       r.tellOffs,
		allowCDFUpdate: r.allowCDFUpdate,
	}
}

// CommitTo writes cursor state back to r.
func (c *Cursor) CommitTo(r *Reader) {
	r.src = c.src
	r.pos = c.pos
	r.dif = c.dif
	r.rng = c.rng
	r.cnt = c.cnt
	r.tellOffs = c.tellOffs
	r.allowCDFUpdate = c.allowCDFUpdate
}

// NewReader initializes an AV1 entropy reader with CDF adaptation enabled.
func NewReader(src []byte) Reader {
	return NewReaderWithCDFUpdate(src, true)
}

// NewReaderWithCDFUpdate initializes an AV1 entropy reader and controls whether
// ReadSymbol adapts CDF state after each decoded symbol.
func NewReaderWithCDFUpdate(src []byte, allowCDFUpdate bool) Reader {
	r := Reader{
		src:            src,
		dif:            1<<(ecWindow-1) - 1,
		rng:            0x8000,
		cnt:            -15,
		tellOffs:       10 - (ecWindow - 8),
		allowCDFUpdate: allowCDFUpdate,
	}
	r.refill()
	return r
}

// AllowCDFUpdate reports whether ReadSymbol mutates CDF state.
func (r *Reader) AllowCDFUpdate() bool {
	return r != nil && r.allowCDFUpdate
}

// SetCDFUpdate controls whether ReadSymbol mutates CDF state.
func (r *Reader) SetCDFUpdate(allow bool) {
	if r != nil {
		r.allowCDFUpdate = allow
	}
}

// BitsRead returns the AV1 entropy bit position reported by the range decoder.
func (r *Reader) BitsRead() int {
	if r == nil {
		return 0
	}
	return r.pos*8 - r.cnt + r.tellOffs
}

// ReadBit decodes one equiprobable bit.
func (r *Reader) ReadBit() (uint8, error) {
	if r == nil {
		return 0, ErrInvalidProbability
	}
	return r.ReadBitTrusted(), nil
}

// ReadBitTrusted decodes one equiprobable bit for callers that have already
// proven the reader is non-nil. It is the hot-path core of ReadBit.
//
//go:nosplit
func (r *Reader) ReadBitTrusted() uint8 {
	rangeValue := r.rng
	dif := r.dif
	split := (rangeValue >> 8) << 7
	split += ecMinProb
	window := split << (ecWindow - 16)

	bit := uint8(1)
	nextRange := split
	if dif >= window {
		dif -= window
		nextRange = rangeValue - split
		bit = 0
	}

	if traceEntropyReads {
		traceBoolRead(CDFProbTop/2, r.dif, r.rng, r.BitsRead())
	}
	shift := 16 - bits.Len32(nextRange)
	r.cnt -= shift
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = nextRange << uint(shift)
	if r.cnt < 0 {
		r.refill()
	}
	return bit
}

// ReadBool decodes one boolean with an AV1 8-bit probability that the returned
// bit is one.
func (r *Reader) ReadBool(prob uint8) (uint8, error) {
	p := (0x7fffff - (uint32(prob) << 15) + uint32(prob)) >> 8
	return r.ReadBoolQ15(uint16(p))
}

// ReadBoolQ15 decodes one boolean with a Q15 probability that the returned bit
// is one.
func (r *Reader) ReadBoolQ15(prob uint16) (uint8, error) {
	if r == nil || prob == 0 || prob >= CDFProbTop {
		return 0, ErrInvalidProbability
	}

	rangeValue := r.rng
	dif := r.dif
	split := ((rangeValue >> 8) * uint32(prob>>ecProbShift)) >> (7 - ecProbShift)
	split += ecMinProb
	window := split << (ecWindow - 16)

	bit := uint8(1)
	nextRange := split
	if dif >= window {
		dif -= window
		nextRange = rangeValue - split
		bit = 0
	}

	if traceEntropyReads {
		traceBoolRead(prob, r.dif, r.rng, r.BitsRead())
	}
	shift := 16 - bits.Len32(nextRange)
	r.cnt -= shift
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = nextRange << uint(shift)
	if r.cnt < 0 {
		r.refill()
	}
	return bit, nil
}

// ReadBits decodes n equiprobable bits MSB-first. n must be at most 32.
func (r *Reader) ReadBits(n uint8) (uint32, error) {
	if n > 32 {
		return 0, ErrInvalidBitCount
	}
	if n == 0 {
		return 0, nil
	}
	if r == nil {
		return 0, ErrInvalidProbability
	}
	return r.ReadBitsTrusted(n), nil
}

// ReadBitsTrusted decodes n equiprobable bits MSB-first for callers that have
// already proven n <= 32 and the reader is non-nil.
//
//go:nosplit
func (r *Reader) ReadBitsTrusted(n uint8) uint32 {
	if n == 0 {
		return 0
	}
	src := r.src
	pos := r.pos
	dif := r.dif
	rng := r.rng
	cnt := r.cnt
	tellOffs := r.tellOffs
	var v uint32
	for range n {
		rangeValue := rng
		traceDif := dif
		split := (rangeValue >> 8) << 7
		split += ecMinProb
		window := split << (ecWindow - 16)

		bit := uint8(1)
		nextRange := split
		if dif >= window {
			dif -= window
			nextRange = rangeValue - split
			bit = 0
		}

		if traceEntropyReads {
			traceBoolRead(CDFProbTop/2, traceDif, rng, pos*8-cnt+tellOffs)
		}
		shift := 16 - bits.Len32(nextRange)
		cnt -= shift
		dif = ((dif + 1) << uint(shift)) - 1
		rng = nextRange << uint(shift)
		if cnt < 0 {
			refillShift := ecWindow - 9 - (cnt + 15)
			for refillShift >= 0 && pos < len(src) {
				dif ^= uint32(src[pos]) << uint(refillShift)
				cnt += 8
				refillShift -= 8
				pos++
			}
			if pos >= len(src) {
				tellOffs += ecLotsBits - cnt
				cnt = ecLotsBits
			}
		}
		v = (v << 1) | uint32(bit)
	}
	r.pos = pos
	r.dif = dif
	r.rng = rng
	r.cnt = cnt
	r.tellOffs = tellOffs
	return v
}

// ReadBitTrusted decodes one equiprobable bit from cursor state.
//
//go:nosplit
func (c *Cursor) ReadBitTrusted() uint8 {
	rangeValue := c.rng
	dif := c.dif
	split := (rangeValue >> 8) << 7
	split += ecMinProb
	window := split << (ecWindow - 16)

	bit := uint8(1)
	nextRange := split
	if dif >= window {
		dif -= window
		nextRange = rangeValue - split
		bit = 0
	}

	if traceEntropyReads {
		traceBoolRead(CDFProbTop/2, c.dif, c.rng, c.pos*8-c.cnt+c.tellOffs)
	}
	shift := 16 - bits.Len32(nextRange)
	c.cnt -= shift
	c.dif = ((dif + 1) << uint(shift)) - 1
	c.rng = nextRange << uint(shift)
	if c.cnt < 0 {
		c.refill()
	}
	return bit
}

// ReadBitsTrusted decodes n equiprobable bits MSB-first from cursor state.
//
//go:nosplit
func (c *Cursor) ReadBitsTrusted(n uint8) uint32 {
	if n == 0 {
		return 0
	}
	src := c.src
	pos := c.pos
	dif := c.dif
	rng := c.rng
	cnt := c.cnt
	tellOffs := c.tellOffs
	var v uint32
	for range n {
		rangeValue := rng
		traceDif := dif
		split := (rangeValue >> 8) << 7
		split += ecMinProb
		window := split << (ecWindow - 16)

		bit := uint8(1)
		nextRange := split
		if dif >= window {
			dif -= window
			nextRange = rangeValue - split
			bit = 0
		}

		if traceEntropyReads {
			traceBoolRead(CDFProbTop/2, traceDif, rng, pos*8-cnt+tellOffs)
		}
		shift := 16 - bits.Len32(nextRange)
		cnt -= shift
		dif = ((dif + 1) << uint(shift)) - 1
		rng = nextRange << uint(shift)
		if cnt < 0 {
			refillShift := ecWindow - 9 - (cnt + 15)
			for refillShift >= 0 && pos < len(src) {
				dif ^= uint32(src[pos]) << uint(refillShift)
				cnt += 8
				refillShift -= 8
				pos++
			}
			if pos >= len(src) {
				tellOffs += ecLotsBits - cnt
				cnt = ecLotsBits
			}
		}
		v = (v << 1) | uint32(bit)
	}
	c.pos = pos
	c.dif = dif
	c.rng = rng
	c.cnt = cnt
	c.tellOffs = tellOffs
	return v
}

// ReadUniform decodes an AV1 ns(n)-style uniformly distributed value in
// [0, n). It consumes equiprobable range-coded bits.
func (r *Reader) ReadUniform(n uint32) (uint32, error) {
	if n == 0 {
		return 0, ErrInvalidRange
	}
	if n == 1 {
		return 0, nil
	}
	width := uint8(bits.Len32(n))
	m := (uint64(1) << width) - uint64(n)
	v, err := r.ReadBits(width - 1)
	if err != nil {
		return 0, err
	}
	if uint64(v) < m {
		return v, nil
	}
	bit, err := r.ReadBit()
	if err != nil {
		return 0, err
	}
	return uint32((uint64(v) << 1) - m + uint64(bit)), nil
}

// ReadSubexp decodes an AV1 finite subexponential code in [0, n).
func (r *Reader) ReadSubexp(n uint32, k uint8) (uint32, error) {
	if n == 0 || k > 31 {
		return 0, ErrInvalidRange
	}

	i := 0
	mk := uint32(0)
	for {
		b := k
		if i > 0 {
			next := int(k) + i - 1
			if next > 31 {
				return 0, ErrInvalidRange
			}
			b = uint8(next)
		}
		a := uint32(1) << b
		if uint64(n) <= uint64(mk)+3*uint64(a) {
			v, err := r.ReadUniform(n - mk)
			if err != nil {
				return 0, err
			}
			return v + mk, nil
		}

		more, err := r.ReadBit()
		if err != nil {
			return 0, err
		}
		if more == 0 {
			v, err := r.ReadBits(b)
			if err != nil {
				return 0, err
			}
			return v + mk, nil
		}
		i++
		mk += a
	}
}

// ReadRefSubexp decodes a finite subexponential code and recenters it around
// ref, returning a value in [0, n).
func (r *Reader) ReadRefSubexp(n uint32, k uint8, ref uint32) (uint32, error) {
	if n == 0 || ref >= n {
		return 0, ErrInvalidRange
	}
	v, err := r.ReadSubexp(n, k)
	if err != nil {
		return 0, err
	}
	return invRecenterFiniteNonNeg(n, ref, v), nil
}

// ReadSignedRefSubexp decodes a value in [-(n-1), n-1] recentered around ref.
func (r *Reader) ReadSignedRefSubexp(n uint32, k uint8, ref int32) (int32, error) {
	if n == 0 || n > 1<<31 {
		return 0, ErrInvalidRange
	}
	scaledN := (n << 1) - 1
	scaledRef := int64(ref) + int64(n) - 1
	if scaledRef < 0 || scaledRef >= int64(scaledN) {
		return 0, ErrInvalidRange
	}
	v, err := r.ReadRefSubexp(scaledN, k, uint32(scaledRef))
	if err != nil {
		return 0, err
	}
	return int32(int64(v) - int64(n) + 1), nil
}

// ReadSymbol decodes one AV1 inverse-CDF-coded symbol. If CDF updates are
// enabled, cdf is adapted in-place after a successful decode.
func (r *Reader) ReadSymbol(cdf []uint16, symbols int) (int, error) {
	if err := ValidateCDF(cdf, symbols); err != nil {
		return 0, err
	}
	if r == nil {
		return 0, ErrInvalidCDF
	}

	// Reslice to the exact window so the compiler proves every cdf[...] index
	// (decode loop and updateCDF) is in bounds: ValidateCDF guarantees
	// len(cdf) >= symbols+1, the decode loop indexes [0, symbols-1], and
	// updateCDFWindow indexes the trailing count at cdf[symbols].
	cdf = cdf[:symbols+1]

	rangeValue := r.rng
	rngHi := rangeValue >> 8
	coded := r.dif >> (ecWindow - 16)
	upper := rangeValue
	lower := uint32(0)
	symbol := 0
	// Derive last from len(cdf) so the compiler relates the loop bound to the
	// backing array: symbol < last == len(cdf)-2 < len(cdf), proving cdf[symbol]
	// in bounds (no per-iteration bounds check). last >= 1 (ValidateCDF
	// guarantees symbols >= 2), so the loop runs at least once and captures the
	// head probability for tracing without a separate cdf[0] bounds check.
	last := len(cdf) - 2
	head := uint16(0)
	// minTerm tracks ecMinProb*(last-symbol); it decrements by ecMinProb each
	// iteration, replacing the per-step multiply with a single subtraction. The
	// arithmetic is identical to ecMinProb*uint32(last-symbol).
	minTerm := ecMinProb * uint32(last)
	for symbol < last {
		c := cdf[symbol]
		if symbol == 0 {
			head = c
		}
		lower = ((rngHi * uint32(c>>ecProbShift)) >> (7 - ecProbShift)) + minTerm
		if coded >= lower {
			break
		}
		symbol++
		upper = lower
		minTerm -= ecMinProb
	}
	if symbol == last {
		// Final symbol: cdf[last] == 0 so its split is 0 and coded >= 0 holds.
		lower = 0
	}
	if lower >= upper {
		return 0, ErrInvalidCDF
	}

	if traceEntropyReads {
		traceCDFRead(head, symbols, r.dif, r.rng, r.BitsRead())
	}
	// Inline the normalize fast path so the no-refill case keeps cdf/symbol live
	// in registers for updateCDFWindow instead of spilling across a call. The
	// arithmetic matches (*Reader).normalize exactly.
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := 16 - bits.Len32(rng)
	r.cnt -= shift
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = rng << uint(shift)
	if r.cnt < 0 {
		r.refill()
	}
	if r.allowCDFUpdate {
		updateCDFWindow(cdf, symbol)
	}
	return symbol, nil
}

// ReadCDF decodes one symbol from caller-owned CDF state.
//
// *CDF state is valid by construction: its only mutators (Init, InitUniform,
// Update, CopyFrom) validate the inverse-CDF shape before storing, and the
// in-place adaptation in updateCDFWindow preserves it. ReadCDF therefore routes
// to the validation-free trusted core instead of re-running the O(symbols)
// ValidateCDF monotonicity scan on every symbol. ReadSymbolTrusted still applies
// the cheap structural guard (nil reader, symbol-count range, slice length), so
// a zero-valued CDF still reports ErrInvalidCDF. Slice callers that need the
// full per-call monotonicity check continue to use ReadSymbol directly.
func (r *Reader) ReadCDF(cdf *CDF) (int, error) {
	if r == nil || cdf == nil {
		return 0, ErrInvalidCDF
	}
	symbols := int(cdf.symbols)
	if symbols < 2 || symbols > MaxSymbols {
		return 0, ErrInvalidCDF
	}
	switch symbols {
	case 2:
		return r.readBinaryCDFKnown(&cdf.values), nil
	case 3:
		return r.readCDF3Known(&cdf.values), nil
	case 4:
		return r.readCDF4Known(&cdf.values), nil
	default:
		return r.readSymbolTrusted(cdf.values[:symbols+1], symbols)
	}
}

// readSymbolTrusted is the validation-free core of ReadSymbol. It assumes the
// caller has already proven that cdf is a well-formed AV1 inverse CDF of exactly
// symbols entries (monotone non-increasing, terminated by cdf[symbols-1] == 0,
// with the adaptation count at cdf[symbols] <= MaxCDFCount) and that r != nil.
// These invariants hold for the coefficient CDFs, which are seeded from valid
// libaom defaults and only ever mutated by updateCDFWindow, which preserves
// them. Skipping the per-symbol monotonicity scan in ValidateCDF is the only
// behavioral difference; the decode math is byte-for-byte identical to
// ReadSymbol.
func (r *Reader) readSymbolTrusted(cdf []uint16, symbols int) (int, error) {
	// Reslice to the exact window so the compiler proves every cdf[...] index
	// is in bounds, exactly as ReadSymbol does.
	cdf = cdf[:symbols+1]

	rangeValue := r.rng
	rngHi := rangeValue >> 8
	coded := r.dif >> (ecWindow - 16)
	upper := rangeValue
	lower := uint32(0)
	symbol := 0
	last := len(cdf) - 2
	head := uint16(0)
	// minTerm tracks ecMinProb*(last-symbol); it decrements by ecMinProb each
	// iteration, replacing the per-step multiply with a single subtraction. The
	// arithmetic is identical to ecMinProb*uint32(last-symbol).
	minTerm := ecMinProb * uint32(last)
	for symbol < last {
		c := cdf[symbol]
		if symbol == 0 {
			head = c
		}
		lower = ((rngHi * uint32(c>>ecProbShift)) >> (7 - ecProbShift)) + minTerm
		if coded >= lower {
			break
		}
		symbol++
		upper = lower
		minTerm -= ecMinProb
	}
	if symbol == last {
		lower = 0
	}
	if lower >= upper {
		return 0, ErrInvalidCDF
	}

	if traceEntropyReads {
		traceCDFRead(head, symbols, r.dif, r.rng, r.BitsRead())
	}
	// Inline the normalize fast path so the no-refill case keeps cdf/symbol live
	// in registers for updateCDFWindow instead of spilling across a call. The
	// arithmetic matches (*Reader).normalize exactly.
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := 16 - bits.Len32(rng)
	r.cnt -= shift
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = rng << uint(shift)
	if r.cnt < 0 {
		r.refill()
	}
	if r.allowCDFUpdate {
		updateCDFWindow(cdf, symbol)
	}
	return symbol, nil
}

// ReadSymbolTrusted decodes one AV1 inverse-CDF-coded symbol while skipping the
// O(symbols) ValidateCDF monotonicity scan that ReadSymbol performs on every
// call. It is intended only for the proven-hot coefficient reads, whose CDFs are
// valid by construction (default-seeded and adapted in place). It still performs
// the cheap structural guard (nil reader, symbol range) so a corrupt symbols
// argument cannot read out of bounds. Use ReadSymbol everywhere validation is
// required.
func (r *Reader) ReadSymbolTrusted(cdf []uint16, symbols int) (int, error) {
	if symbols < 2 || symbols > MaxSymbols || len(cdf) < symbols+1 {
		return 0, ErrInvalidCDF
	}
	if r == nil {
		return 0, ErrInvalidCDF
	}
	return r.readSymbolTrusted(cdf, symbols)
}

// ReadCDFTrusted decodes one symbol from caller-owned CDF state without the
// per-call monotonicity validation. The CDF must be valid by construction; see
// ReadSymbolTrusted.
func (r *Reader) ReadCDFTrusted(cdf *CDF) (int, error) {
	if r == nil || cdf == nil {
		return 0, ErrInvalidCDF
	}
	symbols := int(cdf.symbols)
	if symbols < 2 || symbols > MaxSymbols {
		return 0, ErrInvalidCDF
	}
	switch symbols {
	case 2:
		return r.readBinaryCDFKnown(&cdf.values), nil
	case 3:
		return r.readCDF3Known(&cdf.values), nil
	case 4:
		return r.readCDF4Known(&cdf.values), nil
	default:
		return r.readSymbolTrusted(cdf.values[:symbols+1], symbols)
	}
}

// ReadBinaryCDFUnchecked decodes one symbol from a two-symbol CDF. The caller
// must prove r and cdf are non-nil and that cdf has exactly two symbols.
func (r *Reader) ReadBinaryCDFUnchecked(cdf *CDF) int {
	return r.readBinaryCDFKnown(&cdf.values)
}

// ReadCDF3Trusted decodes one symbol from a three-symbol CDF without the
// generic symbol-search loop. The CDF must be valid by construction.
func (r *Reader) ReadCDF3Trusted(cdf *CDF) (int, error) {
	if r == nil || cdf == nil || cdf.symbols != 3 {
		return 0, ErrInvalidCDF
	}
	return r.readCDF3Known(&cdf.values), nil
}

// ReadCDF3Unchecked decodes one symbol from a three-symbol CDF. The caller must
// prove r and cdf are non-nil and that cdf has exactly three symbols.
func (r *Reader) ReadCDF3Unchecked(cdf *CDF) int {
	return r.readCDF3Known(&cdf.values)
}

// ReadCDF3Unchecked decodes one symbol from a three-symbol CDF using cursor
// state. The caller must prove c and cdf are non-nil and cdf has exactly three
// symbols.
func (c *Cursor) ReadCDF3Unchecked(cdf *CDF) int {
	return c.readCDF3Known(&cdf.values)
}

//go:nosplit
func (r *Reader) readCDF3Known(values *[MaxSymbols + 1]uint16) int {
	rangeValue := r.rng
	rngHi := rangeValue >> 8
	coded := r.dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := values[0]
	lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		c1 := values[1]
		lower = ((rngHi * uint32(c1>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			lower = 0
		}
	}
	if traceEntropyReads {
		traceCDFRead(c0, 3, r.dif, r.rng, r.BitsRead())
	}
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := 16 - bits.Len32(rng)
	r.cnt -= shift
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = rng << uint(shift)
	if r.cnt < 0 {
		r.refill()
	}
	if r.allowCDFUpdate {
		updateCDF3(values, symbol)
	}
	return symbol
}

//go:nosplit
func (c *Cursor) readCDF3Known(values *[MaxSymbols + 1]uint16) int {
	rangeValue := c.rng
	rngHi := rangeValue >> 8
	coded := c.dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := values[0]
	lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		c1 := values[1]
		lower = ((rngHi * uint32(c1>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			lower = 0
		}
	}
	if traceEntropyReads {
		traceCDFRead(c0, 3, c.dif, c.rng, c.pos*8-c.cnt+c.tellOffs)
	}
	dif := c.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := 16 - bits.Len32(rng)
	c.cnt -= shift
	c.dif = ((dif + 1) << uint(shift)) - 1
	c.rng = rng << uint(shift)
	if c.cnt < 0 {
		c.refill()
	}
	if c.allowCDFUpdate {
		updateCDF3(values, symbol)
	}
	return symbol
}

// ReadCDF4Trusted decodes one symbol from a four-symbol CDF without the
// generic symbol-search loop. The CDF must be valid by construction.
func (r *Reader) ReadCDF4Trusted(cdf *CDF) (int, error) {
	if r == nil || cdf == nil || cdf.symbols != 4 {
		return 0, ErrInvalidCDF
	}
	return r.readCDF4Known(&cdf.values), nil
}

// ReadCDF4Unchecked decodes one symbol from a four-symbol CDF. The caller must
// prove r and cdf are non-nil and that cdf has exactly four symbols.
func (r *Reader) ReadCDF4Unchecked(cdf *CDF) int {
	return r.readCDF4Known(&cdf.values)
}

// ReadCDF4Unchecked decodes one symbol from a four-symbol CDF using cursor
// state. The caller must prove c and cdf are non-nil and cdf has exactly four
// symbols.
func (c *Cursor) ReadCDF4Unchecked(cdf *CDF) int {
	return c.readCDF4Known(&cdf.values)
}

// ReadCDF4HighTokenUnchecked decodes the AV1 high-token base-range chain used
// by coefficient BR syntax. It is equivalent to reading the same four-symbol
// CDF up to four times, summing symbols, and stopping early when a symbol below
// 3 is observed. The caller must prove c and cdf are non-nil and cdf has
// exactly four symbols.
func (c *Cursor) ReadCDF4HighTokenUnchecked(cdf *CDF) int {
	return c.readCDF4HighTokenKnown(&cdf.values)
}

//go:nosplit
func (r *Reader) readCDF4Known(values *[MaxSymbols + 1]uint16) int {
	rangeValue := r.rng
	rngHi := rangeValue >> 8
	coded := r.dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := values[0]
	lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		c1 := values[1]
		lower = ((rngHi * uint32(c1>>ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			c2 := values[2]
			lower = ((rngHi * uint32(c2>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
			if coded < lower {
				symbol = 3
				upper = lower
				lower = 0
			}
		}
	}
	if traceEntropyReads {
		traceCDFRead(c0, 4, r.dif, r.rng, r.BitsRead())
	}
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := 16 - bits.Len32(rng)
	r.cnt -= shift
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = rng << uint(shift)
	if r.cnt < 0 {
		r.refill()
	}
	if r.allowCDFUpdate {
		count := values[4]
		rate := uint(5 + (count >> 4))
		c0 := uint32(values[0])
		c1 := uint32(values[1])
		c2 := uint32(values[2])
		switch symbol {
		case 0:
			values[0] = uint16(c0 - (c0 >> rate))
			values[1] = uint16(c1 - (c1 >> rate))
			values[2] = uint16(c2 - (c2 >> rate))
		case 1:
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
			values[1] = uint16(c1 - (c1 >> rate))
			values[2] = uint16(c2 - (c2 >> rate))
		case 2:
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
			values[2] = uint16(c2 - (c2 >> rate))
		default:
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
			values[2] = uint16(c2 + ((CDFProbTop - c2) >> rate))
		}
		if count < MaxCDFCount {
			values[4] = count + 1
		}
	}
	return symbol
}

//go:nosplit
func (c *Cursor) readCDF4Known(values *[MaxSymbols + 1]uint16) int {
	src := c.src
	pos := c.pos
	dif := c.dif
	rng := c.rng
	cnt := c.cnt
	tellOffs := c.tellOffs

	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := values[0]
	lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		c1 := values[1]
		lower = ((rngHi * uint32(c1>>ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			c2 := values[2]
			lower = ((rngHi * uint32(c2>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
			if coded < lower {
				symbol = 3
				upper = lower
				lower = 0
			}
		}
	}
	if traceEntropyReads {
		traceCDFRead(c0, 4, dif, rng, pos*8-cnt+tellOffs)
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := 16 - bits.Len32(rng)
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		refillShift := ecWindow - 9 - (cnt + 15)
		for refillShift >= 0 && pos < len(src) {
			dif ^= uint32(src[pos]) << uint(refillShift)
			cnt += 8
			refillShift -= 8
			pos++
		}
		if pos >= len(src) {
			tellOffs += ecLotsBits - cnt
			cnt = ecLotsBits
		}
	}
	if c.allowCDFUpdate {
		count := values[4]
		rate := uint(5 + (count >> 4))
		c0 := uint32(values[0])
		c1 := uint32(values[1])
		c2 := uint32(values[2])
		switch symbol {
		case 0:
			values[0] = uint16(c0 - (c0 >> rate))
			values[1] = uint16(c1 - (c1 >> rate))
			values[2] = uint16(c2 - (c2 >> rate))
		case 1:
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
			values[1] = uint16(c1 - (c1 >> rate))
			values[2] = uint16(c2 - (c2 >> rate))
		case 2:
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
			values[2] = uint16(c2 - (c2 >> rate))
		default:
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
			values[2] = uint16(c2 + ((CDFProbTop - c2) >> rate))
		}
		if count < MaxCDFCount {
			values[4] = count + 1
		}
	}
	c.pos = pos
	c.dif = dif
	c.rng = rng
	c.cnt = cnt
	c.tellOffs = tellOffs
	return symbol
}

//go:nosplit
func (c *Cursor) readCDF4HighTokenKnown(values *[MaxSymbols + 1]uint16) int {
	src := c.src
	pos := c.pos
	dif := c.dif
	rng := c.rng
	cnt := c.cnt
	tellOffs := c.tellOffs
	allowCDFUpdate := c.allowCDFUpdate

	level := 0
	for i := 0; i < 4; i++ {
		rangeValue := rng
		rngHi := rangeValue >> 8
		coded := dif >> (ecWindow - 16)
		upper := rangeValue
		c0 := values[0]
		lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
		symbol := 0
		if coded < lower {
			symbol = 1
			upper = lower
			c1 := values[1]
			lower = ((rngHi * uint32(c1>>ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
			if coded < lower {
				symbol = 2
				upper = lower
				c2 := values[2]
				lower = ((rngHi * uint32(c2>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
				if coded < lower {
					symbol = 3
					upper = lower
					lower = 0
				}
			}
		}
		if traceEntropyReads {
			traceCDFRead(c0, 4, dif, rng, pos*8-cnt+tellOffs)
		}
		dif -= lower << (ecWindow - 16)
		rng = upper - lower
		shift := 16 - bits.Len32(rng)
		cnt -= shift
		dif = ((dif + 1) << uint(shift)) - 1
		rng <<= uint(shift)
		if cnt < 0 {
			refillShift := ecWindow - 9 - (cnt + 15)
			for refillShift >= 0 && pos < len(src) {
				dif ^= uint32(src[pos]) << uint(refillShift)
				cnt += 8
				refillShift -= 8
				pos++
			}
			if pos >= len(src) {
				tellOffs += ecLotsBits - cnt
				cnt = ecLotsBits
			}
		}
		if allowCDFUpdate {
			count := values[4]
			rate := uint(5 + (count >> 4))
			c0 := uint32(values[0])
			c1 := uint32(values[1])
			c2 := uint32(values[2])
			switch symbol {
			case 0:
				values[0] = uint16(c0 - (c0 >> rate))
				values[1] = uint16(c1 - (c1 >> rate))
				values[2] = uint16(c2 - (c2 >> rate))
			case 1:
				values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
				values[1] = uint16(c1 - (c1 >> rate))
				values[2] = uint16(c2 - (c2 >> rate))
			case 2:
				values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
				values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
				values[2] = uint16(c2 - (c2 >> rate))
			default:
				values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
				values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
				values[2] = uint16(c2 + ((CDFProbTop - c2) >> rate))
			}
			if count < MaxCDFCount {
				values[4] = count + 1
			}
		}
		level += symbol
		if symbol < 3 {
			break
		}
	}

	c.pos = pos
	c.dif = dif
	c.rng = rng
	c.cnt = cnt
	c.tellOffs = tellOffs
	return level
}

// ReadBinaryCDFTrusted decodes one symbol from a two-symbol CDF without taking
// the generic symbol-search path. The CDF must be valid by construction.
func (r *Reader) ReadBinaryCDFTrusted(cdf *CDF) (int, error) {
	if r == nil || cdf == nil || cdf.symbols != 2 {
		return 0, ErrInvalidCDF
	}
	return r.readBinaryCDFKnown(&cdf.values), nil
}

//go:nosplit
func (r *Reader) readBinaryCDFKnown(values *[MaxSymbols + 1]uint16) int {
	rangeValue := r.rng
	rngHi := rangeValue >> 8
	coded := r.dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := values[0]
	lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		lower = 0
	}
	if traceEntropyReads {
		traceCDFRead(c0, 2, r.dif, r.rng, r.BitsRead())
	}
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := 16 - bits.Len32(rng)
	r.cnt -= shift
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = rng << uint(shift)
	if r.cnt < 0 {
		r.refill()
	}
	if r.allowCDFUpdate {
		updateCDF2(values, symbol)
	}
	return symbol
}

// ReadBinaryCDFUnchecked decodes one symbol from a two-symbol CDF using cursor
// state. The caller must prove c and cdf are non-nil and cdf has exactly two
// symbols.
func (c *Cursor) ReadBinaryCDFUnchecked(cdf *CDF) int {
	return c.readBinaryCDFKnown(&cdf.values)
}

//go:nosplit
func (c *Cursor) readBinaryCDFKnown(values *[MaxSymbols + 1]uint16) int {
	rangeValue := c.rng
	rngHi := rangeValue >> 8
	coded := c.dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := values[0]
	lower := ((rngHi * uint32(c0>>ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		lower = 0
	}
	if traceEntropyReads {
		traceCDFRead(c0, 2, c.dif, c.rng, c.pos*8-c.cnt+c.tellOffs)
	}
	dif := c.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := 16 - bits.Len32(rng)
	c.cnt -= shift
	c.dif = ((dif + 1) << uint(shift)) - 1
	c.rng = rng << uint(shift)
	if c.cnt < 0 {
		c.refill()
	}
	if c.allowCDFUpdate {
		updateCDF2(values, symbol)
	}
	return symbol
}

func updateCDF2(cdf *[MaxSymbols + 1]uint16, symbol int) {
	count := cdf[2]
	rate := uint(4 + (count >> 4))
	c0 := uint32(cdf[0])
	if symbol > 0 {
		cdf[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
	} else {
		cdf[0] = uint16(c0 - (c0 >> rate))
	}
	if count < MaxCDFCount {
		cdf[2] = count + 1
	}
}

func updateCDF3(cdf *[MaxSymbols + 1]uint16, symbol int) {
	count := cdf[3]
	rate := uint(4 + (count >> 4))
	c0 := uint32(cdf[0])
	c1 := uint32(cdf[1])
	if symbol > 0 {
		cdf[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
	} else {
		cdf[0] = uint16(c0 - (c0 >> rate))
	}
	if symbol > 1 {
		cdf[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
	} else {
		cdf[1] = uint16(c1 - (c1 >> rate))
	}
	if count < MaxCDFCount {
		cdf[3] = count + 1
	}
}

func updateCDF4(cdf *[MaxSymbols + 1]uint16, symbol int) {
	count := cdf[4]
	rate := uint(5 + (count >> 4))
	c0 := uint32(cdf[0])
	c1 := uint32(cdf[1])
	c2 := uint32(cdf[2])
	if symbol > 0 {
		cdf[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
	} else {
		cdf[0] = uint16(c0 - (c0 >> rate))
	}
	if symbol > 1 {
		cdf[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
	} else {
		cdf[1] = uint16(c1 - (c1 >> rate))
	}
	if symbol > 2 {
		cdf[2] = uint16(c2 + ((CDFProbTop - c2) >> rate))
	} else {
		cdf[2] = uint16(c2 - (c2 >> rate))
	}
	if count < MaxCDFCount {
		cdf[4] = count + 1
	}
}

// ReadSignedDelta decodes the AV1 CDF-coded signed delta core used by
// delta_qindex and delta_lflevel tile syntax.
func (r *Reader) ReadSignedDelta(cdf *CDF, small int) (int, error) {
	if small <= 0 || small >= MaxSymbols {
		return 0, ErrInvalidRange
	}
	if cdf == nil || cdf.Symbols() != small+1 {
		return 0, ErrInvalidCDF
	}

	abs, err := r.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	if abs >= small {
		remBits, err := r.ReadBits(3)
		if err != nil {
			return 0, err
		}
		remBits++
		tail, err := r.ReadBits(uint8(remBits))
		if err != nil {
			return 0, err
		}
		abs = int(tail + (uint32(1) << remBits) + 1)
	}
	if abs == 0 {
		return 0, nil
	}
	sign, err := r.ReadBit()
	if err != nil {
		return 0, err
	}
	if sign != 0 {
		return -abs, nil
	}
	return abs, nil
}

func invRecenterFiniteNonNeg(n uint32, ref uint32, v uint32) uint32 {
	if uint64(ref)*2 <= uint64(n) {
		return invRecenterNonNeg(ref, v)
	}
	return n - 1 - invRecenterNonNeg(n-1-ref, v)
}

func invRecenterNonNeg(ref uint32, v uint32) uint32 {
	if uint64(v) > uint64(ref)*2 {
		return v
	}
	if v&1 == 0 {
		return (v >> 1) + ref
	}
	return ref - ((v + 1) >> 1)
}

func (r *Reader) refill() {
	shift := ecWindow - 9 - (r.cnt + 15)
	for shift >= 0 && r.pos < len(r.src) {
		r.dif ^= uint32(r.src[r.pos]) << uint(shift)
		r.cnt += 8
		shift -= 8
		r.pos++
	}
	if r.pos >= len(r.src) {
		r.tellOffs += ecLotsBits - r.cnt
		r.cnt = ecLotsBits
	}
}

func (c *Cursor) refill() {
	shift := ecWindow - 9 - (c.cnt + 15)
	for shift >= 0 && c.pos < len(c.src) {
		c.dif ^= uint32(c.src[c.pos]) << uint(shift)
		c.cnt += 8
		shift -= 8
		c.pos++
	}
	if c.pos >= len(c.src) {
		c.tellOffs += ecLotsBits - c.cnt
		c.cnt = ecLotsBits
	}
}
