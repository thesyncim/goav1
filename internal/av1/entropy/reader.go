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
	return r.ReadBoolQ15(CDFProbTop / 2)
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

	traceBoolRead(prob, r.dif, r.rng, r.BitsRead())
	r.normalize(dif, nextRange)
	return bit, nil
}

// ReadBits decodes n equiprobable bits MSB-first. n must be at most 32.
func (r *Reader) ReadBits(n uint8) (uint32, error) {
	if n > 32 {
		return 0, ErrInvalidBitCount
	}
	var v uint32
	for range n {
		bit, err := r.ReadBit()
		if err != nil {
			return 0, err
		}
		v = (v << 1) | uint32(bit)
	}
	return v, nil
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

	traceCDFRead(head, symbols, r.dif, r.rng, r.BitsRead())
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
	if cdf == nil {
		return 0, ErrInvalidCDF
	}
	return r.ReadSymbolTrusted(cdf.Values(), cdf.Symbols())
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

	traceCDFRead(head, symbols, r.dif, r.rng, r.BitsRead())
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
	if cdf == nil {
		return 0, ErrInvalidCDF
	}
	return r.ReadSymbolTrusted(cdf.Values(), cdf.Symbols())
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

func (r *Reader) normalize(dif uint32, rng uint32) {
	shift := 16 - bits.Len32(rng)
	r.cnt -= shift
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = rng << uint(shift)
	if r.cnt < 0 {
		r.refill()
	}
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
