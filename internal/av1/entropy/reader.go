// Ported from libaom:
//   aom_dsp/entdec.c
//   aom_dsp/bitreader.h
//
// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package entropy

import (
	"math/bits"
	"unsafe"
)

const (
	ecMinProb   = 4
	ecProbShift = 6
	ecWindow    = 32
	ecLotsBits  = 0x4000

	cdf4HighTokenMax = 12
)

// byteAtUnchecked loads src[pos] after the caller has proven pos < len(src).
// Refill loops already carry that branch; this keeps the hot byte load free of
// a second slice-bounds edge.
func byteAtUnchecked(src []byte, pos int) byte {
	return *(*byte)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(src)), pos))
}

func readerTell(pos int, cnt int32, tellOffs int32) int {
	return pos*8 - int(cnt) + int(tellOffs)
}

// refillState extends dif with the AV1 range decoder's next bytes. The common
// range-decoder case needs two or three bytes, so keep that path branch-collapsed;
// rare short-tail cases fall back to the byte loop.
//
//go:nosplit
func refillState(src []byte, pos int, dif uint32, cnt int32, tellOffs int32) (int, uint32, int32, int32) {
	shift := 8 - cnt
	remaining := len(src) - pos
	if shift >= 8 && shift < 16 {
		if remaining >= 2 {
			dif ^= uint32(byteAtUnchecked(src, pos)) << uint(shift)
			dif ^= uint32(byteAtUnchecked(src, pos+1)) << uint(shift-8)
			pos += 2
			cnt += 16
			if pos >= len(src) {
				tellOffs += ecLotsBits - cnt
				cnt = ecLotsBits
			}
			return pos, dif, cnt, tellOffs
		}
	} else if shift < 24 && remaining >= 3 {
		dif ^= uint32(byteAtUnchecked(src, pos)) << uint(shift)
		dif ^= uint32(byteAtUnchecked(src, pos+1)) << uint(shift-8)
		dif ^= uint32(byteAtUnchecked(src, pos+2)) << uint(shift-16)
		pos += 3
		cnt += 24
		if pos >= len(src) {
			tellOffs += ecLotsBits - cnt
			cnt = ecLotsBits
		}
		return pos, dif, cnt, tellOffs
	}
	return refillStateSlow(src, pos, dif, cnt, tellOffs)
}

//go:noinline
//go:nosplit
func refillStateSlow(src []byte, pos int, dif uint32, cnt int32, tellOffs int32) (int, uint32, int32, int32) {
	shift := ecWindow - 9 - (cnt + 15)
	for shift >= 0 && pos < len(src) {
		dif ^= uint32(byteAtUnchecked(src, pos)) << uint(shift)
		cnt += 8
		shift -= 8
		pos++
	}
	if pos >= len(src) {
		tellOffs += ecLotsBits - cnt
		cnt = ecLotsBits
	}
	return pos, dif, cnt, tellOffs
}

// Reader is an allocation-free AV1 entropy range decoder over one tile payload.
type Reader struct {
	src []byte
	pos uint32

	dif      uint32
	rng      uint16
	cnt      int16
	tellOffs int16

	allowCDFUpdate bool
}

// Cursor is a stack-local copy of Reader state for tight decode loops. Callers
// must CommitTo the source Reader before continuing with non-cursor reads, or
// CommitStateTo when the cursor was created from that same reader and immutable
// reader configuration cannot change during the cursor scope.
type Cursor struct {
	src []byte
	pos uint32

	dif      uint32
	rng      uint16
	cnt      int16
	tellOffs int16

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

// CommitStateTo writes cursor range-decoder state back to r without touching
// immutable reader configuration. Use it only for cursors created from the same
// reader when src and CDF-update mode cannot change during the cursor scope.
func (c *Cursor) CommitStateTo(r *Reader) {
	r.pos = c.pos
	r.dif = c.dif
	r.rng = c.rng
	r.cnt = c.cnt
	r.tellOffs = c.tellOffs
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
	return readerTell(int(r.pos), int32(r.cnt), int32(r.tellOffs))
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
	rangeValue := uint32(r.rng)
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
		traceBoolRead(CDFProbTop/2, r.dif, uint32(r.rng), r.BitsRead())
	}
	shift := int32(16 - bits.Len32(nextRange))
	cnt := int32(r.cnt) - shift
	r.cnt = int16(cnt)
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = uint16(nextRange << uint(shift))
	if cnt < 0 {
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

	rangeValue := uint32(r.rng)
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
		traceBoolRead(prob, r.dif, uint32(r.rng), r.BitsRead())
	}
	shift := int32(16 - bits.Len32(nextRange))
	cnt := int32(r.cnt) - shift
	r.cnt = int16(cnt)
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = uint16(nextRange << uint(shift))
	if cnt < 0 {
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
	pos := int(r.pos)
	dif := r.dif
	rng := uint32(r.rng)
	cnt := int32(r.cnt)
	tellOffs := int32(r.tellOffs)
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
			traceBoolRead(CDFProbTop/2, traceDif, rng, readerTell(pos, cnt, tellOffs))
		}
		shift := int32(16 - bits.Len32(nextRange))
		cnt -= shift
		dif = ((dif + 1) << uint(shift)) - 1
		rng = nextRange << uint(shift)
		if cnt < 0 {
			pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
		}
		v = (v << 1) | uint32(bit)
	}
	r.pos = uint32(pos)
	r.dif = dif
	r.rng = uint16(rng)
	r.cnt = int16(cnt)
	r.tellOffs = int16(tellOffs)
	return v
}

// ReadBitTrusted decodes one equiprobable bit from cursor state.
//
//go:nosplit
func (c *Cursor) ReadBitTrusted() uint8 {
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rangeValue := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)
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
		traceBoolRead(CDFProbTop/2, c.dif, rangeValue, readerTell(pos, cnt, tellOffs))
	}
	shift := int32(16 - bits.Len32(nextRange))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rangeValue = nextRange << uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	c.pos = uint32(pos)
	c.cnt = int16(cnt)
	c.dif = dif
	c.rng = uint16(rangeValue)
	c.tellOffs = int16(tellOffs)
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
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)
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
			traceBoolRead(CDFProbTop/2, traceDif, rng, readerTell(pos, cnt, tellOffs))
		}
		shift := int32(16 - bits.Len32(nextRange))
		cnt -= shift
		dif = ((dif + 1) << uint(shift)) - 1
		rng = nextRange << uint(shift)
		if cnt < 0 {
			pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
		}
		v = (v << 1) | uint32(bit)
	}
	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
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

	rangeValue := uint32(r.rng)
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
		traceCDFRead(head, symbols, r.dif, uint32(r.rng), r.BitsRead())
	}
	// Inline the normalize fast path so the no-refill case keeps cdf/symbol live
	// in registers for updateCDFWindow instead of spilling across a call. The
	// arithmetic matches (*Reader).normalize exactly.
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt := int32(r.cnt) - shift
	r.cnt = int16(cnt)
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = uint16(rng << uint(shift))
	if cnt < 0 {
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
		return r.readSymbolKnown(&cdf.values, symbols)
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

	rangeValue := uint32(r.rng)
	rngHi := rangeValue >> 8
	coded := r.dif >> (ecWindow - 16)
	upper := rangeValue
	lower := uint32(0)
	symbol := 0
	last := len(cdf) - 2
	head := cdf[0]
	// minTerm tracks ecMinProb*(last-symbol); it decrements by ecMinProb each
	// iteration, replacing the per-step multiply with a single subtraction. The
	// arithmetic is identical to ecMinProb*uint32(last-symbol).
	minTerm := ecMinProb * uint32(last)
	for symbol < last {
		c := cdf[symbol]
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
		traceCDFRead(head, symbols, r.dif, uint32(r.rng), r.BitsRead())
	}
	// Inline the normalize fast path so the no-refill case keeps cdf/symbol live
	// in registers for updateCDFWindow instead of spilling across a call. The
	// arithmetic matches (*Reader).normalize exactly.
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt := int32(r.cnt) - shift
	r.cnt = int16(cnt)
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = uint16(rng << uint(shift))
	if cnt < 0 {
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
		return r.readSymbolKnown(&cdf.values, symbols)
	}
}

// ReadCDFUnchecked decodes one symbol from cursor state. The caller must prove
// c and cdf are non-nil and cdf is valid by construction.
func (c *Cursor) ReadCDFUnchecked(cdf *CDF) int {
	symbols := int(cdf.symbols)
	switch symbols {
	case 2:
		return c.readBinaryCDFKnown(&cdf.values)
	case 3:
		return c.readCDF3Known(&cdf.values)
	case 4:
		return c.ReadCDF4Unchecked(cdf)
	case 5:
		return c.readCDF5Known(&cdf.values)
	case 8:
		return c.readCDF8Known(&cdf.values)
	default:
		return c.readSymbolKnown(&cdf.values, symbols)
	}
}

// ReadCDFSymbolsUnchecked decodes one symbol from cursor state with the symbol
// count supplied by a caller that already proved the CDF shape.
func (c *Cursor) ReadCDFSymbolsUnchecked(cdf *CDF, symbols int) int {
	switch symbols {
	case 2:
		return c.readBinaryCDFKnown(&cdf.values)
	case 3:
		return c.readCDF3Known(&cdf.values)
	case 4:
		return c.readCDF4Known(&cdf.values)
	case 5:
		return c.readCDF5Known(&cdf.values)
	case 8:
		return c.readCDF8Known(&cdf.values)
	default:
		return c.readSymbolKnown(&cdf.values, symbols)
	}
}

// readSymbolKnown is the fixed-array core for trusted CDF objects with more
// than four symbols. It matches the cursor helper's raw CDF-row shape while
// preserving Reader's invalid-range guard for public ReadCDF callers.
func (r *Reader) readSymbolKnown(values *[MaxSymbols + 1]uint16, symbols int) (int, error) {
	rangeValue := uint32(r.rng)
	rngHi := rangeValue >> 8
	coded := r.dif >> (ecWindow - 16)
	upper := rangeValue
	lower := uint32(0)
	symbol := 0
	last := symbols - 1
	head := values[0]
	minTerm := ecMinProb * uint32(last)
	for symbol < last {
		v := values[symbol]
		lower = ((rngHi * uint32(v>>ecProbShift)) >> (7 - ecProbShift)) + minTerm
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
		traceCDFRead(head, symbols, r.dif, uint32(r.rng), r.BitsRead())
	}
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt := int32(r.cnt) - shift
	r.cnt = int16(cnt)
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = uint16(rng << uint(shift))
	if cnt < 0 {
		r.refill()
	}
	if r.allowCDFUpdate {
		count := values[symbols]
		rate := uint(5 + (count >> 4))
		for i := 0; i < symbol; i++ {
			v := uint32(values[i])
			values[i] = uint16(v + ((CDFProbTop - v) >> rate))
		}
		for i := symbol; i < last; i++ {
			v := uint32(values[i])
			values[i] = uint16(v - (v >> rate))
		}
		if count < MaxCDFCount {
			values[symbols] = count + 1
		}
	}
	return symbol, nil
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
func (c *Cursor) readSymbolKnown(values *[MaxSymbols + 1]uint16, symbols int) int {
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)

	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
	upper := rangeValue
	lower := uint32(0)
	symbol := 0
	last := symbols - 1
	head := values[0]
	minTerm := ecMinProb * uint32(last)
	for symbol < last {
		v := values[symbol]
		lower = ((rngHi * uint32(v>>ecProbShift)) >> (7 - ecProbShift)) + minTerm
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

	if traceEntropyReads {
		traceCDFRead(head, symbols, dif, rng, readerTell(pos, cnt, tellOffs))
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	if c.allowCDFUpdate {
		count := values[symbols]
		rate := uint(4 + (count >> 4))
		if symbols > 3 {
			rate++
		}
		for i := 0; i < symbol; i++ {
			v := uint32(values[i])
			values[i] = uint16(v + ((CDFProbTop - v) >> rate))
		}
		for i := symbol; i < last; i++ {
			v := uint32(values[i])
			values[i] = uint16(v - (v >> rate))
		}
		if count < MaxCDFCount {
			values[symbols] = count + 1
		}
	}
	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
	return symbol
}

//go:nosplit
func (c *Cursor) readCDF5Known(values *[MaxSymbols + 1]uint16) int {
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)

	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := uint32(values[0])
	c1 := uint32(values[1])
	c2 := uint32(values[2])
	c3 := uint32(values[3])
	lower := ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 4*ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
			if coded < lower {
				symbol = 3
				upper = lower
				lower = ((rngHi * (c3 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
				if coded < lower {
					symbol = 4
					upper = lower
					lower = 0
				}
			}
		}
	}

	if traceEntropyReads {
		traceCDFRead(uint16(c0), 5, dif, rng, readerTell(pos, cnt, tellOffs))
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	if c.allowCDFUpdate {
		count := values[5]
		rate := uint(5 + (count >> 4))
		if symbol > 0 {
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
		} else {
			values[0] = uint16(c0 - (c0 >> rate))
		}
		if symbol > 1 {
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
		} else {
			values[1] = uint16(c1 - (c1 >> rate))
		}
		if symbol > 2 {
			values[2] = uint16(c2 + ((CDFProbTop - c2) >> rate))
		} else {
			values[2] = uint16(c2 - (c2 >> rate))
		}
		if symbol > 3 {
			values[3] = uint16(c3 + ((CDFProbTop - c3) >> rate))
		} else {
			values[3] = uint16(c3 - (c3 >> rate))
		}
		if count < MaxCDFCount {
			values[5] = count + 1
		}
	}
	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
	return symbol
}

//go:nosplit
func (c *Cursor) readCDF8Known(values *[MaxSymbols + 1]uint16) int {
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)

	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := uint32(values[0])
	c1 := uint32(values[1])
	c2 := uint32(values[2])
	c3 := uint32(values[3])
	c4 := uint32(values[4])
	c5 := uint32(values[5])
	c6 := uint32(values[6])
	lower := ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 7*ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 6*ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + 5*ecMinProb
			if coded < lower {
				symbol = 3
				upper = lower
				lower = ((rngHi * (c3 >> ecProbShift)) >> (7 - ecProbShift)) + 4*ecMinProb
				if coded < lower {
					symbol = 4
					upper = lower
					lower = ((rngHi * (c4 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
					if coded < lower {
						symbol = 5
						upper = lower
						lower = ((rngHi * (c5 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
						if coded < lower {
							symbol = 6
							upper = lower
							lower = ((rngHi * (c6 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
							if coded < lower {
								symbol = 7
								upper = lower
								lower = 0
							}
						}
					}
				}
			}
		}
	}

	if traceEntropyReads {
		traceCDFRead(uint16(c0), 8, dif, rng, readerTell(pos, cnt, tellOffs))
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	if c.allowCDFUpdate {
		count := values[8]
		rate := uint(5 + (count >> 4))
		if symbol > 0 {
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
		} else {
			values[0] = uint16(c0 - (c0 >> rate))
		}
		if symbol > 1 {
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
		} else {
			values[1] = uint16(c1 - (c1 >> rate))
		}
		if symbol > 2 {
			values[2] = uint16(c2 + ((CDFProbTop - c2) >> rate))
		} else {
			values[2] = uint16(c2 - (c2 >> rate))
		}
		if symbol > 3 {
			values[3] = uint16(c3 + ((CDFProbTop - c3) >> rate))
		} else {
			values[3] = uint16(c3 - (c3 >> rate))
		}
		if symbol > 4 {
			values[4] = uint16(c4 + ((CDFProbTop - c4) >> rate))
		} else {
			values[4] = uint16(c4 - (c4 >> rate))
		}
		if symbol > 5 {
			values[5] = uint16(c5 + ((CDFProbTop - c5) >> rate))
		} else {
			values[5] = uint16(c5 - (c5 >> rate))
		}
		if symbol > 6 {
			values[6] = uint16(c6 + ((CDFProbTop - c6) >> rate))
		} else {
			values[6] = uint16(c6 - (c6 >> rate))
		}
		if count < MaxCDFCount {
			values[8] = count + 1
		}
	}
	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
	return symbol
}

//go:nosplit
func (r *Reader) readCDF3Known(values *[MaxSymbols + 1]uint16) int {
	rangeValue := uint32(r.rng)
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
		traceCDFRead(c0, 3, r.dif, uint32(r.rng), r.BitsRead())
	}
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt := int32(r.cnt) - shift
	r.cnt = int16(cnt)
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = uint16(rng << uint(shift))
	if cnt < 0 {
		r.refill()
	}
	if r.allowCDFUpdate {
		count := values[3]
		rate := uint(4 + (count >> 4))
		c0 := uint32(values[0])
		c1 := uint32(values[1])
		if symbol > 0 {
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
		} else {
			values[0] = uint16(c0 - (c0 >> rate))
		}
		if symbol > 1 {
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
		} else {
			values[1] = uint16(c1 - (c1 >> rate))
		}
		if count < MaxCDFCount {
			values[3] = count + 1
		}
	}
	return symbol
}

//go:nosplit
func (c *Cursor) readCDF3Known(values *[MaxSymbols + 1]uint16) int {
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)

	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
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
		traceCDFRead(c0, 3, dif, rng, readerTell(pos, cnt, tellOffs))
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	if c.allowCDFUpdate {
		count := values[3]
		rate := uint(4 + (count >> 4))
		c0 := uint32(values[0])
		c1 := uint32(values[1])
		if symbol > 0 {
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
		} else {
			values[0] = uint16(c0 - (c0 >> rate))
		}
		if symbol > 1 {
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
		} else {
			values[1] = uint16(c1 - (c1 >> rate))
		}
		if count < MaxCDFCount {
			values[3] = count + 1
		}
	}
	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
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
	if c.allowCDFUpdate {
		return c.readCDF4UpdateKnown(&cdf.values)
	}
	return c.readCDF4Known(&cdf.values)
}

// ReadCDF4UpdateUnchecked decodes one four-symbol CDF and always adapts it.
// Hot callers that already selected the update-enabled path use this to avoid a
// per-symbol update-mode branch.
func (c *Cursor) ReadCDF4UpdateUnchecked(cdf *CDF) int {
	return c.readCDF4UpdateKnown(&cdf.values)
}

// ReadCDF4NoUpdateUnchecked decodes one four-symbol CDF without adapting it.
// Hot callers that already selected the no-update path use this to avoid a
// per-symbol update-mode branch.
func (c *Cursor) ReadCDF4NoUpdateUnchecked(cdf *CDF) int {
	return c.readCDF4Known(&cdf.values)
}

// ReadCDF4HighTokenUnchecked decodes the AV1 high-token base-range chain used
// by coefficient BR syntax. It is equivalent to reading the same four-symbol
// CDF up to four times, summing symbols, and stopping early when a symbol below
// 3 is observed. The caller must prove c and cdf are non-nil and cdf has
// exactly four symbols.
func (c *Cursor) ReadCDF4HighTokenUnchecked(cdf *CDF) uint8 {
	if c.allowCDFUpdate {
		return c.readCDF4HighTokenUpdateLoop(&cdf.values)
	}
	return c.readCDF4HighTokenKnown(&cdf.values)
}

// ReadCDF4HighTokenUpdateUnchecked decodes a high-token chain and always adapts
// the CDF row after each symbol.
func (c *Cursor) ReadCDF4HighTokenUpdateUnchecked(cdf *CDF) uint8 {
	return c.readCDF4HighTokenUpdateLoop(&cdf.values)
}

// ReadCDF4HighTokenNoUpdateUnchecked decodes a high-token chain without CDF
// adaptation.
func (c *Cursor) ReadCDF4HighTokenNoUpdateUnchecked(cdf *CDF) uint8 {
	return c.readCDF4HighTokenKnown(&cdf.values)
}

//go:nosplit
func (c *Cursor) readCDF4HighTokenUpdateLoop(values *[MaxSymbols + 1]uint16) uint8 {
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)

	level := uint8(0)
	for range 4 {
		rangeValue := rng
		rngHi := rangeValue >> 8
		coded := dif >> (ecWindow - 16)
		upper := rangeValue
		c0 := uint32(values[0])
		c1 := uint32(values[1])
		c2 := uint32(values[2])
		lower := ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
		symbol := uint8(0)
		if coded < lower {
			symbol = 1
			upper = lower
			lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
			if coded < lower {
				symbol = 2
				upper = lower
				lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
				if coded < lower {
					symbol = 3
					upper = lower
					lower = 0
				}
			}
		}
		if traceEntropyReads {
			traceCDFRead(uint16(c0), 4, dif, rng, readerTell(pos, cnt, tellOffs))
		}
		dif -= lower << (ecWindow - 16)
		rng = upper - lower
		shift := int32(16 - bits.Len32(rng))
		cnt -= shift
		dif = ((dif + 1) << uint(shift)) - 1
		rng <<= uint(shift)
		if cnt < 0 {
			pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
		}
		count := values[4]
		rate := uint(5 + (count >> 4))
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
		level += symbol
		if symbol != 3 {
			break
		}
	}

	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
	return level
}

//go:nosplit
func (r *Reader) readCDF4Known(values *[MaxSymbols + 1]uint16) int {
	rangeValue := uint32(r.rng)
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
		traceCDFRead(c0, 4, r.dif, uint32(r.rng), r.BitsRead())
	}
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt := int32(r.cnt) - shift
	r.cnt = int16(cnt)
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = uint16(rng << uint(shift))
	if cnt < 0 {
		r.refill()
	}
	if r.allowCDFUpdate {
		count := values[4]
		rate := uint(5 + (count >> 4))
		c0 := uint32(values[0])
		c1 := uint32(values[1])
		c2 := uint32(values[2])
		if symbol > 0 {
			values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
		} else {
			values[0] = uint16(c0 - (c0 >> rate))
		}
		if symbol > 1 {
			values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
		} else {
			values[1] = uint16(c1 - (c1 >> rate))
		}
		if symbol > 2 {
			values[2] = uint16(c2 + ((CDFProbTop - c2) >> rate))
		} else {
			values[2] = uint16(c2 - (c2 >> rate))
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
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)

	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := uint32(values[0])
	c1 := uint32(values[1])
	c2 := uint32(values[2])
	lower := ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
			if coded < lower {
				symbol = 3
				upper = lower
				lower = 0
			}
		}
	}
	if traceEntropyReads {
		traceCDFRead(uint16(c0), 4, dif, rng, readerTell(pos, cnt, tellOffs))
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
	return symbol
}

//go:nosplit
func (c *Cursor) readCDF4UpdateKnown(values *[MaxSymbols + 1]uint16) int {
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)

	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
	upper := rangeValue
	c0 := uint32(values[0])
	c1 := uint32(values[1])
	c2 := uint32(values[2])
	lower := ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
	symbol := 0
	if coded < lower {
		symbol = 1
		upper = lower
		lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
			if coded < lower {
				symbol = 3
				upper = lower
				lower = 0
			}
		}
	}
	if traceEntropyReads {
		traceCDFRead(uint16(c0), 4, dif, rng, readerTell(pos, cnt, tellOffs))
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	count := values[4]
	rate := uint(5 + (count >> 4))
	if symbol > 0 {
		values[0] = uint16(c0 + ((CDFProbTop - c0) >> rate))
	} else {
		values[0] = uint16(c0 - (c0 >> rate))
	}
	if symbol > 1 {
		values[1] = uint16(c1 + ((CDFProbTop - c1) >> rate))
	} else {
		values[1] = uint16(c1 - (c1 >> rate))
	}
	if symbol > 2 {
		values[2] = uint16(c2 + ((CDFProbTop - c2) >> rate))
	} else {
		values[2] = uint16(c2 - (c2 >> rate))
	}
	if count < MaxCDFCount {
		values[4] = count + 1
	}
	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
	return symbol
}

//go:nosplit
func (c *Cursor) readCDF4HighTokenKnown(values *[MaxSymbols + 1]uint16) uint8 {
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)
	c0 := uint32(values[0])
	c1 := uint32(values[1])
	c2 := uint32(values[2])

	level := uint8(0)
	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
	upper := rangeValue
	lower := ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
	symbol := uint8(0)
	if coded < lower {
		symbol = 1
		upper = lower
		lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
		if coded < lower {
			symbol = 2
			upper = lower
			lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
			if coded < lower {
				symbol = 3
				upper = lower
				lower = 0
			}
		}
	}
	if traceEntropyReads {
		traceCDFRead(uint16(c0), 4, dif, rng, readerTell(pos, cnt, tellOffs))
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	level += symbol
	if symbol == 3 {
		rangeValue = rng
		rngHi = rangeValue >> 8
		coded = dif >> (ecWindow - 16)
		upper = rangeValue
		lower = ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
		symbol = 0
		if coded < lower {
			symbol = 1
			upper = lower
			lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
			if coded < lower {
				symbol = 2
				upper = lower
				lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
				if coded < lower {
					symbol = 3
					upper = lower
					lower = 0
				}
			}
		}
		if traceEntropyReads {
			traceCDFRead(uint16(c0), 4, dif, rng, readerTell(pos, cnt, tellOffs))
		}
		dif -= lower << (ecWindow - 16)
		rng = upper - lower
		shift = int32(16 - bits.Len32(rng))
		cnt -= shift
		dif = ((dif + 1) << uint(shift)) - 1
		rng <<= uint(shift)
		if cnt < 0 {
			pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
		}
		level += symbol
		if symbol == 3 {
			rangeValue = rng
			rngHi = rangeValue >> 8
			coded = dif >> (ecWindow - 16)
			upper = rangeValue
			lower = ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
			symbol = 0
			if coded < lower {
				symbol = 1
				upper = lower
				lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
				if coded < lower {
					symbol = 2
					upper = lower
					lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
					if coded < lower {
						symbol = 3
						upper = lower
						lower = 0
					}
				}
			}
			if traceEntropyReads {
				traceCDFRead(uint16(c0), 4, dif, rng, readerTell(pos, cnt, tellOffs))
			}
			dif -= lower << (ecWindow - 16)
			rng = upper - lower
			shift = int32(16 - bits.Len32(rng))
			cnt -= shift
			dif = ((dif + 1) << uint(shift)) - 1
			rng <<= uint(shift)
			if cnt < 0 {
				pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
			}
			level += symbol
			if symbol == 3 {
				rangeValue = rng
				rngHi = rangeValue >> 8
				coded = dif >> (ecWindow - 16)
				upper = rangeValue
				lower = ((rngHi * (c0 >> ecProbShift)) >> (7 - ecProbShift)) + 3*ecMinProb
				symbol = 0
				if coded < lower {
					symbol = 1
					upper = lower
					lower = ((rngHi * (c1 >> ecProbShift)) >> (7 - ecProbShift)) + 2*ecMinProb
					if coded < lower {
						symbol = 2
						upper = lower
						lower = ((rngHi * (c2 >> ecProbShift)) >> (7 - ecProbShift)) + ecMinProb
						if coded < lower {
							symbol = 3
							upper = lower
							lower = 0
						}
					}
				}
				if traceEntropyReads {
					traceCDFRead(uint16(c0), 4, dif, rng, readerTell(pos, cnt, tellOffs))
				}
				dif -= lower << (ecWindow - 16)
				rng = upper - lower
				shift = int32(16 - bits.Len32(rng))
				cnt -= shift
				dif = ((dif + 1) << uint(shift)) - 1
				rng <<= uint(shift)
				if cnt < 0 {
					pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
				}
				level += symbol
			}
		}
	}

	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
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
	rangeValue := uint32(r.rng)
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
		traceCDFRead(c0, 2, r.dif, uint32(r.rng), r.BitsRead())
	}
	dif := r.dif - (lower << (ecWindow - 16))
	rng := upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt := int32(r.cnt) - shift
	r.cnt = int16(cnt)
	r.dif = ((dif + 1) << uint(shift)) - 1
	r.rng = uint16(rng << uint(shift))
	if cnt < 0 {
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
	src := c.src
	pos := int(c.pos)
	dif := c.dif
	rng := uint32(c.rng)
	cnt := int32(c.cnt)
	tellOffs := int32(c.tellOffs)

	rangeValue := rng
	rngHi := rangeValue >> 8
	coded := dif >> (ecWindow - 16)
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
		traceCDFRead(c0, 2, dif, rng, readerTell(pos, cnt, tellOffs))
	}
	dif -= lower << (ecWindow - 16)
	rng = upper - lower
	shift := int32(16 - bits.Len32(rng))
	cnt -= shift
	dif = ((dif + 1) << uint(shift)) - 1
	rng <<= uint(shift)
	if cnt < 0 {
		pos, dif, cnt, tellOffs = refillState(src, pos, dif, cnt, tellOffs)
	}
	if c.allowCDFUpdate {
		updateCDF2(values, symbol)
	}
	c.pos = uint32(pos)
	c.dif = dif
	c.rng = uint16(rng)
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
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

// ReadSignedDelta decodes the AV1 CDF-coded signed delta core used by
// delta_qindex and delta_lflevel tile syntax.
func (r *Reader) ReadSignedDelta(cdf *CDF, small uint8) (int, error) {
	if small == 0 || small >= MaxSymbols {
		return 0, ErrInvalidRange
	}
	smallInt := int(small)
	if cdf == nil || cdf.Symbols() != smallInt+1 {
		return 0, ErrInvalidCDF
	}

	abs, err := r.ReadCDF(cdf)
	if err != nil {
		return 0, err
	}
	if abs >= smallInt {
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
	pos, dif, cnt, tellOffs := refillState(r.src, int(r.pos), r.dif, int32(r.cnt), int32(r.tellOffs))
	r.pos = uint32(pos)
	r.dif = dif
	r.cnt = int16(cnt)
	r.tellOffs = int16(tellOffs)
}

func (c *Cursor) refill() {
	pos, dif, cnt, tellOffs := refillState(c.src, int(c.pos), c.dif, int32(c.cnt), int32(c.tellOffs))
	c.pos = uint32(pos)
	c.dif = dif
	c.cnt = int16(cnt)
	c.tellOffs = int16(tellOffs)
}
