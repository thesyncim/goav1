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
	for i := uint8(0); i < n; i++ {
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

	rangeValue := r.rng
	coded := r.dif >> (ecWindow - 16)
	upper := rangeValue
	lower := uint32(0)
	symbol := 0
	last := symbols - 1
	for {
		lower = (((rangeValue >> 8) * uint32(cdf[symbol]>>ecProbShift)) >> (7 - ecProbShift)) +
			ecMinProb*uint32(last-symbol)
		if coded >= lower {
			break
		}
		symbol++
		if symbol >= symbols {
			return 0, ErrInvalidCDF
		}
		upper = lower
	}
	if lower >= upper {
		return 0, ErrInvalidCDF
	}

	traceCDFRead(cdf[0], symbols, r.dif, r.rng, r.BitsRead())
	r.normalize(r.dif-(lower<<(ecWindow-16)), upper-lower)
	if r.allowCDFUpdate {
		updateCDF(cdf, symbols, symbol)
	}
	return symbol, nil
}

// ReadCDF decodes one symbol from caller-owned CDF state.
func (r *Reader) ReadCDF(cdf *CDF) (int, error) {
	if cdf == nil {
		return 0, ErrInvalidCDF
	}
	return r.ReadSymbol(cdf.Values(), cdf.Symbols())
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
