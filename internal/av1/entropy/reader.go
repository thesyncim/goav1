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

	r.normalize(r.dif-(lower<<(ecWindow-16)), upper-lower)
	if r.allowCDFUpdate {
		updateCDF(cdf, symbols, symbol)
	}
	return symbol, nil
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
