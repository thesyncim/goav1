package entropy

import (
	"encoding/binary"
	"errors"
	"math/bits"
)

// writer.go is the write side of the AV1 range coder, a source-shaped port of
// libaom's daala range encoder (third_party/upstream/libaom/aom_dsp/entenc.c,
// struct od_ec_enc) and the exact inverse of Reader (a faithful od_ec_dec port,
// byte-exact against libaom on the 226-vector strict suite). Encoding a symbol
// stream here and decoding it with Reader is the oracle round-trip gate. It
// lives beside Reader so both the tile coefficient/mode coders and the header
// emitter share one range coder.
//
// Upstream -> local translation ledger:
//
//	od_ec_enc (struct)            -> Writer
//	od_ec_enc_reset               -> NewWriter
//	od_ec_enc_normalize           -> (*Writer).normalize
//	write_enc_data_to_out_buf     -> (*Writer).writeOut
//	propagate_carry_bwd           -> (*Writer).propagateCarry
//	od_ec_encode_q15              -> (*Writer).encodeQ15
//	od_ec_encode_bool_q15         -> (*Writer).WriteBoolQ15
//	od_ec_encode_cdf_q15          -> (*Writer).WriteSymbol
//	aom_write_bit / aom_write_literal -> (*Writer).WriteLiteral (bool path)
//	update_cdf                    -> updateCDFWindow (shared with Reader) via WriteSymbolAdaptive
//	od_ec_enc_done                -> (*Writer).Finish
//
// Deviations from C widths, all value-preserving (the live range never exceeds
// the C width): rng is uint32 not uint16_t and cnt is int32 not int16_t, matching
// how the C encode functions promote rng/cnt to unsigned/int locals.

// ErrCarryUnderflow signals a carry that would propagate before the first output
// byte, which cannot occur for a well-formed range-coded stream (libaom asserts
// offs > 0 in propagate_carry_bwd).
var ErrCarryUnderflow = errors.New("entropy: range coder carry underflow")

// Writer is the AV1 range/entropy encoder. Callers own the output buffer
// (EncodeInto style): NewWriter takes a destination slice that grows only if it
// is too small, and Finish returns the sub-slice holding the finalized bytes.
type Writer struct {
	buf  []byte // od_ec_enc.buf: output byte buffer (caller-owned)
	offs int    // od_ec_enc.offs: index of the next entropy byte to write
	low  uint64 // od_ec_enc.low: low end of the current range (od_ec_enc_window)
	rng  uint32 // od_ec_enc.rng: values in the current range, in [0x8000,0xFFFF]
	cnt  int32  // od_ec_enc.cnt: number of bits of data in the current value
	err  error  // od_ec_enc.error (sticky)
}

// NewWriter initializes the encoder writing into dst[:0] (its full capacity is
// used as the backing buffer; it grows in place if exhausted). This is
// od_ec_enc_reset: low=0, rng=0x8000, cnt=-9.
func NewWriter(dst []byte) Writer {
	return Writer{
		buf: dst[:cap(dst)],
		rng: 0x8000,
		cnt: -9,
	}
}

// ensure guarantees len(w.buf) >= n, growing (amortized) like the C realloc.
func (w *Writer) ensure(n int) {
	if n <= len(w.buf) {
		return
	}
	grown := max(2*len(w.buf)+8, n)
	next := make([]byte, grown)
	copy(next, w.buf[:w.offs])
	w.buf = next
}

// normalize renormalizes (low, rng) so 0x8000 <= rng < 0x10000, flushing ready
// bytes to the output. Port of od_ec_enc_normalize.
func (w *Writer) normalize(low uint64, rng uint32) {
	if w.err != nil {
		return
	}
	c := w.cnt
	d := int32(16 - bits.Len32(rng)) // 16 - OD_ILOG_NZ(rng)
	s := c + d
	if s >= 40 { // 56 - 16 (see entenc.c rationale)
		w.ensure(w.offs + 8)
		numBytesReady := uint32((s >> 3) + 1)
		c += 24 - int32(numBytesReady<<3)
		output := low >> uint32(c)
		low &= (uint64(1) << uint32(c)) - 1
		mask := uint64(1) << (numBytesReady << 3)
		carry := output & mask
		mask--
		output &= mask
		w.writeOut(output, carry, numBytesReady)
		s = c + d - 24
	}
	w.low = low << uint32(d)
	w.rng = rng << uint32(d)
	w.cnt = s
}

// writeOut writes numBytesReady big-endian bytes of output at w.offs (reserving a
// full 8-byte store whose trailing bytes the next flush overwrites), propagates
// any carry backwards, and advances w.offs. Port of write_enc_data_to_out_buf.
func (w *Writer) writeOut(output, carry uint64, numBytesReady uint32) {
	binary.BigEndian.PutUint64(w.buf[w.offs:w.offs+8], output<<((8-numBytesReady)<<3))
	if carry != 0 {
		w.propagateCarry(w.offs - 1)
	}
	w.offs += int(numBytesReady)
}

// propagateCarry adds one to buf[offs] and ripples toward lower offsets. Port of
// propagate_carry_bwd.
func (w *Writer) propagateCarry(offs int) {
	for {
		if offs < 0 {
			w.err = ErrCarryUnderflow
			return
		}
		sum := uint16(w.buf[offs]) + 1
		w.buf[offs] = byte(sum)
		offs--
		if sum>>8 == 0 {
			return
		}
	}
}

// encodeQ15 codes a symbol whose Q15 inverse-CDF interval is [fl, fh) at index s
// of an nsyms-symbol alphabet. Port of od_ec_encode_q15.
func (w *Writer) encodeQ15(fl, fh uint32, s, nsyms int) {
	l := w.low
	r := w.rng
	n := uint32(nsyms - 1)
	if fl < CDFProbTop {
		u := ((r>>8)*(fl>>ecProbShift))>>(7-ecProbShift) + ecMinProb*(n-uint32(s-1))
		v := ((r>>8)*(fh>>ecProbShift))>>(7-ecProbShift) + ecMinProb*(n-uint32(s))
		l += uint64(r - u)
		r = u - v
	} else {
		r -= ((r>>8)*(fh>>ecProbShift))>>(7-ecProbShift) + ecMinProb*(n-uint32(s))
	}
	w.normalize(l, r)
}

// WriteBoolQ15 codes a single binary value val (0/1) where f is the Q15
// probability that val is one. Port of od_ec_encode_bool_q15.
func (w *Writer) WriteBoolQ15(val int, f uint32) {
	l := w.low
	r := w.rng
	v := ((r >> 8) * (f >> ecProbShift)) >> (7 - ecProbShift)
	v += ecMinProb
	if val != 0 {
		l += uint64(r - v)
		r = v
	} else {
		r -= v
	}
	w.normalize(l, r)
}

// WriteSymbol codes symbol index s using the inverse CDF icdf (icdf[i] ==
// CDF_PROB_TOP - cdf[i], monotone decreasing, icdf[nsyms-1] == 0) without
// adapting it. Port of od_ec_encode_cdf_q15.
func (w *Writer) WriteSymbol(s int, icdf []uint16, nsyms int) {
	fl := uint32(CDFProbTop) // OD_ICDF(0)
	if s > 0 {
		fl = uint32(icdf[s-1])
	}
	w.encodeQ15(fl, uint32(icdf[s]), s, nsyms)
}

// WriteSymbolAdaptive codes symbol s from the adaptive inverse CDF cdf (len ==
// nsyms+1, the trailing slot holding the adaptation count) and then adapts cdf in
// place via the same updateCDFWindow the decoder's ReadSymbol uses, so both sides
// evolve the CDF identically.
func (w *Writer) WriteSymbolAdaptive(s int, cdf []uint16) {
	w.WriteSymbol(s, cdf, len(cdf)-1)
	updateCDFWindow(cdf, s)
}

// WriteCDF codes symbol s using caller-owned adaptive CDF state and adapts it in
// place, the write-side inverse of Reader.ReadCDF and the ReadCDF*Unchecked
// adaptive readers. It always adapts, matching a Reader with CDF update enabled;
// callers coding under disable_cdf_update use WriteSymbol with a non-adapting CDF.
func (w *Writer) WriteCDF(cdf *CDF, s int) {
	n := int(cdf.symbols)
	vals := cdf.values[:n+1]
	w.WriteSymbol(s, vals, n)
	updateCDFWindow(vals, s)
}

// WriteLiteral codes the low n bits of value MSB-first as equiprobable bits,
// matching aom_write_literal -> aom_write_bit -> od_ec_encode_bool_q15(., 1<<14),
// the exact inverse of Reader.ReadBits.
func (w *Writer) WriteLiteral(value uint32, n int) {
	for i := n - 1; i >= 0; i-- {
		w.WriteBoolQ15(int((value>>uint(i))&1), 1<<14)
	}
}

// Finish flushes the minimum number of bits guaranteeing the coded symbols decode
// correctly regardless of trailing bytes and returns the finalized bytes. Port of
// od_ec_enc_done. The writer must not be used after Finish.
func (w *Writer) Finish() ([]byte, error) {
	if w.err != nil {
		return nil, w.err
	}
	l := w.low
	c := w.cnt
	s := int32(10)
	m := uint64(0x3FFF)
	e := ((l + m) &^ m) | (m + 1)
	s += c
	sBits := max(int((s+7)>>3), 0)
	w.ensure(w.offs + sBits)
	if s > 0 {
		n := (uint64(1) << uint32(c+16)) - 1
		for {
			val := uint16(e >> uint32(c+16))
			w.buf[w.offs] = byte(val & 0x00FF)
			if val&0x0100 != 0 {
				w.propagateCarry(w.offs - 1)
			}
			w.offs++
			e &= n
			s -= 8
			c -= 8
			n >>= 8
			if s <= 0 {
				break
			}
		}
	}
	if w.err != nil {
		return nil, w.err
	}
	return w.buf[:w.offs], nil
}
