package encoder

import (
	"encoding/binary"
	"errors"
	"math/bits"
)

// symbol_writer.go is a source-shaped port of the libaom daala range ENCODER,
// pinned at third_party/upstream/libaom/aom_dsp/entenc.c (struct od_ec_enc) with
// constants from aom_dsp/entcode.h and aom_dsp/prob.h. It is the exact write-side
// inverse of the decoder's entropy.Reader (a faithful od_ec_dec port that is
// byte-exact against libaom on the 226-vector strict suite). Encoding a symbol
// stream here and decoding it with entropy.Reader is the oracle round-trip gate.
//
// Upstream -> local translation ledger (status: translated, parity: round-trip
// oracle test in symbol_writer_test.go):
//
//	od_ec_enc (struct)            -> symbolWriter
//	od_ec_enc_reset               -> newSymbolWriter
//	od_ec_enc_normalize           -> (*symbolWriter).normalize
//	write_enc_data_to_out_buf     -> (*symbolWriter).writeOut
//	propagate_carry_bwd           -> (*symbolWriter).propagateCarry
//	od_ec_encode_q15              -> (*symbolWriter).encodeQ15
//	od_ec_encode_bool_q15         -> (*symbolWriter).writeBoolQ15
//	od_ec_encode_cdf_q15          -> (*symbolWriter).writeSymbol
//	aom_write_bit / aom_write_literal -> (*symbolWriter).writeLiteral (bool path)
//	od_ec_enc_done                -> (*symbolWriter).finish
//
// Deviations from C widths, all value-preserving (the live range never exceeds
// the C width): rng is uint32 not uint16_t, cnt is int32 not int16_t. These match
// how the C encode functions promote rng/cnt to `unsigned`/`int` locals; the
// stored values stay within the narrower C width after every normalize.

const (
	ecProbShiftEnc = 6 // EC_PROB_SHIFT (entcode.h)
	ecMinProbEnc   = 4 // EC_MIN_PROB   (entcode.h)
	cdfProbBitsEnc = 15
	cdfProbTopEnc  = 1 << cdfProbBitsEnc // CDF_PROB_TOP == 32768 (prob.h)
)

// errCarryUnderflow signals a carry that would propagate before the first output
// byte, which cannot occur for a well-formed range-coded stream (libaom asserts
// offs > 0 in propagate_carry_bwd).
var errCarryUnderflow = errors.New("encoder: range coder carry underflow")

// symbolWriter is the AV1 range/entropy encoder. Callers own the output buffer
// (EncodeInto style): newSymbolWriter takes a destination slice that is grown
// only if it is too small, and finish returns the buffer sub-slice that holds
// the finalized arithmetic-coded bytes.
type symbolWriter struct {
	buf  []byte // od_ec_enc.buf: the output byte buffer (caller-owned)
	offs int    // od_ec_enc.offs: index of the next entropy byte to write
	low  uint64 // od_ec_enc.low: low end of the current range (od_ec_enc_window)
	rng  uint32 // od_ec_enc.rng: number of values in the current range, in [0x8000,0xFFFF]
	cnt  int32  // od_ec_enc.cnt: number of bits of data in the current value
	err  error  // od_ec_enc.error (sticky)
}

// newSymbolWriter initializes the encoder writing into dst. dst[:0] is used as
// the backing buffer; it grows in place if its capacity is exhausted. This is
// od_ec_enc_reset: low=0, rng=0x8000, cnt=-9 (so cnt crosses zero after one byte
// plus one carry bit is accumulated).
func newSymbolWriter(dst []byte) symbolWriter {
	return symbolWriter{
		buf: dst[:cap(dst)],
		rng: 0x8000,
		cnt: -9,
	}
}

// ensure guarantees len(w.buf) >= n, growing (amortized) like the C realloc to
// 2*storage+8 but never below n.
func (w *symbolWriter) ensure(n int) {
	if n <= len(w.buf) {
		return
	}
	grown := max(2*len(w.buf)+8, n)
	next := make([]byte, grown)
	copy(next, w.buf[:w.offs])
	w.buf = next
}

// normalize renormalizes (low, rng) so that 0x8000 <= rng < 0x10000, flushing
// ready bytes to the output when low has accumulated enough data. Direct port of
// od_ec_enc_normalize (entenc.c).
func (w *symbolWriter) normalize(low uint64, rng uint32) {
	if w.err != nil {
		return
	}
	c := w.cnt
	// d = 16 - OD_ILOG_NZ(rng); OD_ILOG_NZ(x) == 1+get_msb(x) == bits.Len32(x).
	d := int32(16 - bits.Len32(rng))
	s := c + d
	if s >= 40 { // 56 - 16 (see entenc.c rationale)
		w.ensure(w.offs + 8)
		numBytesReady := uint32((s >> 3) + 1)
		// enc->cnt always counts one byte less, so one extra byte is ready.
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
// full 8-byte store, of which the trailing bytes are overwritten by the next
// flush), propagates any carry backwards, and advances w.offs. Port of
// write_enc_data_to_out_buf (entenc.h).
func (w *symbolWriter) writeOut(output, carry uint64, numBytesReady uint32) {
	// HToBE64(output << ((8 - num_bytes_ready) << 3)) then memcpy 8 bytes.
	binary.BigEndian.PutUint64(w.buf[w.offs:w.offs+8], output<<((8-numBytesReady)<<3))
	if carry != 0 {
		w.propagateCarry(w.offs - 1)
	}
	w.offs += int(numBytesReady)
}

// propagateCarry adds one to buf[offs] and ripples the carry toward lower
// offsets. Port of propagate_carry_bwd (entenc.h).
func (w *symbolWriter) propagateCarry(offs int) {
	for {
		if offs < 0 {
			w.err = errCarryUnderflow
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

// encodeQ15 codes a symbol whose Q15 cumulative-frequency interval is [fl, fh)
// in inverse-CDF terms (fl = OD_ICDF of the symbols before s, fh = OD_ICDF up to
// and including s), at index s within an nsyms-symbol alphabet. Port of
// od_ec_encode_q15 (entenc.c).
func (w *symbolWriter) encodeQ15(fl, fh uint32, s, nsyms int) {
	l := w.low
	r := w.rng
	n := uint32(nsyms - 1)
	if fl < cdfProbTopEnc {
		u := ((r>>8)*(fl>>ecProbShiftEnc))>>(7-ecProbShiftEnc) + ecMinProbEnc*(n-uint32(s-1))
		v := ((r>>8)*(fh>>ecProbShiftEnc))>>(7-ecProbShiftEnc) + ecMinProbEnc*(n-uint32(s))
		l += uint64(r - u)
		r = u - v
	} else {
		r -= ((r>>8)*(fh>>ecProbShiftEnc))>>(7-ecProbShiftEnc) + ecMinProbEnc*(n-uint32(s))
	}
	w.normalize(l, r)
}

// writeBoolQ15 codes a single binary value val (0/1) where f is the Q15
// probability that val is one. Port of od_ec_encode_bool_q15 (entenc.c).
func (w *symbolWriter) writeBoolQ15(val int, f uint32) {
	l := w.low
	r := w.rng
	v := ((r >> 8) * (f >> ecProbShiftEnc)) >> (7 - ecProbShiftEnc)
	v += ecMinProbEnc
	if val != 0 {
		l += uint64(r - v)
		r = v
	} else {
		r -= v
	}
	w.normalize(l, r)
}

// writeSymbol codes symbol index s using the inverse CDF icdf (icdf[i] ==
// CDF_PROB_TOP - cdf[i], monotonically decreasing, icdf[nsyms-1] == 0). This is
// the same representation the decoder's entropy.CDF / ReadSymbol uses. Port of
// od_ec_encode_cdf_q15 (entenc.c). CDF adaptation (update_cdf) is intentionally
// not done here yet; callers pass a fixed CDF until the adaptation pass lands.
func (w *symbolWriter) writeSymbol(s int, icdf []uint16, nsyms int) {
	fl := uint32(cdfProbTopEnc) // OD_ICDF(0)
	if s > 0 {
		fl = uint32(icdf[s-1])
	}
	w.encodeQ15(fl, uint32(icdf[s]), s, nsyms)
}

// writeLiteral codes the low n bits of value MSB-first as equiprobable bits,
// matching aom_write_literal -> aom_write_bit -> aom_write(.,128) ->
// od_ec_encode_bool_q15(., 1<<14). This is the exact inverse of the decoder's
// ReadBits/ReadBitsTrusted equiprobable-bit path.
func (w *symbolWriter) writeLiteral(value uint32, n int) {
	for i := n - 1; i >= 0; i-- {
		w.writeBoolQ15(int((value>>uint(i))&1), 1<<14)
	}
}

// finish flushes the minimum number of bits guaranteeing the coded symbols
// decode correctly regardless of trailing bytes, and returns the finalized
// arithmetic-coded bytes. Port of od_ec_enc_done (entenc.c). The writer must not
// be used after finish.
func (w *symbolWriter) finish() ([]byte, error) {
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
