package entropy

import "errors"

// writer_log.go records a tile's range-coder symbol stream so the tile can be
// re-emitted with syntax elements the encoder only decides after the whole
// frame is reconstructed.
//
// SVT-AV1 (and libaom) entropy-code a frame after its in-loop filter
// decisions: Source/Lib/Codec/entropy_coding_process.c runs after the CDEF
// search in cdef_process.c, so write_cdef already knows each 64x64 filter
// block's cdef_idx. goav1's tile coders write symbols while they decide, so
// they cannot know cdef_idx yet. A WriterLog captures every coded symbol's
// od_ec_encode_q15 operands plus markers at the superblock starts and at the
// read_cdef positions; Replay re-runs the range coder over the log and
// splices the cdef_idx literals in. cdef_idx is an equiprobable literal with
// no CDF and no context, so no other symbol's interval changes: the replayed
// stream is the one the encoder would have written had it known the indices
// up front.

// ErrLogReplay signals a cdef_idx marker outside the supplied index table.
var ErrLogReplay = errors.New("entropy: writer log replay out of range")

// WriterLog is the caller-owned symbol log of one Writer. Its storage is
// reused across Reset calls.
type WriterLog struct {
	ops []uint32
}

// A symbol entry keeps exactly what od_ec_encode_q15 reads: fl >> 6 (bits
// 0-8), a flag for fl == 32768 (bit 9, the s == 0 case), fh >> 6 (bits
// 10-18) and nsyms - s (bits 19-23, 1..16), the only use of s and nsyms in
// the EC_MIN_PROB terms. A non-zero kind in bits 24-31 marks a control
// entry instead.
const (
	logFLTop          = 1 << 9
	logKindShift      = 24
	logKindSuperblock = 1 << logKindShift
	logKindCDEFUnit   = 2 << logKindShift
)

// Reset empties the log, keeping its storage.
func (l *WriterLog) Reset() {
	l.ops = l.ops[:0]
}

// Reserve grows the log's storage to at least n entries.
func (l *WriterLog) Reserve(n int) {
	if cap(l.ops) < n {
		l.ops = make([]uint32, 0, n)
	}
}

// Len reports the number of recorded entries.
func (l *WriterLog) Len() int {
	return len(l.ops)
}

func (l *WriterLog) sym(fl, fh uint32, s, nsyms int) {
	op := (fh>>ecProbShift)<<10 | uint32(nsyms-s)<<19
	if fl >= CDFProbTop {
		op |= logFLTop
	} else {
		op |= fl >> ecProbShift
	}
	l.ops = append(l.ops, op)
}

// cdf records symbol s of an n-symbol inverse CDF before adaptation.
func (l *WriterLog) cdf(values *[MaxSymbols + 1]uint16, s, n int) {
	fl := uint32(CDFProbTop)
	if s > 0 {
		fl = uint32(values[s-1])
	}
	l.sym(fl, uint32(values[s]), s, n)
}

// boolQ15 records od_ec_encode_bool_q15(val, f): the two-symbol interval
// with icdf[0] = f, which encodeQ15 codes identically.
func (l *WriterLog) boolQ15(val int, f uint32) {
	if val != 0 {
		l.sym(f, 0, 1, 2)
		return
	}
	l.sym(CDFProbTop, f, 0, 2)
}

// SetLog starts recording every symbol this writer codes into l (nil stops
// recording). The writer keeps coding normally; the log is extra.
func (w *Writer) SetLog(l *WriterLog) {
	w.log = l
}

// Logging reports whether the writer records its symbol stream.
func (w *Writer) Logging() bool {
	return w.log != nil
}

// LogSuperblock marks the start of the next superblock in coding order.
func (w *Writer) LogSuperblock() {
	if w.log != nil {
		w.log.ops = append(w.log.ops, logKindSuperblock)
	}
}

// LogCDEFUnits marks the read_cdef position of a non-skip block: first is
// the block's top-left 64x64 CDEF unit inside the current superblock and
// mask the units it covers (bit u = unit u, row-major 2x2). Replay codes
// the literal only for the first such block of a unit, as the decoder reads
// it.
func (w *Writer) LogCDEFUnits(first uint8, mask uint8) {
	if w.log != nil {
		w.log.ops = append(w.log.ops, logKindCDEFUnit|uint32(first)|uint32(mask)<<8)
	}
}

// replayQ15 is encodeQ15 on the logged operands: u and v depend on fl and
// fh only through fl >> 6 and fh >> 6, and on s and nsyms only through
// nsyms - s.
func (w *Writer) replayQ15(op uint32) {
	l := w.low
	r := w.rng
	ns := op >> 19 & 0x1f
	fh := op >> 10 & 0x1ff
	if op&logFLTop == 0 {
		fl := op & 0x1ff
		u := ((r>>8)*fl)>>(7-ecProbShift) + ecMinProb*ns
		v := ((r>>8)*fh)>>(7-ecProbShift) + ecMinProb*(ns-1)
		l += uint64(r - u)
		r = u - v
	} else {
		r -= ((r>>8)*fh)>>(7-ecProbShift) + ecMinProb*(ns-1)
	}
	w.normalize(l, r)
}

// Replay codes the logged stream into dst (its capacity is reused) and
// returns the finished bytes. At each logged cdef_idx position whose unit
// is still unread in its superblock it writes the bits-wide literal
// cdefIdx[sb*4+unit], sb counting the logged superblocks from zero. With
// bits == 0 the output equals the original writer's bytes.
func (l *WriterLog) Replay(dst []byte, cdefIdx []uint8, bits int) ([]byte, error) {
	w := NewWriter(dst)
	sb := -1
	var seen uint8
	for _, op := range l.ops {
		switch op >> logKindShift {
		case 0:
			w.replayQ15(op)
		case logKindSuperblock >> logKindShift:
			sb++
			seen = 0
		case logKindCDEFUnit >> logKindShift:
			first := uint8(op & 0xff)
			if seen&(1<<first) != 0 || bits == 0 {
				continue
			}
			seen |= uint8(op >> 8)
			i := sb*4 + int(first)
			if sb < 0 || i >= len(cdefIdx) {
				return nil, ErrLogReplay
			}
			w.WriteLiteral(uint32(cdefIdx[i]), bits)
		}
	}
	return w.Finish()
}
