package quantize

import "math/bits"

// BParams is one quantizer's quantize_b constants at one transform scale,
// built once per quantizer the way SVT-AV1 builds its per-qindex quants
// tables (av1_build_quantizer: zbin, round, quant and quant_shift) rather
// than per block. The lane tables hold the vector kernels' broadcast
// constants, which they load from memory instead of rebuilding them from
// general registers at every call.
type BParams struct {
	q       Quantizer
	txScale uint8
	set     bool
	// tag is the tx scale plus one, BParamsCache's key with q (zero for an
	// unset entry).
	tag uint16
	// vector reports that the vector kernels' preconditions hold: positive
	// steps below 1<<16 and log2(step) >= txScale for both steps (the folded
	// shift).
	vector bool
	dc     quantBParams
	// acZbin is the AC zero bin as the kernels' unsigned group test reads
	// it; minZbin is the smaller of the DC and AC bins, the DC group's test.
	acZbin, minZbin uint32
	lanes           bParamLanes
}

// bParamLanes holds the broadcast lanes of the quantize_b AC rule (see
// quantizeBLane4SIMD) and of the fused row-group statistics.
// The dc lanes are the first group's: lane 0 carries the DC constants
// (svt_aom_quantize_b_neon's v_zbins0, v_round0, v_quant0 and quant shift).
type bParamLanes struct {
	quant, round, zbin, shift         [4]int32
	dcQuant, dcRound, dcZbin, dcShift [4]int32
	max, sixteen, signShift           [4]int32
	step, txShift, expShift           [4]int32
	rateBias                          [4]uint32
	one16                             [8]int16
}

// Set builds p for quantizer q at txScale.
func (p *BParams) Set(q Quantizer, txScale uint8) {
	*p = BParams{q: q, txScale: txScale, set: true, tag: uint16(txScale) + 1}
	if q.AC <= 0 || q.DC <= 0 || q.AC >= 1<<16 || q.DC >= 1<<16 || txScale > 2 {
		return
	}
	l := 31 - int32(bits.LeadingZeros32(uint32(q.AC)))
	lDC := 31 - int32(bits.LeadingZeros32(uint32(q.DC)))
	if l < int32(txScale) || lDC < int32(txScale) {
		return
	}
	p.vector = true
	zbinFactor := int32(80)
	if q.DC < 148 {
		zbinFactor = 84
	}
	dcQuant, dcShift := invertQuant(q.DC)
	p.dc = quantBParams{
		zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.DC, 7), txScale),
		round: roundPowerOfTwo((48*q.DC)>>7, txScale),
		quant: dcQuant,
		shift: dcShift,
	}
	acQuant, _ := invertQuant(q.AC)
	acZbin := roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale)
	p.acZbin = uint32(acZbin)
	p.minZbin = uint32(min(acZbin, p.dc.zbin))
	ln := &p.lanes
	for i := range 4 {
		ln.quant[i] = acQuant
		ln.round[i] = roundPowerOfTwo((48*q.AC)>>7, txScale)
		ln.zbin[i] = acZbin
		ln.shift[i] = -(l - int32(txScale))
		ln.max[i] = maxInt16
		ln.sixteen[i] = -16
		ln.signShift[i] = -31
		ln.step[i] = q.AC
		ln.txShift[i] = -int32(txScale)
		ln.expShift[i] = -23
		ln.rateBias[i] = 124
	}
	ln.dcQuant, ln.dcRound, ln.dcZbin, ln.dcShift = ln.quant, ln.round, ln.zbin, ln.shift
	ln.dcQuant[0] = dcQuant
	ln.dcRound[0] = p.dc.round
	ln.dcZbin[0] = p.dc.zbin
	ln.dcShift[0] = -(lDC - int32(txScale))
	for i := range ln.one16 {
		ln.one16[i] = 1
	}
}

// BParamsCache keeps the BParams of the last few quantizers. An encoder's
// blocks cycle through a handful of (quantizer, tx scale) pairs (the three
// planes' steps at the coded transform sizes), so a block finds its
// constants built. The zero value is empty; it is not safe for concurrent
// use.
type BParamsCache struct {
	e    [4]BParams
	last uint8 // the entry the previous Get returned
	next uint8 // the entry the next miss replaces
}

// Get returns the BParams of q at txScale, building them on a miss.
func (c *BParamsCache) Get(q Quantizer, txScale uint8) *BParams {
	p := &c.e[c.last&3]
	if p.q != q || p.tag != uint16(txScale)+1 {
		p = c.find(q, txScale)
	}
	return p
}

func (c *BParamsCache) find(q Quantizer, txScale uint8) *BParams {
	for i := range c.e {
		if p := &c.e[i]; p.set && p.q == q && p.txScale == txScale {
			c.last = uint8(i)
			return p
		}
	}
	i := c.next % uint8(len(c.e))
	c.next++
	c.last = i
	p := &c.e[i]
	p.Set(q, txScale)
	return p
}

// TryQuantizeBBlockEOBTrusted is QuantizeBlockScaledBEOB for count
// contiguous coefficients (a column-major block whose strides equal its
// height; count a multiple of 8) with p's quantizer and tx scale, as
// svt_aom_quantize_b runs over n_coeffs. It adds a trusted live bound: every
// coefficient at or past live (a multiple of 8, 8 <= live <= count) must lie
// below its zero bin, and those are not read (SVT's N2/N4 forward transforms
// leave the high columns zero). iscan must put DC first (iscan[0] == 0), as
// every AV1 scan does. When handled is false nothing has been written and the
// caller must use QuantizeBlockScaledBEOB.
func TryQuantizeBBlockEOBTrusted(qcoeff []int16, coeff []int32, count, live int, p *BParams, iscan []int16) (eob int, handled bool) {
	if quantizeBBlockEOBTrustedImpl == nil {
		return 0, false
	}
	return quantizeBBlockEOBTrustedImpl(qcoeff, coeff, count, live, p, iscan)
}

// quantizeBBlockEOBTrustedImpl binds the architecture's kernel; nil keeps the
// generic path.
var quantizeBBlockEOBTrustedImpl func(qcoeff []int16, coeff []int32, count, live int, p *BParams, iscan []int16) (int, bool)
