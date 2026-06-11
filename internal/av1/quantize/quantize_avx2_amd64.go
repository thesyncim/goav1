//go:build amd64 && !purego

package quantize

import (
	"math/bits"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/dsp/cpu"
)

// quantizeBAVX2Ctx carries the zbin-rule kernel arguments; offsets are
// mirrored by #define in quantize_avx2_amd64.s. Shift is l - txScale.
type quantizeBAVX2Ctx struct {
	Coeff unsafe.Pointer
	Out   unsafe.Pointer
	Count int64
	Quant int64
	Round int64
	Zbin  int64
	Shift int64
}

//go:noescape
func quantizeBAVX2Asm(ctx *quantizeBAVX2Ctx)

// quantizeBBlockAVX2 mirrors quantizeBBlockNEON: the vector kernel covers
// the block eight coefficients at a time and the DC coefficient is redone
// with the scalar rule and DC constants. The 32-bit lane math is exact
// because invert_quant's factor lies in (-32768, 1].
func quantizeBBlockAVX2(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	if count%16 != 0 {
		return false
	}
	l := 31 - int64(bits.LeadingZeros32(uint32(q.AC)))
	if l < int64(txScale) {
		return false
	}
	zbinFactor := int32(80)
	if q.DC < 148 {
		zbinFactor = 84
	}
	acQuant, _ := invertQuant(q.AC)
	ctx := quantizeBAVX2Ctx{
		Coeff: unsafe.Pointer(&coeff[0]),
		Out:   unsafe.Pointer(&qcoeff[0]),
		Count: int64(count),
		Quant: int64(acQuant),
		Round: int64(roundPowerOfTwo((48*q.AC)>>7, txScale)),
		Zbin:  int64(roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.AC, 7), txScale)),
		Shift: l - int64(txScale),
	}
	quantizeBAVX2Asm(&ctx)
	dcQuant, dcShift := invertQuant(q.DC)
	dc := quantBParams{
		zbin:  roundPowerOfTwo(roundPowerOfTwo(zbinFactor*q.DC, 7), txScale),
		round: roundPowerOfTwo((48*q.DC)>>7, txScale),
		quant: dcQuant,
		shift: dcShift,
	}
	qcoeff[0] = quantizeScalarB(coeff[0], &dc, txScale)
	return true
}

// quantizeFPAVX2Ctx carries the fp-rule kernel arguments. ShiftL is
// 1+txScale and ShiftR is 16-txScale.
type quantizeFPAVX2Ctx struct {
	Coeff  unsafe.Pointer
	Out    unsafe.Pointer
	Count  int64
	Quant  int64
	Round  int64
	Deq    int64
	ShiftL int64
	ShiftR int64
}

//go:noescape
func quantizeFPAVX2Asm(ctx *quantizeFPAVX2Ctx)

// quantizeFPBlockAVX2 mirrors quantizeFPBlockNEON, with the same guard that
// keeps (32767 + round) * quant inside int32.
func quantizeFPBlockAVX2(qcoeff []int16, coeff []int32, n int, q Quantizer, txScale uint8) bool {
	count := n * n
	if count%16 != 0 || int64(1<<16)/int64(q.AC) > 1<<14 || int64(1<<16)/int64(q.DC) > 1<<14 {
		return false
	}
	roundAC := roundPowerOfTwo((64*int32(q.AC))>>7, txScale)
	ctx := quantizeFPAVX2Ctx{
		Coeff:  unsafe.Pointer(&coeff[0]),
		Out:    unsafe.Pointer(&qcoeff[0]),
		Count:  int64(count),
		Quant:  int64(1<<16) / int64(q.AC),
		Round:  int64(roundAC),
		Deq:    int64(q.AC),
		ShiftL: int64(1 + txScale),
		ShiftR: int64(16 - int(txScale)),
	}
	quantizeFPAVX2Asm(&ctx)
	quantDC := int64(1<<16) / int64(q.DC)
	roundDC := roundPowerOfTwo((64*int32(q.DC))>>7, txScale)
	qcoeff[0] = quantizeScalarFP(coeff[0], q.DC, quantDC, roundDC, txScale)
	return true
}

// init binds the quantizer dispatch slots when the CPU advertises AVX2; the
// direct-call parity tests verify both kernels regardless of the CPUID
// report (Rosetta hides AVX2 there).
func init() {
	if cpu.Detected.AVX2 {
		quantizeBBlockImpl = quantizeBBlockAVX2
		quantizeFPBlockImpl = quantizeFPBlockAVX2
	}
}
