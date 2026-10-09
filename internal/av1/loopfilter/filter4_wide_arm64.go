// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package loopfilter

import (
	"simd/archsimd"
	"unsafe"
)

// lfFilter4U8Wide follows the byte-domain sixteen-lane dataflow of the old
// NEON path. The block-limit gate excludes 255, so saturating the byte-domain
// threshold sum cannot turn a rejected lane into an accepted one.
//
//go:nocheckptr
func lfFilter4U8Wide(pix []byte, q0Base, step, length int, params filter4Params, changed *uint) {
	limit := archsimd.BroadcastUint8x16(uint8(params.limit))
	blimit := archsimd.BroadcastUint8x16(uint8(params.blimit))
	hevT := archsimd.BroadcastUint8x16(uint8(params.hev))
	bias := archsimd.BroadcastUint8x16(128)
	three16 := archsimd.BroadcastInt16x8(3)
	three8 := archsimd.BroadcastInt8x16(3)
	four8 := archsimd.BroadcastInt8x16(4)
	one8 := archsimd.BroadcastInt8x16(1)
	shift1 := archsimd.BroadcastInt8x16(-1)
	shift3 := archsimd.BroadcastInt8x16(-3)
	p := unsafe.Pointer(unsafe.SliceData(pix))
	for x := 0; x < length; x += 16 {
		base := q0Base + x
		p1p := (*[16]uint8)(unsafe.Add(p, base-2*step))
		p0p := (*[16]uint8)(unsafe.Add(p, base-step))
		q0p := (*[16]uint8)(unsafe.Add(p, base))
		q1p := (*[16]uint8)(unsafe.Add(p, base+step))
		p1 := archsimd.LoadUint8x16Array(p1p)
		p0 := archsimd.LoadUint8x16Array(p0p)
		q0 := archsimd.LoadUint8x16Array(q0p)
		q1 := archsimd.LoadUint8x16Array(q1p)

		d1 := p1.Max(p0).Sub(p1.Min(p0))
		d2 := q1.Max(q0).Sub(q1.Min(q0))
		d0 := p0.Max(q0).Sub(p0.Min(q0))
		d3 := p1.Max(q1).Sub(p1.Min(q1))
		threshold := d0.AddSaturated(d0).AddSaturated(d3.Shift(shift1))
		need := limit.GreaterEqual(d1.Max(d2)).And(blimit.GreaterEqual(threshold))
		mask := need.ToInt8x16().ToBits().ReshapeToUint64s()
		lo, hi := mask.GetElem(0), mask.GetElem(1)
		if lo|hi == 0 {
			continue
		}
		if changed != nil {
			if lo != 0 {
				*changed |= 1 << uint(x/8)
			}
			if hi != 0 {
				*changed |= 1 << uint(x/8+1)
			}
		}
		hev := d1.Max(d2).Greater(hevT)

		p1c := p1.Xor(bias).BitsToInt8()
		p0c := p0.Xor(bias).BitsToInt8()
		q0c := q0.Xor(bias).BitsToInt8()
		q1c := q1.Xor(bias).BitsToInt8()
		outerF := p1c.SubSaturated(q1c).Masked(hev)
		deltaLo := q0.ExtendLo8ToUint16().BitsToInt16().Sub(p0.ExtendLo8ToUint16().BitsToInt16())
		deltaHi := q0.HiToLo().ExtendLo8ToUint16().BitsToInt16().Sub(p0.HiToLo().ExtendLo8ToUint16().BitsToInt16())
		fLo := deltaLo.Mul(three16).Add(outerF.ExtendLo8ToInt16()).SaturateToInt8()
		fHi := deltaHi.Mul(three16).Add(outerF.HiToLo().ExtendLo8ToInt16()).SaturateToInt8()
		fHiBits := fHi.ToBits()
		f := fLo.ToBits().Or(fHiBits.ConcatShiftBytesRight(fHiBits, 8)).BitsToInt8()
		filter1 := f.AddSaturated(four8).Shift(shift3)
		filter2 := f.AddSaturated(three8).Shift(shift3)
		outer := filter1.Add(one8).Shift(shift1)

		np0 := p0c.AddSaturated(filter2).ToBits().Xor(bias).IfElse(need, p0)
		nq0 := q0c.SubSaturated(filter1).ToBits().Xor(bias).IfElse(need, q0)
		np1 := p1c.AddSaturated(outer).ToBits().Xor(bias).IfElse(need.And(hev.Not()), p1)
		nq1 := q1c.SubSaturated(outer).ToBits().Xor(bias).IfElse(need.And(hev.Not()), q1)
		np1.StoreArray(p1p)
		np0.StoreArray(p0p)
		nq0.StoreArray(q0p)
		nq1.StoreArray(q1p)
	}
}
