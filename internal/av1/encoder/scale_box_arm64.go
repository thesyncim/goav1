//go:build goexperiment.simd && arm64 && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

// boxInterpolateRow8 is InterpolateRow_C for 8-bit samples on NEON, as
// libyuv's InterpolateRow_NEON: (s0*(256-f) + s1*f + 128) >> 8 by widening
// multiply-accumulate, and the f == 128 HalfRow as a rounding average.
func boxInterpolateRow8(dst, s0, s1 []uint8, f int) {
	n := len(s0)
	s1, dst = s1[:n], dst[:n]
	x := 0
	if n >= 16 {
		p0, p1 := unsafe.Pointer(&s0[0]), unsafe.Pointer(&s1[0])
		pd := unsafe.Pointer(&dst[0])
		if f == 128 {
			for ; x+16 <= n; x += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p0, x)))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p1, x)))
				a.Average(b).StoreArray((*[16]uint8)(unsafe.Add(pd, x)))
			}
		} else {
			f0 := archsimd.BroadcastUint8x16(uint8(256 - f))
			f1 := archsimd.BroadcastUint8x16(uint8(f))
			bias := archsimd.BroadcastUint16x8(128)
			sh8 := archsimd.BroadcastInt16x8(-8)
			for ; x+16 <= n; x += 16 {
				a := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p0, x)))
				b := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(p1, x)))
				lo := a.MulWidenLo(f0).Add(b.MulWidenLo(f1)).Add(bias).Shift(sh8).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
				hi := a.HiToLo().MulWidenLo(f0.HiToLo()).Add(b.HiToLo().MulWidenLo(f1.HiToLo())).Add(bias).Shift(sh8).SaturateToUint8().ReshapeToUint64s().BitsToFloat64()
				lo.SetElem(1, hi.GetElem(0)).ToBits().ReshapeToUint8s().StoreArray((*[16]uint8)(unsafe.Add(pd, x)))
			}
		}
	}
	if x < n {
		boxInterpolateRow(dst[x:], s0[x:], s1[x:], f, 128)
	}
}

// libyuv's ScaleFilterCols_NEON gathers each output's two taps with ld2 lane
// loads and blends in 32-bit lanes. archsimd has no lane load, so this port
// takes the taps from TBL against the 32-byte source window a 16-output chunk
// can reach (the bilinear step is at most 2: 15 outputs advance at most 30
// samples). The gather indices and the 16-bit blend weights depend on the
// plane geometry only, never on the row, so they are built once per
// (sw, dw, x, dx) and every row of every plane with that geometry runs the
// same table.
//
// The blend is ScaleFilterCols_C's ARM BLENDER,
//
//	a + ((f*(b-a) + 0x8000) >> 16) == (g*a + f*b + 0x8000) >> 16, g = 65536-f,
//
// split into bytes f = 256*fh + fl, g = 256*gh + gl with gh = 255-fh and
// gl = 256-fl so every partial sum fits a 16-bit lane. With
// U = gh*a + fh*b and V = gl*a + fl*b the nested floor division gives
// (U + 128 + (V >> 8)) >> 8, at most 65408.
const (
	boxColsChunk   = 16  // outputs per chunk
	boxColsWindow  = 32  // source bytes a chunk reads
	boxColsEntry   = 192 // table bytes per chunk
	boxColsSlots   = 4   // cached geometries
	boxColsMinRows = 8   // rows that pay for building a table
	boxColsCoefOff = 64  // offset of the weights; the indices come first
)

type boxColsPos struct{ base, out int32 }

// boxColsPlan is the column table of one plane geometry. Per chunk it holds
// four 16-byte TBL index vectors (a and b taps against the window's low and
// high 16 bytes) and, for each 8-output half, the u16 weights gh, fh,
// gl = 256-fl and fl.
type boxColsPlan struct {
	sw, dw, x, dx int
	valid         bool
	pos           []boxColsPos
	tab           []uint8
}

// boxColsCache keeps the plans of the last few plane geometries; the U and V
// planes of a layer share one, so steady state builds nothing.
type boxColsCache struct {
	plans [boxColsSlots]boxColsPlan
	next  int
}

// plan returns the table for a column pass over an sw-sample row producing dw
// samples from x in steps of dx, or nil when the scalar filter must run: the
// row or output is shorter than one chunk, the step is outside [1, 2] (the
// tap window would not cover a chunk), or rows is too few to pay for a build.
func (c *boxColsCache) plan(sw, dw, x, dx, rows int) *boxColsPlan {
	if sw < boxColsWindow || dw < boxColsChunk || x < 0 || dx < 1<<16 || dx > 2<<16 || (x+(dw-1)*dx)>>16 >= sw {
		return nil
	}
	for i := range c.plans {
		if p := &c.plans[i]; p.valid && p.sw == sw && p.dw == dw && p.x == x && p.dx == dx {
			return p
		}
	}
	if rows < boxColsMinRows {
		return nil
	}
	p := &c.plans[c.next]
	c.next = (c.next + 1) % boxColsSlots
	p.build(sw, dw, x, dx)
	return p
}

func (p *boxColsPlan) build(sw, dw, x, dx int) {
	n := (dw + boxColsChunk - 1) / boxColsChunk
	if cap(p.pos) < n {
		p.pos = make([]boxColsPos, n)
		p.tab = make([]uint8, n*boxColsEntry)
	}
	p.pos, p.tab = p.pos[:n], p.tab[:n*boxColsEntry]
	p.sw, p.dw, p.x, p.dx, p.valid = sw, dw, x, dx, true
	for c := range p.pos {
		// The last chunk is pulled back to end at dw; it rewrites some of the
		// previous chunk's outputs with the same values.
		out := min(c*boxColsChunk, dw-boxColsChunk)
		xc := x + out*dx
		base := min(xc>>16, sw-boxColsWindow)
		p.pos[c] = boxColsPos{base: int32(base), out: int32(out)}
		t := p.tab[c*boxColsEntry : (c+1)*boxColsEntry]
		for i := 0; i < boxColsChunk; i++ {
			xx := xc + i*dx
			xi := xx >> 16
			f := xx & 0xffff
			ia := xi - base
			ib := min(xi+1, sw-1) - base // the right tap clamps to the row
			t[i], t[16+i] = uint8(ia), uint8(ia-16)
			t[32+i], t[48+i] = uint8(ib), uint8(ib-16)
			fh, fl := f>>8, f&255
			w := t[boxColsCoefOff+(i>>3)*64+(i&7)*2:]
			w[0] = uint8(255 - fh) // gh
			w[16] = uint8(fh)
			w[32], w[33] = uint8(256-fl), uint8((256-fl)>>8) // gl is 256 at fl == 0
			w[48] = uint8(fl)
		}
	}
}

// boxColsBlend is the blend of one 8-output half; q points at its weights.
func boxColsBlend(a, b archsimd.Uint16x8, q unsafe.Pointer, k128 archsimd.Uint16x8, sh8 archsimd.Int16x8) archsimd.Uint16x8 {
	gh := archsimd.LoadUint16x8Array((*[8]uint16)(q))
	fh := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(q, 16)))
	gl := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(q, 32)))
	fl := archsimd.LoadUint16x8Array((*[8]uint16)(unsafe.Add(q, 48)))
	u := fh.MulAdd(b, gh.MulAdd(a, k128))
	v := fl.MulAdd(b, gl.Mul(a))
	return u.Add(v.Shift(sh8))
}

// filter is boxFilterCols for one row.
func (p *boxColsPlan) filter(dst, s []uint8) {
	dst, s = dst[:p.dw], s[:p.sw]
	ps, pd, pt := unsafe.Pointer(&s[0]), unsafe.Pointer(&dst[0]), unsafe.Pointer(&p.tab[0])
	zero := archsimd.BroadcastUint8x16(0)
	k128 := archsimd.BroadcastUint16x8(128)
	sh8 := archsimd.BroadcastInt16x8(-8)
	for c, pos := range p.pos {
		t := unsafe.Add(pt, c*boxColsEntry)
		w := unsafe.Add(ps, pos.base)
		s0 := archsimd.LoadUint8x16Array((*[16]uint8)(w))
		s1 := archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(w, 16)))
		a := s0.LookupOrZero(archsimd.LoadUint8x16Array((*[16]uint8)(t))).
			Or(s1.LookupOrZero(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(t, 16)))))
		b := s0.LookupOrZero(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(t, 32)))).
			Or(s1.LookupOrZero(archsimd.LoadUint8x16Array((*[16]uint8)(unsafe.Add(t, 48)))))
		lo := boxColsBlend(a.InterleaveLo(zero).ReshapeToUint16s(), b.InterleaveLo(zero).ReshapeToUint16s(), unsafe.Add(t, boxColsCoefOff), k128, sh8)
		hi := boxColsBlend(a.InterleaveHi(zero).ReshapeToUint16s(), b.InterleaveHi(zero).ReshapeToUint16s(), unsafe.Add(t, boxColsCoefOff+64), k128, sh8)
		lo.ReshapeToUint8s().ConcatOdd(hi.ReshapeToUint8s()).StoreArray((*[16]uint8)(unsafe.Add(pd, pos.out)))
	}
}
