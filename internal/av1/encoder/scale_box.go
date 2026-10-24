package encoder

// scale_box.go is the spatial-layer source downscaler: libyuv's
// ScalePlane(..., kFilterBox), the filter libwebrtc's simulcast path uses to
// make each lower layer's picture (I420Buffer::ScaleFrom -> I420Scale with
// kFilterBox). Reference: third_party/upstream/libaom/third_party/libyuv
// source/scale.cc ScalePlane / ScalePlane_16 and scale_common.cc.
//
// ScaleFilterReduce turns Box into Bilinear when both axes scale by 1/2 or
// more, so:
//   - an exact 1/2 step is ScalePlaneDown2 with ScaleRowDown2Box, the
//     (a+b+c+d+2)>>2 average. That is svt_aom_downsample_2d's rounding, so the
//     8-bit step runs svtDownsample2DArch and its output is the next layer's
//     HME half-resolution pyramid plane, sample for sample (svt_hme.go);
//   - a 2/3 step, and an odd-sized chroma 1/2 step, are ScalePlaneBilinearDown;
//   - a step below 1/2 (an odd luma size halved) is ScalePlaneBox.
//
// The 3/4 and 3/8 kernels (ScalePlaneDown34/38) are not ported: no layer
// geometry reaches them. Those ratios take ScalePlaneBilinearDown. BLENDER is
// the ARM (16-bit fraction) variant on every architecture so the encoded
// bytes do not depend on the host. The 8-bit bilinear column pass runs a
// table-driven NEON kernel on arm64 simd builds (scale_box_arm64.go); every
// other build, and 16-bit planes, run boxFilterCols.

type boxScaleSample interface{ ~uint8 | ~uint16 }

// boxScaleScratch holds the row buffers of one plane scale, kept across
// pictures so the steady state allocates nothing.
type boxScaleScratch struct {
	sum  []uint32
	row8 []uint8
	row  []uint16
	cols boxColsCache
}

func (s *boxScaleScratch) sums(n int) []uint32 {
	if cap(s.sum) < n {
		s.sum = make([]uint32, n)
	}
	return s.sum[:n]
}

// scalePlaneBox8 is libyuv ScalePlane with kFilterBox for a downscale.
func scalePlaneBox8(dst []uint8, dstStride, dw, dh int, src []uint8, srcStride, sw, sh int, scratch *boxScaleScratch) {
	switch {
	case dw == sw && dh == sh:
		for y := 0; y < dh; y++ {
			copy(dst[y*dstStride:y*dstStride+dw], src[y*srcStride:y*srcStride+sw])
		}
	case 2*dw == sw && 2*dh == sh:
		svtDownsample2DArch(dst, dstStride, src, srcStride, dw, dh)
	default:
		if cap(scratch.row8) < sw {
			scratch.row8 = make([]uint8, sw)
		}
		scalePlaneBoxAny(dst, dstStride, dw, dh, src, srcStride, sw, sh, scratch, scratch.row8[:sw], 128)
	}
}

// scalePlaneBox16 is libyuv ScalePlane_16 with kFilterBox for a downscale.
func scalePlaneBox16(dst []uint16, dstStride, dw, dh int, src []uint16, srcStride, sw, sh int, scratch *boxScaleScratch) {
	switch {
	case dw == sw && dh == sh:
		for y := 0; y < dh; y++ {
			copy(dst[y*dstStride:y*dstStride+dw], src[y*srcStride:y*srcStride+sw])
		}
	case 2*dw == sw && 2*dh == sh:
		// ScaleRowDown2Box_16_C.
		for y := 0; y < dh; y++ {
			r0 := src[2*y*srcStride:]
			r1 := src[(2*y+1)*srcStride:]
			d := dst[y*dstStride : y*dstStride+dw]
			for x := range d {
				d[x] = uint16((uint32(r0[2*x]) + uint32(r0[2*x+1]) + uint32(r1[2*x]) + uint32(r1[2*x+1]) + 2) >> 2)
			}
		}
	default:
		if cap(scratch.row) < sw {
			scratch.row = make([]uint16, sw)
		}
		// InterpolateRow_16_C does not round.
		scalePlaneBoxAny(dst, dstStride, dw, dh, src, srcStride, sw, sh, scratch, scratch.row[:sw], 0)
	}
}

// libyuv FilterMode after ScaleFilterReduce.
const (
	boxFilterNone = iota
	boxFilterLinear
	boxFilterBilinear
	boxFilterBox
)

// boxScaleFilter is ScaleFilterReduce for a kFilterBox request.
func boxScaleFilter(sw, sh, dw, dh int) int {
	f := boxFilterBox
	if dw*2 >= sw && dh*2 >= sh {
		f = boxFilterBilinear
	}
	if f == boxFilterBilinear {
		if sh == 1 || dh == sh || dh*3 == sh {
			f = boxFilterLinear
		}
		if sw == 1 {
			f = boxFilterNone
		}
	}
	if f == boxFilterLinear && (sw == 1 || dw == sw || dw*3 == sw) {
		f = boxFilterNone
	}
	return f
}

// boxFixedDiv is FixedDiv_C: num / div in 16.16 fixed point.
func boxFixedDiv(num, div int) int { return int((int64(num) << 16) / int64(div)) }

// boxScaleSlope is ScaleSlope for a downscale (dst <= src on both axes).
func boxScaleSlope(sw, sh, dw, dh, filter int) (x, y, dx, dy int) {
	if dw == 1 && sw >= 32768 {
		dw = sw
	}
	if dh == 1 && sh >= 32768 {
		dh = sh
	}
	dx = boxFixedDiv(sw, dw)
	dy = boxFixedDiv(sh, dh)
	switch filter {
	case boxFilterBox:
	case boxFilterBilinear:
		x = (dx >> 1) - 32768
		y = (dy >> 1) - 32768
	case boxFilterLinear:
		x = (dx >> 1) - 32768
		y = dy >> 1
	default:
		x = dx >> 1
		y = dy >> 1
	}
	return x, y, dx, dy
}

// scalePlaneBoxAny is the part of ScalePlane past the exact 1/2 kernel:
// ScalePlaneBox, ScalePlaneBilinearDown or ScalePlaneSimple. row is an
// sw-sample row buffer; rnd is InterpolateRow's rounding (128 for 8-bit, 0
// for 16-bit).
func scalePlaneBoxAny[T boxScaleSample](dst []T, dstStride, dw, dh int, src []T, srcStride, sw, sh int, scratch *boxScaleScratch, row []T, rnd int) {
	filter := boxScaleFilter(sw, sh, dw, dh)
	x, y, dx, dy := boxScaleSlope(sw, sh, dw, dh, filter)
	switch {
	case filter == boxFilterBox && dh*2 < sh:
		scalePlaneBoxFilter(dst, dstStride, dw, dh, src, srcStride, sw, sh, x, y, dx, dy, scratch.sums(sw))
	case filter == boxFilterNone:
		// ScalePlaneSimple / ScaleCols_C.
		for j := 0; j < dh; j++ {
			s := src[(y>>16)*srcStride:]
			d := dst[j*dstStride : j*dstStride+dw]
			xx := x
			for i := range d {
				d[i] = s[xx>>16]
				xx += dx
			}
			y += dy
		}
	default:
		// ScalePlaneBilinearDown (also the 3/4 and 3/8 ratios, see above).
		maxY := (sh - 1) << 16
		y = min(y, maxY)
		var plan *boxColsPlan
		dst8, is8 := any(dst).([]uint8)
		if is8 {
			plan = scratch.cols.plan(sw, dw, x, dx, dh)
		}
		for j := 0; j < dh; j++ {
			yi := y >> 16
			s := src[yi*srcStride : yi*srcStride+sw]
			if filter == boxFilterBilinear {
				if yf := (y >> 8) & 255; yf != 0 {
					s1 := src[(yi+1)*srcStride : (yi+1)*srcStride+sw]
					if r8, ok := any(row).([]uint8); ok {
						boxInterpolateRow8(r8, any(s).([]uint8), any(s1).([]uint8), yf)
					} else {
						boxInterpolateRow(row, s, s1, yf, rnd)
					}
					s = row
				}
			}
			if plan != nil {
				plan.filter(dst8[j*dstStride:j*dstStride+dw], any(s).([]uint8))
			} else {
				boxFilterCols(dst[j*dstStride:j*dstStride+dw], s, x, dx)
			}
			y = min(y+dy, maxY)
		}
	}
}

// boxInterpolateRow is InterpolateRow_C / InterpolateRow_16_C for a non-zero
// fraction (HalfRow at 128).
func boxInterpolateRow[T boxScaleSample](dst, s0, s1 []T, f, rnd int) {
	s1 = s1[:len(s0)]
	dst = dst[:len(s0)]
	if f == 128 {
		for i := range dst {
			dst[i] = T((int(s0[i]) + int(s1[i]) + 1) >> 1)
		}
		return
	}
	f0 := 256 - f
	for i := range dst {
		dst[i] = T((int(s0[i])*f0 + int(s1[i])*f + rnd) >> 8)
	}
}

// boxFilterCols is ScaleFilterCols_C with the ARM BLENDER,
// a + ((f * (b - a) + 0x8000) >> 16). The right tap is clamped to the row:
// libyuv reads one sample past it, which a downscale never weights.
func boxFilterCols[T boxScaleSample](dst, s []T, x, dx int) {
	if dx == 0x18000 && x == 0x4000 && 3*(len(dst)/2) <= len(s) {
		// The 2/3 step: taps repeat every two outputs, fractions 1/4, 3/4.
		n := len(dst) / 2
		d, s3 := dst[:2*n], s[:3*n]
		for m := 0; m < n; m++ {
			a, b, c := int(s3[3*m]), int(s3[3*m+1]), int(s3[3*m+2])
			d[2*m] = T(a + ((0x4000*(b-a) + 0x8000) >> 16))
			d[2*m+1] = T(b + ((0xc000*(c-b) + 0x8000) >> 16))
		}
		if len(dst) == 2*n {
			return
		}
		dst, x = dst[2*n:], x+2*n*dx
	}
	last := len(s) - 1
	for i := range dst {
		xi := x >> 16
		a := int(s[xi])
		b := int(s[min(xi+1, last)])
		dst[i] = T(a + (((x&0xffff)*(b-a) + 0x8000) >> 16))
		x += dx
	}
}

// scalePlaneBoxFilter is ScalePlaneBox: each output sample is the truncated
// average of its 16.16 box (ScaleAddRow + ScaleAddCols2_C; the Cols0/Cols1
// specializations compute the same values).
func scalePlaneBoxFilter[T boxScaleSample](dst []T, dstStride, dw, dh int, src []T, srcStride, sw, sh, x, y, dx, dy int, sum []uint32) {
	maxY := sh << 16
	minBoxW := dx >> 16
	for j := 0; j < dh; j++ {
		iy := y >> 16
		y = min(y+dy, maxY)
		boxH := max((y>>16)-iy, 1)
		clear(sum)
		for k := 0; k < boxH; k++ {
			s := src[(iy+k)*srcStride : (iy+k)*srcStride+sw]
			for i, v := range s {
				sum[i] += uint32(v)
			}
		}
		scale := [2]int{65536 / (max(minBoxW, 1) * boxH), 65536 / (max(minBoxW+1, 1) * boxH)}
		d := dst[j*dstStride : j*dstStride+dw]
		xx := x
		for i := range d {
			ix := xx >> 16
			xx += dx
			boxW := max((xx>>16)-ix, 1)
			var t uint32
			for _, v := range sum[ix : ix+boxW] {
				t += v
			}
			d[i] = T(int(t) * scale[boxW-minBoxW] >> 16)
		}
	}
}
