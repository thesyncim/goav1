package encoder

// sad.go hosts the motion-search SAD kernel dispatch. sad8x8Impl computes the
// sum of absolute differences of one 8x8 block; implementations may ignore
// limit (it is an early-exit hint, not a contract), so callers must compare
// the returned total against their own threshold. Every architecture variant
// must be bit-exact with sad8x8PureGo.
var sad8x8Impl = sad8x8PureGo

// sad16x16Impl computes the full 16x16 SAD (no early exit); used by the
// merge-tier searches where 16x16 and 32x32 SADs dominate the decider.
var sad16x16Impl = sad16x16PureGo

// sad32x32Impl computes the full 32x32 SAD; the full-pel search probes this
// shape often enough that avoiding four separate 16x16 dispatches matters.
var sad32x32Impl = sad32x32PureGo

// sad8x8x4Step4Impl computes four 8x8 SADs against reference blocks whose
// x origins are ref+0, ref+4, ref+8, and ref+12. It mirrors the full-pel
// raster search stride, where candidates are separated by four pixels.
var sad8x8x4Step4Impl = sad8x8x4Step4PureGo

// sad8x8x4Impl computes four 8x8 SADs against arbitrary reference origins.
// It mirrors SVT's sad8x8x4d surface for small non-raster candidate groups.
var sad8x8x4Impl = sad8x8x4PureGo

// sad16x16x4Impl computes four 16x16 SADs against arbitrary reference origins.
// It mirrors SVT's sad16x16x4d surface for larger non-raster candidate groups.
var sad16x16x4Impl = sad16x16x4PureGo

// sad16x16x4Step4Impl computes four 16x16 SADs against reference blocks whose
// x origins are ref+0, ref+4, ref+8, and ref+12.
var sad16x16x4Step4Impl = sad16x16x4Step4PureGo

// sad32x32x4Impl computes four 32x32 SADs against arbitrary reference origins.
// It mirrors SVT's sad32x32x4d surface for larger non-raster candidate groups.
var sad32x32x4Impl = sad32x32x4PureGo

// sad32x32x4Step4Impl computes four 32x32 SADs against reference blocks whose
// x origins are ref+0, ref+4, ref+8, and ref+12.
var sad32x32x4Step4Impl = sad32x32x4Step4PureGo

// sad8x8DualImpl computes the 8x8 SAD between planes with different strides
// (the subpel verifier compares the source plane against the n-stride
// prediction scratch).
var sad8x8DualImpl = sad8x8DualPureGo

// sad16x16DualImpl computes the 16x16 SAD between planes with different
// strides. The subpel refinement hot path scores materialized prediction
// scratch this way.
var sad16x16DualImpl = sad16x16DualPureGo

// sad32x32DualImpl computes the 32x32 SAD between planes with different
// strides.
var sad32x32DualImpl = sad32x32DualPureGo

// sad8x8CompoundAvgBlockImpl computes SAD(src, round((ref0+ref1)/2)) for an
// 8x8 block. It is the compound-reference precheck counterpart to sad8x8Dual.
var sad8x8CompoundAvgBlockImpl = sad8x8CompoundAvgBlockPureGo

// sad8x8PureGo is the portable reference with a row-granular early exit.
func sad8x8PureGo(src, ref []byte, stride int, limit int) int {
	total := 0
	for r := range 8 {
		row := r * stride
		for c := range 8 {
			d := int(src[row+c]) - int(ref[row+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
		if total >= limit {
			return total
		}
	}
	return total
}

// sad16x16PureGo is the portable 16x16 reference.
func sad16x16PureGo(src, ref []byte, stride int) int {
	total := 0
	for r := range 16 {
		row := r * stride
		for c := range 16 {
			d := int(src[row+c]) - int(ref[row+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

// sad32x32PureGo is the portable 32x32 reference.
func sad32x32PureGo(src, ref []byte, stride int) int {
	total := 0
	for r := range 32 {
		row := r * stride
		for c := range 32 {
			d := int(src[row+c]) - int(ref[row+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

// sad64x64Composed composes the square 64x64 shape from the active 32x32
// kernel. It is the portable fallback and the benchmark reference for direct
// 64-wide architecture kernels.
func sad64x64Composed(src, ref []byte, stride int) int {
	return sad32x32(src, ref, stride) +
		sad32x32(src[32:], ref[32:], stride) +
		sad32x32(src[32*stride:], ref[32*stride:], stride) +
		sad32x32(src[32*stride+32:], ref[32*stride+32:], stride)
}

// sad64x64DualComposed composes the two-stride 64x64 shape from the active
// 32x32 dual kernel. Architectures can replace this with one leaf kernel to
// avoid four call/ctx round trips in the subpel scorer.
func sad64x64DualComposed(src []byte, srcStride int, ref []byte, refStride int) int {
	return sad32x32Dual(src, srcStride, ref, refStride) +
		sad32x32Dual(src[32:], srcStride, ref[32:], refStride) +
		sad32x32Dual(src[32*srcStride:], srcStride, ref[32*refStride:], refStride) +
		sad32x32Dual(src[32*srcStride+32:], srcStride, ref[32*refStride+32:], refStride)
}

// sad64x64x4Step4 composes four horizontal 64x64 candidates from the active
// 32x32 x4 kernel, matching the full-pel raster search's step-4 candidate
// grouping without adding a separate assembly surface.
func sad64x64x4Step4(src, ref []byte, stride int) (int, int, int, int) {
	s0, s1, s2, s3 := sad32x32x4Step4(src, ref, stride)
	t0, t1, t2, t3 := sad32x32x4Step4(src[32:], ref[32:], stride)
	s0 += t0
	s1 += t1
	s2 += t2
	s3 += t3
	t0, t1, t2, t3 = sad32x32x4Step4(src[32*stride:], ref[32*stride:], stride)
	s0 += t0
	s1 += t1
	s2 += t2
	s3 += t3
	t0, t1, t2, t3 = sad32x32x4Step4(src[32*stride+32:], ref[32*stride+32:], stride)
	s0 += t0
	s1 += t1
	s2 += t2
	s3 += t3
	return s0, s1, s2, s3
}

// sad64x64x4 composes four arbitrary 64x64 candidates from the active 32x32x4
// kernel, mirroring SVT's sad64x64x4d surface without a separate asm body.
func sad64x64x4(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	s0, s1, s2, s3 := sad32x32x4(src, ref0, ref1, ref2, ref3, stride)
	t0, t1, t2, t3 := sad32x32x4(src[32:], ref0[32:], ref1[32:], ref2[32:], ref3[32:], stride)
	s0 += t0
	s1 += t1
	s2 += t2
	s3 += t3
	t0, t1, t2, t3 = sad32x32x4(src[32*stride:], ref0[32*stride:], ref1[32*stride:], ref2[32*stride:], ref3[32*stride:], stride)
	s0 += t0
	s1 += t1
	s2 += t2
	s3 += t3
	t0, t1, t2, t3 = sad32x32x4(src[32*stride+32:], ref0[32*stride+32:], ref1[32*stride+32:], ref2[32*stride+32:], ref3[32*stride+32:], stride)
	s0 += t0
	s1 += t1
	s2 += t2
	s3 += t3
	return s0, s1, s2, s3
}

// sad8x8x4Step4PureGo is the portable reference for four horizontal 8x8 SAD
// candidates spaced by the raster search's four-pixel x step.
func sad8x8x4Step4PureGo(src, ref []byte, stride int) (int, int, int, int) {
	sum0, sum1, sum2, sum3 := 0, 0, 0, 0
	for r := range 8 {
		row := r * stride
		for c := range 8 {
			s := int(src[row+c])

			d0 := s - int(ref[row+c])
			if d0 < 0 {
				d0 = -d0
			}
			sum0 += d0

			d1 := s - int(ref[row+c+4])
			if d1 < 0 {
				d1 = -d1
			}
			sum1 += d1

			d2 := s - int(ref[row+c+8])
			if d2 < 0 {
				d2 = -d2
			}
			sum2 += d2

			d3 := s - int(ref[row+c+12])
			if d3 < 0 {
				d3 = -d3
			}
			sum3 += d3
		}
	}
	return sum0, sum1, sum2, sum3
}

func sad16x16x4Step4PureGo(src, ref []byte, stride int) (int, int, int, int) {
	sum0, sum1, sum2, sum3 := 0, 0, 0, 0
	for r := range 16 {
		row := r * stride
		for c := range 16 {
			s := int(src[row+c])

			d0 := s - int(ref[row+c])
			if d0 < 0 {
				d0 = -d0
			}
			sum0 += d0

			d1 := s - int(ref[row+c+4])
			if d1 < 0 {
				d1 = -d1
			}
			sum1 += d1

			d2 := s - int(ref[row+c+8])
			if d2 < 0 {
				d2 = -d2
			}
			sum2 += d2

			d3 := s - int(ref[row+c+12])
			if d3 < 0 {
				d3 = -d3
			}
			sum3 += d3
		}
	}
	return sum0, sum1, sum2, sum3
}

func sad8x8x4PureGo(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	sum0, sum1, sum2, sum3 := 0, 0, 0, 0
	for r := range 8 {
		row := r * stride
		for c := range 8 {
			s := int(src[row+c])

			d0 := s - int(ref0[row+c])
			if d0 < 0 {
				d0 = -d0
			}
			sum0 += d0

			d1 := s - int(ref1[row+c])
			if d1 < 0 {
				d1 = -d1
			}
			sum1 += d1

			d2 := s - int(ref2[row+c])
			if d2 < 0 {
				d2 = -d2
			}
			sum2 += d2

			d3 := s - int(ref3[row+c])
			if d3 < 0 {
				d3 = -d3
			}
			sum3 += d3
		}
	}
	return sum0, sum1, sum2, sum3
}

func sad16x16x4PureGo(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	sum0, sum1, sum2, sum3 := 0, 0, 0, 0
	for r := range 16 {
		row := r * stride
		for c := range 16 {
			s := int(src[row+c])

			d0 := s - int(ref0[row+c])
			if d0 < 0 {
				d0 = -d0
			}
			sum0 += d0

			d1 := s - int(ref1[row+c])
			if d1 < 0 {
				d1 = -d1
			}
			sum1 += d1

			d2 := s - int(ref2[row+c])
			if d2 < 0 {
				d2 = -d2
			}
			sum2 += d2

			d3 := s - int(ref3[row+c])
			if d3 < 0 {
				d3 = -d3
			}
			sum3 += d3
		}
	}
	return sum0, sum1, sum2, sum3
}

func sad32x32x4Step4PureGo(src, ref []byte, stride int) (int, int, int, int) {
	sum0, sum1, sum2, sum3 := 0, 0, 0, 0
	for r := range 32 {
		row := r * stride
		for c := range 32 {
			s := int(src[row+c])

			d0 := s - int(ref[row+c])
			if d0 < 0 {
				d0 = -d0
			}
			sum0 += d0

			d1 := s - int(ref[row+c+4])
			if d1 < 0 {
				d1 = -d1
			}
			sum1 += d1

			d2 := s - int(ref[row+c+8])
			if d2 < 0 {
				d2 = -d2
			}
			sum2 += d2

			d3 := s - int(ref[row+c+12])
			if d3 < 0 {
				d3 = -d3
			}
			sum3 += d3
		}
	}
	return sum0, sum1, sum2, sum3
}

func sad32x32x4PureGo(src, ref0, ref1, ref2, ref3 []byte, stride int) (int, int, int, int) {
	sum0, sum1, sum2, sum3 := 0, 0, 0, 0
	for r := range 32 {
		row := r * stride
		for c := range 32 {
			s := int(src[row+c])

			d0 := s - int(ref0[row+c])
			if d0 < 0 {
				d0 = -d0
			}
			sum0 += d0

			d1 := s - int(ref1[row+c])
			if d1 < 0 {
				d1 = -d1
			}
			sum1 += d1

			d2 := s - int(ref2[row+c])
			if d2 < 0 {
				d2 = -d2
			}
			sum2 += d2

			d3 := s - int(ref3[row+c])
			if d3 < 0 {
				d3 = -d3
			}
			sum3 += d3
		}
	}
	return sum0, sum1, sum2, sum3
}

// sad8x8DualPureGo is the portable two-stride 8x8 reference.
func sad8x8DualPureGo(src []byte, srcStride int, ref []byte, refStride int) int {
	total := 0
	for r := range 8 {
		srow := r * srcStride
		rrow := r * refStride
		for c := range 8 {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func sad16x16DualPureGo(src []byte, srcStride int, ref []byte, refStride int) int {
	total := 0
	for r := range 16 {
		srow := r * srcStride
		rrow := r * refStride
		for c := range 16 {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func sad32x32DualPureGo(src []byte, srcStride int, ref []byte, refStride int) int {
	total := 0
	for r := range 32 {
		srow := r * srcStride
		rrow := r * refStride
		for c := range 32 {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func sad64x64DualPureGo(src []byte, srcStride int, ref []byte, refStride int) int {
	total := 0
	for r := range 64 {
		srow := r * srcStride
		rrow := r * refStride
		for c := range 64 {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func sad8x8CompoundAvgBlockPureGo(src []byte, srcStride int, ref0 []byte, ref0Stride int, ref1 []byte, ref1Stride int) int {
	total := 0
	for r := range 8 {
		srow := r * srcStride
		r0row := r * ref0Stride
		r1row := r * ref1Stride
		for c := range 8 {
			pred := (int(ref0[r0row+c]) + int(ref1[r1row+c]) + 1) >> 1
			d := int(src[srow+c]) - pred
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}
