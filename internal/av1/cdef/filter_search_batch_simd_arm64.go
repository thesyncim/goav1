// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package cdef

import "simd/archsimd"

// searchBlocksU8 binds the batched search kernel on arm64.
func searchBlocksU8(src []byte, srcStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, damping int, trials []SearchTrial, rowStep int, sse []uint64) bool {
	return searchBlocksU8SIMD(src, srcStride, input, inputOrigin, blocks, directions, variances, damping, trials, rowStep, sse)
}

// cdefSearchTrialSIMD is one trial's kernel state for the current block: the
// broadcast strengths and shifts of cdefFilterBlock8SIMD plus which tap sets
// the trial reads.
type cdefSearchTrialSIMD struct {
	priStr  archsimd.Uint16x8
	secStr  archsimd.Uint16x8
	priSh   cdefShiftCount
	secSh   cdefShiftCount
	priTap0 archsimd.Int16x8
	priTap1 archsimd.Int16x8
	priOn   bool
	secOn   bool
	// secAtDir is set when the secondary taps follow the block direction
	// (trial primary strength non-zero); otherwise they use direction 0.
	secAtDir bool
}

// cdefLoadSrc8 widens eight source bytes to int16 lanes.
func cdefLoadSrc8(b *[8]byte) archsimd.Int16x8 {
	v := uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
	rows := [2]uint64{v, 0}
	return archsimd.LoadUint64x2Array(&rows).ReshapeToUint8s().ExtendLo8ToUint16().BitsToInt16()
}

// cdefSearchTaps returns the window indices cdefWinReach +- t of the tap
// offsets a and b, checked against the window.
func cdefSearchTaps(a, b int16) (uint, uint, uint, uint) {
	ap, am := uint(cdefWinReach+int(a)), uint(cdefWinReach-int(a))
	bp, bm := uint(cdefWinReach+int(b)), uint(cdefWinReach-int(b))
	if ap > cdefTapMax || am > cdefTapMax || bp > cdefTapMax || bm > cdefTapMax {
		panic(errCDEFWindow)
	}
	return ap, am, bp, bm
}

// cdefSquareSum widens the eight int16 errors to their int32 squares.
func cdefSquareSum(e archsimd.Int16x8) archsimd.Int32x4 {
	return e.MulWidenLo(e).Add(e.HiToLo().MulWidenLo(e.HiToLo()))
}

// searchBlocksU8SIMD is the per-trial search with one kernel entry per
// block: each measured row loads the center and tap samples once, derives the
// clamp bounds once, and then loops the trials over the register-resident
// taps, accumulating each trial's squared error against the source in int32
// lanes (at most 64 blocks of 64 samples at 255^2 fits). The per-row
// arithmetic of a trial is cdefFilterBlock8SIMD's, so every filtered row is
// bit-identical to the per-trial path's.
func searchBlocksU8SIMD(src []byte, srcStride int, input []uint16, inputOrigin int, blocks []BlockPosition, directions *DirectionGrid, variances *VarianceGrid, damping int, trials []SearchTrial, rowStep int, sse []uint64) bool {
	n := len(trials)
	if n == 0 || n > MaxSearchTrials || len(sse) < n {
		return false
	}
	if len(blocks) == 0 {
		for t := range sse[:n] {
			sse[t] = 0
		}
		return true
	}
	// Bounds: block rows 0..7 of the last listed block in src, and the
	// two-sample halo around every block in input.
	maxSrc, maxBY, maxBX := 0, 0, 0
	for _, b := range blocks {
		maxSrc = max(maxSrc, (int(b.BY)*8+7)*srcStride+int(b.BX)*8+7)
		maxBY = max(maxBY, int(b.BY))
		maxBX = max(maxBX, int(b.BX))
	}
	// The origin and stride bounds also make each row window below one
	// compare.
	if inputOrigin < 2*BStride+2 || inputOrigin > 1<<24 || uint(srcStride) > 1<<20 ||
		maxBY >= 8 || maxBX >= 8 || maxSrc >= len(src) ||
		inputOrigin+(maxBY*8+7+2)*BStride+maxBX*8+7+2 >= len(input) {
		return false
	}
	_ = src[maxSrc]
	_ = input[inputOrigin+(maxBY*8+7+2)*BStride+maxBX*8+7+2]
	_ = sse[n-1]

	var st [MaxSearchTrials]cdefSearchTrialSIMD
	var acc [MaxSearchTrials]archsimd.Int32x4
	zero := archsimd.BroadcastInt16x8(0)
	secTap0 := archsimd.BroadcastInt16x8(int16(cdefSecondaryTaps[0]))
	secTap1 := archsimd.BroadcastInt16x8(int16(cdefSecondaryTaps[1]))
	eight := archsimd.BroadcastInt16x8(8)
	noSentinel := archsimd.BroadcastInt16x8(int16(^VeryLarge))
	sar15 := cdefShiftCountOf(15)
	sar4 := cdefShiftCountOf(4)
	for t := range trials {
		s := &st[t]
		sec := int(trials[t].Secondary)
		s.secStr = archsimd.BroadcastInt16x8(int16(sec)).ToBits()
		s.secSh = cdefShiftCountOf(constrainShift(sec, damping))
		s.secOn = sec != 0
		s.secAtDir = trials[t].Primary != 0
		acc[t] = archsimd.BroadcastInt32x4(0)
	}
	// Direction-0 secondary tap window indices (trials with no primary).
	z0p, z0m, z2p, z2m := cdefSearchTaps(cdefDirections[4][0], cdefDirections[4][1])
	z1p, z1m, z3p, z3m := cdefSearchTaps(cdefDirections[0][0], cdefDirections[0][1])
	for _, b := range blocks {
		by, bx := int(b.BY), int(b.BX)
		dir := int(directions[by][bx])
		vari := variances[by][bx]
		p0p, p0m, p1p, p1m := cdefSearchTaps(cdefDirections[dir+2][0], cdefDirections[dir+2][1])
		s0p, s0m, s2p, s2m := cdefSearchTaps(cdefDirections[dir+4][0], cdefDirections[dir+4][1])
		s1p, s1m, s3p, s3m := cdefSearchTaps(cdefDirections[dir][0], cdefDirections[dir][1])
		needDir, needClip, needZero, needIdentity := false, false, false, false
		for t := range trials {
			s := &st[t]
			strength := 0
			if primary := int(trials[t].Primary); primary != 0 {
				strength = adjustStrength(primary, vari)
			}
			s.priOn = strength != 0
			if s.priOn {
				taps := cdefPrimaryTaps[strength&1]
				s.priTap0 = archsimd.BroadcastInt16x8(int16(taps[0]))
				s.priTap1 = archsimd.BroadcastInt16x8(int16(taps[1]))
				s.priStr = archsimd.BroadcastInt16x8(int16(strength)).ToBits()
				s.priSh = cdefShiftCountOf(constrainShift(strength, damping))
			}
			switch {
			case !s.priOn && !s.secOn:
				needIdentity = true
			case s.priOn && s.secOn:
				needDir, needClip = true, true
			case s.priOn || s.secAtDir:
				needDir = true
			default:
				needZero = true
			}
		}
		blockIn := inputOrigin + by*8*BStride + bx*8 - cdefWinReach
		blockSrc := by*8*srcStride + bx*8
		for i := 0; i < 8; i += rowStep {
			// i & 7 is i (the loop keeps it in 0..7); the mask gives the row
			// offsets a range the compiler can use with a variable step.
			r := i & 7
			wo, so := blockIn+r*BStride, blockSrc+r*srcStride
			w := (*[cdefWin1]uint16)(input[wo : wo+cdefWin1])
			x := cdefLdW1(w, cdefWinReach)
			srow := cdefLoadSrc8((*[8]byte)(src[so : so+8]))
			var sqIdentity archsimd.Int32x4
			if needIdentity {
				sqIdentity = cdefSquareSum(x.Sub(srow))
			}
			var t0, t1, t2, t3, s0, s1, s2, s3, s4, s5, s6, s7 archsimd.Int16x8
			mx, mn := x, x
			if needDir {
				t0 = cdefLdW1(w, p0p)
				t1 = cdefLdW1(w, p0m)
				t2 = cdefLdW1(w, p1p)
				t3 = cdefLdW1(w, p1m)
				s0 = cdefLdW1(w, s0p)
				s1 = cdefLdW1(w, s0m)
				s2 = cdefLdW1(w, s1p)
				s3 = cdefLdW1(w, s1m)
				s4 = cdefLdW1(w, s2p)
				s5 = cdefLdW1(w, s2m)
				s6 = cdefLdW1(w, s3p)
				s7 = cdefLdW1(w, s3m)
				if needClip {
					mx = mx.Max(t0.And(noSentinel)).Max(t1.And(noSentinel)).Max(t2.And(noSentinel)).Max(t3.And(noSentinel))
					mn = mn.Min(t0).Min(t1).Min(t2).Min(t3)
					mx = mx.Max(s0.And(noSentinel)).Max(s1.And(noSentinel)).Max(s2.And(noSentinel)).Max(s3.And(noSentinel))
					mn = mn.Min(s0).Min(s1).Min(s2).Min(s3)
					mx = mx.Max(s4.And(noSentinel)).Max(s5.And(noSentinel)).Max(s6.And(noSentinel)).Max(s7.And(noSentinel))
					mn = mn.Min(s4).Min(s5).Min(s6).Min(s7)
				}
			}
			var u0, u1, u2, u3, u4, u5, u6, u7 archsimd.Int16x8
			if needZero {
				u0 = cdefLdW1(w, z0p)
				u1 = cdefLdW1(w, z0m)
				u2 = cdefLdW1(w, z1p)
				u3 = cdefLdW1(w, z1m)
				u4 = cdefLdW1(w, z2p)
				u5 = cdefLdW1(w, z2m)
				u6 = cdefLdW1(w, z3p)
				u7 = cdefLdW1(w, z3m)
			}
			for t := 0; t < n; t++ {
				s := &st[t]
				if !s.priOn && !s.secOn {
					acc[t] = acc[t].Add(sqIdentity)
					continue
				}
				sum := zero
				if s.priOn {
					priStr, priSh := s.priStr, s.priSh
					sum = cdefConstrain(t0, x, priStr, priSh).Add(cdefConstrain(t1, x, priStr, priSh)).Mul(s.priTap0).
						Add(cdefConstrain(t2, x, priStr, priSh).Add(cdefConstrain(t3, x, priStr, priSh)).Mul(s.priTap1))
				}
				if s.secOn {
					secStr, secSh := s.secStr, s.secSh
					if s.secAtDir {
						sum = sum.Add(cdefConstrain(s0, x, secStr, secSh).Add(cdefConstrain(s1, x, secStr, secSh)).
							Add(cdefConstrain(s2, x, secStr, secSh)).Add(cdefConstrain(s3, x, secStr, secSh)).Mul(secTap0))
						sum = sum.Add(cdefConstrain(s4, x, secStr, secSh).Add(cdefConstrain(s5, x, secStr, secSh)).
							Add(cdefConstrain(s6, x, secStr, secSh)).Add(cdefConstrain(s7, x, secStr, secSh)).Mul(secTap1))
					} else {
						sum = sum.Add(cdefConstrain(u0, x, secStr, secSh).Add(cdefConstrain(u1, x, secStr, secSh)).
							Add(cdefConstrain(u2, x, secStr, secSh)).Add(cdefConstrain(u3, x, secStr, secSh)).Mul(secTap0))
						sum = sum.Add(cdefConstrain(u4, x, secStr, secSh).Add(cdefConstrain(u5, x, secStr, secSh)).
							Add(cdefConstrain(u6, x, secStr, secSh)).Add(cdefConstrain(u7, x, secStr, secSh)).Mul(secTap1))
					}
				}
				// y = x + ((8 + sum - (sum<0)) >> 4)
				y := x.Add(cdefSarI16(sum.Add(cdefSarI16(sum, sar15)).Add(eight), sar4))
				if s.priOn && s.secOn {
					y = y.Max(mn).Min(mx)
				}
				acc[t] = acc[t].Add(cdefSquareSum(y.Sub(srow)))
			}
		}
	}
	for t := range trials {
		sse[t] = uint64(uint32(acc[t].ReduceSum()))
	}
	return true
}
