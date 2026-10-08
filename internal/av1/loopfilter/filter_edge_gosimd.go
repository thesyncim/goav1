// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && (arm64 || amd64) && !purego

package loopfilter

// lfBatchGroups is the number of eight-position groups a vertical edge gathers
// into scratch per pass. The scratch row stride is 8*lfBatchGroups samples.
const lfBatchGroups = 4

// lfMaxTaps is the largest tap window (filter14: p6..q6) a kernel gathers.
const lfMaxTaps = 14

// lfKind selects a deblocking filter family. The kernels are chosen by a switch
// over direct generic calls rather than by function values: a function value
// would make the vertical scratch escape to the heap.
type lfKind uint8

const (
	lfKind4 lfKind = iota
	lfKind6
	lfKind8
	lfKind14
)

// lfSpec returns the window geometry of a filter family: before is the number
// of taps preceding q0, taps is the window size, and rows [t0, t1) are the
// window rows a kernel may modify (written back after a vertical batch).
func lfSpec(k lfKind) (before, taps, t0, t1 int) {
	switch k {
	case lfKind4:
		return 2, 4, 0, 4
	case lfKind6:
		return 3, 6, 1, 5
	case lfKind8:
		return 4, 8, 1, 7
	}
	return 7, 14, 1, 13
}

// lfCore runs the horizontal Go-native SIMD kernel of family k over length
// positions (a multiple of eight). Positions are contiguous samples of type S
// and step is the byte distance between adjacent taps.
func lfCore[S lfSample](k lfKind, pix []byte, q0Base int, step int, length int, scale int, params filter4Params) uint {
	switch k {
	case lfKind4:
		return lfFilter4Core[S](pix, q0Base, step, length, scale, params)
	case lfKind6:
		return lfFilter6Core[S](pix, q0Base, step, length, scale, params)
	case lfKind8:
		return lfFilter8Core[S](pix, q0Base, step, length, scale, params)
	default:
		return lfFilter14Core[S](pix, q0Base, step, length, scale, params)
	}
}

// lfPure is the scalar reference of family k for the sub-group tail and for
// layouts the SIMD path does not cover. S selects the 8-bit or 10/12-bit form.
func lfPure[S lfSample](k lfKind, pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	if lfSize[S]() == 1 {
		switch k {
		case lfKind4:
			filter4EdgePureGo(pix, q0Base, step, outer, length, params)
		case lfKind6:
			filter6EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		case lfKind8:
			filter8EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		default:
			filter14EdgePureGo(pix, q0Base, step, outer, length, scale, params)
		}
		return
	}
	switch k {
	case lfKind4:
		filter4Edge16PureGo(pix, q0Base, step, outer, length, params)
	case lfKind6:
		filter6Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
	case lfKind8:
		filter8Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
	default:
		filter14Edge16PureGo(pix, q0Base, step, outer, length, scale, params)
	}
}

// lfEdge runs one deblocking edge of family k through the Go-native SIMD kernel.
//
// A horizontal edge has contiguous positions (outer == sample size) and its taps
// are whole rows apart, so the kernel reads the window in place. A vertical edge
// has contiguous taps (step == sample size) and positions one row apart. Batches
// of up to lfBatchGroups groups are transposed into stack scratch, filtered, and
// transposed back only for groups with an active filter lane. Sub-group tails
// and other layouts take the scalar reference.
func lfEdge[S lfSample](k lfKind, pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	sz := lfSize[S]()
	groups := length / 8
	if outer == sz && groups > 0 {
		_ = lfCore[S](k, pix, q0Base, step, groups*8, scale, params)
		if rem := length - groups*8; rem > 0 {
			lfPure[S](k, pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
		}
		return
	}
	if step != sz || groups == 0 {
		lfPure[S](k, pix, q0Base, step, outer, length, scale, params)
		return
	}
	before, taps, t0, t1 := lfSpec(k)
	var scratch [lfMaxTaps * 8 * lfBatchGroups * 2]byte
	row := 8 * lfBatchGroups * sz
	for g := 0; g < groups; g += lfBatchGroups {
		n := groups - g
		if n > lfBatchGroups {
			n = lfBatchGroups
		}
		src := q0Base + g*8*outer - before*step
		lfVertGather[S](scratch[:], pix, src, outer, taps, n*8, row)
		var changed uint
		if k == lfKind4 && sz == 1 && n >= 2 && params.center == 128 && params.min == -128 && params.max == 127 && params.blimit >= 0 && params.blimit < 255 && params.limit >= 0 && params.limit <= 255 && params.hev >= 0 && params.hev <= 255 {
			done := (n * 8) &^ 15
			lfFilter4U8Wide(scratch[:], before*row, row, done, params, &changed)
			if done < n*8 {
				changed |= lfFilter4Core[uint8](scratch[:], before*row+done, row, n*8-done, 1, params) << uint(done/8)
			}
		} else {
			changed = lfCore[S](k, scratch[:], before*row, row, n*8, scale, params)
		}
		if changed != 0 {
			lfVertScatter[S](pix, scratch[:], src, outer, t0, t1, n*8, row, changed)
		}
	}
	if rem := length - groups*8; rem > 0 {
		lfPure[S](k, pix, q0Base+groups*8*outer, step, outer, rem, scale, params)
	}
}

// lfVertGather copies the taps of n positions from pix into scratch. Taps are
// contiguous within a position (step == sample size), so tap t of position k sits
// at src + k*outer + t*sz in pix and lands at t*row + k*sz in scratch, where the
// horizontal kernel reads it as a contiguous row.
func lfVertGather[S lfSample](scratch []byte, pix []byte, src int, outer int, taps int, n int, row int) {
	if lfSize[S]() == 1 {
		if taps >= 8 {
			for g := 0; g < n/8; g++ {
				in := src + g*8*outer
				out := g * 8
				lfTranspose8x8U8(pix, in, outer, scratch, out, row)
				if taps == 14 {
					lfTranspose8x8U8(pix, in+6, outer, scratch, out+6*row, row)
				}
			}
			return
		}
		for g := 0; g < n/8; g++ {
			lfTranspose8x8U8Short(pix, src+g*8*outer, outer, scratch, g*8, row, taps, 8)
		}
		return
	}
	if taps >= 8 {
		for g := 0; g < n/8; g++ {
			in := src + g*8*outer
			out := g * 16
			lfTranspose8x8U16(pix, in, outer, scratch, out, row)
			if taps == 14 {
				lfTranspose8x8U16(pix, in+12, outer, scratch, out+6*row, row)
			}
		}
		return
	}
	for g := 0; g < n/8; g++ {
		lfTranspose8x8U16Short(pix, src+g*8*outer, outer, scratch, g*16, row, taps, 8)
	}
}

// lfVertScatter writes scratch tap rows [t0, t1) of n positions back to pix,
// the inverse of lfVertGather.
func lfVertScatter[S lfSample](pix []byte, scratch []byte, src int, outer int, t0 int, t1 int, n int, row int, changed uint) {
	if lfSize[S]() == 1 {
		if t1-t0 >= 6 {
			for g := 0; g < n/8; g++ {
				if changed&(1<<uint(g)) == 0 {
					continue
				}
				in := g * 8
				out := src + g*8*outer
				if t1 == 7 { // eight-tap: unchanged outer taps are in scratch too
					lfTranspose8x8U8(scratch, in, row, pix, out, outer)
				} else { // fourteen-tap: only rows 1..12 may change
					lfTranspose8x8U8(scratch, in+row, row, pix, out+1, outer)
					lfTranspose8x8U8(scratch, in+5*row, row, pix, out+5, outer)
				}
			}
			return
		}
		width := 6
		if t1 == 4 {
			width = 4
		}
		for g := 0; g < n/8; g++ {
			if changed&(1<<uint(g)) == 0 {
				continue
			}
			lfTranspose8x8U8Short(scratch, g*8, row, pix, src+g*8*outer, outer, 8, width)
		}
		return
	}
	if t1-t0 >= 6 {
		for g := 0; g < n/8; g++ {
			if changed&(1<<uint(g)) == 0 {
				continue
			}
			in := g * 16
			out := src + g*8*outer
			if t1 == 7 {
				lfTranspose8x8U16(scratch, in, row, pix, out, outer)
			} else {
				lfTranspose8x8U16(scratch, in+row, row, pix, out+2, outer)
				lfTranspose8x8U16(scratch, in+5*row, row, pix, out+10, outer)
			}
		}
		return
	}
	width := 6
	if t1 == 4 {
		width = 4
	}
	for g := 0; g < n/8; g++ {
		if changed&(1<<uint(g)) == 0 {
			continue
		}
		lfTranspose8x8U16Short(scratch, g*16, row, pix, src+g*8*outer, outer, 8, width)
	}
}
