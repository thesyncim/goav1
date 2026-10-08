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
func lfCore[S lfSample](k lfKind, pix []byte, q0Base int, step int, length int, scale int, params filter4Params) {
	switch k {
	case lfKind4:
		lfFilter4Core[S](pix, q0Base, step, length, scale, params)
	case lfKind6:
		lfFilter6Core[S](pix, q0Base, step, length, scale, params)
	case lfKind8:
		lfFilter8Core[S](pix, q0Base, step, length, scale, params)
	default:
		lfFilter14Core[S](pix, q0Base, step, length, scale, params)
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
// has contiguous taps (step == sample size) and positions one row apart, so
// batches of up to lfBatchGroups groups are gathered into a stack scratch laid
// out as a horizontal edge, the kernel runs on the scratch, and the rows the
// kernel may modify are scattered back. Sub-group tails and other layouts take
// the scalar reference.
func lfEdge[S lfSample](k lfKind, pix []byte, q0Base int, step int, outer int, length int, scale int, params filter4Params) {
	sz := lfSize[S]()
	groups := length / 8
	if outer == sz && groups > 0 {
		lfCore[S](k, pix, q0Base, step, groups*8, scale, params)
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
		lfCore[S](k, scratch[:], before*row, row, n*8, scale, params)
		lfVertScatter[S](pix, scratch[:], src, outer, t0, t1, n*8, row)
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
		for k := 0; k < n; k++ {
			r := pix[src+k*outer:][:taps]
			for t, v := range r {
				scratch[t*row+k] = v
			}
		}
		return
	}
	for k := 0; k < n; k++ {
		r := pix[src+k*outer:][:2*taps]
		for t := 0; t < taps; t++ {
			d := t*row + 2*k
			scratch[d] = r[2*t]
			scratch[d+1] = r[2*t+1]
		}
	}
}

// lfVertScatter writes scratch tap rows [t0, t1) of n positions back to pix,
// the inverse of lfVertGather.
func lfVertScatter[S lfSample](pix []byte, scratch []byte, src int, outer int, t0 int, t1 int, n int, row int) {
	if lfSize[S]() == 1 {
		for k := 0; k < n; k++ {
			r := pix[src+k*outer:]
			for t := t0; t < t1; t++ {
				r[t] = scratch[t*row+k]
			}
		}
		return
	}
	for k := 0; k < n; k++ {
		r := pix[src+k*outer:]
		for t := t0; t < t1; t++ {
			s := t*row + 2*k
			r[2*t] = scratch[s]
			r[2*t+1] = scratch[s+1]
		}
	}
}
