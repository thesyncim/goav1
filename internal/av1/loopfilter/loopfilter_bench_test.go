package loopfilter

import (
	"bytes"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"math/rand"
	"testing"
)

// Vertical fourteen-sample benchmarks over a flat plane (the all-flat wide
// path is the heaviest branch and the common case on smooth content). The
// 10-bit case covers the two-byte transpose path; the short case covers a
// single-group batch.

func BenchmarkFilter14EdgeVertical10bit(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter14Edge(plane, 2, 10, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func BenchmarkFilter14EdgeVerticalShort(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter14Edge(plane, 1, 8, EdgeVertical, 32, 0, 8, thresholds)
	}
}

// benchFilter4Edge sets up a horizontal-edge buffer of edgeLen contiguous
// positions (outer == 1, step == stride) with thresholds chosen so most
// positions pass needsFilter4 and split across the hev branch, the worst case
// for the kernel.
func benchFilter4Edge(edgeLen int) ([]byte, int, int, int, filter4Params) {
	const stride = 256
	buf := make([]byte, stride*8)
	rng := rand.New(rand.NewSource(99))
	for i := range buf {
		// Keep adjacent rows close so needsFilter4 passes for most positions.
		buf[i] = byte(100 + rng.Intn(40))
	}
	params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
	q0Base := 3 * stride
	return buf, q0Base, stride, edgeLen, params
}

// BenchmarkFilter4EdgePureGo and BenchmarkFilter4EdgeDispatch bracket the
// per-kernel speedup: the first always runs the pure-Go reference, the second
// runs whatever the dispatcher resolved (Go SIMD on arm64). Both time only the
// inner edge kernel, not the validation wrapper.
func BenchmarkFilter4EdgePureGo(b *testing.B) {
	buf, q0Base, step, length, params := benchFilter4Edge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter4EdgePureGo(buf, q0Base, step, 1, length, params)
	}
}

func BenchmarkFilter4EdgeDispatch(b *testing.B) {
	buf, q0Base, step, length, params := benchFilter4Edge(64)
	b.ReportAllocs()
	for b.Loop() {
		filter4EdgeImpl(buf, q0Base, step, 1, length, params)
	}
}

// BenchmarkFilterAB exercises the public dispatch with one edge geometry. The near-flat alternating
// samples keep the filter active, including the wide fourteen-tap path.
func BenchmarkFilterAB(b *testing.B) {
	type filterFn func(frame.Plane, int, uint8, Edge, int32, int32, int32, Thresholds) error
	for _, f := range []struct {
		name string
		fn   filterFn
	}{
		{"4", Filter4Edge}, {"6", Filter6Edge}, {"8", Filter8Edge}, {"14", Filter14Edge},
	} {
		for _, depth := range []uint8{8, 10, 12} {
			for _, dir := range []struct {
				name string
				edge Edge
				x, y int32
			}{{"H", EdgeHorizontal, 0, 32}, {"V", EdgeVertical, 32, 0}} {
				b.Run(fmt.Sprintf("F%s/%dbit/%s", f.name, depth, dir.name), func(b *testing.B) {
					bytesPerSample := 1
					if depth > 8 {
						bytesPerSample = 2
					}
					plane := testPlane(64, 64, bytesPerSample, 64*bytesPerSample)
					mid := uint16(1 << (depth - 1))
					for y := 0; y < 64; y++ {
						for x := 0; x < 64; x++ {
							setSample(plane, bytesPerSample, x, y, mid+uint16((x+y)&1))
						}
					}
					thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
					b.ReportAllocs()
					for b.Loop() {
						if err := f.fn(plane, bytesPerSample, depth, dir.edge, dir.x, dir.y, 64, thresholds); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

// Vertical six/eight-sample benchmarks over a flat 10-bit plane (the all-flat
// path is the heaviest branch and the common case on smooth content). These
// cover the two-byte transpose path around the horizontal 16-bit kernels; the
// short cases cover a single-group batch.

func BenchmarkFilter6EdgeVertical10bit(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter6Edge(plane, 2, 10, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func BenchmarkFilter8EdgeVertical10bit(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter8Edge(plane, 2, 10, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func BenchmarkFilter6EdgeVertical10bitShort(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter6Edge(plane, 2, 10, EdgeVertical, 32, 0, 8, thresholds)
	}
}

func BenchmarkFilter8EdgeVertical10bitShort(b *testing.B) {
	plane := testPlane(64, 64, 2, 128)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter8Edge(plane, 2, 10, EdgeVertical, 32, 0, 8, thresholds)
	}
}

// benchmarkFilterSB8 filters every 4-pixel column (vertical) or row
// (horizontal) of a 128x128 luma region with dense random masks, the shape of
// one mask region in the frame apply.
func benchmarkFilterSB8(b *testing.B, vertical bool) {
	rng := rand.New(rand.NewSource(9))
	const w, h = 160, 160
	pix := sbTestPlane(rng, w, h)
	lvlStride := w / 4
	lvl := sbTestLevels(rng, lvlStride*(h/4))
	lut := NewSBLUT(0)
	var masks [32][3]uint32
	for i := range masks {
		masks[i] = sbTestMasks(rng, true, 32)
		masks[i][2] &= 0x11111111 // wd16 only on 16-aligned units
	}
	work := append([]byte(nil), pix...)
	for b.Loop() {
		copy(work, pix)
		for i := 1; i < 32; i++ {
			if vertical {
				x := 4 + 4*i
				FilterSBVertical8(work, 8*w+x, w, w-x, masks[i], true, lvl, 2*lvlStride+x/4, lvlStride, 0, &lut)
			} else {
				y := 8 + 4*i
				FilterSBHorizontal8(work, y*w+8, w, w-8, masks[i], true, lvl, (y/4)*lvlStride+2, lvlStride, 1, &lut)
			}
		}
	}
}

func BenchmarkFilterSBVertical8(b *testing.B)   { benchmarkFilterSB8(b, true) }
func BenchmarkFilterSBHorizontal8(b *testing.B) { benchmarkFilterSB8(b, false) }

// These benchmarks reproduce dav1d's tests/checkasm/loopfilter.c check_lpf_sb
// for 8-bit luma and chroma: one superblock call over 32 (luma) or 16
// (chroma) units, edge pixels shaped by init_lpf_border so that the long-flat,
// short-flat, normal/hev and unfiltered cases all occur, unit widths drawn as
// checkasm draws them, and levels in 0..63 with dav1d's level-from-previous
// fallback layout. The ns/op is directly comparable with
// `checkasm -t loopfilter_8bpc -b` (lpf_h_sb_* is FilterSBVertical8, the
// vertical-edge column filter; lpf_v_sb_* is FilterSBHorizontal8).

func checkasmClip(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

type checkasmSBCase struct {
	pix       []byte
	off       int
	stride    int
	rowBytes  int
	vm        [3]uint32
	lvl       []uint8
	lvlOff    int
	lvlStride int
	lut       SBLUT
}

// newCheckasmSBCase builds one check_lpf_sb call. vertical selects the
// vertical-edge (column) filter; strength is checkasm's loop index i.
func newCheckasmSBCase(seed int64, vertical, luma bool, strength int) checkasmSBCase {
	rng := rand.New(rand.NewSource(seed))
	nBlks := 32
	if !luma {
		nBlks = 16
	}
	sharp := rng.Intn(8)
	c := checkasmSBCase{lut: NewSBLUT(uint8(sharp))}
	// checkasm: l[32*2][4]; vertical edges read l[2j+1] (fallback l[2j]),
	// horizontal edges read l[32+j] (fallback l[j]).
	c.lvl = make([]uint8, 4*(64))
	for j := 0; j < nBlks; j++ {
		idx := rng.Intn(strength + 2)
		if idx != 0 {
			c.vm[idx-1] |= 1 << uint(j)
		}
		if vertical {
			c.lvl[4*(2*j)+0] = uint8(rng.Intn(64))
			c.lvl[4*(2*j+1)+0] = uint8(rng.Intn(64))
		} else {
			c.lvl[4*(j)+0] = uint8(rng.Intn(64))
			c.lvl[4*(32+j)+0] = uint8(rng.Intn(64))
		}
	}
	if vertical {
		// 16 taps per row around the edge; a wider row keeps the kernel on the
		// in-frame (16-byte row) path the frame apply takes.
		c.stride = 32
		c.pix = make([]byte, c.stride*(4*nBlks))
		c.off = 8
		c.rowBytes = c.stride - 8
		c.lvlOff, c.lvlStride = 1, 2
	} else {
		c.stride = 4 * nBlks
		c.pix = make([]byte, c.stride*16)
		c.off = 8 * c.stride
		c.rowBytes = c.stride
		c.lvlOff, c.lvlStride = 32, 32
	}
	for i := 0; i < 4*nBlks; i++ {
		x := i >> 2
		var L uint8
		if vertical {
			L = c.lvl[4*(2*x+1)+0]
			if L == 0 {
				L = c.lvl[4*(2*x)+0]
			}
		} else {
			L = c.lvl[4*(32+x)+0]
			if L == 0 {
				L = c.lvl[4*(x)+0]
			}
		}
		E, I := int(c.lut.E[L]), int(c.lut.I[L])
		var base, step int
		if vertical {
			base, step = i*c.stride+c.off, 1
		} else {
			base, step = c.off+i, c.stride
		}
		set := func(k int, v uint8) { c.pix[base+k*step] = v }
		get := func(k int) uint8 { return c.pix[base+k*step] }
		checkasmLPFBorderFull(rng, set, get, E, I)
	}
	return c
}

// checkasmLPFBorderFull is the complete init_lpf_border (all four cases).
func checkasmLPFBorderFull(rng *rand.Rand, set func(i int, v uint8), get func(i int) uint8, E, I int) {
	const F = 1
	filterType := rng.Intn(4)
	edgeDiff := rng.Intn((E+2)*4) - 2*(E+2)
	switch filterType {
	case 0:
		for i := -8; i < 8; i++ {
			set(i, uint8(rng.Intn(256)))
		}
	case 1:
		set(-8, uint8(rng.Intn(256)))
		set(7, uint8(rng.Intn(256)))
		set(0, uint8(rng.Intn(256)))
		set(-1, checkasmClip(int(get(0))+edgeDiff))
		for i := 1; i < 7; i++ {
			set(-(1 + i), checkasmClip(int(get(-1))+rng.Intn(2*(F+1))-(F+1)))
			set(i, checkasmClip(int(get(0))+rng.Intn(2*(F+1))-(F+1)))
		}
	case 2:
		for i := 4; i < 8; i++ {
			set(-(1 + i), uint8(rng.Intn(256)))
			set(i, uint8(rng.Intn(256)))
		}
		set(0, uint8(rng.Intn(256)))
		set(-1, checkasmClip(int(get(0))+edgeDiff))
		for i := 1; i < 4; i++ {
			set(-(1 + i), checkasmClip(int(get(-1))+rng.Intn(2*(F+1))-(F+1)))
			set(i, checkasmClip(int(get(0))+rng.Intn(2*(F+1))-(F+1)))
		}
	case 3:
		for i := 4; i < 8; i++ {
			set(-(1 + i), uint8(rng.Intn(256)))
			set(i, uint8(rng.Intn(256)))
		}
		set(0, uint8(rng.Intn(256)))
		set(-1, checkasmClip(int(get(0))+edgeDiff))
		for i := 1; i < 4; i++ {
			set(-(1 + i), checkasmClip(int(get(-i))+rng.Intn(2*(I+1))-(I+1)))
			set(i, checkasmClip(int(get(i-1))+rng.Intn(2*(I+1))-(I+1)))
		}
	}
}

func benchmarkCheckasmSB(b *testing.B, vertical, luma bool) {
	n := 3
	widths := []int{4, 8, 16}
	if !luma {
		n = 2
		widths = []int{4, 6}
	}
	for s := 0; s < n; s++ {
		b.Run(fmt.Sprintf("w%d", widths[s]), func(b *testing.B) {
			// checkasm's bench alternates between two identical buffers; keep a
			// pool of independent draws so one random case does not dominate.
			const cases = 16
			var cs [cases]checkasmSBCase
			var work [cases][]byte
			for i := range cs {
				cs[i] = newCheckasmSBCase(int64(1000*s+i), vertical, luma, s)
				work[i] = append([]byte(nil), cs[i].pix...)
			}
			i := 0
			for b.Loop() {
				c := &cs[i&(cases-1)]
				w := work[i&(cases-1)]
				if vertical {
					FilterSBVertical8(w, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				} else {
					FilterSBHorizontal8(w, c.off, c.stride, c.rowBytes, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				}
				i++
			}
		})
	}
}

func BenchmarkCheckasmLPFHSBY8(b *testing.B)  { benchmarkCheckasmSB(b, true, true) }
func BenchmarkCheckasmLPFVSBY8(b *testing.B)  { benchmarkCheckasmSB(b, false, true) }
func BenchmarkCheckasmLPFHSBUV8(b *testing.B) { benchmarkCheckasmSB(b, true, false) }
func BenchmarkCheckasmLPFVSBUV8(b *testing.B) { benchmarkCheckasmSB(b, false, false) }

// TestFilterSB8CheckasmMatchesScalar runs the dispatched superblock filters
// against the unit-by-unit C reference on checkasm-shaped content, where the
// flat, normal and hev paths all fire (random pixels mostly fail the filter
// mask). Levels are moved to a random component and the level slice is cut
// right after the last unit, so the vector level loads hit their bounds
// fallback as well.
func TestFilterSB8CheckasmMatchesScalar(t *testing.T) {
	for seed := int64(0); seed < 600; seed++ {
		for _, vertical := range []bool{true, false} {
			for _, luma := range []bool{true, false} {
				n := 3
				if !luma {
					n = 2
				}
				for s := 0; s < n; s++ {
					c := newCheckasmSBCase(seed*7+int64(s), vertical, luma, s)
					comp := int(seed & 3)
					lvl := make([]uint8, len(c.lvl))
					for i := range len(c.lvl) / 4 {
						lvl[4*(i)+comp] = c.lvl[4*(i)+0]
						lvl[4*(i)+(comp+1)&3] = uint8(i) // junk in the other components
					}
					if seed&1 == 1 && !vertical {
						lvl = lvl[:4*(c.lvlOff+31)]
					}
					want := append([]byte(nil), c.pix...)
					got := append([]byte(nil), c.pix...)
					vm := c.vm
					if seed&1 == 1 && !vertical {
						for k := range vm {
							vm[k] &= 0x7fffffff
						}
					}
					if vertical {
						filterSBVertical8Scalar(want, c.off, c.stride, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
						FilterSBVertical8(got, c.off, c.stride, c.rowBytes, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
					} else {
						filterSBHorizontal8Scalar(want, c.off, c.stride, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
						FilterSBHorizontal8(got, c.off, c.stride, c.rowBytes, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
					}
					if !bytes.Equal(got, want) {
						t.Fatalf("seed %d vertical=%v luma=%v s=%d: pixels differ", seed, vertical, luma, s)
					}
				}
			}
		}
	}
}
