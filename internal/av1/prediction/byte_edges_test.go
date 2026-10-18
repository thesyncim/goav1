package prediction

import (
	"bytes"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

// Frozen pre-change pixel walk; do not delegate to the optimized range helper.
func originalDirectionalZ2Ranges(width int, height int, upsampleAbove int, upsampleLeft int, dx int, dy int) (sampleIndexRange, sampleIndexRange, error) {
	minBaseX := -(1 << upsampleAbove)
	minBaseY := -(1 << upsampleLeft)
	fracBitsX := 6 - upsampleAbove
	fracBitsY := 6 - upsampleLeft
	var aboveRange sampleIndexRange
	var leftRange sampleIndexRange
	for row := range height {
		for col := range width {
			y := row + 1
			x := (col << 6) - y*dx
			baseX := x >> fracBitsX
			if baseX >= minBaseX {
				originalRangeAdd(&aboveRange, baseX)
				originalRangeAdd(&aboveRange, baseX+1)
				continue
			}
			x = col + 1
			y = (row << 6) - x*dy
			baseY := y >> fracBitsY
			if baseY < minBaseY {
				return sampleIndexRange{}, sampleIndexRange{}, ErrInvalidPrediction
			}
			originalRangeAdd(&leftRange, baseY)
			originalRangeAdd(&leftRange, baseY+1)
		}
	}
	return aboveRange, leftRange, nil
}

func originalRangeAdd(r *sampleIndexRange, index int) {
	if !r.used {
		r.used, r.min, r.max = true, index, index
		return
	}
	if index < r.min {
		r.min = index
	}
	if index > r.max {
		r.max = index
	}
}

func TestDirectionalZ2EndpointRangesMatchOriginal(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128}
	for angle := 91; angle < 180; angle++ {
		dx, dy := DirectionalDX(angle), DirectionalDY(angle)
		if dx <= 0 || dy <= 0 {
			continue
		}
		for _, w := range sizes {
			for _, h := range sizes {
				for ua := range 2 {
					for ul := range 2 {
						a, l, err := directionalZ2Ranges(w, h, ua, ul, dx, dy)
						wa, wl, we := originalDirectionalZ2Ranges(w, h, ua, ul, dx, dy)
						if a != wa || l != wl || err != we {
							t.Fatalf("%dx%d angle%d up%d,%d: got%+v/%+v/%v want%+v/%+v/%v", w, h, angle, ua, ul, a, l, err, wa, wl, we)
						}
					}
				}
			}
		}
	}
	// Invalid left projections and inputs beyond the algebraic fast domain.
	for _, c := range [][6]int{{16, 16, 0, 0, 1023, 1023}, {129, 3, 0, 1, 64, 64}, {4, 4, 2, 0, 64, 64}, {3, 4, 0, 0, 0, 0}, {2, 1, 0, 0, 2048, 1}, {0, 4, 0, 0, 64, 64}, {4, -1, 0, 0, 64, 64}} {
		a, l, e := directionalZ2Ranges(c[0], c[1], c[2], c[3], c[4], c[5])
		wa, wl, we := originalDirectionalZ2Ranges(c[0], c[1], c[2], c[3], c[4], c[5])
		if a != wa || l != wl || e != we {
			t.Fatalf("fallback%v: got%+v/%+v/%v want%+v/%+v/%v", c, a, l, e, wa, wl, we)
		}
	}
}

func TestByteEdgePredictionMatchesValidated(t *testing.T) {
	for _, shape := range [][4]int{{4, 4, 4, 4}, {15, 7, 16, 8}, {32, 32, 32, 32}, {63, 31, 64, 32}} {
		w, h, pw, ph := shape[0], shape[1], shape[2], shape[3]
		edges := IntraEdges{Above: make([]uint16, pw), Left: make([]uint16, ph), AboveLeft: 255, AboveAvailable: true, LeftAvailable: true, AboveLeftAvailable: true}
		for i := range edges.Above {
			edges.Above[i] = uint16((i*79 + 3) & 255)
		}
		for i := range edges.Left {
			edges.Left[i] = uint16((i*71 + 131) & 255)
		}
		for mode := IntraModeDC; mode <= IntraModeSmoothHorizontal; mode++ {
			for missing := 0; missing < 4; missing++ {
				e := edges
				e.AboveAvailable = missing&1 == 0
				e.LeftAvailable = missing&2 == 0
				want := frame.Plane{Pix: bytes.Repeat([]byte{0x5a}, (w+4)*(h+2)), Stride: w + 4, Width: w + 4, Height: h + 2}
				got := want
				got.Pix = slices.Clone(want.Pix)
				we := PredictIntraPlaneBlockWithExtent(want, 1, 8, 1, 1, w, h, pw, ph, mode, e)
				ge := PredictIntraPlaneBlockWithByteEdges(got, 1, 1, w, h, pw, ph, mode, e)
				if we != ge || !bytes.Equal(want.Pix, got.Pix) {
					t.Fatalf("shape%v mode%d missing%d: error%v/%v or pixels differ", shape, mode, missing, we, ge)
				}
			}
		}
		if pw <= 32 && ph <= 32 {
			for mode := FilterIntraModeDC; mode < FilterIntraModes; mode++ {
				want := frame.Plane{Pix: bytes.Repeat([]byte{0x5a}, (w+4)*(h+2)), Stride: w + 4, Width: w + 4, Height: h + 2}
				got := want
				got.Pix = slices.Clone(want.Pix)
				we := PredictFilterIntraPlaneBlockWithExtent(want, 1, 8, 1, 1, w, h, pw, ph, mode, edges)
				ge := PredictFilterIntraPlaneBlockWithByteEdges(got, 1, 1, w, h, pw, ph, mode, edges)
				if we != ge || !bytes.Equal(want.Pix, got.Pix) {
					t.Fatalf("filter shape%v mode%d: %v/%v or pixels differ", shape, mode, we, ge)
				}
			}
		}
	}
}

func TestByteEdgeFiltersMatchValidated(t *testing.T) {
	for _, n := range []int{1, 2, 3, 4, 5, 8, 17, 33, 65, 129, 257} {
		for strength := uint8(0); strength < 4; strength++ {
			want := make([]uint16, n)
			for i := range want {
				want[i] = uint16((i*79 + 3) & 255)
			}
			got := slices.Clone(want)
			ws, gs := make([]uint16, n+2), make([]uint16, n+2)
			we := FilterIntraEdge(want, ws, strength, 8)
			ge := FilterIntraEdgeFromBytes(got, gs, strength)
			if we != ge || !slices.Equal(want, got) || !slices.Equal(ws, gs) {
				t.Fatalf("filter n%d strength%d mismatch", n, strength)
			}
		}
	}
	for n := 1; n <= 16; n++ {
		want := make([]uint16, 2*n+4)
		for i := range want {
			want[i] = uint16((i*79 + 3) & 255)
		}
		got := slices.Clone(want)
		ws, gs := make([]uint16, n+5), make([]uint16, n+5)
		we := UpsampleIntraEdge(want, 2, n, ws, 8)
		ge := UpsampleIntraEdgeFromBytes(got, 2, n, gs)
		if we != ge || !slices.Equal(want, got) || !slices.Equal(ws, gs) {
			t.Fatalf("upsample n%d mismatch", n)
		}
	}
}
