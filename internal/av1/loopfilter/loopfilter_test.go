package loopfilter

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"math/bits"
	"math/rand"
	"strconv"
	"testing"
	"unsafe"
)

// TestFilterEdgeByWidthAllocs covers the lower-level FilterEdgeByWidth used by
// callers that already hold a resolved level/thresholds pair.
func TestFilterEdgeByWidthAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, uint16((x*7+y*11)&0xff))
		}
	}
	thresholds, err := ThresholdsForLevel(16, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, width := range []uint8{4, 6, 8, 14} {
		t.Run("width="+strconv.Itoa(int(width)), func(t *testing.T) {
			if err := FilterEdgeByWidth(width, plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
				t.Fatalf("warm-up err=%v", err)
			}
			allocs := testing.AllocsPerRun(1000, func() {
				if err := FilterEdgeByWidth(width, plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Fatalf("FilterEdgeByWidth(%d) allocated: %f", width, allocs)
			}
		})
	}
}

// TestFilter14EdgeQ32Row7 reproduces the exact 10-bit pre-LF samples at the
// vertical edge x=16 (corresponds to x=160 in the libaom q32 frame 0 dump) for
// rows 0..15. needsFilter8 must reject row 7 (|p1-p0| > limit) so the call
// leaves samples (16, 7) and (16, 8) unchanged. This regression covers a bug
// where the q32 10-bit deblock applied filter14 fallback (filter4 hev path) at
// rows whose mask check should have rejected the filter.
func TestFilter14EdgeQ32Row7(t *testing.T) {
	const W = 32
	const H = 16
	const bytesPerSample = 2
	const bitDepth = 10

	plane := frame.Plane{
		Width:  W,
		Height: H,
		Stride: W * bytesPerSample,
	}
	plane.Pix = make([]byte, plane.Stride*H)

	// Samples copied from /tmp/libaom-q32-stages/stage-pre.y around x=144..175
	// for rows 0..15. Encoded so the vertical edge falls at local x=16
	// (corresponds to global x=160).
	rows := [H][W]uint16{
		/* y=0  */ {466, 458, 449, 446, 446, 459, 343, 320, 320, 320, 320, 298, 298, 298, 409, 448, 487, 433, 519, 744, 800, 766, 744, 787, 719, 717, 715, 711, 708, 704, 702, 691},
		/* y=1  */ {476, 469, 442, 405, 395, 405, 376, 362, 328, 319, 312, 369, 448, 463, 436, 557, 775, 829, 788, 752, 783, 730, 723, 712, 703, 689, 671, 657, 642, 627, 613, 598},
		/* y=2  */ {487, 469, 442, 408, 387, 364, 360, 361, 333, 328, 311, 321, 436, 447, 460, 620, 805, 821, 802, 786, 759, 749, 731, 700, 691, 675, 656, 639, 620, 603, 588, 571},
		/* y=3  */ {500, 488, 449, 416, 395, 369, 363, 363, 336, 320, 320, 322, 443, 491, 561, 713, 825, 816, 817, 797, 773, 749, 729, 708, 691, 676, 660, 641, 619, 600, 583, 565},
		/* y=4  */ {515, 503, 459, 432, 412, 387, 376, 376, 338, 325, 350, 473, 595, 700, 797, 826, 817, 819, 819, 819, 819, 819, 819, 805, 783, 758, 731, 712, 687, 663, 643, 622},
		/* y=5  */ {531, 519, 489, 460, 426, 416, 411, 384, 337, 328, 376, 522, 704, 801, 827, 819, 816, 817, 819, 819, 819, 819, 819, 819, 808, 785, 762, 744, 721, 696, 670, 643},
		/* y=6  */ {544, 540, 535, 519, 489, 472, 449, 425, 339, 326, 379, 579, 773, 825, 813, 799, 800, 808, 819, 819, 819, 819, 819, 819, 819, 819, 819, 817, 798, 772, 743, 712},
		/* y=7  */ {512, 459, 458, 427, 399, 355, 338, 325, 372, 632, 801, 822, 786, 776, 787, 797, 800, 796, 730, 723, 708, 689, 669, 650, 635, 619, 605, 590, 580, 571, 562, 555},
		/* y=8  */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 819, 823, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826},
		/* y=9  */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 819, 823, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826},
		/* y=10 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 819, 823, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826},
		/* y=11 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 819, 823, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826},
		/* y=12 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 819, 823, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826},
		/* y=13 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 819, 823, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826},
		/* y=14 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 819, 823, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826},
		/* y=15 */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 819, 823, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826, 826},
	}

	for y := range H {
		for x := range W {
			plane.Pix[y*plane.Stride+x*bytesPerSample] = byte(rows[y][x])
			plane.Pix[y*plane.Stride+x*bytesPerSample+1] = byte(rows[y][x] >> 8)
		}
	}

	thresholds, err := ThresholdsForLevel(15, 0)
	if err != nil {
		t.Fatalf("ThresholdsForLevel: %v", err)
	}
	if err := Filter14Edge(plane, bytesPerSample, bitDepth, EdgeVertical, 16, 0, H, thresholds); err != nil {
		t.Fatalf("Filter14Edge: %v", err)
	}

	post7 := uint16(plane.Pix[7*plane.Stride+16*bytesPerSample]) |
		uint16(plane.Pix[7*plane.Stride+16*bytesPerSample+1])<<8
	if post7 != rows[7][16] {
		t.Errorf("(16, 7) was modified: pre=%d post=%d", rows[7][16], post7)
	}
}

func TestFilter14EdgeHorizontalFlat8Bit(t *testing.T) {
	plane := testPlane(4, 14, 1, 4)
	for x := range 4 {
		for y := range 7 {
			setSample(plane, 1, x, y, 90)
		}
		for y := 7; y < 14; y++ {
			setSample(plane, 1, x, y, 100)
		}
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter14Edge(plane, 1, 8, EdgeHorizontal, 0, 7, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	want := []uint16{90, 91, 91, 92, 93, 93, 94, 96, 97, 98, 98, 99, 99, 100}
	for x := range 4 {
		for y, value := range want {
			if got := getSample(plane, 1, x, y); got != value {
				t.Fatalf("sample(%d,%d)=%d want %d", x, y, got, value)
			}
		}
	}
}

func TestFilter14EdgeHighBitDepth(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		plane := testPlane(4, 14, 2, 8)
		for x := range 4 {
			for y := range 7 {
				setSample(plane, 2, x, y, 900)
			}
			for y := 7; y < 14; y++ {
				setSample(plane, 2, x, y, 940)
			}
		}

		thresholds := Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3}
		if err := Filter14Edge(plane, 2, bitDepth, EdgeHorizontal, 0, 7, 4, thresholds); err != nil {
			t.Fatal(err)
		}

		want := []uint16{900, 903, 905, 908, 910, 913, 918, 923, 928, 930, 933, 935, 938, 940}
		for x := range 4 {
			for y, value := range want {
				if got := getSample(plane, 2, x, y); got != value {
					t.Fatalf("bitdepth=%d sample(%d,%d)=%d want %d", bitDepth, x, y, got, value)
				}
			}
		}
	}
}

func TestFilter14EdgeMatchesCReference(t *testing.T) {
	tests := []struct {
		name       string
		edge       Edge
		bitDepth   uint8
		thresholds Thresholds
		samples    [][14]int
	}{
		{
			name:       "horizontal 8 bit",
			edge:       EdgeHorizontal,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][14]int{
				{90, 90, 90, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{80, 80, 80, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{80, 80, 80, 80, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{10, 10, 10, 10, 10, 10, 10, 200, 200, 200, 200, 200, 200, 200},
			},
		},
		{
			name:       "vertical 8 bit",
			edge:       EdgeVertical,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][14]int{
				{90, 90, 90, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{80, 80, 80, 90, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{80, 80, 80, 80, 90, 90, 90, 100, 100, 100, 100, 100, 100, 100},
				{10, 10, 10, 10, 10, 10, 10, 200, 200, 200, 200, 200, 200, 200},
			},
		},
		{
			name:       "horizontal high bit depth",
			edge:       EdgeHorizontal,
			bitDepth:   10,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][14]int{
				{900, 900, 900, 900, 900, 900, 900, 940, 940, 940, 940, 940, 940, 940},
				{850, 850, 850, 900, 900, 900, 900, 940, 940, 940, 940, 940, 940, 940},
				{200, 200, 200, 200, 200, 200, 200, 900, 900, 900, 900, 900, 900, 900},
			},
		},
		{
			name:       "vertical high bit depth",
			edge:       EdgeVertical,
			bitDepth:   12,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][14]int{
				{900, 900, 900, 900, 900, 900, 900, 940, 940, 940, 940, 940, 940, 940},
				{850, 850, 850, 900, 900, 900, 900, 940, 940, 940, 940, 940, 940, 940},
				{200, 200, 200, 200, 200, 200, 200, 900, 900, 900, 900, 900, 900, 900},
			},
		},
	}

	for _, tt := range tests {
		bytesPerSample := 1
		if tt.bitDepth != 8 {
			bytesPerSample = 2
		}

		got := filter14ReferencePlane(tt.edge, bytesPerSample, tt.samples)
		want := cloneTestPlane(got)
		for i, sample := range tt.samples {
			ref := cReferenceFilter14(sample, tt.bitDepth, tt.thresholds)
			writeFilter14ReferenceSample(want, bytesPerSample, tt.edge, i, ref)
		}

		x, y, length := int32(0), int32(7), int32(len(tt.samples))
		if tt.edge == EdgeVertical {
			x, y = 7, 0
		}
		if err := Filter14Edge(got, bytesPerSample, tt.bitDepth, tt.edge, x, y, length, tt.thresholds); err != nil {
			t.Fatalf("%s Filter14Edge: %v", tt.name, err)
		}
		if !bytes.Equal(got.Pix, want.Pix) {
			t.Fatalf("%s got=%v want=%v", tt.name, got.Pix, want.Pix)
		}
	}
}

func TestFilter14EdgeRejectsInvalidInputs(t *testing.T) {
	plane := testPlane(14, 14, 1, 14)
	tests := []struct {
		name           string
		plane          frame.Plane
		bytesPerSample int
		bitDepth       uint8
		edge           Edge
		x              int32
		y              int32
		length         int32
	}{
		{name: "sample width", plane: plane, bytesPerSample: 2, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 7, length: 4},
		{name: "edge", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: Edge(9), x: 0, y: 7, length: 4},
		{name: "length", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 7},
		{name: "horizontal top", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 6, length: 4},
		{name: "horizontal bottom", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 8, length: 4},
		{name: "vertical left", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 6, y: 0, length: 4},
		{name: "vertical right", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 8, y: 0, length: 4},
	}
	for _, tt := range tests {
		err := Filter14Edge(tt.plane, tt.bytesPerSample, tt.bitDepth, tt.edge, tt.x, tt.y, tt.length, Thresholds{})
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestFilter14EdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, 100)
		}
	}
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := Filter14Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Filter14Edge allocated: %f", allocs)
	}
}

func FuzzFilter14Edge(f *testing.F) {
	f.Add(uint8(14), uint8(14), uint8(0), uint8(20), uint8(25), uint8(10))
	f.Add(uint8(16), uint8(16), uint8(1), uint8(63), uint8(132), uint8(3))
	f.Fuzz(func(t *testing.T, rawWidth uint8, rawHeight uint8, rawEdge uint8, rawLimit uint8, rawBlockLimit uint8, rawHEV uint8) {
		width := int(rawWidth%32) + 14
		height := int(rawHeight%32) + 14
		edge := Edge(rawEdge & 1)
		bytesPerSample := 1
		bitDepth := uint8(8)
		stride := width
		max := uint16(0xff)
		if rawEdge&2 != 0 {
			bytesPerSample = 2
			bitDepth = 10
			if rawEdge&4 != 0 {
				bitDepth = 12
			}
			stride = width * bytesPerSample
			max = uint16((1 << bitDepth) - 1)
		}
		plane := testPlane(width, height, bytesPerSample, stride)
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				setSample(plane, bytesPerSample, x, y, uint16((x*19+y*23)&int(max)))
			}
		}
		x := 7
		y := 7
		length := width - x
		if edge == EdgeVertical {
			length = height - y
		}
		err := Filter14Edge(plane, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), Thresholds{
			Limit:            rawLimit,
			BlockLimit:       rawBlockLimit,
			HighEdgeVariance: rawHEV,
		})
		if err != nil {
			t.Fatal(err)
		}
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				if got := getSample(plane, bytesPerSample, x, y); got > max {
					t.Fatalf("sample(%d,%d)=%d max=%d", x, y, got, max)
				}
			}
		}
	})
}

func filter14ReferencePlane(edge Edge, bytesPerSample int, samples [][14]int) frame.Plane {
	if edge == EdgeHorizontal {
		plane := testPlane(len(samples), 14, bytesPerSample, len(samples)*bytesPerSample)
		for i, sample := range samples {
			writeFilter14ReferenceSample(plane, bytesPerSample, edge, i, sample)
		}
		return plane
	}
	plane := testPlane(14, len(samples), bytesPerSample, 14*bytesPerSample)
	for i, sample := range samples {
		writeFilter14ReferenceSample(plane, bytesPerSample, edge, i, sample)
	}
	return plane
}

func writeFilter14ReferenceSample(plane frame.Plane, bytesPerSample int, edge Edge, index int, sample [14]int) {
	for i, value := range sample {
		if edge == EdgeHorizontal {
			setSample(plane, bytesPerSample, index, i, uint16(value))
			continue
		}
		setSample(plane, bytesPerSample, i, index, uint16(value))
	}
}

func cReferenceFilter14(in [14]int, bitDepth uint8, thresholds Thresholds) [14]int {
	scale := 1 << int(bitDepth-8)
	limit := int(thresholds.Limit) * scale
	blimit := int(thresholds.BlockLimit) * scale

	p6, p5, p4, p3 := in[0], in[1], in[2], in[3]
	p2, p1, p0 := in[4], in[5], in[6]
	q0, q1, q2, q3 := in[7], in[8], in[9], in[10]
	q4, q5, q6 := in[11], in[12], in[13]
	if !cReferenceFilterMask4(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3) {
		return in
	}
	flat := cReferenceFlatMask4(scale, p3, p2, p1, p0, q0, q1, q2, q3)
	flat2 := cReferenceFlatMask4(scale, p6, p5, p4, p0, q0, q4, q5, q6)
	if !flat || !flat2 {
		narrow := cReferenceFilter8([8]int{p3, p2, p1, p0, q0, q1, q2, q3}, bitDepth, thresholds)
		in[3], in[4], in[5], in[6] = narrow[0], narrow[1], narrow[2], narrow[3]
		in[7], in[8], in[9], in[10] = narrow[4], narrow[5], narrow[6], narrow[7]
		return in
	}

	in[1] = cRoundPowerOfTwo(p6*7+p5*2+p4*2+p3+p2+p1+p0+q0, 4)
	in[2] = cRoundPowerOfTwo(p6*5+p5*2+p4*2+p3*2+p2+p1+p0+q0+q1, 4)
	in[3] = cRoundPowerOfTwo(p6*4+p5+p4*2+p3*2+p2*2+p1+p0+q0+q1+q2, 4)
	in[4] = cRoundPowerOfTwo(p6*3+p5+p4+p3*2+p2*2+p1*2+p0+q0+q1+q2+q3, 4)
	in[5] = cRoundPowerOfTwo(p6*2+p5+p4+p3+p2*2+p1*2+p0*2+q0+q1+q2+q3+q4, 4)
	in[6] = cRoundPowerOfTwo(p6+p5+p4+p3+p2+p1*2+p0*2+q0*2+q1+q2+q3+q4+q5, 4)
	in[7] = cRoundPowerOfTwo(p5+p4+p3+p2+p1+p0*2+q0*2+q1*2+q2+q3+q4+q5+q6, 4)
	in[8] = cRoundPowerOfTwo(p4+p3+p2+p1+p0+q0*2+q1*2+q2*2+q3+q4+q5+q6*2, 4)
	in[9] = cRoundPowerOfTwo(p3+p2+p1+p0+q0+q1*2+q2*2+q3*2+q4+q5+q6*3, 4)
	in[10] = cRoundPowerOfTwo(p2+p1+p0+q0+q1+q2*2+q3*2+q4*2+q5+q6*4, 4)
	in[11] = cRoundPowerOfTwo(p1+p0+q0+q1+q2+q3*2+q4*2+q5*2+q6*5, 4)
	in[12] = cRoundPowerOfTwo(p0+q0+q1+q2+q3+q4*2+q5*2+q6*7, 4)
	return in
}

func BenchmarkFilter14EdgeHorizontal(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter14Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds)
	}
}

func BenchmarkFilter14EdgeVertical(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter14Edge(plane, 1, 8, EdgeVertical, 32, 0, 64, thresholds)
	}
}

// filter4DispatchParamsCorpus returns a spread of threshold configurations that
// exercise every branch of needsFilter4 and the hev split, including the all-
// pass and all-reject extremes for 8/10/12-bit (the scale only enters through
// the centre/clamp constants, which the SIMD path fixes at the 8-bit values; we
// therefore also assert the 8-bit kernel directly).
func filter4DispatchParamsCorpus() []filter4Params {
	const scale = 1 // the SIMD path is 8-bit only here.
	mk := func(limit, blimit, hev int) filter4Params {
		return filter4Params{
			limit:  int16(limit * scale),
			blimit: int16(blimit * scale),
			hev:    int16(hev * scale),
			min:    -128 * scale,
			max:    128*scale - 1,
			center: 128 * scale,
		}
	}
	return []filter4Params{
		mk(0, 0, 0),
		mk(1, 1, 0),
		mk(2, 5, 1),
		mk(8, 20, 4),
		mk(16, 40, 8),
		mk(32, 80, 16),
		mk(63, 128, 32),
		mk(255, 510, 0),   // effectively always filter, never hev
		mk(255, 510, 255), // always filter, always non-hev
	}
}

// runFilter4Kernel materialises a flat 8-bit buffer wide enough for the chosen
// length, runs both the reference and the dispatched kernel on independent
// copies, and reports the first mismatching byte. The buffer layout mirrors a
// horizontal edge: outer == 1 (positions are contiguous) and step == stride so
// the four taps live on separate rows, exactly as the decoder presents them.
func runFilter4Kernel(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 64 // >= length; row stride in bytes
	const rows = 8    // p3..q3 headroom; we use rows 2..5 for p1,p0,q0,q1
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	for i := range base {
		base[i] = byte(rng.Intn(256))
	}

	step := stride
	outer := 1
	q0Base := 3*stride + 0 // q0 row is row 3; p1 row 1, p0 row 2, q1 row 4

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	filter4EdgePureGo(want, q0Base, step, outer, length, params)
	filter4EdgeImpl(got, q0Base, step, outer, length, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				seed, length, params, i, got[i], want[i])
		}
	}
}

// TestFilter4DispatchMatchesPureGo is the bit-exactness guard for the dispatched
// narrow deblocking kernel. It drives the resolved dispatch slot (which routes
// through the Go SIMD kernel) against the canonical pure-Go reference over a
// spread of edge lengths — including non-multiples of eight to exercise the
// scalar tail — and threshold configurations. Every output byte must match.
func TestFilter4DispatchMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 63}
	var seed int64 = 1
	for _, params := range filter4DispatchParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter4Kernel(t, seed, length, params)
				seed++
			}
		}
	}
}

// runFilter4KernelVertical materialises a flat 8-bit buffer and runs both the
// reference and the dispatched kernel along a vertical edge: the four taps are
// one byte apart (step == 1) and successive positions advance by the row stride
// (outer == stride). This exercises the gather/scatter vertical SIMD path.
func runFilter4KernelVertical(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 64
	const rows = 80
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	for i := range base {
		base[i] = byte(rng.Intn(256))
	}

	step := 1
	outer := stride
	q0Base := 8 // column 8 leaves >=2 bytes of tap headroom on the left

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	filter4EdgePureGo(want, q0Base, step, outer, length, params)
	filter4EdgeImpl(got, q0Base, step, outer, length, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("vertical seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				seed, length, params, i, got[i], want[i])
		}
	}
}

// TestFilter4DispatchVerticalMatchesPureGo is the bit-exactness guard for the
// vertical-edge narrow kernel (gather/scatter SIMD) against the
// pure-Go reference over a spread of edge lengths and threshold configurations.
func TestFilter4DispatchVerticalMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 63}
	var seed int64 = 9000
	for _, params := range filter4DispatchParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter4KernelVertical(t, seed, length, params)
				seed++
			}
		}
	}
}

// TestFilter4DispatchMatchesPureGoForcedPureGo confirms the differential holds
// when the dispatch slot is pinned to the pure-Go branch, so the test still has
// meaning on a SIMD host where the slot would otherwise always pick the SIMD
// kernel. It re-binds the slot for the duration of the check.
func TestFilter4DispatchMatchesPureGoForcedPureGo(t *testing.T) {
	prev := filter4EdgeImpl
	filter4EdgeImpl = filter4EdgePureGo
	defer func() { filter4EdgeImpl = prev }()

	for i, params := range filter4DispatchParamsCorpus() {
		runFilter4Kernel(t, int64(1000+i), 16, params)
	}
}

// filter4Dispatch16ParamsCorpus mirrors filter4DispatchParamsCorpus but at
// 10- and 12-bit scale, exercising the centre/clamp constants the 16-bit kernel
// reads from its context.
func filter4Dispatch16ParamsCorpus() []filter4Params {
	var out []filter4Params
	for _, shift := range []int{2, 4} { // 10-bit, 12-bit
		scale := 1 << shift
		mk := func(limit, blimit, hev int) filter4Params {
			return filter4Params{
				limit:  int16(limit * scale),
				blimit: int16(blimit * scale),
				hev:    int16(hev * scale),
				min:    int16(-128 * scale),
				max:    int16(128*scale - 1),
				center: int16(128 * scale),
			}
		}
		out = append(out,
			mk(0, 0, 0),
			mk(2, 5, 1),
			mk(16, 40, 8),
			mk(63, 128, 32),
			mk(255, 510, 0),
			mk(255, 510, 255),
		)
	}
	return out
}

// runFilter4Kernel16 materialises a two-byte plane buffer, runs both the
// reference and the dispatched 16-bit kernel on independent copies, and reports
// the first mismatching byte. outer == 2 (contiguous two-byte positions) and
// step == stride so the four taps live on separate rows.
func runFilter4Kernel16(t *testing.T, seed int64, length int, params filter4Params) {
	t.Helper()
	const strideSamples = 64
	const stride = strideSamples * 2
	const rows = 8
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	maxVal := int(params.center)*2 - 1
	for i := 0; i+1 < len(base); i += 2 {
		v := rng.Intn(maxVal + 1)
		base[i] = byte(v)
		base[i+1] = byte(v >> 8)
	}
	step := stride
	outer := 2
	q0Base := 3*stride + 0

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	filter4Edge16PureGo(want, q0Base, step, outer, length, params)
	filter4Edge16Impl(got, q0Base, step, outer, length, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				seed, length, params, i, got[i], want[i])
		}
	}
}

// TestFilter4Dispatch16MatchesPureGo is the bit-exactness guard for the
// dispatched 10/12-bit narrow kernel (Go SIMD) against the pure-Go
// reference over a spread of edge lengths and scaled threshold configurations.
func TestFilter4Dispatch16MatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 63}
	var seed int64 = 5000
	for _, params := range filter4Dispatch16ParamsCorpus() {
		for _, length := range lengths {
			for rep := 0; rep < 4; rep++ {
				runFilter4Kernel16(t, seed, length, params)
				seed++
			}
		}
	}
}

// TestFilter4DispatchIsZeroAlloc protects the hot-path contract that the
// dispatched filter does not allocate per edge.
func TestFilter4DispatchIsZeroAlloc(t *testing.T) {
	const stride = 64
	base := make([]byte, stride*8)
	rng := rand.New(rand.NewSource(7))
	for i := range base {
		base[i] = byte(rng.Intn(256))
	}
	params := filter4Params{limit: 16, blimit: 40, hev: 8, min: -128, max: 127, center: 128}
	q0Base := 3 * stride
	allocs := testing.AllocsPerRun(1000, func() {
		filter4EdgeImpl(base, q0Base, stride, 1, 56, params)
	})
	if allocs != 0 {
		t.Fatalf("dispatch allocated %f times per call", allocs)
	}
}

func TestFilter4EdgeHorizontal8Bit(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for x := range 4 {
		setSample(plane, 1, x, 0, 90)
		setSample(plane, 1, x, 1, 90)
		setSample(plane, 1, x, 2, 100)
		setSample(plane, 1, x, 3, 100)
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter4Edge(plane, 1, 8, EdgeHorizontal, 0, 2, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for x := range 4 {
		if got := getSample(plane, 1, x, 0); got != 92 {
			t.Fatalf("p1 sample=%d want 92", got)
		}
		if got := getSample(plane, 1, x, 1); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, x, 2); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, x, 3); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
	}
}

func TestFilter4EdgeVerticalHighVariance8Bit(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for y := range 4 {
		setSample(plane, 1, 0, y, 70)
		setSample(plane, 1, 1, y, 90)
		setSample(plane, 1, 2, y, 100)
		setSample(plane, 1, 3, y, 130)
	}

	thresholds := Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3}
	if err := Filter4Edge(plane, 1, 8, EdgeVertical, 2, 0, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for y := range 4 {
		if got := getSample(plane, 1, 0, y); got != 70 {
			t.Fatalf("p1 sample=%d want 70", got)
		}
		if got := getSample(plane, 1, 1, y); got != 86 {
			t.Fatalf("p0 sample=%d want 86", got)
		}
		if got := getSample(plane, 1, 2, y); got != 104 {
			t.Fatalf("q0 sample=%d want 104", got)
		}
		if got := getSample(plane, 1, 3, y); got != 130 {
			t.Fatalf("q1 sample=%d want 130", got)
		}
	}
}

func TestFilter4EdgeHighBitDepth(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		plane := testPlane(4, 4, 2, 8)
		for x := range 4 {
			setSample(plane, 2, x, 0, 900)
			setSample(plane, 2, x, 1, 900)
			setSample(plane, 2, x, 2, 940)
			setSample(plane, 2, x, 3, 940)
		}

		thresholds := Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3}
		if err := Filter4Edge(plane, 2, bitDepth, EdgeHorizontal, 0, 2, 4, thresholds); err != nil {
			t.Fatal(err)
		}

		for x := range 4 {
			if got := getSample(plane, 2, x, 0); got != 908 {
				t.Fatalf("bitdepth=%d p1 sample=%d want 908", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 1); got != 915 {
				t.Fatalf("bitdepth=%d p0 sample=%d want 915", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 2); got != 925 {
				t.Fatalf("bitdepth=%d q0 sample=%d want 925", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 3); got != 932 {
				t.Fatalf("bitdepth=%d q1 sample=%d want 932", bitDepth, got)
			}
		}
	}
}

func TestFilter4EdgeLeavesMaskedSamples(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	for x := range 4 {
		setSample(plane, 1, x, 0, 10)
		setSample(plane, 1, x, 1, 10)
		setSample(plane, 1, x, 2, 200)
		setSample(plane, 1, x, 3, 200)
	}
	if err := Filter4Edge(plane, 1, 8, EdgeHorizontal, 0, 2, 4, Thresholds{Limit: 63, BlockLimit: 20}); err != nil {
		t.Fatal(err)
	}
	for x := range 4 {
		if got := getSample(plane, 1, x, 1); got != 10 {
			t.Fatalf("p0 sample=%d want 10", got)
		}
		if got := getSample(plane, 1, x, 2); got != 200 {
			t.Fatalf("q0 sample=%d want 200", got)
		}
	}
}

func TestFilter4EdgeRejectsInvalidInputs(t *testing.T) {
	plane := testPlane(4, 4, 1, 4)
	tests := []struct {
		name           string
		plane          frame.Plane
		bytesPerSample int
		bitDepth       uint8
		edge           Edge
		x              int32
		y              int32
		length         int32
	}{
		{name: "sample width", plane: plane, bytesPerSample: 2, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 2, length: 4},
		{name: "edge", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: Edge(9), x: 0, y: 2, length: 4},
		{name: "length", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 2},
		{name: "horizontal top", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 1, length: 4},
		{name: "horizontal width", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 1, y: 2, length: 4},
		{name: "vertical left", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 1, y: 0, length: 4},
		{name: "vertical height", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 2, y: 1, length: 4},
	}
	for _, tt := range tests {
		err := Filter4Edge(tt.plane, tt.bytesPerSample, tt.bitDepth, tt.edge, tt.x, tt.y, tt.length, Thresholds{})
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestFilter4EdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, 100)
		}
	}
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := Filter4Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Filter4Edge allocated: %f", allocs)
	}
}

func FuzzFilter4Edge(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), uint8(20), uint8(25), uint8(10))
	f.Add(uint8(10), uint8(10), uint8(1), uint8(63), uint8(132), uint8(3))
	f.Fuzz(func(t *testing.T, rawWidth uint8, rawHeight uint8, rawEdge uint8, rawLimit uint8, rawBlockLimit uint8, rawHEV uint8) {
		width := int(rawWidth%32) + 4
		height := int(rawHeight%32) + 4
		edge := Edge(rawEdge & 1)
		bytesPerSample := 1
		bitDepth := uint8(8)
		stride := width
		max := uint16(0xff)
		if rawEdge&2 != 0 {
			bytesPerSample = 2
			bitDepth = 10
			if rawEdge&4 != 0 {
				bitDepth = 12
			}
			stride = width * bytesPerSample
			max = uint16((1 << bitDepth) - 1)
		}
		plane := testPlane(width, height, bytesPerSample, stride)
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				setSample(plane, bytesPerSample, x, y, uint16((x*19+y*23)&int(max)))
			}
		}
		x := 2
		y := 2
		length := width - x
		if edge == EdgeVertical {
			length = height - y
		}
		err := Filter4Edge(plane, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), Thresholds{
			Limit:            rawLimit,
			BlockLimit:       rawBlockLimit,
			HighEdgeVariance: rawHEV,
		})
		if err != nil {
			t.Fatal(err)
		}
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				if got := getSample(plane, bytesPerSample, x, y); got > max {
					t.Fatalf("sample(%d,%d)=%d max=%d", x, y, got, max)
				}
			}
		}
	})
}

func BenchmarkFilter4EdgeHorizontal(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter4Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds)
	}
}

func BenchmarkFilter4EdgeVertical(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter4Edge(plane, 1, 8, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func testPlane(width int, height int, bytesPerSample int, stride int) frame.Plane {
	return frame.Plane{
		Pix:    make([]byte, stride*height),
		Stride: stride,
		Width:  width,
		Height: height,
	}
}

func setSample(plane frame.Plane, bytesPerSample int, x int, y int, value uint16) {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		plane.Pix[offset] = byte(value)
		return
	}
	plane.Pix[offset] = byte(value)
	plane.Pix[offset+1] = byte(value >> 8)
}

func getSample(plane frame.Plane, bytesPerSample int, x int, y int) uint16 {
	offset := y*plane.Stride + x*bytesPerSample
	if bytesPerSample == 1 {
		return uint16(plane.Pix[offset])
	}
	return uint16(plane.Pix[offset]) | uint16(plane.Pix[offset+1])<<8
}

func TestFilter6EdgeHorizontalFlat8Bit(t *testing.T) {
	plane := testPlane(4, 6, 1, 4)
	for x := range 4 {
		setSample(plane, 1, x, 0, 90)
		setSample(plane, 1, x, 1, 90)
		setSample(plane, 1, x, 2, 90)
		setSample(plane, 1, x, 3, 100)
		setSample(plane, 1, x, 4, 100)
		setSample(plane, 1, x, 5, 100)
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter6Edge(plane, 1, 8, EdgeHorizontal, 0, 3, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for x := range 4 {
		if got := getSample(plane, 1, x, 0); got != 90 {
			t.Fatalf("p2 sample=%d want 90", got)
		}
		if got := getSample(plane, 1, x, 1); got != 91 {
			t.Fatalf("p1 sample=%d want 91", got)
		}
		if got := getSample(plane, 1, x, 2); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, x, 3); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, x, 4); got != 99 {
			t.Fatalf("q1 sample=%d want 99", got)
		}
		if got := getSample(plane, 1, x, 5); got != 100 {
			t.Fatalf("q2 sample=%d want 100", got)
		}
	}
}

func TestFilter6EdgeVerticalFallback8Bit(t *testing.T) {
	plane := testPlane(6, 4, 1, 6)
	for y := range 4 {
		setSample(plane, 1, 0, y, 80)
		setSample(plane, 1, 1, y, 90)
		setSample(plane, 1, 2, y, 90)
		setSample(plane, 1, 3, y, 100)
		setSample(plane, 1, 4, y, 100)
		setSample(plane, 1, 5, y, 100)
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter6Edge(plane, 1, 8, EdgeVertical, 3, 0, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for y := range 4 {
		if got := getSample(plane, 1, 0, y); got != 80 {
			t.Fatalf("p2 sample=%d want 80", got)
		}
		if got := getSample(plane, 1, 1, y); got != 92 {
			t.Fatalf("p1 sample=%d want 92", got)
		}
		if got := getSample(plane, 1, 2, y); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, 3, y); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, 4, y); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
		if got := getSample(plane, 1, 5, y); got != 100 {
			t.Fatalf("q2 sample=%d want 100", got)
		}
	}
}

func TestFilter6EdgeHighBitDepth(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		plane := testPlane(4, 6, 2, 8)
		for x := range 4 {
			setSample(plane, 2, x, 0, 900)
			setSample(plane, 2, x, 1, 900)
			setSample(plane, 2, x, 2, 900)
			setSample(plane, 2, x, 3, 940)
			setSample(plane, 2, x, 4, 940)
			setSample(plane, 2, x, 5, 940)
		}

		thresholds := Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3}
		if err := Filter6Edge(plane, 2, bitDepth, EdgeHorizontal, 0, 3, 4, thresholds); err != nil {
			t.Fatal(err)
		}

		for x := range 4 {
			if got := getSample(plane, 2, x, 1); got != 905 {
				t.Fatalf("bitdepth=%d p1 sample=%d want 905", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 2); got != 915 {
				t.Fatalf("bitdepth=%d p0 sample=%d want 915", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 3); got != 925 {
				t.Fatalf("bitdepth=%d q0 sample=%d want 925", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 4); got != 935 {
				t.Fatalf("bitdepth=%d q1 sample=%d want 935", bitDepth, got)
			}
		}
	}
}

func TestFilter6EdgeMatchesCReference(t *testing.T) {
	tests := []struct {
		name       string
		edge       Edge
		bitDepth   uint8
		thresholds Thresholds
		samples    [][6]int
	}{
		{
			name:       "horizontal 8 bit",
			edge:       EdgeHorizontal,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][6]int{
				{90, 90, 90, 100, 100, 100},
				{80, 90, 90, 100, 100, 100},
				{70, 70, 90, 100, 130, 130},
				{10, 10, 10, 200, 200, 200},
			},
		},
		{
			name:       "vertical 8 bit",
			edge:       EdgeVertical,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][6]int{
				{90, 90, 90, 100, 100, 100},
				{80, 90, 90, 100, 100, 100},
				{70, 70, 90, 100, 130, 130},
				{10, 10, 10, 200, 200, 200},
			},
		},
		{
			name:       "horizontal high bit depth",
			edge:       EdgeHorizontal,
			bitDepth:   10,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][6]int{
				{900, 900, 900, 940, 940, 940},
				{850, 900, 900, 940, 940, 940},
				{200, 200, 200, 900, 900, 900},
			},
		},
		{
			name:       "vertical high bit depth",
			edge:       EdgeVertical,
			bitDepth:   12,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][6]int{
				{900, 900, 900, 940, 940, 940},
				{850, 900, 900, 940, 940, 940},
				{200, 200, 200, 900, 900, 900},
			},
		},
	}

	for _, tt := range tests {
		bytesPerSample := 1
		if tt.bitDepth != 8 {
			bytesPerSample = 2
		}

		got := filter6ReferencePlane(tt.edge, bytesPerSample, tt.samples)
		want := cloneTestPlane(got)
		for i, sample := range tt.samples {
			ref := cReferenceFilter6(sample, tt.bitDepth, tt.thresholds)
			writeFilter6ReferenceSample(want, bytesPerSample, tt.edge, i, ref)
		}

		x, y, length := int32(0), int32(3), int32(len(tt.samples))
		if tt.edge == EdgeVertical {
			x, y = 3, 0
		}
		if err := Filter6Edge(got, bytesPerSample, tt.bitDepth, tt.edge, x, y, length, tt.thresholds); err != nil {
			t.Fatalf("%s Filter6Edge: %v", tt.name, err)
		}
		if !bytes.Equal(got.Pix, want.Pix) {
			t.Fatalf("%s got=%v want=%v", tt.name, got.Pix, want.Pix)
		}
	}
}

func TestFilter6EdgeRejectsInvalidInputs(t *testing.T) {
	plane := testPlane(6, 6, 1, 6)
	tests := []struct {
		name           string
		plane          frame.Plane
		bytesPerSample int
		bitDepth       uint8
		edge           Edge
		x              int32
		y              int32
		length         int32
	}{
		{name: "sample width", plane: plane, bytesPerSample: 2, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 3, length: 4},
		{name: "edge", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: Edge(9), x: 0, y: 3, length: 4},
		{name: "length", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 3},
		{name: "horizontal top", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 2, length: 4},
		{name: "horizontal bottom", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 4, length: 4},
		{name: "vertical left", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 2, y: 0, length: 4},
		{name: "vertical right", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 4, y: 0, length: 4},
	}
	for _, tt := range tests {
		err := Filter6Edge(tt.plane, tt.bytesPerSample, tt.bitDepth, tt.edge, tt.x, tt.y, tt.length, Thresholds{})
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestFilter6EdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, 100)
		}
	}
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := Filter6Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Filter6Edge allocated: %f", allocs)
	}
}

func FuzzFilter6Edge(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), uint8(20), uint8(25), uint8(10))
	f.Add(uint8(10), uint8(10), uint8(1), uint8(63), uint8(132), uint8(3))
	f.Fuzz(func(t *testing.T, rawWidth uint8, rawHeight uint8, rawEdge uint8, rawLimit uint8, rawBlockLimit uint8, rawHEV uint8) {
		width := int(rawWidth%32) + 6
		height := int(rawHeight%32) + 6
		edge := Edge(rawEdge & 1)
		bytesPerSample := 1
		bitDepth := uint8(8)
		stride := width
		max := uint16(0xff)
		if rawEdge&2 != 0 {
			bytesPerSample = 2
			bitDepth = 10
			if rawEdge&4 != 0 {
				bitDepth = 12
			}
			stride = width * bytesPerSample
			max = uint16((1 << bitDepth) - 1)
		}
		plane := testPlane(width, height, bytesPerSample, stride)
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				setSample(plane, bytesPerSample, x, y, uint16((x*19+y*23)&int(max)))
			}
		}
		x := 3
		y := 3
		length := width - x - 2
		if edge == EdgeVertical {
			length = height - y
		}
		err := Filter6Edge(plane, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), Thresholds{
			Limit:            rawLimit,
			BlockLimit:       rawBlockLimit,
			HighEdgeVariance: rawHEV,
		})
		if err != nil {
			t.Fatal(err)
		}
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				if got := getSample(plane, bytesPerSample, x, y); got > max {
					t.Fatalf("sample(%d,%d)=%d max=%d", x, y, got, max)
				}
			}
		}
	})
}

func filter6ReferencePlane(edge Edge, bytesPerSample int, samples [][6]int) frame.Plane {
	if edge == EdgeHorizontal {
		plane := testPlane(len(samples), 6, bytesPerSample, len(samples)*bytesPerSample)
		for i, sample := range samples {
			writeFilter6ReferenceSample(plane, bytesPerSample, edge, i, sample)
		}
		return plane
	}
	plane := testPlane(6, len(samples), bytesPerSample, 6*bytesPerSample)
	for i, sample := range samples {
		writeFilter6ReferenceSample(plane, bytesPerSample, edge, i, sample)
	}
	return plane
}

func cloneTestPlane(src frame.Plane) frame.Plane {
	pix := make([]byte, len(src.Pix))
	copy(pix, src.Pix)
	return frame.Plane{
		Pix:    pix,
		Stride: src.Stride,
		Width:  src.Width,
		Height: src.Height,
	}
}

func writeFilter6ReferenceSample(plane frame.Plane, bytesPerSample int, edge Edge, index int, sample [6]int) {
	for i, value := range sample {
		if edge == EdgeHorizontal {
			setSample(plane, bytesPerSample, index, i, uint16(value))
			continue
		}
		setSample(plane, bytesPerSample, i, index, uint16(value))
	}
}

func cReferenceFilter6(in [6]int, bitDepth uint8, thresholds Thresholds) [6]int {
	scale := 1 << int(bitDepth-8)
	limit := int(thresholds.Limit) * scale
	blimit := int(thresholds.BlockLimit) * scale
	hev := int(thresholds.HighEdgeVariance) * scale
	min := -128 * scale
	max := 128*scale - 1
	center := 128 * scale

	p2, p1, p0 := in[0], in[1], in[2]
	q0, q1, q2 := in[3], in[4], in[5]
	if !cReferenceFilterMask3(limit, blimit, p2, p1, p0, q0, q1, q2) {
		return in
	}
	if cReferenceFlatMask3(scale, p2, p1, p0, q0, q1, q2) {
		in[1] = cRoundPowerOfTwo(p2*3+p1*2+p0*2+q0, 3)
		in[2] = cRoundPowerOfTwo(p2+p1*2+p0*2+q0*2+q1, 3)
		in[3] = cRoundPowerOfTwo(p1+p0*2+q0*2+q1*2+q2, 3)
		in[4] = cRoundPowerOfTwo(p0+q0*2+q1*2+q2*3, 3)
		return in
	}
	in[1], in[2], in[3], in[4] = cReferenceFilter4(p1, p0, q0, q1, hev, min, max, center)
	return in
}

func cReferenceFilterMask3(limit int, blimit int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int) bool {
	return cAbs(p2-p1) <= limit &&
		cAbs(p1-p0) <= limit &&
		cAbs(q1-q0) <= limit &&
		cAbs(q2-q1) <= limit &&
		cAbs(p0-q0)*2+cAbs(p1-q1)/2 <= blimit
}

func cReferenceFlatMask3(thresh int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int) bool {
	return cAbs(p1-p0) <= thresh &&
		cAbs(q1-q0) <= thresh &&
		cAbs(p2-p0) <= thresh &&
		cAbs(q2-q0) <= thresh
}

func cReferenceFilter4(p1 int, p0 int, q0 int, q1 int, hevThreshold int, min int, max int, center int) (int, int, int, int) {
	ps1 := p1 - center
	ps0 := p0 - center
	qs0 := q0 - center
	qs1 := q1 - center
	hev := cAbs(p1-p0) > hevThreshold || cAbs(q1-q0) > hevThreshold

	filter := 0
	if hev {
		filter = cSignedClamp(ps1-qs1, min, max)
	}
	filter = cSignedClamp(filter+3*(qs0-ps0), min, max)
	filter1 := cSignedClamp(filter+4, min, max) >> 3
	filter2 := cSignedClamp(filter+3, min, max) >> 3

	p0 = cSignedClamp(ps0+filter2, min, max) + center
	q0 = cSignedClamp(qs0-filter1, min, max) + center
	if !hev {
		outer := cRoundPowerOfTwo(filter1, 1)
		p1 = cSignedClamp(ps1+outer, min, max) + center
		q1 = cSignedClamp(qs1-outer, min, max) + center
	}
	return p1, p0, q0, q1
}

func cRoundPowerOfTwo(v int, n uint) int {
	return (v + (1 << (n - 1))) >> n
}

func cSignedClamp(v int, min int, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func cAbs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func BenchmarkFilter6EdgeHorizontal(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter6Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds)
	}
}

func BenchmarkFilter6EdgeVertical(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter6Edge(plane, 1, 8, EdgeVertical, 32, 0, 64, thresholds)
	}
}

func TestFilter8EdgeHorizontalFlat8Bit(t *testing.T) {
	plane := testPlane(4, 8, 1, 4)
	for x := range 4 {
		for y := range 4 {
			setSample(plane, 1, x, y, 90)
		}
		for y := 4; y < 8; y++ {
			setSample(plane, 1, x, y, 100)
		}
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter8Edge(plane, 1, 8, EdgeHorizontal, 0, 4, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for x := range 4 {
		if got := getSample(plane, 1, x, 0); got != 90 {
			t.Fatalf("p3 sample=%d want 90", got)
		}
		if got := getSample(plane, 1, x, 1); got != 91 {
			t.Fatalf("p2 sample=%d want 91", got)
		}
		if got := getSample(plane, 1, x, 2); got != 93 {
			t.Fatalf("p1 sample=%d want 93", got)
		}
		if got := getSample(plane, 1, x, 3); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, x, 4); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, x, 5); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
		if got := getSample(plane, 1, x, 6); got != 99 {
			t.Fatalf("q2 sample=%d want 99", got)
		}
		if got := getSample(plane, 1, x, 7); got != 100 {
			t.Fatalf("q3 sample=%d want 100", got)
		}
	}
}

func TestFilter8EdgeVerticalFallback8Bit(t *testing.T) {
	plane := testPlane(8, 4, 1, 8)
	for y := range 4 {
		setSample(plane, 1, 0, y, 80)
		setSample(plane, 1, 1, y, 90)
		setSample(plane, 1, 2, y, 90)
		setSample(plane, 1, 3, y, 90)
		setSample(plane, 1, 4, y, 100)
		setSample(plane, 1, 5, y, 100)
		setSample(plane, 1, 6, y, 100)
		setSample(plane, 1, 7, y, 100)
	}

	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	if err := Filter8Edge(plane, 1, 8, EdgeVertical, 4, 0, 4, thresholds); err != nil {
		t.Fatal(err)
	}

	for y := range 4 {
		if got := getSample(plane, 1, 0, y); got != 80 {
			t.Fatalf("p3 sample=%d want 80", got)
		}
		if got := getSample(plane, 1, 1, y); got != 90 {
			t.Fatalf("p2 sample=%d want 90", got)
		}
		if got := getSample(plane, 1, 2, y); got != 92 {
			t.Fatalf("p1 sample=%d want 92", got)
		}
		if got := getSample(plane, 1, 3, y); got != 94 {
			t.Fatalf("p0 sample=%d want 94", got)
		}
		if got := getSample(plane, 1, 4, y); got != 96 {
			t.Fatalf("q0 sample=%d want 96", got)
		}
		if got := getSample(plane, 1, 5, y); got != 98 {
			t.Fatalf("q1 sample=%d want 98", got)
		}
		if got := getSample(plane, 1, 6, y); got != 100 {
			t.Fatalf("q2 sample=%d want 100", got)
		}
		if got := getSample(plane, 1, 7, y); got != 100 {
			t.Fatalf("q3 sample=%d want 100", got)
		}
	}
}

func TestFilter8EdgeHighBitDepth(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		plane := testPlane(4, 8, 2, 8)
		for x := range 4 {
			for y := range 4 {
				setSample(plane, 2, x, y, 900)
			}
			for y := 4; y < 8; y++ {
				setSample(plane, 2, x, y, 940)
			}
		}

		thresholds := Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3}
		if err := Filter8Edge(plane, 2, bitDepth, EdgeHorizontal, 0, 4, 4, thresholds); err != nil {
			t.Fatal(err)
		}

		for x := range 4 {
			if got := getSample(plane, 2, x, 1); got != 905 {
				t.Fatalf("bitdepth=%d p2 sample=%d want 905", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 2); got != 910 {
				t.Fatalf("bitdepth=%d p1 sample=%d want 910", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 3); got != 915 {
				t.Fatalf("bitdepth=%d p0 sample=%d want 915", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 4); got != 925 {
				t.Fatalf("bitdepth=%d q0 sample=%d want 925", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 5); got != 930 {
				t.Fatalf("bitdepth=%d q1 sample=%d want 930", bitDepth, got)
			}
			if got := getSample(plane, 2, x, 6); got != 935 {
				t.Fatalf("bitdepth=%d q2 sample=%d want 935", bitDepth, got)
			}
		}
	}
}

func TestFilter8EdgeMatchesCReference(t *testing.T) {
	tests := []struct {
		name       string
		edge       Edge
		bitDepth   uint8
		thresholds Thresholds
		samples    [][8]int
	}{
		{
			name:       "horizontal 8 bit",
			edge:       EdgeHorizontal,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][8]int{
				{90, 90, 90, 90, 100, 100, 100, 100},
				{80, 90, 90, 90, 100, 100, 100, 100},
				{70, 70, 70, 90, 100, 130, 130, 130},
				{10, 10, 10, 10, 200, 200, 200, 200},
			},
		},
		{
			name:       "vertical 8 bit",
			edge:       EdgeVertical,
			bitDepth:   8,
			thresholds: Thresholds{Limit: 63, BlockLimit: 100, HighEdgeVariance: 3},
			samples: [][8]int{
				{90, 90, 90, 90, 100, 100, 100, 100},
				{80, 90, 90, 90, 100, 100, 100, 100},
				{70, 70, 70, 90, 100, 130, 130, 130},
				{10, 10, 10, 10, 200, 200, 200, 200},
			},
		},
		{
			name:       "horizontal high bit depth",
			edge:       EdgeHorizontal,
			bitDepth:   10,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][8]int{
				{900, 900, 900, 900, 940, 940, 940, 940},
				{850, 900, 900, 900, 940, 940, 940, 940},
				{200, 200, 200, 200, 900, 900, 900, 900},
			},
		},
		{
			name:       "vertical high bit depth",
			edge:       EdgeVertical,
			bitDepth:   12,
			thresholds: Thresholds{Limit: 20, BlockLimit: 30, HighEdgeVariance: 3},
			samples: [][8]int{
				{900, 900, 900, 900, 940, 940, 940, 940},
				{850, 900, 900, 900, 940, 940, 940, 940},
				{200, 200, 200, 200, 900, 900, 900, 900},
			},
		},
	}

	for _, tt := range tests {
		bytesPerSample := 1
		if tt.bitDepth != 8 {
			bytesPerSample = 2
		}

		got := filter8ReferencePlane(tt.edge, bytesPerSample, tt.samples)
		want := cloneTestPlane(got)
		for i, sample := range tt.samples {
			ref := cReferenceFilter8(sample, tt.bitDepth, tt.thresholds)
			writeFilter8ReferenceSample(want, bytesPerSample, tt.edge, i, ref)
		}

		x, y, length := int32(0), int32(4), int32(len(tt.samples))
		if tt.edge == EdgeVertical {
			x, y = 4, 0
		}
		if err := Filter8Edge(got, bytesPerSample, tt.bitDepth, tt.edge, x, y, length, tt.thresholds); err != nil {
			t.Fatalf("%s Filter8Edge: %v", tt.name, err)
		}
		if !bytes.Equal(got.Pix, want.Pix) {
			t.Fatalf("%s got=%v want=%v", tt.name, got.Pix, want.Pix)
		}
	}
}

func TestFilter8EdgeRejectsInvalidInputs(t *testing.T) {
	plane := testPlane(8, 8, 1, 8)
	tests := []struct {
		name           string
		plane          frame.Plane
		bytesPerSample int
		bitDepth       uint8
		edge           Edge
		x              int32
		y              int32
		length         int32
	}{
		{name: "sample width", plane: plane, bytesPerSample: 2, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 4, length: 4},
		{name: "edge", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: Edge(9), x: 0, y: 4, length: 4},
		{name: "length", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 4},
		{name: "horizontal top", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 3, length: 4},
		{name: "horizontal bottom", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeHorizontal, x: 0, y: 5, length: 4},
		{name: "vertical left", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 3, y: 0, length: 4},
		{name: "vertical right", plane: plane, bytesPerSample: 1, bitDepth: 8, edge: EdgeVertical, x: 5, y: 0, length: 4},
	}
	for _, tt := range tests {
		err := Filter8Edge(tt.plane, tt.bytesPerSample, tt.bitDepth, tt.edge, tt.x, tt.y, tt.length, Thresholds{})
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestFilter8EdgeAllocs(t *testing.T) {
	plane := testPlane(64, 64, 1, 64)
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			setSample(plane, 1, x, y, 100)
		}
	}
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	allocs := testing.AllocsPerRun(1000, func() {
		if err := Filter8Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("Filter8Edge allocated: %f", allocs)
	}
}

func FuzzFilter8Edge(f *testing.F) {
	f.Add(uint8(8), uint8(8), uint8(0), uint8(20), uint8(25), uint8(10))
	f.Add(uint8(10), uint8(10), uint8(1), uint8(63), uint8(132), uint8(3))
	f.Fuzz(func(t *testing.T, rawWidth uint8, rawHeight uint8, rawEdge uint8, rawLimit uint8, rawBlockLimit uint8, rawHEV uint8) {
		width := int(rawWidth%32) + 8
		height := int(rawHeight%32) + 8
		edge := Edge(rawEdge & 1)
		bytesPerSample := 1
		bitDepth := uint8(8)
		stride := width
		max := uint16(0xff)
		if rawEdge&2 != 0 {
			bytesPerSample = 2
			bitDepth = 10
			if rawEdge&4 != 0 {
				bitDepth = 12
			}
			stride = width * bytesPerSample
			max = uint16((1 << bitDepth) - 1)
		}
		plane := testPlane(width, height, bytesPerSample, stride)
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				setSample(plane, bytesPerSample, x, y, uint16((x*19+y*23)&int(max)))
			}
		}
		x := 4
		y := 4
		length := width - x
		if edge == EdgeVertical {
			length = height - y
		}
		err := Filter8Edge(plane, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), Thresholds{
			Limit:            rawLimit,
			BlockLimit:       rawBlockLimit,
			HighEdgeVariance: rawHEV,
		})
		if err != nil {
			t.Fatal(err)
		}
		for y := 0; y < plane.Height; y++ {
			for x := 0; x < plane.Width; x++ {
				if got := getSample(plane, bytesPerSample, x, y); got > max {
					t.Fatalf("sample(%d,%d)=%d max=%d", x, y, got, max)
				}
			}
		}
	})
}

func filter8ReferencePlane(edge Edge, bytesPerSample int, samples [][8]int) frame.Plane {
	if edge == EdgeHorizontal {
		plane := testPlane(len(samples), 8, bytesPerSample, len(samples)*bytesPerSample)
		for i, sample := range samples {
			writeFilter8ReferenceSample(plane, bytesPerSample, edge, i, sample)
		}
		return plane
	}
	plane := testPlane(8, len(samples), bytesPerSample, 8*bytesPerSample)
	for i, sample := range samples {
		writeFilter8ReferenceSample(plane, bytesPerSample, edge, i, sample)
	}
	return plane
}

func writeFilter8ReferenceSample(plane frame.Plane, bytesPerSample int, edge Edge, index int, sample [8]int) {
	for i, value := range sample {
		if edge == EdgeHorizontal {
			setSample(plane, bytesPerSample, index, i, uint16(value))
			continue
		}
		setSample(plane, bytesPerSample, i, index, uint16(value))
	}
}

func cReferenceFilter8(in [8]int, bitDepth uint8, thresholds Thresholds) [8]int {
	scale := 1 << int(bitDepth-8)
	limit := int(thresholds.Limit) * scale
	blimit := int(thresholds.BlockLimit) * scale
	hev := int(thresholds.HighEdgeVariance) * scale
	min := -128 * scale
	max := 128*scale - 1
	center := 128 * scale

	p3, p2, p1, p0 := in[0], in[1], in[2], in[3]
	q0, q1, q2, q3 := in[4], in[5], in[6], in[7]
	if !cReferenceFilterMask4(limit, blimit, p3, p2, p1, p0, q0, q1, q2, q3) {
		return in
	}
	if cReferenceFlatMask4(scale, p3, p2, p1, p0, q0, q1, q2, q3) {
		in[1] = cRoundPowerOfTwo(p3+p3+p3+2*p2+p1+p0+q0, 3)
		in[2] = cRoundPowerOfTwo(p3+p3+p2+2*p1+p0+q0+q1, 3)
		in[3] = cRoundPowerOfTwo(p3+p2+p1+2*p0+q0+q1+q2, 3)
		in[4] = cRoundPowerOfTwo(p2+p1+p0+2*q0+q1+q2+q3, 3)
		in[5] = cRoundPowerOfTwo(p1+p0+q0+2*q1+q2+q3+q3, 3)
		in[6] = cRoundPowerOfTwo(p0+q0+q1+2*q2+q3+q3+q3, 3)
		return in
	}
	in[2], in[3], in[4], in[5] = cReferenceFilter4(p1, p0, q0, q1, hev, min, max, center)
	return in
}

func cReferenceFilterMask4(limit int, blimit int, p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int) bool {
	return cAbs(p3-p2) <= limit &&
		cAbs(p2-p1) <= limit &&
		cAbs(p1-p0) <= limit &&
		cAbs(q1-q0) <= limit &&
		cAbs(q2-q1) <= limit &&
		cAbs(q3-q2) <= limit &&
		cAbs(p0-q0)*2+cAbs(p1-q1)/2 <= blimit
}

func cReferenceFlatMask4(thresh int, p3 int, p2 int, p1 int, p0 int, q0 int, q1 int, q2 int, q3 int) bool {
	return cAbs(p1-p0) <= thresh &&
		cAbs(q1-q0) <= thresh &&
		cAbs(p2-p0) <= thresh &&
		cAbs(q2-q0) <= thresh &&
		cAbs(p3-p0) <= thresh &&
		cAbs(q3-q0) <= thresh
}

func BenchmarkFilter8EdgeHorizontal(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter8Edge(plane, 1, 8, EdgeHorizontal, 0, 32, 64, thresholds)
	}
}

func BenchmarkFilter8EdgeVertical(b *testing.B) {
	plane := testPlane(64, 64, 1, 64)
	thresholds := Thresholds{Limit: 20, BlockLimit: 25, HighEdgeVariance: 10}
	b.ReportAllocs()
	for b.Loop() {
		_ = Filter8Edge(plane, 1, 8, EdgeVertical, 32, 0, 64, thresholds)
	}
}

// wideFilterParamsCorpus mirrors filter4DispatchParamsCorpus but for the wide
// filters, exercising the needsFilter, flat, hev, and all-pass/all-reject
// branches at 8-bit (scale == 1).
func wideFilterParamsCorpus() []filter4Params {
	mk := func(limit, blimit, hev int) filter4Params {
		return filter4Params{
			limit:  int16(limit),
			blimit: int16(blimit),
			hev:    int16(hev),
			min:    -128,
			max:    127,
			center: 128,
		}
	}
	return []filter4Params{
		mk(0, 0, 0),
		mk(1, 1, 0),
		mk(2, 5, 1),
		mk(8, 20, 4),
		mk(16, 40, 8),
		mk(32, 80, 16),
		mk(63, 128, 32),
		mk(255, 510, 0),
		mk(255, 510, 255),
	}
}

type wideKernel struct {
	name string
	ref  func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	impl func() func(pix []byte, q0Base, step, outer, length, scale int, params filter4Params)
	pad  int // tap headroom rows on each side
}

func wideKernels() []wideKernel {
	return []wideKernel{
		{name: "filter6", ref: filter6EdgePureGo, impl: func() func([]byte, int, int, int, int, int, filter4Params) { return filter6EdgeImpl }, pad: 3},
		{name: "filter8", ref: filter8EdgePureGo, impl: func() func([]byte, int, int, int, int, int, filter4Params) { return filter8EdgeImpl }, pad: 4},
		{name: "filter14", ref: filter14EdgePureGo, impl: func() func([]byte, int, int, int, int, int, filter4Params) { return filter14EdgeImpl }, pad: 7},
	}
}

// fillWideContent fills buf with one of several content regimes. Pure-random
// content almost never satisfies the flat8in/flat8out predicates, so the flat
// and near-flat modes are essential to exercise filter8's flat branch and
// filter14's fourteen-tap wide path (where a divergence once slipped past a
// random-only test and only surfaced on a real monochrome frame).
func fillWideContent(buf []byte, rng *rand.Rand, mode int) {
	for i := range buf {
		switch mode {
		case 0:
			buf[i] = byte(rng.Intn(256)) // random
		case 1:
			buf[i] = byte(120 + rng.Intn(16)) // near-flat
		case 2:
			buf[i] = byte(127 + rng.Intn(3)) // very flat (wide path)
		case 3:
			buf[i] = byte(rng.Intn(4)) // flat near 0 (clamp edge)
		default:
			buf[i] = byte(252 + rng.Intn(4)) // flat near 255 (clamp edge)
		}
	}
}

func runWideKernel(t *testing.T, k wideKernel, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 128
	const rows = 16
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step := stride
	outer := 1
	q0Base := 8 * stride // q0 row 8 leaves >=7 rows of headroom either side

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	k.ref(want, q0Base, step, outer, length, 1, params)
	k.impl()(got, q0Base, step, outer, length, 1, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				k.name, seed, length, params, i, got[i], want[i])
		}
	}
}

// runWideKernelVertical mirrors runWideKernel for a vertical edge: the taps are
// one byte apart (step == 1) and successive positions advance by the row stride
// (outer == stride). This exercises the gather/scatter vertical path.
func runWideKernelVertical(t *testing.T, k wideKernel, seed int64, length int, params filter4Params) {
	t.Helper()
	const stride = 96
	const rows = 80
	rng := rand.New(rand.NewSource(seed))
	base := make([]byte, stride*rows)
	fillWideContent(base, rng, int(seed)%5)
	step := 1
	outer := stride
	q0Base := 16 // column 16 leaves >=7 bytes of tap headroom on the left

	want := append([]byte(nil), base...)
	got := append([]byte(nil), base...)

	k.ref(want, q0Base, step, outer, length, 1, params)
	k.impl()(got, q0Base, step, outer, length, 1, params)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s vertical seed=%d len=%d params=%+v idx=%d got=%d want=%d",
				k.name, seed, length, params, i, got[i], want[i])
		}
	}
}

// TestWideFilterDispatchMatchesPureGo is the bit-exactness guard for the
// dispatched six/eight/fourteen-sample kernels. It drives the resolved dispatch
// slot (Go SIMD where enabled) against the pure-Go reference over a spread of edge
// lengths (including non-multiples of eight to exercise the scalar tail) and
// threshold configurations. Every output byte must match.
func TestWideFilterDispatchMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	for _, k := range wideKernels() {
		var seed int64 = 1
		for _, params := range wideFilterParamsCorpus() {
			for _, length := range lengths {
				for rep := 0; rep < 4; rep++ {
					runWideKernel(t, k, seed, length, params)
					seed++
				}
			}
		}
	}
}

// TestWideFilterDispatchVerticalMatchesPureGo is the bit-exactness guard for the
// vertical-edge six/eight/fourteen-sample kernels (gather/scatter SIMD on
// arm64) against the pure-Go reference over the same length and threshold
// spread. Every output byte must match.
func TestWideFilterDispatchVerticalMatchesPureGo(t *testing.T) {
	lengths := []int{1, 3, 7, 8, 9, 15, 16, 17, 24, 31, 32, 48, 64}
	for _, k := range wideKernels() {
		var seed int64 = 7000
		for _, params := range wideFilterParamsCorpus() {
			for _, length := range lengths {
				for rep := 0; rep < 4; rep++ {
					runWideKernelVertical(t, k, seed, length, params)
					seed++
				}
			}
		}
	}
}

// TestWideFilterDispatchForcedPureGo confirms the differential holds when the
// dispatcher is forced onto the pure-Go branch, so the test still has meaning
// on a SIMD host where the slot would otherwise always pick the kernel.
func TestWideFilterDispatchForcedPureGo(t *testing.T) {
	prev6, prev8, prev14 := filter6EdgeImpl, filter8EdgeImpl, filter14EdgeImpl
	filter6EdgeImpl, filter8EdgeImpl, filter14EdgeImpl = filter6EdgePureGo, filter8EdgePureGo, filter14EdgePureGo
	defer func() { filter6EdgeImpl, filter8EdgeImpl, filter14EdgeImpl = prev6, prev8, prev14 }()

	for _, k := range wideKernels() {
		for i, params := range wideFilterParamsCorpus() {
			runWideKernel(t, k, int64(2000+i), 16, params)
		}
	}
}

func FuzzResolveLevel(f *testing.F) {
	f.Add(uint8(40), uint8(20), uint8(17), int8(-3), int16(-5), int8(-2), int8(3), uint8(0), uint8(4), uint8(1))
	f.Fuzz(func(t *testing.T, levelYV uint8, levelYH uint8, levelU uint8, deltaLF int8, segDelta int16, refDelta int8, modeDelta int8, edgeRaw uint8, refRaw uint8, modeRaw uint8) {
		edge := Edge(edgeRaw & 1)
		ref := refRaw % parser.RefFrames
		mode := ModeDeltaClass(modeRaw & 1)

		params := parser.LoopFilterParams{
			LevelY:              [2]uint8{levelYV % (MaxLevel + 1), levelYH % (MaxLevel + 1)},
			LevelU:              levelU % (MaxLevel + 1),
			ModeRefDeltaEnabled: true,
		}
		params.Deltas.Ref[ref] = refDelta
		params.Deltas.Mode[mode] = modeDelta

		seg := parser.SegmentationParams{Enabled: true}
		seg.Data.Segments[0].DeltaLFYV = segDelta
		seg.Data.Segments[0].DeltaLFYH = segDelta

		got, err := ResolveLevel(params, seg, LevelRequest{
			Plane:     PlaneY,
			Edge:      edge,
			SegmentID: 0,
			RefFrame:  ref,
			Mode:      mode,
			DeltaLF:   deltaLF,
		})
		if err != nil {
			t.Fatal(err)
		}
		if got > MaxLevel {
			t.Fatalf("level=%d", got)
		}
	})
}

func FuzzThresholdsForLevel(f *testing.F) {
	f.Add(uint8(16), uint8(0))
	f.Add(uint8(63), uint8(7))
	f.Fuzz(func(t *testing.T, level uint8, sharpness uint8) {
		thresholds, err := ThresholdsForLevel(level, sharpness)
		if level > MaxLevel || sharpness > MaxSharpness {
			if err == nil {
				t.Fatalf("ThresholdsForLevel(%d,%d) succeeded for invalid input", level, sharpness)
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if thresholds.Limit == 0 || thresholds.BlockLimit < thresholds.Limit || thresholds.HighEdgeVariance != level>>4 {
			t.Fatalf("thresholds=%+v level=%d sharpness=%d", thresholds, level, sharpness)
		}
	})
}

func TestHotStructSizes(t *testing.T) {
	tests := []struct {
		name string
		got  uintptr
		max  uintptr
	}{
		{name: "LevelRequest", got: unsafe.Sizeof(LevelRequest{}), max: 6},
		{name: "filter4Params", got: unsafe.Sizeof(filter4Params{}), max: 12},
	}
	for _, tt := range tests {
		t.Logf("%s size=%d max=%d", tt.name, tt.got, tt.max)
		if tt.got > tt.max {
			t.Fatalf("%s grew to %d bytes; max %d", tt.name, tt.got, tt.max)
		}
	}
}

func TestBaseLevelAndDeltaIndex(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY: [2]uint8{10, 20},
		LevelU: 30,
		LevelV: 40,
	}

	tests := []struct {
		name      string
		plane     Plane
		edge      Edge
		wantLevel uint8
		wantIndex int
	}{
		{name: "y vertical", plane: PlaneY, edge: EdgeVertical, wantLevel: 10, wantIndex: 0},
		{name: "y horizontal", plane: PlaneY, edge: EdgeHorizontal, wantLevel: 20, wantIndex: 1},
		{name: "u vertical", plane: PlaneU, edge: EdgeVertical, wantLevel: 30, wantIndex: 2},
		{name: "u horizontal", plane: PlaneU, edge: EdgeHorizontal, wantLevel: 30, wantIndex: 2},
		{name: "v vertical", plane: PlaneV, edge: EdgeVertical, wantLevel: 40, wantIndex: 3},
		{name: "v horizontal", plane: PlaneV, edge: EdgeHorizontal, wantLevel: 40, wantIndex: 3},
	}
	for _, tt := range tests {
		got, err := BaseLevel(params, tt.plane, tt.edge)
		if err != nil {
			t.Fatalf("%s BaseLevel err=%v", tt.name, err)
		}
		if got != tt.wantLevel {
			t.Fatalf("%s BaseLevel=%d want %d", tt.name, got, tt.wantLevel)
		}
		idx, err := DeltaIndex(tt.plane, tt.edge)
		if err != nil {
			t.Fatalf("%s DeltaIndex err=%v", tt.name, err)
		}
		if idx != tt.wantIndex {
			t.Fatalf("%s DeltaIndex=%d want %d", tt.name, idx, tt.wantIndex)
		}
	}

	if _, err := BaseLevel(params, Plane(9), EdgeVertical); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid plane BaseLevel err=%v want %v", err, ErrInvalidFilter)
	}
	if _, err := DeltaIndex(PlaneY, Edge(9)); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid edge DeltaIndex err=%v want %v", err, ErrInvalidFilter)
	}
}

func TestSelectDelta(t *testing.T) {
	multi := [DeltaCount]int8{-2, 3, 4, -5}
	got, err := SelectDelta(parser.DeltaParams{}, 7, multi, PlaneY, EdgeVertical)
	if err != nil || got != 0 {
		t.Fatalf("disabled SelectDelta=%d err=%v want 0 nil", got, err)
	}
	got, err = SelectDelta(parser.DeltaParams{DeltaLFPresent: true}, 7, multi, PlaneY, EdgeVertical)
	if err != nil || got != 7 {
		t.Fatalf("single SelectDelta=%d err=%v want 7 nil", got, err)
	}
	got, err = SelectDelta(parser.DeltaParams{DeltaLFPresent: true, DeltaLFMulti: true}, 7, multi, PlaneV, EdgeHorizontal)
	if err != nil || got != -5 {
		t.Fatalf("multi SelectDelta=%d err=%v want -5 nil", got, err)
	}
}

func TestSegmentDelta(t *testing.T) {
	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[3] = parser.SegmentData{
		DeltaLFYV: -1,
		DeltaLFYH: 2,
		DeltaLFU:  3,
		DeltaLFV:  -4,
	}

	tests := []struct {
		name  string
		plane Plane
		edge  Edge
		want  int16
	}{
		{name: "y vertical", plane: PlaneY, edge: EdgeVertical, want: -1},
		{name: "y horizontal", plane: PlaneY, edge: EdgeHorizontal, want: 2},
		{name: "u", plane: PlaneU, edge: EdgeHorizontal, want: 3},
		{name: "v", plane: PlaneV, edge: EdgeVertical, want: -4},
	}
	for _, tt := range tests {
		got, err := SegmentDelta(seg, 3, tt.plane, tt.edge)
		if err != nil {
			t.Fatalf("%s SegmentDelta err=%v", tt.name, err)
		}
		if got != tt.want {
			t.Fatalf("%s SegmentDelta=%d want %d", tt.name, got, tt.want)
		}
	}

	got, err := SegmentDelta(parser.SegmentationParams{}, 3, PlaneY, EdgeVertical)
	if err != nil || got != 0 {
		t.Fatalf("disabled SegmentDelta=%d err=%v want 0 nil", got, err)
	}
	if _, err := SegmentDelta(seg, parser.MaxSegments, PlaneY, EdgeVertical); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid segment SegmentDelta err=%v want %v", err, ErrInvalidFilter)
	}
}

func TestResolveLevel(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY:              [2]uint8{40, 20},
		LevelU:              17,
		ModeRefDeltaEnabled: true,
		Deltas: parser.LoopFilterDeltas{
			Ref:  [parser.RefFrames]int8{1, 0, 0, 0, -2},
			Mode: [parser.LoopFilterModeDeltas]int8{0, 3},
		},
	}
	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[2] = parser.SegmentData{
		DeltaLFYV: -5,
		DeltaLFYH: 2,
		DeltaLFU:  7,
		DeltaLFV:  9,
	}

	got, err := ResolveLevel(params, seg, LevelRequest{
		Plane:     PlaneY,
		Edge:      EdgeVertical,
		SegmentID: 2,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   -3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 34 {
		t.Fatalf("resolved yv=%d want 34", got)
	}

	got, err = ResolveLevel(params, seg, LevelRequest{
		Plane:     PlaneY,
		Edge:      EdgeHorizontal,
		SegmentID: 2,
		RefFrame:  0,
		Mode:      ModeDeltaClassMotion,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 23 {
		t.Fatalf("resolved intra=%d want 23", got)
	}

	params.LevelV = 0
	got, err = ResolveLevel(params, seg, LevelRequest{
		Plane:     PlaneV,
		Edge:      EdgeVertical,
		SegmentID: 2,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   20,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("zero chroma level=%d want 0", got)
	}

	params.LevelY = [2]uint8{}
	params.LevelU = 63
	got, err = ResolveLevel(params, seg, LevelRequest{
		Plane:     PlaneU,
		Edge:      EdgeHorizontal,
		SegmentID: 2,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   20,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 0 {
		t.Fatalf("all-zero luma gate level=%d want 0", got)
	}
}

func TestResolveLevelRejectsInvalidInputs(t *testing.T) {
	params := parser.LoopFilterParams{LevelY: [2]uint8{1, 1}}
	tests := []struct {
		name string
		req  LevelRequest
	}{
		{name: "plane", req: LevelRequest{Plane: Plane(9), Edge: EdgeVertical, RefFrame: 0, SegmentID: 0}},
		{name: "edge", req: LevelRequest{Plane: PlaneY, Edge: Edge(9), RefFrame: 0, SegmentID: 0}},
		{name: "segment", req: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, RefFrame: 0, SegmentID: parser.MaxSegments}},
		{name: "ref", req: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, RefFrame: parser.RefFrames, SegmentID: 0}},
		{name: "mode", req: LevelRequest{Plane: PlaneY, Edge: EdgeVertical, RefFrame: 0, SegmentID: 0, Mode: ModeDeltaClass(9)}},
	}
	for _, tt := range tests {
		_, err := ResolveLevel(params, parser.SegmentationParams{}, tt.req)
		if !errors.Is(err, ErrInvalidFilter) {
			t.Fatalf("%s ResolveLevel err=%v want %v", tt.name, err, ErrInvalidFilter)
		}
	}
}

func TestThresholdsForLevel(t *testing.T) {
	tests := []struct {
		level     uint8
		sharpness uint8
		want      Thresholds
	}{
		{level: 0, sharpness: 0, want: Thresholds{Limit: 1, BlockLimit: 5, HighEdgeVariance: 0}},
		{level: 16, sharpness: 0, want: Thresholds{Limit: 16, BlockLimit: 52, HighEdgeVariance: 1}},
		{level: 63, sharpness: 7, want: Thresholds{Limit: 2, BlockLimit: 132, HighEdgeVariance: 3}},
	}
	for _, tt := range tests {
		got, err := ThresholdsForLevel(tt.level, tt.sharpness)
		if err != nil {
			t.Fatalf("ThresholdsForLevel(%d,%d) err=%v", tt.level, tt.sharpness, err)
		}
		if got != tt.want {
			t.Fatalf("ThresholdsForLevel(%d,%d)=%+v want %+v", tt.level, tt.sharpness, got, tt.want)
		}
	}
	if _, err := ThresholdsForLevel(MaxLevel+1, 0); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid level ThresholdsForLevel err=%v want %v", err, ErrInvalidFilter)
	}
	if _, err := ThresholdsForLevel(0, MaxSharpness+1); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid sharpness ThresholdsForLevel err=%v want %v", err, ErrInvalidFilter)
	}
}

func TestLevelHelpersAllocs(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY:              [2]uint8{32, 31},
		LevelU:              12,
		LevelV:              13,
		ModeRefDeltaEnabled: true,
		Deltas: parser.LoopFilterDeltas{
			Ref:  [parser.RefFrames]int8{1, 0, 0, 0, -1, 0, -1, -1},
			Mode: [parser.LoopFilterModeDeltas]int8{0, 1},
		},
	}
	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[1].DeltaLFYV = -2
	req := LevelRequest{
		Plane:     PlaneY,
		Edge:      EdgeVertical,
		SegmentID: 1,
		RefFrame:  4,
		Mode:      ModeDeltaClassMotion,
		DeltaLF:   3,
	}

	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ResolveLevel(params, seg, req); err != nil {
			t.Fatal(err)
		}
		if _, err := ThresholdsForLevel(32, params.Sharpness); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("loopfilter helpers allocated: %f", allocs)
	}
}

func BenchmarkThresholdsForLevel(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = ThresholdsForLevel(uint8(i&MaxLevel), uint8(i&MaxSharpness))
	}
}

func TestPreparedLevelResolverMatchesScalarResolver(t *testing.T) {
	baseCases := []struct {
		params         parser.LoopFilterParams
		segmentationOn bool
	}{
		{params: parser.LoopFilterParams{LevelY: [2]uint8{31, 32}, LevelU: 31, LevelV: 32}},
		{segmentationOn: true, params: parser.LoopFilterParams{LevelY: [2]uint8{255, 1}, LevelU: 200, LevelV: 64}},
		{params: parser.LoopFilterParams{LevelY: [2]uint8{1, 63}, LevelU: 0, LevelV: 63, ModeRefDeltaEnabled: true,
			Deltas: parser.LoopFilterDeltas{Ref: [parser.RefFrames]int8{-128, 127, -127, 126, -1, 1, 0, 0}, Mode: [2]int8{127, -128}}}},
		{segmentationOn: true, params: parser.LoopFilterParams{LevelY: [2]uint8{255, 1}, LevelU: 200, LevelV: 64, ModeRefDeltaEnabled: true,
			Deltas: parser.LoopFilterDeltas{Ref: [parser.RefFrames]int8{127, -128, 127, -128, 63, -64, 1, -1}, Mode: [2]int8{-128, 127}}}},
	}
	segmentValues := []int16{-32768, -275, -1, 0, 1, 275, 32767}
	deltaValues := []int8{-128, -63, -1, 0, 1, 63, 127}
	for caseIndex, tc := range baseCases {
		params := tc.params
		for _, multiDelta := range []bool{false, true} {
			for _, deltaPresent := range []bool{false, true} {
				seg := parser.SegmentationParams{Enabled: tc.segmentationOn}
				for segment := range parser.MaxSegments {
					s := segmentValues[(segment+caseIndex)%len(segmentValues)]
					seg.Data.Segments[segment] = parser.SegmentData{
						DeltaLFYV: s,
						DeltaLFYH: segmentValues[(segment+1+caseIndex)%len(segmentValues)],
						DeltaLFU:  segmentValues[(segment+2+caseIndex)%len(segmentValues)],
						DeltaLFV:  segmentValues[(segment+3+caseIndex)%len(segmentValues)],
					}
				}
				delta := parser.DeltaParams{DeltaLFPresent: deltaPresent, DeltaLFMulti: multiDelta}
				prepared := PrepareLevelResolver(params, seg, delta, false)
				for segmentID := uint8(0); segmentID < parser.MaxSegments; segmentID++ {
					for ref := uint8(0); ref < parser.RefFrames; ref++ {
						for _, mode := range []ModeDeltaClass{ModeDeltaClassZero, ModeDeltaClassMotion} {
							for deltaIndex, deltaFromBase := range deltaValues {
								multi := [DeltaCount]int8{
									deltaValues[(deltaIndex+0)%len(deltaValues)],
									deltaValues[(deltaIndex+1)%len(deltaValues)],
									deltaValues[(deltaIndex+2)%len(deltaValues)],
									deltaValues[(deltaIndex+3)%len(deltaValues)],
								}
								got, err := prepared.ResolveBlockLevels(segmentID, ref, mode, deltaFromBase, multi)
								if err != nil {
									t.Fatalf("prepared case=%d seg=%d ref=%d mode=%d delta=%d multi=%v: %v", caseIndex, segmentID, ref, mode, deltaFromBase, multi, err)
								}
								for plane := PlaneY; plane <= PlaneV; plane++ {
									for edge := EdgeVertical; edge <= EdgeHorizontal; edge++ {
										blockDelta := int8(0)
										if deltaPresent {
											if multiDelta {
												index, _ := DeltaIndex(plane, edge)
												blockDelta = multi[index]
											} else {
												blockDelta = deltaFromBase
											}
										}
										want, err := ResolveLevel(params, seg, LevelRequest{
											Plane: plane, Edge: edge, SegmentID: segmentID,
											RefFrame: ref, Mode: mode, DeltaLF: blockDelta,
										})
										if err != nil {
											t.Fatalf("scalar case=%d seg=%d ref=%d mode=%d plane=%d edge=%d: %v", caseIndex, segmentID, ref, mode, plane, edge, err)
										}
										if got[plane][edge] != want {
											t.Fatalf("case=%d seg=%d ref=%d mode=%d multi=%v delta=%d plane=%d edge=%d: prepared=%d scalar=%d", caseIndex, segmentID, ref, mode, multiDelta, deltaFromBase, plane, edge, got[plane][edge], want)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestPreparedLevelResolverModeRefThresholdAndFlagCombinations(t *testing.T) {
	params := parser.LoopFilterParams{
		LevelY:              [2]uint8{31, 32},
		ModeRefDeltaEnabled: true,
		Deltas:              parser.LoopFilterDeltas{Ref: [parser.RefFrames]int8{0, 1}},
	}
	prepared := PrepareLevelResolver(params, parser.SegmentationParams{}, parser.DeltaParams{}, false)
	got, err := prepared.ResolveBlockLevels(0, 1, ModeDeltaClassZero, 0, [DeltaCount]int8{})
	if err != nil {
		t.Fatalf("mode/ref-only resolve: %v", err)
	}
	if got[PlaneY][EdgeVertical] != 32 || got[PlaneY][EdgeHorizontal] != 34 {
		t.Fatalf("mode/ref threshold levels=%v want Y=(32,34)", got[PlaneY])
	}

	seg := parser.SegmentationParams{Enabled: true}
	seg.Data.Segments[1].DeltaLFYV = 5
	segOnly := PrepareLevelResolver(parser.LoopFilterParams{LevelY: [2]uint8{20, 21}}, seg, parser.DeltaParams{}, false)
	segGot, err := segOnly.ResolveBlockLevels(1, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{})
	if err != nil || segGot[PlaneY][EdgeVertical] != 25 || segGot[PlaneY][EdgeHorizontal] != 21 {
		t.Fatalf("segmentation-only levels=%v err=%v want Y=(25,21)", segGot[PlaneY], err)
	}
}

func TestPreparedLevelResolverMonochromeAndRawBaseDomain(t *testing.T) {
	for raw := 0; raw <= 255; raw++ {
		params := parser.LoopFilterParams{
			LevelY: [2]uint8{uint8(raw), uint8(255 - raw)},
			LevelU: uint8(raw), LevelV: uint8(255 - raw),
		}
		prepared := PrepareLevelResolver(params, parser.SegmentationParams{}, parser.DeltaParams{}, true)
		got, err := prepared.ResolveBlockLevels(0, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{})
		if err != nil {
			t.Fatalf("raw base=%d resolve: %v", raw, err)
		}
		for plane := PlaneY; plane <= PlaneV; plane++ {
			for edge := EdgeVertical; edge <= EdgeHorizontal; edge++ {
				want, err := ResolveLevel(params, parser.SegmentationParams{}, LevelRequest{
					Plane: plane, Edge: edge, RefFrame: 0, Mode: ModeDeltaClassZero,
				})
				if err != nil {
					t.Fatalf("raw base=%d plane=%d edge=%d scalar: %v", raw, plane, edge, err)
				}
				if got[plane][edge] != want {
					t.Fatalf("raw base=%d plane=%d edge=%d prepared=%d scalar=%d", raw, plane, edge, got[plane][edge], want)
				}
			}
		}
	}

	base := parser.LoopFilterParams{LevelY: [2]uint8{20, 24}, LevelU: 31, LevelV: 47}
	noFlags := PrepareLevelResolver(base, parser.SegmentationParams{}, parser.DeltaParams{}, true)
	got, err := noFlags.ResolveBlockLevels(0, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{})
	if err != nil || got[PlaneU] != [2]uint8{31, 31} || got[PlaneV] != [2]uint8{47, 47} {
		t.Fatalf("monochrome no-flags legacy chroma fast return levels=%v err=%v", got, err)
	}

	cases := []struct {
		name  string
		seg   parser.SegmentationParams
		delta parser.DeltaParams
		mode  parser.LoopFilterParams
	}{
		{name: "delta", delta: parser.DeltaParams{DeltaLFPresent: true}},
		{name: "segmentation", seg: parser.SegmentationParams{Enabled: true}},
		{name: "mode-ref", mode: parser.LoopFilterParams{ModeRefDeltaEnabled: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := base
			params.ModeRefDeltaEnabled = tc.mode.ModeRefDeltaEnabled
			params.Deltas = tc.mode.Deltas
			seg := tc.seg
			if seg.Enabled {
				seg.Data.Segments[0].DeltaLFU = 7
				seg.Data.Segments[0].DeltaLFV = -9
			}
			prepared := PrepareLevelResolver(params, seg, tc.delta, true)
			levels, err := prepared.ResolveBlockLevels(0, 0, ModeDeltaClassZero, 1, [DeltaCount]int8{1, 2, 3, 4})
			if err != nil || levels[PlaneU] != [2]uint8{} || levels[PlaneV] != [2]uint8{} {
				t.Fatalf("monochrome %s gated chroma levels=%v err=%v", tc.name, levels, err)
			}
		})
	}
}

func TestPreparedLevelResolverLumaZeroFastReturnKeepsLegacyChromaBehavior(t *testing.T) {
	params := parser.LoopFilterParams{LevelU: 40, LevelV: 50}
	for _, mono := range []bool{false, true} {
		prepared := PrepareLevelResolver(params, parser.SegmentationParams{}, parser.DeltaParams{}, mono)
		got, err := prepared.ResolveBlockLevels(parser.MaxSegments, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{})
		if err != nil || got != ([3][2]uint8{}) {
			t.Fatalf("mono=%v levels=%v err=%v want luma-zero fast return", mono, got, err)
		}
	}
}

func TestPreparedLevelResolverFastReturnValidationOrder(t *testing.T) {
	params := parser.LoopFilterParams{}
	seg := parser.SegmentationParams{Enabled: true}
	prepared := PrepareLevelResolver(params, seg, parser.DeltaParams{}, true)
	if _, err := prepared.ResolveBlockLevels(0, parser.RefFrames, ModeDeltaClassZero, 0, [DeltaCount]int8{}); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid reference error=%v want %v", err, ErrInvalidFilter)
	}
	if _, err := prepared.ResolveBlockLevels(0, 0, ModeDeltaClass(2), 0, [DeltaCount]int8{}); !errors.Is(err, ErrInvalidFilter) {
		t.Fatalf("invalid mode error=%v want %v", err, ErrInvalidFilter)
	}
	if got, err := prepared.ResolveBlockLevels(parser.MaxSegments, 0, ModeDeltaClassZero, 0, [DeltaCount]int8{}); err != nil || got != ([3][2]uint8{}) {
		t.Fatalf("luma-zero invalid-segment fast return levels=%v err=%v", got, err)
	}
}

const (
	libaomLPFDeterministicSeed = 0xbaba
	libaomLPFNumCoeffs         = 1024
	libaomLPFStride            = 32
	libaomLPFIterations        = 64
)

func TestLoopFilterEdgesMatchLibaomOperationCorpus(t *testing.T) {
	for _, width := range []int{4, 6, 8, 14} {
		for _, bitDepth := range []uint8{8, 10, 12} {
			bytesPerSample := 1
			if bitDepth != 8 {
				bytesPerSample = 2
			}
			mask := (1 << int(bitDepth)) - 1
			for _, edge := range []Edge{EdgeHorizontal, EdgeVertical} {
				rnd := newLibaomTestRandom(libaomLPFDeterministicSeed)
				for i := range libaomLPFIterations {
					blimit := uint8(rnd.pseudoUniform(3*MaxLevel + 5))
					limit := uint8(rnd.pseudoUniform(MaxLevel + 1))
					hev := uint8(rnd.pseudoUniform(MaxLevel+1) >> 4)
					thresholds := Thresholds{Limit: limit, BlockLimit: blimit, HighEdgeVariance: hev}

					got := libaomInitLoopFilterInput(bytesPerSample, rnd, limit, mask, i)
					want := cloneTestPlane(got)
					applyCReferenceFilterByWidth(want, bytesPerSample, bitDepth, edge, width, 8, 8, 4, thresholds)

					if err := applyLoopFilterByWidth(width, got, bytesPerSample, bitDepth, edge, 8, 8, 4, thresholds); err != nil {
						t.Fatalf("width=%d bitdepth=%d edge=%d iteration=%d: %v", width, bitDepth, edge, i, err)
					}
					if !bytes.Equal(got.Pix, want.Pix) {
						t.Fatalf("width=%d bitdepth=%d edge=%d iteration=%d got=%v want=%v", width, bitDepth, edge, i, got.Pix, want.Pix)
					}
				}
			}
		}
	}
}

type libaomTestRandom struct {
	state uint32
}

func newLibaomTestRandom(seed uint32) *libaomTestRandom {
	return &libaomTestRandom{state: seed}
}

func (r *libaomTestRandom) generate(randomRange uint32) uint32 {
	r.state = (1103515245*r.state + 12345) & ((1 << 31) - 1)
	return r.state % randomRange
}

func (r *libaomTestRandom) rand16() uint16 {
	return uint16((r.generate(1<<31) >> 15) & 0xffff)
}

func (r *libaomTestRandom) rand8() uint8 {
	return uint8((r.generate(1<<31) >> 23) & 0xff)
}

func (r *libaomTestRandom) pseudoUniform(randomRange int) int {
	return int(r.generate(uint32(randomRange)))
}

func libaomInitLoopFilterInput(bytesPerSample int, rnd *libaomTestRandom, limit uint8, mask int, iteration int) frame.Plane {
	tmp := make([]uint16, libaomLPFNumCoeffs)
	for j := 0; j < libaomLPFNumCoeffs; {
		val := rnd.rand8()
		if val&0x80 != 0 {
			tmp[j] = rnd.rand16()
			j++
			continue
		}
		for k := 0; k < int(val&0x1f)+1 && j < libaomLPFNumCoeffs; k++ {
			if j < 1 {
				tmp[j] = rnd.rand16()
			} else if val&0x20 != 0 {
				tmp[j] = uint16(int(tmp[j-1]) + int(limit) - 1)
			} else {
				tmp[j] = uint16(int(tmp[j-1]) - (int(limit) - 1))
			}
			j++
		}
	}

	for j := 0; j < libaomLPFNumCoeffs; {
		val := rnd.rand8()
		if val&0x80 != 0 {
			j++
			continue
		}
		for k := 0; k < int(val&0x1f)+1 && j < libaomLPFNumCoeffs; k++ {
			if j < 1 {
				tmp[j] = rnd.rand16()
			} else {
				dst := (j%libaomLPFStride)*libaomLPFStride + j/libaomLPFStride
				prev := ((j - 1) % libaomLPFStride * libaomLPFStride) + (j-1)/libaomLPFStride
				if val&0x20 != 0 {
					tmp[dst] = uint16(int(tmp[prev]) + int(limit) - 1)
				} else {
					tmp[dst] = uint16(int(tmp[prev]) - (int(limit) - 1))
				}
			}
			j++
		}
	}

	plane := testPlane(libaomLPFStride, libaomLPFStride, bytesPerSample, libaomLPFStride*bytesPerSample)
	for j := range libaomLPFNumCoeffs {
		src := j
		if iteration%2 == 0 {
			src = libaomLPFStride*(j%libaomLPFStride) + j/libaomLPFStride
		}
		value := tmp[src] & uint16(mask)
		setSample(plane, bytesPerSample, j%libaomLPFStride, j/libaomLPFStride, value)
	}
	return plane
}

func applyCReferenceFilterByWidth(plane frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, width int, x int, y int, length int, thresholds Thresholds) {
	for i := range length {
		q0, step := filter4SampleOffset(plane, bytesPerSample, edge, x, y, i)
		sample := make([]int, width)
		for j := range width {
			sample[j] = readSample(plane.Pix, bytesPerSample, q0+(j-width/2)*step)
		}
		ref := cReferenceFilterByWidth(width, sample, bitDepth, thresholds)
		for j, value := range ref {
			writeSample(plane.Pix, bytesPerSample, q0+(j-width/2)*step, value)
		}
	}
}

func applyLoopFilterByWidth(width int, plane frame.Plane, bytesPerSample int, bitDepth uint8, edge Edge, x int, y int, length int, thresholds Thresholds) error {
	switch width {
	case 4:
		return Filter4Edge(plane, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
	case 6:
		return Filter6Edge(plane, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
	case 8:
		return Filter8Edge(plane, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
	case 14:
		return Filter14Edge(plane, bytesPerSample, bitDepth, edge, int32(x), int32(y), int32(length), thresholds)
	default:
		return ErrInvalidFilter
	}
}

func cReferenceFilterByWidth(width int, sample []int, bitDepth uint8, thresholds Thresholds) []int {
	out := make([]int, width)
	switch width {
	case 4:
		var in [4]int
		copy(in[:], sample)
		ref := cReferenceFilter4Samples(in, bitDepth, thresholds)
		copy(out, ref[:])
	case 6:
		var in [6]int
		copy(in[:], sample)
		ref := cReferenceFilter6(in, bitDepth, thresholds)
		copy(out, ref[:])
	case 8:
		var in [8]int
		copy(in[:], sample)
		ref := cReferenceFilter8(in, bitDepth, thresholds)
		copy(out, ref[:])
	case 14:
		var in [14]int
		copy(in[:], sample)
		ref := cReferenceFilter14(in, bitDepth, thresholds)
		copy(out, ref[:])
	}
	return out
}

func cReferenceFilter4Samples(in [4]int, bitDepth uint8, thresholds Thresholds) [4]int {
	scale := 1 << int(bitDepth-8)
	limit := int(thresholds.Limit) * scale
	blimit := int(thresholds.BlockLimit) * scale
	hev := int(thresholds.HighEdgeVariance) * scale
	min := -128 * scale
	max := 128*scale - 1
	center := 128 * scale

	p1, p0 := in[0], in[1]
	q0, q1 := in[2], in[3]
	if !cReferenceFilterMask2(limit, blimit, p1, p0, q0, q1) {
		return in
	}
	in[0], in[1], in[2], in[3] = cReferenceFilter4(p1, p0, q0, q1, hev, min, max, center)
	return in
}

func cReferenceFilterMask2(limit int, blimit int, p1 int, p0 int, q0 int, q1 int) bool {
	return cAbs(p1-p0) <= limit &&
		cAbs(q1-q0) <= limit &&
		cAbs(p0-q0)*2+cAbs(p1-q1)/2 <= blimit
}

// checkasmSB16Case is checkasmSBCase for 10/12-bit planes: dav1d's
// tests/checkasm/loopfilter.c check_lpf_sb at 16bpc, with init_lpf_border's
// limits and flatness scaled by bitdepth_min_8 and samples stored as
// little-endian uint16 (off, stride and rowBytes in bytes).
type checkasmSB16Case struct {
	checkasmSBCase
	bitDepth uint8
}

func newCheckasmSB16Case(seed int64, vertical, luma bool, strength int) checkasmSB16Case {
	rng := rand.New(rand.NewSource(seed))
	nBlks := 32
	if !luma {
		nBlks = 16
	}
	sharp := rng.Intn(8)
	c := checkasmSB16Case{checkasmSBCase: checkasmSBCase{lut: NewSBLUT(uint8(sharp))}, bitDepth: 10}
	if rng.Intn(2) == 1 {
		c.bitDepth = 12
	}
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
		c.stride = 2 * 32
		c.pix = make([]byte, c.stride*(4*nBlks))
		c.off = 2 * 8
		c.rowBytes = c.stride - c.off
		c.lvlOff, c.lvlStride = 1, 2
	} else {
		c.stride = 2 * 4 * nBlks
		c.pix = make([]byte, c.stride*16)
		c.off = 8 * c.stride
		c.rowBytes = c.stride
		c.lvlOff, c.lvlStride = 32, 32
	}
	bd8 := int(c.bitDepth) - 8
	maxV := 1<<c.bitDepth - 1
	clip := func(v int) int { return min(max(v, 0), maxV) }
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
		E, I := int(c.lut.E[L])<<bd8, int(c.lut.I[L])<<bd8
		F := 1 << bd8
		var base, step int
		if vertical {
			base, step = i*c.stride+c.off, 2
		} else {
			base, step = c.off+2*i, c.stride
		}
		set := func(k, v int) { binary.LittleEndian.PutUint16(c.pix[base+k*step:], uint16(v)) }
		get := func(k int) int { return int(binary.LittleEndian.Uint16(c.pix[base+k*step:])) }
		edgeDiff := rng.Intn((E+2)*4) - 2*(E+2)
		switch rng.Intn(4) {
		case 0:
			for k := -8; k < 8; k++ {
				set(k, rng.Intn(maxV+1))
			}
		case 1:
			set(-8, rng.Intn(maxV+1))
			set(7, rng.Intn(maxV+1))
			set(0, rng.Intn(maxV+1))
			set(-1, clip(get(0)+edgeDiff))
			for k := 1; k < 7; k++ {
				set(-(1 + k), clip(get(-1)+rng.Intn(2*(F+1))-(F+1)))
				set(k, clip(get(0)+rng.Intn(2*(F+1))-(F+1)))
			}
		case 2:
			for k := 4; k < 8; k++ {
				set(-(1 + k), rng.Intn(maxV+1))
				set(k, rng.Intn(maxV+1))
			}
			set(0, rng.Intn(maxV+1))
			set(-1, clip(get(0)+edgeDiff))
			for k := 1; k < 4; k++ {
				set(-(1 + k), clip(get(-1)+rng.Intn(2*(F+1))-(F+1)))
				set(k, clip(get(0)+rng.Intn(2*(F+1))-(F+1)))
			}
		case 3:
			for k := 4; k < 8; k++ {
				set(-(1 + k), rng.Intn(maxV+1))
				set(k, rng.Intn(maxV+1))
			}
			set(0, rng.Intn(maxV+1))
			set(-1, clip(get(0)+edgeDiff))
			for k := 1; k < 4; k++ {
				set(-(1 + k), clip(get(-k)+rng.Intn(2*(I+1))-(I+1)))
				set(k, clip(get(k-1)+rng.Intn(2*(I+1))-(I+1)))
			}
		}
	}
	return c
}

// TestFilterSB16CheckasmMatchesScalar is TestFilterSB8CheckasmMatchesScalar
// for the 10/12-bit superblock filters.
func TestFilterSB16CheckasmMatchesScalar(t *testing.T) {
	for seed := int64(0); seed < 400; seed++ {
		for _, vertical := range []bool{true, false} {
			for _, luma := range []bool{true, false} {
				n := 3
				if !luma {
					n = 2
				}
				for s := 0; s < n; s++ {
					c := newCheckasmSB16Case(seed*7+int64(s), vertical, luma, s)
					comp := int(seed & 3)
					lvl := make([]uint8, len(c.lvl))
					for i := range len(c.lvl) / 4 {
						lvl[4*(i)+comp] = c.lvl[4*(i)+0]
						lvl[4*(i)+(comp+1)&3] = uint8(i)
					}
					vm := c.vm
					if seed&1 == 1 && !vertical {
						lvl = lvl[:4*(c.lvlOff+31)]
						for k := range vm {
							vm[k] &= 0x7fffffff
						}
					}
					want := append([]byte(nil), c.pix...)
					got := append([]byte(nil), c.pix...)
					if vertical {
						filterSBVertical16Scalar(want, c.off, c.stride, c.bitDepth, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
						FilterSBVertical16(got, c.off, c.stride, c.rowBytes, c.bitDepth, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
					} else {
						filterSBHorizontal16Scalar(want, c.off, c.stride, c.bitDepth, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
						FilterSBHorizontal16(got, c.off, c.stride, c.rowBytes, c.bitDepth, vm, luma, lvl, c.lvlOff, c.lvlStride, comp, &c.lut)
					}
					if !bytes.Equal(got, want) {
						t.Fatalf("seed %d vertical=%v luma=%v s=%d bd=%d: pixels differ", seed, vertical, luma, s, c.bitDepth)
					}
				}
			}
		}
	}
}

func benchmarkCheckasmSB16(b *testing.B, vertical, luma bool) {
	n := 3
	widths := []int{4, 8, 16}
	if !luma {
		n = 2
		widths = []int{4, 6}
	}
	for s := 0; s < n; s++ {
		b.Run(fmt.Sprintf("w%d", widths[s]), func(b *testing.B) {
			const cases = 16
			var cs [cases]checkasmSB16Case
			var work [cases][]byte
			for i := range cs {
				cs[i] = newCheckasmSB16Case(int64(1000*s+i), vertical, luma, s)
				work[i] = append([]byte(nil), cs[i].pix...)
			}
			i := 0
			for b.Loop() {
				c := &cs[i&(cases-1)]
				w := work[i&(cases-1)]
				if vertical {
					FilterSBVertical16(w, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				} else {
					FilterSBHorizontal16(w, c.off, c.stride, c.rowBytes, c.bitDepth, c.vm, luma, c.lvl, c.lvlOff, c.lvlStride, 0, &c.lut)
				}
				i++
			}
		})
	}
}

func BenchmarkCheckasmLPFHSBY16(b *testing.B)  { benchmarkCheckasmSB16(b, true, true) }
func BenchmarkCheckasmLPFVSBY16(b *testing.B)  { benchmarkCheckasmSB16(b, false, true) }
func BenchmarkCheckasmLPFHSBUV16(b *testing.B) { benchmarkCheckasmSB16(b, true, false) }
func BenchmarkCheckasmLPFVSBUV16(b *testing.B) { benchmarkCheckasmSB16(b, false, false) }

// sbTestPlane16 builds a little-endian uint16 plane at the given bit depth
// from 8x8 tiles that are either flat (one background shared by every flat
// tile, noise below the scaled flatness threshold), lightly textured, or
// random, so every filter width including the 13-tap flat path fires across
// both edge directions.
func sbTestPlane16(rng *rand.Rand, w, h int, bitDepth uint8) []byte {
	shift := uint(bitDepth - 8)
	maxPix := 1<<bitDepth - 1
	flatF := 1 << shift
	bg := rng.Intn(maxPix + 1 - flatF)
	tw, th := (w+7)/8, (h+7)/8
	modes := make([]int, tw*th)
	for i := range modes {
		modes[i] = rng.Intn(5)
	}
	pix := make([]byte, 2*w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var s int
			switch modes[(y/8)*tw+x/8] {
			case 0, 1, 2:
				s = bg + rng.Intn(flatF)
			case 3:
				s = bg + rng.Intn(12*flatF) - 6*flatF
			default:
				s = rng.Intn(maxPix + 1)
			}
			if rng.Intn(256) == 0 {
				s = maxPix
			}
			s = min(max(s, 0), maxPix)
			pix[2*(y*w+x)] = byte(s)
			pix[2*(y*w+x)+1] = byte(s >> 8)
		}
	}
	return pix
}

// TestFilterSB16MatchesScalar checks the dispatched high-bit-depth superblock
// filters against the unit-by-unit dav1d C reference.
func TestFilterSB16MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const w, h = 160, 160
	const stride = 2 * w
	for iter := 0; iter < 4000; iter++ {
		bitDepth := uint8(10)
		if rng.Intn(2) == 0 {
			bitDepth = 12
		}
		luma := rng.Intn(2) == 0
		lut := NewSBLUT(uint8(rng.Intn(MaxSharpness + 1)))
		pix := sbTestPlane16(rng, w, h, bitDepth)
		lvlStride := w / 4
		lvl := sbTestLevels(rng, lvlStride*(h/4))
		comp := rng.Intn(4)
		units := 1 + rng.Intn(32)
		vm := sbTestMasks(rng, luma, units)
		vertical := rng.Intn(2) == 0
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		if vertical {
			x := 4 * (1 + rng.Intn(w/4-1))
			if vm[2] != 0 {
				x = 8 * (1 + rng.Intn(w/8-2))
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off := y0*stride + 2*x
			lvlOff := (y0/4)*lvlStride + x/4
			filterSBVertical16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			FilterSBVertical16(got, off, stride, 2*(w-x), bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			y := 4 * (1 + rng.Intn(h/4-1))
			if vm[2] != 0 {
				y = 8 * (1 + rng.Intn(h/8-2))
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off := y*stride + 2*x0
			lvlOff := (y/4)*lvlStride + x0/4
			filterSBHorizontal16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			FilterSBHorizontal16(got, off, stride, 2*(w-x0), bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		if !bytes.Equal(got, want) {
			for i := 0; i < len(got); i += 2 {
				if got[i] != want[i] || got[i+1] != want[i+1] {
					px := i / 2
					t.Fatalf("iter %d vertical=%v luma=%v bd=%d vm=%x: first diff at (%d,%d): got %d want %d", iter, vertical, luma, bitDepth, vm, px%w, px/w, ld16(got, i), ld16(want, i))
				}
			}
		}
	}
}

// TestFilterSB16ScalarMatchesEdgeKernels checks the high-bit-depth C
// reference against the package's per-edge kernels for every width.
func TestFilterSB16ScalarMatchesEdgeKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	const w, h = 64, 64
	const stride = 2 * w
	for iter := 0; iter < 3000; iter++ {
		bitDepth := uint8(10)
		if rng.Intn(2) == 0 {
			bitDepth = 12
		}
		sharp := uint8(rng.Intn(MaxSharpness + 1))
		lut := NewSBLUT(sharp)
		level := uint8(1 + rng.Intn(MaxLevel))
		th, err := ThresholdsForLevel(level, sharp)
		if err != nil {
			t.Fatal(err)
		}
		widths := []int{4, 6, 8, 16}
		wd := widths[rng.Intn(len(widths))]
		pix := sbTestPlane16(rng, w, h, bitDepth)
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		edge := EdgeVertical
		x, y := 16+4*rng.Intn(8), 16+4*rng.Intn(8)
		if rng.Intn(2) == 0 {
			edge = EdgeHorizontal
		}
		plane := frame.Plane{Pix: want, Stride: stride, Width: w, Height: h}
		kw := uint8(wd)
		if wd == 16 {
			kw = 14
		}
		if err := FilterEdgeByWidth(kw, plane, 2, bitDepth, edge, int32(x), int32(y), 4, th); err != nil {
			t.Fatal(err)
		}
		off := y*stride + 2*x
		if edge == EdgeVertical {
			lpfUnit16(got, off, stride, 2, bitDepth, int(lut.E[level]), int(lut.I[level]), int(level>>4), wd)
		} else {
			lpfUnit16(got, off, 2, stride, bitDepth, int(lut.E[level]), int(lut.I[level]), int(level>>4), wd)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("iter %d: width %d edge %v level %d sharp %d bd %d mismatch", iter, wd, edge, level, sharp, bitDepth)
		}
	}
}

// TestFilterSB16TightBuffers is TestFilterSB8TightBuffers for 10/12-bit
// planes: planes and level maps end right after the last sample and level
// the selected units use, so calls reaching a buffer end take the edge path.
func TestFilterSB16TightBuffers(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	const w, h = 160, 160
	const stride = 2 * w
	for iter := 0; iter < 4000; iter++ {
		bitDepth := uint8(10)
		if rng.Intn(2) == 0 {
			bitDepth = 12
		}
		luma := rng.Intn(2) == 0
		lut := NewSBLUT(uint8(rng.Intn(MaxSharpness + 1)))
		full := sbTestPlane16(rng, w, h, bitDepth)
		lvlStride := w / 4
		lvlFull := sbTestLevels(rng, lvlStride*(h/4))
		comp := rng.Intn(4)
		units := 1 + rng.Intn(32)
		vm := sbTestMasks(rng, luma, units)
		all := vm[0] | vm[1] | vm[2]
		if all == 0 {
			continue
		}
		hi := 31 - bits.LeadingZeros32(all)
		vertical := rng.Intn(2) == 0
		var off, rowBytes, lvlOff, pixLen, lvlLen int
		if vertical {
			x := 4 * (1 + rng.Intn(w/4-1))
			if vm[2] != 0 {
				x = 8 + 4*rng.Intn(w/4-3)
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off, rowBytes, lvlOff = y0*stride+2*x, 2*(w-x), (y0/4)*lvlStride+x/4
			pixLen = off + (4*hi+3)*stride + 14
			lvlLen = lvlOff + hi*lvlStride + 1
		} else {
			y := 4 * (1 + rng.Intn(h/4-1))
			if vm[2] != 0 {
				y = 28 + 4*rng.Intn(h/4-8)
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off, rowBytes, lvlOff = y*stride+2*x0, 2*(w-x0), (y/4)*lvlStride+x0/4
			pixLen = off + 6*stride + 8*(hi+1)
			lvlLen = lvlOff + hi + 1
		}
		pixLen = min(pixLen, len(full))
		if rng.Intn(2) == 0 {
			lvlLen = len(lvlFull) / 4
		}
		lvl := append([]uint8(nil), lvlFull[:4*lvlLen]...)
		want := append([]byte(nil), full[:pixLen]...)
		got := append([]byte(nil), full[:pixLen]...)
		if vertical {
			filterSBVertical16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			FilterSBVertical16(got, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			filterSBHorizontal16Scalar(want, off, stride, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			FilterSBHorizontal16(got, off, stride, rowBytes, bitDepth, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("iter %d vertical=%v luma=%v bd=%d vm=%x: samples differ from the reference", iter, vertical, luma, bitDepth, vm)
		}
	}
}

// sbExtremeRun filters one superblock call of 32 units (128 lanes) whose
// taps come from tap(lane, k) (k = -8..7, k < 0 above/left of the edge) with
// every unit at the maximum level and the given sharpness, and checks the
// dispatched filter against the unit-by-unit reference in both edge
// directions.
func sbExtremeRun(t *testing.T, name string, sharp uint8, luma bool, vm [3]uint32, tap func(lane, k int) uint8) {
	t.Helper()
	lut := NewSBLUT(sharp)
	for _, vertical := range []bool{false, true} {
		var pix []byte
		var off, stride, rowBytes, lvlOff, lvlStride int
		if vertical {
			// 128 rows of 32 bytes, edge at x = 8 (16-byte rows fit).
			stride, off, rowBytes = 32, 8, 24
			pix = make([]byte, stride*128)
			lvlStride, lvlOff = 8, 2
			for lane := 0; lane < 128; lane++ {
				for k := -8; k < 8; k++ {
					pix[lane*stride+off+k] = tap(lane, k)
				}
			}
		} else {
			// 16 rows of 128 lanes, edge at y = 8.
			stride, off, rowBytes = 128, 8*128, 128
			pix = make([]byte, stride*16)
			lvlStride, lvlOff = 32, 2*32
			for lane := 0; lane < 128; lane++ {
				for k := -8; k < 8; k++ {
					pix[off+k*stride+lane] = tap(lane, k)
				}
			}
		}
		lvl := make([]uint8, 4*(256))
		for i := range len(lvl) / 4 {
			lvl[4*(i)+0] = MaxLevel
		}
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		if vertical {
			filterSBVertical8Scalar(want, off, stride, vm, luma, lvl, lvlOff, lvlStride, 0, &lut)
			FilterSBVertical8(got, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, 0, &lut)
		} else {
			filterSBHorizontal8Scalar(want, off, stride, vm, luma, lvl, lvlOff, lvlStride, 0, &lut)
			FilterSBHorizontal8(got, off, stride, rowBytes, vm, luma, lvl, lvlOff, lvlStride, 0, &lut)
		}
		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("%s vertical=%v luma=%v vm=%x: first diff at byte %d: got %d want %d", name, vertical, luma, vm, i, got[i], want[i])
				}
			}
		}
	}
}

// TestFilterSB8ExtremeTaps drives the 8-bit superblock kernels at the
// boundaries of their exact rewrites: flat lanes whose taps sit at 254/255
// (the scaled flat accumulators peak 128 below 2^16) or 0/1, and the narrow
// filter's saturating f = iclip_diff(3*(q0-p0) + iclip_diff(p1-q1)) over every
// p0, q0 pair with p1, q1 at the int8 clamp points.
func TestFilterSB8ExtremeTaps(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	widthMasks := func(luma bool) [][3]uint32 {
		all := ^uint32(0)
		m := [][3]uint32{{all, 0, 0}, {0, all, 0}}
		if luma {
			m = append(m, [3]uint32{0, 0, all})
		}
		var mixed [3]uint32
		for u := 0; u < 32; u++ {
			w := rng.Intn(len(m))
			mixed[w] |= 1 << uint(u)
		}
		return append(m, mixed)
	}
	for _, luma := range []bool{true, false} {
		for _, vm := range widthMasks(luma) {
			for iter := 0; iter < 8; iter++ {
				for _, lo := range []uint8{0, 254} {
					sbExtremeRun(t, "flat", uint8(rng.Intn(MaxSharpness+1)), luma, vm, func(lane, k int) uint8 {
						return lo + uint8(rng.Intn(2))
					})
				}
				// Flat inner taps around a step: flat8in passes, flat8out
				// does not on some lanes.
				sbExtremeRun(t, "flatin", uint8(rng.Intn(MaxSharpness+1)), luma, vm, func(lane, k int) uint8 {
					if k < -4 || k > 3 {
						return uint8(rng.Intn(256))
					}
					return 254 + uint8(rng.Intn(2))
				})
			}
			// Sharpness 0 keeps I = 63, so lanes with |q0 - p0| up to 96 and
			// p1, q1 up to 63 away still pass the filter mask: 3*(q0-p0) and
			// p1 - q1 both reach the int8 clamp.
			dev := []int{-63, -40, 0, 40, 63}
			clip := func(v int) uint8 { return uint8(max(0, min(255, v))) }
			for rep := 0; rep < 2; rep++ {
				for p0 := 0; p0 < 256; p0++ {
					for half := 0; half < 2; half++ {
						d1 := make([]int, 128)
						d2 := make([]int, 128)
						for i := range d1 {
							d1[i], d2[i] = dev[rng.Intn(len(dev))], dev[rng.Intn(len(dev))]
						}
						q0 := func(lane int) int { return (lane*2 + half) & 255 }
						if rep == 1 {
							// Concentrate q0 within the fm window around p0.
							q0 = func(lane int) int { return p0 - 96 + (lane*3+half)%193 }
						}
						sbExtremeRun(t, "narrow", 0, luma, vm, func(lane, k int) uint8 {
							switch k {
							case -1:
								return uint8(p0)
							case 0:
								return clip(q0(lane))
							case -2:
								return clip(p0 + d1[lane])
							case 1:
								return clip(q0(lane) + d2[lane])
							}
							return uint8(rng.Intn(256))
						})
					}
				}
			}
		}
	}
}

// sbTestPlane fills a plane with smooth content plus noise so every filter
// width (including the flat 13-tap path) is exercised.
func sbTestPlane(rng *rand.Rand, w, h int) []byte {
	pix := make([]byte, w*h)
	for y := 0; y < h; y++ {
		base := rng.Intn(256)
		for x := 0; x < w; x++ {
			v := base
			switch rng.Intn(4) {
			case 0:
				v += rng.Intn(3) - 1
			case 1:
				v += rng.Intn(9) - 4
			case 2:
				v = rng.Intn(256)
			}
			if x%16 >= 8 && rng.Intn(3) == 0 {
				v += 20
			}
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			pix[y*w+x] = uint8(v)
		}
	}
	return pix
}

func sbTestMasks(rng *rand.Rand, luma bool, units int) [3]uint32 {
	var vm [3]uint32
	for n := 0; n < units; n++ {
		bit := uint32(1) << uint(n)
		switch rng.Intn(5) {
		case 0:
		case 1:
			vm[0] |= bit
		case 2, 3:
			vm[1] |= bit
		case 4:
			if luma {
				vm[2] |= bit
			} else {
				vm[1] |= bit
			}
		}
	}
	return vm
}

func sbTestLevels(rng *rand.Rand, n int) []uint8 {
	lvl := make([]uint8, 4*(n))
	for i := range len(lvl) / 4 {
		for c := 0; c < 4; c++ {
			if rng.Intn(6) != 0 {
				lvl[4*(i)+c] = uint8(rng.Intn(MaxLevel + 1))
			}
		}
	}
	return lvl
}

// TestFilterSB8MatchesScalar checks the dispatched superblock filters against
// the unit-by-unit dav1d C reference on random content, masks and levels.
func TestFilterSB8MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const w, h = 160, 160
	for iter := 0; iter < 4000; iter++ {
		luma := rng.Intn(2) == 0
		sharp := uint8(rng.Intn(MaxSharpness + 1))
		lut := NewSBLUT(sharp)
		pix := sbTestPlane(rng, w, h)
		lvlStride := w / 4
		lvl := sbTestLevels(rng, lvlStride*(h/4))
		comp := rng.Intn(4)
		units := 1 + rng.Intn(32)
		vm := sbTestMasks(rng, luma, units)
		vertical := rng.Intn(2) == 0
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		if vertical {
			// Column x in 4..(w-8) so every width's taps fit; rows from y0.
			x := 4 * (2 + rng.Intn(w/4-4))
			if vm[2] != 0 {
				x = 16 * (1 + rng.Intn(w/16-2))
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off := y0*w + x
			lvlOff := (y0/4)*lvlStride + x/4
			filterSBVertical8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			FilterSBVertical8(got, off, w, w-x, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			y := 4 * (2 + rng.Intn(h/4-4))
			if vm[2] != 0 {
				y = 16 * (1 + rng.Intn(h/16-2))
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off := y*w + x0
			lvlOff := (y/4)*lvlStride + x0/4
			filterSBHorizontal8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			FilterSBHorizontal8(got, off, w, w-x0, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		if !bytes.Equal(got, want) {
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("iter %d vertical=%v luma=%v vm=%x: first diff at (%d,%d): got %d want %d", iter, vertical, luma, vm, i%w, i/w, got[i], want[i])
				}
			}
		}
	}
}

// TestFilterSB8ScalarMatchesEdgeKernels checks the C reference against the
// package's existing per-edge kernels, unit by unit, for every width.
func TestFilterSB8ScalarMatchesEdgeKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const w, h = 64, 64
	for iter := 0; iter < 3000; iter++ {
		sharp := uint8(rng.Intn(MaxSharpness + 1))
		lut := NewSBLUT(sharp)
		level := uint8(1 + rng.Intn(MaxLevel))
		th, err := ThresholdsForLevel(level, sharp)
		if err != nil {
			t.Fatal(err)
		}
		widths := []int{4, 6, 8, 16}
		wd := widths[rng.Intn(len(widths))]
		pix := sbTestPlane(rng, w, h)
		want := append([]byte(nil), pix...)
		got := append([]byte(nil), pix...)
		edge := EdgeVertical
		x, y := 16+4*rng.Intn(8), 16+4*rng.Intn(8)
		if rng.Intn(2) == 0 {
			edge = EdgeHorizontal
		}
		plane := frame.Plane{Pix: want, Stride: w, Width: w, Height: h}
		kw := uint8(wd)
		if wd == 16 {
			kw = 14
		}
		if err := FilterEdgeByWidth(kw, plane, 1, 8, edge, int32(x), int32(y), 4, th); err != nil {
			t.Fatal(err)
		}
		if edge == EdgeVertical {
			lpfUnit8(got, y*w+x, w, 1, int(lut.E[level]), int(lut.I[level]), int(level>>4), wd)
		} else {
			lpfUnit8(got, y*w+x, 1, w, int(lut.E[level]), int(lut.I[level]), int(level>>4), wd)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("iter %d: width %d edge %v level %d sharp %d mismatch", iter, wd, edge, level, sharp)
		}
	}
}

// TestFilterSB8TightBuffers runs the superblock filters on planes and level
// maps cut right after the last sample and level the selected units use, so
// calls whose groups reach the end of either buffer (the bottom and right
// of a frame, an unpadded level map) take the bounds-checked edge path and
// still match the unit-by-unit reference.
func TestFilterSB8TightBuffers(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const w, h = 160, 160
	for iter := 0; iter < 4000; iter++ {
		luma := rng.Intn(2) == 0
		lut := NewSBLUT(uint8(rng.Intn(MaxSharpness + 1)))
		full := sbTestPlane(rng, w, h)
		lvlStride := w / 4
		lvlFull := sbTestLevels(rng, lvlStride*(h/4))
		comp := rng.Intn(4)
		units := 1 + rng.Intn(32)
		vm := sbTestMasks(rng, luma, units)
		all := vm[0] | vm[1] | vm[2]
		if all == 0 {
			continue
		}
		hi := 31 - bits.LeadingZeros32(all)
		vertical := rng.Intn(2) == 0
		var off, rowBytes, lvlOff, pixLen, lvlLen int
		if vertical {
			x := 4 * (2 + rng.Intn(w/4-3))
			if vm[2] != 0 {
				x = 8 + 4*rng.Intn(w/4-3)
			}
			y0 := 4 * rng.Intn(h/4-units+1)
			off, rowBytes, lvlOff = y0*w+x, w-x, (y0/4)*lvlStride+x/4
			pixLen = off + (4*hi+3)*w + 7
			lvlLen = lvlOff + hi*lvlStride + 1
		} else {
			y := 4 * (2 + rng.Intn(h/4-4))
			if vm[2] != 0 {
				y = 28 + 4*rng.Intn(h/4-8)
			}
			x0 := 4 * rng.Intn(w/4-units+1)
			off, rowBytes, lvlOff = y*w+x0, w-x0, (y/4)*lvlStride+x0/4
			pixLen = off + 6*w + 4*(hi+1)
			lvlLen = lvlOff + hi + 1
		}
		pixLen = min(pixLen, len(full))
		if rng.Intn(2) == 0 {
			lvlLen = len(lvlFull) / 4
		}
		lvl := append([]uint8(nil), lvlFull[:4*lvlLen]...)
		want := append([]byte(nil), full[:pixLen]...)
		got := append([]byte(nil), full[:pixLen]...)
		if vertical {
			filterSBVertical8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			FilterSBVertical8(got, off, w, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		} else {
			filterSBHorizontal8Scalar(want, off, w, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
			FilterSBHorizontal8(got, off, w, rowBytes, vm, luma, lvl, lvlOff, lvlStride, comp, &lut)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("iter %d vertical=%v luma=%v vm=%x: samples differ from the reference", iter, vertical, luma, vm)
		}
	}
}
