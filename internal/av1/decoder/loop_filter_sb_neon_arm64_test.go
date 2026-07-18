//go:build arm64 && !purego

package decoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/loopfilter"
)

func TestFilterMaskLine8NEONMatchesTrustedKernels(t *testing.T) {
	const (
		stride = 192
		width  = 160
		height = 160
		cols   = 40
		x4     = 8
		y4     = 8
	)
	lut := lfmask.CalcEIH(3)
	for _, tc := range []struct {
		name    string
		chroma  bool
		edge    loopfilter.Edge
		classes int
	}{
		{name: "luma-horizontal", edge: loopfilter.EdgeHorizontal, classes: 3},
		{name: "luma-vertical", edge: loopfilter.EdgeVertical, classes: 3},
		{name: "chroma-horizontal", chroma: true, edge: loopfilter.EdgeHorizontal, classes: 2},
		{name: "chroma-vertical", chroma: true, edge: loopfilter.EdgeVertical, classes: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(0x51b00 + int64(tc.edge)*17))
			wantPix := make([]byte, stride*height)
			for y := 0; y < height; y++ {
				for x := 0; x < width; x++ {
					wantPix[y*stride+x] = 105
					if (tc.edge == loopfilter.EdgeVertical && x < x4*4) || (tc.edge == loopfilter.EdgeHorizontal && y < y4*4) {
						wantPix[y*stride+x] = 100
					}
				}
			}
			gotPix := append([]byte(nil), wantPix...)
			want := frame.Plane{Pix: wantPix, Stride: stride, Width: width, Height: height}
			got := frame.Plane{Pix: gotPix, Stride: stride, Width: width, Height: height}
			levels := make([][4]uint8, cols*cols)
			for i := range levels {
				for c := range levels[i] {
					levels[i][c] = uint8(rng.Intn(64))
					if (i+c)%11 == 0 {
						levels[i][c] = 0
					}
					if c == 0 && i%3 == 0 {
						levels[i][c] |= 0x80
					}
				}
			}

			component := int(tc.edge)
			if tc.chroma {
				component += 2
			}
			levelIndex := y4*cols + x4
			q0Base := y4*4*stride + x4*4
			if tc.chroma {
				var mask [2][2]uint16
				for off := 0; off < 32; off++ {
					if off%5 != 0 {
						class := (off*7 + int(tc.edge)) % tc.classes
						mask[class][off>>4] |= 1 << (off & 15)
					}
				}
				applyMaskLineReference(t, want, &mask, nil, levels, levelIndex, component, cols, x4, y4, tc.edge, 3)
				packed := [2]uint32{
					uint32(mask[0][0]) | uint32(mask[0][1])<<16,
					uint32(mask[1][0]) | uint32(mask[1][1])<<16,
				}
				filterChromaMaskLine8Trusted(got, q0Base, &packed, levels, levelIndex, component, cols, tc.edge, &lut)
			} else {
				var mask [3][2]uint16
				for off := 0; off < 32; off++ {
					if off%5 != 0 {
						class := (off*7 + int(tc.edge)) % tc.classes
						mask[class][off>>4] |= 1 << (off & 15)
					}
				}
				applyMaskLineReference(t, want, nil, &mask, levels, levelIndex, component, cols, x4, y4, tc.edge, 3)
				packed := [3]uint32{
					uint32(mask[0][0]) | uint32(mask[0][1])<<16,
					uint32(mask[1][0]) | uint32(mask[1][1])<<16,
					uint32(mask[2][0]) | uint32(mask[2][1])<<16,
				}
				filterLumaMaskLine8Trusted(got, q0Base, &packed, levels, levelIndex, component, cols, tc.edge, &lut)
			}
			for i := range wantPix {
				if wantPix[i] != gotPix[i] {
					t.Fatalf("pixel %d: got %d want %d", i, gotPix[i], wantPix[i])
				}
			}
		})
	}
}

func TestFilterMaskLine8NEONCapturedHorizontal4(t *testing.T) {
	const stride, cols = 128, 32
	pix := make([]byte, stride*16)
	samples := [...]byte{221, 226, 230, 234, 236, 244, 246, 86, 5, 9, 13, 20}
	for y, sample := range samples {
		pix[y*stride+108] = sample
	}
	wantPix := append([]byte(nil), pix...)
	want := frame.Plane{Pix: wantPix, Stride: stride, Width: stride, Height: 16}
	got := frame.Plane{Pix: pix, Stride: stride, Width: stride, Height: 16}
	levels := make([][4]uint8, cols*16)
	for x := 0; x < cols; x++ {
		levels[x][1] = 25
		levels[cols+x][1] = 25
	}
	var mask [3][2]uint16
	mask[0][1] = 0xc00
	applyMaskLineReference(t, want, nil, &mask, levels, cols, 1, cols, 0, 1, loopfilter.EdgeHorizontal, 0)
	runPix := append([]byte(nil), pix...)
	run := frame.Plane{Pix: runPix, Stride: stride, Width: stride, Height: 16}
	th, err := loopfilter.ThresholdsForLevel(25, 0)
	if err != nil {
		t.Fatal(err)
	}
	loopfilter.Filter4EdgeTrusted(run, 1, 8, loopfilter.EdgeHorizontal, 104, 4, 8, th)
	for i := range runPix {
		if runPix[i] != wantPix[i] {
			t.Fatalf("run pixel %d: got %d want %d", i, runPix[i], wantPix[i])
		}
	}
	lut := lfmask.CalcEIH(0)
	packed := [3]uint32{uint32(mask[0][0]) | uint32(mask[0][1])<<16}
	filterLumaMaskLine8Trusted(got, 4*stride, &packed, levels, cols, 1, cols, loopfilter.EdgeHorizontal, &lut)
	for i := range pix {
		if pix[i] != wantPix[i] {
			t.Fatalf("pixel %d: got %d want %d", i, pix[i], wantPix[i])
		}
	}
}

func applyMaskLineReference(t *testing.T, dst frame.Plane, chroma *[2][2]uint16, luma *[3][2]uint16, levels [][4]uint8, levelIndex, component, levelStride, x4, y4 int, edge loopfilter.Edge, sharpness uint8) {
	t.Helper()
	classes := 3
	if chroma != nil {
		classes = 2
	}
	for off := 0; off < 32; off++ {
		class := -1
		for c := 0; c < classes; c++ {
			var set bool
			if chroma != nil {
				set = chroma[c][off>>4]&(1<<(off&15)) != 0
			} else {
				set = luma[c][off>>4]&(1<<(off&15)) != 0
			}
			if set {
				class = c
			}
		}
		if class < 0 {
			continue
		}
		idx := levelIndex + off
		if edge == loopfilter.EdgeVertical {
			idx = levelIndex + off*levelStride
		}
		level := levels[idx][component] & loopfilter.MaxLevel
		if level == 0 {
			if edge == loopfilter.EdgeVertical {
				level = levels[idx-1][component] & loopfilter.MaxLevel
			} else {
				level = levels[idx-levelStride][component] & loopfilter.MaxLevel
			}
		}
		if level == 0 {
			continue
		}
		th, err := loopfilter.ThresholdsForLevel(level, sharpness)
		if err != nil {
			t.Fatal(err)
		}
		x := x4 * 4
		y := y4 * 4
		if edge == loopfilter.EdgeVertical {
			y += off * 4
		} else {
			x += off * 4
		}
		width := 4
		if chroma != nil && class == 1 {
			width = 6
		} else if chroma == nil && class == 1 {
			width = 8
		} else if chroma == nil && class == 2 {
			width = 14
		}
		switch width {
		case 4:
			loopfilter.Filter4EdgeTrusted(dst, 1, 8, edge, x, y, 4, th)
		case 6:
			loopfilter.Filter6EdgeTrusted(dst, 1, 8, edge, x, y, 4, th)
		case 8:
			loopfilter.Filter8EdgeTrusted(dst, 1, 8, edge, x, y, 4, th)
		case 14:
			loopfilter.Filter14EdgeTrusted(dst, 1, 8, edge, x, y, 4, th)
		}
	}
}
