package encoder

// svt_screen.go ports SVT-AV1's screen-content detection
// (Source/Lib/Codec/pic_analysis_process.c svt_aom_is_screen_content, the
// screen_content_mode 2 default) and the per-picture class bookkeeping of
// pd_process.c perform_sc_detection:
//
//   - Detection runs on I-slices only; every later picture takes the last
//     I-slice's classes (ctx->last_i_picture_sc_class*).
//   - screen_content_mode 0 forces every class off, 1 forces every class on,
//     2 detects on pictures up to INPUT_SIZE_1080p_RANGE.
//   - The 16x16 pass counts blocks with 2..4 luma colors (counts_1) and,
//     among them, those whose per-pixel variance exceeds 0 (counts_2). The
//     8x8 pass repeats it with a variance threshold of 16, reading the
//     variance through the BLOCK_16X16 function pointer (a 16x16 window at
//     the 8x8 block, scaled by the 8x8 pel count) exactly as upstream does.
//     The window past the right or bottom edge reads SVT's replicated
//     picture padding.

// svtSCClass holds a picture's sc_class0..sc_class4.
type svtSCClass struct {
	class0, class1, class2, class3, class4 bool
}

// svtSCCounts are svt_aom_is_screen_content's block counters, kept for the
// differential test against SVT's own counts.
type svtSCCounts struct {
	c16Colors, c16Var int
	c8Colors, c8Var   int
}

// svtInputSize1080pRange is INPUT_SIZE_1080p_RANGE.
const svtInputSize1080pRange = 4

// svtSCDetector holds the color-count scratch so detection allocates nothing.
type svtSCDetector struct {
	stamp [256]uint32
	gen   uint32
}

// validPaletteColors is is_valid_palette_nb_colors: true when the rows x cols
// block at src has more than one and at most threshold distinct values.
func (d *svtSCDetector) validPaletteColors(src []byte, stride, rows, cols, threshold int) bool {
	d.gen++
	if d.gen == 0 {
		d.stamp = [256]uint32{}
		d.gen = 1
	}
	g := d.gen
	n := 0
	for r := 0; r < rows; r++ {
		row := src[r*stride : r*stride+cols]
		for _, v := range row {
			if d.stamp[v] != g {
				d.stamp[v] = g
				n++
				if n > threshold {
					return false
				}
			}
		}
	}
	return n > 1
}

// svtPerPixelVariance16 is svt_av1_get_sby_perpixel_variance through the
// BLOCK_16X16 variance function (svt_aom_variance16x16 against the constant
// 128 block): the 16x16 window at (x, y), samples outside the w x h picture
// replicated from its edge, rounded by 2^log2Pels.
func svtPerPixelVariance16(src []byte, stride, w, h, x, y int, log2Pels uint) uint32 {
	var sum int32
	var sse uint32
	if x+16 <= w && y+16 <= h {
		for r := 0; r < 16; r++ {
			row := src[(y+r)*stride+x : (y+r)*stride+x+16]
			for _, v := range row {
				d := int32(v) - 128
				sum += d
				sse += uint32(d * d)
			}
		}
	} else {
		for r := 0; r < 16; r++ {
			yy := min(y+r, h-1)
			row := src[yy*stride : yy*stride+w]
			for c := 0; c < 16; c++ {
				d := int32(row[min(x+c, w-1)]) - 128
				sum += d
				sse += uint32(d * d)
			}
		}
	}
	v := sse - uint32((int64(sum)*int64(sum))/256)
	return (v + (1 << log2Pels >> 1)) >> log2Pels
}

// detect is svt_aom_is_screen_content on the w x h luma plane.
func (d *svtSCDetector) detect(y []byte, stride, w, h int) (svtSCClass, svtSCCounts) {
	var cnt svtSCCounts
	for r := 0; r+16 <= h; r += 16 {
		for c := 0; c+16 <= w; c += 16 {
			if d.validPaletteColors(y[r*stride+c:], stride, 16, 16, 4) {
				cnt.c16Colors++
				if svtPerPixelVariance16(y, stride, w, h, c, r, 8) > 0 {
					cnt.c16Var++
				}
			}
		}
	}
	area := w * h
	var cls svtSCClass
	cls.class0 = cnt.c16Colors*16*16*10 > area
	cls.class1 = cls.class0 && cnt.c16Var*16*16*12 > area
	cls.class2 = cls.class1 || (cnt.c16Colors*16*16*10 > area*4 && cnt.c16Var*16*16*30 > area)
	cls.class3 = cls.class1 || (cnt.c16Colors*16*16*8 > area && cnt.c16Var*16*16*50 > area)

	for r := 0; r+8 <= h; r += 8 {
		for c := 0; c+8 <= w; c += 8 {
			if d.validPaletteColors(y[r*stride+c:], stride, 8, 8, 4) {
				cnt.c8Colors++
				if svtPerPixelVariance16(y, stride, w, h, c, r, 6) > 16 {
					cnt.c8Var++
				}
			}
		}
	}
	cls.class4 = cnt.c8Colors*8*8*18 > area && cnt.c8Var*8*8*20 > area
	return cls, cnt
}

// svtScreenClasses is perform_sc_detection's I-slice step for the given
// content mode on the coded-size luma plane.
func (d *svtSCDetector) classes(content ContentHint, y []byte, stride, w, h int) svtSCClass {
	switch content {
	case ContentCamera:
		return svtSCClass{}
	case ContentScreen:
		return svtSCClass{true, true, true, true, true}
	}
	if svtInputResolution(w, h) > svtInputSize1080pRange {
		return svtSCClass{}
	}
	cls, _ := d.detect(y, stride, w, h)
	return cls
}
