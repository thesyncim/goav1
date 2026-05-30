// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package loopfilter

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

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
