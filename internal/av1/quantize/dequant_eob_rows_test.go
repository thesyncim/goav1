package quantize

import (
	"slices"
	"testing"
)

// Independent scalar statement of the original flat dequant + separate row
// traversal. Do not use dequantScalar or the fused row collection here.
func dequantEOBRowsOriginal(dst []int32, coeff []int16, scan []int16, eob, height int, q Quantizer, txScale, bitDepth uint8) int {
	clear(dst)
	for i := 0; i < eob; i++ {
		pos := int(scan[i])
		factor := int64(q.AC)
		if pos == 0 {
			factor = int64(q.DC)
		}
		level := int64(coeff[pos])
		negative := level < 0
		if negative {
			level = -level
		}
		value := ((level * factor) & 0xffffff) >> txScale
		if negative {
			value = -value
		}
		limit := int64(1) << (7 + bitDepth)
		if value < -limit {
			value = -limit
		}
		if value >= limit {
			value = limit - 1
		}
		dst[pos] = int32(value)
	}
	rows := 0
	for _, pos := range scan[:eob] {
		row := int(pos) % height
		if row >= rows {
			rows = row + 1
		}
	}
	return rows
}

func TestDequantEOBRowsFlatOriginal(t *testing.T) {
	levels := []int16{-32768, -32767, -1025, -17, -1, 0, 1, 17, 1025, 32766, 32767}
	for _, dims := range [][2]int{{4, 4}, {4, 8}, {8, 4}, {8, 8}, {4, 16}, {16, 4}, {8, 32}, {32, 8}, {16, 16}, {16, 32}, {32, 16}, {32, 32}} {
		width, height := dims[0], dims[1]
		n := width * height
		coeff := make([]int16, n)
		for i := range coeff {
			coeff[i] = levels[(i*7+height)%len(levels)]
		}
		for _, order := range []string{"columns", "rows", "permuted", "zeros_last_row"} {
			scan := make([]int16, n)
			for i := range scan {
				switch order {
				case "columns":
					scan[i] = int16(i)
				case "rows":
					scan[i] = int16((i%width)*height + i/width)
				case "permuted":
					scan[i] = int16((i*13 + 7) % n)
				case "zeros_last_row":
					scan[i] = int16(n - 1 - i)
				}
			}
			c := slices.Clone(coeff)
			if order == "zeros_last_row" {
				for i := height - 1; i < n; i += height {
					c[i] = 0
				}
			}
			for _, q := range []Quantizer{{4, 8}, {1336, 1828}, {21387, 29247}, {1<<30 + 1, 1<<30 - 1}} {
				for _, bd := range []uint8{8, 10, 12} {
					for ts := uint8(0); ts <= 2; ts++ {
						for eob := 0; eob <= n; eob++ {
							if n > 64 && eob > 3 && eob < n-2 && eob%(n/8) != 0 {
								continue
							}
							want, got := make([]int32, n+8), make([]int32, n+8)
							for i := range got {
								got[i] = int32(i*29 - 12345)
								want[i] = got[i]
							}
							wc, ws := slices.Clone(c), slices.Clone(scan)
							wr := dequantEOBRowsOriginal(want[3:n+3:n+3], wc, ws, eob, height, q, ts, bd)
							gr := DequantizeBlockScaledBitDepthEOBRowsFlatTrusted(got[3:n+3:n+3], c, scan, eob, height, q, ts, bd)
							if gr != wr || !slices.Equal(got, want) {
								t.Fatalf("%dx%d %s q%v bd%d ts%d eob%d rows=%d want=%d: result/guard mismatch", width, height, order, q, bd, ts, eob, gr, wr)
							}
							if !slices.Equal(c, wc) || !slices.Equal(scan, ws) {
								t.Fatal("input mutated")
							}
						}
					}
				}
			}
		}
	}
}

func TestDequantEOBRowsFlatPartialPanicState(t *testing.T) {
	for _, scan := range [][]int16{{0, 3, -1, 2}, {0, 3, 16, 2}, {0, 3}} {
		coeff := make([]int16, 16)
		for i := range coeff {
			coeff[i] = int16(i + 1)
		}
		got, want := make([]int32, 16), make([]int32, 16)
		for i := range got {
			got[i] = 999
			want[i] = 999
		}
		panics := func(f func()) (yes bool) { defer func() { yes = recover() != nil }(); f(); return }
		wp := panics(func() { dequantEOBRowsOriginal(want, coeff, scan, 4, 4, Quantizer{7, 11}, 0, 8) })
		gp := panics(func() {
			DequantizeBlockScaledBitDepthEOBRowsFlatTrusted(got, coeff, scan, 4, 4, Quantizer{7, 11}, 0, 8)
		})
		if !wp || !gp || !slices.Equal(got, want) {
			t.Fatal("panic or partial destination state differs")
		}
	}
}
