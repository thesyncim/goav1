package transform

import (
	"math/rand"
	"testing"
)

// libaomCospi12 is libaom's av1_cospi_arr_data[cos_bit-cos_bit_min] for the
// cos_bit=12 row used by every 1D inverse transform.
var libaomCospi12 = [...]int32{
	4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973,
	3948, 3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564,
	3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896,
	2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019,
	1931, 1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995,
	897, 799, 700, 601, 501, 401, 301, 201, 101,
}

// libaomHalfBtf reproduces libaom's half_btf for cos_bit=12.
func libaomHalfBtf(w0, in0, w1, in1 int32) int32 {
	v := int64(w0)*int64(in0) + int64(w1)*int64(in1)
	v += int64(1) << 11
	return int32(v >> 12)
}

// libaomClamp16 mirrors libaom's clamp_value with bit=16.
func libaomClamp16(v int32) int32 {
	if v < -(1 << 15) {
		return -(1 << 15)
	}
	if v > (1<<15)-1 {
		return (1 << 15) - 1
	}
	return v
}

// libaomIDCT4 reproduces libaom's av1_idct4 for cos_bit=12 and 16-bit stage
// range.
func libaomIDCT4(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	output[0] = input[0]
	output[1] = input[2]
	output[2] = input[1]
	output[3] = input[3]
	var step [4]int32
	step[0] = libaomHalfBtf(cospi[32], output[0], cospi[32], output[1])
	step[1] = libaomHalfBtf(cospi[32], output[0], -cospi[32], output[1])
	step[2] = libaomHalfBtf(cospi[48], output[2], -cospi[16], output[3])
	step[3] = libaomHalfBtf(cospi[16], output[2], cospi[48], output[3])
	output[0] = libaomClamp16(step[0] + step[3])
	output[1] = libaomClamp16(step[1] + step[2])
	output[2] = libaomClamp16(step[1] - step[2])
	output[3] = libaomClamp16(step[0] - step[3])
}

// libaomIDCT8 reproduces libaom's av1_idct8 for cos_bit=12 and 16-bit stage
// range.
func libaomIDCT8(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	var step [8]int32
	bf0 := output
	bf1 := output
	bf1[0] = input[0]
	bf1[1] = input[4]
	bf1[2] = input[2]
	bf1[3] = input[6]
	bf1[4] = input[1]
	bf1[5] = input[5]
	bf1[6] = input[3]
	bf1[7] = input[7]

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = libaomHalfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = libaomHalfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = libaomHalfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = libaomHalfBtf(cospi[8], bf0[4], cospi[56], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = libaomHalfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = libaomHalfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = libaomHalfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = libaomHalfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = libaomClamp16(bf0[4] + bf0[5])
	bf1[5] = libaomClamp16(bf0[4] - bf0[5])
	bf1[6] = libaomClamp16(-bf0[6] + bf0[7])
	bf1[7] = libaomClamp16(bf0[6] + bf0[7])

	bf0 = output
	bf1 = step[:]
	bf1[0] = libaomClamp16(bf0[0] + bf0[3])
	bf1[1] = libaomClamp16(bf0[1] + bf0[2])
	bf1[2] = libaomClamp16(bf0[1] - bf0[2])
	bf1[3] = libaomClamp16(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = libaomHalfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = libaomHalfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]

	bf0 = step[:]
	bf1 = output
	bf1[0] = libaomClamp16(bf0[0] + bf0[7])
	bf1[1] = libaomClamp16(bf0[1] + bf0[6])
	bf1[2] = libaomClamp16(bf0[2] + bf0[5])
	bf1[3] = libaomClamp16(bf0[3] + bf0[4])
	bf1[4] = libaomClamp16(bf0[3] - bf0[4])
	bf1[5] = libaomClamp16(bf0[2] - bf0[5])
	bf1[6] = libaomClamp16(bf0[1] - bf0[6])
	bf1[7] = libaomClamp16(bf0[0] - bf0[7])
}

// libaomIADST8 reproduces libaom's av1_iadst8 for cos_bit=12 and 16-bit stage
// range.
func libaomIADST8(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [8]int32
	bf0 := output
	bf1 := output

	bf1[0] = input[7]
	bf1[1] = input[0]
	bf1[2] = input[5]
	bf1[3] = input[2]
	bf1[4] = input[3]
	bf1[5] = input[4]
	bf1[6] = input[1]
	bf1[7] = input[6]

	bf0 = output
	bf1 = step[:]
	bf1[0] = halfBtf(cospi[4], bf0[0], cospi[60], bf0[1])
	bf1[1] = halfBtf(cospi[60], bf0[0], -cospi[4], bf0[1])
	bf1[2] = halfBtf(cospi[20], bf0[2], cospi[44], bf0[3])
	bf1[3] = halfBtf(cospi[44], bf0[2], -cospi[20], bf0[3])
	bf1[4] = halfBtf(cospi[36], bf0[4], cospi[28], bf0[5])
	bf1[5] = halfBtf(cospi[28], bf0[4], -cospi[36], bf0[5])
	bf1[6] = halfBtf(cospi[52], bf0[6], cospi[12], bf0[7])
	bf1[7] = halfBtf(cospi[12], bf0[6], -cospi[52], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[4])
	bf1[1] = clamp(bf0[1] + bf0[5])
	bf1[2] = clamp(bf0[2] + bf0[6])
	bf1[3] = clamp(bf0[3] + bf0[7])
	bf1[4] = clamp(bf0[0] - bf0[4])
	bf1[5] = clamp(bf0[1] - bf0[5])
	bf1[6] = clamp(bf0[2] - bf0[6])
	bf1[7] = clamp(bf0[3] - bf0[7])

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = halfBtf(cospi[16], bf0[4], cospi[48], bf0[5])
	bf1[5] = halfBtf(cospi[48], bf0[4], -cospi[16], bf0[5])
	bf1[6] = halfBtf(-cospi[48], bf0[6], cospi[16], bf0[7])
	bf1[7] = halfBtf(cospi[16], bf0[6], cospi[48], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[2])
	bf1[1] = clamp(bf0[1] + bf0[3])
	bf1[2] = clamp(bf0[0] - bf0[2])
	bf1[3] = clamp(bf0[1] - bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[6])
	bf1[5] = clamp(bf0[5] + bf0[7])
	bf1[6] = clamp(bf0[4] - bf0[6])
	bf1[7] = clamp(bf0[5] - bf0[7])

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = halfBtf(cospi[32], bf0[2], cospi[32], bf0[3])
	bf1[3] = halfBtf(cospi[32], bf0[2], -cospi[32], bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = halfBtf(cospi[32], bf0[6], cospi[32], bf0[7])
	bf1[7] = halfBtf(cospi[32], bf0[6], -cospi[32], bf0[7])

	bf0 = step[:]
	bf1 = output
	bf1[0] = bf0[0]
	bf1[1] = -bf0[4]
	bf1[2] = bf0[6]
	bf1[3] = -bf0[2]
	bf1[4] = bf0[3]
	bf1[5] = -bf0[7]
	bf1[6] = bf0[5]
	bf1[7] = -bf0[1]
}

// libaomIDCT16 reproduces libaom's av1_idct16 for cos_bit=12 and 16-bit stage
// range.
func libaomIDCT16(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [16]int32
	bf0 := output
	bf1 := output

	bf1[0] = input[0]
	bf1[1] = input[8]
	bf1[2] = input[4]
	bf1[3] = input[12]
	bf1[4] = input[2]
	bf1[5] = input[10]
	bf1[6] = input[6]
	bf1[7] = input[14]
	bf1[8] = input[1]
	bf1[9] = input[9]
	bf1[10] = input[5]
	bf1[11] = input[13]
	bf1[12] = input[3]
	bf1[13] = input[11]
	bf1[14] = input[7]
	bf1[15] = input[15]

	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = bf0[6]
	bf1[7] = bf0[7]
	bf1[8] = halfBtf(cospi[60], bf0[8], -cospi[4], bf0[15])
	bf1[9] = halfBtf(cospi[28], bf0[9], -cospi[36], bf0[14])
	bf1[10] = halfBtf(cospi[44], bf0[10], -cospi[20], bf0[13])
	bf1[11] = halfBtf(cospi[12], bf0[11], -cospi[52], bf0[12])
	bf1[12] = halfBtf(cospi[52], bf0[11], cospi[12], bf0[12])
	bf1[13] = halfBtf(cospi[20], bf0[10], cospi[44], bf0[13])
	bf1[14] = halfBtf(cospi[36], bf0[9], cospi[28], bf0[14])
	bf1[15] = halfBtf(cospi[4], bf0[8], cospi[60], bf0[15])

	bf0 = step[:]
	bf1 = output
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = bf0[2]
	bf1[3] = bf0[3]
	bf1[4] = halfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = halfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = halfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = halfBtf(cospi[8], bf0[4], cospi[56], bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[9])
	bf1[9] = clamp(bf0[8] - bf0[9])
	bf1[10] = clamp(-bf0[10] + bf0[11])
	bf1[11] = clamp(bf0[10] + bf0[11])
	bf1[12] = clamp(bf0[12] + bf0[13])
	bf1[13] = clamp(bf0[12] - bf0[13])
	bf1[14] = clamp(-bf0[14] + bf0[15])
	bf1[15] = clamp(bf0[14] + bf0[15])

	bf0 = output
	bf1 = step[:]
	bf1[0] = halfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = halfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = halfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = halfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[5])
	bf1[5] = clamp(bf0[4] - bf0[5])
	bf1[6] = clamp(-bf0[6] + bf0[7])
	bf1[7] = clamp(bf0[6] + bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = halfBtf(-cospi[16], bf0[9], cospi[48], bf0[14])
	bf1[10] = halfBtf(-cospi[48], bf0[10], -cospi[16], bf0[13])
	bf1[11] = bf0[11]
	bf1[12] = bf0[12]
	bf1[13] = halfBtf(-cospi[16], bf0[10], cospi[48], bf0[13])
	bf1[14] = halfBtf(cospi[48], bf0[9], cospi[16], bf0[14])
	bf1[15] = bf0[15]

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[3])
	bf1[1] = clamp(bf0[1] + bf0[2])
	bf1[2] = clamp(bf0[1] - bf0[2])
	bf1[3] = clamp(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = halfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = halfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]
	bf1[8] = clamp(bf0[8] + bf0[11])
	bf1[9] = clamp(bf0[9] + bf0[10])
	bf1[10] = clamp(bf0[9] - bf0[10])
	bf1[11] = clamp(bf0[8] - bf0[11])
	bf1[12] = clamp(-bf0[12] + bf0[15])
	bf1[13] = clamp(-bf0[13] + bf0[14])
	bf1[14] = clamp(bf0[13] + bf0[14])
	bf1[15] = clamp(bf0[12] + bf0[15])

	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[7])
	bf1[1] = clamp(bf0[1] + bf0[6])
	bf1[2] = clamp(bf0[2] + bf0[5])
	bf1[3] = clamp(bf0[3] + bf0[4])
	bf1[4] = clamp(bf0[3] - bf0[4])
	bf1[5] = clamp(bf0[2] - bf0[5])
	bf1[6] = clamp(bf0[1] - bf0[6])
	bf1[7] = clamp(bf0[0] - bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = halfBtf(-cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[11] = halfBtf(-cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[12] = halfBtf(cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[13] = halfBtf(cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[14] = bf0[14]
	bf1[15] = bf0[15]

	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[15])
	bf1[1] = clamp(bf0[1] + bf0[14])
	bf1[2] = clamp(bf0[2] + bf0[13])
	bf1[3] = clamp(bf0[3] + bf0[12])
	bf1[4] = clamp(bf0[4] + bf0[11])
	bf1[5] = clamp(bf0[5] + bf0[10])
	bf1[6] = clamp(bf0[6] + bf0[9])
	bf1[7] = clamp(bf0[7] + bf0[8])
	bf1[8] = clamp(bf0[7] - bf0[8])
	bf1[9] = clamp(bf0[6] - bf0[9])
	bf1[10] = clamp(bf0[5] - bf0[10])
	bf1[11] = clamp(bf0[4] - bf0[11])
	bf1[12] = clamp(bf0[3] - bf0[12])
	bf1[13] = clamp(bf0[2] - bf0[13])
	bf1[14] = clamp(bf0[1] - bf0[14])
	bf1[15] = clamp(bf0[0] - bf0[15])
}

func TestInverseDCT4BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := range 5000 {
		var input [4]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseDCT4(got[:], 1, minInt16, maxInt16)
		var want [4]int32
		libaomIDCT4(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d input=%v got=%v want=%v", trial, input, got, want)
			}
		}
	}
}

func TestInverseDCT8BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := range 5000 {
		var input [8]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseDCT8(got[:], 1, minInt16, maxInt16)
		var want [8]int32
		libaomIDCT8(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d input=%v got=%v want=%v", trial, input, got, want)
			}
		}
	}
}

func TestInverseDCT16BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := range 5000 {
		var input [16]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseDCT16(got[:], 1, minInt16, maxInt16)
		var want [16]int32
		libaomIDCT16(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d coeff=%d got=%d want=%d", trial, i, got[i], want[i])
			}
		}
	}
}

func TestInverseADST8BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := range 5000 {
		var input [8]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseADST8(got[:], 1, minInt16, maxInt16)
		var want [8]int32
		libaomIADST8(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d input=%v\n  goav1=%v\n  libaom=%v", trial, input, got, want)
			}
		}
	}
}

// libaomIDCT32 reproduces libaom's av1_idct32 for cos_bit=12 and 16-bit stage
// range. Faithfully follows av1/common/av1_inv_txfm1d.c:av1_idct32.
func libaomIDCT32(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [32]int32
	bf0 := output
	bf1 := output

	// stage 1: bit reverse
	bf1[0] = input[0]
	bf1[1] = input[16]
	bf1[2] = input[8]
	bf1[3] = input[24]
	bf1[4] = input[4]
	bf1[5] = input[20]
	bf1[6] = input[12]
	bf1[7] = input[28]
	bf1[8] = input[2]
	bf1[9] = input[18]
	bf1[10] = input[10]
	bf1[11] = input[26]
	bf1[12] = input[6]
	bf1[13] = input[22]
	bf1[14] = input[14]
	bf1[15] = input[30]
	bf1[16] = input[1]
	bf1[17] = input[17]
	bf1[18] = input[9]
	bf1[19] = input[25]
	bf1[20] = input[5]
	bf1[21] = input[21]
	bf1[22] = input[13]
	bf1[23] = input[29]
	bf1[24] = input[3]
	bf1[25] = input[19]
	bf1[26] = input[11]
	bf1[27] = input[27]
	bf1[28] = input[7]
	bf1[29] = input[23]
	bf1[30] = input[15]
	bf1[31] = input[31]

	// stage 2
	bf0 = output
	bf1 = step[:]
	for i := 0; i < 16; i++ {
		bf1[i] = bf0[i]
	}
	bf1[16] = halfBtf(cospi[62], bf0[16], -cospi[2], bf0[31])
	bf1[17] = halfBtf(cospi[30], bf0[17], -cospi[34], bf0[30])
	bf1[18] = halfBtf(cospi[46], bf0[18], -cospi[18], bf0[29])
	bf1[19] = halfBtf(cospi[14], bf0[19], -cospi[50], bf0[28])
	bf1[20] = halfBtf(cospi[54], bf0[20], -cospi[10], bf0[27])
	bf1[21] = halfBtf(cospi[22], bf0[21], -cospi[42], bf0[26])
	bf1[22] = halfBtf(cospi[38], bf0[22], -cospi[26], bf0[25])
	bf1[23] = halfBtf(cospi[6], bf0[23], -cospi[58], bf0[24])
	bf1[24] = halfBtf(cospi[58], bf0[23], cospi[6], bf0[24])
	bf1[25] = halfBtf(cospi[26], bf0[22], cospi[38], bf0[25])
	bf1[26] = halfBtf(cospi[42], bf0[21], cospi[22], bf0[26])
	bf1[27] = halfBtf(cospi[10], bf0[20], cospi[54], bf0[27])
	bf1[28] = halfBtf(cospi[50], bf0[19], cospi[14], bf0[28])
	bf1[29] = halfBtf(cospi[18], bf0[18], cospi[46], bf0[29])
	bf1[30] = halfBtf(cospi[34], bf0[17], cospi[30], bf0[30])
	bf1[31] = halfBtf(cospi[2], bf0[16], cospi[62], bf0[31])

	// stage 3
	bf0 = step[:]
	bf1 = output
	for i := 0; i < 8; i++ {
		bf1[i] = bf0[i]
	}
	bf1[8] = halfBtf(cospi[60], bf0[8], -cospi[4], bf0[15])
	bf1[9] = halfBtf(cospi[28], bf0[9], -cospi[36], bf0[14])
	bf1[10] = halfBtf(cospi[44], bf0[10], -cospi[20], bf0[13])
	bf1[11] = halfBtf(cospi[12], bf0[11], -cospi[52], bf0[12])
	bf1[12] = halfBtf(cospi[52], bf0[11], cospi[12], bf0[12])
	bf1[13] = halfBtf(cospi[20], bf0[10], cospi[44], bf0[13])
	bf1[14] = halfBtf(cospi[36], bf0[9], cospi[28], bf0[14])
	bf1[15] = halfBtf(cospi[4], bf0[8], cospi[60], bf0[15])
	bf1[16] = clamp(bf0[16] + bf0[17])
	bf1[17] = clamp(bf0[16] - bf0[17])
	bf1[18] = clamp(-bf0[18] + bf0[19])
	bf1[19] = clamp(bf0[18] + bf0[19])
	bf1[20] = clamp(bf0[20] + bf0[21])
	bf1[21] = clamp(bf0[20] - bf0[21])
	bf1[22] = clamp(-bf0[22] + bf0[23])
	bf1[23] = clamp(bf0[22] + bf0[23])
	bf1[24] = clamp(bf0[24] + bf0[25])
	bf1[25] = clamp(bf0[24] - bf0[25])
	bf1[26] = clamp(-bf0[26] + bf0[27])
	bf1[27] = clamp(bf0[26] + bf0[27])
	bf1[28] = clamp(bf0[28] + bf0[29])
	bf1[29] = clamp(bf0[28] - bf0[29])
	bf1[30] = clamp(-bf0[30] + bf0[31])
	bf1[31] = clamp(bf0[30] + bf0[31])

	// stage 4
	bf0 = output
	bf1 = step[:]
	for i := 0; i < 4; i++ {
		bf1[i] = bf0[i]
	}
	bf1[4] = halfBtf(cospi[56], bf0[4], -cospi[8], bf0[7])
	bf1[5] = halfBtf(cospi[24], bf0[5], -cospi[40], bf0[6])
	bf1[6] = halfBtf(cospi[40], bf0[5], cospi[24], bf0[6])
	bf1[7] = halfBtf(cospi[8], bf0[4], cospi[56], bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[9])
	bf1[9] = clamp(bf0[8] - bf0[9])
	bf1[10] = clamp(-bf0[10] + bf0[11])
	bf1[11] = clamp(bf0[10] + bf0[11])
	bf1[12] = clamp(bf0[12] + bf0[13])
	bf1[13] = clamp(bf0[12] - bf0[13])
	bf1[14] = clamp(-bf0[14] + bf0[15])
	bf1[15] = clamp(bf0[14] + bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = halfBtf(-cospi[8], bf0[17], cospi[56], bf0[30])
	bf1[18] = halfBtf(-cospi[56], bf0[18], -cospi[8], bf0[29])
	bf1[19] = bf0[19]
	bf1[20] = bf0[20]
	bf1[21] = halfBtf(-cospi[40], bf0[21], cospi[24], bf0[26])
	bf1[22] = halfBtf(-cospi[24], bf0[22], -cospi[40], bf0[25])
	bf1[23] = bf0[23]
	bf1[24] = bf0[24]
	bf1[25] = halfBtf(-cospi[40], bf0[22], cospi[24], bf0[25])
	bf1[26] = halfBtf(cospi[24], bf0[21], cospi[40], bf0[26])
	bf1[27] = bf0[27]
	bf1[28] = bf0[28]
	bf1[29] = halfBtf(-cospi[8], bf0[18], cospi[56], bf0[29])
	bf1[30] = halfBtf(cospi[56], bf0[17], cospi[8], bf0[30])
	bf1[31] = bf0[31]

	// stage 5
	bf0 = step[:]
	bf1 = output
	bf1[0] = halfBtf(cospi[32], bf0[0], cospi[32], bf0[1])
	bf1[1] = halfBtf(cospi[32], bf0[0], -cospi[32], bf0[1])
	bf1[2] = halfBtf(cospi[48], bf0[2], -cospi[16], bf0[3])
	bf1[3] = halfBtf(cospi[16], bf0[2], cospi[48], bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[5])
	bf1[5] = clamp(bf0[4] - bf0[5])
	bf1[6] = clamp(-bf0[6] + bf0[7])
	bf1[7] = clamp(bf0[6] + bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = halfBtf(-cospi[16], bf0[9], cospi[48], bf0[14])
	bf1[10] = halfBtf(-cospi[48], bf0[10], -cospi[16], bf0[13])
	bf1[11] = bf0[11]
	bf1[12] = bf0[12]
	bf1[13] = halfBtf(-cospi[16], bf0[10], cospi[48], bf0[13])
	bf1[14] = halfBtf(cospi[48], bf0[9], cospi[16], bf0[14])
	bf1[15] = bf0[15]
	bf1[16] = clamp(bf0[16] + bf0[19])
	bf1[17] = clamp(bf0[17] + bf0[18])
	bf1[18] = clamp(bf0[17] - bf0[18])
	bf1[19] = clamp(bf0[16] - bf0[19])
	bf1[20] = clamp(-bf0[20] + bf0[23])
	bf1[21] = clamp(-bf0[21] + bf0[22])
	bf1[22] = clamp(bf0[21] + bf0[22])
	bf1[23] = clamp(bf0[20] + bf0[23])
	bf1[24] = clamp(bf0[24] + bf0[27])
	bf1[25] = clamp(bf0[25] + bf0[26])
	bf1[26] = clamp(bf0[25] - bf0[26])
	bf1[27] = clamp(bf0[24] - bf0[27])
	bf1[28] = clamp(-bf0[28] + bf0[31])
	bf1[29] = clamp(-bf0[29] + bf0[30])
	bf1[30] = clamp(bf0[29] + bf0[30])
	bf1[31] = clamp(bf0[28] + bf0[31])

	// stage 6
	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[3])
	bf1[1] = clamp(bf0[1] + bf0[2])
	bf1[2] = clamp(bf0[1] - bf0[2])
	bf1[3] = clamp(bf0[0] - bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = halfBtf(-cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[6] = halfBtf(cospi[32], bf0[5], cospi[32], bf0[6])
	bf1[7] = bf0[7]
	bf1[8] = clamp(bf0[8] + bf0[11])
	bf1[9] = clamp(bf0[9] + bf0[10])
	bf1[10] = clamp(bf0[9] - bf0[10])
	bf1[11] = clamp(bf0[8] - bf0[11])
	bf1[12] = clamp(-bf0[12] + bf0[15])
	bf1[13] = clamp(-bf0[13] + bf0[14])
	bf1[14] = clamp(bf0[13] + bf0[14])
	bf1[15] = clamp(bf0[12] + bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = bf0[17]
	bf1[18] = halfBtf(-cospi[16], bf0[18], cospi[48], bf0[29])
	bf1[19] = halfBtf(-cospi[16], bf0[19], cospi[48], bf0[28])
	bf1[20] = halfBtf(-cospi[48], bf0[20], -cospi[16], bf0[27])
	bf1[21] = halfBtf(-cospi[48], bf0[21], -cospi[16], bf0[26])
	bf1[22] = bf0[22]
	bf1[23] = bf0[23]
	bf1[24] = bf0[24]
	bf1[25] = bf0[25]
	bf1[26] = halfBtf(-cospi[16], bf0[21], cospi[48], bf0[26])
	bf1[27] = halfBtf(-cospi[16], bf0[20], cospi[48], bf0[27])
	bf1[28] = halfBtf(cospi[48], bf0[19], cospi[16], bf0[28])
	bf1[29] = halfBtf(cospi[48], bf0[18], cospi[16], bf0[29])
	bf1[30] = bf0[30]
	bf1[31] = bf0[31]

	// stage 7
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[7])
	bf1[1] = clamp(bf0[1] + bf0[6])
	bf1[2] = clamp(bf0[2] + bf0[5])
	bf1[3] = clamp(bf0[3] + bf0[4])
	bf1[4] = clamp(bf0[3] - bf0[4])
	bf1[5] = clamp(bf0[2] - bf0[5])
	bf1[6] = clamp(bf0[1] - bf0[6])
	bf1[7] = clamp(bf0[0] - bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = halfBtf(-cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[11] = halfBtf(-cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[12] = halfBtf(cospi[32], bf0[11], cospi[32], bf0[12])
	bf1[13] = halfBtf(cospi[32], bf0[10], cospi[32], bf0[13])
	bf1[14] = bf0[14]
	bf1[15] = bf0[15]
	bf1[16] = clamp(bf0[16] + bf0[23])
	bf1[17] = clamp(bf0[17] + bf0[22])
	bf1[18] = clamp(bf0[18] + bf0[21])
	bf1[19] = clamp(bf0[19] + bf0[20])
	bf1[20] = clamp(bf0[19] - bf0[20])
	bf1[21] = clamp(bf0[18] - bf0[21])
	bf1[22] = clamp(bf0[17] - bf0[22])
	bf1[23] = clamp(bf0[16] - bf0[23])
	bf1[24] = clamp(-bf0[24] + bf0[31])
	bf1[25] = clamp(-bf0[25] + bf0[30])
	bf1[26] = clamp(-bf0[26] + bf0[29])
	bf1[27] = clamp(-bf0[27] + bf0[28])
	bf1[28] = clamp(bf0[27] + bf0[28])
	bf1[29] = clamp(bf0[26] + bf0[29])
	bf1[30] = clamp(bf0[25] + bf0[30])
	bf1[31] = clamp(bf0[24] + bf0[31])

	// stage 8
	bf0 = output
	bf1 = step[:]
	bf1[0] = clamp(bf0[0] + bf0[15])
	bf1[1] = clamp(bf0[1] + bf0[14])
	bf1[2] = clamp(bf0[2] + bf0[13])
	bf1[3] = clamp(bf0[3] + bf0[12])
	bf1[4] = clamp(bf0[4] + bf0[11])
	bf1[5] = clamp(bf0[5] + bf0[10])
	bf1[6] = clamp(bf0[6] + bf0[9])
	bf1[7] = clamp(bf0[7] + bf0[8])
	bf1[8] = clamp(bf0[7] - bf0[8])
	bf1[9] = clamp(bf0[6] - bf0[9])
	bf1[10] = clamp(bf0[5] - bf0[10])
	bf1[11] = clamp(bf0[4] - bf0[11])
	bf1[12] = clamp(bf0[3] - bf0[12])
	bf1[13] = clamp(bf0[2] - bf0[13])
	bf1[14] = clamp(bf0[1] - bf0[14])
	bf1[15] = clamp(bf0[0] - bf0[15])
	bf1[16] = bf0[16]
	bf1[17] = bf0[17]
	bf1[18] = bf0[18]
	bf1[19] = bf0[19]
	bf1[20] = halfBtf(-cospi[32], bf0[20], cospi[32], bf0[27])
	bf1[21] = halfBtf(-cospi[32], bf0[21], cospi[32], bf0[26])
	bf1[22] = halfBtf(-cospi[32], bf0[22], cospi[32], bf0[25])
	bf1[23] = halfBtf(-cospi[32], bf0[23], cospi[32], bf0[24])
	bf1[24] = halfBtf(cospi[32], bf0[23], cospi[32], bf0[24])
	bf1[25] = halfBtf(cospi[32], bf0[22], cospi[32], bf0[25])
	bf1[26] = halfBtf(cospi[32], bf0[21], cospi[32], bf0[26])
	bf1[27] = halfBtf(cospi[32], bf0[20], cospi[32], bf0[27])
	bf1[28] = bf0[28]
	bf1[29] = bf0[29]
	bf1[30] = bf0[30]
	bf1[31] = bf0[31]

	// stage 9
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[31])
	bf1[1] = clamp(bf0[1] + bf0[30])
	bf1[2] = clamp(bf0[2] + bf0[29])
	bf1[3] = clamp(bf0[3] + bf0[28])
	bf1[4] = clamp(bf0[4] + bf0[27])
	bf1[5] = clamp(bf0[5] + bf0[26])
	bf1[6] = clamp(bf0[6] + bf0[25])
	bf1[7] = clamp(bf0[7] + bf0[24])
	bf1[8] = clamp(bf0[8] + bf0[23])
	bf1[9] = clamp(bf0[9] + bf0[22])
	bf1[10] = clamp(bf0[10] + bf0[21])
	bf1[11] = clamp(bf0[11] + bf0[20])
	bf1[12] = clamp(bf0[12] + bf0[19])
	bf1[13] = clamp(bf0[13] + bf0[18])
	bf1[14] = clamp(bf0[14] + bf0[17])
	bf1[15] = clamp(bf0[15] + bf0[16])
	bf1[16] = clamp(bf0[15] - bf0[16])
	bf1[17] = clamp(bf0[14] - bf0[17])
	bf1[18] = clamp(bf0[13] - bf0[18])
	bf1[19] = clamp(bf0[12] - bf0[19])
	bf1[20] = clamp(bf0[11] - bf0[20])
	bf1[21] = clamp(bf0[10] - bf0[21])
	bf1[22] = clamp(bf0[9] - bf0[22])
	bf1[23] = clamp(bf0[8] - bf0[23])
	bf1[24] = clamp(bf0[7] - bf0[24])
	bf1[25] = clamp(bf0[6] - bf0[25])
	bf1[26] = clamp(bf0[5] - bf0[26])
	bf1[27] = clamp(bf0[4] - bf0[27])
	bf1[28] = clamp(bf0[3] - bf0[28])
	bf1[29] = clamp(bf0[2] - bf0[29])
	bf1[30] = clamp(bf0[1] - bf0[30])
	bf1[31] = clamp(bf0[0] - bf0[31])
}

func TestInverseDCT32BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for trial := range 1000 {
		var input [32]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseDCT32(got[:], 1, minInt16, maxInt16)
		var want [32]int32
		libaomIDCT32(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d coeff=%d got=%d want=%d\n input=%v\n  got=%v\n want=%v",
					trial, i, got[i], want[i], input, got, want)
			}
		}
	}
}

// libaomIADST16 reproduces libaom's av1_iadst16 for cos_bit=12 and 16-bit
// stage range. Mirrors av1/common/av1_inv_txfm1d.c:av1_iadst16.
func libaomIADST16(input []int32, output []int32) {
	cospi := libaomCospi12[:]
	halfBtf := libaomHalfBtf
	clamp := libaomClamp16
	var step [16]int32
	bf0 := output
	bf1 := output

	// stage 1
	bf1[0] = input[15]
	bf1[1] = input[0]
	bf1[2] = input[13]
	bf1[3] = input[2]
	bf1[4] = input[11]
	bf1[5] = input[4]
	bf1[6] = input[9]
	bf1[7] = input[6]
	bf1[8] = input[7]
	bf1[9] = input[8]
	bf1[10] = input[5]
	bf1[11] = input[10]
	bf1[12] = input[3]
	bf1[13] = input[12]
	bf1[14] = input[1]
	bf1[15] = input[14]

	// stage 2
	bf0 = output
	bf1 = step[:]
	bf1[0] = halfBtf(cospi[2], bf0[0], cospi[62], bf0[1])
	bf1[1] = halfBtf(cospi[62], bf0[0], -cospi[2], bf0[1])
	bf1[2] = halfBtf(cospi[10], bf0[2], cospi[54], bf0[3])
	bf1[3] = halfBtf(cospi[54], bf0[2], -cospi[10], bf0[3])
	bf1[4] = halfBtf(cospi[18], bf0[4], cospi[46], bf0[5])
	bf1[5] = halfBtf(cospi[46], bf0[4], -cospi[18], bf0[5])
	bf1[6] = halfBtf(cospi[26], bf0[6], cospi[38], bf0[7])
	bf1[7] = halfBtf(cospi[38], bf0[6], -cospi[26], bf0[7])
	bf1[8] = halfBtf(cospi[34], bf0[8], cospi[30], bf0[9])
	bf1[9] = halfBtf(cospi[30], bf0[8], -cospi[34], bf0[9])
	bf1[10] = halfBtf(cospi[42], bf0[10], cospi[22], bf0[11])
	bf1[11] = halfBtf(cospi[22], bf0[10], -cospi[42], bf0[11])
	bf1[12] = halfBtf(cospi[50], bf0[12], cospi[14], bf0[13])
	bf1[13] = halfBtf(cospi[14], bf0[12], -cospi[50], bf0[13])
	bf1[14] = halfBtf(cospi[58], bf0[14], cospi[6], bf0[15])
	bf1[15] = halfBtf(cospi[6], bf0[14], -cospi[58], bf0[15])

	// stage 3
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[8])
	bf1[1] = clamp(bf0[1] + bf0[9])
	bf1[2] = clamp(bf0[2] + bf0[10])
	bf1[3] = clamp(bf0[3] + bf0[11])
	bf1[4] = clamp(bf0[4] + bf0[12])
	bf1[5] = clamp(bf0[5] + bf0[13])
	bf1[6] = clamp(bf0[6] + bf0[14])
	bf1[7] = clamp(bf0[7] + bf0[15])
	bf1[8] = clamp(bf0[0] - bf0[8])
	bf1[9] = clamp(bf0[1] - bf0[9])
	bf1[10] = clamp(bf0[2] - bf0[10])
	bf1[11] = clamp(bf0[3] - bf0[11])
	bf1[12] = clamp(bf0[4] - bf0[12])
	bf1[13] = clamp(bf0[5] - bf0[13])
	bf1[14] = clamp(bf0[6] - bf0[14])
	bf1[15] = clamp(bf0[7] - bf0[15])

	// stage 4
	bf0 = output
	bf1 = step[:]
	for i := 0; i < 8; i++ {
		bf1[i] = bf0[i]
	}
	bf1[8] = halfBtf(cospi[8], bf0[8], cospi[56], bf0[9])
	bf1[9] = halfBtf(cospi[56], bf0[8], -cospi[8], bf0[9])
	bf1[10] = halfBtf(cospi[40], bf0[10], cospi[24], bf0[11])
	bf1[11] = halfBtf(cospi[24], bf0[10], -cospi[40], bf0[11])
	bf1[12] = halfBtf(-cospi[56], bf0[12], cospi[8], bf0[13])
	bf1[13] = halfBtf(cospi[8], bf0[12], cospi[56], bf0[13])
	bf1[14] = halfBtf(-cospi[24], bf0[14], cospi[40], bf0[15])
	bf1[15] = halfBtf(cospi[40], bf0[14], cospi[24], bf0[15])

	// stage 5
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[4])
	bf1[1] = clamp(bf0[1] + bf0[5])
	bf1[2] = clamp(bf0[2] + bf0[6])
	bf1[3] = clamp(bf0[3] + bf0[7])
	bf1[4] = clamp(bf0[0] - bf0[4])
	bf1[5] = clamp(bf0[1] - bf0[5])
	bf1[6] = clamp(bf0[2] - bf0[6])
	bf1[7] = clamp(bf0[3] - bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[12])
	bf1[9] = clamp(bf0[9] + bf0[13])
	bf1[10] = clamp(bf0[10] + bf0[14])
	bf1[11] = clamp(bf0[11] + bf0[15])
	bf1[12] = clamp(bf0[8] - bf0[12])
	bf1[13] = clamp(bf0[9] - bf0[13])
	bf1[14] = clamp(bf0[10] - bf0[14])
	bf1[15] = clamp(bf0[11] - bf0[15])

	// stage 6
	bf0 = output
	bf1 = step[:]
	for i := 0; i < 4; i++ {
		bf1[i] = bf0[i]
	}
	bf1[4] = halfBtf(cospi[16], bf0[4], cospi[48], bf0[5])
	bf1[5] = halfBtf(cospi[48], bf0[4], -cospi[16], bf0[5])
	bf1[6] = halfBtf(-cospi[48], bf0[6], cospi[16], bf0[7])
	bf1[7] = halfBtf(cospi[16], bf0[6], cospi[48], bf0[7])
	for i := 8; i < 12; i++ {
		bf1[i] = bf0[i]
	}
	bf1[12] = halfBtf(cospi[16], bf0[12], cospi[48], bf0[13])
	bf1[13] = halfBtf(cospi[48], bf0[12], -cospi[16], bf0[13])
	bf1[14] = halfBtf(-cospi[48], bf0[14], cospi[16], bf0[15])
	bf1[15] = halfBtf(cospi[16], bf0[14], cospi[48], bf0[15])

	// stage 7
	bf0 = step[:]
	bf1 = output
	bf1[0] = clamp(bf0[0] + bf0[2])
	bf1[1] = clamp(bf0[1] + bf0[3])
	bf1[2] = clamp(bf0[0] - bf0[2])
	bf1[3] = clamp(bf0[1] - bf0[3])
	bf1[4] = clamp(bf0[4] + bf0[6])
	bf1[5] = clamp(bf0[5] + bf0[7])
	bf1[6] = clamp(bf0[4] - bf0[6])
	bf1[7] = clamp(bf0[5] - bf0[7])
	bf1[8] = clamp(bf0[8] + bf0[10])
	bf1[9] = clamp(bf0[9] + bf0[11])
	bf1[10] = clamp(bf0[8] - bf0[10])
	bf1[11] = clamp(bf0[9] - bf0[11])
	bf1[12] = clamp(bf0[12] + bf0[14])
	bf1[13] = clamp(bf0[13] + bf0[15])
	bf1[14] = clamp(bf0[12] - bf0[14])
	bf1[15] = clamp(bf0[13] - bf0[15])

	// stage 8
	bf0 = output
	bf1 = step[:]
	bf1[0] = bf0[0]
	bf1[1] = bf0[1]
	bf1[2] = halfBtf(cospi[32], bf0[2], cospi[32], bf0[3])
	bf1[3] = halfBtf(cospi[32], bf0[2], -cospi[32], bf0[3])
	bf1[4] = bf0[4]
	bf1[5] = bf0[5]
	bf1[6] = halfBtf(cospi[32], bf0[6], cospi[32], bf0[7])
	bf1[7] = halfBtf(cospi[32], bf0[6], -cospi[32], bf0[7])
	bf1[8] = bf0[8]
	bf1[9] = bf0[9]
	bf1[10] = halfBtf(cospi[32], bf0[10], cospi[32], bf0[11])
	bf1[11] = halfBtf(cospi[32], bf0[10], -cospi[32], bf0[11])
	bf1[12] = bf0[12]
	bf1[13] = bf0[13]
	bf1[14] = halfBtf(cospi[32], bf0[14], cospi[32], bf0[15])
	bf1[15] = halfBtf(cospi[32], bf0[14], -cospi[32], bf0[15])

	// stage 9
	bf0 = step[:]
	bf1 = output
	bf1[0] = bf0[0]
	bf1[1] = -bf0[8]
	bf1[2] = bf0[12]
	bf1[3] = -bf0[4]
	bf1[4] = bf0[6]
	bf1[5] = -bf0[14]
	bf1[6] = bf0[10]
	bf1[7] = -bf0[2]
	bf1[8] = bf0[3]
	bf1[9] = -bf0[11]
	bf1[10] = bf0[15]
	bf1[11] = -bf0[7]
	bf1[12] = bf0[5]
	bf1[13] = -bf0[13]
	bf1[14] = bf0[9]
	bf1[15] = -bf0[1]
}

func TestInverseADST16BitExactLibaom(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for trial := range 1000 {
		var input [16]int32
		for i := range input {
			input[i] = int32(rng.Intn(65536) - 32768)
		}
		got := input
		inverseADST16(got[:], 1, minInt16, maxInt16)
		var want [16]int32
		libaomIADST16(input[:], want[:])
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial=%d coeff=%d got=%d want=%d\n input=%v\n  got=%v\n want=%v",
					trial, i, got[i], want[i], input, got, want)
			}
		}
	}
}
