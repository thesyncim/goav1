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
	for trial := 0; trial < 5000; trial++ {
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
	for trial := 0; trial < 5000; trial++ {
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
	for trial := 0; trial < 5000; trial++ {
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
	for trial := 0; trial < 5000; trial++ {
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
