package transform

import (
	"math"
	"testing"
)

func TestInverseADST1DRoundTripMatchesLibaomShape(t *testing.T) {
	for _, length := range []int{4, 8, 16} {
		input := make([]int32, length)
		for i := range input {
			input[i] = int32(((i*41+23)%37)-18) * 3
		}
		got := referenceADST1DInt(input)
		inverseADST1D(got, 1, length, minInt16, maxInt16)
		normalizationShift := getMaxBit(length) - 1
		for i := range got {
			normalized := clipInt32(roundShift(int64(got[i]), normalizationShift))
			if diff := abs32(normalized - input[i]); diff > 2 {
				t.Fatalf("length=%d coeff=%d got=%d normalized=%d want=%d diff=%d", length, i, got[i], normalized, input[i], diff)
			}
		}
	}
}

func TestInverseADST1DStrided(t *testing.T) {
	compact := []int32{100, -20, 30, 7}
	want := append([]int32(nil), compact...)
	inverseADST1D(want, 1, 4, minInt16, maxInt16)

	values := []int32{100, 99, -20, 99, 30, 99, 7}
	inverseADST1D(values, 2, 4, minInt16, maxInt16)
	for i := 0; i < len(want); i++ {
		if values[i*2] != want[i] {
			t.Fatalf("values[%d]=%d want %d", i*2, values[i*2], want[i])
		}
	}
	for _, i := range []int{1, 3, 5} {
		if values[i] != 99 {
			t.Fatalf("sentinel values[%d]=%d want 99", i, values[i])
		}
	}
}

func TestInverseFlipADST1DReversesADST(t *testing.T) {
	for _, length := range []int{4, 8, 16} {
		input := make([]int32, length)
		for i := range input {
			input[i] = int32(((i*17+9)%29)-14) * 6
		}
		adst := append([]int32(nil), input...)
		flip := append([]int32(nil), input...)
		inverseADST1D(adst, 1, length, minInt16, maxInt16)
		inverseFlipADST1D(flip, 1, length, minInt16, maxInt16)
		for i := 0; i < length; i++ {
			if flip[i] != adst[length-1-i] {
				t.Fatalf("length=%d flip[%d]=%d want ADST[%d]=%d", length, i, flip[i], length-1-i, adst[length-1-i])
			}
		}
	}
}

func TestInverseADST4OverflowRegressionFromLibaom(t *testing.T) {
	values := []int32{300000, 0, 300000, 300000}
	inverseADST1D(values, 1, 4, minInt32, maxInt32)
	if values[0] <= 0 {
		t.Fatalf("ADST4 overflow regression values[0]=%d want > 0", values[0])
	}
}

func referenceADST1DInt(input []int32) []int32 {
	size := len(input)
	if size == 4 {
		return referenceFADST4New(input)
	}
	out := make([]int32, size)
	for k := 0; k < size; k++ {
		sum := 0.0
		for n := 0; n < size; n++ {
			angle := math.Pi * float64((2*n+1)*(2*k+1)) / float64(4*size)
			sum += float64(input[n]) * math.Sin(angle)
		}
		out[k] = int32(math.Round(sum))
	}
	return out
}

func referenceFADST4New(input []int32) []int32 {
	x0 := int64(input[0])
	x1 := int64(input[1])
	x2 := int64(input[2])
	x3 := int64(input[3])
	if x0|x1|x2|x3 == 0 {
		return []int32{0, 0, 0, 0}
	}

	s0 := int64(5283) * x0
	s1 := int64(15212) * x0
	s2 := int64(9929) * x1
	s3 := int64(5283) * x1
	s4 := int64(13377) * x2
	s5 := int64(15212) * x3
	s6 := int64(9929) * x3
	s7 := x0 + x1 - x3

	x0 = s0 + s2 + s5
	x1 = int64(13377) * s7
	x2 = s1 - s3 + s6
	x3 = s4

	s0 = x0 + x3
	s1 = x1
	s2 = x2 - x3
	s3 = x2 - x0 + x3

	return []int32{
		clipInt32(roundShift(s0, 14)),
		clipInt32(roundShift(s1, 14)),
		clipInt32(roundShift(s2, 14)),
		clipInt32(roundShift(s3, 14)),
	}
}

func getMaxBit(x int) int {
	maxBit := -1
	for x != 0 {
		x >>= 1
		maxBit++
	}
	return maxBit
}
