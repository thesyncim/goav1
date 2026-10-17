package quantize

import "testing"

func TestDequantizeCoefficientWithBoundsTrustedMatchesFormula(t *testing.T) {
	levels := []int32{-32768, -32767, -1024, -2, -1, 0, 1, 2, 1024, 32767}
	for _, bitDepth := range []uint8{8, 10, 12} {
		dqMin, dqMax := DequantCoefficientBoundsTrusted(bitDepth)
		for _, qIndex := range []uint8{1, 32, 127, 255} {
			dc, err := DCQuant(qIndex, 0, bitDepth)
			if err != nil {
				t.Fatal(err)
			}
			ac, err := ACQuant(qIndex, 0, bitDepth)
			if err != nil {
				t.Fatal(err)
			}
			for _, factor := range []int32{dc, ac} {
				for _, txScale := range []uint8{0, 1, 2} {
					for _, level := range levels {
						got := DequantizeCoefficientWithBoundsTrusted(level, factor, txScale, dqMin, dqMax)
						want := testDequantCoefficientFormula(level, factor, txScale, bitDepth)
						if got != want {
							t.Fatalf("level=%d factor=%d txScale=%d bitDepth=%d got=%d want=%d", level, factor, txScale, bitDepth, got, want)
						}
					}
				}
			}
		}
	}
}

func TestDequantizeMagnitudeWithBoundsTrustedMatchesSignedPath(t *testing.T) {
	levels := []int32{0, 1, 2, 3, 14, 15, 16, 31, 127, 255, 1024, 8192, 16383, 16384, 16385, 32767, 32768}
	for _, bitDepth := range []uint8{8, 10, 12} {
		dqMin, dqMax := DequantCoefficientBoundsTrusted(bitDepth)
		for qIndex := 0; qIndex <= MaxQIndex; qIndex++ {
			dc, err := DCQuant(uint8(qIndex), 0, bitDepth)
			if err != nil {
				t.Fatal(err)
			}
			ac, err := ACQuant(uint8(qIndex), 0, bitDepth)
			if err != nil {
				t.Fatal(err)
			}
			for _, factor := range []int32{dc, ac} {
				for _, txScale := range []uint8{0, 1, 2} {
					for _, magnitude := range levels {
						for _, negative := range []bool{false, true} {
							level := magnitude
							if negative {
								level = -level
							}
							got := DequantizeMagnitudeWithBoundsTrusted(magnitude, negative, factor, txScale, dqMax)
							want := DequantizeCoefficientWithBoundsTrusted(level, factor, txScale, dqMin, dqMax)
							if got != want {
								t.Fatalf("q=%d bitDepth=%d factor=%d txScale=%d magnitude=%d negative=%t got=%d want=%d", qIndex, bitDepth, factor, txScale, magnitude, negative, got, want)
							}
						}
					}
				}
			}
		}
		// Exhaust the signed-int16 domain for representative DC/AC factors.
		for _, qIndex := range []uint8{0, 1, 32, 128, 255} {
			dc, _ := DCQuant(qIndex, 0, bitDepth)
			ac, _ := ACQuant(qIndex, 0, bitDepth)
			for _, factor := range []int32{dc, ac} {
				for _, txScale := range []uint8{0, 1, 2} {
					for signed := int32(-32768); signed <= 32767; signed++ {
						negative := signed < 0
						magnitude := signed
						if negative {
							magnitude = -magnitude
						}
						got := DequantizeMagnitudeWithBoundsTrusted(magnitude, negative, factor, txScale, dqMax)
						want := DequantizeCoefficientWithBoundsTrusted(signed, factor, txScale, dqMin, dqMax)
						if got != want {
							t.Fatalf("exhaustive q=%d bitDepth=%d factor=%d txScale=%d signed=%d got=%d want=%d", qIndex, bitDepth, factor, txScale, signed, got, want)
						}
					}
				}
			}
		}
	}
}

func testDequantCoefficientFormula(level int32, factor int32, txScale uint8, bitDepth uint8) int32 {
	negative := level < 0
	magnitude := int64(level)
	if negative {
		magnitude = -magnitude
	}
	value := int32((magnitude * int64(factor) & 0xffffff) >> txScale)
	if negative {
		value = -value
	}
	max := int32(1)<<(7+bitDepth) - 1
	min := -int32(1) << (7 + bitDepth)
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
