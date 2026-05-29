package transform

import "testing"

func BenchmarkSizeFromDimensions(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := SizeFromDimensions(32, 32); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkScanSize(b *testing.B) {
	size := Size{Width: 64, Height: 64}
	for i := 0; i < b.N; i++ {
		if _, err := ScanSize(size); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSizeShift(b *testing.B) {
	size := Size{Width: 32, Height: 32}
	for i := 0; i < b.N; i++ {
		if _, err := size.Shift(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDefaultScanMode(b *testing.B) {
	size := Size{Width: 16, Height: 32}
	for i := 0; i < b.N; i++ {
		if _, err := DefaultScanMode(size, Class2D); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFillDefaultScan32x32(b *testing.B) {
	var scan [32 * 32]int16
	var inverse [32 * 32]int16
	size := Size{Width: 32, Height: 32}
	for i := 0; i < b.N; i++ {
		if err := FillDefaultScan(scan[:], inverse[:], size, Class2D); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFillDefaultScan4x4(b *testing.B) {
	var scan [16]int16
	var inverse [16]int16
	size := Size{Width: 4, Height: 4}
	for i := 0; i < b.N; i++ {
		if err := FillDefaultScan(scan[:], inverse[:], size, Class2D); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMaxEOB(b *testing.B) {
	size := Size{Width: 64, Height: 64}
	for i := 0; i < b.N; i++ {
		if _, err := MaxEOB(size); err != nil {
			b.Fatal(err)
		}
	}
}
