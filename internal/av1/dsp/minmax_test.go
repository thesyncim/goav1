package dsp

import (
	"errors"
	"testing"
)

func TestMinMaxAbsDiff8x8MinValue(t *testing.T) {
	var a [64]byte
	var b [64]byte
	for i := range 64 {
		for j := range a {
			a[j] = 0
			b[j] = 255
		}
		b[i] = byte(i)

		minDiff, maxDiff, err := MinMaxAbsDiff8x8(a[:], 8, b[:], 8, 1)
		if err != nil {
			t.Fatal(err)
		}
		if minDiff != uint16(i) || maxDiff != 255 {
			t.Fatalf("i=%d min=%d max=%d", i, minDiff, maxDiff)
		}
	}
}

func TestMinMaxAbsDiff8x8MaxValue(t *testing.T) {
	var a [64]byte
	var b [64]byte
	for i := range 64 {
		for j := range a {
			a[j] = 0
			b[j] = 0
		}
		b[i] = byte(i)

		minDiff, maxDiff, err := MinMaxAbsDiff8x8(a[:], 8, b[:], 8, 1)
		if err != nil {
			t.Fatal(err)
		}
		if minDiff != 0 || maxDiff != uint16(i) {
			t.Fatalf("i=%d min=%d max=%d", i, minDiff, maxDiff)
		}
	}
}

func TestMinMaxAbsDiff8x8CompareReference(t *testing.T) {
	var a [64]byte
	var b [64]byte
	fillMinMaxBytes(a[:], 0x12345678)
	fillMinMaxBytes(b[:], 0x90abcdef)
	wantMin, wantMax := referenceMinMaxAbsDiff8x8(a[:], 8, b[:], 8, 1)

	minDiff, maxDiff, err := MinMaxAbsDiff8x8(a[:], 8, b[:], 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	if minDiff != wantMin || maxDiff != wantMax {
		t.Fatalf("min/max=%d/%d want %d/%d", minDiff, maxDiff, wantMin, wantMax)
	}
}

func TestMinMaxAbsDiff8x8CompareReferenceAndVaryStride(t *testing.T) {
	var a [8 * 64]byte
	var b [8 * 64]byte
	fillMinMaxBytes(a[:], 0x0badcafe)
	fillMinMaxBytes(b[:], 0xfeedface)
	for aStride := 8; aStride <= 64; aStride += 8 {
		for bStride := 8; bStride <= 64; bStride += 8 {
			wantMin, wantMax := referenceMinMaxAbsDiff8x8(a[:], aStride, b[:], bStride, 1)
			minDiff, maxDiff, err := MinMaxAbsDiff8x8(a[:], aStride, b[:], bStride, 1)
			if err != nil {
				t.Fatal(err)
			}
			if minDiff != wantMin || maxDiff != wantMax {
				t.Fatalf("stride=%d/%d min/max=%d/%d want %d/%d", aStride, bStride, minDiff, maxDiff, wantMin, wantMax)
			}
		}
	}
}

func TestMinMaxAbsDiff8x8HighBitDepth(t *testing.T) {
	var a [64 * 2]byte
	var b [64 * 2]byte
	for i := range 64 {
		for j := range 64 {
			put16(a[:], j, 0)
			put16(b[:], j, 65535)
		}
		put16(b[:], i, uint16(i))

		minDiff, maxDiff, err := MinMaxAbsDiff8x8(a[:], 16, b[:], 16, 2)
		if err != nil {
			t.Fatal(err)
		}
		if minDiff != uint16(i) || maxDiff != 65535 {
			t.Fatalf("i=%d min=%d max=%d", i, minDiff, maxDiff)
		}
	}
}

func TestMinMaxAbsDiff8x8HighBitDepthMaxValue(t *testing.T) {
	var a [64 * 2]byte
	var b [64 * 2]byte
	for i := range 64 {
		for j := range 64 {
			put16(a[:], j, 0)
			put16(b[:], j, 0)
		}
		put16(b[:], i, uint16(i))

		minDiff, maxDiff, err := MinMaxAbsDiff8x8(a[:], 16, b[:], 16, 2)
		if err != nil {
			t.Fatal(err)
		}
		if minDiff != 0 || maxDiff != uint16(i) {
			t.Fatalf("i=%d min=%d max=%d", i, minDiff, maxDiff)
		}
	}
}

func TestMinMaxAbsDiff8x8HighBitDepthCompareReferenceAndVaryStride(t *testing.T) {
	var a [8 * 64 * 2]byte
	var b [8 * 64 * 2]byte
	fillMinMax16(a[:], 0x13579bdf)
	fillMinMax16(b[:], 0x2468ace0)
	for aStrideSamples := 8; aStrideSamples <= 64; aStrideSamples += 8 {
		for bStrideSamples := 8; bStrideSamples <= 64; bStrideSamples += 8 {
			aStride := aStrideSamples * 2
			bStride := bStrideSamples * 2
			wantMin, wantMax := referenceMinMaxAbsDiff8x8(a[:], aStride, b[:], bStride, 2)
			minDiff, maxDiff, err := MinMaxAbsDiff8x8(a[:], aStride, b[:], bStride, 2)
			if err != nil {
				t.Fatal(err)
			}
			if minDiff != wantMin || maxDiff != wantMax {
				t.Fatalf("stride=%d/%d min/max=%d/%d want %d/%d", aStrideSamples, bStrideSamples, minDiff, maxDiff, wantMin, wantMax)
			}
		}
	}
}

func TestMinMaxAbsDiff8x8RejectsInvalid(t *testing.T) {
	var a [64]byte
	var b [64]byte
	for _, tc := range []struct {
		name           string
		aStride        int
		bStride        int
		bytesPerSample int
		a              []byte
		b              []byte
	}{
		{name: "bad sample width", aStride: 8, bStride: 8, bytesPerSample: 3, a: a[:], b: b[:]},
		{name: "short stride", aStride: 7, bStride: 8, bytesPerSample: 1, a: a[:], b: b[:]},
		{name: "short source", aStride: 8, bStride: 8, bytesPerSample: 1, a: a[:63], b: b[:]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := MinMaxAbsDiff8x8(tc.a, tc.aStride, tc.b, tc.bStride, tc.bytesPerSample)
			if !errors.Is(err, ErrInvalidBlock) {
				t.Fatalf("err=%v want %v", err, ErrInvalidBlock)
			}
		})
	}
}

func TestMinMaxAbsDiff8x8Allocs(t *testing.T) {
	var a [8 * 64]byte
	var b [8 * 64]byte
	fillMinMaxBytes(a[:], 0x11111111)
	fillMinMaxBytes(b[:], 0x22222222)
	allocs := testing.AllocsPerRun(1000, func() {
		_, _, err := MinMaxAbsDiff8x8(a[:], 64, b[:], 64, 1)
		if err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("MinMaxAbsDiff8x8 allocated: %f", allocs)
	}
}

func FuzzMinMaxAbsDiff8x8(f *testing.F) {
	f.Add([]byte{0, 1, 2, 3}, []byte{3, 2, 1, 0}, uint8(1))
	f.Add([]byte{0xff, 0x00, 0xaa}, []byte{0x00, 0xff, 0x55}, uint8(2))
	f.Fuzz(func(t *testing.T, aSeed []byte, bSeed []byte, rawBPS uint8) {
		bytesPerSample := int(rawBPS%2) + 1
		stride := 8 * bytesPerSample
		a := make([]byte, stride*8)
		b := make([]byte, stride*8)
		for i := range a {
			if len(aSeed) != 0 {
				a[i] = aSeed[i%len(aSeed)]
			}
			if len(bSeed) != 0 {
				b[i] = bSeed[i%len(bSeed)]
			}
		}
		wantMin, wantMax := referenceMinMaxAbsDiff8x8(a, stride, b, stride, bytesPerSample)
		minDiff, maxDiff, err := MinMaxAbsDiff8x8(a, stride, b, stride, bytesPerSample)
		if err != nil {
			t.Fatalf("MinMaxAbsDiff8x8 err=%v", err)
		}
		if minDiff != wantMin || maxDiff != wantMax {
			t.Fatalf("min/max=%d/%d want %d/%d", minDiff, maxDiff, wantMin, wantMax)
		}
	})
}

func fillMinMaxBytes(dst []byte, seed uint32) {
	x := seed
	for i := range dst {
		x = x*1664525 + 1013904223
		dst[i] = byte(x >> 24)
	}
}

func fillMinMax16(dst []byte, seed uint32) {
	x := seed
	for i := 0; i+1 < len(dst); i += 2 {
		x = x*1664525 + 1013904223
		v := uint16(x >> 16)
		dst[i] = byte(v)
		dst[i+1] = byte(v >> 8)
	}
}

func referenceMinMaxAbsDiff8x8(a []byte, aStride int, b []byte, bStride int, bytesPerSample int) (uint16, uint16) {
	minDiff := uint16(^uint16(0))
	var maxDiff uint16
	for row := range 8 {
		for col := range 8 {
			var diff uint16
			if bytesPerSample == 1 {
				diff = absDiff8(a[row*aStride+col], b[row*bStride+col])
			} else {
				ai := row*aStride + col*2
				bi := row*bStride + col*2
				av := uint16(a[ai]) | uint16(a[ai+1])<<8
				bv := uint16(b[bi]) | uint16(b[bi+1])<<8
				diff = absDiff16(av, bv)
			}
			if diff < minDiff {
				minDiff = diff
			}
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	return minDiff, maxDiff
}

func put16(dst []byte, index int, value uint16) {
	dst[index*2] = byte(value)
	dst[index*2+1] = byte(value >> 8)
}
