package transform

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestTXB8x8PrepTablesMatchDefaultScan(t *testing.T) {
	var scan [64]int16
	var inverse [64]int16
	if err := FillDefaultScan(scan[:], inverse[:], Size{Width: 8, Height: 8}, Class2D); err != nil {
		t.Fatal(err)
	}
	for i, pos := range scan {
		if got := txb8x8Scan2D[i]; got != uint8(pos) {
			t.Fatalf("scan[%d]=%d want %d", i, got, pos)
		}
	}
}

func TestTXBPrep8x8Levels2DMatchesScalar(t *testing.T) {
	cases := make([][64]int16, 0, 38)
	cases = append(cases, [64]int16{})
	var edges [64]int16
	edgeValues := []int16{-32768, -4096, -128, -127, -1, 0, 1, 2, 14, 15, 127, 128, 4096, 32767}
	for i := range edges {
		edges[i] = edgeValues[i%len(edgeValues)]
	}
	cases = append(cases, edges)

	rng := rand.New(rand.NewSource(0x7a8b8c))
	for range 36 {
		var coeffs [64]int16
		for i := range coeffs {
			switch rng.Intn(8) {
			case 0, 1:
				coeffs[i] = 0
			case 2:
				coeffs[i] = int16(rng.Intn(9) - 4)
			case 3:
				coeffs[i] = int16(rng.Intn(260) - 130)
			default:
				coeffs[i] = int16(rng.Intn(8192) - 4096)
			}
		}
		cases = append(cases, coeffs)
	}

	for i := range cases {
		coeffs := cases[i]
		eob := txbPrep8x8TestEOB(&coeffs)
		var gotLevels, wantLevels [256]uint8
		var gotAbs, wantAbs [64]uint16
		got := PrepTXB8x8Levels2D(&coeffs, &gotLevels, &gotAbs, eob)
		want := txbPrep8x8Levels2DPureGo(&coeffs, &wantLevels, &wantAbs, eob)

		if got != want {
			t.Fatalf("case %d result=%+v want %+v", i, got, want)
		}
		if !reflect.DeepEqual(gotLevels, wantLevels) {
			for j := range gotLevels {
				if gotLevels[j] != wantLevels[j] {
					t.Fatalf("case %d levels[%d]=%d want %d", i, j, gotLevels[j], wantLevels[j])
				}
			}
		}
		if !reflect.DeepEqual(gotAbs, wantAbs) {
			t.Fatalf("case %d absLevels mismatch\ngot  %v\nwant %v", i, gotAbs, wantAbs)
		}
	}
}

func TestTXBPrep8x8Levels2DZeroAlloc(t *testing.T) {
	var coeffs [64]int16
	for i := range coeffs {
		coeffs[i] = int16((i*47)%511 - 255)
	}
	eob := txbPrep8x8TestEOB(&coeffs)
	var levels [256]uint8
	var absLevels [64]uint16
	allocs := testing.AllocsPerRun(1000, func() {
		levels = [256]uint8{}
		absLevels = [64]uint16{}
		_ = PrepTXB8x8Levels2D(&coeffs, &levels, &absLevels, eob)
	})
	if allocs != 0 {
		t.Fatalf("TXB 8x8 prep allocated: %f", allocs)
	}
}

func BenchmarkTXBPrep8x8Levels2D(b *testing.B) {
	rng := rand.New(rand.NewSource(0x48b8))
	var blocks [256][64]int16
	var eobs [256]int
	for i := range blocks {
		for c := range blocks[i] {
			switch rng.Intn(6) {
			case 0:
				blocks[i][c] = 0
			case 1:
				blocks[i][c] = int16(rng.Intn(7) - 3)
			default:
				blocks[i][c] = int16(rng.Intn(1024) - 512)
			}
		}
		eobs[i] = txbPrep8x8TestEOB(&blocks[i])
	}

	bench := func(b *testing.B, fn func(*[64]int16, *[256]uint8, *[64]uint16, int) TXB8x8PrepResult) {
		var levels [256]uint8
		var absLevels [64]uint16
		var sink TXB8x8PrepResult
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := i & 255
			levels = [256]uint8{}
			sink = fn(&blocks[idx], &levels, &absLevels, eobs[idx])
		}
		if sink.CulLevel == 255 {
			b.Fatal(sink)
		}
	}

	b.Run("impl", func(b *testing.B) {
		bench(b, PrepTXB8x8Levels2D)
	})
	b.Run("scalar", func(b *testing.B) {
		bench(b, txbPrep8x8Levels2DPureGo)
	})
}

func txbPrep8x8TestEOB(coeffs *[64]int16) int {
	for c := len(txb8x8Scan2D) - 1; c >= 0; c-- {
		if coeffs[txb8x8Scan2D[c]] != 0 {
			return c + 1
		}
	}
	return 0
}
