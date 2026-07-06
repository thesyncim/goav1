// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant and NOTICE for the AOM attribution.

package entropy

import (
	"math/rand"
	"testing"
)

// randomAdversarialCDF4 builds a strictly decreasing four-symbol inverse CDF
// with a random adaptation count, biased toward extreme states (near-saturated
// heads and near-zero tails) so the high-token chain and its adaptation rates
// are exercised across the whole reachable range.
func randomAdversarialCDF4(rng *rand.Rand) CDF {
	var row [MaxSymbols + 1]uint16
	switch rng.Intn(4) {
	case 0:
		// near-saturated: symbol 3 chains fire almost every read
		row[0] = uint16(32700 + rng.Intn(67))
		row[1] = row[0] - uint16(1+rng.Intn(16))
		row[2] = row[1] - uint16(1+rng.Intn(16))
	case 1:
		// near-zero probabilities: symbol 0 dominates, minimum splits
		row[0] = uint16(3 + rng.Intn(48))
		row[1] = row[0] * 2 / 3
		row[2] = row[0] / 3
		if row[1] <= row[2] {
			row[1] = row[2] + 1
		}
		if row[0] <= row[1] {
			row[0] = row[1] + 1
		}
	default:
		// general random strictly-decreasing row
		a := 1 + rng.Intn(32766)
		b := 1 + rng.Intn(32766)
		c := 1 + rng.Intn(32766)
		if a < b {
			a, b = b, a
		}
		if b < c {
			b, c = c, b
		}
		if a < b {
			a, b = b, a
		}
		if b == a {
			b = a - 1
		}
		if c >= b {
			c = b - 1
		}
		if c < 1 {
			c = 1
		}
		row[0], row[1], row[2] = uint16(a), uint16(b), uint16(c)
	}
	row[3] = 0
	row[4] = uint16(rng.Intn(int(MaxCDFCount) + 1))
	return CDF{values: row, symbols: 4}
}

// readerStatesEqual compares every range-decoder state field (src is shared
// by construction in these tests).
func readerStatesEqual(a, b *Reader) bool {
	return a.pos == b.pos && a.dif == b.dif && a.rng == b.rng &&
		a.cnt == b.cnt && a.tellOffs == b.tellOffs
}

// TestCursorArchBitKernelMatchesPureGo differentially tests the arch
// ReadBitTrusted entry (arm64 asm when built in) against the pure-Go Reader
// core over long random streams, comparing the decoded bit and the full
// range-decoder state after every read, including reads past the end of the
// buffer where the ecLotsBits tell accounting takes over.
func TestCursorArchBitKernelMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xd1ab17))
	lengths := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 64, 100, 4096}
	for _, n := range lengths {
		for trial := 0; trial < 24; trial++ {
			src := make([]byte, n)
			rng.Read(src)
			ref := NewReader(src)
			archReader := NewReader(src)
			cur := archReader.Cursor()
			reads := 8*n + 256
			if reads > 4096 {
				reads = 4096
			}
			for i := 0; i < reads; i++ {
				want := ref.ReadBitTrusted()
				got := cur.ReadBitTrusted()
				if got != want {
					t.Fatalf("len=%d trial=%d read=%d bit=%d want %d", n, trial, i, got, want)
				}
				cur.CommitTo(&archReader)
				if !readerStatesEqual(&archReader, &ref) {
					t.Fatalf("len=%d trial=%d read=%d state: got pos=%d dif=%016x rng=%04x cnt=%d tell=%d want pos=%d dif=%016x rng=%04x cnt=%d tell=%d",
						n, trial, i,
						archReader.pos, archReader.dif, archReader.rng, archReader.cnt, archReader.tellOffs,
						ref.pos, ref.dif, ref.rng, ref.cnt, ref.tellOffs)
				}
			}
		}
	}
}

// TestCursorArchHighTokenKernelMatchesPureGo differentially tests the arch
// high-token chain (arm64 asm when built in) against the pure-Go update loop
// over long random streams and adversarial CDF rows, comparing the level, the
// adapted CDF row, and the full range-decoder state after every chain.
func TestCursorArchHighTokenKernelMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(0xd1a41))
	lengths := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 64, 100, 4096}
	for _, n := range lengths {
		for trial := 0; trial < 24; trial++ {
			src := make([]byte, n)
			rng.Read(src)
			refReader := NewReader(src)
			refCur := refReader.Cursor()
			archReader := NewReader(src)
			archCur := archReader.Cursor()
			chains := 2*n + 64
			if chains > 1024 {
				chains = 1024
			}
			for i := 0; i < chains; i++ {
				cdf := randomAdversarialCDF4(rng)
				refCDF := cdf
				archCDF := cdf
				want := refCur.readCDF4HighTokenUpdateLoop(&refCDF.values)
				got := archCur.ReadCDF4HighTokenUpdateUnchecked(&archCDF)
				if got != want {
					t.Fatalf("len=%d trial=%d chain=%d level=%d want %d (cdf=%v)", n, trial, i, got, want, cdf.values)
				}
				if archCDF != refCDF {
					t.Fatalf("len=%d trial=%d chain=%d cdf: got %v want %v (start %v)", n, trial, i, archCDF.values, refCDF.values, cdf.values)
				}
				refCur.CommitTo(&refReader)
				archCur.CommitTo(&archReader)
				if !readerStatesEqual(&archReader, &refReader) {
					t.Fatalf("len=%d trial=%d chain=%d state: got pos=%d dif=%016x rng=%04x cnt=%d tell=%d want pos=%d dif=%016x rng=%04x cnt=%d tell=%d",
						n, trial, i,
						archReader.pos, archReader.dif, archReader.rng, archReader.cnt, archReader.tellOffs,
						refReader.pos, refReader.dif, refReader.rng, refReader.cnt, refReader.tellOffs)
				}
			}
		}
	}
}
