package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

// TestEOBPosTokenInverts checks the EOB position-token decomposition is exactly
// invertible across the full eob range: the token's group start plus the coded
// offset reconstructs eob, and the offset fits in the token's offset-bit budget
// (the invariant the decoder relies on when it reassembles eob from the token
// plus eob_extra bits).
func TestEOBPosTokenInverts(t *testing.T) {
	const maxEOB = 1024 // largest AV1 transform: 32x32 coded coefficients
	for eob := 1; eob <= maxEOB; eob++ {
		token, extra := eobPosToken(eob)
		if token < 0 || token >= len(eobGroupStart) {
			t.Fatalf("eob=%d: token %d out of range", eob, token)
		}
		if extra < 0 {
			t.Fatalf("eob=%d: negative extra %d", eob, extra)
		}
		if got := eobGroupStart[token] + extra; got != eob {
			t.Fatalf("eob=%d: groupStart[%d]+extra=%d, want %d", eob, token, got, eob)
		}
		if bits := eobOffsetBits[token]; extra >= (1 << bits) {
			t.Fatalf("eob=%d token=%d: extra %d does not fit %d offset bits", eob, token, extra, bits)
		}
	}
}

// TestWriteGolombRoundTrip encodes coefficient-level tails with writeGolomb and
// decodes them with the real decoder range coder (entropy.Reader), replaying the
// exact loop of tile.readCoeffGolombCursor. This proves writeGolomb produces the
// bitstream the decoder reads back.
func TestWriteGolombRoundTrip(t *testing.T) {
	levels := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 100, 255, 1000, 65535, 1 << 17}
	for i := 2; i < 18; i++ {
		levels = append(levels, (1<<i)-1, 1<<i, (1<<i)+1)
	}

	w := newSymbolWriter(make([]byte, 0, 1<<16))
	for _, lv := range levels {
		writeGolomb(&w, lv)
	}
	buf, err := w.finish()
	if err != nil {
		t.Fatalf("finish: %v", err)
	}

	r := entropy.NewReader(buf)
	for idx, want := range levels {
		got, err := decodeGolomb(&r)
		if err != nil {
			t.Fatalf("decodeGolomb[%d]: %v", idx, err)
		}
		if got != want {
			t.Fatalf("golomb[%d] = %d, want %d", idx, got, want)
		}
	}
}

// decodeGolomb mirrors tile.readCoeffGolombCursor using the public entropy.Reader
// API, so the test exercises the same range-coder bit reads the decoder uses.
func decodeGolomb(r *entropy.Reader) (int, error) {
	x := 1
	length := 0
	for {
		bit, err := r.ReadBit()
		if err != nil {
			return 0, err
		}
		length++
		if length > 20 {
			return 0, errCarryUnderflow // reuse a sentinel; never hit on valid input
		}
		if bit != 0 {
			break
		}
	}
	if length > 1 {
		suffix, err := r.ReadBits(uint8(length - 1))
		if err != nil {
			return 0, err
		}
		x = (1 << (length - 1)) | int(suffix)
	}
	return x - 1, nil
}

func TestWriteGolombZeroAlloc(t *testing.T) {
	dst := make([]byte, 0, 4096)
	allocs := testing.AllocsPerRun(100, func() {
		w := newSymbolWriter(dst)
		for lv := range 500 {
			writeGolomb(&w, lv)
		}
		if _, err := w.finish(); err != nil {
			t.Fatalf("finish: %v", err)
		}
	})
	if allocs != 0 {
		t.Fatalf("writeGolomb path allocated %v objects/run, want 0", allocs)
	}
}
