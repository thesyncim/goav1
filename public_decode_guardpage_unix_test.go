// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build unix

package goav1_test

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"sort"
	"testing"

	av1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/guardpage"
)

// guardVector is one committed IVF clip split into temporal-unit payloads.
type guardVector struct {
	name     string
	ivf      []byte
	payloads [][]byte
}

func loadGuardVectors(t *testing.T) []guardVector {
	t.Helper()
	files, err := filepath.Glob(profileClipPath("*.ivf"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no profile clips: %v", err)
	}
	var out []guardVector
	for _, f := range files {
		ivf, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		it, err := av1.NewIVFIterator(ivf)
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		v := guardVector{name: filepath.Base(f), ivf: ivf}
		for {
			frame, ok, err := it.Next()
			if err != nil {
				t.Fatalf("%s: %v", f, err)
			}
			if !ok {
				break
			}
			v.payloads = append(v.payloads, slices.Clone(frame.Payload))
		}
		out = append(out, v)
	}
	sort.Slice(out, func(i, j int) bool { return len(out[i].ivf) < len(out[j].ivf) })
	return out
}

// guarded copies b into a mapping whose last byte is followed by (or, with
// before, whose first byte follows) an inaccessible page.
func guarded(t *testing.T, b []byte, before bool) []byte {
	if len(b) == 0 {
		return b
	}
	var g []byte
	if before {
		g = guardpage.Before[byte](t, len(b))
	} else {
		g = guardpage.After[byte](t, len(b))
	}
	copy(g, b)
	return g
}

// decodeDigests decodes payloads through the public API and returns the
// visible-plane digest of every output frame. A panic (including a fault on
// a guard page, which debug.SetPanicOnFault turns into one on the calling
// goroutine) fails the test with name.
func decodeDigests(t *testing.T, name string, payloads [][]byte, workers int, layered bool) (digests [][16]byte, err error) {
	t.Helper()
	defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("%s: decoder panicked: %v", name, r)
		}
	}()
	opts := []av1.Option{av1.WithWorkers(workers)}
	if layered {
		d, err := av1.NewLayeredDecoder(payloads, opts...)
		if err != nil {
			return nil, err
		}
		defer d.Close()
		for {
			frames, ok, err := d.DecodeNextWithMetadata()
			for _, f := range frames {
				digests = append(digests, frameMD5Visible(f.Frame))
			}
			if err != nil || !ok {
				return digests, err
			}
		}
	}
	d, err := av1.NewDecoder(payloads, opts...)
	if err != nil {
		return nil, err
	}
	defer d.Close()
	for {
		frames, ok, err := d.DecodeNext()
		for _, f := range frames {
			digests = append(digests, frameMD5Visible(f))
		}
		if err != nil || !ok {
			return digests, err
		}
	}
}

// TestPublicDecodeGuardedPayloads decodes every committed clip with each
// temporal unit placed against a guard page on either side, through
// NewDecoder, NewLayeredDecoder, NewDecoderFromIVF and the RTP payload path,
// at one and eight workers, and checks the output matches a heap decode.
func TestPublicDecodeGuardedPayloads(t *testing.T) {
	for _, v := range loadGuardVectors(t) {
		var want [2][][16]byte
		var wantErr [2]error
		for i, layered := range []bool{false, true} {
			want[i], wantErr[i] = decodeDigests(t, v.name, v.payloads, 1, layered)
		}
		if wantErr[0] != nil {
			t.Fatalf("%s: heap decode: %v", v.name, wantErr[0])
		}
		for _, before := range []bool{false, true} {
			g := make([][]byte, len(v.payloads))
			for i, p := range v.payloads {
				g[i] = guarded(t, p, before)
			}
			for _, workers := range []int{1, 8} {
				for i, layered := range []bool{false, true} {
					name := fmt.Sprintf("%s before=%v workers=%d layered=%v", v.name, before, workers, layered)
					got, err := decodeDigests(t, name, g, workers, layered)
					if (err == nil) != (wantErr[i] == nil) || !slices.Equal(got, want[i]) {
						t.Fatalf("%s: guarded decode differs (err %v, %d frames, want %d)", name, err, len(got), len(want[i]))
					}
				}
			}
			ivf := guarded(t, v.ivf, before)
			func() {
				defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
				d, err := av1.NewDecoderFromIVF(ivf, av1.WithWorkers(8))
				if err != nil {
					t.Fatalf("%s: NewDecoderFromIVF: %v", v.name, err)
				}
				defer d.Close()
				frames, err := d.DecodeAll()
				if err != nil || len(frames) != len(want[0]) {
					t.Fatalf("%s before=%v: guarded IVF decode: %v (%d frames)", v.name, before, err, len(frames))
				}
			}()
		}
		var seq uint16
		var rtp [][]byte
		for i, p := range v.payloads {
			payloads, _ := publicPacketizeLowOverheadRTP(t, p, av1.RTPPayloadSizeLimits{MaxPayloadLen: 200}, i == 0, &seq, uint32(i))
			for _, r := range payloads {
				rtp = append(rtp, guarded(t, r, false))
			}
		}
		func() {
			defer debug.SetPanicOnFault(debug.SetPanicOnFault(true))
			d, err := av1.NewDecoderFromRTPPayloads(rtp, av1.WithWorkers(1))
			if err != nil {
				t.Fatalf("%s: NewDecoderFromRTPPayloads: %v", v.name, err)
			}
			defer d.Close()
			if _, err := d.DecodeAll(); err != nil {
				t.Fatalf("%s: guarded RTP decode: %v", v.name, err)
			}
		}()
	}
}

// TestPublicDecodeTruncatedAndCorruptPayloads feeds each clip with one
// temporal unit cut short or bit-flipped, ending at a guard page. Decoding
// may fail, but must neither read past the payload nor panic. Every cut
// offset of every temporal unit is tried for the two smallest clips; the
// others get 32 random cuts and 32 random bit flips each, at one worker and
// (for every eighth case) at eight.
func TestPublicDecodeTruncatedAndCorruptPayloads(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	vectors := loadGuardVectors(t)
	for vi, v := range vectors {
		// One guard mapping per temporal unit: a cut of length c is the
		// unit's first c bytes copied to the tail, so it still ends at the
		// guard page.
		tails := make([][]byte, len(v.payloads))
		for i, p := range v.payloads {
			tails[i] = guardpage.After[byte](t, max(len(p), 1))
		}
		run := func(k, c int, flip int, workers int) {
			p := v.payloads[k]
			buf := tails[k][len(tails[k])-c:]
			copy(buf, p[:c])
			if flip >= 0 {
				buf[flip/8] ^= 1 << (flip % 8)
			}
			payloads := append(slices.Clone(v.payloads[:k]), buf)
			if flip >= 0 {
				payloads = append(payloads, v.payloads[k+1:]...)
			}
			decodeDigests(t, fmt.Sprintf("%s unit %d cut %d flip %d workers %d", v.name, k, c, flip, workers), payloads, workers, false)
		}
		n := 0
		if vi < 2 {
			for k, p := range v.payloads {
				for c := range len(p) {
					run(k, c, -1, 1+7*(n%8/7))
					n++
				}
			}
			continue
		}
		for range 32 {
			k := rng.Intn(len(v.payloads))
			if len(v.payloads[k]) == 0 {
				continue
			}
			run(k, rng.Intn(len(v.payloads[k])), -1, 1+7*(n%8/7))
			n++
		}
		for range 32 {
			k := rng.Intn(len(v.payloads))
			if len(v.payloads[k]) == 0 {
				continue
			}
			run(k, len(v.payloads[k]), rng.Intn(8*len(v.payloads[k])), 1+7*(n%8/7))
			n++
		}
	}
}
