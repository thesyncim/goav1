package goav1

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internaltile "github.com/thesyncim/goav1/internal/av1/tile"
)

// truncateLastTileOBU returns tu with its last frame or tile-group OBU cut to
// half its payload, its size field rewritten so the OBU framing stays valid.
// The tile data then ends mid-symbol, which libaom and dav1d reject as a
// symbol-decoder overread.
func truncateLastTileOBU(t *testing.T, tu []byte) []byte {
	t.Helper()
	var units []internalobu.Unit
	it := internalobu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse OBU: %v", err)
		}
		if !ok {
			break
		}
		units = append(units, unit)
	}
	last := -1
	for i, unit := range units {
		if unit.Header.Type == internalobu.TypeFrame || unit.Header.Type == internalobu.TypeTileGroup {
			last = i
		}
	}
	if last < 0 {
		t.Fatal("temporal unit has no frame or tile-group OBU")
	}
	var out []byte
	for i, unit := range units {
		if i != last {
			out = append(out, unit.Raw...)
			continue
		}
		header := unit.Header
		header.HasSizeField = true
		var headerBytes [2]byte
		n, err := internalobu.PutHeader(headerBytes[:], header)
		if err != nil {
			t.Fatalf("put OBU header: %v", err)
		}
		payload := unit.Payload[:len(unit.Payload)/2]
		out = append(out, headerBytes[:n]...)
		out = appendPublicLEB128(out, uint32(len(payload)))
		out = append(out, payload...)
	}
	return out
}

// TestDecoderRejectsTileDataOverread damages one temporal unit's tile data
// and checks the decoders fail it with ErrTileDataOverread (dav1d's
// msac.cnt <= -15 check in dav1d_decode_tile_sbrow) instead of decoding
// padding into the frame and every frame that references it.
func TestDecoderRejectsTileDataOverread(t *testing.T) {
	ivf, err := os.ReadFile(filepath.Join(featureVectorDir, "rt-cbr.ivf"))
	if err != nil {
		t.Fatalf("read clip: %v", err)
	}
	payloads := featureVectorPayloads(t, ivf)
	const damaged = 3
	if len(payloads) <= damaged+1 {
		t.Fatalf("clip has %d temporal units", len(payloads))
	}
	payloads[damaged] = truncateLastTileOBU(t, payloads[damaged])

	newDecoders := []struct {
		name string
		open func(workers int) (frameSource, error)
	}{
		{"Decoder", func(workers int) (frameSource, error) { return NewDecoder(payloads, WithWorkers(workers)) }},
		{"LayeredDecoder", func(workers int) (frameSource, error) { return NewLayeredDecoder(payloads, WithWorkers(workers)) }},
	}
	for _, nd := range newDecoders {
		for _, workers := range []int{1, 4} {
			t.Run(fmt.Sprintf("%s/workers=%d", nd.name, workers), func(t *testing.T) {
				dec, err := nd.open(workers)
				if err != nil {
					t.Fatalf("open: %v", err)
				}
				defer dec.Close()
				// Frame threading may surface the error up to the frame delay
				// (at most the worker count) calls after the damaged unit.
				outputs := 0
				for call := 0; call <= damaged+workers; call++ {
					frames, ok, err := dec.DecodeNext()
					if err != nil {
						if !errors.Is(err, internaltile.ErrTileDataOverread) {
							t.Fatalf("DecodeNext call %d: err=%v want %v", call, err, internaltile.ErrTileDataOverread)
						}
						if outputs > damaged {
							t.Fatalf("output %d frames before the error, want at most %d", outputs, damaged)
						}
						return
					}
					if !ok {
						t.Fatalf("stream ended after %d calls without an error", call+1)
					}
					outputs += len(frames)
				}
				t.Fatalf("no error within %d DecodeNext calls (%d frames output)", damaged+workers+1, outputs)
			})
		}
	}
}
