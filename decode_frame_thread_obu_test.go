package goav1

import (
	"crypto/sha256"
	"encoding/binary"
	"os"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/bitstream"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// splitFrameOBUs rewrites an IVF stream so every OBU_FRAME becomes the
// equivalent OBU_FRAME_HEADER + OBU_TILE_GROUP pair (frame_obu() is
// frame_header_obu(), byte_alignment() and tile_group_obu(), AV1 5.10).
func splitFrameOBUs(t *testing.T, ivf []byte) []byte {
	t.Helper()
	if len(ivf) < 32 {
		t.Fatal("short IVF")
	}
	out := append([]byte(nil), ivf[:32]...)
	var stream internaldecoder.Stream
	var event internaldecoder.Event
	for off := 32; off < len(ivf); {
		if off+12 > len(ivf) {
			t.Fatal("short IVF frame header")
		}
		size := int(binary.LittleEndian.Uint32(ivf[off:]))
		frameHeader := ivf[off : off+12]
		tu := ivf[off+12 : off+12+size]
		off += 12 + size

		var units []byte
		for p := 0; p < len(tu); {
			unit, n, err := obu.ParseLowOverhead(tu[p:])
			if err != nil {
				t.Fatal(err)
			}
			p += n
			if err := stream.PushUnitInto(&event, unit, false); err != nil {
				t.Fatal(err)
			}
			if unit.Header.Type != obu.TypeFrame {
				units = append(units, unit.Raw...)
				continue
			}
			bits := int(event.FilmGrain.BitsRead)
			headerBytes := (bits + 7) / 8
			// frame_header_obu() ends in trailing_bits(): a one bit, then
			// zeros to the byte boundary (the zero byte_alignment bits of
			// the frame OBU are already in place).
			fh := append([]byte(nil), unit.Payload[:headerBytes]...)
			if bits%8 == 0 {
				fh = append(fh, 0x80)
			} else {
				fh[bits/8] |= 0x80 >> uint(bits%8)
			}
			units = appendOBU(t, units, unit.Header, obu.TypeFrameHeader, fh)
			units = appendOBU(t, units, unit.Header, obu.TypeTileGroup, unit.Payload[headerBytes:])
		}
		var hdr [12]byte
		copy(hdr[:], frameHeader)
		binary.LittleEndian.PutUint32(hdr[:], uint32(len(units)))
		out = append(out, hdr[:]...)
		out = append(out, units...)
	}
	return out
}

func appendOBU(t *testing.T, dst []byte, h obu.Header, typ obu.Type, payload []byte) []byte {
	t.Helper()
	h.Type = typ
	h.HasSizeField = true
	var hb [2]byte
	n, err := obu.PutHeader(hb[:], h)
	if err != nil {
		t.Fatal(err)
	}
	dst = append(dst, hb[:n]...)
	dst = bitstream.AppendLEB128(dst, uint32(len(payload)))
	return append(dst, payload...)
}

func decodeFrameThreadDigestBytes(t *testing.T, raw []byte, opts ...Option) ([]byte, int) {
	t.Helper()
	dec, err := NewDecoderFromIVF(raw, opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	sum := sha256.New()
	frames := 0
	for {
		got, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatal(err)
		}
		if !ok {
			break
		}
		for _, f := range got {
			frames++
			hashDecodedFrame(sum, f)
		}
	}
	return sum.Sum(nil), frames
}

// TestFrameThreadedFrameHeaderTileGroup: a frame sent as a frame header OBU
// and one tile group OBU holding every tile takes the frame-threaded path
// like an OBU_FRAME (dav1d submits a frame once all its tiles are in), and
// decodes byte-identically to the serial decode.
func TestFrameThreadedFrameHeaderTileGroup(t *testing.T) {
	clips := []string{
		"internal/av1/testvector/testdata/profiles/multitile-2x2-rows-framethread-256x256.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-segmentation-temporal-64x64.ivf",
		"internal/av1/testvector/testdata/profiles/profile0-420-8bit-filmgrain-showexisting-128x128.ivf",
		"testdata/benchcorpus/p360_inter_q32.ivf",
	}
	for _, clip := range clips {
		raw, err := os.ReadFile(clip)
		if err != nil {
			t.Logf("read %s: %v; skipped", clip, err)
			continue
		}
		split := splitFrameOBUs(t, raw)
		want, wantFrames := decodeFrameThreadDigestBytes(t, raw, WithWorkers(1))
		if got, frames := decodeFrameThreadDigestBytes(t, split, WithWorkers(1)); frames != wantFrames || string(got) != string(want) {
			t.Fatalf("%s: split stream decodes differently (%d frames, want %d)", clip, frames, wantFrames)
		}
		// The parse hook serializes submissions behind each entropy pass, so it
		// only checks that the ring is used; the digests run without it.
		threaded := 0
		decoderFramePipelineParseHook = func(*decoderFramePipeline, *decoderFrameContext, []*Frame) func() {
			threaded++
			return nil
		}
		decodeFrameThreadDigestBytes(t, split, WithWorkers(4), WithMaxFrameDelay(3))
		decoderFramePipelineParseHook = nil
		if threaded == 0 {
			t.Fatalf("%s: no frame took the frame-threaded path", clip)
		}
		for _, workers := range []int{2, 4, 8} {
			for _, delay := range []int{0, 2, 4} {
				got, frames := decodeFrameThreadDigestBytes(t, split, WithWorkers(workers), WithMaxFrameDelay(delay))
				if frames != wantFrames || string(got) != string(want) {
					t.Fatalf("%s workers=%d delay=%d: %d frames, digest mismatch=%v (want %d frames)",
						clip, workers, delay, frames, string(got) != string(want), wantFrames)
				}
			}
		}
	}
}
