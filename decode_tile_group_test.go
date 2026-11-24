package goav1

import (
	"path/filepath"
	"testing"
	"time"

	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/obu"
)

// Tile-group handling follows dav1d (src/obu.c): a frame is decoded only when
// its tile groups have carried every tile, and a frame that never completes
// is dropped without an error, an output picture or a reference update, as if
// its OBUs were absent. These tests need no dav1d: each damaged stream must
// decode exactly like the same stream with the dropped frame's OBUs removed.

func tgClip(t *testing.T, dir, name string) [][]byte {
	t.Helper()
	return vdLoadClip(t, name, filepath.Join(dir, name+".ivf"), 0, false).tus
}

func tgFeatureClip(t *testing.T, name string) [][]byte {
	return tgClip(t, featureVectorDir, name)
}

func tgProfileClip(t *testing.T, name string) [][]byte {
	return tgClip(t, filepath.Join("internal", "av1", "testvector", "testdata", "profiles"), name)
}

func tgOBUs(t *testing.T, tu []byte) []vdOBU {
	t.Helper()
	obus, ok := vdSplitOBUs(tu)
	if !ok {
		t.Fatal("unsplittable temporal unit")
	}
	return obus
}

// tgDropOBUs returns tu without the OBUs whose indexes are listed.
func tgDropOBUs(t *testing.T, tu []byte, drop ...int) []byte {
	t.Helper()
	var out []byte
	for i, o := range tgOBUs(t, tu) {
		skip := false
		for _, d := range drop {
			skip = skip || d == i
		}
		if !skip {
			out = append(out, tu[o.start:o.end]...)
		}
	}
	return out
}

// tgDropFrame removes every frame OBU of tu (the temporal delimiter stays).
func tgDropFrame(t *testing.T, tu []byte) []byte {
	t.Helper()
	var drop []int
	for i, o := range tgOBUs(t, tu) {
		switch o.typ {
		case obu.TypeFrameHeader, obu.TypeFrame, obu.TypeTileGroup:
			drop = append(drop, i)
		}
	}
	return tgDropOBUs(t, tu, drop...)
}

// tgPayload returns a copy of OBU i's payload.
func tgPayload(t *testing.T, tu []byte, i int) []byte {
	t.Helper()
	o := tgOBUs(t, tu)[i]
	return append([]byte(nil), tu[o.payOff:o.end]...)
}

func tgSetPayload(t *testing.T, tu []byte, i int, pay []byte) []byte {
	t.Helper()
	return vdRebuild(tu, tgOBUs(t, tu)[i], pay)
}

// tgEvents parses the unmutated clip with goav1's front end and returns the
// events of temporal unit n.
func tgEvents(t *testing.T, tus [][]byte, n int) []internaldecoder.Event {
	t.Helper()
	var s internaldecoder.Stream
	events := make([]internaldecoder.Event, decoderMaxPayloadEvents)
	for i := 0; i <= n; i++ {
		count, err := s.PushLowOverhead(tus[i], events)
		if err != nil {
			t.Fatalf("tu %d: %v", i, err)
		}
		if i == n {
			return events[:count]
		}
	}
	return nil
}

func tgSameDecode(t *testing.T, name string, got, want [][]byte) {
	t.Helper()
	for _, workers := range []int{1, 8} {
		for _, layered := range []bool{false, true} {
			g := vdRunGuarded(got, workers, layered, time.Minute)
			w := vdRunGuarded(want, workers, layered, time.Minute)
			if g.panicked || g.hung || w.panicked || w.hung {
				t.Fatalf("%s workers=%d layered=%v: panic or hang: %s / %s", name, workers, layered, g.err, w.err)
			}
			if g.errTU != w.errTU || !vdSameFrames(g.frames, w.frames) || len(g.frames) != len(w.frames) {
				t.Fatalf("%s workers=%d layered=%v: err@%d f%d [%s], want err@%d f%d [%s]",
					name, workers, layered, g.errTU, len(g.frames), g.err, w.errTU, len(w.frames), w.err)
			}
			// Frame threading delays output by a frame-dependent count, so the
			// temporal unit that returns a picture is compared single-threaded.
			for i := range g.frames {
				if workers == 1 && g.frames[i].tu != w.frames[i].tu {
					t.Fatalf("%s workers=%d layered=%v: frame %d from tu %d, want tu %d", name, workers, layered, i, g.frames[i].tu, w.frames[i].tu)
				}
			}
		}
	}
}

// A frame whose last tile group is lost, or whose last group ends before the
// last tile, is never decoded; the next frame then decodes against the
// references the lost frame would have replaced (dav1d errors on it there,
// not a frame later).
func TestDecodeIncompleteTileGroupFrameIsAbsent(t *testing.T) {
	tus := tgProfileClip(t, "multitile-2x2-tilegroups-256x256")
	absent := vdClone(tus)
	absent[3] = tgDropFrame(t, tus[3])

	lost := vdClone(tus)
	lost[3] = tgDropOBUs(t, tus[3], 3)
	tgSameDecode(t, "lost last tile group", lost, absent)

	// tile_end 3 -> 2 in the second group: tile 3 never arrives.
	short := vdClone(tus)
	pay := tgPayload(t, tus[3], 3)
	pay[0] ^= 1 << 3
	short[3] = tgSetPayload(t, tus[3], 3, pay)
	tgSameDecode(t, "tile group ending early", short, absent)

	got := vdRun(lost, 1, false)
	if got.errTU != 4 || len(got.frames) != 3 {
		t.Fatalf("lost last tile group: err@%d f%d [%s], want err@4 f3 as dav1d", got.errTU, len(got.frames), got.err)
	}
}

// When the last temporal unit's frame misses tile 3, dav1d never decodes the
// frame: no picture and no error, even though its tile 2 data is damaged.
func TestDecodeIncompleteLastFrameIsDroppedSilently(t *testing.T) {
	for _, tc := range []struct {
		clip string
		cut  func(t *testing.T, tu []byte) []byte
	}{
		{"tile-groups-2", func(t *testing.T, tu []byte) []byte {
			// tile_end 3 -> 2 in the group carrying tiles 2-3.
			pay := tgPayload(t, tu, 3)
			pay[0] ^= 1 << 3
			return tgSetPayload(t, tu, 3, pay)
		}},
		{"tile-groups-4", func(t *testing.T, tu []byte) []byte {
			// The group carrying tile 3 is lost and tile 2 is one byte short.
			tu = tgDropOBUs(t, tu, 5)
			pay := tgPayload(t, tu, 4)
			return tgSetPayload(t, tu, 4, pay[:len(pay)-1])
		}},
	} {
		tus := tgFeatureClip(t, tc.clip)
		last := len(tus) - 1
		damaged := vdClone(tus)
		damaged[last] = tc.cut(t, tus[last])
		absent := vdClone(tus)
		absent[last] = tgDropFrame(t, tus[last])
		tgSameDecode(t, tc.clip, damaged, absent)

		got := vdRun(damaged, 1, false)
		if got.errTU >= 0 || len(got.frames) != last {
			t.Fatalf("%s: err@%d f%d [%s], want no error and %d frames", tc.clip, got.errTU, len(got.frames), got.err, last)
		}
		base := vdRun(tus, 1, false)
		if !vdSameFrames(got.frames, base.frames[:last]) {
			t.Fatalf("%s: frames before the dropped frame differ from the clean decode", tc.clip)
		}
	}
}

// dav1d parse_tile_hdr reads tile_start_and_end_present_flag in an OBU_FRAME
// as in an OBU_TILE_GROUP, and dav1d_bytealign_get_bits does not check the
// alignment bits. Streams using either decode like the original.
func TestDecodeTileGroupHeaderAsDav1d(t *testing.T) {
	t.Run("explicit range in OBU_FRAME", func(t *testing.T) {
		tus := tgFeatureClip(t, "tiles-2x2")
		events := tgEvents(t, tus, 0)
		idx := -1
		for i := range events {
			if events[i].Kind == internaldecoder.EventFrame {
				idx = i
			}
		}
		if idx < 0 || events[idx].TileInfo.Cols*events[idx].TileInfo.Rows != 4 {
			t.Fatal("tu 0 has no 2x2 OBU_FRAME")
		}
		pay := tgPayload(t, tus[0], idx)
		at := int(events[idx].TileGroup.DataOffset) - 1
		if pay[at] != 0 {
			t.Fatalf("tile group header byte %#x, want 0", pay[at])
		}
		pay[at] = 0b1_00_11_000 // present, tile_start 0, tile_end 3
		got := vdClone(tus)
		got[0] = tgSetPayload(t, tus[0], idx, pay)
		tgSameDecode(t, "explicit", got, tus)
	})

	t.Run("OBU_FRAME carrying the first tiles only", func(t *testing.T) {
		tus := tgFeatureClip(t, "tiles-2x2")
		events := tgEvents(t, tus, 0)
		idx := -1
		for i := range events {
			if events[i].Kind == internaldecoder.EventFrame {
				idx = i
			}
		}
		e := events[idx]
		pay := tgPayload(t, tus[0], idx)
		at := int(e.TileGroup.DataOffset)
		n := int(e.TileInfo.TileSizeBytes)
		// Tile spans: every tile but the last has a size prefix.
		var start, size [4]int
		off := at
		for i := range 3 {
			v := 0
			for j := range n {
				v |= int(pay[off+j]) << (8 * j)
			}
			start[i], size[i] = off+n, v+1
			off = start[i] + size[i]
		}
		start[3], size[3] = off, len(pay)-off

		head := append([]byte(nil), pay[:at-1]...)
		head = append(head, 0b1_00_01_000) // tiles 0-1
		head = append(head, pay[start[0]-n:start[0]+size[0]]...)
		head = append(head, pay[start[1]:start[1]+size[1]]...)
		tail := []byte{0b1_10_11_000} // tiles 2-3
		tail = append(tail, pay[start[2]-n:start[2]+size[2]]...)
		tail = append(tail, pay[start[3]:start[3]+size[3]]...)

		o := tgOBUs(t, tus[0])[idx]
		tu := append([]byte(nil), tus[0][:o.start]...)
		tu = append(tu, tus[0][o.start:o.sizeOff]...)
		tu = append(tu, vdLEB(len(head))...)
		tu = append(tu, head...)
		tu = append(tu, byte(obu.TypeTileGroup)<<3|2)
		tu = append(tu, vdLEB(len(tail))...)
		tu = append(tu, tail...)
		tu = append(tu, tus[0][o.end:]...)
		got := vdClone(tus)
		got[0] = tu
		tgSameDecode(t, "split", got, tus)
	})

	t.Run("nonzero alignment bits", func(t *testing.T) {
		tus := tgProfileClip(t, "multitile-2x2-tilegroups-256x256")
		got := vdClone(tus)
		for n := range got {
			obus := tgOBUs(t, got[n])
			for i := range obus {
				if obus[i].typ != obu.TypeTileGroup {
					continue
				}
				pay := tgPayload(t, got[n], i)
				pay[0] |= 0b111 // the 3 alignment bits after the 5-bit header
				got[n] = tgSetPayload(t, got[n], i, pay)
			}
		}
		tgSameDecode(t, "alignment", got, tus)
	})
}
