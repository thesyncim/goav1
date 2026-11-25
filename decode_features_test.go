package goav1

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"hash"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// featureVectorDir holds one small clip per coding tool or header mode,
// encoded by aomenc and SvtAv1EncApp, with dav1d's output MD5 for each in
// manifest.txt. scripts/decoder_feature_sweep.py -x regenerates the set; the
// sweep found the quantizer-matrix, intra-block-copy CDF and layered
// temporal-MV bugs that the conformance vectors missed.
var featureVectorDir = filepath.Join("internal", "av1", "testvector", "testdata", "features")

type featureVector struct {
	name    string
	md5Hex  string
	layered bool
}

func loadFeatureVectors(t *testing.T) []featureVector {
	t.Helper()
	f, err := os.Open(filepath.Join(featureVectorDir, "manifest.txt"))
	if err != nil {
		t.Fatalf("open manifest: %v", err)
	}
	defer f.Close()
	var vectors []featureVector
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) != 3 || len(fields[1]) != 32 || (fields[2] != "0" && fields[2] != "1") {
			t.Fatalf("bad manifest line %q", sc.Text())
		}
		vectors = append(vectors, featureVector{name: fields[0], md5Hex: fields[1], layered: fields[2] == "1"})
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if len(vectors) == 0 {
		t.Fatal("empty feature-vector manifest")
	}
	return vectors
}

// hashFrameVisible appends a frame's visible planes to h in the layout of
// dav1d's yuv muxer (and aom-go-dec): Y, U, V rows, little-endian 16-bit
// samples above 8 bits, no chroma for monochrome.
func hashFrameVisible(h hash.Hash, f *Frame) {
	bps := f.Layout.BytesPerSample
	for _, plane := range []FramePlane{f.Y, f.U, f.V} {
		if plane.Width == 0 || plane.Height == 0 || len(plane.Pix) == 0 {
			continue
		}
		rowBytes := plane.Width * bps
		for row := 0; row < plane.Height; row++ {
			off := row * plane.Stride
			h.Write(plane.Pix[off : off+rowBytes])
		}
	}
}

type frameSource interface {
	DecodeNext() ([]*Frame, bool, error)
	Close()
}

func decodeStreamMD5(t *testing.T, dec frameSource) string {
	t.Helper()
	defer dec.Close()
	h := md5.New()
	n := 0
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext after %d frames: %v", n, err)
		}
		if !ok {
			break
		}
		for _, f := range frames {
			hashFrameVisible(h, f)
			n++
		}
	}
	if n == 0 {
		t.Fatal("decoded no frames")
	}
	return hex.EncodeToString(h.Sum(nil))
}

func TestFeatureVectorsMatchDav1d(t *testing.T) {
	for _, v := range loadFeatureVectors(t) {
		t.Run(v.name, func(t *testing.T) {
			t.Parallel()
			ivf, err := os.ReadFile(filepath.Join(featureVectorDir, v.name+".ivf"))
			if err != nil {
				t.Fatalf("read clip: %v", err)
			}
			// One worker, frame threading at eight, and the layered decoder
			// on the 4:2:0 clips it accepts.
			for _, workers := range []int{1, 8} {
				dec, err := NewDecoderFromIVF(ivf, WithWorkers(workers))
				if err != nil {
					t.Fatalf("workers=%d NewDecoderFromIVF: %v", workers, err)
				}
				if got := decodeStreamMD5(t, dec); got != v.md5Hex {
					t.Errorf("workers=%d md5 %s, dav1d %s", workers, got, v.md5Hex)
				}
			}
			if !v.layered {
				return
			}
			payloads := featureVectorPayloads(t, ivf)
			dec, err := NewLayeredDecoder(payloads, WithWorkers(4))
			if err != nil {
				t.Fatalf("NewLayeredDecoder: %v", err)
			}
			if got := decodeStreamMD5(t, dec); got != v.md5Hex {
				t.Errorf("layered md5 %s, dav1d %s", got, v.md5Hex)
			}
		})
	}
}

func featureVectorPayloads(t *testing.T, ivf []byte) [][]byte {
	t.Helper()
	it, err := NewIVFIterator(ivf)
	if err != nil {
		t.Fatalf("NewIVFIterator: %v", err)
	}
	var payloads [][]byte
	for {
		frame, ok, err := it.Next()
		if err != nil {
			t.Fatalf("ivf next: %v", err)
		}
		if !ok {
			return payloads
		}
		payloads = append(payloads, frame.Payload)
	}
}

// TestFeatureVectorManifestMatchesClips keeps the manifest and the clips in
// step: every listed clip exists and every clip is listed.
func TestFeatureVectorManifestMatchesClips(t *testing.T) {
	listed := map[string]bool{}
	for _, v := range loadFeatureVectors(t) {
		if listed[v.name] {
			t.Fatalf("%s listed twice", v.name)
		}
		listed[v.name] = true
	}
	clips, err := filepath.Glob(filepath.Join(featureVectorDir, "*.ivf"))
	if err != nil {
		t.Fatal(err)
	}
	for _, clip := range clips {
		name := strings.TrimSuffix(filepath.Base(clip), ".ivf")
		if !listed[name] {
			t.Errorf("%s is not in manifest.txt", name)
		}
		delete(listed, name)
	}
	for name := range listed {
		t.Errorf("manifest lists %s but %s.ivf is missing", name, name)
	}
	if t.Failed() {
		t.Log("regenerate with scripts/decoder_feature_sweep.py -x", featureVectorDir)
	}
}

// A new coded video sequence may change the bit depth, the chroma format and
// the superblock size. dav1d decodes such a stream as the concatenation of its
// sequences: obu.c drops every reference at the new sequence header and the
// picture pool hands out a buffer per picture whatever its format. The cases
// below concatenate clips; each sequence decodes on its own (dav1d's output for
// each concatenation is byte-identical to its clips' outputs one after the
// other), so the expected frames are those of the clips decoded alone, whose
// whole-stream MD5s are dav1d's, and the whole concatenation's MD5 is dav1d
// 1.5.4's (`dav1d --filmgrain 1 --muxer md5`) on the concatenated IVF.
var formatChangeCases = []struct {
	name     string
	clips    []string
	dav1dMD5 string
}{
	{"444-to-420-10bit", []string{"444", "10bit"}, "fba127f141fe118ec0aa48e779e706ef"},
	{"sb64-to-sb128", []string{"allintra", "sb128"}, "edc29697bd9c185cb70f6c022ca3580a"},
	{"10bit-to-8bit", []string{"10bit", "lowdelay"}, "a75dcfc5d09a1c0de2c07b43e4a4e2dc"},
	{"420-444-420", []string{"lowdelay", "444", "allintra"}, "7095b982ea1efb78ffb2230f7163f080"},
	{"8bit-640x360-to-10bit-352x288", []string{"size-640x360", "10bit"}, "415fda8dba453bad377afa8c8de929d0"},
	// The same format, but a sequence header that differs: the reference
	// reset alone, with frames of the old sequence still in flight.
	{"same-format-new-sequence", []string{"lowdelay", "cdef-off"}, "32a109f9e7050e021ddc2d83ceff2b87"},
	// Film-grain display pictures change format with their source.
	{"film-grain-10bit-444-8bit", []string{"film-grain-10bit", "film-grain-444", "film-grain-8"}, "7d4c9b0fab401885b1e329e38b33460c"},
	// Super-resolution output pictures change format too.
	{"superres-444-10bit-to-420-8bit", []string{formatChangeSuperRes444, formatChangeSuperRes420}, "c5bf72fb586c6b0884fdd0643b5d37a8"},
}

const (
	formatChangeSuperRes444 = "../profiles/profile1-444-10bit-superres-160x128"
	formatChangeSuperRes420 = "../profiles/profile0-420-8bit-superres-resize-random-192x108"
)

// formatChangeClips returns the payloads of each clip and the per-frame
// digests of each clip decoded alone, after checking that clip's whole-stream
// MD5 against dav1d's.
func formatChangeClips(t *testing.T, names []string) (clips [][][]byte, want []string) {
	t.Helper()
	// dav1d 1.5.4's whole-stream MD5s: the feature clips' from manifest.txt,
	// the profile clips' measured with `dav1d --muxer md5`.
	manifest := map[string]string{
		formatChangeSuperRes444: "8539ebd666196c6f57f6ce526e1ae4a9",
		formatChangeSuperRes420: "fbaeff6240dade0d60ffd8a79295b99b",
	}
	for _, v := range loadFeatureVectors(t) {
		manifest[v.name] = v.md5Hex
	}
	for _, name := range names {
		ivf, err := os.ReadFile(filepath.Join(featureVectorDir, name+".ivf"))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		payloads := featureVectorPayloads(t, ivf)
		dec, err := NewDecoder(payloads)
		if err != nil {
			t.Fatalf("%s: NewDecoder: %v", name, err)
		}
		frames, whole := formatChangeDigests(t, dec)
		if whole != manifest[name] {
			t.Fatalf("%s decoded alone: md5 %s, dav1d %s", name, whole, manifest[name])
		}
		clips = append(clips, payloads)
		want = append(want, frames...)
	}
	return clips, want
}

// formatChangeDigests decodes every frame of dec and returns each frame's MD5
// and the whole stream's.
func formatChangeDigests(t *testing.T, dec frameSource) ([]string, string) {
	t.Helper()
	defer dec.Close()
	whole := md5.New()
	var frames []string
	for {
		got, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext after %d frames: %v", len(frames), err)
		}
		if !ok {
			break
		}
		for _, f := range got {
			h := md5.New()
			hashFrameVisible(h, f)
			hashFrameVisible(whole, f)
			frames = append(frames, hex.EncodeToString(h.Sum(nil)))
		}
	}
	return frames, hex.EncodeToString(whole.Sum(nil))
}

func formatChangeCheck(t *testing.T, label string, dec frameSource, want []string, dav1dMD5 string) {
	t.Helper()
	got, whole := formatChangeDigests(t, dec)
	if len(got) != len(want) {
		t.Fatalf("%s: %d frames, want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: frame %d md5 %s, want %s", label, i, got[i], want[i])
		}
	}
	if whole != dav1dMD5 {
		t.Fatalf("%s: stream md5 %s, dav1d %s", label, whole, dav1dMD5)
	}
}

// formatChangeIVF writes the concatenated clips as one IVF stream.
func formatChangeIVF(tus [][]byte) []byte {
	var b bytes.Buffer
	hdr := make([]byte, IVFFileHeaderSize)
	copy(hdr, "DKIF")
	binary.LittleEndian.PutUint16(hdr[6:], IVFFileHeaderSize)
	copy(hdr[8:], "AV01")
	binary.LittleEndian.PutUint16(hdr[12:], 352)
	binary.LittleEndian.PutUint16(hdr[14:], 288)
	binary.LittleEndian.PutUint32(hdr[16:], 30)
	binary.LittleEndian.PutUint32(hdr[20:], 1)
	binary.LittleEndian.PutUint32(hdr[24:], uint32(len(tus)))
	b.Write(hdr)
	for i, tu := range tus {
		var fh [IVFFrameHeaderSize]byte
		binary.LittleEndian.PutUint32(fh[:4], uint32(len(tu)))
		binary.LittleEndian.PutUint64(fh[4:], uint64(i))
		b.Write(fh[:])
		b.Write(tu)
	}
	return b.Bytes()
}

func TestDecoderFormatChangeMatchesDav1d(t *testing.T) {
	for _, tc := range formatChangeCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			clips, want := formatChangeClips(t, tc.clips)
			var tus [][]byte
			for _, c := range clips {
				tus = append(tus, c...)
			}
			if dir := os.Getenv("GOAV1_FORMAT_CHANGE_DUMP"); dir != "" {
				if err := os.WriteFile(filepath.Join(dir, tc.name+".ivf"), formatChangeIVF(tus), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			// One worker, and frame threading at eight: the frames of the old
			// sequence still in flight drain before the new one starts.
			for _, workers := range []int{1, 8} {
				dec, err := NewDecoder(tus, WithWorkers(workers))
				if err != nil {
					t.Fatalf("workers=%d NewDecoder: %v", workers, err)
				}
				formatChangeCheck(t, "NewDecoder workers="+strconv.Itoa(workers), dec, want, tc.dav1dMD5)
			}
			ivf := formatChangeIVF(tus)
			dec, _, err := NewDecoderFromIVFReaderAt(bytes.NewReader(ivf), int64(len(ivf)), WithWorkers(8))
			if err != nil {
				t.Fatalf("NewDecoderFromIVFReaderAt: %v", err)
			}
			formatChangeCheck(t, "NewDecoderFromIVFReaderAt", dec, want, tc.dav1dMD5)

			// RTP, with the N bit on the first packet of each sequence.
			var payloads [][]byte
			seq := uint16(0x1200)
			for _, c := range clips {
				for i, tu := range c {
					p, _ := publicPacketizeLowOverheadRTP(t, tu, RTPPayloadSizeLimits{MaxPayloadLen: 1200}, i == 0, &seq, uint32(i))
					payloads = append(payloads, p...)
				}
			}
			for _, workers := range []int{1, 8} {
				dec, err := NewDecoderFromRTPPayloads(payloads, WithWorkers(workers))
				if err != nil {
					t.Fatalf("workers=%d NewDecoderFromRTPPayloads: %v", workers, err)
				}
				formatChangeCheck(t, "NewDecoderFromRTPPayloads workers="+strconv.Itoa(workers), dec, want, tc.dav1dMD5)
			}
		})
	}
}

// TestDecoderFormatChangeFrameSizeLimit checks that WithFrameSizeLimit covers
// the frames of every sequence, not only the first one's.
func TestDecoderFormatChangeFrameSizeLimit(t *testing.T) {
	clips, _ := formatChangeClips(t, []string{"10bit", "size-640x360"})
	tus := append(append([][]byte{}, clips[0]...), clips[1]...)
	limit := WithFrameSizeLimit(352 * 288)
	if _, err := NewDecoder(tus, limit); !errors.Is(err, ErrDecoderFrameSizeLimit) {
		t.Fatalf("NewDecoder err=%v, want %v", err, ErrDecoderFrameSizeLimit)
	}
	if _, err := NewDecoder(tus, limit, WithWorkers(8)); !errors.Is(err, ErrDecoderFrameSizeLimit) {
		t.Fatalf("NewDecoder workers=8 err=%v, want %v", err, ErrDecoderFrameSizeLimit)
	}
	if _, err := NewDecoderFromRTPPayloads(tus, limit); err == nil {
		t.Fatal("NewDecoderFromRTPPayloads accepted a stream over the frame size limit")
	}
	if _, err := NewDecoder(clips[0], limit); err != nil {
		t.Fatalf("NewDecoder of the first sequence alone: %v", err)
	}
}

// TestDecoderFormatChangeSteadyStateZeroAlloc checks that a stream which
// changes format at a new sequence decodes without allocating, the change
// included: the pool slots and scratch are sized for every sequence when the
// decoder is built.
func TestDecoderFormatChangeSteadyStateZeroAlloc(t *testing.T) {
	clips, _ := formatChangeClips(t, []string{"lowdelay", "444", "10bit", "film-grain-8"})
	var tus [][]byte
	for _, c := range clips {
		tus = append(tus, c...)
	}
	for _, workers := range []int{1, 8} {
		dec, err := NewDecoder(tus, WithWorkers(workers))
		if err != nil {
			t.Fatal(err)
		}
		pass := func() {
			for {
				_, ok, err := dec.DecodeNext()
				if err != nil {
					t.Fatal(err)
				}
				if !ok {
					break
				}
			}
			if err := dec.Reset(); err != nil {
				t.Fatal(err)
			}
		}
		pass() // warm-up
		if allocs := testing.AllocsPerRun(3, pass); allocs != 0 {
			t.Errorf("workers=%d: %v allocations per pass, want 0", workers, allocs)
		}
		dec.Close()
	}
}

// encodeLimitTestStream encodes n small frames and returns the temporal units.
func encodeLimitTestStream(t *testing.T, w, h, n int) [][]byte {
	t.Helper()
	enc, err := NewVideoEncoder(VideoEncoderConfig{Width: w, Height: h, QIndex: 80})
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	var stream [][]byte
	for i := range n {
		frame := I420Frame{
			Y: make([]byte, w*h), U: make([]byte, w/2*h/2), V: make([]byte, w/2*h/2),
			YStride: w, ChromaStride: w / 2, Width: w, Height: h,
		}
		for j := range frame.Y {
			frame.Y[j] = uint8(30 + (j*7+i*11)%180)
		}
		for j := range frame.U {
			frame.U[j], frame.V[j] = 118, 134
		}
		out, err := enc.Encode(frame, false)
		if err != nil {
			t.Fatal(err)
		}
		stream = append(stream, append([]byte(nil), out.Data...))
	}
	return stream
}

func TestDecoderFrameSizeLimit(t *testing.T) {
	const w, h = 192, 128
	stream := encodeLimitTestStream(t, w, h, 3)

	for _, tc := range []struct {
		name  string
		limit int
		want  error
	}{
		{name: "unlimited", limit: 0},
		{name: "exact", limit: w * h},
		{name: "one below", limit: w*h - 1, want: ErrDecoderFrameSizeLimit},
		{name: "tiny", limit: 1, want: ErrDecoderFrameSizeLimit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dec, err := NewDecoder(stream, WithFrameSizeLimit(tc.limit))
			if !errors.Is(err, tc.want) {
				t.Fatalf("NewDecoder: err=%v want %v", err, tc.want)
			}
			if err == nil {
				frames, err := dec.DecodeAll()
				dec.Close()
				if err != nil || len(frames) != len(stream) {
					t.Fatalf("DecodeAll: %d frames, err=%v", len(frames), err)
				}
			}
			layered, err := NewLayeredDecoder(stream, WithFrameSizeLimit(tc.limit))
			if !errors.Is(err, tc.want) {
				t.Fatalf("NewLayeredDecoder: err=%v want %v", err, tc.want)
			}
			if err == nil {
				layered.Close()
			}
		})
	}

	if _, err := NewDecoder(stream, WithFrameSizeLimit(-1)); err == nil {
		t.Fatal("negative frame size limit accepted")
	}
}

// TestDecoderBoundsEventsPerPayload feeds payloads made of thousands of
// two-byte OBUs. Each OBU is parsed into a multi-kilobyte event, so the
// constructors must reject the payload instead of sizing buffers by its OBU
// count.
func TestDecoderBoundsEventsPerPayload(t *testing.T) {
	const obus = 20000
	const budget = 8 << 20

	// RTP: aggregation header, then length-prefixed one-byte OBU elements.
	for _, header := range []byte{0x78, 0x28, 0x10} { // padding, metadata, temporal delimiter
		rtp := make([]byte, 0, 1+2*obus)
		rtp = append(rtp, 0x00)
		for range obus {
			rtp = append(rtp, 0x01, header)
		}
		allocated := allocatedBy(func() {
			if dec, err := NewDecoderFromRTPPayloads([][]byte{rtp}); err == nil {
				dec.Close()
				t.Errorf("header %#x: NewDecoderFromRTPPayloads accepted %d OBUs in one payload", header, obus)
			}
			if dec, err := NewLayeredDecoderFromRTPPayloads([][]byte{rtp}); err == nil {
				dec.Close()
				t.Errorf("header %#x: NewLayeredDecoderFromRTPPayloads accepted %d OBUs in one payload", header, obus)
			}
		})
		if allocated > budget {
			t.Errorf("header %#x: RTP constructors allocated %d bytes for a %d-byte payload", header, allocated, len(rtp))
		}
	}

	// Low-overhead: temporal delimiters with a zero size field.
	low := make([]byte, 0, 2*obus)
	for range obus {
		low = append(low, 0x12, 0x00)
	}
	allocated := allocatedBy(func() {
		if dec, err := NewDecoder([][]byte{low}); err == nil {
			dec.Close()
			t.Errorf("NewDecoder accepted %d OBUs in one payload", obus)
		}
		if dec, err := NewLayeredDecoder([][]byte{low}); err == nil {
			dec.Close()
			t.Errorf("NewLayeredDecoder accepted %d OBUs in one payload", obus)
		}
	})
	if allocated > budget {
		t.Errorf("low-overhead constructors allocated %d bytes for a %d-byte payload", allocated, len(low))
	}
}

func allocatedBy(f func()) uint64 {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

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

// TestDecoderWorkersSteadyStateZeroAlloc pins the zero-allocation design on
// the multi-worker paths: SB-row wavefront lanes, the pooled CDEF and loop
// filter bands, multi-tile dispatch and the external super-res output.
// wavefrontTestVector has enough SB rows to engage wavefront lanes at the
// production threshold.
func TestDecoderWorkersSteadyStateZeroAlloc(t *testing.T) {
	clips := []string{
		profileClipPath("profile1-444-8bit-cdef-restoration-160x128.ivf"),
		profileClipPath("profile1-444-8bit-edgemv-130x130.ivf"),
		profileClipPath("profile1-444-8bit-multitile-2x1-256x256.ivf"),
		profileClipPath("profile1-444-8bit-superres-inter-160x128.ivf"),
		profileClipPath("profile1-444-8bit-filmgrain-96x96.ivf"),
		profileClipPath("profile1-444-10bit-cdef-restoration-160x128.ivf"),
		profileClipPath("superblock128-420-8bit-128x128.ivf"),
		profileClipPath("superres-restoration-420-8bit-160x128.ivf"),
		wavefrontTestVector,
	}
	for _, workers := range []int{2, 4} {
		for _, path := range clips {
			t.Run(fmt.Sprintf("%s/workers=%d", filepath.Base(path), workers), func(t *testing.T) {
				ivf, err := os.ReadFile(path)
				if err != nil {
					t.Fatalf("read clip: %v", err)
				}
				dec, err := NewDecoderFromIVF(ivf, WithWorkers(workers))
				if err != nil {
					t.Fatalf("NewDecoderFromIVF: %v", err)
				}
				defer dec.Close()
				decodeAll := func() {
					if err := dec.Reset(); err != nil {
						t.Fatalf("Reset: %v", err)
					}
					for {
						_, ok, err := dec.DecodeNext()
						if err != nil {
							t.Fatalf("DecodeNext: %v", err)
						}
						if !ok {
							return
						}
					}
				}
				for range 3 {
					decodeAll()
				}
				if allocs := testing.AllocsPerRun(10, decodeAll); allocs != 0 {
					t.Fatalf("DecodeNext allocs/run=%v want 0", allocs)
				}
			})
		}
	}
}

func TestDecoderOwnedIntraContextsPoisonAndHookFallback(t *testing.T) {
	path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", "profile0-420-8bit-sframe-64x64.ivf")
	ivf, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	owned, err := NewDecoderFromIVF(ivf, WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	defer owned.Close()
	generic, err := NewDecoderFromIVF(ivf, WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	defer generic.Close()
	hooks := 0
	generic.batch.Request.Tile.AfterBlock = func(tile.BlockLoopVisit) error { hooks++; return nil }
	for repeat := 0; repeat < 2; repeat++ {
		if repeat != 0 {
			if err := owned.Reset(); err != nil {
				t.Fatal(err)
			}
			if err := generic.Reset(); err != nil {
				t.Fatal(err)
			}
		}
		for _, dec := range []*Decoder{owned, generic} {
			for i := range dec.batch.TileScratch {
				m := &dec.batch.TileScratch[i].Loop.Mode
				for y := range m.GridMotionValid {
					for x := range m.GridMotionValid[y] {
						m.GridMotionValid[y][x] = 0xa5
						m.GridBlockSizeVisited[y][x] = 0xa5
					}
				}
				m.SBTopRightValid = true
				for x := range m.SBTopRightIntra {
					m.SBTopRightIntra[x] = 0xa5
					m.SBTopRightMotionValid[x] = 0xa5
					m.SBTopRightBlockSize[x] = tile.BlockSize(255)
				}
			}
		}
		for frame := 0; ; frame++ {
			got, ok, ge := owned.DecodeNext()
			want, wok, we := generic.DecodeNext()
			if ge != nil || we != nil || ok != wok || len(got) != len(want) {
				t.Fatalf("repeat%d frame%d owned=%v/%v generic=%v/%v", repeat, frame, ok, ge, wok, we)
			}
			if !ok {
				if frame != 8 {
					t.Fatalf("frames=%d", frame)
				}
				break
			}
			for i := range got {
				if visibleFrameMD5(got[i]) != visibleFrameMD5(want[i]) {
					t.Fatalf("repeat%d frame%d pixels differ from generic path", repeat, frame)
				}
			}
			if frame == 0 {
				if owned.batch.TileScratch[0].Loop.Mode.GridMotionValid[0][0] != 0xa5 || owned.batch.TileScratch[0].Loop.Mode.SBTopRightIntra[0] != 0xa5 || !owned.batch.TileScratch[0].Loop.Mode.SBTopRightValid {
					t.Fatal("first key frame did not select private contexts")
				}
				if generic.batch.TileScratch[0].Loop.Mode.GridMotionValid[0][0] == 0xa5 || hooks == 0 {
					t.Fatal("caller hook did not force complete context path")
				}
			}
		}
	}
}
