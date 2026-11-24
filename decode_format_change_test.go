package goav1

import (
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

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
