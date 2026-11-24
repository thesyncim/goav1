package goav1

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
)

// validationClipOBUs returns a feature clip's temporal units, each a fresh
// copy, with the byte offset of every OBU header in each unit.
func validationClipOBUs(t *testing.T, name string) ([][]byte, [][]int) {
	t.Helper()
	ivf, err := os.ReadFile(filepath.Join(featureVectorDir, name+".ivf"))
	if err != nil {
		t.Fatal(err)
	}
	var tus [][]byte
	var headers [][]int
	for _, p := range featureVectorPayloads(t, ivf) {
		tu := append([]byte(nil), p...)
		var offs []int
		for pos := 0; pos < len(tu); {
			_, n, err := internalobu.ParseLowOverhead(tu[pos:])
			if err != nil {
				t.Fatalf("ParseLowOverhead: %v", err)
			}
			offs = append(offs, pos)
			pos += n
		}
		tus = append(tus, tu)
		headers = append(headers, offs)
	}
	return tus, headers
}

func validationDecodeMD5(t *testing.T, tus [][]byte) string {
	t.Helper()
	dec, err := NewDecoder(tus)
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	return decodeStreamMD5(t, dec)
}

// TestDecoderIgnoresOBUReservedBits sets obu_reserved_1bit, and the
// extension header's reserved bits where there is one, on every OBU of a
// clip: the spec says decoders ignore them, as libaom and dav1d do, so the
// output must not change.
func TestDecoderIgnoresOBUReservedBits(t *testing.T) {
	tus, headers := validationClipOBUs(t, "444")
	want := validationDecodeMD5(t, tus)
	for i, offs := range headers {
		for _, off := range offs {
			tus[i][off] |= 0x01
			if tus[i][off]&0x04 != 0 {
				tus[i][off+1] |= 0x07
			}
		}
	}
	if got := validationDecodeMD5(t, tus); got != want {
		t.Fatalf("reserved bits changed the output: md5 %s, want %s", got, want)
	}
}

// TestDecoderRejectsOperatingPointWithoutLayers flips one bit of the
// sequence header's operating_point_idc so it names a temporal layer but no
// spatial layer, which dav1d's parse_seq_hdr rejects.
func TestDecoderRejectsOperatingPointWithoutLayers(t *testing.T) {
	tus, headers := validationClipOBUs(t, "444")
	seq := -1
	for _, off := range headers[0] {
		if internalobu.Type((tus[0][off]>>3)&0xf) == internalobu.TypeSequenceHeader {
			seq = off
		}
	}
	if seq < 0 {
		t.Fatal("no sequence header in the first temporal unit")
	}
	// The clip's header: seq_profile(3) still_picture reduced_still_picture
	// timing_info_present initial_display_delay_present
	// operating_points_cnt_minus_1(5), then operating_point_idc[0] at bit 12.
	unit, _, err := internalobu.ParseLowOverhead(tus[0][seq:])
	if err != nil {
		t.Fatal(err)
	}
	sh, err := internalparser.ParseSequenceHeader(unit.Payload)
	if err != nil || sh.TimingInfoPresent || sh.OperatingPointsCount != 1 || sh.OperatingPoints[0].IDC != 0 {
		t.Fatalf("unexpected clip header %+v err=%v", sh, err)
	}
	payload := seq + int(unit.HeaderLen) + int(unit.SizeLen)
	tus[0][payload+(12+11)/8] ^= 0x80 >> ((12 + 11) % 8) // idc = 0x001
	dec, err := NewDecoder(tus)
	if err == nil {
		dec.Close()
		t.Fatal("NewDecoder accepted operating_point_idc 0x001")
	}
	if !errors.Is(err, internalparser.ErrInvalidSequenceHeader) {
		t.Fatalf("NewDecoder err=%v want %v", err, internalparser.ErrInvalidSequenceHeader)
	}
}

// TestDecoderNewSequenceReleasesExternalReferences decodes two super-res
// clips back to back. The second sequence header starts a new coded video
// sequence, which drops every reference; on the super-res path the reference
// slots hold output-pool surface IDs, which the reset used to hand to the
// coded pool ("frame: invalid pool slot"). dav1d decodes the concatenation,
// and its output is each clip's output in turn.
func TestDecoderNewSequenceReleasesExternalReferences(t *testing.T) {
	dir := filepath.Join("internal", "av1", "testvector", "testdata", "profiles")
	for _, pair := range [][2]string{
		{"superres-420-8bit-160x128", "superres-restoration-420-8bit-160x128"},
		{"profile1-444-10bit-superres-160x128", "profile1-444-10bit-superres-restoration-160x128"},
	} {
		var tus [][]byte
		var want []byte
		for _, name := range pair {
			ivf, err := os.ReadFile(filepath.Join(dir, name+".ivf"))
			if err != nil {
				t.Fatal(err)
			}
			payloads := featureVectorPayloads(t, ivf)
			tus = append(tus, payloads...)
			want = append(want, validationDecodeRaw(t, payloads, 1)...)
		}
		for _, workers := range []int{1, 8} {
			if got := validationDecodeRaw(t, tus, workers); !bytes.Equal(got, want) {
				t.Errorf("%s+%s workers=%d: concatenated decode differs from the clips decoded apart", pair[0], pair[1], workers)
			}
		}
	}
}

// validationDecodeRaw returns the visible planes of every decoded frame.
func validationDecodeRaw(t *testing.T, tus [][]byte, workers int) []byte {
	t.Helper()
	dec, err := NewDecoder(tus, WithWorkers(workers))
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	defer dec.Close()
	var out bytes.Buffer
	for i := 0; ; i++ {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			t.Fatalf("DecodeNext %d: %v", i, err)
		}
		if !ok {
			return out.Bytes()
		}
		for _, f := range frames {
			bps := f.Layout.BytesPerSample
			for _, p := range []FramePlane{f.Y, f.U, f.V} {
				for row := 0; row < p.Height && len(p.Pix) != 0; row++ {
					out.Write(p.Pix[row*p.Stride : row*p.Stride+p.Width*bps])
				}
			}
		}
	}
}
