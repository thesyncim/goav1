package goav1

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	internaldecoder "github.com/thesyncim/goav1/internal/av1/decoder"
	internalobu "github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/obu"
	internalparser "github.com/thesyncim/goav1/internal/av1/parser"
	"hash/fnv"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
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

type vdClip struct {
	name    string
	header  []byte // 32-byte IVF header
	tus     [][]byte
	layered bool
}

type vdCase struct {
	clip   *vdClip
	name   string // clip/mutation
	kind   string // mutation family
	tus    [][]byte
	path   string
	dav1d  vdOracle
	goav1  vdResult
	goav18 vdResult
	layer  vdResult
	class  string
}

type vdFrame struct {
	tu   int
	hash uint64
}

type vdOracle struct {
	errs   []int
	frames []vdFrame
	msg    string
}

type vdResult struct {
	ran       bool
	construct bool // the error came from the constructor's probe
	errTU     int  // -1 when the stream decoded cleanly
	err       string
	panicked  bool
	hung      bool
	frames    []vdFrame
	allocMB   uint64
}

type vdOBU struct {
	start, end   int // whole OBU, header through payload
	typ          obu.Type
	hasSize      bool
	sizeOff      int
	sizeLen      int
	payOff, size int
}

func vdSplitOBUs(tu []byte) ([]vdOBU, bool) {
	var out []vdOBU
	pos := 0
	for pos < len(tu) {
		o := vdOBU{start: pos}
		h := tu[pos]
		o.typ = obu.Type((h >> 3) & 0xf)
		ext := h&4 != 0
		o.hasSize = h&2 != 0
		p := pos + 1
		if ext {
			p++
		}
		if p > len(tu) || !o.hasSize {
			return out, false
		}
		o.sizeOff = p
		var v uint64
		i := 0
		for ; i < 8; i++ {
			if p+i >= len(tu) {
				return out, false
			}
			b := tu[p+i]
			v |= uint64(b&0x7f) << (7 * i)
			if b&0x80 == 0 {
				break
			}
		}
		o.sizeLen = i + 1
		o.payOff = p + o.sizeLen
		if v > uint64(len(tu)-o.payOff) {
			return out, false
		}
		o.size = int(v)
		o.end = o.payOff + o.size
		out = append(out, o)
		pos = o.end
	}
	return out, true
}

func vdLEB(v int) []byte {
	var b []byte
	for {
		c := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			b = append(b, c|0x80)
			continue
		}
		return append(b, c)
	}
}

// vdRebuild returns tu with OBU o's payload replaced by pay (size field
// re-encoded, so the framing stays valid).
func vdRebuild(tu []byte, o vdOBU, pay []byte) []byte {
	out := append([]byte{}, tu[:o.sizeOff]...)
	out = append(out, vdLEB(len(pay))...)
	out = append(out, pay...)
	return append(out, tu[o.end:]...)
}

func vdClone(tus [][]byte) [][]byte {
	out := make([][]byte, len(tus))
	for i, tu := range tus {
		out[i] = append([]byte(nil), tu...)
	}
	return out
}

func vdWriteIVF(path string, header []byte, tus [][]byte) error {
	var buf bytes.Buffer
	h := append([]byte(nil), header...)
	binary.LittleEndian.PutUint32(h[24:], uint32(len(tus)))
	buf.Write(h)
	for i, tu := range tus {
		var fh [12]byte
		binary.LittleEndian.PutUint32(fh[0:], uint32(len(tu)))
		binary.LittleEndian.PutUint64(fh[4:], uint64(i))
		buf.Write(fh[:])
		buf.Write(tu)
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

func vdLoadClip(t *testing.T, name, path string, maxTUs int, layered bool) *vdClip {
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	payloads := featureVectorPayloads(t, data)
	if maxTUs > 0 && len(payloads) > maxTUs {
		payloads = payloads[:maxTUs]
	}
	hdrLen := int(binary.LittleEndian.Uint16(data[6:]))
	header := append([]byte(nil), data[:32]...)
	if hdrLen != 32 {
		binary.LittleEndian.PutUint16(header[6:], 32)
	}
	return &vdClip{name: name, header: header, tus: vdClone(payloads), layered: layered}
}

// vdHeaderBits reports, for every TU, the frame-header bit count of each
// OBU_FRAME/OBU_FRAME_HEADER, keyed by the OBU index within the TU. It runs
// goav1's own parser over the unmutated clip.
func vdHeaderBits(clip *vdClip) []map[int]int {
	out := make([]map[int]int, len(clip.tus))
	var s internaldecoder.Stream
	events := make([]internaldecoder.Event, decoderMaxPayloadEvents)
	for i, tu := range clip.tus {
		out[i] = map[int]int{}
		n, err := s.PushLowOverhead(tu, events)
		if err != nil {
			return out
		}
		for j := range n {
			e := &events[j]
			switch e.Kind {
			case internaldecoder.EventFrame, internaldecoder.EventFrameHeader:
				out[i][j] = int(e.FilmGrain.BitsRead)
			}
		}
	}
	return out
}

func vdMutations(clip *vdClip) []vdCase {
	rng := rand.New(rand.NewSource(int64(fnvString(clip.name))))
	n := len(clip.tus)
	var cases []vdCase
	add := func(kind, name string, tus [][]byte) {
		cases = append(cases, vdCase{clip: clip, kind: kind, name: clip.name + "/" + name, tus: tus})
	}
	pick := []int{0}
	if n > 2 {
		pick = append(pick, n/2)
	}
	if n > 1 {
		pick = append(pick, n-1)
	}
	hdrBits := vdHeaderBits(clip)

	// Truncate a whole TU (the last OBU's framing becomes invalid).
	for _, i := range pick {
		for _, cut := range []int{len(clip.tus[i]) / 2, len(clip.tus[i]) - 1} {
			if cut <= 0 {
				continue
			}
			tus := vdClone(clip.tus)
			tus[i] = tus[i][:cut]
			add("trunc-tu", fmt.Sprintf("trunc-tu%d-%d", i, cut), tus)
		}
	}
	for _, i := range pick {
		obus, ok := vdSplitOBUs(clip.tus[i])
		if !ok {
			continue
		}
		for j, o := range obus {
			// Truncate one OBU's payload but keep the framing valid.
			if o.size > 0 {
				cuts := []int{o.size - 1, o.size / 2}
				if o.typ == obu.TypeFrame || o.typ == obu.TypeFrameHeader {
					if hb, ok := hdrBits[i][j]; ok {
						cuts = append(cuts, (hb+7)/8, hb/8)
					}
				}
				seen := map[int]bool{}
				for _, cut := range cuts {
					if cut < 0 || cut >= o.size || seen[cut] {
						continue
					}
					seen[cut] = true
					tus := vdClone(clip.tus)
					tus[i] = vdRebuild(clip.tus[i], o, clip.tus[i][o.payOff:o.payOff+cut])
					add("trunc-obu", fmt.Sprintf("trunc-obu%d.%d-%s-%d", i, j, o.typ, cut), tus)
				}
			}
			// Corrupt the size field (the TU keeps its length).
			for _, delta := range []int{1, -1} {
				v := o.size + delta
				if v < 0 {
					continue
				}
				enc := vdLEB(v)
				if len(enc) > o.sizeLen {
					continue
				}
				for len(enc) < o.sizeLen { // pad to the original width
					enc[len(enc)-1] |= 0x80
					enc = append(enc, 0)
				}
				tus := vdClone(clip.tus)
				copy(tus[i][o.sizeOff:], enc)
				add("obu-size", fmt.Sprintf("obu-size%d.%d-%s%+d", i, j, o.typ, delta), tus)
			}
			{
				tus := vdClone(clip.tus)
				tu := append([]byte{}, clip.tus[i][:o.sizeOff]...)
				tu = append(tu, 0xff, 0xff, 0xff, 0xff, 0x0f)
				tus[i] = append(tu, clip.tus[i][o.payOff:]...)
				add("obu-size", fmt.Sprintf("obu-size%d.%d-%s-huge", i, j, o.typ), tus)
			}
			// Drop the OBU.
			{
				tus := vdClone(clip.tus)
				tus[i] = append(append([]byte{}, clip.tus[i][:o.start]...), clip.tus[i][o.end:]...)
				if len(tus[i]) > 0 {
					add("drop-obu", fmt.Sprintf("drop-obu%d.%d-%s", i, j, o.typ), tus)
				}
			}
			// Bit flips by region.
			flip := func(kind string, lo, hi, count int) {
				if hi <= lo {
					return
				}
				positions := map[int]bool{}
				if hi-lo <= count {
					for b := lo; b < hi; b++ {
						positions[b] = true
					}
				} else {
					for len(positions) < count {
						positions[lo+rng.Intn(hi-lo)] = true
					}
				}
				keys := make([]int, 0, len(positions))
				for b := range positions {
					keys = append(keys, b)
				}
				sort.Ints(keys)
				for _, b := range keys {
					tus := vdClone(clip.tus)
					tus[i][o.payOff+b/8] ^= 0x80 >> (b % 8)
					add(kind, fmt.Sprintf("%s%d.%d-bit%d", kind, i, j, b), tus)
				}
			}
			switch o.typ {
			case obu.TypeSequenceHeader:
				if i == 0 {
					flip("flip-seq", 0, o.size*8, 24)
				}
			case obu.TypeFrame, obu.TypeFrameHeader:
				hb, ok := hdrBits[i][j]
				if !ok {
					hb = min(o.size*8, 128)
				}
				flip("flip-fh", 0, hb, 12)
				if o.typ == obu.TypeFrame {
					start := (hb + 7) / 8 * 8
					flip("flip-tg", start, min(o.size*8, start+16), 6)
				}
			case obu.TypeTileGroup:
				flip("flip-tg", 0, min(o.size*8, 16), 8)
			}
		}
	}
	// Duplicate a TU, and drop one.
	for _, i := range pick {
		tus := vdClone(clip.tus[:i+1])
		tus = append(tus, vdClone(clip.tus[i:])...)
		add("dup-tu", fmt.Sprintf("dup-tu%d", i), tus)
		if n > 1 {
			tus := vdClone(clip.tus[:i])
			tus = append(tus, vdClone(clip.tus[i+1:])...)
			add("drop-tu", fmt.Sprintf("drop-tu%d", i), tus)
		}
	}
	return cases
}

func fnvString(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// vdHashFrame matches the oracle's picture hash.
func vdHashFrame(f *Frame) uint64 {
	h := fnv.New64a()
	var dims [8]byte
	binary.LittleEndian.PutUint32(dims[0:], uint32(f.Y.Width))
	binary.LittleEndian.PutUint32(dims[4:], uint32(f.Y.Height))
	h.Write(dims[:])
	hashFrameVisible(h, f)
	return h.Sum64()
}

type vdDecoder interface {
	DecodeNext() ([]*Frame, bool, error)
	Close()
}

func vdNew(tus [][]byte, workers int, layered bool) (vdDecoder, error) {
	if layered {
		return NewLayeredDecoder(tus, WithWorkers(workers))
	}
	return NewDecoder(tus, WithWorkers(workers))
}

// vdRun decodes tus with goav1. A constructor failure is located by
// bisecting on the payload prefix, and the clean prefix is then decoded so
// its frames can be compared.
func vdRun(tus [][]byte, workers int, layered bool) vdResult {
	res := vdResult{ran: true, errTU: -1}
	dec, err := vdNew(tus, workers, layered)
	limit := len(tus)
	if err != nil {
		res.construct = true
		res.err = err.Error()
		lo, hi := 0, len(tus) // prefix tus[:lo] constructs (or is empty), tus[:hi] fails
		for hi-lo > 1 {
			mid := (lo + hi) / 2
			d, e := vdNew(tus[:mid], workers, layered)
			if e == nil {
				d.Close()
				lo = mid
			} else {
				hi = mid
			}
		}
		res.errTU = hi - 1
		limit = hi - 1
		if limit == 0 {
			return res
		}
		dec, err = vdNew(tus[:limit], workers, layered)
		if err != nil {
			return res
		}
	}
	defer dec.Close()
	for i := 0; ; i++ {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			// The clean prefix before a constructor failure can still fail
			// to decode earlier; the first rejecting TU is the verdict.
			if res.errTU < 0 || i < res.errTU {
				res.errTU = i
				res.err = err.Error()
				res.construct = false
			}
			return res
		}
		if !ok {
			return res
		}
		for _, f := range frames {
			res.frames = append(res.frames, vdFrame{tu: i, hash: vdHashFrame(f)})
		}
	}
}

func vdRunGuarded(tus [][]byte, workers int, layered bool, timeout time.Duration) vdResult {
	done := make(chan vdResult, 1)
	go func() {
		var res vdResult
		defer func() {
			if r := recover(); r != nil {
				res = vdResult{ran: true, panicked: true, errTU: -1, err: fmt.Sprint(r)}
			}
			done <- res
		}()
		res = vdRun(tus, workers, layered)
	}()
	select {
	case r := <-done:
		return r
	case <-time.After(timeout):
		return vdResult{ran: true, hung: true, errTU: -1, err: "timeout"}
	}
}

func vdParseOracleLine(line string) (string, vdOracle, error) {
	f := strings.Fields(line)
	if len(f) != 5 {
		return "", vdOracle{}, fmt.Errorf("bad oracle line %q", line)
	}
	o := vdOracle{msg: strings.TrimPrefix(f[4], "msg=")}
	if errs := strings.TrimPrefix(f[2], "errs="); errs != "-" {
		for _, s := range strings.Split(errs, ",") {
			v, err := strconv.Atoi(s)
			if err != nil {
				return "", o, err
			}
			o.errs = append(o.errs, v)
		}
	}
	if pics := strings.TrimPrefix(f[3], "pics="); pics != "-" {
		for _, s := range strings.Split(pics, ",") {
			tu, h, ok := strings.Cut(s, ":")
			if !ok {
				return "", o, fmt.Errorf("bad pic %q", s)
			}
			ti, err := strconv.Atoi(tu)
			if err != nil {
				return "", o, err
			}
			hv, err := strconv.ParseUint(h, 16, 64)
			if err != nil {
				return "", o, err
			}
			o.frames = append(o.frames, vdFrame{tu: ti, hash: hv})
		}
	}
	return f[0], o, nil
}

func vdRunOracle(t *testing.T, oracle string, cases []vdCase) {
	var in bytes.Buffer
	index := map[string]int{}
	for i := range cases {
		in.WriteString(cases[i].path + "\n")
		index[cases[i].path] = i
	}
	cmd := exec.Command(oracle)
	cmd.Stdin = &in
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("oracle: %v", err)
	}
	sc := bufio.NewScanner(bytes.NewReader(out))
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	got := 0
	for sc.Scan() {
		path, o, err := vdParseOracleLine(sc.Text())
		if err != nil {
			t.Fatalf("%v", err)
		}
		i, ok := index[path]
		if !ok {
			t.Fatalf("oracle reported unknown path %q", path)
		}
		cases[i].dav1d = o
		got++
	}
	if got != len(cases) {
		t.Fatalf("oracle reported %d of %d cases", got, len(cases))
	}
}

func vdFramesBefore(frames []vdFrame, tu int) []vdFrame {
	if tu < 0 {
		return frames
	}
	for i, f := range frames {
		if f.tu >= tu {
			return frames[:i]
		}
	}
	return frames
}

func vdSameFrames(a, b []vdFrame) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].hash != b[i].hash {
			return false
		}
	}
	return true
}

// vdClassify compares one goav1 result with dav1d's.
//
//	agree        both accept with identical output, or both reject at the same TU
//	a            dav1d rejects and goav1 accepts the stream
//	a-late       both reject, goav1 at a later TU (it accepted a TU dav1d rejected)
//	b            goav1 rejects and dav1d accepts the stream
//	b-early      both reject, goav1 at an earlier TU
//	c            both accept with different output
//	c-prefix     the frames before the first error differ
//	crash        goav1 panicked or hung
func vdClassify(d vdOracle, g vdResult) string {
	if g.panicked || g.hung {
		return "crash"
	}
	dErr := -1
	if len(d.errs) > 0 {
		dErr = d.errs[0]
	}
	switch {
	case dErr < 0 && g.errTU < 0:
		if vdSameFrames(d.frames, g.frames) {
			return "agree"
		}
		return "c"
	case dErr >= 0 && g.errTU < 0:
		return "a"
	case dErr < 0 && g.errTU >= 0:
		return "b"
	}
	m := min(dErr, g.errTU)
	if !vdSameFrames(vdFramesBefore(d.frames, m), vdFramesBefore(g.frames, m)) {
		return "c-prefix"
	}
	switch {
	case g.errTU > dErr:
		return "a-late"
	case g.errTU < dErr:
		return "b-early"
	}
	return "agree"
}

func vdClips(t *testing.T) []*vdClip {
	var clips []*vdClip
	for _, v := range loadFeatureVectors(t) {
		clips = append(clips, vdLoadClip(t, "features/"+v.name, filepath.Join(featureVectorDir, v.name+".ivf"), 0, v.layered))
	}
	profiles, _ := filepath.Glob(filepath.Join("internal", "av1", "testvector", "testdata", "profiles", "*.ivf"))
	for _, p := range profiles {
		clips = append(clips, vdLoadClip(t, "profiles/"+strings.TrimSuffix(filepath.Base(p), ".ivf"), p, 24, false))
	}
	for _, name := range []string{
		"aomgood_testsrc2_360p_20f", "aomgood_testsrc2_360p_still", "aomgood_johnny_360p_600k_c2",
		"svtrtc_johnny_360p_500k_p12", "goav1_johnny_360p_500k", "svtra_mtbike_1080p_crf35_p8_2tile",
		"svtra_oldtown_720p_crf35_p8_fgs8", "aomrt_fourpeople_360p_500k_s9",
	} {
		p := filepath.Join("testdata", "realcorpus", "streams", name+".ivf")
		if _, err := os.Stat(p); err == nil {
			clips = append(clips, vdLoadClip(t, "corpus/"+name, p, 12, false))
		}
	}
	return clips
}

func TestDecoderValidationDiff(t *testing.T) {
	oracle := os.Getenv("GOAV1_DAV1D_ORACLE")
	if oracle == "" {
		t.Skip("set GOAV1_DAV1D_ORACLE to the tools/dav1d_validation_oracle binary")
	}
	filter := os.Getenv("GOAV1_VALIDATION_FILTER")
	workers8 := os.Getenv("GOAV1_VALIDATION_WORKERS8") != ""
	par := runtime.GOMAXPROCS(0)
	if v, err := strconv.Atoi(os.Getenv("GOAV1_VALIDATION_PAR")); err == nil && v > 0 {
		par = v
	}
	dir := os.Getenv("GOAV1_VALIDATION_KEEP")
	if dir == "" {
		dir = t.TempDir()
	} else if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	var clips []*vdClip
	for _, c := range vdClips(t) {
		if filter == "" || strings.Contains(c.name, filter) {
			clips = append(clips, c)
		}
	}

	// Baselines: the unmutated clips must agree exactly before any mutation
	// result is trusted.
	var cases []vdCase
	for _, c := range clips {
		cases = append(cases, vdCase{clip: c, kind: "baseline", name: c.name + "/baseline", tus: c.tus})
	}
	nBase := len(cases)
	for _, c := range clips {
		cases = append(cases, vdMutations(c)...)
	}
	for i := range cases {
		cases[i].path = filepath.Join(dir, fmt.Sprintf("%05d_%s.ivf", i, strings.ReplaceAll(cases[i].name, "/", "__")))
		if err := vdWriteIVF(cases[i].path, cases[i].clip.header, cases[i].tus); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d clips, %d cases", len(clips), len(cases))
	vdRunOracle(t, oracle, cases)

	var out *os.File
	if p := os.Getenv("GOAV1_VALIDATION_OUT"); p != "" {
		var err error
		if out, err = os.Create(p); err != nil {
			t.Fatal(err)
		}
		defer out.Close()
	}
	var mu sync.Mutex
	work := make(chan int)
	var wg sync.WaitGroup
	for range par {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				c := &cases[i]
				var before runtime.MemStats
				if par == 1 {
					runtime.ReadMemStats(&before)
				}
				c.goav1 = vdRunGuarded(c.tus, 1, false, 60*time.Second)
				if par == 1 {
					// Only meaningful sequentially: TotalAlloc is process-wide.
					var after runtime.MemStats
					runtime.ReadMemStats(&after)
					c.goav1.allocMB = (after.TotalAlloc - before.TotalAlloc) >> 20
				}
				c.class = vdClassify(c.dav1d, c.goav1)
				if workers8 {
					c.goav18 = vdRunGuarded(c.tus, 8, false, 60*time.Second)
				}
				if c.clip.layered {
					c.layer = vdRunGuarded(c.tus, 4, true, 60*time.Second)
				}
				mu.Lock()
				if out != nil {
					fmt.Fprintln(out, vdLine(c))
				}
				mu.Unlock()
			}
		}()
	}
	for i := range cases {
		work <- i
	}
	close(work)
	wg.Wait()

	counts := map[string]map[string]int{}
	bump := func(kind, class string) {
		if counts[kind] == nil {
			counts[kind] = map[string]int{}
		}
		counts[kind][class]++
		if counts["ALL"] == nil {
			counts["ALL"] = map[string]int{}
		}
		counts["ALL"][class]++
	}
	for i := range cases {
		c := &cases[i]
		if i < nBase && c.class != "agree" {
			t.Errorf("baseline %s: %s (%s)", c.name, c.class, vdLine(c))
		}
		bump(c.kind, c.class)
		// The accept/reject verdict must not depend on the worker count or
		// on the decoder front end.
		if c.goav18.ran && (c.goav18.errTU < 0) != (c.goav1.errTU < 0) {
			bump(c.kind, "workers8-verdict")
		}
		if c.goav18.ran && (c.goav18.panicked || c.goav18.hung) {
			bump(c.kind, "workers8-crash")
		}
		if c.layer.ran {
			if c.layer.panicked || c.layer.hung {
				bump(c.kind, "layered-crash")
			} else if (c.layer.errTU < 0) != (c.goav1.errTU < 0) {
				bump(c.kind, "layered-verdict")
			}
		}
	}
	kinds := make([]string, 0, len(counts))
	for k := range counts {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, k := range kinds {
		classes := make([]string, 0, len(counts[k]))
		for c := range counts[k] {
			classes = append(classes, c)
		}
		sort.Strings(classes)
		var parts []string
		for _, c := range classes {
			parts = append(parts, fmt.Sprintf("%s=%d", c, counts[k][c]))
		}
		t.Logf("%-10s %s", k, strings.Join(parts, " "))
	}
}

func vdLine(c *vdCase) string {
	dErr := -1
	if len(c.dav1d.errs) > 0 {
		dErr = c.dav1d.errs[0]
	}
	desc := func(r vdResult) string {
		if !r.ran {
			return "-"
		}
		s := fmt.Sprintf("err@%d/f%d", r.errTU, len(r.frames))
		if r.construct {
			s += "/ctor"
		}
		if r.panicked {
			s += "/PANIC"
		}
		if r.hung {
			s += "/HANG"
		}
		if r.allocMB != 0 {
			s += fmt.Sprintf("/%dMB", r.allocMB)
		}
		if r.err != "" {
			s += " [" + strings.ReplaceAll(r.err, " ", "_") + "]"
		}
		return s
	}
	return fmt.Sprintf("%-8s %-10s %s dav1d=err@%d/f%d(errs=%d)[%s] goav1=%s w8=%s layered=%s",
		c.class, c.kind, c.name, dErr, len(c.dav1d.frames), len(c.dav1d.errs), c.dav1d.msg, desc(c.goav1), desc(c.goav18), desc(c.layer))
}

// TestDecoderValidationDump prints goav1's parse of one stream, TU by TU, for
// triaging a disagreement: GOAV1_VALIDATION_DUMP=<file.ivf>.
func TestDecoderValidationDump(t *testing.T) {
	path := os.Getenv("GOAV1_VALIDATION_DUMP")
	if path == "" {
		t.Skip("set GOAV1_VALIDATION_DUMP")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var s internaldecoder.Stream
	events := make([]internaldecoder.Event, decoderMaxPayloadEvents)
	for i, tu := range featureVectorPayloads(t, data) {
		n, err := s.PushLowOverhead(tu, events)
		for j := range n {
			e := &events[j]
			t.Logf("tu %d obu %d %s kind=%d show=%v type=%d hdrbits=%d tg=%+v size=%dx%d tiles=%dx%d",
				i, j, e.Type, e.Kind, e.FrameHeader.ShowFrame, e.FrameHeader.FrameType, e.FilmGrain.BitsRead,
				e.TileGroup, e.FrameSize.UpscaledWidth, e.FrameSize.Height, e.TileInfo.Cols, e.TileInfo.Rows)
		}
		if err != nil {
			t.Logf("tu %d: %v", i, err)
			return
		}
	}
}
