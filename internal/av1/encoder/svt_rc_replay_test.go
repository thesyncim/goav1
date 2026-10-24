package encoder

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// TestSVTRateControlReplaysSVT410Traces drives svtRateControl with the
// per-frame inputs an instrumented SVT-AV1 v4.1.0 fed its own one-pass CBR
// controller and requires the same qindex and frame target on every picture.
//
// The traces in testdata/svtrc are SvtAv1EncApp runs (preset 13, --rtc 1,
// --pred-struct 1, --rc 2, --buf-initial-sz 500) on the first 120 frames of the
// Xiph 1280x720 60 fps FourPeople and Johnny clips at 3 Mbps, once with the
// default three-layer hierarchy and once flat (--hierarchical-levels 0).
// svt410_rc_trace.patch is the logging added to rc_vbr_cbr.c: per picture the
// state svt_av1_rc_calc_qindex_rate_control chose q from (ME distortion,
// reference q / layer / slice type, rate-correction factors) and, after the
// frame, its coded size and cyclic-refresh segment statistics. The replay feeds
// the recorded ME distortion, reference, size and cyclic-refresh inputs, so a
// pass proves the controller arithmetic (buffer model, active-quality
// bounds, q regulation, flat and layered adjust_q_cbr, correction factors) is
// SVT's; it says nothing about the sizes the encoder produces.
func TestSVTRateControlReplaysSVT410Traces(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("testdata", "svtrc", "*.csv"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no SVT rate-control traces: %v", err)
	}
	for _, file := range files {
		name := filepath.Base(file)
		t.Run(name, func(t *testing.T) {
			replaySVTRCTrace(t, file, strings.HasPrefix(name, "flat_"))
		})
	}
}

func replaySVTRCTrace(t *testing.T, file string, flat bool) {
	t.Helper()
	fh, err := os.Open(file)
	if err != nil {
		t.Fatal(err)
	}
	defer fh.Close()
	var rows [][]int
	sc := bufio.NewScanner(fh)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Split(line, ",")
		row := make([]int, len(fields))
		for i, f := range fields {
			v, err := strconv.Atoi(f)
			if err != nil {
				t.Fatalf("%s: bad field %q in %q", file, f, line)
			}
			row[i] = v
		}
		if len(row) != 18 {
			t.Fatalf("%s: %d fields in %q, want 18", file, len(row), line)
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(rows) != 120 {
		t.Fatalf("%s: %d pictures, want 120", file, len(rows))
	}

	const (
		colPic = iota
		colKey
		colTL
		colQ
		colTarget
		colBits
		colME
		colPrevME
		colRefs
		colRefQ
		colRefTL
		colRefSlice
		colCRApply
		colCRSeg1
		colCRSeg2
		colCRDelta1
		colCRDelta2
		colCRCount
	)
	const iSlice = 1 // SVT-AV1 SliceType I_SLICE
	var rc svtRateControl
	rc.init(3_000_000, 60, 16, 255, 1280, 720, rcBuffers{initialMs: 500})
	for _, r := range rows {
		pic := r[colPic]
		f := svtRCFrame{
			key:         r[colKey] == 1,
			flat:        flat,
			hierLevels:  2,
			frameOffset: pic,
			width:       1280,
			height:      720,
			gfUpdate:    pic%lpd1MaxGFInterval == 0,
			refValid:    r[colRefs] > 0,
			refBaseQ:    r[colRefQ],
			refLayer:    r[colRefTL],
			refIsI:      r[colRefSlice] == iSlice,
		}
		if !flat {
			f.temporalLayer = r[colTL]
		}
		rc.curAvgBaseMEDist = uint32(r[colME])
		rc.prevAvgBaseMEDist = uint32(r[colPrevME])
		rc.target(&f)
		q := int(rc.pickQ(&f))
		if f.thisFrame != r[colTarget] || q != r[colQ] {
			t.Fatalf("picture %d: target %d q %d, SVT-AV1 v4.1.0 target %d q %d", pic, f.thisFrame, q, r[colTarget], r[colQ])
		}
		var cr svtCRStats
		if r[colCRApply] == 1 {
			cr = svtCRStats{apply: true, seg1: r[colCRSeg1], seg2: r[colCRSeg2], delta1: r[colCRDelta1], delta2: r[colCRDelta2], sbCount: r[colCRCount]}
		}
		rc.postencode(&f, r[colQ], r[colBits], cr)
	}
}

func TestRCBuffersResolve(t *testing.T) {
	got, err := rcBuffers{}.resolve()
	if err != nil || got != (rcBuffers{initialMs: 600, optimalMs: 600, maxMs: 1000}) {
		t.Fatalf("zero value resolves to %+v, %v; want SVT-AV1's 600 / 600 / 1000", got, err)
	}
	got, err = rcBuffers{initialMs: 500, optimalMs: 600, maxMs: 1000}.resolve()
	if err != nil || got != (rcBuffers{initialMs: 500, optimalMs: 600, maxMs: 1000}) {
		t.Fatalf("500 / 600 / 1000 resolves to %+v, %v", got, err)
	}
	// enc_settings.c clamps an initial or optimal size that is not below the
	// maximum to maximum-1.
	got, err = rcBuffers{initialMs: 800, optimalMs: 900, maxMs: 700}.resolve()
	if err != nil || got != (rcBuffers{initialMs: 699, optimalMs: 699, maxMs: 700}) {
		t.Fatalf("800 / 900 / 700 resolves to %+v, %v; want 699 / 699 / 700", got, err)
	}
	// The default initial size is clamped the same way under a short maximum.
	got, err = rcBuffers{maxMs: 500}.resolve()
	if err != nil || got != (rcBuffers{initialMs: 499, optimalMs: 499, maxMs: 500}) {
		t.Fatalf("max 500 resolves to %+v, %v; want 499 / 499 / 500", got, err)
	}
	for _, bad := range []rcBuffers{{initialMs: -1}, {initialMs: 19}, {optimalMs: 10001}, {maxMs: 5}, {maxMs: 20000}} {
		if _, err := bad.resolve(); err == nil {
			t.Fatalf("%+v resolved without error", bad)
		}
	}
}

func TestRateControlConfigBuffersReachController(t *testing.T) {
	rc := RateControlConfig{TargetBitsPerSecond: 3_000_000, FramesPerSecond: 60, MinQIndex: 16, MaxQIndex: 255}
	def, err := newVideoEncoderCBRWithColor(320, 240, rc, defaultVideoColorConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer def.Close()
	if got := def.svtRC.startingBufferLevel; got != 1_800_000 {
		t.Fatalf("default starting buffer %d, want 600 ms of 3 Mbps", got)
	}
	rc.BufferInitialMs, rc.BufferOptimalMs, rc.BufferMaxMs = 500, 600, 1000
	enc, err := newVideoEncoderCBRWithColor(320, 240, rc, defaultVideoColorConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	if got, want := enc.svtRC.startingBufferLevel, int64(1_500_000); got != want {
		t.Fatalf("starting buffer %d, want %d", got, want)
	}
	if enc.svtRC.optimalBufferLevel != 1_800_000 || enc.svtRC.maximumBufferSize != 3_000_000 {
		t.Fatalf("optimal %d maximum %d", enc.svtRC.optimalBufferLevel, enc.svtRC.maximumBufferSize)
	}
	rc.BufferInitialMs = 5
	if _, err := newVideoEncoderCBRWithColor(320, 240, rc, defaultVideoColorConfig()); err == nil {
		t.Fatal("an initial buffer below 20 ms was accepted")
	}
}
