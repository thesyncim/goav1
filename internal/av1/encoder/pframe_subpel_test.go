package encoder_test

import (
	"bytes"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	goav1 "github.com/thesyncim/goav1"
	"github.com/thesyncim/goav1/internal/av1/encoder"
	avframe "github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/motion"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestEncodePFrameSubpelMotion is the subpel gate: the second frame is the
// first resampled at a half-pixel offset, so only subpel motion vectors can
// predict it well. The decode must match the encoder reconstruction
// bit-for-bit (the encoder predicts through the decoder's own convolve), the
// P-frame must compress far below the keyframe (the refinement found the
// half-pel motion), and aomdec must agree when present.
func TestEncodePFrameSubpelMotion(t *testing.T) {
	const w, h = 192, 128
	cw, ch := w/2, h/2
	src1, src2 := makeHalfPelMotionFrames(t, w, h)

	const qIndex = 60
	keyTU, keyRecon, err := encoder.EncodeKeyframe(src1, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	pTU, pRecon, err := encoder.EncodePFrame(src2, keyRecon, qIndex)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("key %d bytes, half-pel P %d bytes", len(keyTU), len(pTU))
	if len(pTU)*6 >= len(keyTU) {
		t.Fatalf("P %d bytes not well below key %d: subpel motion not engaged?", len(pTU), len(keyTU))
	}

	dec, err := goav1.NewDecoder([][]byte{keyTU, pTU})
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	f := frames[1]
	comparePlane(t, "Y", f.Y, pRecon.Y, w, h, w)
	comparePlane(t, "U", f.U, pRecon.U, cw, ch, cw)
	comparePlane(t, "V", f.V, pRecon.V, cw, ch, cw)
	psnr := planePSNR(src2.Y, pRecon.Y)
	t.Logf("half-pel P luma PSNR %.2f dB", psnr)
	if psnr < 33 {
		t.Fatalf("PSNR %.2f below floor", psnr)
	}

	if aomdec, err := exec.LookPath("aomdec"); err == nil {
		stream := ivf.AppendFileHeader(nil, w, h, 30, 1, 2)
		stream = ivf.AppendFrame(stream, keyTU, 0)
		stream = ivf.AppendFrame(stream, pTU, 1)
		dir := t.TempDir()
		p := filepath.Join(dir, "s.ivf")
		os.WriteFile(p, stream, 0o644)
		outPath := filepath.Join(dir, "o.yuv")
		if out, err := exec.Command(aomdec, "--rawvideo", "-o", outPath, p).CombinedOutput(); err != nil {
			t.Fatalf("aomdec: %v\n%s", err, out)
		}
		got, _ := os.ReadFile(outPath)
		want := append(append(append([]byte(nil), keyRecon.Y...), keyRecon.U...), keyRecon.V...)
		want = append(append(append(want, pRecon.Y...), pRecon.U...), pRecon.V...)
		if string(got) != string(want) {
			t.Fatal("aomdec output differs from reconstruction")
		}
		t.Log("aomdec: half-pel stream bit-exact")
	}
}

func TestVideoEncoderMinEffortSkipsSubpelMotion(t *testing.T) {
	const w, h = 192, 128
	src1, src2 := makeHalfPelMotionFrames(t, w, h)

	encode := func(t *testing.T, effort int8) ([]byte, []byte) {
		t.Helper()
		enc, err := encoder.NewVideoEncoder(w, h, 60)
		if err != nil {
			t.Fatal(err)
		}
		defer enc.Close()
		enc.SetMaxThreads(1)
		enc.SetScreenContentSelection(true)
		if err := enc.SetContentHint(encoder.ContentCamera); err != nil {
			t.Fatalf("SetContentHint: %v", err)
		}
		if err := enc.SetEffortLevel(effort); err != nil {
			t.Fatalf("SetEffortLevel: %v", err)
		}
		keyTU, key, err := enc.Encode(src1, true)
		if err != nil {
			t.Fatalf("Encode key: %v", err)
		}
		if !key {
			t.Fatal("first frame was not key")
		}
		keyCopy := append([]byte(nil), keyTU...)
		pTU, key, err := enc.Encode(src2, false)
		if err != nil {
			t.Fatalf("Encode delta: %v", err)
		}
		if key {
			t.Fatal("second frame unexpectedly became key")
		}
		return keyCopy, append([]byte(nil), pTU...)
	}

	defaultKey, defaultP := encode(t, 0)
	minKey, minP := encode(t, encoder.WebRTCMinEffortLevel)
	if bytes.Equal(defaultP, minP) {
		t.Fatal("min effort produced the same delta frame as default effort")
	}
	if len(minP) <= len(defaultP) {
		t.Fatalf("min effort delta %d bytes, default %d: want faster full-pel path to spend more bits on half-pel motion", len(minP), len(defaultP))
	}

	seq := parseFirstSequenceHeader(t, minKey)
	prefix := parseFirstFramePrefix(t, minP, seq)
	if prefix.AllowScreenContentTools || prefix.ForceIntegerMV {
		t.Fatalf("min effort changed camera content MV signaling: prefix=%+v", prefix)
	}
	defaultLF, _ := parseFirstInterFrameFilters(t, defaultP, seq, w, h)
	if defaultLF.LevelY == [2]uint8{} {
		t.Fatalf("default effort loop filter levels=%+v want enabled", defaultLF.LevelY)
	}
	minLF, minCDEF := parseFirstInterFrameFilters(t, minP, seq, w, h)
	if minLF.LevelY != [2]uint8{} || minLF.LevelU != 0 || minLF.LevelV != 0 {
		t.Fatalf("min effort loop filter=%+v want disabled", minLF)
	}
	if minCDEF.StrengthCount != 1 || minCDEF.YStrength[0] != 0 || minCDEF.UVStrength[0] != 0 {
		t.Fatalf("min effort cdef=%+v want no-op strengths", minCDEF)
	}
	decodeTemporalUnits(t, defaultKey, defaultP)
	decodeTemporalUnits(t, minKey, minP)
}

func makeHalfPelMotionFrames(t *testing.T, w, h int) (encoder.SourceFrame420, encoder.SourceFrame420) {
	t.Helper()
	cw, ch := w/2, h/2
	rng := rand.New(rand.NewSource(31))
	src1 := encoder.SourceFrame420{
		Y:            make([]byte, w*h),
		U:            make([]byte, cw*ch),
		V:            make([]byte, cw*ch),
		YStride:      w,
		ChromaStride: cw,
		Width:        w,
		Height:       h,
	}
	for y := range h {
		for x := range w {
			src1.Y[y*w+x] = uint8(50 + rng.Intn(150)) // aperiodic texture
		}
	}
	for i := range src1.U {
		src1.U[i] = 120
		src1.V[i] = 130
	}

	// Frame 2: frame 1 shifted by half a pixel horizontally using the codec's
	// own 8-tap half-pel filter, so full-pel-only search cannot match it.
	src2 := src1
	src2.Y = make([]byte, w*h)
	dstP := avframe.Plane{Pix: src2.Y, Stride: w, Width: w, Height: h}
	refP := avframe.Plane{Pix: src1.Y, Stride: w, Width: w, Height: h}
	halfPel := motion.Vector{Col: 4}
	for py := 0; py < h; py += 8 {
		for px := 0; px < w; px += 8 {
			refX, refY, subX, subY, err := motion.ReferenceOriginSubsampled(px, py, halfPel, false, false)
			if err != nil {
				t.Fatal(err)
			}
			if err := motion.PredictInterPlaneBlockFromOriginWithFilterBitDepth(dstP, refP, 1, 8, px, py, refX, refY, 8, 8, subX, subY, motion.InterpFilters{}); err != nil {
				t.Fatal(err)
			}
		}
	}
	src2.U = append([]byte(nil), src1.U...)
	src2.V = append([]byte(nil), src1.V...)
	return src1, src2
}

func parseFirstSequenceHeader(t *testing.T, tu []byte) parser.SequenceHeader {
	t.Helper()
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			break
		}
		if unit.Header.Type == obu.TypeSequenceHeader {
			seq, err := parser.ParseSequenceHeader(unit.Payload)
			if err != nil {
				t.Fatalf("ParseSequenceHeader: %v", err)
			}
			return seq
		}
	}
	t.Fatal("missing sequence header")
	return parser.SequenceHeader{}
}

func parseFirstFramePrefix(t *testing.T, tu []byte, seq parser.SequenceHeader) parser.FrameHeaderPrefix {
	t.Helper()
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			break
		}
		if unit.Header.Type == obu.TypeFrameHeader || unit.Header.Type == obu.TypeFrame {
			prefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, seq)
			if err != nil {
				t.Fatalf("ParseFrameHeaderPrefix: %v", err)
			}
			return prefix
		}
	}
	t.Fatal("missing frame header")
	return parser.FrameHeaderPrefix{}
}

func parseFirstInterFrameFilters(t *testing.T, tu []byte, seq parser.SequenceHeader, width, height int) (parser.LoopFilterParams, parser.CDEFParams) {
	t.Helper()
	it := obu.NewLowOverheadIterator(tu)
	for {
		unit, ok, err := it.Next()
		if err != nil {
			t.Fatalf("parse low-overhead OBU: %v", err)
		}
		if !ok {
			break
		}
		if unit.Header.Type != obu.TypeFrameHeader && unit.Header.Type != obu.TypeFrame {
			continue
		}
		prefix, err := parser.ParseFrameHeaderPrefix(unit.Payload, seq)
		if err != nil {
			t.Fatalf("ParseFrameHeaderPrefix: %v", err)
		}
		refs := parserReferenceStateForFrame(width, height)
		size, err := parser.ParseFrameSize(unit.Payload, seq, prefix, &refs, unit.Header.TemporalID, unit.Header.SpatialID)
		if err != nil {
			t.Fatalf("ParseFrameSize: %v", err)
		}
		tiles, err := parser.ParseTileInfo(unit.Payload, seq, prefix, size)
		if err != nil {
			t.Fatalf("ParseTileInfo: %v", err)
		}
		quant, err := parser.ParseQuantizationParams(unit.Payload, seq, tiles)
		if err != nil {
			t.Fatalf("ParseQuantizationParams: %v", err)
		}
		seg, err := parser.ParseSegmentationParams(unit.Payload, prefix, quant, nil)
		if err != nil {
			t.Fatalf("ParseSegmentationParams: %v", err)
		}
		delta, err := parser.ParseDeltaParams(unit.Payload, size, quant, seg)
		if err != nil {
			t.Fatalf("ParseDeltaParams: %v", err)
		}
		lf, err := parser.ParseLoopFilterParams(unit.Payload, seq, prefix, size, seg, delta, nil)
		if err != nil {
			t.Fatalf("ParseLoopFilterParams: %v", err)
		}
		cdef, err := parser.ParseCDEFParams(unit.Payload, seq, size, seg, lf)
		if err != nil {
			t.Fatalf("ParseCDEFParams: %v", err)
		}
		return lf, cdef
	}
	t.Fatal("missing frame header")
	return parser.LoopFilterParams{}, parser.CDEFParams{}
}

func parserReferenceStateForFrame(width, height int) parser.ReferenceState {
	var refs parser.ReferenceState
	defaultGlobal := parser.DefaultGlobalMotionParams()
	for i := range parser.RefFrames {
		refs.Frames[i] = parser.ReferenceFrame{
			Valid:        true,
			GlobalMotion: defaultGlobal,
			Size: parser.FrameSize{
				CodedWidth:          uint32(width),
				UpscaledWidth:       uint32(width),
				Height:              uint32(height),
				RenderWidth:         uint32(width),
				RenderHeight:        uint32(height),
				SuperResDenominator: 8,
			},
		}
	}
	return refs
}

func decodeTemporalUnits(t *testing.T, payloads ...[]byte) {
	t.Helper()
	dec, err := goav1.NewDecoder(payloads)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Close()
	frames, err := dec.DecodeAll()
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(frames) != len(payloads) {
		t.Fatalf("decoded %d frames, want %d", len(frames), len(payloads))
	}
}
