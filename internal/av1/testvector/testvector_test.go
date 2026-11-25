package testvector

import (
	"bytes"
	"context"
	"crypto/md5"
	"crypto/sha1"
	"errors"
	"fmt"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/ivf"
	"github.com/thesyncim/goav1/internal/av1/obu"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/rtp"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestCoreSuiteVectors(t *testing.T) {
	suite := CoreSuite()
	if suite.Name != "goav1-core" {
		t.Fatalf("suite name=%q", suite.Name)
	}
	if err := suite.Manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	for _, vector := range suite.Manifest.Vectors {
		switch vector.Kind {
		case KindOBU:
			checkOBUVector(t, vector)
		case KindRTP:
			checkRTPVector(t, vector)
		case KindIVF:
			checkIVFVector(t, vector)
		case KindParser:
			checkParserVector(t, vector)
		default:
			t.Fatalf("unhandled vector kind=%d", vector.Kind)
		}
	}
}

func TestCoreSuiteTagLookup(t *testing.T) {
	suite := CoreSuite()
	for i, vector := range suite.Manifest.Vectors {
		got, ok := CoreVector(vector.Tag)
		if !ok {
			t.Fatalf("missing vector tag=%x", vector.Tag)
		}
		if got.Tag != vector.Tag || got.Kind != vector.Kind || got.Name != vector.Name ||
			!bytes.Equal(got.Input, vector.Input) || !bytes.Equal(got.Want, vector.Want) {
			t.Fatalf("vector[%d]=%+v want %+v", i, got, vector)
		}
	}
	if got, ok := CoreVector(Tag(0xffff)); ok || got.Tag != 0 || got.Kind != 0 || got.Name != "" || got.Input != nil || got.Want != nil {
		t.Fatalf("invalid vector=%+v ok=%v", got, ok)
	}

	digest, ok := CoreFrameDigest(TagIVFSingleFrameAV1, 0)
	if !ok {
		t.Fatal("missing core digest")
	}
	if digest.MD5 != coreIVFSingleFrameMD5 {
		t.Fatalf("digest=%x want %x", digest.MD5, coreIVFSingleFrameMD5)
	}
	if got, ok := CoreFrameDigest(TagIVFSingleFrameAV1, 1); ok || got != (FrameDigest{}) {
		t.Fatalf("invalid digest=%+v ok=%v", got, ok)
	}
}

func TestCoreSuiteAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		suite := CoreSuite()
		if len(suite.Manifest.Vectors) != len(coreVectors) || len(suite.Manifest.Digests) != len(coreDigests) {
			t.Fatalf("suite=%+v", suite)
		}
		if _, ok := CoreVector(TagParserSequenceFullLowOverhead); !ok {
			t.Fatal("missing vector")
		}
		if _, ok := CoreFrameDigest(TagIVFSingleFrameAV1, 0); !ok {
			t.Fatal("missing digest")
		}
	})
	if allocs != 0 {
		t.Fatalf("CoreSuite allocated: %f", allocs)
	}
}

func checkOBUVector(t *testing.T, vector Vector) {
	t.Helper()
	unit, consumed, err := obu.ParseLowOverhead(vector.Input)
	if err != nil {
		t.Fatalf("%s ParseLowOverhead: %v", vector.Name, err)
	}
	if consumed != len(vector.Input) {
		t.Fatalf("%s consumed=%d want %d", vector.Name, consumed, len(vector.Input))
	}
	if unit.Header.Type != obu.TypeTemporalDelimiter {
		t.Fatalf("%s type=%d want %d", vector.Name, unit.Header.Type, obu.TypeTemporalDelimiter)
	}
	if !bytes.Equal(unit.Payload, vector.Want) {
		t.Fatalf("%s payload=%x want %x", vector.Name, unit.Payload, vector.Want)
	}
}

func checkRTPVector(t *testing.T, vector Vector) {
	t.Helper()
	it, err := rtp.NewIterator(vector.Input)
	if err != nil {
		t.Fatalf("%s NewIterator: %v", vector.Name, err)
	}
	elem, ok, err := it.Next()
	if err != nil {
		t.Fatalf("%s Next: %v", vector.Name, err)
	}
	if !ok {
		t.Fatalf("%s missing element", vector.Name)
	}
	if !bytes.Equal(elem.Data, vector.Want) {
		t.Fatalf("%s data=%x want %x", vector.Name, elem.Data, vector.Want)
	}
}

func checkIVFVector(t *testing.T, vector Vector) {
	t.Helper()
	it, err := ivf.NewIterator(vector.Input)
	if err != nil {
		t.Fatalf("%s NewIterator: %v", vector.Name, err)
	}
	frame, ok, err := it.Next()
	if err != nil {
		t.Fatalf("%s Next: %v", vector.Name, err)
	}
	if !ok {
		t.Fatalf("%s missing frame", vector.Name)
	}
	if !bytes.Equal(frame.Payload, vector.Want) {
		t.Fatalf("%s payload=%x want %x", vector.Name, frame.Payload, vector.Want)
	}
	digest, ok := CoreFrameDigest(vector.Tag, frame.Index)
	if !ok {
		t.Fatalf("%s missing digest", vector.Name)
	}
	if got := MD5(md5.Sum(frame.Payload)); got != digest.MD5 {
		t.Fatalf("%s md5=%x want %x", vector.Name, got, digest.MD5)
	}
}

func checkParserVector(t *testing.T, vector Vector) {
	t.Helper()
	payload := sequencePayloadFromVector(t, vector)
	if !bytes.Equal(payload, vector.Want) {
		t.Fatalf("%s payload=%x want %x", vector.Name, payload, vector.Want)
	}
	sh, err := parser.ParseSequenceHeader(payload)
	if err != nil {
		t.Fatalf("%s sequence header parse: %v", vector.Name, err)
	}
	checkLibaomSequenceConfig(t, vector, sh)
}

func sequencePayloadFromVector(t *testing.T, vector Vector) []byte {
	t.Helper()
	switch vector.Tag {
	case TagParserSequenceFullLowOverhead, TagParserSequenceReducedLowOverhead:
		unit, consumed, err := obu.ParseLowOverhead(vector.Input)
		if err != nil {
			t.Fatalf("%s ParseLowOverhead: %v", vector.Name, err)
		}
		if consumed != len(vector.Input) {
			t.Fatalf("%s consumed=%d want %d", vector.Name, consumed, len(vector.Input))
		}
		if unit.Header.Type != obu.TypeSequenceHeader {
			t.Fatalf("%s type=%d want %d", vector.Name, unit.Header.Type, obu.TypeSequenceHeader)
		}
		return unit.Payload
	default:
		t.Fatalf("%s unhandled parser vector tag=%x", vector.Name, vector.Tag)
		return nil
	}
}

func checkLibaomSequenceConfig(t *testing.T, vector Vector, sh parser.SequenceHeader) {
	t.Helper()
	reduced := vector.Tag == TagParserSequenceReducedLowOverhead
	if sh.SeqProfile != 0 || sh.OperatingPointsCount != 1 || sh.OperatingPoints[0].SeqLevelIdx != 0 || sh.OperatingPoints[0].SeqTier != 0 {
		t.Fatalf("%s profile/op=%+v", vector.Name, sh)
	}
	if sh.StillPicture != reduced || sh.ReducedStillPictureHeader != reduced {
		t.Fatalf("%s still/reduced=%v/%v want %v", vector.Name, sh.StillPicture, sh.ReducedStillPictureHeader, reduced)
	}
	if sh.InitialDisplayDelayPresent || sh.OperatingPoints[0].InitialDisplayDelayPresent || sh.OperatingPoints[0].InitialDisplayDelayMinus1 != 0 {
		t.Fatalf("%s initial delay=%+v", vector.Name, sh.OperatingPoints[0])
	}
	cc := sh.ColorConfig
	if cc.HighBitdepth || cc.TwelveBit || cc.MonoChrome || !cc.SubsamplingX || !cc.SubsamplingY || cc.ChromaSamplePosition != 0 {
		t.Fatalf("%s color config=%+v", vector.Name, cc)
	}
}

func TestFrameMD5MatchesVisibleRows(t *testing.T) {
	f, expected := testDigestFrame(t, frame.Format{
		Width:        5,
		Height:       3,
		BitDepth:     8,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        8,
	})

	got, err := FrameMD5(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := MD5(md5.Sum(expected)); got != want {
		t.Fatalf("md5=%x want %x", got, want)
	}
}

func TestFrameMD5MatchesMonochrome(t *testing.T) {
	f, expected := testDigestFrame(t, frame.Format{
		Width:      4,
		Height:     2,
		BitDepth:   8,
		MonoChrome: true,
		Align:      8,
	})

	got, err := FrameMD5(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := MD5(md5.Sum(expected)); got != want {
		t.Fatalf("md5=%x want %x", got, want)
	}
}

func TestFrameMD5MatchesSubsampledMonochromeNeutralChroma(t *testing.T) {
	f, expected := testDigestFrame(t, frame.Format{
		Width:        5,
		Height:       3,
		BitDepth:     8,
		MonoChrome:   true,
		SubsamplingX: true,
		SubsamplingY: true,
		Align:        8,
	})

	got, err := FrameMD5(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := MD5(md5.Sum(expected)); got != want {
		t.Fatalf("md5=%x want %x", got, want)
	}
}

func TestFrameMD5MatchesHighBitDepthRows(t *testing.T) {
	f, expected := testDigestFrame(t, frame.Format{
		Width:        3,
		Height:       2,
		BitDepth:     10,
		SubsamplingX: true,
		Align:        16,
	})

	got, err := FrameMD5(f)
	if err != nil {
		t.Fatal(err)
	}
	if want := MD5(md5.Sum(expected)); got != want {
		t.Fatalf("md5=%x want %x", got, want)
	}
}

func TestFrameMD5RejectsInvalidPlane(t *testing.T) {
	_, err := FrameMD5(frame.Frame{
		Format: frame.Format{Width: 2, Height: 1, BitDepth: 8},
		Layout: frame.Layout{
			BytesPerSample: 1,
		},
		Y: frame.Plane{Pix: []byte{0x01}, Stride: 1, Width: 2, Height: 1},
	})
	if !errors.Is(err, frame.ErrInvalidPlane) {
		t.Fatalf("FrameMD5 err=%v want %v", err, frame.ErrInvalidPlane)
	}
}

func testDigestFrame(t *testing.T, format frame.Format) (frame.Frame, []byte) {
	t.Helper()
	layout, err := frame.RequiredSize(format)
	if err != nil {
		t.Fatal(err)
	}
	backing := make([]byte, layout.Size)
	for i := range backing {
		backing[i] = 0xee
	}
	f, err := frame.Bind(backing, format)
	if err != nil {
		t.Fatal(err)
	}

	expected := make([]byte, 0, visibleFrameBytes(f))
	fillDigestPlane(&f.Y, f.Layout.BytesPerSample, 0x10, &expected)
	if f.Format.MonoChrome {
		fillNeutralChromaDigest(f.Format, f.Layout.BytesPerSample, &expected)
	} else {
		fillDigestPlane(&f.U, f.Layout.BytesPerSample, 0x70, &expected)
		fillDigestPlane(&f.V, f.Layout.BytesPerSample, 0xa0, &expected)
	}
	return f, expected
}

func fillDigestPlane(plane *frame.Plane, bytesPerSample int, seed byte, expected *[]byte) {
	for y := 0; y < plane.Height; y++ {
		for x := 0; x < plane.Width; x++ {
			off := y*plane.Stride + x*bytesPerSample
			v := uint16(seed) + uint16(y*7+x)
			plane.Pix[off] = byte(v)
			if bytesPerSample == 2 {
				plane.Pix[off+1] = byte(v >> 8)
			}
		}
		start := y * plane.Stride
		end := start + plane.Width*bytesPerSample
		*expected = append(*expected, plane.Pix[start:end]...)
	}
}

func fillNeutralChromaDigest(format frame.Format, bytesPerSample int, expected *[]byte) {
	width := format.Width
	height := format.Height
	if format.SubsamplingX {
		width = (width + 1) >> 1
	}
	if format.SubsamplingY {
		height = (height + 1) >> 1
	}
	neutral := uint16(1) << (format.BitDepth - 1)
	for range 2 {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				*expected = append(*expected, byte(neutral))
				if bytesPerSample == 2 {
					*expected = append(*expected, byte(neutral>>8))
				}
			}
		}
	}
}

func visibleFrameBytes(f frame.Frame) int {
	bytes := f.Y.Width * f.Y.Height * f.Layout.BytesPerSample
	if f.Format.MonoChrome {
		width := f.Format.Width
		height := f.Format.Height
		if f.Format.SubsamplingX {
			width = (width + 1) >> 1
		}
		if f.Format.SubsamplingY {
			height = (height + 1) >> 1
		}
		bytes += 2 * width * height * f.Layout.BytesPerSample
	} else {
		bytes += (f.U.Width*f.U.Height + f.V.Width*f.V.Height) * f.Layout.BytesPerSample
	}
	return bytes
}

func TestLibaomRemoteManifest(t *testing.T) {
	manifest := LibaomRemoteManifest()
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	if manifest.BaseURL != LibaomTestDataURL {
		t.Fatalf("base url=%q want %q", manifest.BaseURL, LibaomTestDataURL)
	}
	if len(manifest.Vectors) != len(libaomRemoteVectors) {
		t.Fatalf("vectors=%d want %d", len(manifest.Vectors), len(libaomRemoteVectors))
	}
	for _, vector := range manifest.Vectors {
		if vector.Stream.Name == "" || vector.MD5.Name != vector.Stream.Name+".md5" {
			t.Fatalf("vector files=%+v", vector)
		}
		if vector.Oracle != OracleFrameMD5 {
			t.Fatalf("%s oracle=%d", vector.Name, vector.Oracle)
		}
	}
}

func TestLibaomRemoteVectorLookupAndLabels(t *testing.T) {
	vector, ok := LibaomRemoteVector(TagDecoderLibaomQuantizer00)
	if !ok {
		t.Fatal("missing quantizer vector")
	}
	if vector.Stream.Name != "av1-1-b8-00-quantizer-00.ivf" || vector.Labels&VectorLabelQuantizer == 0 {
		t.Fatalf("quantizer vector=%+v", vector)
	}
	if got, ok := LibaomRemoteVector(Tag(0xffff)); ok || got.Tag != 0 {
		t.Fatalf("invalid vector=%+v ok=%v", got, ok)
	}

	manifest := LibaomRemoteManifest()
	fast := manifest.SelectRemote(SuiteLevelFast, 0, nil)
	if len(fast) != 14 {
		t.Fatalf("fast vectors=%d want 14", len(fast))
	}
	motion := manifest.SelectRemote(SuiteLevelRelevant, VectorLabelMotion, nil)
	if len(motion) != 2 {
		t.Fatalf("motion vectors=%d want 2", len(motion))
	}
	highBitDepth := manifest.SelectRemote(SuiteLevelRelevant, VectorLabelHighBitDepth, nil)
	if len(highBitDepth) != 3 {
		t.Fatalf("high-bit-depth vectors=%d want 3", len(highBitDepth))
	}
	full := manifest.SelectRemote(SuiteLevelFull, 0, nil)
	if len(full) != len(libaomRemoteVectors) {
		t.Fatalf("full vectors=%d want %d", len(full), len(libaomRemoteVectors))
	}
	extended := manifest.SelectRemote(SuiteLevelExtended, 0, nil)
	if len(extended) != 226 {
		t.Fatalf("extended vectors=%d want 226", len(extended))
	}
	// Extended cohort must be opt-in: every extended vector also has
	// SuiteLevelFull set, but none of them carries SuiteLevelFast or
	// SuiteLevelRelevant.
	for _, vector := range extended {
		if vector.Levels&(SuiteLevelFast|SuiteLevelRelevant) != 0 {
			t.Fatalf("extended vector %q leaks into default tiers: levels=%b", vector.Name, vector.Levels)
		}
		if vector.Levels&SuiteLevelFull == 0 {
			t.Fatalf("extended vector %q missing SuiteLevelFull: levels=%b", vector.Name, vector.Levels)
		}
	}
	extendedHighBitDepth := manifest.SelectRemote(SuiteLevelExtended, VectorLabelHighBitDepth, nil)
	if len(extendedHighBitDepth) != 63 {
		t.Fatalf("extended high-bit-depth vectors=%d want 63", len(extendedHighBitDepth))
	}
	extendedSize := manifest.SelectRemote(SuiteLevelExtended, VectorLabelSize, nil)
	if len(extendedSize) != 99 {
		t.Fatalf("extended size vectors=%d want 99", len(extendedSize))
	}
	extendedSVC := manifest.SelectRemote(SuiteLevelExtended, VectorLabelSVC, nil)
	if len(extendedSVC) != 2 {
		t.Fatalf("extended svc vectors=%d want 2", len(extendedSVC))
	}
}

func TestParseLibaomMD5Digests(t *testing.T) {
	digests, err := ParseLibaomMD5Digests(TagDecoderLibaomQuantizer00, []byte(
		"5a68de997d60afa9083b17fe00f7cdf2  frame0\n"+
			"00112233445566778899aabbccddeeff  frame1\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(digests) != 2 {
		t.Fatalf("digests=%d want 2", len(digests))
	}
	if digests[0].FrameIndex != 0 || digests[1].FrameIndex != 1 {
		t.Fatalf("digests=%+v", digests)
	}
	if _, err := ParseLibaomMD5Digests(TagDecoderLibaomQuantizer00, nil); err != ErrMissingDigest {
		t.Fatalf("empty err=%v want %v", err, ErrMissingDigest)
	}
	if _, err := ParseLibaomMD5Digests(TagDecoderLibaomQuantizer00, []byte("bad")); err == nil {
		t.Fatal("bad md5 err=nil")
	}
}

func TestRemoteManifestValidateFindAndSelect(t *testing.T) {
	content := []byte("stream")
	md5 := []byte("md5")
	manifest := RemoteManifest{
		Name:    "local",
		BaseURL: "https://example.invalid",
		Vectors: []RemoteVector{{
			Tag:    TagDecoderLibaomQuantizer00,
			Kind:   KindDecoder,
			Name:   "quantizer",
			Stream: RemoteFile{Name: "stream.ivf", SHA1: sha1Hex(content)},
			MD5:    RemoteFile{Name: "stream.ivf.md5", SHA1: sha1Hex(md5)},
			Oracle: OracleFrameMD5,
			Levels: SuiteLevelFast | SuiteLevelFull,
			Labels: VectorLabelQuantizer,
		}},
	}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	vector, ok := manifest.FindRemote(TagDecoderLibaomQuantizer00)
	if !ok {
		t.Fatal("missing remote vector")
	}
	if vector.Stream.Name != "stream.ivf" {
		t.Fatalf("vector=%+v", vector)
	}
	selected := manifest.SelectRemote(SuiteLevelFast, VectorLabelQuantizer, nil)
	if len(selected) != 1 || selected[0].Tag != TagDecoderLibaomQuantizer00 {
		t.Fatalf("selected=%+v", selected)
	}
	if selected := manifest.SelectRemote(SuiteLevelRelevant, VectorLabelMotion, nil); len(selected) != 0 {
		t.Fatalf("unexpected selection=%+v", selected)
	}
}

func TestRemoteManifestRejectsInvalidMetadata(t *testing.T) {
	if err := (RemoteManifest{}).Validate(); !errors.Is(err, ErrInvalidRemote) {
		t.Fatalf("empty manifest err=%v want %v", err, ErrInvalidRemote)
	}
	if err := (RemoteFile{Name: "../bad.ivf", SHA1: sha1Hex([]byte("x"))}).Validate(); !errors.Is(err, ErrInvalidRemote) {
		t.Fatalf("bad path err=%v want %v", err, ErrInvalidRemote)
	}
	if err := (RemoteFile{Name: "bad.ivf", SHA1: "bad"}).Validate(); !errors.Is(err, ErrInvalidRemote) {
		t.Fatalf("bad sha1 err=%v want %v", err, ErrInvalidRemote)
	}

	file := RemoteFile{Name: "same.ivf", SHA1: sha1Hex([]byte("x"))}
	dup := RemoteManifest{
		Name:    "dup",
		BaseURL: "https://example.invalid",
		Vectors: []RemoteVector{
			{Tag: TagDecoderLibaomQuantizer00, Kind: KindDecoder, Name: "one", Stream: file, MD5: RemoteFile{Name: "one.md5", SHA1: sha1Hex([]byte("1"))}, Oracle: OracleFrameMD5, Levels: SuiteLevelFast},
			{Tag: TagDecoderLibaomQuantizer00, Kind: KindDecoder, Name: "two", Stream: RemoteFile{Name: "two.ivf", SHA1: sha1Hex([]byte("2"))}, MD5: RemoteFile{Name: "two.md5", SHA1: sha1Hex([]byte("3"))}, Oracle: OracleFrameMD5, Levels: SuiteLevelFast},
		},
	}
	if err := dup.Validate(); !errors.Is(err, ErrDuplicateTag) {
		t.Fatalf("duplicate err=%v want %v", err, ErrDuplicateTag)
	}
}

func TestEnsureRemoteFileDownloadsCachesAndRepairs(t *testing.T) {
	payload := []byte("remote payload")
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Path != "/vector.ivf" {
			t.Fatalf("path=%q want /vector.ivf", r.URL.Path)
		}
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	dir := t.TempDir()
	file := RemoteFile{Name: "vector.ivf", SHA1: sha1Hex(payload)}
	path, err := EnsureRemoteFile(context.Background(), dir, server.URL, file)
	if err != nil {
		t.Fatal(err)
	}
	if path != filepath.Join(dir, file.Name) {
		t.Fatalf("path=%q", path)
	}
	if requests != 1 {
		t.Fatalf("requests=%d want 1", requests)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatalf("payload=%q want %q", got, payload)
	}

	path, err = EnsureRemoteFile(context.Background(), dir, server.URL, file)
	if err != nil {
		t.Fatal(err)
	}
	if requests != 1 {
		t.Fatalf("cached requests=%d want 1", requests)
	}
	if err := os.WriteFile(path, []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := EnsureRemoteFile(context.Background(), dir, server.URL, file); err != nil {
		t.Fatal(err)
	}
	if requests != 2 {
		t.Fatalf("repair requests=%d want 2", requests)
	}
}

func TestLoadRemoteSuiteDownloadsSelectedStreamsAndDigests(t *testing.T) {
	files := map[string][]byte{
		"quantizer.ivf":     []byte("quantizer stream"),
		"quantizer.ivf.md5": []byte("00112233445566778899aabbccddeeff  frame0\nffeeddccbbaa99887766554433221100  frame1\n"),
		"motion.ivf":        []byte("motion stream"),
		"motion.ivf.md5":    []byte("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  frame0\n"),
	}
	requests := make(map[string]int)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := r.URL.Path
		if len(name) > 0 && name[0] == '/' {
			name = name[1:]
		}
		payload, ok := files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		requests[name]++
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	manifest := RemoteManifest{
		Name:    "remote-suite",
		BaseURL: server.URL,
		Vectors: []RemoteVector{
			{
				Tag:    TagDecoderLibaomQuantizer00,
				Kind:   KindDecoder,
				Name:   "quantizer",
				Stream: RemoteFile{Name: "quantizer.ivf", SHA1: sha1Hex(files["quantizer.ivf"])},
				MD5:    RemoteFile{Name: "quantizer.ivf.md5", SHA1: sha1Hex(files["quantizer.ivf.md5"])},
				Oracle: OracleFrameMD5,
				Levels: SuiteLevelFast | SuiteLevelFull,
				Labels: VectorLabelQuantizer,
			},
			{
				Tag:    TagDecoderLibaomMV,
				Kind:   KindDecoder,
				Name:   "motion",
				Stream: RemoteFile{Name: "motion.ivf", SHA1: sha1Hex(files["motion.ivf"])},
				MD5:    RemoteFile{Name: "motion.ivf.md5", SHA1: sha1Hex(files["motion.ivf.md5"])},
				Oracle: OracleFrameMD5,
				Levels: SuiteLevelRelevant | SuiteLevelFull,
				Labels: VectorLabelMotion,
			},
		},
	}

	suite, err := LoadRemoteSuite(context.Background(), manifest, t.TempDir(), SuiteLevelFast, 0)
	if err != nil {
		t.Fatal(err)
	}
	if suite.Name != manifest.Name {
		t.Fatalf("suite name=%q want %q", suite.Name, manifest.Name)
	}
	if err := suite.Manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	if len(suite.Manifest.Vectors) != 1 || string(suite.Manifest.Vectors[0].Input) != "quantizer stream" {
		t.Fatalf("vectors=%+v", suite.Manifest.Vectors)
	}
	if len(suite.Manifest.Digests) != 2 || suite.Manifest.Digests[1].FrameIndex != 1 {
		t.Fatalf("digests=%+v", suite.Manifest.Digests)
	}
	if requests["quantizer.ivf"] != 1 || requests["quantizer.ivf.md5"] != 1 ||
		requests["motion.ivf"] != 0 || requests["motion.ivf.md5"] != 0 {
		t.Fatalf("requests=%+v", requests)
	}

	motion, err := LoadRemoteSuite(context.Background(), manifest, t.TempDir(), SuiteLevelRelevant, VectorLabelMotion)
	if err != nil {
		t.Fatal(err)
	}
	if len(motion.Manifest.Vectors) != 1 || motion.Manifest.Vectors[0].Tag != TagDecoderLibaomMV {
		t.Fatalf("motion vectors=%+v", motion.Manifest.Vectors)
	}
	if digest, ok := motion.Manifest.FindDigest(TagDecoderLibaomMV, 0); !ok || digest.MD5[0] != 0xaa {
		t.Fatalf("motion digest=%x ok=%v", digest.MD5, ok)
	}
}

func TestLoadRemoteSuiteRejectsEmptySelection(t *testing.T) {
	manifest := RemoteManifest{
		Name:    "empty-selection",
		BaseURL: "https://example.invalid",
		Vectors: []RemoteVector{{
			Tag:    TagDecoderLibaomQuantizer00,
			Kind:   KindDecoder,
			Name:   "quantizer",
			Stream: RemoteFile{Name: "quantizer.ivf", SHA1: sha1Hex([]byte("stream"))},
			MD5:    RemoteFile{Name: "quantizer.ivf.md5", SHA1: sha1Hex([]byte("md5"))},
			Oracle: OracleFrameMD5,
			Levels: SuiteLevelFast,
			Labels: VectorLabelQuantizer,
		}},
	}
	_, err := LoadRemoteSuite(context.Background(), manifest, t.TempDir(), SuiteLevelRelevant, VectorLabelMotion)
	if !errors.Is(err, ErrMissingVector) {
		t.Fatalf("err=%v want %v", err, ErrMissingVector)
	}
}

func TestEnsureRemoteFileRejectsChecksumMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("wrong"))
	}))
	defer server.Close()

	file := RemoteFile{Name: "vector.ivf", SHA1: sha1Hex([]byte("right"))}
	if _, err := EnsureRemoteFile(context.Background(), t.TempDir(), server.URL, file); !errors.Is(err, ErrChecksum) {
		t.Fatalf("err=%v want %v", err, ErrChecksum)
	}
}

func sha1Hex(data []byte) string {
	sum := sha1.Sum(data)
	return fmt.Sprintf("%x", sum)
}

func TestManifestFindAndValidate(t *testing.T) {
	manifest := Manifest{Vectors: []Vector{
		{Tag: TagOBULowOverheadTemporalDelimiter, Kind: KindOBU, Name: "temporal delimiter", Input: []byte{0x12}, Want: []byte{0x12}},
		{Tag: TagRTPPayloadSingleOBU, Kind: KindRTP, Name: "single obu", Input: []byte{0x10, 0xaa}, Want: []byte{0xaa}},
	}}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	vector, ok := manifest.Find(TagRTPPayloadSingleOBU)
	if !ok {
		t.Fatal("missing vector")
	}
	if vector.Kind != KindRTP || vector.Name != "single obu" {
		t.Fatalf("vector=%+v", vector)
	}
}

func TestManifestFindDigest(t *testing.T) {
	md5, err := ParseMD5Hex([]byte("5a68de997d60afa9083b17fe00f7cdf2"))
	if err != nil {
		t.Fatal(err)
	}
	manifest := Manifest{Digests: []FrameDigest{{
		Tag:        TagIVFSingleFrameAV1,
		FrameIndex: 3,
		MD5:        md5,
	}}}
	if err := manifest.Validate(); err != nil {
		t.Fatal(err)
	}
	digest, ok := manifest.FindDigest(TagIVFSingleFrameAV1, 3)
	if !ok {
		t.Fatal("missing digest")
	}
	if digest.MD5 != md5 {
		t.Fatalf("digest=%x want %x", digest.MD5, md5)
	}
}

func TestManifestRejectsInvalidTags(t *testing.T) {
	err := (Manifest{Vectors: []Vector{{}}}).Validate()
	if !errors.Is(err, ErrInvalidTag) {
		t.Fatalf("err=%v want %v", err, ErrInvalidTag)
	}
	err = (Manifest{Vectors: []Vector{
		{Tag: TagRTPPayloadSingleOBU},
		{Tag: TagRTPPayloadSingleOBU},
	}}).Validate()
	if !errors.Is(err, ErrDuplicateTag) {
		t.Fatalf("err=%v want %v", err, ErrDuplicateTag)
	}
	err = (Manifest{Digests: []FrameDigest{{}}}).Validate()
	if !errors.Is(err, ErrInvalidTag) {
		t.Fatalf("digest err=%v want %v", err, ErrInvalidTag)
	}
	err = (Manifest{Digests: []FrameDigest{
		{Tag: TagIVFSingleFrameAV1, FrameIndex: 0},
		{Tag: TagIVFSingleFrameAV1, FrameIndex: 0},
	}}).Validate()
	if !errors.Is(err, ErrDuplicateTag) {
		t.Fatalf("duplicate digest err=%v want %v", err, ErrDuplicateTag)
	}
}

func TestParseMD5Hex(t *testing.T) {
	md5, err := ParseMD5Hex([]byte("5A68DE997D60AFA9083B17FE00F7CDF2 trailing"))
	if err != nil {
		t.Fatal(err)
	}
	want := MD5{0x5a, 0x68, 0xde, 0x99, 0x7d, 0x60, 0xaf, 0xa9, 0x08, 0x3b, 0x17, 0xfe, 0x00, 0xf7, 0xcd, 0xf2}
	if md5 != want {
		t.Fatalf("md5=%x want %x", md5, want)
	}
	if _, err := ParseMD5Hex([]byte("short")); !errors.Is(err, ErrInvalidMD5) {
		t.Fatalf("short err=%v want %v", err, ErrInvalidMD5)
	}
	if _, err := ParseMD5Hex([]byte("zz68de997d60afa9083b17fe00f7cdf2")); !errors.Is(err, ErrInvalidMD5) {
		t.Fatalf("invalid err=%v want %v", err, ErrInvalidMD5)
	}
}
