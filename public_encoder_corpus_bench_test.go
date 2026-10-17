package goav1_test

import (
	"crypto/md5"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"hash"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

const (
	publicEncoderP720Width          = 1280
	publicEncoderP720Height         = 720
	publicEncoderP720Frames         = 48
	publicEncoderP720FrameBytes     = publicEncoderP720Width * publicEncoderP720Height * 3 / 2
	publicEncoderP720RawEnv         = "GOAV1_ENCODER_P720_I420"
	publicEncoderP720ExpectedTUEnv  = "GOAV1_ENCODER_P720_EXPECT_BITSTREAM_SHA256"
	publicEncoderP720ExpectedRecEnv = "GOAV1_ENCODER_P720_EXPECT_RECON_SHA256"
)

var publicEncoderP720SampleSink uint64

type publicEncoderP720Fixture struct {
	frames []av1.I420Frame
	rawSHA string
	rawMD5 string
}

type publicEncoderP720Digest struct {
	bitstreamSHA string
	reconSHA     string
	bitstreamLen int64
	frames       int
}

// TestPublicEncoderP720CorpusParity is opt-in because its decoded corpus input
// is intentionally kept outside the repository. With GOAV1_ENCODER_P720_I420
// set, it fails on missing, malformed, or mismatched input and logs exact
// stream/reconstruction digests for comparison between immutable binaries.
func TestPublicEncoderP720CorpusParity(t *testing.T) {
	fixture := loadPublicEncoderP720Fixture(t, false)
	if fixture == nil {
		return
	}
	got, err := encodePublicEncoderP720Digest(fixture.frames)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("public_encoder_p720 frames=%d raw_sha256=%s source_stream_md5=%s bitstream_sha256=%s reconstruction_sha256=%s bitstream_bytes=%d",
		got.frames, fixture.rawSHA, fixture.rawMD5, got.bitstreamSHA, got.reconSHA, got.bitstreamLen)
	checkPublicEncoderP720Expected(t, publicEncoderP720ExpectedTUEnv, got.bitstreamSHA)
	checkPublicEncoderP720Expected(t, publicEncoderP720ExpectedRecEnv, got.reconSHA)
}

// BenchmarkPublicEncoderP720CorpusStream measures one complete 48-frame
// persistent-encoder stream per operation. Each operation constructs and
// prewarms the public encoder, emits the keyframe and all 47 inter frames,
// observes each returned temporal unit, then closes the encoder to join pending
// filter work and release workers. Input loading, framing, and MD5 validation
// are outside the timer. This QIndex-80, one-thread corpus case is distinct
// from the CBR Pan benchmark.
func BenchmarkPublicEncoderP720CorpusStream(b *testing.B) {
	fixture := loadPublicEncoderP720Fixture(b, true)
	b.ReportAllocs()
	b.ResetTimer()
	var lastBytes int64
	for b.Loop() {
		observed, bitstreamBytes, err := encodePublicEncoderP720Observed(fixture.frames)
		if err != nil {
			b.Fatal(err)
		}
		publicEncoderP720SampleSink = publicEncoderP720SampleSink*0x9e3779b185ebca87 + observed
		lastBytes = bitstreamBytes
	}
	b.ReportMetric(publicEncoderP720Frames, "frames/op")
	b.ReportMetric(float64(lastBytes), "bitstream_B/stream")
}

func loadPublicEncoderP720Fixture(tb testing.TB, required bool) *publicEncoderP720Fixture {
	tb.Helper()
	rawPath := os.Getenv(publicEncoderP720RawEnv)
	if rawPath == "" {
		if required {
			tb.Fatalf("set %s to the decoded p720_inter_q32 I420 fixture", publicEncoderP720RawEnv)
		}
		tb.Skipf("set %s to run the public p720 encoder parity preflight", publicEncoderP720RawEnv)
		return nil
	}
	raw, err := os.ReadFile(rawPath)
	if err != nil {
		tb.Fatalf("read %s=%q: %v", publicEncoderP720RawEnv, rawPath, err)
	}
	wantBytes := publicEncoderP720FrameBytes
	if len(raw) != wantBytes*publicEncoderP720Frames {
		tb.Fatalf("%s has %d bytes; want exactly %d bytes for %d packed I420 frames at %dx%d",
			rawPath, len(raw), wantBytes*publicEncoderP720Frames, publicEncoderP720Frames, publicEncoderP720Width, publicEncoderP720Height)
	}

	corpusDir := os.Getenv("GOAV1_BENCH_CORPUS_DIR")
	if corpusDir == "" {
		_, filename, _, ok := runtime.Caller(0)
		if !ok {
			tb.Fatal("runtime.Caller failed while locating p720 MD5 sidecar")
		}
		corpusDir = filepath.Join(filepath.Dir(filename), "testdata", "benchcorpus")
	}
	sidecarPath := filepath.Join(corpusDir, "p720_inter_q32.md5")
	sidecar, err := os.ReadFile(sidecarPath)
	if err != nil {
		tb.Fatalf("read p720 stream-MD5 sidecar %q: %v", sidecarPath, err)
	}
	wantMD5 := strings.TrimSpace(string(sidecar))
	if len(wantMD5) != md5.Size*2 {
		tb.Fatalf("p720 sidecar %q must contain one stream MD5 token, got %q", sidecarPath, wantMD5)
	}
	if _, err := hex.DecodeString(wantMD5); err != nil {
		tb.Fatalf("parse p720 stream-MD5 sidecar %q: %v", sidecarPath, err)
	}
	gotMD5 := fmt.Sprintf("%x", md5.Sum(raw))
	if gotMD5 != wantMD5 {
		tb.Fatalf("decoded raw I420 stream MD5=%s, sidecar %s has %s", gotMD5, sidecarPath, wantMD5)
	}

	frames := make([]av1.I420Frame, publicEncoderP720Frames)
	yBytes := publicEncoderP720Width * publicEncoderP720Height
	cBytes := yBytes / 4
	for i := range frames {
		base := i * wantBytes
		y := raw[base : base+yBytes]
		u := raw[base+yBytes : base+yBytes+cBytes]
		v := raw[base+yBytes+cBytes : base+wantBytes]
		frames[i] = av1.I420Frame{
			Y: y, U: u, V: v,
			YStride: publicEncoderP720Width, ChromaStride: publicEncoderP720Width / 2,
			Width: publicEncoderP720Width, Height: publicEncoderP720Height,
		}
	}
	rawSHA := fmt.Sprintf("%x", sha256.Sum256(raw))
	return &publicEncoderP720Fixture{frames: frames, rawSHA: rawSHA, rawMD5: gotMD5}
}

func publicEncoderP720Config() av1.VideoEncoderConfig {
	return av1.VideoEncoderConfig{
		Width: publicEncoderP720Width, Height: publicEncoderP720Height,
		QIndex: 80, MaxThreads: 1, DisableSceneCutKeyframes: true,
	}
}

func encodePublicEncoderP720Digest(frames []av1.I420Frame) (publicEncoderP720Digest, error) {
	enc, err := av1.NewVideoEncoder(publicEncoderP720Config())
	if err != nil {
		return publicEncoderP720Digest{}, fmt.Errorf("NewVideoEncoder: %w", err)
	}
	streamHash := sha256.New()
	reconHash := sha256.New()
	var result publicEncoderP720Digest
	for i, src := range frames {
		encoded, encodeErr := enc.Encode(src, i == 0)
		if encodeErr != nil {
			_ = enc.Close()
			return publicEncoderP720Digest{}, fmt.Errorf("Encode frame %d: %w", i, encodeErr)
		}
		if encoded.Keyframe != (i == 0) {
			_ = enc.Close()
			return publicEncoderP720Digest{}, fmt.Errorf("frame %d keyframe=%v, want %v", i, encoded.Keyframe, i == 0)
		}
		if len(encoded.Data) == 0 {
			_ = enc.Close()
			return publicEncoderP720Digest{}, fmt.Errorf("frame %d returned an empty temporal unit", i)
		}
		_, _ = streamHash.Write(encoded.Data)
		result.bitstreamLen += int64(len(encoded.Data))
		recon := enc.Reconstruction()
		if recon.Width != publicEncoderP720Width || recon.Height != publicEncoderP720Height {
			_ = enc.Close()
			return publicEncoderP720Digest{}, fmt.Errorf("frame %d reconstruction is %dx%d, want %dx%d", i, recon.Width, recon.Height, publicEncoderP720Width, publicEncoderP720Height)
		}
		if err := writePublicEncoderP720VisibleFrame(reconHash, recon); err != nil {
			_ = enc.Close()
			return publicEncoderP720Digest{}, fmt.Errorf("frame %d reconstruction: %w", i, err)
		}
		result.frames++
	}
	if result.frames != publicEncoderP720Frames {
		_ = enc.Close()
		return publicEncoderP720Digest{}, fmt.Errorf("encoded %d frames, want %d", result.frames, publicEncoderP720Frames)
	}
	if err := enc.Close(); err != nil {
		return publicEncoderP720Digest{}, fmt.Errorf("VideoEncoder.Close: %w", err)
	}
	result.bitstreamSHA = fmt.Sprintf("%x", streamHash.Sum(nil))
	result.reconSHA = fmt.Sprintf("%x", reconHash.Sum(nil))
	return result, nil
}

func encodePublicEncoderP720Observed(frames []av1.I420Frame) (uint64, int64, error) {
	enc, err := av1.NewVideoEncoder(publicEncoderP720Config())
	if err != nil {
		return 0, 0, fmt.Errorf("NewVideoEncoder: %w", err)
	}
	var observed uint64
	var bitstreamBytes int64
	for i, src := range frames {
		encoded, encodeErr := enc.Encode(src, i == 0)
		if encodeErr != nil {
			_ = enc.Close()
			return 0, 0, fmt.Errorf("Encode frame %d: %w", i, encodeErr)
		}
		if encoded.Keyframe != (i == 0) || len(encoded.Data) == 0 {
			_ = enc.Close()
			return 0, 0, fmt.Errorf("frame %d returned keyframe=%v, temporal-unit bytes=%d", i, encoded.Keyframe, len(encoded.Data))
		}
		bitstreamBytes += int64(len(encoded.Data))
		observed = observed*0x100000001b3 + uint64(len(encoded.Data))
		observed = observed*0x100000001b3 + uint64(encoded.Data[0])<<8 + uint64(encoded.Data[len(encoded.Data)-1])
	}
	if len(frames) != publicEncoderP720Frames {
		_ = enc.Close()
		return 0, 0, fmt.Errorf("got %d frames, want %d", len(frames), publicEncoderP720Frames)
	}
	if err := enc.Close(); err != nil {
		return 0, 0, fmt.Errorf("VideoEncoder.Close: %w", err)
	}
	return observed, bitstreamBytes, nil
}

func writePublicEncoderP720VisibleFrame(h hash.Hash, frame av1.I420Frame) error {
	if err := writePublicEncoderP720Plane(h, frame.Y, frame.YStride, publicEncoderP720Width, publicEncoderP720Height); err != nil {
		return fmt.Errorf("Y plane: %w", err)
	}
	if err := writePublicEncoderP720Plane(h, frame.U, frame.ChromaStride, publicEncoderP720Width/2, publicEncoderP720Height/2); err != nil {
		return fmt.Errorf("U plane: %w", err)
	}
	return writePublicEncoderP720Plane(h, frame.V, frame.ChromaStride, publicEncoderP720Width/2, publicEncoderP720Height/2)
}

func writePublicEncoderP720Plane(h hash.Hash, plane []byte, stride, width, height int) error {
	if stride < width || height <= 0 || len(plane) < (height-1)*stride+width {
		return fmt.Errorf("plane bytes=%d stride=%d visible=%dx%d", len(plane), stride, width, height)
	}
	for y := range height {
		if _, err := h.Write(plane[y*stride : y*stride+width]); err != nil {
			return err
		}
	}
	return nil
}

func checkPublicEncoderP720Expected(t *testing.T, envName, got string) {
	t.Helper()
	want := strings.ToLower(strings.TrimSpace(os.Getenv(envName)))
	if want == "" {
		return
	}
	if len(want) != sha256.Size*2 {
		t.Fatalf("%s must be a 64-character SHA-256, got %q", envName, want)
	}
	if _, err := hex.DecodeString(want); err != nil {
		t.Fatalf("parse %s: %v", envName, err)
	}
	if got != want {
		t.Fatalf("digest mismatch for %s: got %s, want %s", envName, got, want)
	}
}
