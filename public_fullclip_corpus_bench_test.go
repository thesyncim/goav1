package goav1_test

import (
	"bufio"
	"crypto/md5"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	av1 "github.com/thesyncim/goav1"
)

const (
	publicFullClipExpectedClips  = 18
	publicFullClipExpectedFrames = 864
	publicFullClipWarmups        = 1
	publicFullClipSamples        = 9
)

var publicFullClipSampleSink uint64

type publicFullClipOracle struct {
	path      string
	streamMD5 [16]byte
	frameMD5s [][16]byte
	perFrame  bool
}

type publicFullClip struct {
	name   string
	data   []byte
	oracle publicFullClipOracle
}

type publicFullClipResult struct {
	frames    int
	streamMD5 [16]byte
	frameMD5s [][16]byte
	observed  uint64
}

// TestPublicDecoderFullClipCorpusMD5 checks all public Decoder output against
// the independent aomdec/dav1d MD5 sidecars. Set GOAV1_PUBLIC_FULLCLIP=1 to run
// it. The sampler below repeats this check before collecting any timings.
func TestPublicDecoderFullClipCorpusMD5(t *testing.T) {
	if os.Getenv("GOAV1_PUBLIC_FULLCLIP") != "1" {
		t.Skip("set GOAV1_PUBLIC_FULLCLIP=1 to verify the full public decoder corpus")
	}
	clips := loadPublicFullClipCorpus(t)
	totalFrames := verifyPublicFullClipCorpus(t, clips)
	if totalFrames != publicFullClipExpectedFrames {
		t.Fatalf("public corpus produced %d frames, want %d", totalFrames, publicFullClipExpectedFrames)
	}
	t.Logf("public decoder MD5 preflight passed: clips=%d frames=%d workers=1", len(clips), totalFrames)
}

// TestPublicDecoderFullClipCorpusSamples is an opt-in full-clip sampling
// harness. It verifies every sidecar first, then collects one warmup and nine
// whole-clip samples for each cold IVF, cold pre-parsed payload, and warm
// Reset+DecodeNext path. Input reads, IVF payload extraction, and MD5
// verification are outside the timed sections.
func TestPublicDecoderFullClipCorpusSamples(t *testing.T) {
	if os.Getenv("GOAV1_PUBLIC_FULLCLIP_SAMPLES") != "1" {
		t.Skip("set GOAV1_PUBLIC_FULLCLIP_SAMPLES=1 to sample full-clip public decoding")
	}
	clips := loadPublicFullClipCorpus(t)
	totalFrames := verifyPublicFullClipCorpus(t, clips)
	if totalFrames != publicFullClipExpectedFrames {
		t.Fatalf("public corpus produced %d frames, want %d", totalFrames, publicFullClipExpectedFrames)
	}
	t.Logf("public decoder MD5 preflight passed: clips=%d frames=%d workers=1", len(clips), totalFrames)
	t.Logf("sample protocol: workers=1 warmups=%d samples=%d; cold_ivf=NewDecoderFromIVF+DecodeNext+Close; cold_payload=NewDecoder(payloads)+DecodeNext+Close (IVF parse/copy excluded); warm=Reset+DecodeNext; MD5 preflight excluded", publicFullClipWarmups, publicFullClipSamples)

	for _, clip := range clips {
		sinkBefore := publicFullClipSampleSink
		cold, err := samplePublicFullClipCold(clip)
		if err != nil {
			t.Fatalf("%s cold samples: %v", clip.name, err)
		}
		coldPayload, err := samplePublicFullClipColdPayload(clip)
		if err != nil {
			t.Fatalf("%s cold pre-parsed payload samples: %v", clip.name, err)
		}
		if coldPayload.frames != cold.frames {
			t.Fatalf("%s NewDecoder(payloads) emitted %d frames, cold IVF path emitted %d", clip.name, coldPayload.frames, cold.frames)
		}
		warm, err := samplePublicFullClipWarm(clip)
		if err != nil {
			t.Fatalf("%s warm samples: %v", clip.name, err)
		}
		t.Logf("public_fullclip clip=%s frames=%d cold_fresh_decoder_median=%s cold_samples=%s cold_payload_decoder_median=%s cold_payload_samples=%s warm_reset_decode_median=%s warm_samples=%s sink=%d",
			clip.name, cold.frames, medianPublicFullClipDuration(cold.samples), formatPublicFullClipDurations(cold.samples),
			medianPublicFullClipDuration(coldPayload.samples), formatPublicFullClipDurations(coldPayload.samples),
			medianPublicFullClipDuration(warm.samples), formatPublicFullClipDurations(warm.samples), publicFullClipSampleSink-sinkBefore)
	}
}

// BenchmarkPublicDecoderFullClip exposes cold-IVF, cold-preparsed-payload, and
// warm sub-benchmarks for each corpus clip. Select a single clip/path with
// -bench, for example
// `-bench 'BenchmarkPublicDecoderFullClip/p720_inter_q32/cold_preparsed_payload_decoder'`.
// Run TestPublicDecoderFullClipCorpusMD5 separately first to keep MD5 work out
// of CPU profiles and benchmark timing.
func BenchmarkPublicDecoderFullClip(b *testing.B) {
	clips := loadPublicFullClipCorpus(b)
	for _, clip := range clips {
		clip := clip
		b.Run(clip.name, func(b *testing.B) {
			b.Run("cold_fresh_decoder", func(b *testing.B) {
				warmup, err := decodePublicFullClipFresh(clip.data, false)
				if err != nil {
					b.Fatalf("warmup decode: %v", err)
				}
				if warmup.frames == 0 {
					b.Fatal("warmup produced no visible frames")
				}
				wantFrames := warmup.frames
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					result, err := decodePublicFullClipFresh(clip.data, false)
					if err != nil {
						b.Fatalf("decode: %v", err)
					}
					if result.frames != wantFrames {
						b.Fatalf("decoded %d frames, want %d", result.frames, wantFrames)
					}
					publicFullClipSampleSink += result.observed
				}
				b.ReportMetric(float64(wantFrames), "frames/op")
			})
			b.Run("warm_reset_decode", func(b *testing.B) {
				dec, err := av1.NewDecoderFromIVF(clip.data, av1.WithWorkers(1))
				if err != nil {
					b.Fatalf("NewDecoderFromIVF: %v", err)
				}
				b.Cleanup(dec.Close)
				warmup, err := decodePublicFullClipAfterReset(dec)
				if err != nil {
					b.Fatalf("warmup Reset+DecodeNext: %v", err)
				}
				if warmup.frames == 0 {
					b.Fatal("warmup produced no visible frames")
				}
				wantFrames := warmup.frames
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					result, err := decodePublicFullClipAfterReset(dec)
					if err != nil {
						b.Fatalf("Reset+DecodeNext: %v", err)
					}
					if result.frames != wantFrames {
						b.Fatalf("decoded %d frames, want %d", result.frames, wantFrames)
					}
					publicFullClipSampleSink += result.observed
				}
				b.ReportMetric(float64(wantFrames), "frames/op")
			})
			b.Run("cold_preparsed_payload_decoder", func(b *testing.B) {
				payloads, err := publicFullClipPayloads(clip.data)
				if err != nil {
					b.Fatalf("parse IVF payloads: %v", err)
				}
				warmup, err := decodePublicFullClipPayloads(payloads, false)
				if err != nil {
					b.Fatalf("warmup NewDecoder(payloads): %v", err)
				}
				if warmup.frames == 0 {
					b.Fatal("warmup produced no visible frames")
				}
				wantFrames := warmup.frames
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					result, err := decodePublicFullClipPayloads(payloads, false)
					if err != nil {
						b.Fatalf("NewDecoder(payloads): %v", err)
					}
					if result.frames != wantFrames {
						b.Fatalf("decoded %d frames, want %d", result.frames, wantFrames)
					}
					publicFullClipSampleSink += result.observed
				}
				b.ReportMetric(float64(wantFrames), "frames/op")
			})
		})
	}
}

func loadPublicFullClipCorpus(t testing.TB) []publicFullClip {
	t.Helper()
	dir := os.Getenv("GOAV1_BENCH_CORPUS_DIR")
	if dir == "" {
		_, filename, _, ok := runtime.Caller(0)
		if !ok {
			t.Fatal("runtime.Caller failed")
		}
		dir = filepath.Join(filepath.Dir(filename), "testdata", "benchcorpus")
	}
	paths, err := filepath.Glob(filepath.Join(dir, "*.ivf"))
	if err != nil {
		t.Fatalf("glob public corpus %q: %v", dir, err)
	}
	sort.Strings(paths)
	if len(paths) != publicFullClipExpectedClips {
		t.Fatalf("public corpus %q has %d IVF clips, want %d", dir, len(paths), publicFullClipExpectedClips)
	}
	clips := make([]publicFullClip, 0, len(paths))
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read public corpus clip %s: %v", path, err)
		}
		oracle, err := loadPublicFullClipOracle(path)
		if err != nil {
			t.Fatalf("load public corpus sidecar for %s: %v", path, err)
		}
		clips = append(clips, publicFullClip{
			name:   strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
			data:   data,
			oracle: oracle,
		})
	}
	return clips
}

func loadPublicFullClipOracle(ivfPath string) (publicFullClipOracle, error) {
	stem := strings.TrimSuffix(ivfPath, filepath.Ext(ivfPath))
	paths := []string{stem + ".md5", ivfPath + ".md5", stem + ".framemd5", ivfPath + ".framemd5"}
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return publicFullClipOracle{}, fmt.Errorf("read %s: %w", path, err)
		}
		digests, err := parsePublicFullClipMD5Tokens(raw)
		if err != nil {
			return publicFullClipOracle{}, fmt.Errorf("parse %s: %w", path, err)
		}
		oracle := publicFullClipOracle{path: path}
		if strings.EqualFold(filepath.Ext(path), ".framemd5") || len(digests) > 1 {
			oracle.perFrame = true
			oracle.frameMD5s = digests
		} else {
			oracle.streamMD5 = digests[0]
		}
		return oracle, nil
	}
	return publicFullClipOracle{}, fmt.Errorf("no supported MD5 sidecar found; tried %s", strings.Join(paths, ", "))
}

func parsePublicFullClipMD5Tokens(src []byte) ([][16]byte, error) {
	var digests [][16]byte
	scanner := bufio.NewScanner(strings.NewReader(string(src)))
	for scanner.Scan() {
		line := scanner.Bytes()
		trimmed := strings.TrimSpace(string(line))
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		for i := 0; i+md5.Size*2 <= len(line); i++ {
			if i > 0 && isPublicFullClipHex(line[i-1]) {
				continue
			}
			end := i + md5.Size*2
			if end < len(line) && isPublicFullClipHex(line[end]) {
				continue
			}
			var digest [16]byte
			if _, err := hex.Decode(digest[:], line[i:end]); err != nil {
				continue
			}
			digests = append(digests, digest)
			i = end - 1
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(digests) == 0 {
		return nil, errors.New("no 32-character MD5 token found")
	}
	return digests, nil
}

func isPublicFullClipHex(b byte) bool {
	return b >= '0' && b <= '9' || b >= 'a' && b <= 'f' || b >= 'A' && b <= 'F'
}

func verifyPublicFullClipCorpus(t *testing.T, clips []publicFullClip) int {
	t.Helper()
	totalFrames := 0
	for _, clip := range clips {
		result, err := decodePublicFullClipFresh(clip.data, true)
		if err != nil {
			t.Fatalf("%s public decode: %v", clip.name, err)
		}
		if err := comparePublicFullClipOracle(result, clip.oracle); err != nil {
			t.Fatalf("%s sidecar %s: %v", clip.name, clip.oracle.path, err)
		}
		t.Logf("public MD5 verified: clip=%s frames=%d digest=%x", clip.name, result.frames, result.streamMD5)

		payloads, err := publicFullClipPayloads(clip.data)
		if err != nil {
			t.Fatalf("%s IVF payload parse: %v", clip.name, err)
		}
		payloadResult, err := decodePublicFullClipPayloads(payloads, true)
		if err != nil {
			t.Fatalf("%s NewDecoder payload preflight: %v", clip.name, err)
		}
		if payloadResult.frames != result.frames {
			t.Fatalf("%s NewDecoder(payloads) emitted %d frames, NewDecoderFromIVF emitted %d",
				clip.name, payloadResult.frames, result.frames)
		}
		if err := comparePublicFullClipOracle(payloadResult, clip.oracle); err != nil {
			t.Fatalf("%s NewDecoder(payloads) sidecar %s: %v", clip.name, clip.oracle.path, err)
		}
		t.Logf("public payload constructor MD5 verified: clip=%s frames=%d digest=%x",
			clip.name, payloadResult.frames, payloadResult.streamMD5)
		totalFrames += result.frames
	}
	return totalFrames
}

func comparePublicFullClipOracle(result publicFullClipResult, oracle publicFullClipOracle) error {
	if oracle.perFrame {
		if len(result.frameMD5s) != len(oracle.frameMD5s) {
			return fmt.Errorf("decoded %d frame digests, sidecar has %d", len(result.frameMD5s), len(oracle.frameMD5s))
		}
		for i := range oracle.frameMD5s {
			if result.frameMD5s[i] != oracle.frameMD5s[i] {
				return fmt.Errorf("frame %d MD5=%x, want %x", i, result.frameMD5s[i], oracle.frameMD5s[i])
			}
		}
		return nil
	}
	if result.streamMD5 != oracle.streamMD5 {
		return fmt.Errorf("stream MD5=%x, want %x", result.streamMD5, oracle.streamMD5)
	}
	return nil
}

func decodePublicFullClipFresh(data []byte, hashFrames bool) (publicFullClipResult, error) {
	dec, err := av1.NewDecoderFromIVF(data, av1.WithWorkers(1))
	if err != nil {
		return publicFullClipResult{}, fmt.Errorf("NewDecoderFromIVF: %w", err)
	}
	result, decodeErr := decodePublicFullClip(dec, hashFrames)
	dec.Close()
	return result, decodeErr
}

// publicFullClipPayloads copies each IVF frame payload before a timed
// NewDecoder(payloads) run. NewIVFIterator returns views into the IVF bytes.
func publicFullClipPayloads(ivfBytes []byte) ([][]byte, error) {
	it, err := av1.NewIVFIterator(ivfBytes)
	if err != nil {
		return nil, fmt.Errorf("NewIVFIterator: %w", err)
	}
	payloads := make([][]byte, 0)
	for {
		frame, ok, err := it.Next()
		if err != nil {
			return nil, fmt.Errorf("IVFIterator.Next: %w", err)
		}
		if !ok {
			break
		}
		payloads = append(payloads, append([]byte(nil), frame.Payload...))
	}
	if len(payloads) == 0 {
		return nil, errors.New("IVF stream produced no frame payloads")
	}
	return payloads, nil
}

func TestPublicFullClipPayloadsRejectsInvalidIVF(t *testing.T) {
	_, err := publicFullClipPayloads([]byte("DKIF"))
	if !errors.Is(err, av1.ErrIVFShortHeader) {
		t.Fatalf("publicFullClipPayloads error=%v, want %v", err, av1.ErrIVFShortHeader)
	}
}

func TestDecodePublicFullClipPayloadsPropagatesInvalidPayload(t *testing.T) {
	_, err := decodePublicFullClipPayloads([][]byte{{0xff}}, false)
	if err == nil {
		t.Fatal("decodePublicFullClipPayloads accepted an invalid AV1 payload")
	}
}

func decodePublicFullClipPayloads(payloads [][]byte, hashFrames bool) (publicFullClipResult, error) {
	dec, err := av1.NewDecoder(payloads, av1.WithWorkers(1))
	if err != nil {
		return publicFullClipResult{}, fmt.Errorf("NewDecoder: %w", err)
	}
	defer dec.Close()
	return decodePublicFullClip(dec, hashFrames)
}

func decodePublicFullClip(dec *av1.Decoder, hashFrames bool) (publicFullClipResult, error) {
	var streamHash hash.Hash
	if hashFrames {
		streamHash = md5.New()
	}
	result := publicFullClipResult{}
	for {
		frames, ok, err := dec.DecodeNext()
		if err != nil {
			return publicFullClipResult{}, err
		}
		if !ok {
			break
		}
		for _, f := range frames {
			if f == nil {
				continue
			}
			result.frames++
			result.observed += observePublicFullClipFrame(f)
			if hashFrames {
				frameHash := md5.New()
				if err := writePublicFullClipFrameMD5(frameHash, f); err != nil {
					return publicFullClipResult{}, err
				}
				var frameDigest [16]byte
				copy(frameDigest[:], frameHash.Sum(nil))
				result.frameMD5s = append(result.frameMD5s, frameDigest)
				if err := writePublicFullClipFrameMD5(streamHash, f); err != nil {
					return publicFullClipResult{}, err
				}
			}
		}
	}
	if hashFrames {
		copy(result.streamMD5[:], streamHash.Sum(nil))
	}
	return result, nil
}

func writePublicFullClipFrameMD5(h hash.Hash, f *av1.Frame) error {
	bytesPerSample := f.Layout.BytesPerSample
	if bytesPerSample != 1 && bytesPerSample != 2 {
		return fmt.Errorf("unsupported bytes per sample: %d", bytesPerSample)
	}
	if err := writePublicFullClipPlaneMD5(h, f.Y, bytesPerSample); err != nil {
		return err
	}
	if f.Format.MonoChrome {
		return writePublicFullClipNeutralChromaMD5(h, f.Format, bytesPerSample)
	}
	if err := writePublicFullClipPlaneMD5(h, f.U, bytesPerSample); err != nil {
		return err
	}
	return writePublicFullClipPlaneMD5(h, f.V, bytesPerSample)
}

func writePublicFullClipPlaneMD5(h hash.Hash, plane av1.FramePlane, bytesPerSample int) error {
	if plane.Width < 0 || plane.Height < 0 || plane.Stride < 0 {
		return errors.New("invalid plane dimensions")
	}
	if plane.Width == 0 || plane.Height == 0 {
		if len(plane.Pix) != 0 {
			return errors.New("empty plane has pixel data")
		}
		return nil
	}
	rowBytes := plane.Width * bytesPerSample
	if rowBytes/bytesPerSample != plane.Width || plane.Stride < rowBytes {
		return errors.New("invalid visible plane stride")
	}
	last := (plane.Height-1)*plane.Stride + rowBytes
	if last < rowBytes || last > len(plane.Pix) {
		return errors.New("visible plane rows exceed pixel buffer")
	}
	for y := 0; y < plane.Height; y++ {
		row := y * plane.Stride
		if _, err := h.Write(plane.Pix[row : row+rowBytes]); err != nil {
			return err
		}
	}
	return nil
}

func writePublicFullClipNeutralChromaMD5(h hash.Hash, format av1.FrameFormat, bytesPerSample int) error {
	width, height := format.Width, format.Height
	if format.SubsamplingX {
		width = (width + 1) >> 1
	}
	if format.SubsamplingY {
		height = (height + 1) >> 1
	}
	rowBytes := width * bytesPerSample
	if width <= 0 || height <= 0 || rowBytes/bytesPerSample != width {
		return errors.New("invalid monochrome chroma dimensions")
	}
	neutral := uint16(1) << (format.BitDepth - 1)
	var row [4096]byte
	for i := 0; i+bytesPerSample <= len(row); i += bytesPerSample {
		row[i] = byte(neutral)
		if bytesPerSample == 2 {
			row[i+1] = byte(neutral >> 8)
		}
	}
	for range 2 {
		for y := 0; y < height; y++ {
			remaining := rowBytes
			for remaining > 0 {
				chunk := min(remaining, len(row))
				if _, err := h.Write(row[:chunk]); err != nil {
					return err
				}
				remaining -= chunk
			}
		}
	}
	return nil
}

func observePublicFullClipFrame(f *av1.Frame) uint64 {
	value := uint64(f.Format.Width)<<32 | uint64(uint32(f.Format.Height))
	bytesPerSample := f.Layout.BytesPerSample
	for _, plane := range []av1.FramePlane{f.Y, f.U, f.V} {
		if plane.Width <= 0 || plane.Height <= 0 || len(plane.Pix) == 0 {
			continue
		}
		rowBytes := plane.Width * bytesPerSample
		last := (plane.Height-1)*plane.Stride + rowBytes - 1
		if last >= 0 && last < len(plane.Pix) {
			value += uint64(plane.Pix[0]) + uint64(plane.Pix[last])
		}
	}
	return value
}

type publicFullClipTiming struct {
	frames  int
	samples []time.Duration
}

func samplePublicFullClipCold(clip publicFullClip) (publicFullClipTiming, error) {
	for i := 0; i < publicFullClipWarmups; i++ {
		result, err := decodePublicFullClipFresh(clip.data, false)
		if err != nil {
			return publicFullClipTiming{}, err
		}
		if result.frames == 0 {
			return publicFullClipTiming{}, errors.New("warmup produced no visible frames")
		}
		publicFullClipSampleSink += result.observed
	}
	timing := publicFullClipTiming{frames: -1, samples: make([]time.Duration, 0, publicFullClipSamples)}
	for i := 0; i < publicFullClipSamples; i++ {
		start := time.Now()
		result, err := decodePublicFullClipFresh(clip.data, false)
		elapsed := time.Since(start)
		if err != nil {
			return publicFullClipTiming{}, err
		}
		if result.frames == 0 {
			return publicFullClipTiming{}, errors.New("sample produced no visible frames")
		}
		if timing.frames == -1 {
			timing.frames = result.frames
		} else if timing.frames != result.frames {
			return publicFullClipTiming{}, fmt.Errorf("sample produced %d frames, earlier sample produced %d", result.frames, timing.frames)
		}
		publicFullClipSampleSink += result.observed
		timing.samples = append(timing.samples, elapsed)
	}
	return timing, nil
}

func samplePublicFullClipWarm(clip publicFullClip) (publicFullClipTiming, error) {
	dec, err := av1.NewDecoderFromIVF(clip.data, av1.WithWorkers(1))
	if err != nil {
		return publicFullClipTiming{}, fmt.Errorf("NewDecoderFromIVF: %w", err)
	}
	defer dec.Close()
	for i := 0; i < publicFullClipWarmups; i++ {
		result, err := decodePublicFullClipAfterReset(dec)
		if err != nil {
			return publicFullClipTiming{}, err
		}
		if result.frames == 0 {
			return publicFullClipTiming{}, errors.New("warmup produced no visible frames")
		}
		publicFullClipSampleSink += result.observed
	}
	timing := publicFullClipTiming{frames: -1, samples: make([]time.Duration, 0, publicFullClipSamples)}
	for i := 0; i < publicFullClipSamples; i++ {
		start := time.Now()
		result, err := decodePublicFullClipAfterReset(dec)
		elapsed := time.Since(start)
		if err != nil {
			return publicFullClipTiming{}, err
		}
		if result.frames == 0 {
			return publicFullClipTiming{}, errors.New("sample produced no visible frames")
		}
		if timing.frames == -1 {
			timing.frames = result.frames
		} else if timing.frames != result.frames {
			return publicFullClipTiming{}, fmt.Errorf("sample produced %d frames, earlier sample produced %d", result.frames, timing.frames)
		}
		publicFullClipSampleSink += result.observed
		timing.samples = append(timing.samples, elapsed)
	}
	return timing, nil
}

// samplePublicFullClipColdPayload parses and copies the IVF payloads before
// timing. Each timed sample includes NewDecoder, all DecodeNext calls, and
// Close, but excludes IVF parsing/copying and sidecar verification.
func samplePublicFullClipColdPayload(clip publicFullClip) (publicFullClipTiming, error) {
	payloads, err := publicFullClipPayloads(clip.data)
	if err != nil {
		return publicFullClipTiming{}, err
	}
	warmup, err := decodePublicFullClipPayloads(payloads, false)
	if err != nil {
		return publicFullClipTiming{}, fmt.Errorf("warmup NewDecoder(payloads): %w", err)
	}
	if warmup.frames == 0 {
		return publicFullClipTiming{}, errors.New("warmup produced no visible frames")
	}
	publicFullClipSampleSink += warmup.observed

	timing := publicFullClipTiming{frames: -1, samples: make([]time.Duration, 0, publicFullClipSamples)}
	for i := 0; i < publicFullClipSamples; i++ {
		start := time.Now()
		result, err := decodePublicFullClipPayloads(payloads, false)
		elapsed := time.Since(start)
		if err != nil {
			return publicFullClipTiming{}, fmt.Errorf("NewDecoder(payloads) sample %d: %w", i, err)
		}
		if result.frames == 0 {
			return publicFullClipTiming{}, errors.New("sample produced no visible frames")
		}
		if timing.frames == -1 {
			timing.frames = result.frames
		} else if timing.frames != result.frames {
			return publicFullClipTiming{}, fmt.Errorf("sample produced %d frames, earlier sample produced %d", result.frames, timing.frames)
		}
		publicFullClipSampleSink += result.observed
		timing.samples = append(timing.samples, elapsed)
	}
	return timing, nil
}

func decodePublicFullClipAfterReset(dec *av1.Decoder) (publicFullClipResult, error) {
	if err := dec.Reset(); err != nil {
		return publicFullClipResult{}, fmt.Errorf("Reset: %w", err)
	}
	return decodePublicFullClip(dec, false)
}

func medianPublicFullClipDuration(samples []time.Duration) time.Duration {
	ordered := append([]time.Duration(nil), samples...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	if len(ordered) == 0 {
		return 0
	}
	return ordered[len(ordered)/2]
}

func formatPublicFullClipDurations(samples []time.Duration) string {
	parts := make([]string, len(samples))
	for i, sample := range samples {
		parts[i] = sample.String()
	}
	return "[" + strings.Join(parts, ",") + "]"
}
