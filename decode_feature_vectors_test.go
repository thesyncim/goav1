package goav1

import (
	"bufio"
	"crypto/md5"
	"encoding/hex"
	"hash"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
