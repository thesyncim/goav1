package goav1_test

import (
	"os"
	"path/filepath"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

// The sidecars copy the independent libaom aomdec per-frame MD5s already
// recorded in conformance/publicpath_test.go and testvector/profiles.
// Each operation decodes all three frames through the exported Decoder API.
func loadPublicPaletteClips(t testing.TB) []publicFullClip {
	t.Helper()
	var clips []publicFullClip
	for _, item := range []struct{ name, file string }{
		{"palette_8bit", "profile1-444-8bit-palette-64x64.ivf"},
		{"palette_10bit", "profile1-444-10bit-palette-64x64.ivf"},
		{"control_8bit", "profile1-444-8bit-64x64.ivf"},
	} {
		path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", item.file)
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		oracle, err := loadPublicFullClipOracle(path)
		if err != nil {
			t.Fatal(err)
		}
		clips = append(clips, publicFullClip{name: item.name, data: data, oracle: oracle})
	}
	return clips
}

func TestPublicDecoderPaletteMD5(t *testing.T) {
	for _, clip := range loadPublicPaletteClips(t) {
		t.Run(clip.name, func(t *testing.T) {
			dec, err := av1.NewDecoderFromIVF(clip.data, av1.WithWorkers(1))
			if err != nil {
				t.Fatal(err)
			}
			defer dec.Close()
			for round := range 2 {
				if round != 0 {
					if err := dec.Reset(); err != nil {
						t.Fatal(err)
					}
				}
				result, err := decodePublicFullClip(dec, true)
				if err != nil {
					t.Fatal(err)
				}
				if err := comparePublicFullClipOracle(result, clip.oracle); err != nil {
					t.Fatal(err)
				}
				t.Logf("round=%d frames=%d stream=%x frameMD5=%x", round, result.frames, result.streamMD5, result.frameMD5s)
			}
		})
	}
}

func BenchmarkPublicDecoderPalette(b *testing.B) {
	for _, clip := range loadPublicPaletteClips(b) {
		b.Run(clip.name, func(b *testing.B) {
			b.Run("warm_reset_decode", func(b *testing.B) {
				dec, err := av1.NewDecoderFromIVF(clip.data, av1.WithWorkers(1))
				if err != nil {
					b.Fatal(err)
				}
				b.Cleanup(dec.Close)
				warmup, err := decodePublicFullClip(dec, true)
				if err != nil {
					b.Fatal(err)
				}
				if err := comparePublicFullClipOracle(warmup, clip.oracle); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					result, err := decodePublicFullClipAfterReset(dec)
					if err != nil {
						b.Fatal(err)
					}
					if result.frames != warmup.frames {
						b.Fatalf("frames=%d want %d", result.frames, warmup.frames)
					}
					publicFullClipSampleSink += result.observed
				}
				b.ReportMetric(float64(warmup.frames), "frames/op")
			})
		})
	}
}
