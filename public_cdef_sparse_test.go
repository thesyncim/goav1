package goav1_test

import (
	"os"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

func TestPublicDecoderCDEFSparseMD5(t *testing.T) {
	if os.Getenv("GOAV1_PUBLIC_FULLCLIP") != "1" {
		t.Skip("set GOAV1_PUBLIC_FULLCLIP=1 for the focused public corpus gate")
	}
	selected := 0
	for _, clip := range loadPublicFullClipCorpus(t) {
		if clip.name != "p360_intra_q32" && clip.name != "p720_inter_q32" {
			continue
		}
		selected++
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
				if result.frames != 48 {
					t.Fatalf("frames=%d want 48", result.frames)
				}
				t.Logf("round=%d frames=%d streamMD5=%x", round, result.frames, result.streamMD5)
			}
		})
	}
	if selected != 2 {
		t.Fatalf("selected=%d want 2", selected)
	}
}
