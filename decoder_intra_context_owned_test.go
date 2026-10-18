package goav1

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/tile"
)

func TestDecoderOwnedIntraContextsPoisonAndHookFallback(t *testing.T) {
	path := filepath.Join("internal", "av1", "testvector", "testdata", "profiles", "profile0-420-8bit-sframe-64x64.ivf")
	ivf, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	owned, err := NewDecoderFromIVF(ivf, WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	defer owned.Close()
	generic, err := NewDecoderFromIVF(ivf, WithWorkers(1))
	if err != nil {
		t.Fatal(err)
	}
	defer generic.Close()
	hooks := 0
	generic.batch.Request.Tile.AfterBlock = func(tile.BlockLoopVisit) error { hooks++; return nil }
	for repeat := 0; repeat < 2; repeat++ {
		if repeat != 0 {
			if err := owned.Reset(); err != nil {
				t.Fatal(err)
			}
			if err := generic.Reset(); err != nil {
				t.Fatal(err)
			}
		}
		for _, dec := range []*Decoder{owned, generic} {
			for i := range dec.batch.TileScratch {
				m := &dec.batch.TileScratch[i].Loop.Mode
				for y := range m.GridMotionValid {
					for x := range m.GridMotionValid[y] {
						m.GridMotionValid[y][x] = 0xa5
						m.GridBlockSizeVisited[y][x] = 0xa5
					}
				}
				m.SBTopRightValid = true
				for x := range m.SBTopRightIntra {
					m.SBTopRightIntra[x] = 0xa5
					m.SBTopRightMotionValid[x] = 0xa5
					m.SBTopRightBlockSize[x] = tile.BlockSize(255)
				}
			}
		}
		for frame := 0; ; frame++ {
			got, ok, ge := owned.DecodeNext()
			want, wok, we := generic.DecodeNext()
			if ge != nil || we != nil || ok != wok || len(got) != len(want) {
				t.Fatalf("repeat%d frame%d owned=%v/%v generic=%v/%v", repeat, frame, ok, ge, wok, we)
			}
			if !ok {
				if frame != 8 {
					t.Fatalf("frames=%d", frame)
				}
				break
			}
			for i := range got {
				if visibleFrameMD5(got[i]) != visibleFrameMD5(want[i]) {
					t.Fatalf("repeat%d frame%d pixels differ from generic path", repeat, frame)
				}
			}
			if frame == 0 {
				if owned.batch.TileScratch[0].Loop.Mode.GridMotionValid[0][0] != 0xa5 || owned.batch.TileScratch[0].Loop.Mode.SBTopRightIntra[0] != 0xa5 || !owned.batch.TileScratch[0].Loop.Mode.SBTopRightValid {
					t.Fatal("first key frame did not select private contexts")
				}
				if generic.batch.TileScratch[0].Loop.Mode.GridMotionValid[0][0] == 0xa5 || hooks == 0 {
					t.Fatal("caller hook did not force complete context path")
				}
			}
		}
	}
}
