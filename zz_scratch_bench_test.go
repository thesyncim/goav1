//go:build goav1_scratchbench

package goav1_test

import (
	"os"
	"strings"
	"testing"

	av1 "github.com/thesyncim/goav1"
)

// BenchmarkZZScratchFiles decodes the IVF files listed in GOAV1_ZZ_FILES
// (colon separated), warm, single worker. Scratch only; never staged.
func BenchmarkZZScratchFiles(b *testing.B) {
	files := strings.Split(os.Getenv("GOAV1_ZZ_FILES"), ":")
	var decs []*av1.Decoder
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			b.Skip(err)
		}
		dec, err := av1.NewDecoderFromIVF(data, av1.WithWorkers(1))
		if err != nil {
			b.Fatal(err)
		}
		decs = append(decs, dec)
	}
	b.ResetTimer()
	for b.Loop() {
		for _, dec := range decs {
			if _, err := decodePublicFullClipAfterReset(dec); err != nil {
				b.Fatal(err)
			}
		}
	}
}
