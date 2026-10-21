package tile

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
)

func TestWriteSignedDeltaRoundTrip(t *testing.T) {
	values := []int{0, 1, -1, 2, -2, 3, -3, 4, -5, 7, 8, -9, 15, 16, -31, 63, 64, -100, 127, 200, -255, 0, 3, -3}
	var wcdf, rcdf entropy.CDF
	if err := wcdf.InitDefaultDelta(); err != nil {
		t.Fatal(err)
	}
	if err := rcdf.InitDefaultDelta(); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 256))
	for _, v := range values {
		if err := WriteSignedDelta(&w, &wcdf, v); err != nil {
			t.Fatalf("write %d: %v", v, err)
		}
	}
	payload, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}
	r := entropy.NewReader(payload)
	for _, want := range values {
		got, err := r.ReadSignedDelta(&rcdf, entropy.DeltaSmall)
		if err != nil {
			t.Fatalf("read %d: %v", want, err)
		}
		if got != want {
			t.Fatalf("read %d want %d", got, want)
		}
	}
}
