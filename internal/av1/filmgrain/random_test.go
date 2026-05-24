package filmgrain

import (
	"errors"
	"testing"
)

func TestRandomNumber(t *testing.T) {
	rng := NewRandom(0x1234)
	wantValues := []uint16{0x89, 0x44, 0x22, 0x91, 0xc8, 0xe4, 0xf2, 0xf9}
	wantRegisters := []uint16{0x891a, 0x448d, 0x2246, 0x9123, 0xc891, 0xe448, 0xf224, 0xf912}
	for i := range wantValues {
		got, err := rng.Number(8)
		if err != nil {
			t.Fatal(err)
		}
		if got != wantValues[i] || rng.Register() != wantRegisters[i] {
			t.Fatalf("step %d value=%#x register=%#x want %#x %#x", i, got, rng.Register(), wantValues[i], wantRegisters[i])
		}
	}
}

func TestRandomNumberElevenBits(t *testing.T) {
	rng := NewRandom(0x1234)
	for i, want := range []uint16{1096, 548, 274, 1161, 1604} {
		got, err := rng.Number(11)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("step %d value=%d want %d", i, got, want)
		}
	}
}

func TestRandomSeeds(t *testing.T) {
	cb, err := NewPlaneRandom(0x1234, 1)
	if err != nil {
		t.Fatal(err)
	}
	if cb.Register() != 0xa710 {
		t.Fatalf("cb register=%#x", cb.Register())
	}
	cr, err := NewPlaneRandom(0x1234, 2)
	if err != nil {
		t.Fatal(err)
	}
	if cr.Register() != 0x5bec {
		t.Fatalf("cr register=%#x", cr.Register())
	}
	stripe, err := NewStripeRandom(0x1234, 32)
	if err != nil {
		t.Fatal(err)
	}
	if stripe.Register() != 0xc522 {
		t.Fatalf("stripe register=%#x", stripe.Register())
	}
}

func TestRandomRejectsInvalidInputs(t *testing.T) {
	rng := NewRandom(0)
	if _, err := rng.Number(0); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("Number(0) err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := rng.Number(17); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("Number(17) err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := NewPlaneRandom(0, 0); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("NewPlaneRandom err=%v want %v", err, ErrInvalidParams)
	}
	if _, err := NewStripeRandom(0, -1); !errors.Is(err, ErrInvalidParams) {
		t.Fatalf("NewStripeRandom err=%v want %v", err, ErrInvalidParams)
	}
}

func TestRandomAllocs(t *testing.T) {
	allocs := testing.AllocsPerRun(1000, func() {
		rng := NewRandom(0x1234)
		for i := 0; i < 64; i++ {
			if _, err := rng.Number(11); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := NewPlaneRandom(rng.Register(), 1); err != nil {
			t.Fatal(err)
		}
		if _, err := NewStripeRandom(rng.Register(), 64); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("filmgrain random allocated: %f", allocs)
	}
}

func BenchmarkRandomNumber(b *testing.B) {
	rng := NewRandom(0x1234)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = rng.Number(11)
	}
}
