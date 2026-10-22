package entropy

import (
	"bytes"
	"math/rand"
	"testing"
)

// logTestOp is one randomized writer call; kind selects the Writer method.
type logTestOp struct {
	kind int
	s    int
	f    uint32
	cdf  int
}

func logTestCDFs(t *testing.T) []CDF {
	t.Helper()
	sizes := []int{2, 3, 4, 4, 5, 7, 8, 16}
	out := make([]CDF, len(sizes))
	for i, n := range sizes {
		if err := out[i].InitUniform(n); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

// writeLogTestOps codes ops through every hooked Writer entry point.
func writeLogTestOps(w *Writer, cdfs []CDF, ops []logTestOp, markEvery int) {
	for i, op := range ops {
		if markEvery > 0 && i%markEvery == 0 {
			w.LogSuperblock()
		}
		c := &cdfs[op.cdf]
		switch op.kind {
		case 0:
			w.WriteBoolQ15(op.s&1, op.f)
		case 1:
			w.WriteBit(op.s & 1)
		case 2:
			w.WriteCDF(c, op.s%int(c.symbols))
		case 3:
			w.WriteSymbol(op.s%int(c.symbols), c.values[:], int(c.symbols))
		case 4:
			switch c.symbols {
			case 2:
				w.WriteBinaryCDFTrusted(c, op.s%2)
			case 3:
				w.WriteCDF3(c, op.s%3)
			case 4:
				if op.s%5 == 4 {
					w.WriteCDF4Zero(c)
				} else {
					w.WriteCDF4(c, op.s%4)
				}
			case 5:
				w.WriteCDF5(c, op.s%5)
			case 7:
				w.WriteCDF7(c, op.s%7)
			default:
				w.WriteCDF(c, op.s%int(c.symbols))
			}
		case 5:
			w.WriteLiteral(uint32(op.s), 5)
		}
		if markEvery > 0 && i%markEvery == 1 {
			w.LogCDEFUnits(0, 1)
			w.LogCDEFUnits(0, 1) // later blocks of the unit read nothing
		}
	}
}

func randomLogTestOps(rng *rand.Rand, n int, cdfCount int) []logTestOp {
	ops := make([]logTestOp, n)
	for i := range ops {
		ops[i] = logTestOp{
			kind: rng.Intn(6),
			s:    rng.Intn(64),
			f:    uint32(1 + rng.Intn(CDFProbTop-1)),
			cdf:  rng.Intn(cdfCount),
		}
	}
	return ops
}

// TestWriterLogReplayIdentity: replaying the log with no literals reproduces
// the original writer's bytes exactly, for every hooked symbol writer.
func TestWriterLogReplayIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 200; trial++ {
		ops := randomLogTestOps(rng, 1+rng.Intn(4000), 8)
		orig := NewWriter(nil)
		var log WriterLog
		orig.SetLog(&log)
		writeLogTestOps(&orig, logTestCDFs(t), ops, 37)
		want, err := orig.Finish()
		if err != nil {
			t.Fatal(err)
		}
		got, err := log.Replay(nil, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("trial %d: replay %d bytes != original %d bytes", trial, len(got), len(want))
		}
	}
}

// TestWriterLogReplayLiterals: the spliced cdef_idx literals decode at the
// marked positions (once per superblock unit) and every other symbol still
// decodes to the written value.
func TestWriterLogReplayLiterals(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const markEvery = 29
	for trial := 0; trial < 100; trial++ {
		ops := randomLogTestOps(rng, 1+rng.Intn(3000), 8)
		bits := 1 + rng.Intn(3)
		w := NewWriter(nil)
		var log WriterLog
		w.SetLog(&log)
		writeLogTestOps(&w, logTestCDFs(t), ops, markEvery)
		if _, err := w.Finish(); err != nil {
			t.Fatal(err)
		}
		sbs := (len(ops) + markEvery - 1) / markEvery
		idx := make([]uint8, sbs*4)
		for i := range idx {
			idx[i] = uint8(rng.Intn(1 << bits))
		}
		data, err := log.Replay(nil, idx, bits)
		if err != nil {
			t.Fatal(err)
		}
		r := NewReader(data)
		cdfs := logTestCDFs(t)
		sb := -1
		for i, op := range ops {
			if i%markEvery == 0 {
				sb++
			}
			c := &cdfs[op.cdf]
			var got, want int
			switch op.kind {
			case 0:
				v, _ := r.ReadBoolQ15(uint16(op.f))
				got, want = int(v), op.s&1
			case 1, 5:
				n := uint8(1)
				want = op.s & 1
				if op.kind == 5 {
					n, want = 5, op.s&31
				}
				v, _ := r.ReadBits(n)
				got = int(v)
			case 2, 4:
				want = op.s % int(c.symbols)
				if op.kind == 4 && c.symbols == 2 {
					want = op.s % 2
				}
				if op.kind == 4 && c.symbols == 4 && op.s%5 == 4 {
					want = 0
				}
				got, _ = r.ReadCDF(c)
			case 3:
				want = op.s % int(c.symbols)
				tmp := c.values
				got, _ = r.ReadSymbol(tmp[:], int(c.symbols))
			}
			if got != want {
				t.Fatalf("trial %d op %d kind %d: decoded %d want %d", trial, i, op.kind, got, want)
			}
			if i%markEvery == 1 {
				v, _ := r.ReadBits(uint8(bits))
				if uint8(v) != idx[sb*4] {
					t.Fatalf("trial %d sb %d: cdef_idx %d want %d", trial, sb, v, idx[sb*4])
				}
			}
		}
	}
}
