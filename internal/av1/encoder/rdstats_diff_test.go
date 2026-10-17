package encoder

import (
	"math/rand"
	"testing"
)

// TestRDStatsDiffMatchesBlockStats checks the difference-only statistics
// against rdStatsBlockPureGo's dskip - dcode, rate and all-zero flag.
func TestRDStatsDiffMatchesBlockStats(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for trial := range 4000 {
		count := []int{16, 64, 256, 1024, 24, 7}[trial%6]
		tran := make([]int32, count)
		qc := make([]int16, count)
		step := int32(rng.Intn(3000) + 1)
		ts := uint8(rng.Intn(3))
		density := []int{0, 1, 4, 16, 64}[trial%5]
		for i := range tran {
			tran[i] = rng.Int31n(1<<18) - 1<<17
			if density > 0 && rng.Intn(64) < density {
				qc[i] = int16(rng.Intn(65536) - 32768)
			}
		}
		if trial%7 == 0 {
			qc[0] = -32768
		}
		dskip, dcode, wantRate, wantZero := rdStatsBlockPureGo(tran, qc, count, step, ts)
		for name, fn := range map[string]func([]int32, []int16, int, int32, uint8) (int64, int64, bool){
			"scalar":   rdStatsDiffPureGo,
			"dispatch": rdStatsDiffImpl,
		} {
			diff, rate, zero := fn(tran, qc, count, step, ts)
			if diff != dskip-dcode || rate != wantRate || zero != wantZero {
				t.Fatalf("%s trial %d: diff %d rate %d zero %v, want %d %d %v", name, trial, diff, rate, zero, dskip-dcode, wantRate, wantZero)
			}
		}
	}
}
