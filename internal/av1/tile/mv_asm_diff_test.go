// M-D3 D3-e differential harness (see coeff_asm_arm64_spec.go §7).
//
// It drives randomized and adversarial motion-vector residual streams through
// the arm64 MV kernel and the pure-Go chain in lockstep and asserts the full
// decode record — returned vector, MVResidualResult, error, the range-decoder
// state snapshot and the post-read adapted MVCDFs image — is identical. It
// extends the TXB lockstep in coeff_asm_diff_test.go (whose payload shapes,
// CDF scramblers and GOAV1_TXB_DIFF_TXBS soak knob it reuses) to the MV
// symbol chain: joint, sign, 11-symbol class, class-0 integer or the
// class-long bit loop, fraction and high-precision reads with chained CDF
// adaptation, refill boundaries and past-end ecLotsBits accounting.
package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// readLockstepMV reads one MV residual, forcing the pure-Go chain when kernel
// is false — the same mechanism as the GOAV1_DISABLE_COEFF_ASM=mv kill-switch
// position, so the lockstep proves kernel-on and kernel-off byte-identical.
func readLockstepMV(s *DecodeState, cdfs *MVCDFs, ref motion.Vector, precision MVSubpelPrecision, kernel bool) (motion.Vector, MVResidualResult, error) {
	if !kernel {
		saved := mvResidualKernel
		mvResidualKernel = false
		defer func() { mvResidualKernel = saved }()
	}
	return s.ReadMotionVector(cdfs, ref, precision)
}

// scrambleMVCDFs randomizes every row the MV chain reads (adversarial-but-
// valid shapes warmed to randomized adaptation counts, per scrambleCDF).
func scrambleMVCDFs(t *testing.T, rng *rand.Rand, cdfs *MVCDFs) {
	t.Helper()
	scrambleCDF(t, rng, &cdfs.Joint)
	for i := range cdfs.Components {
		c := &cdfs.Components[i]
		scrambleCDF(t, rng, &c.Classes)
		scrambleCDF(t, rng, &c.Class0FP[0])
		scrambleCDF(t, rng, &c.Class0FP[1])
		scrambleCDF(t, rng, &c.FP)
		scrambleCDF(t, rng, &c.Sign)
		scrambleCDF(t, rng, &c.Class0HP)
		scrambleCDF(t, rng, &c.HP)
		scrambleCDF(t, rng, &c.Class0)
		for j := range c.Bits {
			scrambleCDF(t, rng, &c.Bits[j])
		}
	}
}

// seedClassHeavyMVCDFs forces the D3-e hard cases: tail-heavy joint rows make
// both components decode nearly every read, and tail-heavy class rows push
// the class toward 10 so the integer part runs the full ten-bit loop and the
// magnitude reaches the ±16384 boundary of the int16 packing.
func seedClassHeavyMVCDFs(t *testing.T, rng *rand.Rand, cdfs *MVCDFs) {
	t.Helper()
	scrambleCDFTailHeavy(t, rng, &cdfs.Joint)
	for i := range cdfs.Components {
		c := &cdfs.Components[i]
		scrambleCDFTailHeavy(t, rng, &c.Classes)
		scrambleCDF(t, rng, &c.Class0FP[0])
		scrambleCDF(t, rng, &c.Class0FP[1])
		scrambleCDF(t, rng, &c.FP)
		scrambleCDF(t, rng, &c.Sign)
		scrambleCDF(t, rng, &c.Class0HP)
		scrambleCDF(t, rng, &c.HP)
		scrambleCDF(t, rng, &c.Class0)
		for j := range c.Bits {
			scrambleCDF(t, rng, &c.Bits[j])
		}
	}
}

// TestReadMotionVectorKernelLockstep is the D3-e differential gate: the
// kernel and pure-Go MV chains must produce identical records over randomized
// + adversarial streams, CDF states, precisions and update modes, including
// the reference-clamp error path (extreme refs push ref+diff outside the
// (-16384, 16384) open interval). GOAV1_TXB_DIFF_TXBS scales the stream count
// per (precision, update, seed) combination for soak runs.
func TestReadMotionVectorKernelLockstep(t *testing.T) {
	if !entropy.HasMVResidual {
		t.Skip("arm64 MV residual kernel not built in")
	}
	streams := txbDiffIters(24)
	if testing.Short() {
		streams = 4
	}
	rng := rand.New(rand.NewSource(0xd3e214))
	for _, precision := range []MVSubpelPrecision{MVSubpelNone, MVSubpelLow, MVSubpelHigh} {
		for _, update := range []bool{true, false} {
			for seed := 0; seed < 3; seed++ {
				for stream := 0; stream < streams; stream++ {
					payload := txbDiffPayload(rng, stream+seed)
					var cdfs MVCDFs
					if err := cdfs.InitDefault(); err != nil {
						t.Fatal(err)
					}
					switch seed {
					case 1:
						scrambleMVCDFs(t, rng, &cdfs)
					case 2:
						seedClassHeavyMVCDFs(t, rng, &cdfs)
					}
					var sa, sb DecodeState
					opts := DecodeOptions{DisableCDFUpdate: !update}
					if err := sa.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, opts); err != nil {
						t.Fatal(err)
					}
					if err := sb.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, opts); err != nil {
						t.Fatal(err)
					}
					ca, cb := cdfs, cdfs
					tag := "precision=" + string(rune('0'+int(precision+1))) +
						"/update=" + map[bool]string{true: "on", false: "off"}[update] +
						"/seed=" + string(rune('0'+seed)) +
						"/stream=" + string(rune('0'+stream%10))
					for read := 0; read < 64; read++ {
						ref := motion.Vector{
							Row: int16(rng.Intn(1<<12) - 1<<11),
							Col: int16(rng.Intn(1<<12) - 1<<11),
						}
						if rng.Intn(8) == 0 {
							// Extreme refs steer ref+diff into the clamp error.
							ref = motion.Vector{Row: 16383, Col: -16383}
						}
						mvA, resA, errA := readLockstepMV(&sa, &ca, ref, precision, true)
						mvB, resB, errB := readLockstepMV(&sb, &cb, ref, precision, false)
						if (errA != nil) != (errB != nil) ||
							(errA != nil && errA.Error() != errB.Error()) {
							t.Fatalf("%s read=%d: error mismatch: %v vs %v", tag, read, errA, errB)
						}
						if mvA != mvB {
							t.Fatalf("%s read=%d: mv mismatch: %+v vs %+v", tag, read, mvA, mvB)
						}
						if resA != resB {
							t.Fatalf("%s read=%d: result mismatch:\nkernel: %+v\npure:   %+v", tag, read, resA, resB)
						}
						if sa.Reader.State() != sb.Reader.State() {
							t.Fatalf("%s read=%d: reader state mismatch: %+v vs %+v", tag, read, sa.Reader.State(), sb.Reader.State())
						}
						if ca != cb {
							t.Fatalf("%s read=%d: post-read MVCDFs images differ", tag, read)
						}
						// The clamp error commits the reader state (compared
						// above) and leaves it valid, so the stream continues
						// through it — error-path coverage mid-stream.
					}
				}
			}
		}
	}
}

// BenchmarkReadMotionVector measures the full ReadMotionVector chain (joint +
// both component reads at high precision) over a random stream with default
// nmv CDFs — the D3-e kernel's same-binary A/B hook via
// GOAV1_DISABLE_COEFF_ASM=mv.
func BenchmarkReadMotionVector(b *testing.B) {
	payload := make([]byte, 1<<20)
	rng := rand.New(rand.NewSource(0xd3e5eed))
	rng.Read(payload)
	var cdfs MVCDFs
	if err := cdfs.InitDefault(); err != nil {
		b.Fatal(err)
	}
	var s DecodeState
	reset := func() {
		if err := s.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			b.Fatal(err)
		}
		if err := cdfs.InitDefault(); err != nil {
			b.Fatal(err)
		}
	}
	reset()
	ref := motion.Vector{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i&8191 == 8191 {
			reset()
		}
		if _, _, err := s.ReadMotionVector(&cdfs, ref, MVSubpelHigh); err != nil {
			// A maximal class-10 magnitude can clamp against the ±16384
			// open interval; restart the stream and keep measuring.
			reset()
		}
	}
}
