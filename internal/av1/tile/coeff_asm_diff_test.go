// M-D3 differential harness (see coeff_asm_arm64_spec.go).
//
// It drives randomized and adversarial TXB decode streams through every
// independent implementation of the coefficient spine and asserts the full
// decode record — TXBDecodeResult, coeffs, dirty/level-dirty lists, the
// range-decoder state snapshot, and the post-TXB adapted CDF images — is
// identical between them. Level scratch compares its semantic low nibble;
// the assembly base walk caches min(level,3) in the otherwise-unused high
// nibble until the next sparse clear. Today this cross-checks the pure-Go bodies
// (tracked2D, WithGeo trusted scan-hot, WithGeo untrusted); the D3-b/D3-c
// arm64 kernels join the same lockstep as further variants so they can be
// diffed bit-for-bit. It extends (does not duplicate) the per-read entropy
// kernel differential in internal/av1/entropy/reader_arch_diff_test.go: that
// test pins single symbol reads; this one pins whole-TXB sequences with
// chained CDF adaptation, refill boundaries, and past-end ecLotsBits
// accounting. GOAV1_TXB_DIFF_TXBS scales the stream volume of BOTH lockstep
// tests below for soak runs (the value is streams per (size, class, update)
// combination; each stream decodes up to 48 back-to-back TXBs per variant).
package tile

import (
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

type txbDiffVariant int

const (
	txbVariantTracked txbDiffVariant = iota
	txbVariantGeoTrusted
	txbVariantGeoUntrusted
	// The *Go variants force the pure-Go loops even when the arm64 kernels
	// (D3-b base-levels and D3-c sign/golomb) are built in — the same
	// mechanism as the GOAV1_DISABLE_COEFF_ASM kill switch — so the lockstep
	// proves kernels-on and kernels-off byte-identical.
	txbVariantTrackedGo
	txbVariantGeoTrustedGo
)

func (v txbDiffVariant) String() string {
	switch v {
	case txbVariantTracked:
		return "tracked2D"
	case txbVariantGeoTrusted:
		return "geoTrusted"
	case txbVariantGeoUntrusted:
		return "geoUntrusted"
	case txbVariantTrackedGo:
		return "tracked2D-purego"
	case txbVariantGeoTrustedGo:
		return "geoTrusted-purego"
	default:
		return "unknown"
	}
}

// txbDiffState is one variant's complete mutable decode state for a stream of
// back-to-back TXBs (mirrors the production LumaCoeffTreeScratch reuse shape:
// dirty replay clears coeffs, level-dirty sparse-clears the levels scratch).
type txbDiffState struct {
	cdfs          CoeffCDFs
	state         DecodeState
	sizeVal       TransformSize
	coeffs        []int16
	scan          []int16
	levels        []uint8
	dirty         [maxCoeffScanLen]int16
	dirtyLen      uint16
	levelDirty    [maxCoeffScanLen]int16
	levelDirtyLen uint16
	levelScratch  [maxCoeffScratchLen]uint8
}

func newTXBDiffState(t *testing.T, payload []byte, size TransformSize, class transform.Class, cdfs *CoeffCDFs, update bool) *txbDiffState {
	t.Helper()
	txSize, err := size.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, levels := coeffScanAndScratch(t, size, txSize, class)
	s := &txbDiffState{
		cdfs:    *cdfs,
		sizeVal: size,
		coeffs:  make([]int16, len(scan)),
		scan:    scan,
	}
	// The sparse level clear consumes levelDirtyScratch; production points it
	// at the same backing array as the in-use levels window, so mirror that.
	s.levels = s.levelScratch[:len(levels)]
	if err := s.state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{DisableCDFUpdate: !update}); err != nil {
		t.Fatal(err)
	}
	return s
}

// decodeOne decodes the next TXB from the shared stream through the selected
// implementation. Between TXBs it replays the production reuse discipline:
// zero the previous nonzero coeffs from the dirty list and reset its length
// (the level-dirty list is consumed by the decode itself).
func (s *txbDiffState) decodeOne(v txbDiffVariant, class transform.Class, dcSignCtx uint8, eobCtx uint8) (TXBDecodeResult, error) {
	if v == txbVariantTrackedGo || v == txbVariantGeoTrustedGo {
		savedBase := coeffBaseLevelsKernel
		savedSign := coeffSignGolombKernel
		coeffBaseLevelsKernel = false
		coeffSignGolombKernel = false
		defer func() {
			coeffBaseLevelsKernel = savedBase
			coeffSignGolombKernel = savedSign
		}()
	}
	for i := 0; i < int(s.dirtyLen); i++ {
		pos := int(s.dirty[i]) & coeffDirtyPosMask
		s.coeffs[pos] = 0
	}
	s.dirtyLen = 0

	req := TXBDecodeRequest{
		Size:                  s.sizeVal,
		Plane:                 CoeffPlaneY,
		Class:                 class,
		DCSignContext:         dcSignCtx,
		EOBMultiContext:       eobCtx,
		TXBSkipKnown:          true,
		TXBSkip:               false,
		SkipAllZeroCoeffClear: true,
		skipNonZeroCoeffClear: true,
		knownNonZero:          true,
		trustedScan:           v != txbVariantGeoUntrusted,
		coeffDirtyPos:         &s.dirty,
		coeffDirtyLen:         &s.dirtyLen,
		levelDirtyPos:         &s.levelDirty,
		levelDirtyLen:         &s.levelDirtyLen,
		levelDirtyScratch:     &s.levelScratch,
	}
	geo, ok := coeffGeoPtr(req.Size)
	if !ok {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	// The production tracked path only serves Class2D with a trusted scan;
	// everything else goes through the general body.
	if v == txbVariantTracked || v == txbVariantTrackedGo {
		return s.state.readCoefficientsTXBTracked2DWithGeo(&s.cdfs, req, s.coeffs, s.scan, s.levels, geo)
	}
	return s.state.readCoefficientsTXBWithGeo(&s.cdfs, req, s.coeffs, s.scan, s.levels, geo)
}

// compareTXBDiffStates asserts two variants observed the identical decode
// record after the same TXB.
func compareTXBDiffStates(t *testing.T, tag string, txb int, va, vb txbDiffVariant, a, b *txbDiffState, ra, rb TXBDecodeResult, ea, eb error) {
	t.Helper()
	fail := func(field string, av, bv any) {
		t.Fatalf("%s txb=%d %s vs %s: %s mismatch: %v vs %v", tag, txb, va, vb, field, av, bv)
	}
	if (ea != nil) != (eb != nil) || (ea != nil && ea.Error() != eb.Error()) {
		fail("error", ea, eb)
	}
	if ea != nil {
		return
	}
	if ra != rb {
		fail("result", ra, rb)
	}
	if a.state.Reader.State() != b.state.Reader.State() {
		fail("reader state", a.state.Reader.State(), b.state.Reader.State())
	}
	for i := range a.coeffs {
		if a.coeffs[i] != b.coeffs[i] {
			fail("coeffs", i, a.coeffs[i]-b.coeffs[i])
		}
	}
	if a.dirtyLen != b.dirtyLen {
		fail("dirtyLen", a.dirtyLen, b.dirtyLen)
	}
	for i := 0; i < int(a.dirtyLen); i++ {
		if a.dirty[i] != b.dirty[i] {
			fail("dirty", i, []int16{a.dirty[i], b.dirty[i]})
		}
	}
	if a.levelDirtyLen != b.levelDirtyLen {
		fail("levelDirtyLen", a.levelDirtyLen, b.levelDirtyLen)
	}
	for i := 0; i < int(a.levelDirtyLen); i++ {
		if a.levelDirty[i] != b.levelDirty[i] {
			fail("levelDirty", i, []int16{a.levelDirty[i], b.levelDirty[i]})
		}
	}
	for i := range a.levels {
		if a.levels[i]&15 != b.levels[i]&15 {
			fail("levels", i, []uint8{a.levels[i], b.levels[i]})
		}
	}
	if a.cdfs != b.cdfs {
		fail("cdfs", "post-TXB CDF images differ", "")
	}
}

// scrambleCDF re-seeds one CDF row into an adversarial-but-valid state:
// extreme probability shapes (tail-heavy rows force base-symbol saturation,
// BR chains and golomb tails; head-heavy rows force minimum-split decodes)
// warmed to a randomized adaptation count through the public Update path so
// every state is reachable by a real decode.
func scrambleCDF(t *testing.T, rng *rand.Rand, cdf *entropy.CDF) {
	t.Helper()
	symbols := cdf.Symbols()
	if symbols < 2 {
		t.Fatalf("scrambleCDF: invalid row (symbols=%d)", symbols)
	}
	cumulative := make([]uint16, symbols-1)
	switch rng.Intn(4) {
	case 0:
		// tail-heavy: tiny cumulative head, last symbol dominates.
		v := 0
		for i := range cumulative {
			v += 1 + rng.Intn(3)
			cumulative[i] = uint16(v)
		}
	case 1:
		// head-heavy: cumulative packed against 32768, symbol 0 dominates.
		v := entropy.CDFProbTop
		for i := len(cumulative) - 1; i >= 0; i-- {
			v -= 1 + rng.Intn(3)
			cumulative[i] = uint16(v)
		}
	default:
		// random strictly-increasing row.
		vals := make([]int, 0, len(cumulative))
		used := map[int]bool{}
		for len(vals) < len(cumulative) {
			v := 1 + rng.Intn(entropy.CDFProbTop-1)
			if !used[v] {
				used[v] = true
				vals = append(vals, v)
			}
		}
		for i := 0; i < len(vals); i++ {
			for j := i + 1; j < len(vals); j++ {
				if vals[j] < vals[i] {
					vals[i], vals[j] = vals[j], vals[i]
				}
			}
		}
		for i, v := range vals {
			cumulative[i] = uint16(v)
		}
	}
	if err := cdf.Init(cumulative); err != nil {
		t.Fatalf("scrambleCDF Init: %v (cumulative=%v)", err, cumulative)
	}
	for warm := rng.Intn(40); warm > 0; warm-- {
		if err := cdf.Update(rng.Intn(symbols)); err != nil {
			t.Fatalf("scrambleCDF Update: %v", err)
		}
	}
}

// scrambleCoeffCDFs randomizes every CDF family the TXB spine reads.
func scrambleCoeffCDFs(t *testing.T, rng *rand.Rand, cdfs *CoeffCDFs) {
	t.Helper()
	for tx := range CoeffTxSizeContexts {
		for plane := range CoeffPlaneTypes {
			for ctx := range CoeffBaseContexts {
				scrambleCDF(t, rng, &cdfs.CoeffBase[tx][plane][ctx])
			}
			for ctx := range CoeffBRContexts {
				scrambleCDF(t, rng, &cdfs.CoeffBR[tx][plane][ctx])
			}
			for ctx := range EOBBaseContexts {
				scrambleCDF(t, rng, &cdfs.CoeffBaseEOB[tx][plane][ctx])
			}
			for ctx := range EOBCoefContexts {
				scrambleCDF(t, rng, &cdfs.EOBExtra[tx][plane][ctx])
			}
		}
	}
	for plane := range CoeffPlaneTypes {
		for ctx := range 3 {
			scrambleCDF(t, rng, &cdfs.DCSign[plane][ctx])
		}
		for ctx := range maxEOBFlagContexts {
			scrambleCDF(t, rng, &cdfs.EOBFlag16[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag32[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag64[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag128[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag256[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag512[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag1024[plane][ctx])
		}
	}
}

// txbDiffPayload builds one stream. Shapes: random bytes, all-zero (drives
// max-probability paths), all-ones (minimum splits, saturating symbols,
// golomb tails), alternating, and short tails that push the reader past the
// end of the buffer into ecLotsBits tell accounting mid-TXB.
func txbDiffPayload(rng *rand.Rand, shape int) []byte {
	var n int
	switch shape % 5 {
	case 0, 1:
		n = 1 + rng.Intn(4096)
	default:
		n = 1 + rng.Intn(17)
	}
	p := make([]byte, n)
	switch shape % 4 {
	case 0:
		rng.Read(p)
	case 1:
		for i := range p {
			p[i] = 0xff
		}
	case 2:
		// all zero
	case 3:
		for i := range p {
			p[i] = 0xaa
		}
	}
	return p
}

func txbDiffIters(defaultStreams int) int {
	if v := os.Getenv("GOAV1_TXB_DIFF_TXBS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultStreams
}

// scrambleCDFTailHeavy forces one row into the tail-heavy shape (the last
// symbol dominates) warmed to a randomized adaptation count. Seeding the
// CoeffBase/CoeffBaseEOB/CoeffBR families this way makes nearly every base
// read saturate, runs the BR chain to its cap and pushes the replayed levels
// to MaxBaseBRRange — forcing the DC-sign/sign-bit path to append an
// Exp-Golomb tail for nearly every nonzero coefficient (the D3-c kernel's
// hard cases).
func scrambleCDFTailHeavy(t *testing.T, rng *rand.Rand, cdf *entropy.CDF) {
	t.Helper()
	symbols := cdf.Symbols()
	if symbols < 2 {
		t.Fatalf("scrambleCDFTailHeavy: invalid row (symbols=%d)", symbols)
	}
	cumulative := make([]uint16, symbols-1)
	v := 0
	for i := range cumulative {
		v += 1 + rng.Intn(2)
		cumulative[i] = uint16(v)
	}
	if err := cdf.Init(cumulative); err != nil {
		t.Fatalf("scrambleCDFTailHeavy Init: %v (cumulative=%v)", err, cumulative)
	}
	for warm := rng.Intn(40); warm > 0; warm-- {
		if err := cdf.Update(symbols - 1); err != nil {
			t.Fatalf("scrambleCDFTailHeavy Update: %v", err)
		}
	}
}

// seedGolombHeavyCDFs builds the D3-c adversarial CDF state: every level
// read saturates (tail-heavy base/BR rows), the DC-sign rows carry scrambled
// probability shapes and warmed adaptation counts, and the EOB families stay
// randomly scrambled so the dirty-list lengths spread across 1..maxEOB.
func seedGolombHeavyCDFs(t *testing.T, rng *rand.Rand, cdfs *CoeffCDFs) {
	t.Helper()
	for tx := range CoeffTxSizeContexts {
		for plane := range CoeffPlaneTypes {
			for ctx := range CoeffBaseContexts {
				scrambleCDFTailHeavy(t, rng, &cdfs.CoeffBase[tx][plane][ctx])
			}
			for ctx := range CoeffBRContexts {
				scrambleCDFTailHeavy(t, rng, &cdfs.CoeffBR[tx][plane][ctx])
			}
			for ctx := range EOBBaseContexts {
				scrambleCDFTailHeavy(t, rng, &cdfs.CoeffBaseEOB[tx][plane][ctx])
			}
			for ctx := range EOBCoefContexts {
				scrambleCDF(t, rng, &cdfs.EOBExtra[tx][plane][ctx])
			}
		}
	}
	for plane := range CoeffPlaneTypes {
		for ctx := range 3 {
			scrambleCDF(t, rng, &cdfs.DCSign[plane][ctx])
		}
		for ctx := range maxEOBFlagContexts {
			scrambleCDF(t, rng, &cdfs.EOBFlag16[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag32[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag64[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag128[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag256[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag512[plane][ctx])
			scrambleCDF(t, rng, &cdfs.EOBFlag1024[plane][ctx])
		}
	}
}

// TestReadCoefficientsTXBVariantsLockstep is the M-D3 differential gate: all
// TXB spine implementations must produce identical records over randomized +
// adversarial streams. GOAV1_TXB_DIFF_TXBS scales the stream count per
// (size, class, update) combination for soak runs.
func TestReadCoefficientsTXBVariantsLockstep(t *testing.T) {
	streams := txbDiffIters(4)
	if testing.Short() {
		streams = 1
	}
	rng := rand.New(rand.NewSource(0xd3a5be))
	for size := TransformSize(0); size < transformSizeCount; size++ {
		if _, ok := coeffGeo(size); !ok {
			continue
		}
		for _, class := range []transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert} {
			for _, update := range []bool{true, false} {
				variants := []txbDiffVariant{txbVariantGeoTrusted, txbVariantGeoUntrusted}
				if class == transform.Class2D {
					variants = append([]txbDiffVariant{txbVariantTracked}, variants...)
					variants = append(variants, txbVariantTrackedGo, txbVariantGeoTrustedGo)
				}
				for stream := 0; stream < streams; stream++ {
					payload := txbDiffPayload(rng, stream+int(size))
					cdfs := new(CoeffCDFs)
					if err := cdfs.InitDefault(uint8(rng.Intn(256))); err != nil {
						t.Fatal(err)
					}
					if stream%2 == 0 {
						scrambleCoeffCDFs(t, rng, cdfs)
					}
					states := make([]*txbDiffState, len(variants))
					for i := range variants {
						states[i] = newTXBDiffState(t, payload, size, class, cdfs, update)
					}
					tag := "size=" + strconv.Itoa(int(size)) + "/class=" + strconv.Itoa(int(class)) + "/update=" + strconv.FormatBool(update) + "/stream=" + strconv.Itoa(stream)
					maxTXBs := 48
					for txb := 0; txb < maxTXBs; txb++ {
						dcSignCtx := uint8(rng.Intn(3))
						eobCtx := uint8(rng.Intn(maxEOBFlagContexts))
						results := make([]TXBDecodeResult, len(variants))
						errs := make([]error, len(variants))
						for i, v := range variants {
							results[i], errs[i] = states[i].decodeOne(v, class, dcSignCtx, eobCtx)
						}
						for i := 1; i < len(variants); i++ {
							compareTXBDiffStates(t, tag, txb, variants[0], variants[i], states[0], states[i], results[0], results[i], errs[0], errs[i])
						}
						if errs[0] != nil {
							break
						}
					}
				}
			}
		}
	}
}

// TestReadCoefficientsTXBSignGolombLockstep is the D3-c-focused differential:
// every stream runs with golomb-heavy CDF state (seedGolombHeavyCDFs), so the
// sign/golomb replay decodes an Exp-Golomb tail for nearly every nonzero
// coefficient. The four payload shapes pin the tail length classes: 0xff
// streams terminate the unary prefix immediately (length 1, value 0), 0x00
// streams run the prefix past maxLength into the validation-error unwind,
// random/0xaa streams spread the lengths in between, and the short-tail
// shapes push golomb bit reads past the end of the buffer into ecLotsBits
// tell accounting. Dirty-list lengths spread across 1..maxEOB via the
// scrambled EOB families. Same lockstep record and GOAV1_TXB_DIFF_TXBS soak
// knob as the main test.
func TestReadCoefficientsTXBSignGolombLockstep(t *testing.T) {
	streams := txbDiffIters(3)
	if testing.Short() {
		streams = 1
	}
	rng := rand.New(rand.NewSource(0xd3c516))
	for size := TransformSize(0); size < transformSizeCount; size++ {
		if _, ok := coeffGeo(size); !ok {
			continue
		}
		for _, class := range []transform.Class{transform.Class2D, transform.ClassHoriz, transform.ClassVert} {
			for _, update := range []bool{true, false} {
				variants := []txbDiffVariant{txbVariantGeoTrusted, txbVariantGeoUntrusted}
				if class == transform.Class2D {
					variants = append([]txbDiffVariant{txbVariantTracked}, variants...)
					variants = append(variants, txbVariantTrackedGo, txbVariantGeoTrustedGo)
				}
				for stream := 0; stream < streams; stream++ {
					payload := txbDiffPayload(rng, stream+int(size))
					cdfs := new(CoeffCDFs)
					if err := cdfs.InitDefault(uint8(rng.Intn(256))); err != nil {
						t.Fatal(err)
					}
					seedGolombHeavyCDFs(t, rng, cdfs)
					states := make([]*txbDiffState, len(variants))
					for i := range variants {
						states[i] = newTXBDiffState(t, payload, size, class, cdfs, update)
					}
					tag := "golomb/size=" + strconv.Itoa(int(size)) + "/class=" + strconv.Itoa(int(class)) + "/update=" + strconv.FormatBool(update) + "/stream=" + strconv.Itoa(stream)
					maxTXBs := 48
					for txb := 0; txb < maxTXBs; txb++ {
						dcSignCtx := uint8(rng.Intn(3))
						eobCtx := uint8(rng.Intn(maxEOBFlagContexts))
						results := make([]TXBDecodeResult, len(variants))
						errs := make([]error, len(variants))
						for i, v := range variants {
							results[i], errs[i] = states[i].decodeOne(v, class, dcSignCtx, eobCtx)
						}
						for i := 1; i < len(variants); i++ {
							compareTXBDiffStates(t, tag, txb, variants[0], variants[i], states[0], states[i], results[0], results[i], errs[0], errs[i])
						}
						if errs[0] != nil {
							break
						}
					}
				}
			}
		}
	}
}
