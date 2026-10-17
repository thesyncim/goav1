// Coefficient-spine differential harness.
//
// It drives randomized and adversarial TXB decode streams through the
// independent implementations of the coefficient spine (tracked2D, WithGeo
// trusted scan-hot, WithGeo untrusted) and asserts the full decode record —
// TXBDecodeResult, coeffs, dirty/level-dirty lists, the range-decoder state
// snapshot, and the post-TXB adapted CDF images — is identical between them.
// It pins whole-TXB sequences with chained CDF adaptation, refill boundaries,
// and past-end ecLotsBits accounting. GOAV1_TXB_DIFF_TXBS scales the stream
// volume of the lockstep tests below for soak runs (the value is streams per
// (size, class, update) combination; each stream decodes up to 48
// back-to-back TXBs per variant).
package tile

import (
	"math/rand"
	"os"
	"slices"
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
)

func (v txbDiffVariant) String() string {
	switch v {
	case txbVariantTracked:
		return "tracked2D"
	case txbVariantGeoTrusted:
		return "geoTrusted"
	case txbVariantGeoUntrusted:
		return "geoUntrusted"
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
	levelsPacked  bool
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
		coeffs:  make([]int16, maxCoeffScanLen),
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
	var result TXBDecodeResult
	var err error
	if v == txbVariantTracked && class == transform.Class2D {
		result, err = s.state.readCoefficientsTXBTracked2DWithGeo(&s.cdfs, req, s.coeffs, s.scan, s.levels, geo)
	} else {
		result, err = s.state.readCoefficientsTXBWithGeo(&s.cdfs, req, s.coeffs, s.scan, s.levels, geo)
	}
	if err == nil && result.EOB > 1 {
		s.levelsPacked = v == txbVariantTracked && class == transform.Class2D
	}
	return result, err
}

// decodeShape resets only the range reader and CDF image for this synthetic
// transition step. Coefficient, level, and dirty-list scratch remain shared
// across transform sizes/classes, matching the production reusable scratch.
func (s *txbDiffState) decodeShape(t *testing.T, v txbDiffVariant, size TransformSize, class transform.Class, payload []byte, allZero bool) (TXBDecodeResult, error) {
	t.Helper()
	if !allZero {
		for i := 0; i < int(s.dirtyLen); i++ {
			s.coeffs[coeffDirtyPackedPos(s.dirty[i])] = 0
		}
		s.dirtyLen = 0
	}

	txSize, err := size.TransformSize()
	if err != nil {
		return TXBDecodeResult{}, err
	}
	s.scan, s.levels = coeffScanAndScratch(t, size, txSize, class)
	s.levels = s.levelScratch[:len(s.levels)]
	s.sizeVal = size
	if err := s.cdfs.InitDefault(0); err != nil {
		return TXBDecodeResult{}, err
	}
	if err := s.state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
		return TXBDecodeResult{}, err
	}
	req := TXBDecodeRequest{
		Size:                  size,
		Plane:                 CoeffPlaneY,
		Class:                 class,
		TXBSkipContext:        0,
		DCSignContext:         0,
		EOBMultiContext:       0,
		TXBSkipKnown:          !allZero,
		TXBSkip:               false,
		SkipAllZeroCoeffClear: true,
		skipNonZeroCoeffClear: true,
		coeffDirtyPos:         &s.dirty,
		coeffDirtyLen:         &s.dirtyLen,
		levelDirtyPos:         &s.levelDirty,
		levelDirtyLen:         &s.levelDirtyLen,
		levelDirtyScratch:     &s.levelScratch,
		trustedScan:           true,
		knownNonZero:          !allZero,
	}
	if allZero {
		// Let the real skip CDF take the all-zero return while retaining both
		// scratch backings exactly as the production caller does.
		req.TXBSkipKnown = false
	}
	tracked2D := v == txbVariantTracked && class == transform.Class2D && !allZero
	var result TXBDecodeResult
	if tracked2D {
		geo, ok := coeffGeoPtr(size)
		if !ok {
			return TXBDecodeResult{}, ErrInvalidDecodeState
		}
		result, err = s.state.readCoefficientsTXBTracked2DWithGeo(&s.cdfs, req, s.coeffs, s.scan, s.levels, geo)
	} else {
		result, err = s.state.ReadCoefficientsTXB(&s.cdfs, req, s.coeffs, s.scan, s.levels)
	}
	if err == nil && result.EOB > 1 {
		s.levelsPacked = tracked2D
		if s.levelsPacked {
			assertTracked2DPackedLevels(t, s)
		}
	}
	return result, err
}

func assertTracked2DPackedLevels(t *testing.T, s *txbDiffState) {
	t.Helper()
	for i := 0; i < int(s.levelDirtyLen); i++ {
		pos := int(s.levelDirty[i])
		if pos < 0 || pos >= len(s.levels) {
			t.Fatalf("tracked2D level-dirty position %d outside active levels len=%d", pos, len(s.levels))
		}
		if !validPackedTracked2DLevel(s.levels[pos]) {
			t.Fatalf("tracked2D level at %d is not packed: %#02x", pos, s.levels[pos])
		}
	}
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
	if a.state.Reader.State() != b.state.Reader.State() {
		fail("reader state", a.state.Reader.State(), b.state.Reader.State())
	}
	if a.cdfs != b.cdfs {
		fail("cdfs", "post-TXB CDF images differ", "")
	}
	if ea != nil {
		return
	}
	if ra != rb {
		fail("result", ra, rb)
	}
	if rb.EOB > 1 && vb == txbVariantTracked && b.levelsPacked {
		assertTracked2DPackedLevels(t, b)
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
	if len(a.levels) != len(b.levels) {
		fail("levels len", len(a.levels), len(b.levels))
	}
	for i := range a.levels {
		av := a.levels[i]
		bv := b.levels[i]
		if a.levelsPacked {
			if !validPackedTracked2DLevel(av) {
				fail("invalid packed level", i, av)
			}
			av &= 0x3f
		}
		if b.levelsPacked {
			if !validPackedTracked2DLevel(bv) {
				fail("invalid packed level", i, bv)
			}
			bv &= 0x3f
		}
		if av != bv {
			fail("levels(raw)", i, []uint8{av, bv})
		}
	}
}

func validPackedTracked2DLevel(packed uint8) bool {
	raw := packed & 0x3f
	return raw <= 15 && packed>>6 == min(raw, uint8(3))
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

func TestTracked2DPackedScratchTransitions(t *testing.T) {
	type step struct {
		name    string
		size    TransformSize
		class   transform.Class
		payload []byte
		zero    bool
	}
	steps := []step{
		{name: "large_2d", size: TransformSize32x32, class: transform.Class2D, payload: tracked2DTransitionPayload(t, TransformSize32x32, transform.Class2D, 32)},
		{name: "small_dc", size: TransformSize4x4, class: transform.Class2D, payload: tracked2DTransitionPayload(t, TransformSize4x4, transform.Class2D, 1)},
		{name: "horizontal", size: TransformSize8x8, class: transform.ClassHoriz, payload: tracked2DTransitionPayload(t, TransformSize8x8, transform.ClassHoriz, 4)},
		{name: "vertical", size: TransformSize8x8, class: transform.ClassVert, payload: tracked2DTransitionPayload(t, TransformSize8x8, transform.ClassVert, 4)},
		{name: "all_zero", size: TransformSize8x8, class: transform.Class2D, payload: tracked2DAllZeroPayload(t, TransformSize8x8), zero: true},
		{name: "larger_2d", size: TransformSize64x64, class: transform.Class2D, payload: tracked2DTransitionPayload(t, TransformSize64x64, transform.Class2D, 64)},
	}
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	base := newTXBDiffState(t, steps[0].payload, steps[0].size, steps[0].class, &cdfs, true)
	tracked := newTXBDiffState(t, steps[0].payload, steps[0].size, steps[0].class, &cdfs, true)
	for i, step := range steps {
		t.Run(step.name, func(t *testing.T) {
			baseBefore := slices.Clone(base.levelScratch[:])
			trackedBefore := slices.Clone(tracked.levelScratch[:])
			rb, eb := base.decodeShape(t, txbVariantGeoTrusted, step.size, step.class, step.payload, step.zero)
			rt, et := tracked.decodeShape(t, txbVariantTracked, step.size, step.class, step.payload, step.zero)
			if eb != nil || et != nil {
				t.Fatalf("transition decode errors base=%v tracked=%v", eb, et)
			}
			if step.zero && (eb == nil && !rb.AllZero || et == nil && !rt.AllZero) {
				t.Fatalf("all-zero step results base=%+v tracked=%+v", rb, rt)
			}
			compareTXBDiffStates(t, "shared-scratch", i, txbVariantGeoTrusted, txbVariantTracked, base, tracked, rb, rt, eb, et)
			if rb.EOB <= 1 || rb.AllZero {
				assertLevelScratchUnchanged(t, "base", baseBefore, base.levelScratch[:])
				assertLevelScratchUnchanged(t, "tracked", trackedBefore, tracked.levelScratch[:])
				return
			}
			assertLevelScratchTailZero(t, "base", base.levelScratch[:], len(base.levels))
			assertLevelScratchTailZero(t, "tracked", tracked.levelScratch[:], len(tracked.levels))
		})
	}
}

func assertLevelScratchUnchanged(t *testing.T, name string, before, after []uint8) {
	t.Helper()
	if !slices.Equal(before, after) {
		t.Fatalf("%s level scratch changed on DC-only/all-zero transition", name)
	}
}

func assertLevelScratchTailZero(t *testing.T, name string, scratch []uint8, validLen int) {
	t.Helper()
	if validLen < 0 || validLen > len(scratch) {
		t.Fatalf("%s invalid active level length %d for scratch len %d", name, validLen, len(scratch))
	}
	for i, v := range scratch[validLen:] {
		if v != 0 {
			t.Fatalf("%s level scratch canary at %d was not cleared: %#02x", name, validLen+i, v)
		}
	}
}

func tracked2DTransitionPayload(t *testing.T, size TransformSize, class transform.Class, minEOB uint16) []byte {
	t.Helper()
	txSize, err := size.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, levels := coeffScanAndScratch(t, size, txSize, class)
	coeffs := make([]int16, len(scan))
	for seed := 0; seed < 8192; seed++ {
		payload := []byte{byte(seed), byte(seed*73 + 19), byte(seed ^ 0xa5), byte(seed*29 + 7), byte(seed*11 + 3), 0x5a}
		clear(coeffs)
		clear(levels)
		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		req := TXBDecodeRequest{Size: size, Plane: CoeffPlaneY, Class: class, TXBSkipKnown: true, knownNonZero: true, trustedScan: true}
		result, err := state.ReadCoefficientsTXB(&cdfs, req, coeffs, scan, levels)
		if err == nil && !result.AllZero && (minEOB == 1 && result.EOB == 1 || minEOB > 1 && result.EOB >= minEOB) {
			return slices.Clone(payload)
		}
	}
	t.Fatalf("could not find transition payload size=%d class=%d eob=%d", size, class, minEOB)
	return nil
}

func tracked2DAllZeroPayload(t *testing.T, size TransformSize) []byte {
	t.Helper()
	for seed := 0; seed < 8192; seed++ {
		payload := []byte{byte(seed), byte(seed*73 + 19), byte(seed ^ 0xa5), byte(seed*29 + 7), byte(seed*11 + 3), 0x5a}
		var cdfs CoeffCDFs
		if err := cdfs.InitDefault(0); err != nil {
			t.Fatal(err)
		}
		var state DecodeState
		if err := state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
			t.Fatal(err)
		}
		allZero, err := state.ReadTXBSkip(&cdfs, TXBSkipRequest{Size: size, Context: 0})
		if err == nil && allZero {
			return slices.Clone(payload)
		}
	}
	t.Fatalf("could not find all-zero transition payload size=%d", size)
	return nil
}
