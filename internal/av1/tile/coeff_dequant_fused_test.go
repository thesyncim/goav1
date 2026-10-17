package tile

import (
	"errors"
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

type coeffDequantTestProvider struct {
	target  *CoeffDequantTarget
	allow   bool
	factors [3][2]int32
}

type coeffDequantLifecycleSelector struct {
	typ       transform.Type
	recordErr error
}

func (s coeffDequantLifecycleSelector) SelectCoeffTransform(CoeffTransformRequest) (transform.Type, error) {
	return s.typ, nil
}

func (s coeffDequantLifecycleSelector) RecordCoeffTransform(CoeffTransformRequest, transform.Type) error {
	return s.recordErr
}

type coeffDequantLifecycleOutcome struct {
	state   DecodeState
	cdfs    CoeffCDFs
	ctx     CoeffEntropyContext
	scratch LumaCoeffTreeScratch
	typ     transform.Type
	result  TXBDecodeResult
	coeffs  []int16
	err     error
}

func decodeCoeffDequantLifecycleTXB(t *testing.T, payload []byte, target *CoeffDequantTarget, selector CoeffTransformSelector, typ transform.Type, useType bool) coeffDequantLifecycleOutcome {
	t.Helper()
	out := coeffDequantLifecycleOutcome{}
	if err := out.cdfs.InitDefault(0); err != nil {
		t.Fatal(err)
	}
	if err := out.state.Reset(payload, Job{Offset: 0, Size: uint32(len(payload))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	ctxReq := CoeffContextRequest{
		PlaneBlock: BlockSize4x4,
		Size:       TransformSize4x4,
		VisibleW4:  1,
		VisibleH4:  1,
	}
	txDims, blockDims, visibleW, visibleH, err := validateCoeffContextRequestWithSpan(ctxReq)
	if err != nil {
		t.Fatal(err)
	}
	known := coeffContextKnown{Plane: CoeffPlaneY, TXDims: txDims, Block: blockDims, VisibleW: visibleW, VisibleH: visibleH}
	transformReq := CoeffTransformRequest{Block: TransformBlock{Size: TransformSize4x4, VisibleW4: 1, VisibleH4: 1}}
	out.typ, out.result, out.coeffs, _, out.err = out.state.decodeCoeffTXBWithKnownContext(
		&out.cdfs, &out.ctx, &out.scratch, ctxReq,
		TXBDecodeRequest{SkipAllZeroCoeffClear: true, dequant: target}, known,
		selector, typ, useType, transform.Class2D, transformReq,
	)
	return out
}

func (p coeffDequantTestProvider) PrepareCoeffDequantPlane(plane uint8) bool {
	if !p.allow || p.target == nil || plane > 2 {
		return false
	}
	p.target.Factors[plane] = p.factors[plane]
	p.target.PreparedPlanes |= 1 << plane
	return true
}

func (s *txbDiffState) decodeOneWithDequant(t *testing.T, target *CoeffDequantTarget, dequantPlane uint8, class transform.Class, dcSignCtx uint8, eobCtx uint8, dcOnly bool) (TXBDecodeResult, error) {
	t.Helper()
	for i := 0; i < int(s.dirtyLen); i++ {
		s.coeffs[coeffDirtyPackedPos(s.dirty[i])] = 0
	}
	s.dirtyLen = 0
	s.setDecodeShape(t, s.sizeVal, class)
	resetCoeffDequantTXB(target)
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
		trustedScan:           true,
		coeffDirtyPos:         &s.dirty,
		coeffDirtyLen:         &s.dirtyLen,
		levelDirtyPos:         &s.levelDirty,
		levelDirtyLen:         &s.levelDirtyLen,
		levelDirtyScratch:     &s.levelScratch,
		dequant:               target,
		dequantPlane:          dequantPlane,
		dequantDCOnly:         dcOnly,
	}
	geo, ok := coeffGeoPtr(req.Size)
	if !ok {
		return TXBDecodeResult{}, ErrInvalidDecodeState
	}
	var result TXBDecodeResult
	var err error
	if class == transform.Class2D {
		result, err = s.state.readCoefficientsTXBTracked2DWithGeo(&s.cdfs, req, s.coeffs, s.scan, s.levels, geo)
	} else {
		result, err = s.state.readCoefficientsTXBWithGeo(&s.cdfs, req, s.coeffs, s.scan, s.levels, geo)
	}
	if err == nil && result.EOB > 1 {
		s.levelsPacked = class == transform.Class2D
		if s.levelsPacked {
			assertTracked2DPackedLevels(t, s)
		}
	}
	return result, err
}

func (s *txbDiffState) setDecodeShape(t *testing.T, size TransformSize, class transform.Class) {
	t.Helper()
	txSize, err := size.TransformSize()
	if err != nil {
		t.Fatal(err)
	}
	scan, levels := coeffScanAndScratch(t, size, txSize, class)
	s.sizeVal = size
	s.scan = scan
	s.levels = s.levelScratch[:len(levels)]
}

func TestReadTracked2DDequantFusionMatchesQuantizedOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(0xd3f053))
	for size := TransformSize(0); size < transformSizeCount; size++ {
		geo, ok := coeffGeoPtr(size)
		if !ok {
			continue
		}
		for _, bitDepth := range []uint8{8, 10, 12} {
			for _, update := range []bool{true, false} {
				for sample := range 3 {
					payload := txbDiffPayload(rng, sample+int(size)*3)
					cdfs := new(CoeffCDFs)
					if err := cdfs.InitDefault(uint8(rng.Intn(256))); err != nil {
						t.Fatal(err)
					}
					if sample != 0 {
						scrambleCoeffCDFs(t, rng, cdfs)
					}
					base := newTXBDiffState(t, payload, size, transform.Class2D, cdfs, update)
					fused := newTXBDiffState(t, payload, size, transform.Class2D, cdfs, update)
					factors := [3][2]int32{
						{79 + int32(sample*3), 101 + int32(sample*5)},
						{83 + int32(sample*7), 109 + int32(sample*2)},
						{89 + int32(sample*5), 127 + int32(sample*3)},
					}
					target := &CoeffDequantTarget{
						Dst:      make([]int32, maxCoeffScanLen),
						BitDepth: bitDepth,
					}
					target.Provider = coeffDequantTestProvider{target: target, allow: sample != 2, factors: factors}
					for i := range target.Dst {
						target.Dst[i] = int32(0x5a5a0000 + i)
					}

					for txb := range 3 {
						dcCtx := uint8(rng.Intn(3))
						eobCtx := uint8(rng.Intn(maxEOBFlagContexts))
						dcOnly := txb%2 == 0
						beforeDst := slices.Clone(target.Dst)
						rb, eb := base.decodeOne(txbVariantTracked, transform.Class2D, dcCtx, eobCtx)
						rf, ef := fused.decodeOneWithDequant(t, target, 0, transform.Class2D, dcCtx, eobCtx, dcOnly)
						if (eb == nil) != (ef == nil) || (eb != nil && eb.Error() != ef.Error()) {
							t.Fatalf("size=%d bd=%d update=%t sample=%d txb=%d errors differ: baseline=%v fused=%v", size, bitDepth, update, sample, txb, eb, ef)
						}
						if base.state.Reader.State() != fused.state.Reader.State() || base.cdfs != fused.cdfs {
							t.Fatalf("size=%d bd=%d update=%t sample=%d txb=%d reader/CDF state differs", size, bitDepth, update, sample, txb)
						}
						if eb != nil {
							if target.decoded || target.Ready || target.ActiveRows != 0 || target.TxScale != 0 {
								t.Fatalf("failed TXB published dequant target: %+v", target)
							}
							break
						}
						if rb != rf {
							t.Fatalf("size=%d bd=%d update=%t sample=%d txb=%d result baseline=%+v fused=%+v", size, bitDepth, update, sample, txb, rb, rf)
						}

						if sample == 2 {
							// A failed lazy preparation must leave the established int16
							// path and its dirty-list contract unchanged.
							if target.decoded || target.Ready || base.dirtyLen != fused.dirtyLen ||
								!slices.Equal(base.coeffs[:int(geo.maxEOB)], fused.coeffs[:int(geo.maxEOB)]) ||
								!slices.Equal(target.Dst, beforeDst) {
								t.Fatalf("provider fallback changed the quantized path: size=%d txb=%d", size, txb)
							}
							continue
						}
						if !target.decoded || target.Ready || fused.dirtyLen != 0 {
							t.Fatalf("successful reader state target=%+v coeffDirtyLen=%d", target, fused.dirtyLen)
						}
						txSize, err := size.TransformSize()
						if err != nil {
							t.Fatal(err)
						}
						txScale, err := quantize.TransformScale(int(txSize.Width), int(txSize.Height))
						if err != nil {
							t.Fatal(err)
						}
						want := make([]int32, int(geo.maxEOB))
						if err := quantize.DequantizeBlockScaledBitDepthEOB(want, int(geo.scanHeight), base.coeffs, int(geo.scanHeight), base.scan, int(rb.EOB), int(geo.scanWidth), int(geo.scanHeight), quantize.Quantizer{DC: factors[0][0], AC: factors[0][1]}, txScale, bitDepth); err != nil {
							t.Fatalf("dequant oracle size=%d eob=%d: %v", size, rb.EOB, err)
						}
						compareLen := len(want)
						if rb.EOB == 1 && dcOnly {
							compareLen = 1
							for i := 1; i < int(geo.maxEOB); i++ {
								if target.Dst[i] != beforeDst[i] {
									t.Fatalf("DC-only EOB1 cleared unrelated dequant scratch at %d", i)
								}
							}
						}
						if !slices.Equal(target.Dst[:compareLen], want[:compareLen]) {
							t.Fatalf("dequant mismatch size=%d bd=%d txb=%d eob=%d", size, bitDepth, txb, rb.EOB)
						}
						for i, coeff := range fused.coeffs[:int(geo.maxEOB)] {
							if coeff != 0 {
								t.Fatalf("target-only fused read wrote int16 coeff[%d]=%d", i, coeff)
							}
						}
						wantRows := 0
						for pos, coeff := range base.coeffs[:int(geo.maxEOB)] {
							if coeff == 0 {
								continue
							}
							row := pos & (int(geo.scanHeight) - 1)
							if row+1 > wantRows {
								wantRows = row + 1
							}
						}
						if target.ActiveRows != wantRows || target.TxScale != txScale {
							t.Fatalf("size=%d activeRows=%d want=%d txScale=%d want=%d", size, target.ActiveRows, wantRows, target.TxScale, txScale)
						}
						for i := int(geo.maxEOB); i < len(target.Dst); i++ {
							if target.Dst[i] != beforeDst[i] {
								t.Fatalf("size=%d wrote past dequant prefix at %d", size, i)
							}
						}
					}
				}
			}
		}
	}
}

func TestReadTracked2DDequantFusionUsesPerPlaneFactors(t *testing.T) {
	rng := rand.New(rand.NewSource(0x71a9))
	payload := txbDiffPayload(rng, 0)
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(73); err != nil {
		t.Fatal(err)
	}
	factors := [3][2]int32{{17, 31}, {29, 43}, {47, 59}}
	for plane := uint8(0); plane < 3; plane++ {
		base := newTXBDiffState(t, payload, TransformSize16x16, transform.Class2D, &cdfs, true)
		fused := newTXBDiffState(t, payload, TransformSize16x16, transform.Class2D, &cdfs, true)
		target := &CoeffDequantTarget{Dst: make([]int32, maxCoeffScanLen), BitDepth: 10}
		target.Provider = coeffDequantTestProvider{target: target, allow: true, factors: factors}
		rBase, errBase := base.decodeOne(txbVariantTracked, transform.Class2D, 1, 0)
		rFused, errFused := fused.decodeOneWithDequant(t, target, plane, transform.Class2D, 1, 0, false)
		if (errBase == nil) != (errFused == nil) || (errBase != nil && errBase.Error() != errFused.Error()) {
			t.Fatalf("plane=%d baseline error=%v fused error=%v", plane, errBase, errFused)
		}
		if errBase != nil {
			t.Fatalf("plane=%d fixture did not decode: %v", plane, errBase)
		}
		if rBase != rFused || base.state.Reader.State() != fused.state.Reader.State() || base.cdfs != fused.cdfs {
			t.Fatalf("plane=%d decoded state diverged", plane)
		}
		if target.PreparedPlanes != 1<<plane || target.Factors[plane] != factors[plane] {
			t.Fatalf("plane=%d prepared=%03b factors=%v want=%v", plane, target.PreparedPlanes, target.Factors[plane], factors[plane])
		}
		geo, ok := coeffGeoPtr(TransformSize16x16)
		if !ok {
			t.Fatal("missing test geometry")
		}
		want := make([]int32, int(geo.maxEOB))
		if err := quantize.DequantizeBlockScaledBitDepthEOB(want, int(geo.scanHeight), base.coeffs, int(geo.scanHeight), base.scan, int(rBase.EOB), int(geo.scanWidth), int(geo.scanHeight), quantize.Quantizer{DC: factors[plane][0], AC: factors[plane][1]}, 0, 10); err != nil {
			t.Fatalf("plane=%d dequant oracle: %v", plane, err)
		}
		if !slices.Equal(target.Dst[:len(want)], want) {
			t.Fatalf("plane=%d output did not use its DC/AC factors", plane)
		}
	}
}

func TestReadTracked2DDequantFusionScratchTransitions(t *testing.T) {
	rng := rand.New(rand.NewSource(0xf0510))
	payload := make([]byte, 8192)
	if _, err := rng.Read(payload); err != nil {
		t.Fatal(err)
	}
	var cdfs CoeffCDFs
	if err := cdfs.InitDefault(91); err != nil {
		t.Fatal(err)
	}
	base := newTXBDiffState(t, payload, TransformSize16x16, transform.Class2D, &cdfs, true)
	fused := newTXBDiffState(t, payload, TransformSize16x16, transform.Class2D, &cdfs, true)
	target := &CoeffDequantTarget{
		Dst:      make([]int32, maxCoeffScanLen),
		BitDepth: 10,
	}
	target.Provider = coeffDequantTestProvider{
		target:  target,
		allow:   true,
		factors: [3][2]int32{{73, 89}, {97, 113}, {127, 149}},
	}
	steps := []struct {
		name  string
		size  TransformSize
		class transform.Class
	}{
		{name: "fused-16x16", size: TransformSize16x16, class: transform.Class2D},
		{name: "fallback-horizontal-8x8", size: TransformSize8x8, class: transform.ClassHoriz},
		{name: "fused-32x32", size: TransformSize32x32, class: transform.Class2D},
	}
	for i, step := range steps {
		t.Run(step.name, func(t *testing.T) {
			base.setDecodeShape(t, step.size, step.class)
			fused.setDecodeShape(t, step.size, step.class)
			dcCtx := uint8(i % 3)
			eobCtx := uint8(i % maxEOBFlagContexts)
			variant := txbVariantTracked
			if step.class != transform.Class2D {
				variant = txbVariantGeoTrusted
			}
			rBase, errBase := base.decodeOne(variant, step.class, dcCtx, eobCtx)
			rFused, errFused := fused.decodeOneWithDequant(t, target, 0, step.class, dcCtx, eobCtx, false)
			if (errBase == nil) != (errFused == nil) || (errBase != nil && errBase.Error() != errFused.Error()) {
				t.Fatalf("baseline error=%v fused error=%v", errBase, errFused)
			}
			if errBase != nil {
				t.Fatalf("transition decode failed: %v", errBase)
			}
			if rBase != rFused || base.state.Reader.State() != fused.state.Reader.State() || base.cdfs != fused.cdfs {
				t.Fatal("transition decode record or entropy state differs")
			}
			geo, ok := coeffGeoPtr(step.size)
			if !ok {
				t.Fatal("missing transition geometry")
			}
			if step.class != transform.Class2D {
				if target.Ready || target.decoded || fused.dirtyLen != base.dirtyLen || !slices.Equal(fused.coeffs[:int(geo.maxEOB)], base.coeffs[:int(geo.maxEOB)]) {
					t.Fatal("generic H/V fallback changed quantized scratch or retained the fused target")
				}
				return
			}
			if !target.decoded || target.Ready || fused.dirtyLen != 0 {
				t.Fatalf("fused transition target=%+v coeffDirtyLen=%d", target, fused.dirtyLen)
			}
			for pos, coeff := range fused.coeffs[:int(geo.maxEOB)] {
				if coeff != 0 {
					t.Fatalf("target-only path wrote coeff[%d]=%d", pos, coeff)
				}
			}
			txSize, err := step.size.TransformSize()
			if err != nil {
				t.Fatal(err)
			}
			txScale, err := quantize.TransformScale(int(txSize.Width), int(txSize.Height))
			if err != nil {
				t.Fatal(err)
			}
			want := make([]int32, int(geo.maxEOB))
			if err := quantize.DequantizeBlockScaledBitDepthEOB(want, int(geo.scanHeight), base.coeffs, int(geo.scanHeight), base.scan, int(rBase.EOB), int(geo.scanWidth), int(geo.scanHeight), quantize.Quantizer{DC: 73, AC: 89}, txScale, 10); err != nil {
				t.Fatalf("transition dequant oracle: %v", err)
			}
			if !slices.Equal(target.Dst[:len(want)], want) {
				t.Fatal("transition dequant output differs from quantized oracle")
			}
		})
	}
}

func TestDecodeCoeffTXBDequantTargetLifecycleAndFallback(t *testing.T) {
	t.Run("skip-clears-stale-target", func(t *testing.T) {
		payload := mustTXBSkipPayload(t)
		target := &CoeffDequantTarget{Ready: true, ActiveRows: 3, TxScale: 2, decoded: true}
		got := decodeCoeffDequantLifecycleTXB(t, payload, target, nil, transform.TypeDCTDCT, false)
		want := decodeCoeffDequantLifecycleTXB(t, payload, nil, nil, transform.TypeDCTDCT, false)
		if got.err != nil || !got.result.AllZero {
			t.Fatalf("skip result=%+v err=%v", got.result, got.err)
		}
		assertCoeffDequantLifecycleStateEqual(t, got, want)
		if target.Ready || target.decoded || target.ActiveRows != 0 || target.TxScale != 0 {
			t.Fatalf("TXBSkip retained stale target metadata: %+v", target)
		}
	})

	rng := rand.New(rand.NewSource(0x8f41))
	payload := txbDiffPayload(rng, 0)
	for _, tc := range []struct {
		name      string
		typ       transform.Type
		recordErr error
	}{
		{name: "horizontal-fallback", typ: transform.TypeHDCT},
		{name: "record-error", typ: transform.TypeDCTDCT, recordErr: errors.New("test recorder failure")},
		{name: "record-success", typ: transform.TypeDCTDCT},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target := &CoeffDequantTarget{
				Dst:        make([]int32, maxCoeffScanLen),
				BitDepth:   8,
				Ready:      true,
				ActiveRows: 5,
				TxScale:    2,
				Factors:    [3][2]int32{{71, 83}, {97, 109}, {127, 139}},
			}
			for i := range target.Dst {
				target.Dst[i] = 0x13579
			}
			target.Provider = coeffDequantTestProvider{
				target:  target,
				allow:   true,
				factors: [3][2]int32{{71, 83}, {97, 109}, {127, 139}},
			}
			selector := coeffDequantLifecycleSelector{typ: tc.typ, recordErr: tc.recordErr}
			got := decodeCoeffDequantLifecycleTXB(t, payload, target, selector, tc.typ, false)
			want := decodeCoeffDequantLifecycleTXB(t, payload, nil, selector, tc.typ, false)
			if tc.recordErr != nil {
				if !errors.Is(got.err, tc.recordErr) || want.err == nil || got.err.Error() != want.err.Error() {
					t.Fatalf("record error candidate=%v baseline=%v want=%v", got.err, want.err, tc.recordErr)
				}
			} else if got.err != nil || want.err != nil || got.result.AllZero || want.result.AllZero {
				t.Fatalf("fixture did not reach nonzero selected transform: candidate=%+v/%v baseline=%+v/%v", got.result, got.err, want.result, want.err)
			}
			assertCoeffDequantLifecycleStateEqual(t, got, want)
			if tc.name == "record-success" {
				if !target.Ready || !target.decoded || target.ActiveRows <= 0 || target.TxScale != 0 {
					t.Fatalf("successful record did not publish target metadata: %+v", target)
				}
			} else if target.Ready || target.decoded || target.ActiveRows != 0 || target.TxScale != 0 {
				t.Fatalf("fallback/error retained stale target metadata: %+v", target)
			}
			if tc.name == "horizontal-fallback" && (!slices.Equal(got.coeffs, want.coeffs) || got.result != want.result) {
				t.Fatal("horizontal fallback changed quantized coefficient output")
			}
		})
	}
}

func assertCoeffDequantLifecycleStateEqual(t *testing.T, a, b coeffDequantLifecycleOutcome) {
	t.Helper()
	if a.state.Reader.State() != b.state.Reader.State() || a.cdfs != b.cdfs || a.ctx != b.ctx {
		t.Fatal("reader, CDF, or coefficient context differs")
	}
}
