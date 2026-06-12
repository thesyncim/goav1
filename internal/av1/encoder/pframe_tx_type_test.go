package encoder

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestChooseInter8x8TXTypeCanSelectHybrid(t *testing.T) {
	st := newTXTypeTestState(t, 96)
	src := SourceFrame420{
		Y:            make([]byte, 64),
		U:            make([]byte, 16),
		V:            make([]byte, 16),
		YStride:      8,
		ChromaStride: 4,
		Width:        8,
		Height:       8,
	}
	for i := range st.predY[:64] {
		st.predY[i] = 128
	}
	for i := range st.predU[:16] {
		st.predU[i] = 128
		st.predV[i] = 128
		src.U[i] = 128
		src.V[i] = 128
	}

	rng := rand.New(rand.NewSource(37))
	for attempt := range 2048 {
		for i := range src.Y {
			src.Y[i] = uint8(rng.Intn(256))
		}
		dctDcode := prepareTXTypeDCTCandidate(st, src)
		typ := st.chooseInter8x8TXType(src, 0, 0, dctDcode)
		if typ != transform.TypeDCTDCT {
			t.Logf("attempt %d selected %v", attempt, typ)
			return
		}
	}
	t.Fatal("selector never chose a hybrid tx_type")
}

func TestChooseInter8x8TXTypeSeededMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	for attempt := range 128 {
		seeded := newTXTypeTestState(t, 96)
		reference := newTXTypeTestState(t, 96)
		src := SourceFrame420{
			Y:            make([]byte, 64),
			U:            make([]byte, 16),
			V:            make([]byte, 16),
			YStride:      8,
			ChromaStride: 4,
			Width:        8,
			Height:       8,
		}
		for i := range src.Y {
			src.Y[i] = uint8(rng.Intn(256))
			seeded.predY[i] = uint8(rng.Intn(256))
			reference.predY[i] = seeded.predY[i]
		}
		for i := range src.U {
			src.U[i] = uint8(rng.Intn(256))
			src.V[i] = uint8(rng.Intn(256))
			seeded.predU[i] = uint8(rng.Intn(256))
			seeded.predV[i] = uint8(rng.Intn(256))
			reference.predU[i] = seeded.predU[i]
			reference.predV[i] = seeded.predV[i]
		}

		dctDcode := prepareTXTypeDCTCandidate(seeded, src)
		got := seeded.chooseInter8x8TXType(src, 0, 0, dctDcode)
		want := chooseInter8x8TXTypeReference(reference, src, 0, 0)
		if got != want ||
			!slices.Equal(seeded.lumaQ[:64], reference.lumaQ[:64]) ||
			!slices.Equal(seeded.uQ[:16], reference.uQ[:16]) ||
			!slices.Equal(seeded.vQ[:16], reference.vQ[:16]) {
			t.Fatalf("attempt %d: seeded type=%v reference type=%v qcoeffs equal Y=%t U=%t V=%t",
				attempt, got, want,
				slices.Equal(seeded.lumaQ[:64], reference.lumaQ[:64]),
				slices.Equal(seeded.uQ[:16], reference.uQ[:16]),
				slices.Equal(seeded.vQ[:16], reference.vQ[:16]))
		}
	}
}

func TestTrialTXBBitsInterMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(91))
	for attempt := range 128 {
		qcoeff := make([]int16, 64)
		for i := range qcoeff {
			if rng.Intn(4) != 0 {
				continue
			}
			v := int16(1 + rng.Intn(7))
			if rng.Intn(2) == 0 {
				v = -v
			}
			qcoeff[i] = v
		}
		for _, typ := range [...]transform.Type{
			transform.TypeDCTDCT,
			transform.TypeADSTDCT,
			transform.TypeDCTADST,
			transform.TypeADSTADST,
			transform.TypeIDTX,
		} {
			fast := newTXTypeTestState(t, 96)
			reference := newTXTypeTestState(t, 96)
			txBefore := fast.txCDFs
			got := fast.trialTXBBitsInter(qcoeff, 8, tile.TransformSize8x8, typ)
			want := trialTXBBitsInterReference(reference, qcoeff, 8, tile.TransformSize8x8, typ)
			if got != want {
				t.Fatalf("attempt %d typ %v: trial bits=%d want %d", attempt, typ, got, want)
			}
			if fast.txCDFs != txBefore {
				t.Fatalf("attempt %d typ %v: trial mutated transform cdfs", attempt, typ)
			}
		}
	}
}

func BenchmarkChooseInter8x8TXType(b *testing.B) {
	st := newTXTypeTestState(b, 96)
	src := SourceFrame420{
		Y:            make([]byte, 64),
		U:            make([]byte, 16),
		V:            make([]byte, 16),
		YStride:      8,
		ChromaStride: 4,
		Width:        8,
		Height:       8,
	}
	rng := rand.New(rand.NewSource(109))
	for i := range src.Y {
		src.Y[i] = uint8(rng.Intn(256))
		st.predY[i] = uint8(rng.Intn(256))
	}
	for i := range src.U {
		src.U[i] = uint8(rng.Intn(256))
		src.V[i] = uint8(rng.Intn(256))
		st.predU[i] = uint8(rng.Intn(256))
		st.predV[i] = uint8(rng.Intn(256))
	}
	b.ReportAllocs()
	b.ResetTimer()
	var typ transform.Type
	for range b.N {
		dctDcode := prepareTXTypeDCTCandidate(st, src)
		typ = st.chooseInter8x8TXType(src, 0, 0, dctDcode)
	}
	if typ == transform.TypeCount {
		b.Fatal(typ)
	}
}

func prepareTXTypeDCTCandidate(st *lossyEncodeState, src SourceFrame420) int64 {
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(src.Y, st.predY[:64], 8, src.YStride, 0, 0, 8, 8, st.yQuant, st.lumaQ[:64])
	st.prepareInterTXB(src.U, st.predU[:16], 4, src.ChromaStride, 0, 0, 4, 4, st.uQuant, st.uQ[:16])
	st.prepareInterTXB(src.V, st.predV[:16], 4, src.ChromaStride, 0, 0, 4, 4, st.vQuant, st.vQ[:16])
	return st.rdDcode
}

func chooseInter8x8TXTypeReference(st *lossyEncodeState, src SourceFrame420, lumaPX, lumaPY int) transform.Type {
	if !st.armTrial() {
		return transform.TypeDCTDCT
	}
	bestType := transform.TypeDCTDCT
	bestCost := int64(1 << 62)
	saveDcode, saveDskip, saveRcode := st.rdDcode, st.rdDskip, st.rdRcode
	baseTrialCDFs := st.trialCDFs
	tmpY := st.lumaQ2[:64]
	tmpU := st.lumaQ2[64:80]
	tmpV := st.lumaQ2[80:96]
	for _, typ := range [...]transform.Type{
		transform.TypeDCTDCT,
		transform.TypeADSTDCT,
		transform.TypeDCTADST,
		transform.TypeADSTADST,
		transform.TypeIDTX,
	} {
		st.trialCDFs = baseTrialCDFs
		st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
		lumaZero := st.prepareInterTXBTyped(src.Y, st.predY[:64], 8, src.YStride, lumaPX, lumaPY, 8, 8, st.yQuant, tmpY, typ)
		if typ != transform.TypeDCTDCT && lumaZero {
			continue
		}
		st.prepareInterTXBTyped(src.U, st.predU[:16], 4, src.ChromaStride, lumaPX/2, lumaPY/2, 4, 4, st.uQuant, tmpU, typ)
		st.prepareInterTXBTyped(src.V, st.predV[:16], 4, src.ChromaStride, lumaPX/2, lumaPY/2, 4, 4, st.vQuant, tmpV, typ)
		cost := st.rdDcode << 7
		cost += st.trialTXBBitsInter(tmpY, 8, tile.TransformSize8x8, typ)
		cost += st.trialTXBBits(tile.CoeffPlaneUV, tmpU, 4)
		cost += st.trialTXBBits(tile.CoeffPlaneUV, tmpV, 4)
		if cost < bestCost {
			bestCost = cost
			bestType = typ
			copy(st.lumaQ[:64], tmpY)
			copy(st.uQ[:16], tmpU)
			copy(st.vQ[:16], tmpV)
		}
	}
	st.rdDcode, st.rdDskip, st.rdRcode = saveDcode, saveDskip, saveRcode
	st.trialCDFs = baseTrialCDFs
	return bestType
}

func trialTXBBitsInterReference(st *lossyEncodeState, qcoeff []int16, n int, size tile.TransformSize, typ transform.Type) int64 {
	scan := st.scan4
	if n == 8 {
		scan = st.scan8
	}
	buf := make([]byte, 0, 1<<14)
	tw := entropy.NewWriter(buf)
	txCDFs := st.txCDFs
	txReq := tile.InterTransformTypeRequest{
		Size:        size,
		QIndexKnown: true,
		QIndex:      st.qIndex,
	}
	afterSkip := func() error {
		return tile.WriteInterTransformType(&tw, &txCDFs, txReq, typ)
	}
	base := tw.Tell()
	if _, err := tile.WriteCoefficientsTXB(&tw, &st.trialCDFs, tile.TXBEncodeRequest{
		Size: size, Plane: tile.CoeffPlaneY, Class: transform.Class2D, AfterSkip: afterSkip,
	}, qcoeff[:n*n], scan, st.levels); err != nil {
		return 1 << 59
	}
	bits := int64(tw.Tell() - base)
	return ((bits<<9)*st.rdMult + 256) >> 9
}

func newTXTypeTestState(t interface {
	Helper()
	Fatal(args ...any)
}, qIndex uint8) *lossyEncodeState {
	t.Helper()
	var st lossyEncodeState
	st.qIndex = qIndex
	st.color = parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	for plane, dst := range []*quantize.Quantizer{&st.yQuant, &st.uQuant, &st.vQuant} {
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.Plane(plane))
		if err != nil {
			t.Fatal(err)
		}
		*dst = q
	}
	if err := st.initScans(); err != nil {
		t.Fatal(err)
	}
	if err := st.txCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := st.trialCDFs.InitDefault(qIndex); err != nil {
		t.Fatal(err)
	}
	st.trialReady = true
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	return &st
}
