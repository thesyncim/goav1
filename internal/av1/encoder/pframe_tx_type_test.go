package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
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
		typ := st.chooseInter8x8TXType(src, 0, 0)
		if typ != transform.TypeDCTDCT {
			t.Logf("attempt %d selected %v", attempt, typ)
			return
		}
	}
	t.Fatal("selector never chose a hybrid tx_type")
}

func newTXTypeTestState(t *testing.T, qIndex uint8) *lossyEncodeState {
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
	st.trialBuf = make([]byte, 1<<14)
	st.trialReady = true
	dcq := float64(st.yQuant.DC)
	st.rdMult = int64(dcq * dcq * (3.2 + 0.0015*dcq))
	return &st
}
