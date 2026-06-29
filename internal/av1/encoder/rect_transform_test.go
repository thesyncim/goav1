package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/quantize"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

func TestLossyEncodeStateScansCoverThinChromaMaxTransformSizes(t *testing.T) {
	var st lossyEncodeState
	if err := st.initScans(); err != nil {
		t.Fatal(err)
	}
	color420 := parser.ColorConfig{SubsamplingX: true, SubsamplingY: true}
	cases := []struct {
		name  string
		block tile.BlockSize
		color parser.ColorConfig
		plane int
		want  tile.TransformSize
	}{
		{name: "420_8x32", block: tile.BlockSize8x32, color: color420, plane: 1, want: tile.TransformSize4x16},
		{name: "420_32x8", block: tile.BlockSize32x8, color: color420, plane: 1, want: tile.TransformSize16x4},
		{name: "420_16x64", block: tile.BlockSize16x64, color: color420, plane: 1, want: tile.TransformSize8x32},
		{name: "420_64x16", block: tile.BlockSize64x16, color: color420, plane: 1, want: tile.TransformSize32x8},
		{name: "luma_16x64", block: tile.BlockSize16x64, plane: 0, want: tile.TransformSize16x64},
		{name: "luma_64x16", block: tile.BlockSize64x16, plane: 0, want: tile.TransformSize64x16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tile.MaxTransformSize(tc.block, tc.color, tc.plane)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("MaxTransformSize=%d want %d", got, tc.want)
			}
			dims, ok := got.Dimensions()
			if !ok {
				t.Fatalf("invalid transform size %d", got)
			}
			scan, ok := st.scanForTransformSize(got)
			if !ok {
				scan, ok = st.scanForThinTransformSize(got)
			}
			if !ok {
				t.Fatalf("missing scan for transform %d", got)
			}
			txSize, err := got.TransformSize()
			if err != nil {
				t.Fatal(err)
			}
			scanSize, err := transform.ScanSize(txSize)
			if err != nil {
				t.Fatal(err)
			}
			wantLen := int(scanSize.Width) * int(scanSize.Height)
			if len(scan) != wantLen {
				t.Fatalf("scan len=%d want %d", len(scan), wantLen)
			}
			if wantLen > int(dims.W4)*4*int(dims.H4)*4 {
				t.Fatalf("adjusted scan len=%d larger than transform area", wantLen)
			}
		})
	}
}

func TestForwardDCTBlockSupportsThinRectangles(t *testing.T) {
	cases := []struct {
		name string
		w    int
		h    int
	}{
		{name: "4x16", w: 4, h: 16},
		{name: "16x4", w: 16, h: 4},
		{name: "8x32", w: 8, h: 32},
		{name: "32x8", w: 32, h: 8},
		{name: "16x64", w: 16, h: 64},
		{name: "64x16", w: 64, h: 16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			size, err := transform.SizeFromDimensions(tc.w, tc.h)
			if err != nil {
				t.Fatal(err)
			}
			scanSize, err := transform.ScanSize(size)
			if err != nil {
				t.Fatal(err)
			}
			n := tc.w * tc.h
			residual := make([]int16, n)
			for i := range residual {
				residual[i] = int16((i*17)%511 - 255)
			}
			coeff := make([]int32, int(scanSize.Width)*int(scanSize.Height))
			if err := forwardDCTBlock(coeff, residual, tc.w, tc.h); err != nil {
				t.Fatalf("forwardDCTBlock(%dx%d): %v", tc.w, tc.h, err)
			}
			if coeff[0] == 0 {
				t.Fatalf("forwardDCTBlock(%dx%d) produced zero DC", tc.w, tc.h)
			}
		})
	}
}

func TestTXBlockGeometryUsesAdjusted64ExtentCoeffPlane(t *testing.T) {
	cases := []struct {
		name        string
		size        tile.TransformSize
		wantW       int
		wantH       int
		wantCoeffW  int
		wantCoeffH  int
		wantScale   uint8
		wantSamples int
		wantCoeffs  int
	}{
		{name: "16x64", size: tile.TransformSize16x64, wantW: 16, wantH: 64, wantCoeffW: 16, wantCoeffH: 32, wantScale: 1, wantSamples: 1024, wantCoeffs: 512},
		{name: "64x16", size: tile.TransformSize64x16, wantW: 64, wantH: 16, wantCoeffW: 32, wantCoeffH: 16, wantScale: 1, wantSamples: 1024, wantCoeffs: 512},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			geo, err := txBlockGeometryForTransformSize(tc.size)
			if err != nil {
				t.Fatal(err)
			}
			if geo.width != tc.wantW || geo.height != tc.wantH ||
				geo.coeffWidth != tc.wantCoeffW || geo.coeffHeight != tc.wantCoeffH ||
				geo.sampleCount != tc.wantSamples || geo.coeffCount != tc.wantCoeffs ||
				geo.txScale != tc.wantScale {
				t.Fatalf("geo=%+v", geo)
			}
			if scratchLen, err := tile.CoeffLevelsScratchLen(tc.size); err != nil {
				t.Fatal(err)
			} else if scratchLen > len(newReducedTXBTestState(t, 80).levels) {
				t.Fatalf("scratch len=%d exceeds encoder levels len=%d", scratchLen, len(newReducedTXBTestState(t, 80).levels))
			}
		})
	}
}

func TestReduced64ExtentTXBEncoderHelpers(t *testing.T) {
	cases := []struct {
		name  string
		size  tile.TransformSize
		w     int
		h     int
		block tile.BlockSize
	}{
		{name: "16x64", size: tile.TransformSize16x64, w: 16, h: 64, block: tile.BlockSize16x64},
		{name: "64x16", size: tile.TransformSize64x16, w: 64, h: 16, block: tile.BlockSize64x16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := newReducedTXBTestState(t, 80)
			scan, ok := st.scanForThinTransformSize(tc.size)
			if !ok {
				t.Fatalf("missing scan for %s", tc.name)
			}
			geo, err := txBlockGeometryForTransformSize(tc.size)
			if err != nil {
				t.Fatal(err)
			}
			if len(scan) != geo.coeffCount {
				t.Fatalf("scan len=%d coeffCount=%d", len(scan), geo.coeffCount)
			}
			src := make([]byte, tc.w*tc.h)
			recon := make([]byte, tc.w*tc.h)
			pred := make([]byte, tc.w*tc.h)
			for i := range src {
				src[i] = uint8((i*29 + 17) & 0xff)
				pred[i] = uint8((i*7 + 91) & 0xff)
			}
			ctxReq := tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: tc.block,
				Size:       tc.size,
			}
			var coeffCtx tile.CoeffEntropyContext
			if err := st.encodeTXBPredRect(recon, src, tc.w, 0, 0, tc.w, tc.h, st.yQuant, ctxReq, &coeffCtx, scan, nil, pred); err != nil {
				t.Fatalf("encodeTXBPredRect: %v", err)
			}

			st = newReducedTXBTestState(t, 80)
			qcoeff := st.lumaQ[:geo.coeffCount]
			if st.prepareInterTXB(src, pred, tc.w, tc.w, 0, 0, tc.w, tc.h, st.yQuant, qcoeff) {
				t.Fatal("prepareInterTXB unexpectedly produced all-zero coefficients")
			}
			for i := geo.coeffCount; i < geo.sampleCount; i++ {
				st.lumaQ[i] = int16(12345)
			}
			st.interTxTypeReq.Size = tc.size
			st.interTxType = transform.TypeDCTDCT
			st.afterSkipInter = func() error {
				return tile.WriteInterTransformType(st.w, &st.txCDFs, st.interTxTypeReq, st.interTxType)
			}
			var interCtx tile.CoeffEntropyContext
			recon = make([]byte, tc.w*tc.h)
			if err := st.finishInterTXB(recon, pred, tc.w, tc.w, 0, 0, tc.w, tc.h, st.yQuant, qcoeff, ctxReq, &interCtx, scan, st.afterSkipInter); err != nil {
				t.Fatalf("finishInterTXB: %v", err)
			}
		})
	}
}

func TestReduced64ExtentHighBDTXBEncoderHelper(t *testing.T) {
	cases := []struct {
		name  string
		size  tile.TransformSize
		w     int
		h     int
		block tile.BlockSize
	}{
		{name: "16x64", size: tile.TransformSize16x64, w: 16, h: 64, block: tile.BlockSize16x64},
		{name: "64x16", size: tile.TransformSize64x16, w: 64, h: 16, block: tile.BlockSize64x16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const bitDepth = 10
			st := newReducedTXBTestState(t, 80)
			q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, 80, bitDepth, quantize.PlaneY)
			if err != nil {
				t.Fatal(err)
			}
			st.yQuant = q
			scan, ok := st.scanForThinTransformSize(tc.size)
			if !ok {
				t.Fatalf("missing scan for %s", tc.name)
			}
			src := make([]uint16, tc.w*tc.h)
			recon := make([]uint16, tc.w*tc.h)
			pred := make([]uint16, tc.w*tc.h)
			for i := range src {
				src[i] = uint16((i*41 + 77) & ((1 << bitDepth) - 1))
				pred[i] = uint16((i*11 + 303) & ((1 << bitDepth) - 1))
			}
			ctxReq := tile.CoeffContextRequest{
				Plane:      0,
				PlaneBlock: tc.block,
				Size:       tc.size,
			}
			var coeffCtx tile.CoeffEntropyContext
			if err := st.encodeTXBPredRect16(recon, src, tc.w, 0, 0, tc.w, tc.h, bitDepth, st.yQuant, ctxReq, &coeffCtx, scan, nil, pred); err != nil {
				t.Fatalf("encodeTXBPredRect16: %v", err)
			}
		})
	}
}

func newReducedTXBTestState(t *testing.T, qIndex uint8) *lossyEncodeState {
	t.Helper()
	st := newTXTypeTestState(t, qIndex)
	if err := st.coeffCDFs.InitDefault(qIndex); err != nil {
		t.Fatal(err)
	}
	for plane, dst := range []*quantize.Quantizer{&st.yQuant, &st.uQuant, &st.vQuant} {
		q, err := quantize.PlaneQuantizer(parser.QuantizationParams{}, qIndex, 8, quantize.Plane(plane))
		if err != nil {
			t.Fatal(err)
		}
		*dst = q
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<16))
	st.w = &w
	return st
}
