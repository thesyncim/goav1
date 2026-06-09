package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestWriteModeSymbolsRoundTrip is the oracle gate for the block mode-symbol
// writers: a mixed stream of skip_transform, luma intra mode (keyframe and
// inter), angle delta, and chroma intra mode (incl. CfL alphas) decisions coded
// with adapting CDFs must decode back exactly through the corresponding readers,
// with decoder CDFs and the shared neighbor context in lockstep.
func TestWriteModeSymbolsRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(13))

	// Shared neighbor context with randomized skip and mode neighbors. The
	// writers and readers derive contexts from the same object, so any fill
	// exercises the derivation as long as both sides see identical state.
	var ctx BlockModeContext
	for i := range MaxBlockModeSlots {
		ctx.AboveSkip[i] = uint8(rng.Intn(2))
		ctx.LeftSkip[i] = uint8(rng.Intn(2))
		ctx.AboveMode[i] = IntraMode(rng.Intn(int(intraModeCount)))
		ctx.LeftMode[i] = IntraMode(rng.Intn(int(intraModeCount)))
	}

	type op struct {
		kind int // 0 skip, 1 skip forced (no symbol), 2 luma key, 3 luma inter, 4 angle, 5 chroma
		req  BlockModeRequest
		luma LumaIntraModeRequest
		ang  IntraAngleDeltaRequest
		chr  ChromaIntraModeRequest

		skip  bool
		mode  IntraMode
		delta int8
		uv    ChromaIntraMode
		alpha CFLAlphaResult
	}

	allSizes := make([]BlockSize, 0, int(blockSizeCount))
	for s := BlockSize(0); s < blockSizeCount; s++ {
		allSizes = append(allSizes, s)
	}

	const n = 6000
	ops := make([]op, 0, n)

	var encMode BlockModeCDFs
	var encIntra IntraModeCDFs
	if err := encMode.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := encIntra.InitDefault(); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<16))

	for range n {
		size := allSizes[rng.Intn(len(allSizes))]
		x4 := uint8(rng.Intn(MaxBlockModeSlots))
		y4 := uint8(rng.Intn(MaxBlockModeSlots))
		var o op
		switch rng.Intn(6) {
		case 0:
			o = op{kind: 0, req: BlockModeRequest{Size: size, X4: x4, Y4: y4}, skip: rng.Intn(2) == 0}
			if err := WriteSkipTransform(&w, &encMode, &ctx, o.req, false, o.skip); err != nil {
				t.Fatalf("WriteSkipTransform: %v", err)
			}
		case 1:
			// Segment-forced skip: no symbol coded, decoder derives true.
			o = op{kind: 1, req: BlockModeRequest{Size: size, X4: x4, Y4: y4, SegmentationEnabled: true, Segment: parser.SegmentData{Skip: true}}, skip: true}
			if err := WriteSkipTransform(&w, &encMode, &ctx, o.req, false, true); err != nil {
				t.Fatalf("WriteSkipTransform forced: %v", err)
			}
		case 2:
			o = op{kind: 2, luma: LumaIntraModeRequest{FrameType: parser.FrameTypeKey, Size: size, X4: x4, Y4: y4},
				mode: IntraMode(rng.Intn(int(intraModeCount)))}
			if err := WriteLumaIntraMode(&w, &encIntra, &ctx, o.luma, o.mode); err != nil {
				t.Fatalf("WriteLumaIntraMode key: %v", err)
			}
		case 3:
			o = op{kind: 3, luma: LumaIntraModeRequest{FrameType: parser.FrameTypeInter, Size: size, X4: x4, Y4: y4},
				mode: IntraMode(rng.Intn(int(intraModeCount)))}
			if err := WriteLumaIntraMode(&w, &encIntra, &ctx, o.luma, o.mode); err != nil {
				t.Fatalf("WriteLumaIntraMode inter: %v", err)
			}
		case 4:
			mode := IntraMode(rng.Intn(int(intraModeCount)))
			o = op{kind: 4, ang: IntraAngleDeltaRequest{Size: size, Mode: mode}}
			if write, err := shouldReadIntraAngleDelta(size, mode); err == nil && write {
				o.delta = int8(rng.Intn(2*AngleDeltaMax+1) - AngleDeltaMax)
			}
			if err := WriteIntraAngleDelta(&w, &encIntra, o.ang, o.delta); err != nil {
				t.Fatalf("WriteIntraAngleDelta: %v", err)
			}
		default:
			luma := IntraMode(rng.Intn(int(intraModeCount)))
			cflAllowed := rng.Intn(2) == 0
			modes := int(chromaIntraModeCount) - 1
			if cflAllowed {
				modes = int(chromaIntraModeCount)
			}
			uv := ChromaIntraMode(rng.Intn(modes))
			var alpha CFLAlphaResult
			if uv == ChromaIntraModeCFL {
				alpha = randomCFLAlpha(rng)
			}
			o = op{kind: 5, chr: ChromaIntraModeRequest{Size: size, LumaMode: luma, CFLAllowed: cflAllowed}, uv: uv, alpha: alpha}
			if err := WriteChromaIntraMode(&w, &encIntra, o.chr, o.uv, o.alpha); err != nil {
				t.Fatalf("WriteChromaIntraMode: %v", err)
			}
		}
		ops = append(ops, o)
	}

	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decMode BlockModeCDFs
	var decIntra IntraModeCDFs
	if err := decMode.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := decIntra.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	for i, o := range ops {
		switch o.kind {
		case 0, 1:
			got, err := state.ReadSkipTransform(&decMode, &ctx, o.req, false)
			if err != nil {
				t.Fatalf("op %d ReadSkipTransform: %v", i, err)
			}
			if got != o.skip {
				t.Fatalf("op %d skip=%v want %v", i, got, o.skip)
			}
		case 2, 3:
			got, err := state.ReadLumaIntraMode(&decIntra, &ctx, o.luma)
			if err != nil {
				t.Fatalf("op %d ReadLumaIntraMode: %v", i, err)
			}
			if got != o.mode {
				t.Fatalf("op %d luma mode=%d want %d", i, got, o.mode)
			}
		case 4:
			got, err := state.ReadIntraAngleDelta(&decIntra, o.ang)
			if err != nil {
				t.Fatalf("op %d ReadIntraAngleDelta: %v", i, err)
			}
			if got != o.delta {
				t.Fatalf("op %d angle delta=%d want %d", i, got, o.delta)
			}
		default:
			gotMode, gotAlpha, err := state.ReadChromaIntraMode(&decIntra, o.chr)
			if err != nil {
				t.Fatalf("op %d ReadChromaIntraMode: %v", i, err)
			}
			if gotMode != o.uv {
				t.Fatalf("op %d uv mode=%d want %d", i, gotMode, o.uv)
			}
			if gotAlpha != o.alpha {
				t.Fatalf("op %d cfl alpha=%+v want %+v", i, gotAlpha, o.alpha)
			}
		}
	}
}

// randomCFLAlpha builds a valid CFLAlphaResult: a joint sign in [0, CFLJointSigns)
// and, for each plane whose sign is non-zero, a random alpha index nibble.
func randomCFLAlpha(rng *rand.Rand) CFLAlphaResult {
	jointSign := rng.Intn(CFLJointSigns)
	var idx uint8
	if cflSignU(jointSign) != cflSignZero {
		idx = uint8(rng.Intn(CFLAlphabetSize)) << 4
	}
	if cflSignV(jointSign) != cflSignZero {
		idx += uint8(rng.Intn(CFLAlphabetSize))
	}
	return CFLAlphaResult{Index: idx, JointSign: int8(jointSign)}
}
