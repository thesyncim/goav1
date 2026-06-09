package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestWriteInterRefSymbolsRoundTrip is the oracle gate for the is-inter flag
// and single-reference writers: random decisions against a randomized shared
// neighbor context decode back exactly through ReadIntraFlagResult and
// ReadInterReferences, decoder CDFs adapting in lockstep. Both SINGLE (no
// reference-mode symbol) and SELECT (explicit single symbol) frame modes are
// covered across all seven single references.
func TestWriteInterRefSymbolsRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(29))

	var ctx BlockModeContext
	for i := range MaxBlockModeSlots {
		ctx.AboveIntra[i] = uint8(rng.Intn(2))
		ctx.LeftIntra[i] = uint8(rng.Intn(2))
		for r := range 2 {
			ctx.AboveRef[r][i] = ReferenceFrame(rng.Intn(7))
			ctx.LeftRef[r][i] = ReferenceFrame(rng.Intn(7))
		}
		ctx.AboveCompound[i] = uint8(rng.Intn(2))
		ctx.LeftCompound[i] = uint8(rng.Intn(2))
	}

	singles := []ReferenceFrame{
		ReferenceFrameLast, ReferenceFrameLast2, ReferenceFrameLast3,
		ReferenceFrameGolden, ReferenceFrameBWD, ReferenceFrameAltref2, ReferenceFrameAltref,
	}
	refModes := []parser.ReferenceMode{parser.ReferenceModeSingle, parser.ReferenceModeSelect}

	type op struct {
		kind  int // 0 intra flag, 1 references
		flag  IntraFlagRequest
		intra bool

		refReq InterReferenceRequest
		refs   InterReferencesResult
	}
	const n = 5000
	ops := make([]op, 0, n)

	var encIntra IntraModeCDFs
	var encRefs InterRefCDFs
	if err := encIntra.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := encRefs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	w := entropy.NewWriter(make([]byte, 0, 1<<15))

	for range n {
		x4 := uint8(rng.Intn(MaxBlockModeSlots))
		y4 := uint8(rng.Intn(MaxBlockModeSlots))
		haveTop := rng.Intn(2) == 0
		haveLeft := rng.Intn(2) == 0
		var o op
		if rng.Intn(2) == 0 {
			o = op{kind: 0, flag: IntraFlagRequest{
				FrameType: parser.FrameTypeInter,
				X4:        x4, Y4: y4, HaveTop: haveTop, HaveLeft: haveLeft,
			}, intra: rng.Intn(2) == 0}
			if err := WriteIntraFlag(&w, &encIntra, &ctx, o.flag, o.intra); err != nil {
				t.Fatalf("WriteIntraFlag: %v", err)
			}
		} else {
			req := InterReferenceRequest{
				Size:          BlockSize(rng.Intn(int(blockSizeCount))),
				ReferenceMode: refModes[rng.Intn(len(refModes))],
				X4:            x4, Y4: y4, HaveTop: haveTop, HaveLeft: haveLeft,
			}
			refs := InterReferencesResult{Ref: [2]ReferenceFrame{singles[rng.Intn(len(singles))], ReferenceFrameNone}}
			o = op{kind: 1, refReq: req, refs: refs}
			if err := WriteInterReferences(&w, &encRefs, &ctx, req, refs); err != nil {
				t.Fatalf("WriteInterReferences size=%v mode=%v ref=%v: %v", req.Size, req.ReferenceMode, refs.Ref[0], err)
			}
		}
		ops = append(ops, o)
	}
	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decIntra IntraModeCDFs
	var decRefs InterRefCDFs
	if err := decIntra.InitDefault(); err != nil {
		t.Fatal(err)
	}
	if err := decRefs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}
	for i, o := range ops {
		switch o.kind {
		case 0:
			got, err := state.ReadIntraFlagResult(&decIntra, &ctx, o.flag)
			if err != nil {
				t.Fatalf("op %d ReadIntraFlagResult: %v", i, err)
			}
			if got.Intra != o.intra {
				t.Fatalf("op %d intra=%v want %v", i, got.Intra, o.intra)
			}
		default:
			got, err := state.ReadInterReferences(&decRefs, &ctx, o.refReq)
			if err != nil {
				t.Fatalf("op %d ReadInterReferences: %v", i, err)
			}
			if got.Compound != o.refs.Compound || got.Ref != o.refs.Ref {
				t.Fatalf("op %d refs=%+v want %+v", i, got, o.refs)
			}
		}
	}
}
