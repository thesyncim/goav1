package tile

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/entropy"
	"github.com/thesyncim/goav1/internal/av1/parser"
)

// TestWriteTransformTreeRoundTrip proves WriteTransformTree is the exact
// forward of DecodeTransformTree: a random mix of switchable inter var-tx
// trees (random one- and two-level splits), skip blocks, and intra selected
// transform sizes round-trips bit-exactly with identical adaptive CDF
// evolution and identical transform context arrays on both sides.
func TestWriteTransformTreeRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(59))
	color := parser.ColorConfig{BitDepth: 8, SubsamplingX: true, SubsamplingY: true}
	sizes := []BlockSize{BlockSize8x8, BlockSize16x16, BlockSize32x32, BlockSize64x64}

	type op struct {
		req  TransformTreeRequest
		tree TransformTreeResult
	}

	var encCDFs TransformCDFs
	if err := encCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var encCtx BlockModeContext
	w := entropy.NewWriter(make([]byte, 0, 1<<16))

	const n = 4000
	ops := make([]op, 0, n)
	for range n {
		size := sizes[rng.Intn(len(sizes))]
		dims, _ := size.Dimensions()
		x4 := uint8(rng.Intn(MaxBlockModeSlots - int(dims.W4) + 1))
		y4 := uint8(rng.Intn(MaxBlockModeSlots - int(dims.H4) + 1))
		req := TransformTreeRequest{
			Size:          size,
			X4:            x4,
			Y4:            y4,
			VisibleW4:     dims.W4,
			VisibleH4:     dims.H4,
			HaveTop:       rng.Intn(2) == 0,
			HaveLeft:      rng.Intn(2) == 0,
			Color:         color,
			TransformMode: parser.TransformModeSwitchable,
			Inter:         rng.Intn(3) != 0,
			SkipTransform: rng.Intn(4) == 0,
		}
		var tree TransformTreeResult
		maxY, err := MaxTransformSize(size, parser.ColorConfig{}, 0)
		if err != nil {
			t.Fatal(err)
		}
		tree.Y = maxY
		if req.Inter && !req.SkipTransform {
			// Random splits on the nodes the walk actually visits: square
			// blocks have one depth-0 node (bit 0) and, when split, four
			// depth-1 children at offsets {0,1,4,5}. No symbol is coded
			// below depth 2 or at 4x4, so deeper masks must stay clear.
			tree.Split[0] = uint16(rng.Intn(2))
			if md, _ := maxY.Dimensions(); tree.Split[0] != 0 && md.Sub > TransformSize4x4 {
				for _, b := range []int{0, 1, 4, 5} {
					if rng.Intn(2) == 0 {
						tree.Split[1] |= 1 << b
					}
				}
			}
		} else if !req.Inter {
			// Random selected depth (max two steps, never below 4x4).
			steps := rng.Intn(3)
			for range steps {
				sd, ok := tree.Y.Dimensions()
				if !ok || sd.Sub == tree.Y {
					break
				}
				tree.Y = sd.Sub
			}
		}
		if _, err := WriteTransformTree(&w, &encCDFs, &encCtx, req, tree); err != nil {
			t.Fatalf("WriteTransformTree req=%+v tree=%+v: %v", req, tree, err)
		}
		ops = append(ops, op{req: req, tree: tree})
	}

	buf, err := w.Finish()
	if err != nil {
		t.Fatal(err)
	}

	var decCDFs TransformCDFs
	if err := decCDFs.InitDefault(); err != nil {
		t.Fatal(err)
	}
	var decCtx BlockModeContext
	var state DecodeState
	if err := state.Reset(buf, Job{Offset: 0, Size: uint32(len(buf))}, DecodeOptions{}); err != nil {
		t.Fatal(err)
	}

	for i, o := range ops {
		got, err := state.DecodeTransformTree(&decCDFs, &decCtx, o.req)
		if err != nil {
			t.Fatalf("op %d: DecodeTransformTree req=%+v: %v", i, o.req, err)
		}
		if o.req.Inter && !o.req.SkipTransform {
			if got.Split != o.tree.Split {
				t.Fatalf("op %d: split %v want %v (req=%+v)", i, got.Split, o.tree.Split, o.req)
			}
		} else if !o.req.Inter {
			if got.Y != o.tree.Y {
				t.Fatalf("op %d: tx %v want %v (req=%+v)", i, got.Y, o.tree.Y, o.req)
			}
		}
	}

	if encCtx.AboveTx != decCtx.AboveTx || encCtx.LeftTx != decCtx.LeftTx ||
		encCtx.AboveTxIntra != decCtx.AboveTxIntra || encCtx.LeftTxIntra != decCtx.LeftTxIntra {
		t.Fatal("transform context arrays diverge between writer and reader")
	}
}
