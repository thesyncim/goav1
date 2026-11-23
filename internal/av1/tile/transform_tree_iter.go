package tile

import (
	"encoding/binary"
	"errors"
)

// Iterative forms of the variable transform-tree walks. The tree is at most
// three levels deep (Split has two levels), so an explicit stack replaces
// recursion. Besides avoiding call overhead, this keeps the walk state and
// the decoded result on the stack under compilers whose escape analysis gives
// up on recursive methods (TinyGo heap-allocated both on every inter block).

// lumaTXBWindowCap bounds the transform blocks collected for one 64x64
// coefficient window. A window holds one maximum-size root transform per
// block, and two split levels give at most 16 leaves per root.
const lumaTXBWindowCap = 64

// errTXBWindowFull reports that a window produced more transform blocks than
// lumaTXBWindowCap; callers then fall back to the visitor walk.
var errTXBWindowFull = errors.New("tile: transform block window capacity exceeded")

type txTreeNode struct {
	size     TransformSize
	depth    uint8
	haveTop  bool
	haveLeft bool
	x4, y4   int
	xOff     int
	yOff     int
}

// txTreeChildren appends the visible children of a split node in decode
// order: top-left, top-right, bottom-left, bottom-right. The right child
// always has a left neighbour inside the parent and the bottom children a
// top neighbour.
func txTreeChildren(req TransformTreeRequest, n txTreeNode, dims TransformDimensions, kids *[4]txTreeNode) (int, error) {
	sub := dims.Sub
	subDims, ok := sub.Dimensions()
	if !ok {
		return 0, ErrInvalidDecodeState
	}
	w, h := int(subDims.W4), int(subDims.H4)
	d := n.depth + 1
	k := 0
	kids[k] = txTreeNode{size: sub, depth: d, haveTop: n.haveTop, haveLeft: n.haveLeft, x4: n.x4, y4: n.y4, xOff: n.xOff * 2, yOff: n.yOff * 2}
	k++
	if dims.Log2W >= dims.Log2H && transformChildVisible(req, n.x4+w, n.y4) {
		kids[k] = txTreeNode{size: sub, depth: d, haveTop: n.haveTop, haveLeft: true, x4: n.x4 + w, y4: n.y4, xOff: n.xOff*2 + 1, yOff: n.yOff * 2}
		k++
	}
	if dims.Log2H >= dims.Log2W && transformChildVisible(req, n.x4, n.y4+h) {
		kids[k] = txTreeNode{size: sub, depth: d, haveTop: true, haveLeft: n.haveLeft, x4: n.x4, y4: n.y4 + h, xOff: n.xOff * 2, yOff: n.yOff*2 + 1}
		k++
		if dims.Log2W >= dims.Log2H && transformChildVisible(req, n.x4+w, n.y4+h) {
			kids[k] = txTreeNode{size: sub, depth: d, haveTop: true, haveLeft: true, x4: n.x4 + w, y4: n.y4 + h, xOff: n.xOff*2 + 1, yOff: n.yOff*2 + 1}
			k++
		}
	}
	return k, nil
}

// txTreeStack is a depth-first work stack; children are pushed in reverse so
// they pop in decode order.
type txTreeStack struct {
	n     [16]txTreeNode
	depth int
}

func (s *txTreeStack) pushChildren(kids *[4]txTreeNode, k int) error {
	if s.depth+k > len(s.n) {
		return ErrInvalidDecodeState
	}
	for i := k - 1; i >= 0; i-- {
		s.n[s.depth] = kids[i]
		s.depth++
	}
	return nil
}

// txTreeReader carries the block-invariant state of dav1d's read_tx_tree
// recursion: the tile state, the split masks being built (b->tx_split0/1),
// and the block's visible extent in superblock-local 4x4 units.
type txTreeReader struct {
	s      *DecodeState
	cdfs   *TransformCDFs
	ctx    *BlockModeContext
	split  [2]uint16
	right  int
	bottom int
}

// read is dav1d's read_tx_tree (src/decode.c): the var-tx split flag of one
// node, recursing into its visible children in decode order and setting the
// above/left transform contexts at each leaf. haveTop and haveLeft gate the
// split context as libaom's edge-seeded txfm contexts do. The caller
// guarantees the node lies inside the superblock-local context arrays.
func (t *txTreeReader) read(from TransformSize, depth uint8, xOff, yOff, x4, y4 int, haveTop, haveLeft bool) {
	tDim := from.Dims()
	txw, txh := tDim.Log2W, tDim.Log2H
	ctx := t.ctx
	isSplit := false
	if depth < 2 && from > TransformSize4x4 {
		cat := 2*(int(TransformSize64x64)-int(tDim.Max)) - int(depth)
		a, l := 0, 0
		if haveTop && ctx.A.Tx[x4&(MaxBlockModeSlots-1)] < txw {
			a = 1
		}
		if haveLeft && ctx.L.Tx[y4&(MaxBlockModeSlots-1)] < txh {
			l = 1
		}
		isSplit = t.s.Reader.ReadBinaryCDFUnchecked(&t.cdfs.TxPartition[cat][a+l]) != 0
		if isSplit {
			t.split[depth&1] |= 1 << (yOff*4 + xOff)
		}
	}
	if isSplit && tDim.Max > 1 {
		sub := tDim.Sub
		subDim := sub.Dims()
		txsw, txsh := int(subDim.W4), int(subDim.H4)
		d := depth + 1
		t.read(sub, d, xOff*2, yOff*2, x4, y4, haveTop, haveLeft)
		if txw >= txh && x4+txsw < t.right {
			t.read(sub, d, xOff*2+1, yOff*2, x4+txsw, y4, haveTop, true)
		}
		if txh >= txw && y4+txsh < t.bottom {
			t.read(sub, d, xOff*2, yOff*2+1, x4, y4+txsh, true, haveLeft)
			if txw >= txh && x4+txsw < t.right {
				t.read(sub, d, xOff*2+1, yOff*2+1, x4+txsw, y4+txsh, true, true)
			}
		}
		return
	}
	markW, markH := txw, txh
	if isSplit {
		markW, markH = 0, 0
	}
	setCtxPow2(ctx.A.Tx[x4&(MaxBlockModeSlots-1):], txw, markW)
	setCtxPow2(ctx.L.Tx[y4&(MaxBlockModeSlots-1):], txh, markH)
}

// setCtxPow2 ports dav1d_memset_pow2: dst[:1<<log2n] = v with one store per
// up to eight bytes; log2n is at most 5.
func setCtxPow2(dst []uint8, log2n uint8, v uint8) {
	x := uint64(v) * 0x0101010101010101
	switch log2n {
	case 0:
		dst[0] = v
	case 1:
		binary.LittleEndian.PutUint16(dst, uint16(x))
	case 2:
		binary.LittleEndian.PutUint32(dst, uint32(x))
	case 3:
		binary.LittleEndian.PutUint64(dst, x)
	case 4:
		binary.LittleEndian.PutUint64(dst, x)
		binary.LittleEndian.PutUint64(dst[8:], x)
	default:
		binary.LittleEndian.PutUint64(dst, x)
		binary.LittleEndian.PutUint64(dst[8:], x)
		binary.LittleEndian.PutUint64(dst[16:], x)
		binary.LittleEndian.PutUint64(dst[24:], x)
	}
}

// collectLumaTXBsInWindow writes the luma transform blocks of a variable tree
// whose top-left lies inside window to out, in coefficient decode order (the
// order forEachLumaTXBInWindow visits them). It returns errTXBWindowFull if
// out is too small.
func (r TransformTreeResult) collectLumaTXBsInWindow(req TransformTreeRequest, window coeffUnitWindow, out *[lumaTXBWindowCap]TransformBlock) (int, error) {
	if _, err := validateTransformTreeRequest(&req); err != nil {
		return 0, err
	}
	if !r.Y.Valid() || !r.Variable {
		return 0, ErrInvalidDecodeState
	}
	if req.SkipTransform {
		return 0, nil
	}
	dims, ok := r.Y.Dimensions()
	if !ok {
		return 0, ErrInvalidDecodeState
	}
	reqX4 := int(req.X4)
	reqY4 := int(req.Y4)
	yStart := maxInt(int(window.Y4Start), reqY4)
	yEnd := minInt(int(window.Y4End), reqY4+int(req.VisibleH4))
	xStart := maxInt(int(window.X4Start), reqX4)
	xEnd := minInt(int(window.X4End), reqX4+int(req.VisibleW4))

	count := 0
	var stack txTreeStack
	var kids [4]txTreeNode
	for y := yStart; y < yEnd; y += int(dims.H4) {
		for x := xStart; x < xEnd; x += int(dims.W4) {
			stack.n[0] = txTreeNode{size: r.Y, x4: x, y4: y, xOff: (x - reqX4) / int(dims.W4), yOff: (y - reqY4) / int(dims.H4)}
			stack.depth = 1
			for stack.depth > 0 {
				stack.depth--
				n := stack.n[stack.depth]
				nDims, ok := n.size.Dimensions()
				if !ok {
					return count, ErrInvalidDecodeState
				}
				split := false
				if int(n.depth) < len(r.Split) && n.yOff < 4 {
					split = r.Split[n.depth]&(1<<(n.yOff*4+n.xOff)) != 0
				}
				if split && n.size > TransformSize4x4 {
					k, err := txTreeChildren(req, n, nDims, &kids)
					if err != nil {
						return count, err
					}
					if err := stack.pushChildren(&kids, k); err != nil {
						return count, err
					}
					continue
				}
				block, err := transformBlockAt(req, n.x4, n.y4, n.size)
				if err != nil {
					return count, err
				}
				if count == len(out) {
					return count, errTXBWindowFull
				}
				out[count] = block
				count++
			}
		}
	}
	return count, nil
}
