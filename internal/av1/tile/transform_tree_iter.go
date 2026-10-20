package tile

import "errors"

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

// readVariableTransformTree decodes one root of a variable transform tree
// depth first: split reads, Split bits and context marks in libaom's
// read_tx_size_vartx order.
func (s *DecodeState) readVariableTransformTree(cdfs *TransformCDFs, ctx *BlockModeContext, req TransformTreeRequest, result *TransformTreeResult, root txTreeNode) error {
	var stack txTreeStack
	var kids [4]txTreeNode
	stack.n[0] = root
	stack.depth = 1
	for stack.depth > 0 {
		stack.depth--
		n := stack.n[stack.depth]
		dims, ok := n.size.Dimensions()
		if !ok {
			return ErrInvalidDecodeState
		}
		split, err := s.ReadTransformPartitionSplit(cdfs, ctx, TransformPartitionRequest{
			Size:     req.Size,
			From:     n.size,
			Depth:    n.depth,
			X4:       uint8(n.x4),
			Y4:       uint8(n.y4),
			HaveTop:  n.haveTop,
			HaveLeft: n.haveLeft,
		})
		if err != nil {
			return err
		}
		if split {
			result.Split[n.depth] |= 1 << (n.yOff*4 + n.xOff)
		}
		if split && dims.Max > 1 {
			k, err := txTreeChildren(req, n, dims, &kids)
			if err != nil {
				return err
			}
			if err := stack.pushChildren(&kids, k); err != nil {
				return err
			}
			continue
		}
		markW, markH := dims.Log2W, dims.Log2H
		if split {
			markW, markH = 0, 0
		}
		if err := ctx.MarkTransformArea(n.x4, n.y4, int(dims.W4), int(dims.H4), markW, markH, false); err != nil {
			return err
		}
	}
	return nil
}

// collectLumaTXBsInWindow writes the luma transform blocks of a variable tree
// whose top-left lies inside window to out, in coefficient decode order (the
// order forEachLumaTXBInWindow visits them). It returns errTXBWindowFull if
// out is too small.
func (r TransformTreeResult) collectLumaTXBsInWindow(req TransformTreeRequest, window coeffUnitWindow, out *[lumaTXBWindowCap]TransformBlock) (int, error) {
	if _, err := validateTransformTreeRequest(req); err != nil {
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
