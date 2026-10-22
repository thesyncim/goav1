package tile

// This file ports dav1d's src/intra_edge.{c,h}: a static tree, built once,
// that carries per-partition-position edge availability flags down the
// partition walk so each block learns whether its top-right and bottom-left
// neighbours are already decoded without recomputing it from its position.

// Edge flags (dav1d enum EdgeFlags).
const (
	edgeI444TopHasRight   uint8 = 1 << 0
	edgeI422TopHasRight   uint8 = 1 << 1
	edgeI420TopHasRight   uint8 = 1 << 2
	edgeI444LeftHasBottom uint8 = 1 << 3
	edgeI422LeftHasBottom uint8 = 1 << 4
	edgeI420LeftHasBottom uint8 = 1 << 5

	edgeAllTopHasRight   = edgeI444TopHasRight | edgeI422TopHasRight | edgeI420TopHasRight
	edgeAllLeftHasBottom = edgeI444LeftHasBottom | edgeI422LeftHasBottom | edgeI420LeftHasBottom
	edgeAllTRAndBL       = edgeAllTopHasRight | edgeAllLeftHasBottom
)

// intraEdgeNode merges dav1d's EdgeNode, EdgeBranch and EdgeTip: o/h/v for
// every node, h4/v4 and the split children for branches (16x16 and up), and
// the three split flags for 8x8 tips.
type intraEdgeNode struct {
	o     uint8
	h     [2]uint8
	v     [2]uint8
	h4    uint8
	v4    uint8
	tip   [3]uint8
	split [4]uint16
}

const (
	intraEdgeNodesSB128 = 1 + 4 + 16 + 64 + 256
	intraEdgeNodesSB64  = 1 + 4 + 16 + 64
)

var intraEdgeTree struct {
	nodes [intraEdgeNodesSB128 + intraEdgeNodesSB64]intraEdgeNode
	n     uint16
	root  [2]uint16 // BlockLevel128x128, BlockLevel64x64
}

func init() {
	intraEdgeTree.root[0] = initIntraEdgeModeNode(BlockLevel128x128, true, false)
	intraEdgeTree.root[1] = initIntraEdgeModeNode(BlockLevel64x64, true, false)
}

// intraEdgeRoot returns the tree root for a superblock root level.
// Roots below 64x64 (test-only walks) start at the top-left descendant of the
// 64x64 tree at that level.
func intraEdgeRoot(root BlockLevel) uint16 {
	if root == BlockLevel128x128 {
		return intraEdgeTree.root[0]
	}
	idx := intraEdgeTree.root[1]
	for l := BlockLevel64x64; l < root && l < BlockLevel8x8; l++ {
		idx = intraEdgeTree.nodes[idx].split[0]
	}
	return idx
}

func newIntraEdgeNode() uint16 {
	idx := intraEdgeTree.n
	intraEdgeTree.n++
	return idx
}

// initIntraEdges ports init_edges.
func initIntraEdges(node *intraEdgeNode, level BlockLevel, flags uint8) {
	node.o = flags
	node.h[0] = flags | edgeAllLeftHasBottom
	node.v[0] = flags | edgeAllTopHasRight
	if level == BlockLevel8x8 {
		node.h[1] = flags & (edgeAllLeftHasBottom | edgeI420TopHasRight)
		node.v[1] = flags & (edgeAllTopHasRight | edgeI420LeftHasBottom | edgeI422LeftHasBottom)
		node.tip[0] = (flags & edgeAllTopHasRight) | edgeI422LeftHasBottom
		node.tip[1] = flags | edgeI444TopHasRight
		node.tip[2] = flags & (edgeI420TopHasRight | edgeI420LeftHasBottom | edgeI422LeftHasBottom)
		return
	}
	node.h[1] = flags & edgeAllLeftHasBottom
	node.v[1] = flags & edgeAllTopHasRight
	node.h4 = edgeAllLeftHasBottom
	node.v4 = edgeAllTopHasRight
	if level == BlockLevel16x16 {
		node.h4 |= flags & edgeI420TopHasRight
		node.v4 |= flags & (edgeI420LeftHasBottom | edgeI422LeftHasBottom)
	}
}

func intraEdgeFlags(topHasRight, leftHasBottom bool) uint8 {
	var flags uint8
	if topHasRight {
		flags |= edgeAllTopHasRight
	}
	if leftHasBottom {
		flags |= edgeAllLeftHasBottom
	}
	return flags
}

// initIntraEdgeModeNode ports init_mode_node and returns the node index.
func initIntraEdgeModeNode(level BlockLevel, topHasRight, leftHasBottom bool) uint16 {
	idx := newIntraEdgeNode()
	initIntraEdges(&intraEdgeTree.nodes[idx], level, intraEdgeFlags(topHasRight, leftHasBottom))
	for n := range 4 {
		childTopHasRight := !(n == 3 || (n == 1 && !topHasRight))
		childLeftHasBottom := n == 0 || (n == 2 && leftHasBottom)
		var child uint16
		if level == BlockLevel16x16 {
			child = newIntraEdgeNode()
			initIntraEdges(&intraEdgeTree.nodes[child], level+1, intraEdgeFlags(childTopHasRight, childLeftHasBottom))
		} else {
			child = initIntraEdgeModeNode(level+1, childTopHasRight, childLeftHasBottom)
		}
		intraEdgeTree.nodes[idx].split[n] = child
	}
	return idx
}
