package motion

// ConvolveScratch carries reusable temporary storage for translational 2D
// inter-prediction convolution. It keeps the large int16 intermediate block out
// of hot call frames on framework decode paths. edge is the dav1d-style
// emulated-edge window (src/mc_tmpl.c emu_edge_c): clamped blocks materialize
// their halo into it once and then run the plain SIMD kernels over it.
type ConvolveScratch struct {
	im   [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
	edge [emuEdgeStride * emuEdgeRows]byte
	// imHBD is the int32 intermediate for the high-bit-depth 2D convolve
	// (convolve2DHighBD*). Keeping it caller-owned avoids zero-filling a ~69KB
	// stack array on every 2D HBD block, which dominated 10-bit decode CPU.
	imHBD [(maxBlockSize + filterTaps - 1) * maxBlockSize]int32
	// edge16 is the 16bpc emulated-edge window (dav1d src/mc_tmpl.c
	// emu_edge_c) for clamped high-bit-depth blocks, the 2-byte-sample sibling
	// of edge.
	edge16 emuEdge16Buf
}
