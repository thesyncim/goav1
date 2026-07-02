package motion

// ConvolveScratch carries reusable temporary storage for translational 2D
// inter-prediction convolution. It keeps the large int16 intermediate block out
// of hot call frames on framework decode paths. edge is the dav1d-style
// emulated-edge window (src/mc_tmpl.c emu_edge_c): clamped blocks materialize
// their halo into it once and then run the plain SIMD kernels over it.
type ConvolveScratch struct {
	im   [(maxBlockSize + filterTaps - 1) * maxBlockSize]int16
	edge [emuEdgeStride * emuEdgeRows]byte
}
