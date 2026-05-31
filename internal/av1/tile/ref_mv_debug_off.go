//go:build !goav1_debug_refmv

package tile

const debugRefMVEnabled = false

func debugReferenceMVStack(_ ReferenceMVStackRequest, _ *ReferenceMVStackResult) {
}
