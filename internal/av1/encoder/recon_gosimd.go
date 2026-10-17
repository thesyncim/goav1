//go:build goexperiment.simd && (arm64 || amd64) && !purego

package encoder

import (
	"simd/archsimd"
	"unsafe"
)

func init() {
	if !gosimdKernelsSupported() {
		return
	}
	copyPredToReconBlockImpl = copyPredToReconBlockSIMD
	addConstantResidualToPredReconBlockImpl = addConstantResidualToPredReconBlockSIMD
}

// copyPredToReconBlockSIMD is copyPredToReconBlockPureGo with 16-byte vector
// rows (8-byte words for 8-wide blocks) instead of a memmove call per row.
func copyPredToReconBlockSIMD(reconPlane, pred []byte, predStride, stride, px, py, w, h int) {
	if h <= 0 || w&7 != 0 {
		copyPredToReconBlockPureGo(reconPlane, pred, predStride, stride, px, py, w, h)
		return
	}
	_ = reconPlane[(py+h-1)*stride+px+w-1]
	_ = pred[(h-1)*predStride+w-1]
	dp := unsafe.Pointer(&reconPlane[py*stride+px])
	sp := unsafe.Pointer(&pred[0])
	if w == 8 {
		for row := 0; row < h; row++ {
			*(*uint64)(dp) = *(*uint64)(sp)
			if row+1 < h {
				dp, sp = step(dp, stride), step(sp, predStride)
			}
		}
		return
	}
	if w&15 != 0 {
		copyPredToReconBlockPureGo(reconPlane, pred, predStride, stride, px, py, w, h)
		return
	}
	for row := 0; row < h; row++ {
		for c := 0; c < w; c += 16 {
			load16(step(sp, c)).StoreArray((*[16]uint8)(step(dp, c)))
		}
		if row+1 < h {
			dp, sp = step(dp, stride), step(sp, predStride)
		}
	}
}

// addConstantResidualToPredReconBlockSIMD is
// addConstantResidualToPredReconBlockPureGo on 16-byte vectors: with the
// residual clamped to [-255, 255] (beyond which the clamp already pins every
// sample), pred + r clamped to [0, 255] is a saturating byte add of r or a
// saturating byte subtract of -r.
func addConstantResidualToPredReconBlockSIMD(reconPlane, pred []byte, predStride, stride, px, py, w, h int, residual int16) {
	if h <= 0 || (w&15 != 0 && (w != 8 || h&1 != 0)) {
		addConstantResidualToPredReconBlockPureGo(reconPlane, pred, predStride, stride, px, py, w, h, residual)
		return
	}
	_ = reconPlane[(py+h-1)*stride+px+w-1]
	_ = pred[(h-1)*predStride+w-1]
	dp := unsafe.Pointer(&reconPlane[py*stride+px])
	sp := unsafe.Pointer(&pred[0])
	r := max(min(int(residual), 255), -255)
	if w == 8 {
		// Two 8-byte rows per vector, staged through memory (cheaper than
		// lane inserts on Apple cores).
		var pair [16]uint8
		add := archsimd.BroadcastUint8x16(uint8(max(r, 0)))
		sub := archsimd.BroadcastUint8x16(uint8(max(-r, 0)))
		for row := 0; row < h; row += 2 {
			*(*uint64)(unsafe.Pointer(&pair[0])) = *(*uint64)(sp)
			*(*uint64)(unsafe.Pointer(&pair[8])) = *(*uint64)(step(sp, predStride))
			archsimd.LoadUint8x16Array(&pair).AddSaturated(add).SubSaturated(sub).StoreArray(&pair)
			*(*uint64)(dp) = *(*uint64)(unsafe.Pointer(&pair[0]))
			*(*uint64)(step(dp, stride)) = *(*uint64)(unsafe.Pointer(&pair[8]))
			if row+2 < h {
				dp, sp = step(dp, 2*stride), step(sp, 2*predStride)
			}
		}
		return
	}
	if r >= 0 {
		v := archsimd.BroadcastUint8x16(uint8(r))
		for row := 0; row < h; row++ {
			for c := 0; c < w; c += 16 {
				load16(step(sp, c)).AddSaturated(v).StoreArray((*[16]uint8)(step(dp, c)))
			}
			if row+1 < h {
				dp, sp = step(dp, stride), step(sp, predStride)
			}
		}
		return
	}
	v := archsimd.BroadcastUint8x16(uint8(-r))
	for row := 0; row < h; row++ {
		for c := 0; c < w; c += 16 {
			load16(step(sp, c)).SubSaturated(v).StoreArray((*[16]uint8)(step(dp, c)))
		}
		if row+1 < h {
			dp, sp = step(dp, stride), step(sp, predStride)
		}
	}
}
