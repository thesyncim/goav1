//go:build !(goexperiment.simd && (arm64 || amd64) && !purego)

package encoder

func boxInterpolateRow8(dst, s0, s1 []uint8, f int) { boxInterpolateRow(dst, s0, s1, f, 128) }

// boxColsCache and boxColsPlan are the arm64 SIMD column filter's table; the
// other builds run boxFilterCols.
type boxColsCache struct{}

type boxColsPlan struct{}

func (*boxColsCache) plan(sw, dw, x, dx, rows int) *boxColsPlan { return nil }

func (*boxColsPlan) filter(dst, s []uint8) {}
