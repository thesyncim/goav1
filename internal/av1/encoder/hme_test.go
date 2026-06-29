package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

func TestBuildQuarterPlaneMatchesPureGo(t *testing.T) {
	for _, tc := range []struct {
		name      string
		qw, qh    int
		srcStride int
		salt      int
	}{
		{name: "tiny scalar", qw: 7, qh: 5, srcStride: 37, salt: 3},
		{name: "neon aligned", qw: 32, qh: 9, srcStride: 160, salt: 11},
		{name: "neon with tail", qw: 31, qh: 7, srcStride: 133, salt: 23},
		{name: "1080p row shape", qw: 480, qh: 8, srcStride: 1920, salt: 41},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := make([]byte, tc.srcStride*tc.qh*4)
			for y := 0; y < tc.qh*4; y++ {
				for x := 0; x < tc.qw*4; x++ {
					src[y*tc.srcStride+x] = byte((x*29 + y*17 + x*y*tc.salt + 97) & 255)
				}
			}
			got := make([]byte, tc.qw*tc.qh)
			want := make([]byte, tc.qw*tc.qh)
			buildQuarterPlane(got, src, tc.srcStride, tc.qw, tc.qh)
			buildQuarterPlanePureGo(want, src, tc.srcStride, tc.qw, tc.qh)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("quarter[%d]=%d want %d", i, got[i], want[i])
				}
			}
		})
	}
}

func BenchmarkBuildQuarterPlane1080p(b *testing.B) {
	const w, h = 1920, 1080
	const qw, qh = w / 4, h / 4
	src := make([]byte, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src[y*w+x] = byte((x*13 + y*31 + x*y*7 + 5) & 255)
		}
	}
	b.Run("dispatch", func(b *testing.B) {
		dst := make([]byte, qw*qh)
		b.ReportAllocs()
		b.SetBytes(int64(w * h))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buildQuarterPlane(dst, src, w, qw, qh)
		}
	})
	b.Run("purego", func(b *testing.B) {
		dst := make([]byte, qw*qh)
		b.ReportAllocs()
		b.SetBytes(int64(w * h))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			buildQuarterPlanePureGo(dst, src, w, qw, qh)
		}
	})
}

func TestHMEQuarterMeshSearchMatchesLibaomReference(t *testing.T) {
	const qw, qh = 37, 29
	cols := (qw*4 + 31) / 32
	rows := (qh*4 + 31) / 32
	srcQ := make([]byte, qw*qh)
	refQ := make([]byte, qw*qh)
	rng := rand.New(rand.NewSource(41))
	for i := range srcQ {
		srcQ[i] = uint8(rng.Intn(256))
		refQ[i] = uint8(rng.Intn(256))
	}

	// Region (1,0) has a clean raw-SAD match at +8 quarter pels, while
	// zero-motion is already close. The libaom L1 MV cost should keep this
	// low-texture false motion at zero without a separate static shortcut.
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			v := uint8(40 + (x*9+y*11)%130)
			srcQ[y*qw+8+x] = v
			refQ[y*qw+8+x] = v + 1
			refQ[y*qw+16+x] = v
		}
	}
	// Region (0,1) is a true large motion match. Its zero-motion SAD is far
	// above the MV cost, so the same costed mesh still picks +8.
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			v := uint8(20 + (x*7+y*13)%40)
			srcQ[(8+y)*qw+x] = v
			refQ[(8+y)*qw+x] = v + 180
			refQ[(8+y)*qw+8+x] = v
		}
	}

	h := hmeState{
		srcQ:     srcQ,
		refQ:     refQ,
		qw:       qw,
		qh:       qh,
		cols:     cols,
		rows:     rows,
		seeds:    make([]motion.Vector, cols*rows),
		seedSADs: make([]int32, cols*rows),
	}
	h.searchRows(2, 0, rows)

	expectedStatic, expectedCut := 0, 0
	for ry := 0; ry < rows; ry++ {
		for rx := 0; rx < cols; rx++ {
			idx := ry*cols + rx
			qx := min(rx*8, qw-8)
			qy := min(ry*8, qh-8)
			dx, dy, sad := hmeQuarterMeshSearchReference(srcQ[qy*qw+qx:], refQ, qw, qw, qh, qx, qy)
			if got := h.seeds[idx]; got.Col != int16(dx*4) || got.Row != int16(dy*4) {
				t.Fatalf("seed[%d,%d] = (%d,%d), want (%d,%d)", rx, ry, got.Col, got.Row, dx*4, dy*4)
			}
			if got := int(h.seedSADs[idx]); got != sad {
				t.Fatalf("seedSAD[%d,%d] = %d, want %d", rx, ry, got, sad)
			}
			zero := hmeSAD8x8Reference(srcQ[qy*qw+qx:], qw, refQ[qy*qw+qx:], qw)
			if zero <= 8*8*2 {
				expectedStatic++
			}
			if sad > hmeCutRegionSAD {
				expectedCut++
			}
		}
	}
	if h.seeds[1].Col != 0 || h.seedSADs[1] != 64 {
		t.Fatalf("low-SAD false motion was not MV-costed: seed=%+v sad=%d", h.seeds[1], h.seedSADs[1])
	}
	if idx := cols; h.seeds[idx].Col != 32 || h.seedSADs[idx] != 0 {
		t.Fatalf("true large motion was not selected: seed=%+v sad=%d", h.seeds[idx], h.seedSADs[idx])
	}
	if got := h.bandStatic[2]; got != expectedStatic {
		t.Fatalf("bandStatic = %d, want %d", got, expectedStatic)
	}
	if got := h.bandCut[2]; got != expectedCut {
		t.Fatalf("bandCut = %d, want %d", got, expectedCut)
	}
}

func hmeQuarterMeshSearchReference(srcBlock, ref []byte, stride, width, height, qx, qy int) (int, int, int) {
	bestDX, bestDY, _ := hmeQuarterExhaustiveMeshSearchReference(srcBlock, ref, stride, width, height, qx, qy, 0, 0, 8, 2)
	return hmeQuarterExhaustiveMeshSearchReference(srcBlock, ref, stride, width, height, qx, qy, bestDX, bestDY, 1, 1)
}

func hmeQuarterExhaustiveMeshSearchReference(srcBlock, ref []byte, stride, width, height, qx, qy int, startDX, startDY, searchRange, step int) (int, int, int) {
	if step < 1 {
		step = 1
	}
	startDX = min(max(startDX, -qx), width-8-qx)
	startDY = min(max(startDY, -qy), height-8-qy)
	minDX, maxDX := -qx, width-8-qx
	minDY, maxDY := -qy, height-8-qy
	startCol := max(-searchRange, minDX-startDX)
	endCol := min(searchRange, maxDX-startDX)
	startRow := max(-searchRange, minDY-startDY)
	endRow := min(searchRange, maxDY-startDY)

	bestDX, bestDY := startDX, startDY
	bestRawSAD := hmeSAD8x8Reference(srcBlock, stride, ref[(qy+startDY)*stride+qx+startDX:], stride)
	bestCost := bestRawSAD + hmeQuarterMVCostReference(startDX, startDY, width, height)
	colStep := step
	if step <= 1 {
		colStep = 4
	}
	for row := startRow; row <= endRow; row += step {
		for col := startCol; col <= endCol; col += colStep {
			if step > 1 {
				dx, dy := startDX+col, startDY+row
				s := hmeSAD8x8Reference(srcBlock, stride, ref[(qy+dy)*stride+qx+dx:], stride)
				hmeQuarterUpdateBestReference(s, dx, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
				continue
			}
			if col+3 <= endCol {
				for i := 0; i < 4; i++ {
					dx, dy := startDX+col+i, startDY+row
					s := hmeSAD8x8Reference(srcBlock, stride, ref[(qy+dy)*stride+qx+dx:], stride)
					hmeQuarterUpdateBestReference(s, dx, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
				}
				continue
			}
			for i := 0; i < endCol-col; i++ {
				dx, dy := startDX+col+i, startDY+row
				s := hmeSAD8x8Reference(srcBlock, stride, ref[(qy+dy)*stride+qx+dx:], stride)
				hmeQuarterUpdateBestReference(s, dx, dy, width, height, &bestCost, &bestRawSAD, &bestDX, &bestDY)
			}
		}
	}
	return bestDX, bestDY, bestRawSAD
}

func hmeQuarterUpdateBestReference(rawSAD, dx, dy, width, height int, bestCost, bestRawSAD, bestDX, bestDY *int) {
	if rawSAD >= *bestCost {
		return
	}
	cost := rawSAD + hmeQuarterMVCostReference(dx, dy, width, height)
	if cost < *bestCost {
		*bestCost, *bestRawSAD, *bestDX, *bestDY = cost, rawSAD, dx, dy
	}
}

func hmeQuarterMVCostReference(dx, dy, width, height int) int {
	minFrame := min(width, height) * 4
	lambda := 32
	if minFrame >= 720 {
		lambda = 8
	} else if minFrame >= 480 {
		lambda = 15
	}
	dist := dx
	if dist < 0 {
		dist = -dist
	}
	ady := dy
	if ady < 0 {
		ady = -ady
	}
	return lambda * (dist + ady)
}

func hmeSAD8x8Reference(src []byte, srcStride int, ref []byte, refStride int) int {
	s := 0
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			d := int(src[y*srcStride+x]) - int(ref[y*refStride+x])
			if d < 0 {
				d = -d
			}
			s += d
		}
	}
	return s
}
