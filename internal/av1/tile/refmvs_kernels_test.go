package tile

import (
	"math/rand/v2"
	"testing"
)

// splatScalarRef is dav1d's splat_mv_c over the tile row table.
func splatScalarRef(rt *refmvsTile, mv uint64, rest uint32, by4, bx4, bw4, bh4 int32) {
	base := (by4 & 31) + 5
	for y := int32(0); y < bh4; y++ {
		r := rt.row(base + y)[bx4 : bx4+bw4]
		for i := range r {
			r[i].set(mv, rest)
		}
	}
}

// saveTmvsScalarRef is dav1d's save_tmvs_c with goav1's entry layout.
func saveTmvsScalarRef(rt *refmvsTile, dst *ReferenceMVFrame, refFrameSide *[referenceFrameCount]int8, colStart8, colEnd8, rowStart8, rowEnd8 int32) {
	rowEnd8 = min(rowEnd8, int32(dst.Rows))
	colEnd8 = min(colEnd8, int32(dst.Cols))
	saveable := func(m refmvsMV) bool {
		r, c := m.row(), m.col()
		return max(r, -r)|max(c, -c) < 4096
	}
	stride := int32(dst.Stride)
	for y := rowStart8; y < rowEnd8; y++ {
		b := rt.row(6 + (y&15)*2)
		rp := dst.Entries[y*stride : y*stride+colEnd8]
		for x := colStart8; x < colEnd8; {
			cand := &b[x*2+1]
			end := min(x+(int32(blockDimensions[cand.bs].W4)+1)>>1, colEnd8)
			tmv := ReferenceMVEntry{Ref: ReferenceFrameNone}
			if r := cand.ref.ref1(); r > 0 && refFrameSide[r-1] == 0 && saveable(cand.mv[1]) {
				tmv = ReferenceMVEntry{Ref: ReferenceFrame(r - 1), MV: cand.mv[1].vector(), Valid: true}
			} else if r := cand.ref.ref0(); r > 0 && refFrameSide[r-1] == 0 && saveable(cand.mv[0]) {
				tmv = ReferenceMVEntry{Ref: ReferenceFrame(r - 1), MV: cand.mv[0].vector(), Valid: true}
			}
			for ; x < end; x++ {
				rp[x] = tmv
			}
		}
	}
}

func newKernelTestTile(t *testing.T, iw4, ih4 int32) *refmvsTile {
	t.Helper()
	rt := &refmvsTile{}
	rt.bindFrame(refmvsFrame{iw4: iw4, ih4: ih4}, iw4*4, 0, iw4, 0, ih4)
	return rt
}

var kernelTestMVComponents = []int16{0, 1, -1, 7, -8, 4095, -4095, 4096, -4096, 4097, 32767, -32768, -32767}

func kernelTestMV(rng *rand.Rand) refmvsMV {
	pick := func() int32 {
		if rng.IntN(2) == 0 {
			return int32(kernelTestMVComponents[rng.IntN(len(kernelTestMVComponents))])
		}
		return int32(int16(rng.Uint32()))
	}
	return makeRefmvsMV(pick(), pick())
}

func TestRefmvsSplatMatchesScalar(t *testing.T) {
	const iw4, ih4 = 120, 68
	rng := rand.New(rand.NewPCG(1, 2))
	got := newKernelTestTile(t, iw4, ih4)
	want := newKernelTestTile(t, iw4, ih4)
	for sby := int32(0); sby < 4; sby++ {
		got.sbrowInit(sby, 16)
		want.sbrowInit(sby, 16)
		for bs := BlockSize(0); bs < blockSizeCount; bs++ {
			d := blockDimensions[bs]
			bw4, bh4 := int32(d.W4), int32(d.H4)
			if bh4 > 16 {
				continue // 128-tall blocks need a 32-row superblock
			}
			for trial := 0; trial < 20; trial++ {
				bx4 := int32(rng.IntN(int(iw4/bw4))) * bw4
				by4 := sby*16 + int32(rng.IntN(int(16/bh4)))*bh4
				mv := uint64(kernelTestMV(rng)) | uint64(kernelTestMV(rng))<<32
				rest := refmvsSplatRest(int8(rng.IntN(9)-1), int8(rng.IntN(9)-1), bs, uint8(rng.IntN(4)))
				got.splat(mv, rest, by4, bx4, bw4, bh4)
				splatScalarRef(want, mv, rest, by4, bx4, bw4, bh4)
			}
		}
		for i := range got.buf {
			if got.buf[i] != want.buf[i] {
				t.Fatalf("sby %d: cell %d = %+v, want %+v", sby, i, got.buf[i], want.buf[i])
			}
		}
	}
	// 128x128 superblocks (sbsz 32).
	got = newKernelTestTile(t, iw4, ih4)
	want = newKernelTestTile(t, iw4, ih4)
	for sby := int32(0); sby < 2; sby++ {
		got.sbrowInit(sby, 32)
		want.sbrowInit(sby, 32)
		for bs := BlockSize(0); bs < blockSizeCount; bs++ {
			d := blockDimensions[bs]
			bw4, bh4 := int32(d.W4), int32(d.H4)
			for trial := 0; trial < 10; trial++ {
				bx4 := int32(rng.IntN(int(iw4/bw4))) * bw4
				by4 := sby*32 + int32(rng.IntN(int(32/bh4)))*bh4
				mv := uint64(kernelTestMV(rng)) | uint64(kernelTestMV(rng))<<32
				rest := refmvsSplatRest(int8(rng.IntN(9)-1), int8(rng.IntN(9)-1), bs, uint8(rng.IntN(4)))
				got.splat(mv, rest, by4, bx4, bw4, bh4)
				splatScalarRef(want, mv, rest, by4, bx4, bw4, bh4)
			}
		}
		for i := range got.buf {
			if got.buf[i] != want.buf[i] {
				t.Fatalf("sb128 sby %d: cell %d = %+v, want %+v", sby, i, got.buf[i], want.buf[i])
			}
		}
	}
}

func TestRefmvsSaveTmvsMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, sz := range [][2]int32{{120, 68}, {37, 21}, {480, 270}} {
		iw4, ih4 := sz[0], sz[1]
		rt := newKernelTestTile(t, iw4, ih4)
		cols, rows := uint16(iw4>>1), uint16(ih4>>1)
		stride := cols + uint16(rng.IntN(3))
		mk := func() *ReferenceMVFrame {
			f := &ReferenceMVFrame{Rows: rows, Cols: cols, Stride: stride, Entries: make([]ReferenceMVEntry, int(rows)*int(stride))}
			for i := range f.Entries {
				f.Entries[i] = ReferenceMVEntry{MV: kernelTestMV(rng).vector(), Ref: 3, Valid: true}
			}
			return f
		}
		for sby := int32(0); sby*16 < ih4; sby++ {
			rt.sbrowInit(sby, 16)
			// Fill the superblock row's cells with random blocks, tiled the
			// way the partition would leave them.
			for y4 := sby * 16; y4 < min(sby*16+16, ih4); y4++ {
				row := rt.row((y4 & 31) + 5)
				for x4 := int32(0); x4 < int32(len(row)); x4++ {
					bs := BlockSize(rng.IntN(int(blockSizeCount)))
					r0 := int8(rng.IntN(9) - 1)
					r1 := int8(rng.IntN(9) - 1)
					if rng.IntN(3) == 0 {
						r1 = -1
					}
					row[x4] = refmvsBlock{mv: [2]refmvsMV{kernelTestMV(rng), kernelTestMV(rng)}, ref: makeRefmvsRefPair(r0, r1), bs: bs}
				}
			}
			for trial := 0; trial < 8; trial++ {
				var side [referenceFrameCount]int8
				for i := range side {
					side[i] = int8(rng.IntN(2))
				}
				colStart8 := int32(rng.IntN(int(cols)))
				colEnd8 := colStart8 + int32(rng.IntN(int(cols)-int(colStart8)+8))
				if trial == 0 {
					colStart8, colEnd8 = 0, int32(cols)+16 // clamp at the right edge
				}
				got, want := mk(), mk()
				copy(want.Entries, got.Entries)
				rt.saveTmvs(got, &side, colStart8, colEnd8, sby*8, sby*8+8)
				saveTmvsScalarRef(rt, want, &side, colStart8, colEnd8, sby*8, sby*8+8)
				for i := range got.Entries {
					if got.Entries[i] != want.Entries[i] {
						t.Fatalf("%dx%d sby %d trial %d: entry %d = %+v, want %+v", iw4, ih4, sby, trial, i, got.Entries[i], want.Entries[i])
					}
				}
			}
		}
	}
}

func BenchmarkRefmvsSplat(b *testing.B) {
	rt := &refmvsTile{}
	rt.bindFrame(refmvsFrame{iw4: 480, ih4: 270}, 1920, 0, 480, 0, 270)
	rt.sbrowInit(0, 16)
	sizes := []BlockSize{BlockSize8x8, BlockSize16x16, BlockSize32x32, BlockSize64x64, BlockSize16x8, BlockSize4x4, BlockSize8x16, BlockSize64x32}
	var x4 int32
	for b.Loop() {
		for _, bs := range sizes {
			d := blockDimensions[bs]
			x4 = (x4 + 16) % 464
			rt.splat(0x00010002_00030004, refmvsSplatRest(1, -1, bs, 0), 0, x4&^15, int32(d.W4), int32(d.H4))
		}
	}
}

func BenchmarkRefmvsSaveTmvs(b *testing.B) {
	// A superblock row tiled with mixed block sizes, about a tenth intra,
	// single and compound references on both sides.
	rng := rand.New(rand.NewPCG(5, 6))
	rt := &refmvsTile{}
	rt.bindFrame(refmvsFrame{iw4: 480, ih4: 272}, 1920, 0, 480, 0, 272)
	rt.sbrowInit(0, 16)
	sizes := []BlockSize{BlockSize4x4, BlockSize8x8, BlockSize8x8, BlockSize16x16, BlockSize16x16, BlockSize32x32, BlockSize64x64, BlockSize16x8, BlockSize8x16}
	for sbx := int32(0); sbx < 480; sbx += 16 {
		var fill func(x4, y4, w4 int32)
		fill = func(x4, y4, w4 int32) {
			bs := sizes[rng.IntN(len(sizes))]
			d := blockDimensions[bs]
			if int32(d.W4) < w4 && rng.IntN(3) != 0 {
				h := w4 / 2
				for _, o := range [4][2]int32{{0, 0}, {h, 0}, {0, h}, {h, h}} {
					fill(x4+o[0], y4+o[1], h)
				}
				return
			}
			bs = map[int32]BlockSize{1: BlockSize4x4, 2: BlockSize8x8, 4: BlockSize16x16, 8: BlockSize32x32, 16: BlockSize64x64}[w4]
			blk := refmvsBlock{mv: [2]refmvsMV{kernelTestMV(rng) & 0x03ff03ff, kernelTestMV(rng) & 0x03ff03ff}, ref: makeRefmvsRefPair(int8(1+rng.IntN(7)), -1), bs: bs}
			switch rng.IntN(10) {
			case 0:
				blk = refmvsBlock{mv: [2]refmvsMV{refmvsInvalidMV, 0}, ref: makeRefmvsRefPair(0, -1), bs: bs}
			case 1, 2:
				blk.ref = makeRefmvsRefPair(int8(1+rng.IntN(4)), int8(5+rng.IntN(3)))
			}
			for y := y4; y < y4+w4; y++ {
				row := rt.row(y + 5)
				for x := x4; x < x4+w4; x++ {
					row[x] = blk
				}
			}
		}
		fill(sbx, 0, 16)
	}
	dst := &ReferenceMVFrame{Rows: 136, Cols: 240, Stride: 240, Entries: make([]ReferenceMVEntry, 136*240)}
	side := [referenceFrameCount]int8{0, 0, 0, 0, 1, 1, 1}
	for b.Loop() {
		rt.saveTmvs(dst, &side, 0, 240, 0, 8)
	}
}
