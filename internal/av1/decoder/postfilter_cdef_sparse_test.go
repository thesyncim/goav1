package decoder

import (
	"fmt"
	"slices"
	"testing"
	"unsafe"

	"github.com/thesyncim/goav1/internal/av1/cdef"
	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

func TestCDEFSparseStageRectFrozen(t *testing.T) {
	cases := 0
	for _, shape := range [][2]int{{8, 8}, {4, 4}, {4, 8}} {
		bw, bh := shape[0], shape[1]
		for _, dims := range [][2]int{{3 * 8 * bw, 3 * 8 * bh}, {17 * bw, 17 * bh}, {16*bw + 1, 16*bh + 1}, {8 * bw, 8 * bh}} {
			w, h := dims[0], dims[1]
			stride := w + 13
			pix := make([]byte, stride*h)
			for i := range pix {
				pix[i] = byte((i*79 ^ (i >> 2)) & 255)
			}
			plane := frame.Plane{Pix: pix, Width: w, Height: h, Stride: stride}
			for uy := 0; uy < h; uy += 8 * bh {
				for ux := 0; ux < w; ux += 8 * bw {
					uw, uh := min(8*bw, w-ux), min(8*bh, h-uy)
					cols, rows := uw/bw, uh/bh
					for _, span := range [][4]int{{0, 0, 1, 1}, {cols - 1, rows - 1, cols, rows}, {0, 0, cols, 1}, {0, rows - 1, cols, rows}, {cols / 2, rows / 2, cols, rows}, {0, 0, cols, rows}} {
						blocks := []cdef.BlockPosition{{BX: uint8(span[0]), BY: uint8(span[1])}, {BX: uint8(span[2] - 1), BY: uint8(span[3] - 1)}}
						rect := frameWorkCDEFSparseStageRect(blocks, uw, uh, bw, bh)
						if !rect.active {
							continue
						}
						for mode := range 4 {
							top, bottom, left := make([]byte, 2*w), make([]byte, 2*w), make([]byte, cdef.HorizontalBorder*uh)
							for i := range top {
								top[i] = byte(i*13 + 31)
							}
							for i := range bottom {
								bottom[i] = byte(i*7 + 199)
							}
							for i := range left {
								left[i] = byte(i*29 + 53)
							}
							if mode&1 == 0 {
								bottom = nil
							}
							leftBackup := mode&2 != 0
							want := make([]uint16, cdef.InputBufferSize)
							storage := make([]uint16, cdef.InputBufferSize+2)
							for i := range storage {
								storage[i] = 0xaffe
							}
							got := storage[1 : len(storage)-1]
							if err := frameWorkAssembleCDEFInputU8(want, plane, top, bottom, left, leftBackup, uy/(8*bh), ux, uy, uw, uh); err != nil {
								t.Fatal(err)
							}
							if !frameWorkAssembleCDEFSparseInputU8(got, plane, top, bottom, left, leftBackup, uy/(8*bh), ux, uy, uw, uh, rect) {
								t.Fatal("valid sparse wide stage declined")
							}
							assertCDEFSparseRegion(t, got, want, rect, 8)
							if storage[0] != 0xaffe || storage[len(storage)-1] != 0xaffe {
								t.Fatal("wide canary")
							}
							if ux >= 2 && uy >= 2 && ux+uw+2 <= w && uy+uh+2 <= h {
								wb, gb := make([]byte, cdef.InputBufferSize), make([]byte, cdef.InputBufferSize)
								for i := range gb {
									gb[i] = 0xe7
								}
								if err := frameWorkAssembleCDEFSecondaryByteInputU8(wb, plane, top, bottom, left, leftBackup, ux, uy, uw, uh); err != nil {
									t.Fatal(err)
								}
								if !frameWorkAssembleCDEFSparseByteInputU8(gb, plane, top, bottom, left, leftBackup, ux, uy, uw, uh, rect) {
									t.Fatal("valid sparse byte stage declined")
								}
								for i := range gb {
									x, y := i%cdef.BStride-cdef.HorizontalBorder, i/cdef.BStride-cdef.VerticalBorder
									inside := x >= int(rect.x0)-2 && x < int(rect.x1)+2 && y >= int(rect.y0)-2 && y < int(rect.y1)+2
									if inside && gb[i] != wb[i] || !inside && gb[i] != 0xe7 {
										t.Fatalf("byte bounds/copy mismatch at %d", i)
									}
								}
							}
							cases++
						}
					}
				}
			}
		}
	}
	t.Logf("full-stage oracle and untouched-region matches: %d", cases)
}

func assertCDEFSparseRegion(t *testing.T, got, want []uint16, rect frameWorkCDEFStageRect, halo int) {
	t.Helper()
	for i := range got {
		x, y := i%cdef.BStride-cdef.HorizontalBorder, i/cdef.BStride-cdef.VerticalBorder
		inside := x >= int(rect.x0)-halo && x < int(rect.x1)+halo && y >= int(rect.y0)-2 && y < int(rect.y1)+2
		if inside && got[i] != want[i] || !inside && got[i] != 0xaffe {
			t.Fatalf("staging/canary at %d x=%d y=%d: got=%x want=%x inside=%t rect=%+v", i, x, y, got[i], want[i], inside, rect)
		}
	}
}

func TestCDEFSparseFullPlaneSnapshot(t *testing.T) {
	patterns := []uint64{1, 1 << 63, 0x00003c3c3c3c0000, 0x00000000ffffffff, 0x8040201008040201, 0x0101010101010101, 0x8080808080808080, 0xffffffffffffffff, 0}
	for _, shape := range [][2]int{{1, 1}, {1, 0}, {0, 0}} {
		for _, strength := range [][2]uint8{{2, 2}, {9 << 2, 5 << 2}, {9<<2 | 2, 5<<2 | 1}, {0, 5<<2 | 2}} {
			t.Run(fmt.Sprintf("%dx%d-%d-%d", shape[0], shape[1], strength[0], strength[1]), func(t *testing.T) {
				params := parser.CDEFParams{Damping: 4, StrengthCount: 1}
				params.YStrength[0], params.UVStrength[0] = strength[0], strength[1]
				runCDEFU8WalkDifferential(t, cdefU8WalkCase{lumaW: 192, lumaH: 192, xDecC: shape[0], yDecC: shape[1], params: params, readProb: 100, withSkip: true, activeMasks: patterns, poisonInput: true, lowRange: true})
			})
		}
	}
}

// Compare the entire backing arrays and error/panic outcome with the frozen
// original walk, including overlapping uint16 line/input storage and an
// intentionally overlapping byte plane. The candidate must take its old path.
func TestCDEFSparseAliasesAndErrorsFrozen(t *testing.T) {
	for _, mode := range []string{"disjoint", "input-lines", "input-plane", "input-top", "input-bottom", "band-first", "band-middle", "band-last", "short-input", "short-lines", "bad-index", "bad-plane"} {
		t.Run(mode, func(t *testing.T) {
			run := func(frozen bool) (backing []uint16, pix []byte, outcome string) {
				defer func() {
					if p := recover(); p != nil {
						outcome = fmt.Sprint(p)
					}
				}()
				const width, height, stride = 192, 192, 208
				backing = make([]uint16, stride*height+3*cdef.InputBufferSize)
				for i := range backing {
					backing[i] = uint16(i*31 + 7)
				}
				pix = make([]byte, stride*height)
				for i := range pix {
					pix[i] = byte((i * 29) ^ (i >> 3))
				}
				input := backing[:cdef.InputBufferSize]
				lines := backing[cdef.InputBufferSize : 2*cdef.InputBufferSize]
				if mode == "input-lines" {
					lines = input
				}
				if mode == "input-plane" {
					pix = unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(backing))), len(pix))
				}
				if mode == "short-input" {
					input = input[:3]
				}
				if mode == "short-lines" {
					lines = lines[:3]
				}
				plane := frame.Plane{Pix: pix, Width: width, Height: height, Stride: stride}
				if mode == "bad-plane" {
					plane.Stride = 2
				}
				indices := threading.FrameWorkCDEFIndexMap{Stride: 3, Rows: 3, Index: make([]uint8, 9), Read: []bool{true, true, true, true, true, true, true, true, true}}
				if mode == "bad-index" {
					indices.Index[2] = 9
				}
				params := parser.CDEFParams{Damping: 4, StrengthCount: 1}
				params.YStrength[0] = 2
				recs := make([]threading.FrameWorkLoopFilterBlockRecord, (width/4)*(height/4))
				for i := range recs {
					recs[i].Valid = true
					recs[i].SkipTransform = true
				}
				for uy := range 3 {
					for ux := range 3 {
						x, y := 4, 4
						if mode == "band-first" || mode == "band-middle" || mode == "band-last" {
							x, y = 14, 14
							if ux == 0 {
								y = 0
							}
							if ux == 1 {
								x = 0
							}
						}
						recs[(uy*16+y)*(width/4)+ux*16+x].SkipTransform = false
					}
				}
				skip := &FrameWorkLoopFilterMap{Records: recs, Stride: width / 4, Rows: height / 4}
				var blocks [64]cdef.BlockPosition
				var dirs cdef.DirectionGrid
				var vars cdef.VarianceGrid
				apply := frameWorkApplyCDEFPlaneRowsU8
				if frozen {
					apply = frameWorkApplyCDEFPlaneRowsU8FrozenSparse
				}
				rowStart, rowEnd := 0, 3
				var boundary FrameWorkCDEFPostFilterU8BandBoundary
				if mode == "band-first" || mode == "band-middle" || mode == "band-last" {
					switch mode {
					case "band-first":
						rowStart, rowEnd = 0, 1
					case "band-middle":
						rowStart, rowEnd = 1, 2
					case "band-last":
						rowStart, rowEnd = 2, 3
					}
					if rowStart > 0 {
						boundary.Top[0] = make([]byte, 2*width)
						for r := range 2 {
							copy(boundary.Top[0][r*width:][:width], pix[((rowStart*64)-2+r)*stride:][:width])
						}
					}
					if rowEnd < 3 {
						boundary.Bottom[0] = make([]byte, 2*width)
						for r := range 2 {
							copy(boundary.Bottom[0][r*width:][:width], pix[(rowEnd*64+r)*stride:][:width])
						}
					}
				}
				if mode == "input-top" || mode == "input-bottom" {
					rowStart, rowEnd = 1, 2
					boundary.Top[0] = make([]byte, 2*width)
					boundary.Bottom[0] = make([]byte, 2*width)
					view := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(input))), 2*width)
					if mode == "input-top" {
						boundary.Top[0] = view
					} else {
						boundary.Bottom[0] = view
					}
				}
				u, b, err := apply(params, indices, skip, 3, 3, rowStart, rowEnd, plane, lines, boundary, input, blocks[:], &dirs, &vars, nil, nil, 0, 0, 0, false, nil)
				outcome = fmt.Sprintf("%d/%d/%v/%v/%v", u, b, err, dirs, vars)
				return
			}
			wb, wp, we := run(true)
			gb, gp, ge := run(false)
			if we != ge || !slices.Equal(wp, gp) {
				t.Fatalf("outcome/pixels differ: %s vs %s", we, ge)
			}
			// Independent input is temporary staging: unused samples are left
			// untouched, including when a later unit reports an invalid index.
			// Line backups/canaries still match; alias routes match everything.
			if mode != "input-lines" && mode != "input-plane" && mode != "input-top" && mode != "input-bottom" {
				wb, gb = wb[cdef.InputBufferSize:], gb[cdef.InputBufferSize:]
			}
			if !slices.Equal(wb, gb) {
				t.Fatal("alias/line/canary backing mutation differs")
			}
		})
	}
}

func TestCDEFSparseDeclineBeforeMutation(t *testing.T) {
	const width, height, stride = 192, 192, 208
	plane := frame.Plane{Pix: make([]byte, stride*height), Width: width, Height: height, Stride: stride}
	for _, mode := range []string{"short-input", "short-top", "short-bottom", "short-left", "inactive", "empty-rect", "wide-rect", "bad-plane"} {
		t.Run(mode, func(t *testing.T) {
			input := make([]uint16, cdef.InputBufferSize)
			bytes := make([]byte, 2*cdef.InputBufferSize)
			for i := range input {
				input[i] = 0xaced
			}
			for i := range bytes {
				bytes[i] = 0xda
			}
			top, bottom, left := make([]byte, 2*width), make([]byte, 2*width), make([]byte, 8*64)
			r := frameWorkCDEFStageRect{8, 8, 24, 24, true}
			p := plane
			switch mode {
			case "short-input":
				input, bytes = input[:3], bytes[:3]
			case "short-top":
				top = top[:1]
			case "short-bottom":
				bottom = bottom[:1]
			case "short-left":
				left = left[:1]
			case "inactive":
				r.active = false
			case "empty-rect":
				r.x0 = r.x1
			case "wide-rect":
				r.x1 = 65
			case "bad-plane":
				p.Stride = 1
			}
			if frameWorkAssembleCDEFSparseInputU8(input, p, top, bottom, left, true, 1, 64, 64, 64, 64, r) || frameWorkAssembleCDEFSparseByteInputU8(bytes, p, top, bottom, left, true, 64, 64, 64, 64, r) {
				t.Fatal("invalid stage accepted")
			}
			for _, x := range input {
				if x != 0xaced {
					t.Fatal("wide mutation before fallback")
				}
			}
			for _, x := range bytes {
				if x != 0xda {
					t.Fatal("byte mutation before fallback")
				}
			}
		})
	}
}

func TestCDEFSparseHelpersNoAllocs(t *testing.T) {
	const width, height, stride = 192, 192, 208
	plane := frame.Plane{Pix: make([]byte, stride*height), Width: width, Height: height, Stride: stride}
	input := make([]uint16, cdef.InputBufferSize)
	bytes := make([]byte, 2*cdef.InputBufferSize)
	top, bottom, left := make([]byte, 2*width), make([]byte, 2*width), make([]byte, 8*64)
	blocks := []cdef.BlockPosition{{BX: 2, BY: 3}, {BX: 4, BY: 5}}
	if n := testing.AllocsPerRun(100, func() {
		r := frameWorkCDEFSparseStageRect(blocks, 64, 64, 8, 8)
		if !frameWorkCDEFSparseScratchIndependent(input, plane.Pix, top, top, bottom) || !frameWorkAssembleCDEFSparseInputU8(input, plane, top, bottom, left, true, 1, 64, 64, 64, 64, r) || !frameWorkAssembleCDEFSparseByteInputU8(bytes, plane, top, bottom, left, true, 64, 64, 64, 64, r) {
			panic("valid stage declined")
		}
	}); n != 0 {
		t.Fatalf("allocations=%v", n)
	}
}

func TestCDEFSparseFullPlaneNoAllocs(t *testing.T) {
	for _, strength := range []uint8{2, 38} {
		params := parser.CDEFParams{Damping: 4, StrengthCount: 1}
		params.YStrength[0] = strength
		masks := []uint64{1, 1 << 63, 0x00003c3c3c3c0000, 0x00000000ffffffff, 0x00003c3c3c3c0000, 0x0101010101010101, 0x8080808080808080, ^uint64(0), 0}
		tc := cdefU8WalkCase{lumaW: 192, lumaH: 192, params: params, activeMasks: masks}
		if n := cdefPlaneWalkU8Allocs(t, tc); n != 0 {
			t.Fatalf("strength=%d allocations=%v", strength, n)
		}
	}
}
