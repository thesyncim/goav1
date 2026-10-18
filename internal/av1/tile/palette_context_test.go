package tile

import (
	"fmt"
	"testing"
)

// The oracle is frozen from 10f4a6ae956e881e8d9e85a4797c55a441cd27c3.
// Keep its weighted scoring and stable insertion sort independent of the
// production equality cases, including invalid-relative-to-size neighbors.
func paletteColorIndexContextFrozen(colorMap []uint8, stride int, row int, col int, paletteSize int) (int, [PaletteMaxSize]uint8, error) {
	if paletteSize < PaletteMinSize || paletteSize > PaletteMaxSize || (row == 0 && col == 0) {
		return 0, [PaletteMaxSize]uint8{}, ErrInvalidDecodeState
	}
	neighbors := [3]int{-1, -1, -1}
	if col > 0 {
		neighbors[0] = int(colorMap[row*stride+col-1])
	}
	if col > 0 && row > 0 {
		neighbors[1] = int(colorMap[(row-1)*stride+col-1])
	}
	if row > 0 {
		neighbors[2] = int(colorMap[(row-1)*stride+col])
	}
	weights := [3]int{2, 1, 2}
	var scores [PaletteMaxSize]int
	for i, n := range neighbors {
		if n >= 0 {
			scores[n] += weights[i]
		}
	}
	var order [PaletteMaxSize]uint8
	var inverse [PaletteMaxSize]uint8
	for i := range PaletteMaxSize {
		order[i] = uint8(i)
		inverse[i] = uint8(i)
	}
	for i := range 3 {
		maxScore := scores[i]
		maxIdx := i
		for j := i + 1; j < paletteSize; j++ {
			if scores[j] > maxScore {
				maxScore = scores[j]
				maxIdx = j
			}
		}
		if maxIdx != i {
			maxOrder := order[maxIdx]
			for k := maxIdx; k > i; k-- {
				scores[k] = scores[k-1]
				order[k] = order[k-1]
				inverse[order[k]] = uint8(k)
			}
			scores[i] = maxScore
			order[i] = maxOrder
			inverse[order[i]] = uint8(i)
		}
	}
	hash := scores[0] + 2*scores[1] + 2*scores[2]
	lookup := [...]int{-1, -1, 0, -1, -1, 4, 3, 2, 1}
	if hash <= 0 || hash >= len(lookup) || lookup[hash] < 0 {
		return 0, [PaletteMaxSize]uint8{}, ErrInvalidDecodeState
	}
	return lookup[hash], order, nil
}

type paletteContextOutcome struct {
	context   int
	order     [PaletteMaxSize]uint8
	err       error
	panicText string
}

func observePaletteContext(fn func([]uint8, int, int, int, int) (int, [PaletteMaxSize]uint8, error), data []uint8, stride, row, col, size int) (out paletteContextOutcome) {
	defer func() {
		if p := recover(); p != nil {
			out.panicText = fmt.Sprint(p)
		}
	}()
	out.context, out.order, out.err = fn(data, stride, row, col, size)
	return
}

func TestPaletteColorIndexContextFrozenExhaustive(t *testing.T) {
	count := 0
	check := func(data []uint8, stride, row, col, size int) {
		t.Helper()
		got := observePaletteContext(paletteColorIndexContext, data, stride, row, col, size)
		want := observePaletteContext(paletteColorIndexContextFrozen, data, stride, row, col, size)
		if got != want {
			t.Fatalf("map=%v stride=%d row=%d col=%d size=%d: got %+v want %+v", data, stride, row, col, size, got, want)
		}
		count++
	}
	for size := PaletteMinSize; size <= PaletteMaxSize; size++ {
		for left := range PaletteMaxSize {
			for diagonal := range PaletteMaxSize {
				for top := range PaletteMaxSize {
					check([]uint8{uint8(diagonal), uint8(top), uint8(left), 0}, 2, 1, 1, size)
				}
			}
			check([]uint8{uint8(left), 0}, 2, 0, 1, size)
			check([]uint8{uint8(left), 0, 0}, 2, 1, 0, size)
		}
		check(nil, 0, 0, 0, size)
	}
	// Preserve old validation, missing-neighbor, slice-bound and score-bound
	// behavior even for inputs that decoded palette maps cannot produce.
	for _, size := range []int{-1, 0, 1, 2, 8, 9, 255} {
		for _, geometry := range [][3]int{{2, 0, 0}, {2, -1, -1}, {0, -1, 1}, {0, 1, -1}, {2, 1, 1}, {0, 1, 1}} {
			for _, data := range [][]uint8{nil, {0}, {0, 1, 2, 3}} {
				check(data, geometry[0], geometry[1], geometry[2], size)
			}
		}
	}
	for color := PaletteMaxSize; color <= 255; color++ {
		for neighbor := range 3 {
			data := []uint8{0, 0, 0, 0}
			data[neighbor] = uint8(color)
			check(data, 2, 1, 1, PaletteMaxSize)
		}
		check([]uint8{uint8(color), 0}, 2, 0, 1, 2)
		check([]uint8{uint8(color), 0, 0}, 2, 1, 0, 2)
	}
	t.Logf("frozen context/order/error/panic outcomes matched: %d", count)
}

var paletteContextSink [PaletteMaxSize]uint8

func TestPaletteColorIndexContextNoAllocs(t *testing.T) {
	data := []uint8{2, 7, 3, 0}
	if allocs := testing.AllocsPerRun(100, func() {
		_, paletteContextSink, _ = paletteColorIndexContext(data, 2, 1, 1, 8)
	}); allocs != 0 {
		t.Fatalf("allocs=%v want 0", allocs)
	}
}
