//go:build goexperiment.simd && (amd64 || arm64) && !purego

package transform

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"runtime"
	"simd/archsimd"
	"slices"
	"testing"
)

func fourLaneSIMDAvailable() bool {
	return runtime.GOARCH == "arm64" || archsimd.X86.AVX2()
}

func TestNativeDCTLanes4Parity(t *testing.T) {
	if !fourLaneSIMDAvailable() {
		t.Skip("native SIMD unavailable")
	}
	random := rand.New(rand.NewSource(0x32644))
	clamps := append(slices.Clone(stageClampSets),
		[2]int32{-itxNarrowBound, itxNarrowBound - 1},
		[2]int32{-itxNarrowBound - 1, itxNarrowBound},
		[2]int32{-itxWideBound, itxWideBound - 1},
		[2]int32{-3, 5},
		[2]int32{-itxWideBound - 1, itxWideBound},
		[2]int32{math.MinInt32, math.MaxInt32})
	for _, kernel := range []struct {
		name         string
		size         int
		column       func([]int32, int, int32, int32)
		row          func([]int32, []int32, []int32, []int32, int32, int32)
		scalarColumn func([]int32, int, int32, int32)
		scalarRow    func([]int32, []int32, []int32, []int32, int32, int32)
	}{
		{"DCT32", 32, inverseDCT32Lanes4ColSIMD, inverseDCT32Lanes4RowSIMD, inverseDCT32Col4PureGo, inverseDCT32Row4PureGo},
		{"DCT64", 64, inverseDCT64Col4SIMD, inverseDCT64Row4SIMD, inverseDCT64Col4PureGo, inverseDCT64Row4PureGo},
	} {
		t.Run(kernel.name, func(t *testing.T) {
			for _, clamp := range clamps {
				for _, stride := range []int{4, 5, 11, 64} {
					for pattern := 0; pattern < 4; pattern++ {
						length := (kernel.size-1)*stride + 4
						actual := make([]int32, length+6)
						for index := range actual {
							actual[index] = 0x12345678
						}
						for index := 0; index < kernel.size; index++ {
							for lane := 0; lane < 4; lane++ {
								value := clamp[0] + int32(random.Int63n(int64(clamp[1])-int64(clamp[0])+1))
								if pattern == 1 {
									value = clamp[(index+lane)%2]
								} else if pattern == 2 {
									value = 0
									if index == lane {
										value = clamp[1]
									}
								} else if pattern == 3 {
									value = int32((index+lane)%3 - 1)
								}
								actual[3+index*stride+lane] = value
							}
						}
						expected := slices.Clone(actual)
						kernel.scalarColumn(expected[3:3+length], stride, clamp[0], clamp[1])
						kernel.column(actual[3:3+length], stride, clamp[0], clamp[1])
						if !slices.Equal(actual, expected) {
							t.Fatalf("column clamp=%v stride=%d pattern=%d mismatch", clamp, stride, pattern)
						}
						rows := make([]int32, 4*(kernel.size+2))
						for index := range rows {
							rows[index] = clamp[0] + int32(random.Int63n(int64(clamp[1])-int64(clamp[0])+1))
						}
						expectedRows := slices.Clone(rows)
						rowStride := kernel.size + 2
						kernel.scalarRow(expectedRows[1:1+kernel.size], expectedRows[rowStride+1:rowStride+1+kernel.size], expectedRows[2*rowStride+1:2*rowStride+1+kernel.size], expectedRows[3*rowStride+1:3*rowStride+1+kernel.size], clamp[0], clamp[1])
						kernel.row(rows[1:1+kernel.size], rows[rowStride+1:rowStride+1+kernel.size], rows[2*rowStride+1:2*rowStride+1+kernel.size], rows[3*rowStride+1:3*rowStride+1+kernel.size], clamp[0], clamp[1])
						if !slices.Equal(rows, expectedRows) {
							t.Fatalf("row clamp=%v mismatch", clamp)
						}
					}
				}
			}
		})
	}
}

func TestNativeDCTLanes4NoAlloc(t *testing.T) {
	if !fourLaneSIMDAvailable() {
		t.Skip("native SIMD unavailable")
	}
	for _, kernel := range []struct {
		name   string
		size   int
		column func([]int32, int, int32, int32)
		row    func([]int32, []int32, []int32, []int32, int32, int32)
	}{
		{"DCT32", 32, inverseDCT32Lanes4ColSIMD, inverseDCT32Lanes4RowSIMD},
		{"DCT64", 64, inverseDCT64Col4SIMD, inverseDCT64Row4SIMD},
	} {
		t.Run(kernel.name, func(t *testing.T) {
			buffer := make([]int32, kernel.size*4)
			if allocations := testing.AllocsPerRun(20, func() {
				kernel.column(buffer, 4, -itxNarrowBound, itxNarrowBound-1)
				kernel.row(buffer[:kernel.size], buffer[kernel.size:2*kernel.size], buffer[2*kernel.size:3*kernel.size], buffer[3*kernel.size:], -itxWideBound, itxWideBound-1)
			}); allocations != 0 {
				t.Fatalf("allocated %g objects per row/column pair", allocations)
			}
		})
	}
}

func TestAMD64DCTLanes4Dispatch(t *testing.T) {
	if runtime.GOARCH != "amd64" || !archsimd.X86.AVX2() {
		t.Skip("AMD64 AVX2 unavailable")
	}
	for _, binding := range []struct{ actual, expected any }{
		{inverseDCT32Col4Impl, inverseDCT32Lanes4ColSIMD},
		{inverseDCT32Row4Impl, inverseDCT32Lanes4RowSIMD},
		{inverseDCT64Col4Impl, inverseDCT64Col4SIMD},
		{inverseDCT64Row4Impl, inverseDCT64Row4SIMD},
	} {
		if reflect.ValueOf(binding.actual).Pointer() != reflect.ValueOf(binding.expected).Pointer() {
			t.Fatal("four-lane dispatch is not bound to Go SIMD")
		}
	}
}

func TestNativeDCTLanes4ShortRows(t *testing.T) {
	if !fourLaneSIMDAvailable() {
		t.Skip("native SIMD unavailable")
	}
	for _, kernel := range []struct {
		name   string
		size   int
		row    func([]int32, []int32, []int32, []int32, int32, int32)
		scalar func([]int32, []int32, []int32, []int32, int32, int32)
	}{
		{"DCT32", 32, inverseDCT32Lanes4RowSIMD, inverseDCT32Row4PureGo},
		{"DCT64", 64, inverseDCT64Row4SIMD, inverseDCT64Row4PureGo},
	} {
		t.Run(kernel.name, func(t *testing.T) {
			for shortLane := 0; shortLane < 4; shortLane++ {
				for _, length := range []int{0, 1, kernel.size - 1, kernel.size, kernel.size + 3} {
					var actual, expected [4][]int32
					for lane := range actual {
						rowLength := kernel.size + 3
						if lane == shortLane {
							rowLength = length
						}
						actual[lane] = make([]int32, rowLength)
						for index := range actual[lane] {
							actual[lane][index] = int32(index*17 + lane*31 - 100)
						}
						expected[lane] = slices.Clone(actual[lane])
					}
					kernel.scalar(expected[0], expected[1], expected[2], expected[3], -32768, 32767)
					kernel.row(actual[0], actual[1], actual[2], actual[3], -32768, 32767)
					for lane := range actual {
						if !slices.Equal(actual[lane], expected[lane]) {
							t.Fatalf("short lane=%d length=%d output lane=%d mismatch", shortLane, length, lane)
						}
					}
				}
			}
		})
	}
}

func BenchmarkNativeDCT32Lanes4(b *testing.B) {
	if !fourLaneSIMDAvailable() {
		b.Skip("native SIMD unavailable")
	}
	for _, bounds := range [][2]int32{{-itxNarrowBound, itxNarrowBound - 1}, {-itxWideBound, itxWideBound - 1}} {
		b.Run(fmt.Sprint(bounds), func(b *testing.B) {
			buffer := make([]int32, 32*4)
			b.ReportAllocs()
			for b.Loop() {
				for index := range buffer {
					buffer[index] = int32(index%251 - 125)
				}
				inverseDCT32Lanes4ColSIMD(buffer, 4, bounds[0], bounds[1])
			}
		})
	}
}
