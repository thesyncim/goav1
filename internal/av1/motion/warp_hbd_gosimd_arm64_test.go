// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

//go:build goexperiment.simd && arm64 && !purego

package motion

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
)

type warpHBDResidentKernel func(tmp *warpTmp, ref frame.Plane, ix4 int, sx4 int, iy4 int, sy4 int, alpha int, beta int, reduceBitsHoriz int, offsetBitsHoriz int) int

var warpHBDResidentSIMDKernels = []struct {
	name string
	run  warpHBDResidentKernel
}{
	{name: "go-simd", run: warpHorizontalHighBDResidentGoSIMD},
}

func residentRefPlaneHighBD(rng *rand.Rand, bitDepth uint8, oddBase bool) frame.Plane {
	const width, height, stride = 96, 96, 208
	pix := make([]byte, stride*height+1)
	if oddBase {
		pix = pix[1:]
	}
	ref := frame.Plane{Pix: pix, Stride: stride, Width: width, Height: height}
	mask := uint16((1 << bitDepth) - 1)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			storeHighBDSample(ref, x, y, uint16(rng.Intn(int(mask)+1)))
		}
	}
	return ref
}

func TestWarpHorizontalHighBDGoSIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5A1D0FF))
	phases := []int{-1 << 18, -257, -1, 0, 511, 32768, 65535, 1 << 18}
	steps := []int{-256, -96, -1, 0, 1, 96, 256}
	tested := 0
	for _, bitDepth := range []uint8{10, 12} {
		reduceBitsHoriz, _ := highBDRoundBits(bitDepth)
		offsetBitsHoriz := int(bitDepth) + filterBits - 1
		for _, oddBase := range []bool{false, true} {
			ref := residentRefPlaneHighBD(rng, bitDepth, oddBase)
			for pi, sx4 := range phases {
				for si, alpha := range steps {
					beta := steps[(pi+si)%len(steps)]
					ix4, iy4 := 24+(pi%7), 24+(si%7)
					var want warpTmp
					wantSY := warpHorizontalHighBDResident(&want, ref, ix4, sx4, iy4, 12345, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
					for _, kernel := range warpHBDResidentSIMDKernels {
						var got warpTmp
						gotSY := kernel.run(&got, ref, ix4, sx4, iy4, 12345, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
						if gotSY != wantSY || got != want {
							for i := range want {
								if got[i] != want[i] {
									t.Fatalf("kernel=%s bd=%d oddBase=%v ix4=%d iy4=%d sx4=%d alpha=%d beta=%d tmp[%d]=%d want %d",
										kernel.name, bitDepth, oddBase, ix4, iy4, sx4, alpha, beta, i, got[i], want[i])
								}
							}
							t.Fatalf("kernel=%s bd=%d oddBase=%v got sy %d want %d", kernel.name, bitDepth, oddBase, gotSY, wantSY)
						}
					}
					tested++
				}
			}
		}
	}
	if tested != 224 {
		t.Fatalf("tested %d cases, want 224", tested)
	}
}

func TestWarpHorizontalHighBDGoSIMDExtremesAndResidentBoundaries(t *testing.T) {
	for _, bitDepth := range []uint8{10, 12} {
		maxSample := uint16((1 << bitDepth) - 1)
		reduceBitsHoriz, _ := highBDRoundBits(bitDepth)
		offsetBitsHoriz := int(bitDepth) + filterBits - 1
		for _, stride := range []int{48, 49} {
			for pattern := 0; pattern < 3; pattern++ {
				ref, _ := testPlane(24, 24, 2, stride)
				for y := 0; y < ref.Height; y++ {
					for x := 0; x < ref.Width; x++ {
						var sample uint16
						switch pattern {
						case 0:
							sample = 0
						case 1:
							sample = maxSample
						case 2:
							if (x+y)&1 != 0 {
								sample = maxSample
							}
						}
						storeHighBDSample(ref, x, y, sample)
					}
				}
				for _, phase := range []int{-1 << 18, -1, 0, 32768, 65535, 1 << 18} {
					for _, alpha := range []int{-256, -1, 0, 1, 256} {
						for _, beta := range []int{-256, 0, 256} {
							var want warpTmp
							wantSY := warpHorizontalHighBDResident(&want, ref, 16, phase, 16, 321, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
							for _, kernel := range warpHBDResidentSIMDKernels {
								var got warpTmp
								gotSY := kernel.run(&got, ref, 16, phase, 16, 321, alpha, beta, reduceBitsHoriz, offsetBitsHoriz)
								if gotSY != wantSY || got != want {
									t.Fatalf("kernel=%s bd=%d stride=%d pattern=%d phase=%d alpha=%d beta=%d got=%v want=%v", kernel.name, bitDepth, stride, pattern, phase, alpha, beta, got, want)
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestWarpHorizontalHighBDGoSIMDZeroAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA110C))
	ref := residentRefPlaneHighBD(rng, 10, false)
	var tmp warpTmp
	const ix4, sx4, iy4, sy4, alpha, beta = 40, 32768, 40, 0, 96, -64
	for _, kernel := range warpHBDResidentSIMDKernels {
		allocs := testing.AllocsPerRun(1000, func() {
			kernel.run(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, 3, 10+filterBits-1)
		})
		if allocs != 0 {
			t.Errorf("resident Go SIMD horizontal %s allocated: %f", kernel.name, allocs)
		}
	}
}

func TestWarpHorizontalHighBDGoSIMDFullUint16AndExactFootprint(t *testing.T) {
	for filterIndex, filter := range warpedFilter {
		var sum, absSum int
		for _, coeff := range filter {
			sum += int(coeff)
			if coeff < 0 {
				absSum -= int(coeff)
			} else {
				absSum += int(coeff)
			}
		}
		if sum != 128 {
			t.Fatalf("warpedFilter[%d] sums to %d, want 128 for centered uint16 SIMD arithmetic", filterIndex, sum)
		}
		if absSum > 222 {
			t.Fatalf("warpedFilter[%d] L1=%d exceeds centered SIMD accumulator proof", filterIndex, absSum)
		}
	}

	const width, height, stride = 15, 15, 30
	ref := frame.Plane{Pix: make([]byte, stride*height), Stride: stride, Width: width, Height: height}
	values := [...]uint16{0, 1, 32767, 32768, 65535}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			storeHighBDSample(ref, x, y, values[(3*x+2*y)%len(values)])
		}
	}
	for _, bitDepth := range []uint8{10, 12} {
		reduceBitsHoriz, _ := highBDRoundBits(bitDepth)
		offsetBitsHoriz := int(bitDepth) + filterBits - 1
		for _, sx4 := range []int{-1 << 18, -1, 0, 32768, 65535, 1 << 18} {
			var want warpTmp
			wantSY := warpHorizontalHighBDResident(&want, ref, 7, sx4, 7, 1234, 97, -31, reduceBitsHoriz, offsetBitsHoriz)
			for _, kernel := range warpHBDResidentSIMDKernels {
				var got warpTmp
				gotSY := kernel.run(&got, ref, 7, sx4, 7, 1234, 97, -31, reduceBitsHoriz, offsetBitsHoriz)
				if gotSY != wantSY || got != want {
					t.Fatalf("kernel=%s bd=%d sx4=%d exact 15x15 footprint got=%v want=%v", kernel.name, bitDepth, sx4, got, want)
				}
			}
		}
	}
}

func warpHBDTmpChecksum(tmp *warpTmp) int64 {
	var sum int64
	for _, v := range tmp {
		sum += int64(v)
	}
	return sum
}

var warpHBDTmpSink int64

func benchWarpHorizontalHighBDInputs() (frame.Plane, int, int, int, int, int, int) {
	rng := rand.New(rand.NewSource(1))
	ref := residentRefPlaneHighBD(rng, 10, false)
	const i, j = 16, 16
	const alpha, beta, gamma, delta = -320, -64, 0, -64
	ix4, sx4, iy4, sy4 := warpBlockOrigin(i, j, [6]int32{42805, -7571, 65230, -57, 0, 65509}, alpha, beta, gamma, delta, 0, 0)
	return ref, ix4, sx4, iy4, sy4, alpha, beta
}

func BenchmarkWarpHorizontalHighBDResidentScalar(b *testing.B) {
	ref, ix4, sx4, iy4, sy4, alpha, beta := benchWarpHorizontalHighBDInputs()
	var tmp warpTmp
	b.ReportAllocs()
	for b.Loop() {
		warpHorizontalHighBDResident(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, 3, 10+filterBits-1)
	}
	warpHBDTmpSink = warpHBDTmpChecksum(&tmp)
}

func BenchmarkWarpHorizontalHighBDResidentGoSIMD(b *testing.B) {
	ref, ix4, sx4, iy4, sy4, alpha, beta := benchWarpHorizontalHighBDInputs()
	var tmp warpTmp
	b.ReportAllocs()
	for b.Loop() {
		warpHorizontalHighBDResidentGoSIMD(&tmp, ref, ix4, sx4, iy4, sy4, alpha, beta, 3, 10+filterBits-1)
	}
	warpHBDTmpSink = warpHBDTmpChecksum(&tmp)
}
