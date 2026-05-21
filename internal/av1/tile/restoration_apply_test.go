package tile

import (
	"errors"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	av1restoration "github.com/thesyncim/goav1/internal/av1/restoration"
)

func TestRestorationUnitScratchLenMatchesPrimitives(t *testing.T) {
	got, err := RestorationUnitScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	wantWiener, err := av1restoration.WienerScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	wantSGR, err := av1restoration.SelfguidedScratchLen(64, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got != (RestorationUnitScratchSize{Wiener: wantWiener, SGRProj: wantSGR}) {
		t.Fatalf("scratch=%+v want wiener=%d sgr=%d", got, wantWiener, wantSGR)
	}
}

func TestApplyRestorationUnitNoneCopies(t *testing.T) {
	const width, height = 5, 4
	const srcStride, dstStride = 11, 8
	const origin = 13
	src := make([]uint16, origin+(height-1)*srcStride+width)
	dst := make([]uint16, dstStride*height)
	for i := range dst {
		dst[i] = 0xffff
	}
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			src[origin+row*srcStride+col] = uint16(20 + row*7 + col)
		}
	}

	result, err := ApplyRestorationUnit(src, srcStride, origin, dst, dstStride, width, height, RestorationUnit{Type: parser.RestorationNone}, 8, RestorationUnitScratch{})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitApplyResult{Type: parser.RestorationNone}) {
		t.Fatalf("result=%+v", result)
	}
	for row := 0; row < height; row++ {
		for col := 0; col < width; col++ {
			if got, want := dst[row*dstStride+col], src[origin+row*srcStride+col]; got != want {
				t.Fatalf("dst[%d,%d]=%d want %d", row, col, got, want)
			}
		}
		for col := width; col < dstStride; col++ {
			if dst[row*dstStride+col] != 0xffff {
				t.Fatalf("padding row=%d col=%d overwritten", row, col)
			}
		}
	}
}

func TestApplyRestorationUnitWienerMatchesPrimitive(t *testing.T) {
	const width, height = 16, 12
	const bitDepth = 10
	stride := width + 2*av1restoration.WienerHalfwin
	origin := av1restoration.WienerHalfwin*stride + av1restoration.WienerHalfwin
	src := makeRestorationApplySource(stride, height+2*av1restoration.WienerHalfwin, bitDepth)
	unit := RestorationUnit{
		Type: parser.RestorationWiener,
		Wiener: av1restoration.WienerInfo{
			VFilter: av1restoration.NewWienerFilter(-3, -11, 22),
			HFilter: av1restoration.NewWienerFilter(6, -9, 18),
		},
	}
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]uint16, width*height)
	want := make([]uint16, width*height)
	result, err := ApplyRestorationUnit(src, stride, origin, got, width, width, height, unit, bitDepth, RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitApplyResult{Type: parser.RestorationWiener, Filtered: true}) {
		t.Fatalf("result=%+v", result)
	}
	if err := av1restoration.ApplyWienerRestoration(src, stride, origin, want, width, width, height, unit.Wiener, bitDepth, make([]uint16, sizes.Wiener)); err != nil {
		t.Fatal(err)
	}
	assertSamplesEqual(t, got, want)
}

func TestApplyRestorationUnitSGRProjMatchesPrimitive(t *testing.T) {
	const width, height = 13, 9
	const bitDepth = 8
	stride := width + 2*av1restoration.SGRProjBorderHorz
	origin := av1restoration.SGRProjBorderVert*stride + av1restoration.SGRProjBorderHorz
	src := makeRestorationApplySource(stride, height+2*av1restoration.SGRProjBorderVert, bitDepth)
	unit := RestorationUnit{
		Type:    parser.RestorationSGRProj,
		SGRProj: SGRProjInfo{ParamsIndex: 1, XQD: [2]int{13, -9}},
	}
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]uint16, width*height)
	want := make([]uint16, width*height)
	result, err := ApplyRestorationUnit(src, stride, origin, got, width, width, height, unit, bitDepth, RestorationUnitScratch{SGRProj: make([]int32, sizes.SGRProj)})
	if err != nil {
		t.Fatal(err)
	}
	if result != (RestorationUnitApplyResult{Type: parser.RestorationSGRProj, Filtered: true}) {
		t.Fatalf("result=%+v", result)
	}
	if err := av1restoration.ApplySelfguidedRestoration(src, stride, origin, want, width, width, height, int(unit.SGRProj.ParamsIndex), unit.SGRProj.XQD, bitDepth, make([]int32, sizes.SGRProj)); err != nil {
		t.Fatal(err)
	}
	assertSamplesEqual(t, got, want)
}

func TestApplyRestorationUnitRejectsInvalidInputs(t *testing.T) {
	const width, height = 8, 8
	stride := width + 2*av1restoration.WienerHalfwin
	origin := av1restoration.WienerHalfwin*stride + av1restoration.WienerHalfwin
	src := makeRestorationApplySource(stride, height+2*av1restoration.WienerHalfwin, 8)
	dst := make([]uint16, width*height)
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, RestorationUnit{Type: parser.RestorationSwitchable}, 8, RestorationUnitScratch{}); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("invalid type err=%v want %v", err, ErrInvalidPlan)
	}
	bad := append([]uint16(nil), src...)
	bad[origin] = 256
	if _, err := ApplyRestorationUnit(bad, stride, origin, dst, width, width, height, RestorationUnit{Type: parser.RestorationNone}, 8, RestorationUnitScratch{}); !errors.Is(err, ErrInvalidPlan) {
		t.Fatalf("bad none sample err=%v want %v", err, ErrInvalidPlan)
	}
	unit := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, unit, 8, RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener-1)}); !errors.Is(err, av1restoration.ErrInvalidRestoration) {
		t.Fatalf("short scratch err=%v want %v", err, av1restoration.ErrInvalidRestoration)
	}
}

func TestApplyRestorationUnitAllocs(t *testing.T) {
	const width, height = 8, 8
	const stride = width + 2*av1restoration.WienerHalfwin
	const origin = av1restoration.WienerHalfwin*stride + av1restoration.WienerHalfwin
	src := makeRestorationApplySource(stride, height+2*av1restoration.WienerHalfwin, 8)
	dst := make([]uint16, width*height)
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		t.Fatal(err)
	}
	unit := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	scratch := RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)}
	allocs := testing.AllocsPerRun(1000, func() {
		if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, unit, 8, scratch); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("ApplyRestorationUnit allocated: %f", allocs)
	}
}

func FuzzApplyRestorationUnitNone(f *testing.F) {
	f.Add(uint8(5), uint8(4), uint8(0), []byte{0, 1, 2, 3, 255})
	f.Add(uint8(31), uint8(17), uint8(2), []byte{255, 1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, rawW uint8, rawH uint8, rawDepth uint8, data []byte) {
		width := int(rawW%32) + 1
		height := int(rawH%32) + 1
		bitDepths := [...]uint8{8, 10, 12}
		bitDepth := bitDepths[rawDepth%uint8(len(bitDepths))]
		max := uint16((1 << bitDepth) - 1)
		stride := width + 3
		origin := stride + 1
		src := make([]uint16, origin+(height-1)*stride+width)
		for i := range src {
			src[i] = fuzzRestorationApplySample(data, i, max)
		}
		dst := make([]uint16, width*height)
		if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, RestorationUnit{Type: parser.RestorationNone}, bitDepth, RestorationUnitScratch{}); err != nil {
			t.Fatalf("ApplyRestorationUnit none err=%v", err)
		}
		for row := 0; row < height; row++ {
			for col := 0; col < width; col++ {
				if got, want := dst[row*width+col], src[origin+row*stride+col]; got != want {
					t.Fatalf("row=%d col=%d got=%d want %d", row, col, got, want)
				}
			}
		}
	})
}

func BenchmarkApplyRestorationUnitWiener(b *testing.B) {
	const width, height = 64, 64
	const bitDepth = 12
	stride := width + 2*av1restoration.WienerHalfwin
	origin := av1restoration.WienerHalfwin*stride + av1restoration.WienerHalfwin
	src := makeRestorationApplySource(stride, height+2*av1restoration.WienerHalfwin, bitDepth)
	dst := make([]uint16, width*height)
	sizes, err := RestorationUnitScratchLen(width, height)
	if err != nil {
		b.Fatal(err)
	}
	unit := RestorationUnit{Type: parser.RestorationWiener, Wiener: av1restoration.DefaultWienerInfo()}
	scratch := RestorationUnitScratch{Wiener: make([]uint16, sizes.Wiener)}
	b.ReportAllocs()
	b.SetBytes(width * height * 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ApplyRestorationUnit(src, stride, origin, dst, width, width, height, unit, bitDepth, scratch); err != nil {
			b.Fatal(err)
		}
	}
}

func makeRestorationApplySource(stride int, height int, bitDepth uint8) []uint16 {
	max := uint16((1 << bitDepth) - 1)
	src := make([]uint16, stride*height)
	for row := 0; row < height; row++ {
		for col := 0; col < stride; col++ {
			src[row*stride+col] = uint16((row*37 + col*19 + row*col) & int(max))
		}
	}
	return src
}

func assertSamplesEqual(t *testing.T, got []uint16, want []uint16) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("sample[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func fuzzRestorationApplySample(data []byte, index int, max uint16) uint16 {
	if len(data) == 0 {
		return 0
	}
	lo := uint16(data[index%len(data)])
	hi := uint16(data[(index+1)%len(data)])
	return (lo | hi<<8) & max
}
