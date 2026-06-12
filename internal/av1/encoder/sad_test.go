package encoder

import (
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/frame"
	"github.com/thesyncim/goav1/internal/av1/motion"
)

// TestSAD8x8ImplMatchesPureGo proves the dispatched SAD kernel is bit-exact
// with the portable reference across random data, strides, and offsets
// (including totals far above any early-exit threshold).
func TestSAD8x8ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	const stride = 73 // deliberately odd
	src := make([]byte, stride*64)
	ref := make([]byte, stride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*48) + rng.Intn(stride-8)
		want := sad8x8PureGo(src[off:], ref[off:], stride, 1<<30)
		got := sad8x8Impl(src[off:], ref[off:], stride, 1<<30)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

// TestSAD16x16ImplMatchesPureGo proves the 16x16 kernel bit-exact with the
// portable reference.
func TestSAD16x16ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	const stride = 91
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*64) + rng.Intn(stride-16)
		want := sad16x16PureGo(src[off:], ref[off:], stride)
		got := sad16x16Impl(src[off:], ref[off:], stride)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

// TestSAD32x32ImplMatchesPureGo proves the 32x32 kernel bit-exact with the
// portable reference.
func TestSAD32x32ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(45))
	const stride = 113
	src := make([]byte, stride*128)
	ref := make([]byte, stride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*80) + rng.Intn(stride-32)
		want := sad32x32PureGo(src[off:], ref[off:], stride)
		got := sad32x32Impl(src[off:], ref[off:], stride)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

// TestSAD8x8DualImplMatchesPureGo proves the two-stride kernel bit-exact
// with the portable reference.
func TestSAD8x8DualImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(47))
	const srcStride, refStride = 83, 19
	src := make([]byte, srcStride*64)
	ref := make([]byte, refStride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*48) + rng.Intn(srcStride-8)
		ro := rng.Intn(refStride*48) + rng.Intn(refStride-8)
		want := sad8x8DualPureGo(src[so:], srcStride, ref[ro:], refStride)
		got := sad8x8DualImpl(src[so:], srcStride, ref[ro:], refStride)
		if got != want {
			t.Fatalf("so %d ro %d: impl %d want %d", so, ro, got, want)
		}
	}
}

func TestSAD16x16DualImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(4716))
	const srcStride, refStride = 83, 29
	src := make([]byte, srcStride*96)
	ref := make([]byte, refStride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*64) + rng.Intn(srcStride-16)
		ro := rng.Intn(refStride*64) + rng.Intn(refStride-16)
		want := sad16x16DualPureGo(src[so:], srcStride, ref[ro:], refStride)
		got := sad16x16DualImpl(src[so:], srcStride, ref[ro:], refStride)
		if got != want {
			t.Fatalf("so %d ro %d: impl %d want %d", so, ro, got, want)
		}
	}
}

func TestSAD32x32DualImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(4732))
	const srcStride, refStride = 117, 41
	src := make([]byte, srcStride*128)
	ref := make([]byte, refStride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*80) + rng.Intn(srcStride-32)
		ro := rng.Intn(refStride*80) + rng.Intn(refStride-32)
		want := sad32x32DualPureGo(src[so:], srcStride, ref[ro:], refStride)
		got := sad32x32DualImpl(src[so:], srcStride, ref[ro:], refStride)
		if got != want {
			t.Fatalf("so %d ro %d: impl %d want %d", so, ro, got, want)
		}
	}
}

// TestSAD8x8CompoundAvgBlockImplMatchesPureGo proves the compound average SAD
// kernel is bit-exact with the portable rounded-average reference.
func TestSAD8x8CompoundAvgBlockImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(48))
	const srcStride, ref0Stride, ref1Stride = 83, 31, 47
	src := make([]byte, srcStride*64)
	ref0 := make([]byte, ref0Stride*64)
	ref1 := make([]byte, ref1Stride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref0 {
		ref0[i] = uint8(rng.Intn(256))
	}
	for i := range ref1 {
		ref1[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		so := rng.Intn(srcStride*48) + rng.Intn(srcStride-8)
		r0o := rng.Intn(ref0Stride*48) + rng.Intn(ref0Stride-8)
		r1o := rng.Intn(ref1Stride*48) + rng.Intn(ref1Stride-8)
		want := sad8x8CompoundAvgBlockPureGo(src[so:], srcStride, ref0[r0o:], ref0Stride, ref1[r1o:], ref1Stride)
		got := sad8x8CompoundAvgBlockImpl(src[so:], srcStride, ref0[r0o:], ref0Stride, ref1[r1o:], ref1Stride)
		if got != want {
			t.Fatalf("so %d r0o %d r1o %d: impl %d want %d", so, r0o, r1o, got, want)
		}
	}
}

func TestSADRectBlockMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(49))
	const stride = 97
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for _, size := range [][2]int{{16, 8}, {8, 16}, {32, 16}, {16, 32}, {24, 8}} {
		bw, bh := size[0], size[1]
		for range 500 {
			base := rng.Intn(96-bh)*stride + rng.Intn(stride-bw)
			refBase := rng.Intn(96-bh)*stride + rng.Intn(stride-bw)
			want := sadRectBlockReference(src, ref, base, refBase, stride, bw, bh)
			got := sadRectBlock(src, ref, base, refBase, stride, bw, bh, 1<<30)
			if got != want {
				t.Fatalf("%dx%d base=%d refBase=%d: got %d want %d", bw, bh, base, refBase, got, want)
			}
		}
	}
}

func sadRectBlockReference(src, ref []byte, base, refBase, stride, bw, bh int) int {
	total := 0
	for r := range bh {
		row := base + r*stride
		refRow := refBase + r*stride
		for c := range bw {
			d := int(src[row+c]) - int(ref[refRow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func TestFullPelDiamondSearchSeededMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(53))
	const (
		width  = 96
		height = 80
		stride = 112
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for _, n := range []int{8, 12, 16, 32} {
		for _, reach := range []int{4, 8} {
			for range 200 {
				px := rng.Intn((width-n)/8+1) * 8
				py := rng.Intn((height-n)/8+1) * 8
				seedDX := rng.Intn(41) - 20
				seedDY := rng.Intn(41) - 20
				wantDX, wantDY, wantSAD := fullPelDiamondSearchSeededReference(src, ref, stride, width, height, px, py, n, seedDX, seedDY, reach)
				gotDX, gotDY, gotSAD := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, n, seedDX, seedDY, reach)
				if gotDX != wantDX || gotDY != wantDY || gotSAD != wantSAD {
					t.Fatalf("n=%d reach=%d px=%d py=%d seed=(%d,%d): got (%d,%d,%d) want (%d,%d,%d)",
						n, reach, px, py, seedDX, seedDY, gotDX, gotDY, gotSAD, wantDX, wantDY, wantSAD)
				}
			}
		}
	}
}

func fullPelDiamondSearchSeededReference(src, ref []byte, stride, width, height, px, py, n int, seedDX, seedDY, reach int) (int, int, int) {
	sad := func(dx, dy, limit int) int {
		base := py*stride + px
		refBase := (py+dy)*stride + px + dx
		switch n {
		case 8:
			return sad8x8Impl(src[base:], ref[refBase:], stride, limit)
		case 16:
			return sad16x16Impl(src[base:], ref[refBase:], stride)
		case 32:
			return sad16x16Impl(src[base:], ref[refBase:], stride) +
				sad16x16Impl(src[base+16:], ref[refBase+16:], stride) +
				sad16x16Impl(src[base+16*stride:], ref[refBase+16*stride:], stride) +
				sad16x16Impl(src[base+16*stride+16:], ref[refBase+16*stride+16:], stride)
		}
		total := 0
		for r := range n {
			row := base + r*stride
			refRow := refBase + r*stride
			for c := range n {
				d := int(src[row+c]) - int(ref[refRow+c])
				if d < 0 {
					d = -d
				}
				total += d
			}
			if total >= limit {
				return total
			}
		}
		return total
	}
	clampLo := func(v, lo int) int {
		if v < lo {
			return lo
		}
		return v
	}
	clampHi := func(v, hi int) int {
		if v > hi {
			return hi
		}
		return v
	}
	seedDX = clampHi(clampLo(seedDX, -px), width-n-px) &^ 1
	seedDY = clampHi(clampLo(seedDY, -py), height-n-py) &^ 1
	minDX := clampLo(seedDX-reach, -px)
	maxDX := clampHi(seedDX+reach, width-n-px)
	minDY := clampLo(seedDY-reach, -py)
	maxDY := clampHi(seedDY+reach, height-n-py)

	bestDX, bestDY := 0, 0
	bestSAD := sad(0, 0, 1<<30)
	if bestSAD <= n*n*2 {
		return 0, 0, bestSAD
	}
	for dy := minDY &^ 1; dy <= maxDY; dy += 4 {
		for dx := minDX &^ 1; dx <= maxDX; dx += 4 {
			if dx == 0 && dy == 0 {
				continue
			}
			if s := sad(dx, dy, bestSAD); s < bestSAD {
				bestSAD, bestDX, bestDY = s, dx, dy
			}
		}
	}
	for _, cand := range [4][2]int{{bestDX + 2, bestDY}, {bestDX - 2, bestDY}, {bestDX, bestDY + 2}, {bestDX, bestDY - 2}} {
		dx, dy := cand[0], cand[1]
		if dx < minDX || dx > maxDX || dy < minDY || dy > maxDY {
			continue
		}
		if s := sad(dx, dy, bestSAD); s < bestSAD {
			bestSAD, bestDX, bestDY = s, dx, dy
		}
	}
	return bestDX, bestDY, bestSAD
}

func TestSubpelRefineMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(59))
	const (
		width  = 128
		height = 96
		stride = 144
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for _, n := range []int{8, 16, 32} {
		for range 200 {
			px := 16 + rng.Intn((width-n-32)/8+1)*8
			py := 16 + rng.Intn((height-n-32)/8+1)*8
			mv := motion.Vector{
				Row: int16((rng.Intn(5) - 2) * 16),
				Col: int16((rng.Intn(5) - 2) * 16),
			}
			base := py*stride + px
			refBase := (py+int(mv.Row)/8)*stride + px + int(mv.Col)/8
			bestSAD := sadBlock(src, ref, base, refBase, stride, n, 1<<30)

			var gotState, wantState lossyEncodeState
			gotMV, gotSAD := gotState.subpelRefine(src, ref, stride, width, height, px, py, n, mv, bestSAD)
			wantMV, wantSAD := wantState.subpelRefineReference(src, ref, stride, width, height, px, py, n, mv, bestSAD)
			if gotMV != wantMV || gotSAD != wantSAD {
				t.Fatalf("n=%d px=%d py=%d mv=%+v best=%d: got (%+v,%d) want (%+v,%d)",
					n, px, py, mv, bestSAD, gotMV, gotSAD, wantMV, wantSAD)
			}
		}
	}
}

func (st *lossyEncodeState) subpelRefineReference(src, refPlane []byte, stride, width, height, px, py, n int, mv motion.Vector, bestSAD int) (motion.Vector, int) {
	st.prober.Init(frame.Plane{
		Pix: refPlane, Stride: stride, Width: width, Height: height,
	}, px+int(mv.Col)>>3, py+int(mv.Row)>>3, n)
	startMV := mv
	exact := func(cand motion.Vector) int {
		if !st.prober.Predict(st.sadScratch[:n*n], motion.Vector{Row: cand.Row - startMV.Row, Col: cand.Col - startMV.Col}) {
			if err := predictInto(st.sadScratch[:n*n], refPlane, stride, width, height, px, py, n, n, cand, false, false); err != nil {
				return -1
			}
		}
		base := py*stride + px
		s := 0
		for r := 0; r < n; r += 8 {
			for c := 0; c < n; c += 8 {
				s += sad8x8DualImpl(src[base+r*stride+c:], stride, st.sadScratch[r*n+c:], n)
			}
		}
		return s
	}
	start := mv
	center := bestSAD
	var half [4]int
	offs := [4]motion.Vector{
		{Row: start.Row, Col: start.Col - 4},
		{Row: start.Row, Col: start.Col + 4},
		{Row: start.Row - 4, Col: start.Col},
		{Row: start.Row + 4, Col: start.Col},
	}
	for i, cand := range offs {
		s := exact(cand)
		half[i] = s
		if s >= 0 && s < bestSAD {
			bestSAD, mv = s, cand
		}
	}
	quarterAxis := func(sl, sr int) int {
		if sl < 0 || sr < 0 {
			return 0
		}
		den := sl + sr - 2*center
		if den <= 0 {
			return 0
		}
		est := (sl - sr) * 2 / den
		if est > 4 {
			est = 4
		}
		if est < -4 {
			est = -4
		}
		return est
	}
	estX := quarterAxis(half[0], half[1])
	estY := quarterAxis(half[2], half[3])
	for _, e := range [2][2]int{{estX &^ 1, estY &^ 1}, {(estX + 1) &^ 1, (estY + 1) &^ 1}} {
		if e[0] == 0 && e[1] == 0 {
			continue
		}
		cand := motion.Vector{Row: start.Row + int16(e[1]), Col: start.Col + int16(e[0])}
		if cand == mv || cand == start {
			continue
		}
		if s := exact(cand); s >= 0 && s < bestSAD {
			bestSAD, mv = s, cand
		}
	}
	return mv, bestSAD
}

func BenchmarkSAD8x8(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad8x8Impl(src, ref, 64, 1<<30)
	}
}

func BenchmarkSAD16x16(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad16x16Impl(src, ref, 64)
	}
}

func BenchmarkSAD32x32(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad32x32Impl(src, ref, 64)
	}
}

func BenchmarkSAD32x32Composed16x16(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = sad16x16Impl(src, ref, 64) +
			sad16x16Impl(src[16:], ref[16:], 64) +
			sad16x16Impl(src[16*64:], ref[16*64:], 64) +
			sad16x16Impl(src[16*64+16:], ref[16*64+16:], 64)
	}
}

func BenchmarkSAD16x16Dual(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 16*64)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad16x16Dual(src, 64, ref, 16)
	}
}

func BenchmarkSAD16x16DualComposed8x8(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 16*64)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = sad8x8Dual(src, 64, ref, 16) +
			sad8x8Dual(src[8:], 64, ref[8:], 16) +
			sad8x8Dual(src[8*64:], 64, ref[8*16:], 16) +
			sad8x8Dual(src[8*64+8:], 64, ref[8*16+8:], 16)
	}
}

func BenchmarkSAD32x32Dual(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 32*64)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad32x32Dual(src, 64, ref, 32)
	}
}

func BenchmarkSAD32x32DualComposed8x8(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 32*64)
	for i := range src {
		src[i] = uint8(i * 7)
	}
	for i := range ref {
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sum := 0
		for r := 0; r < 32; r += 8 {
			for c := 0; c < 32; c += 8 {
				sum += sad8x8Dual(src[r*64+c:], 64, ref[r*32+c:], 32)
			}
		}
		_ = sum
	}
}

func BenchmarkSADRect32x16(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sadRectBlock(src, ref, 0, 0, 64, 32, 16, 1<<30)
	}
}

func BenchmarkSADRect32x16ScalarReference(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sadRectBlockReference(src, ref, 0, 0, 64, 32, 16)
	}
}

func BenchmarkSAD8x8CompoundAvgBlock(b *testing.B) {
	src := make([]byte, 64*64)
	ref0 := make([]byte, 64*64)
	ref1 := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref0[i] = uint8(i * 13)
		ref1[i] = uint8(i*17 + 3)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad8x8CompoundAvgBlock(src, 64, ref0, 64, ref1, 64)
	}
}

func BenchmarkSAD8x8CompoundAvgBlockScalarReference(b *testing.B) {
	src := make([]byte, 64*64)
	ref0 := make([]byte, 64*64)
	ref1 := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref0[i] = uint8(i * 13)
		ref1[i] = uint8(i*17 + 3)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad8x8CompoundAvgBlockPureGo(src, 64, ref0, 64, ref1, 64)
	}
}
