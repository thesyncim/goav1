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

func TestSAD8x8x4Step4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const stride = 77
	src := make([]byte, stride*64)
	ref := make([]byte, stride*64)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*48) + rng.Intn(stride-27)
		w0, w1, w2, w3 := sad8x8x4Step4PureGo(src[off:], ref[off:], stride)
		g0, g1, g2, g3 := sad8x8x4Step4Impl(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD8x8x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(424))
	const stride = 79
	src := make([]byte, stride*80)
	ref := make([]byte, stride*80)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		row := rng.Intn(56) + 8
		col := rng.Intn(stride-24) + 8
		off := row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0, w1, w2, w3 := sad8x8x4PureGo(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		g0, g1, g2, g3 := sad8x8x4Impl(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD16x16x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(425))
	const stride = 91
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		row := rng.Intn(56) + 16
		col := rng.Intn(stride-40) + 16
		off := row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0, w1, w2, w3 := sad16x16x4PureGo(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		g0, g1, g2, g3 := sad16x16x4Impl(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD32x32x4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(426))
	const stride = 137
	src := make([]byte, stride*128)
	ref := make([]byte, stride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		row := rng.Intn(56) + 32
		col := rng.Intn(stride-72) + 32
		off := row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0, w1, w2, w3 := sad32x32x4PureGo(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		g0, g1, g2, g3 := sad32x32x4Impl(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
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

func TestSAD16x16x4Step4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(44))
	const stride = 93
	src := make([]byte, stride*96)
	ref := make([]byte, stride*96)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*64) + rng.Intn(stride-35)
		w0, w1, w2, w3 := sad16x16x4Step4PureGo(src[off:], ref[off:], stride)
		g0, g1, g2, g3 := sad16x16x4Step4Impl(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
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

func TestSAD64x64ComposedMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4564))
	const stride = 151
	src := make([]byte, stride*160)
	ref := make([]byte, stride*160)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*96) + rng.Intn(stride-64)
		want := sadRectBlockReference(src, ref, off, off, stride, 64, 64)
		got := sad64x64(src[off:], ref[off:], stride)
		if got != want {
			t.Fatalf("off %d: impl %d want %d", off, got, want)
		}
	}
}

func TestSAD64x64x4Step4MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4565))
	const (
		stride = 151
		height = 160
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		row := rng.Intn(height - 64)
		col := rng.Intn(stride - 76)
		off := row*stride + col
		w0 := sadRectBlockReference(src, ref, off, off, stride, 64, 64)
		w1 := sadRectBlockReference(src, ref, off, off+4, stride, 64, 64)
		w2 := sadRectBlockReference(src, ref, off, off+8, stride, 64, 64)
		w3 := sadRectBlockReference(src, ref, off, off+12, stride, 64, 64)
		g0, g1, g2, g3 := sad64x64x4Step4(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD64x64x4MatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4566))
	const (
		stride = 160
		height = 192
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 500 {
		row := rng.Intn(56) + 64
		col := rng.Intn(stride-136) + 64
		off := row*stride + col
		ref0 := off + 2
		ref1 := off - 2
		ref2 := off + 2*stride
		ref3 := off - 2*stride
		w0 := sadRectBlockReference(src, ref, off, ref0, stride, 64, 64)
		w1 := sadRectBlockReference(src, ref, off, ref1, stride, 64, 64)
		w2 := sadRectBlockReference(src, ref, off, ref2, stride, 64, 64)
		w3 := sadRectBlockReference(src, ref, off, ref3, stride, 64, 64)
		g0, g1, g2, g3 := sad64x64x4(src[off:], ref[ref0:], ref[ref1:], ref[ref2:], ref[ref3:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
		}
	}
}

func TestSAD32x32x4Step4ImplMatchesPureGo(t *testing.T) {
	rng := rand.New(rand.NewSource(46))
	const stride = 117
	src := make([]byte, stride*128)
	ref := make([]byte, stride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
		ref[i] = uint8(rng.Intn(256))
	}
	for range 2000 {
		off := rng.Intn(stride*80) + rng.Intn(stride-51)
		w0, w1, w2, w3 := sad32x32x4Step4PureGo(src[off:], ref[off:], stride)
		g0, g1, g2, g3 := sad32x32x4Step4Impl(src[off:], ref[off:], stride)
		if g0 != w0 || g1 != w1 || g2 != w2 || g3 != w3 {
			t.Fatalf("off %d: impl (%d,%d,%d,%d) want (%d,%d,%d,%d)",
				off, g0, g1, g2, g3, w0, w1, w2, w3)
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

func TestSADRectDualBlockMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(4901))
	const srcStride, refStride = 97, 67
	src := make([]byte, srcStride*128)
	ref := make([]byte, refStride*128)
	for i := range src {
		src[i] = uint8(rng.Intn(256))
	}
	for i := range ref {
		ref[i] = uint8(rng.Intn(256))
	}
	for _, size := range [][2]int{{8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 16}, {16, 32}, {32, 32}, {64, 64}, {24, 8}} {
		bw, bh := size[0], size[1]
		for range 500 {
			so := rng.Intn(128-bh)*srcStride + rng.Intn(srcStride-bw)
			ro := rng.Intn(128-bh)*refStride + rng.Intn(refStride-bw)
			want := sadRectDualBlockReference(src[so:], srcStride, ref[ro:], refStride, bw, bh)
			got := sadRectDualBlock(src[so:], srcStride, ref[ro:], refStride, bw, bh)
			if got != want {
				t.Fatalf("%dx%d so=%d ro=%d: got %d want %d", bw, bh, so, ro, got, want)
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

func sadRectDualBlockReference(src []byte, srcStride int, ref []byte, refStride int, bw, bh int) int {
	total := 0
	for r := range bh {
		srow := r * srcStride
		rrow := r * refStride
		for c := range bw {
			d := int(src[srow+c]) - int(ref[rrow+c])
			if d < 0 {
				d = -d
			}
			total += d
		}
	}
	return total
}

func TestFullPelDiamondSearchSeededMatchesLibaomReference(t *testing.T) {
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
	for _, n := range []int{8, 12, 16, 32, 64} {
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
	seedDX = minInt(maxInt(seedDX, -px), width-n-px)
	seedDY = minInt(maxInt(seedDY, -py), height-n-py)

	mesh := func(startDX, startDY, searchRange, step int) (int, int, int) {
		if step < 1 {
			step = 1
		}
		startDX = minInt(maxInt(startDX, -px), width-n-px)
		startDY = minInt(maxInt(startDY, -py), height-n-py)
		startCol := maxInt(-searchRange, -px-startDX)
		endCol := minInt(searchRange, width-n-px-startDX)
		startRow := maxInt(-searchRange, -py-startDY)
		endRow := minInt(searchRange, height-n-py-startDY)
		bestDX, bestDY := startDX, startDY
		bestSAD := sad(bestDX, bestDY, 1<<30)
		colStep := step
		if step <= 1 {
			colStep = 4
		}
		for row := startRow; row <= endRow; row += step {
			for col := startCol; col <= endCol; col += colStep {
				dx := startDX + col
				dy := startDY + row
				if step <= 1 && col+3 <= endCol {
					for i := 0; i < 4; i++ {
						if s := sad(dx+i, dy, bestSAD); s < bestSAD {
							bestSAD, bestDX, bestDY = s, dx+i, dy
						}
					}
					continue
				}
				if s := sad(dx, dy, bestSAD); s < bestSAD {
					bestSAD, bestDX, bestDY = s, dx, dy
				}
			}
		}
		return bestDX, bestDY, bestSAD
	}
	bestDX, bestDY, _ := mesh(seedDX, seedDY, reach, 4)
	return mesh(bestDX, bestDY, 2, 2)
}

func TestSubpelRefineMatchesLibaomTreeReference(t *testing.T) {
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
			wantMV, wantSAD := wantState.subpelRefineReference(src, ref, stride, width, height, px, py, n, mv, bestSAD, realtimeSubpelStopQuarter)
			if gotMV != wantMV || gotSAD != wantSAD {
				t.Fatalf("n=%d px=%d py=%d mv=%+v best=%d: got (%+v,%d) want (%+v,%d)",
					n, px, py, mv, bestSAD, gotMV, gotSAD, wantMV, wantSAD)
			}
			gotHalfMV, gotHalfSAD := gotState.subpelRefineWithStop(src, ref, stride, width, height, px, py, n, mv, bestSAD, realtimeSubpelStopHalf)
			wantHalfMV, wantHalfSAD := wantState.subpelRefineReference(src, ref, stride, width, height, px, py, n, mv, bestSAD, realtimeSubpelStopHalf)
			if gotHalfMV != wantHalfMV || gotHalfSAD != wantHalfSAD {
				t.Fatalf("half n=%d px=%d py=%d mv=%+v best=%d: got (%+v,%d) want (%+v,%d)",
					n, px, py, mv, bestSAD, gotHalfMV, gotHalfSAD, wantHalfMV, wantHalfSAD)
			}
			gotFullMV, gotFullSAD := gotState.subpelRefineWithStop(src, ref, stride, width, height, px, py, n, mv, bestSAD, realtimeSubpelStopFull)
			if gotFullMV != mv || gotFullSAD != bestSAD {
				t.Fatalf("full stop n=%d: got (%+v,%d) want (%+v,%d)", n, gotFullMV, gotFullSAD, mv, bestSAD)
			}
		}
	}
}

func (st *lossyEncodeState) subpelRefineReference(src, refPlane []byte, stride, width, height, px, py, n int, mv motion.Vector, bestSAD int, stop realtimeSubpelStop) (motion.Vector, int) {
	if stop == realtimeSubpelStopFull {
		return mv, bestSAD
	}
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
	const invalid = int(^uint(0) >> 1)
	check := func(cand motion.Vector) int {
		s := exact(cand)
		if s < 0 {
			return invalid
		}
		if s < bestSAD {
			bestSAD, mv = s, cand
		}
		return s
	}
	checkChanged := func(cand motion.Vector) bool {
		oldSAD, oldMV := bestSAD, mv
		check(cand)
		return bestSAD != oldSAD || mv != oldMV
	}
	diagStep := func(step int16, left, right, up, down int) motion.Vector {
		diag := motion.Vector{Row: step, Col: step}
		if up <= down {
			diag.Row = -step
		}
		if left <= right {
			diag.Col = -step
		}
		return diag
	}
	firstLevel := func(center motion.Vector, step int16) motion.Vector {
		left := check(motion.Vector{Row: center.Row, Col: center.Col - step})
		right := check(motion.Vector{Row: center.Row, Col: center.Col + step})
		up := check(motion.Vector{Row: center.Row - step, Col: center.Col})
		down := check(motion.Vector{Row: center.Row + step, Col: center.Col})
		diag := diagStep(step, left, right, up, down)
		check(motion.Vector{Row: center.Row + diag.Row, Col: center.Col + diag.Col})
		return diag
	}
	secondLevel := func(center motion.Vector, diag motion.Vector) {
		if center == mv {
			return
		}
		if center.Row == mv.Row {
			diag.Row *= -1
		} else if center.Col == mv.Col {
			diag.Col *= -1
		}
		rowBias := motion.Vector{Row: mv.Row + diag.Row, Col: mv.Col}
		colBias := motion.Vector{Row: mv.Row, Col: mv.Col + diag.Col}
		diagBias := motion.Vector{Row: mv.Row + diag.Row, Col: mv.Col + diag.Col}
		hasBetter := checkChanged(rowBias)
		if checkChanged(colBias) {
			hasBetter = true
		}
		if hasBetter {
			check(diagBias)
		}
	}

	for step := int16(4); ; step >>= 1 {
		center := mv
		diag := firstLevel(center, step)
		if center != mv {
			secondLevel(center, diag)
		}
		if stop == realtimeSubpelStopHalf || step == 2 {
			break
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

func BenchmarkSAD8x8x4Step4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad8x8x4Step4Impl(src, ref, 64)
	}
}

func BenchmarkSAD8x8x4Step4Composed(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = sad8x8Impl(src, ref, 64, 1<<30) +
			sad8x8Impl(src, ref[4:], 64, 1<<30) +
			sad8x8Impl(src, ref[8:], 64, 1<<30) +
			sad8x8Impl(src, ref[12:], 64, 1<<30)
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

func BenchmarkSAD16x16x4Step4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad16x16x4Step4Impl(src, ref, 64)
	}
}

func BenchmarkSAD16x16x4Step4Composed(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = sad16x16Impl(src, ref, 64) +
			sad16x16Impl(src, ref[4:], 64) +
			sad16x16Impl(src, ref[8:], 64) +
			sad16x16Impl(src, ref[12:], 64)
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

func BenchmarkSAD32x32x4Step4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad32x32x4Step4Impl(src, ref, 64)
	}
}

func BenchmarkSAD32x32x4Step4Composed(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_ = sad32x32Impl(src, ref, 64) +
			sad32x32Impl(src, ref[4:], 64) +
			sad32x32Impl(src, ref[8:], 64) +
			sad32x32Impl(src, ref[12:], 64)
	}
}

func BenchmarkSAD32x32x4(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 32*96 + 32
	b.ReportAllocs()
	for b.Loop() {
		sad32x32x4(src[off:], ref[off+2:], ref[off-2:], ref[off+2*96:], ref[off-2*96:], 96)
	}
}

func BenchmarkSAD32x32x4ScalarReference(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 32*96 + 32
	b.ReportAllocs()
	for b.Loop() {
		sad32x32x4PureGo(src[off:], ref[off+2:], ref[off-2:], ref[off+2*96:], ref[off-2*96:], 96)
	}
}

func BenchmarkSAD32x32x4Composed(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 32*96 + 32
	b.ReportAllocs()
	for b.Loop() {
		_ = sad32x32(src[off:], ref[off+2:], 96) +
			sad32x32(src[off:], ref[off-2:], 96) +
			sad32x32(src[off:], ref[off+2*96:], 96) +
			sad32x32(src[off:], ref[off-2*96:], 96)
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

func BenchmarkSAD64x64(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sad64x64(src, ref, 96)
	}
}

func BenchmarkSAD64x64x4Step4(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad64x64x4Step4(src, ref, 96)
	}
}

func BenchmarkSAD64x64x4(b *testing.B) {
	const stride = 160
	src := make([]byte, stride*160)
	ref := make([]byte, stride*160)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 64*stride + 64
	b.ReportAllocs()
	for b.Loop() {
		_, _, _, _ = sad64x64x4(src[off:], ref[off+2:], ref[off-2:], ref[off+2*stride:], ref[off-2*stride:], stride)
	}
}

func BenchmarkSAD64x64x4ScalarReference(b *testing.B) {
	const stride = 160
	src := make([]byte, stride*160)
	ref := make([]byte, stride*160)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 64*stride + 64
	b.ReportAllocs()
	for b.Loop() {
		_ = sadRectBlockReference(src, ref, off, off+2, stride, 64, 64) +
			sadRectBlockReference(src, ref, off, off-2, stride, 64, 64) +
			sadRectBlockReference(src, ref, off, off+2*stride, stride, 64, 64) +
			sadRectBlockReference(src, ref, off, off-2*stride, stride, 64, 64)
	}
}

func BenchmarkSAD64x64x4Composed(b *testing.B) {
	const stride = 160
	src := make([]byte, stride*160)
	ref := make([]byte, stride*160)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 64*stride + 64
	b.ReportAllocs()
	for b.Loop() {
		_ = sad64x64(src[off:], ref[off+2:], stride) +
			sad64x64(src[off:], ref[off-2:], stride) +
			sad64x64(src[off:], ref[off+2*stride:], stride) +
			sad64x64(src[off:], ref[off-2*stride:], stride)
	}
}

func BenchmarkSAD64x64ScalarReference(b *testing.B) {
	src := make([]byte, 96*96)
	ref := make([]byte, 96*96)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i * 13)
	}
	b.ReportAllocs()
	for b.Loop() {
		sadRectBlockReference(src, ref, 0, 0, 96, 64, 64)
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

func BenchmarkSADRectDual32x16(b *testing.B) {
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
		sadRectDualBlock(src, 64, ref, 32, 32, 16)
	}
}

func BenchmarkSADRectDual32x16Composed8x8(b *testing.B) {
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
		for r := 0; r < 16; r += 8 {
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

func BenchmarkSAD8x8x4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		sad8x8x4(src[off:], ref[off+2:], ref[off-2:], ref[off+2*64:], ref[off-2*64:], 64)
	}
}

func BenchmarkSAD8x8x4ScalarReference(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		sad8x8x4PureGo(src[off:], ref[off+2:], ref[off-2:], ref[off+2*64:], ref[off-2*64:], 64)
	}
}

func BenchmarkSAD16x16x4(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		sad16x16x4(src[off:], ref[off+2:], ref[off-2:], ref[off+2*64:], ref[off-2*64:], 64)
	}
}

func BenchmarkSAD16x16x4ScalarReference(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		sad16x16x4PureGo(src[off:], ref[off+2:], ref[off-2:], ref[off+2*64:], ref[off-2*64:], 64)
	}
}

func BenchmarkSAD16x16x4Composed(b *testing.B) {
	src := make([]byte, 64*64)
	ref := make([]byte, 64*64)
	for i := range src {
		src[i] = uint8(i * 7)
		ref[i] = uint8(i*13 + 5)
	}
	off := 16*64 + 16
	b.ReportAllocs()
	for b.Loop() {
		_ = sad16x16(src[off:], ref[off+2:], 64) +
			sad16x16(src[off:], ref[off-2:], 64) +
			sad16x16(src[off:], ref[off+2*64:], 64) +
			sad16x16(src[off:], ref[off-2*64:], 64)
	}
}

var fullPelDiamondBenchSink int

func BenchmarkFullPelDiamondSearch8(b *testing.B) {
	const (
		width  = 96
		height = 80
		stride = 112
		px     = 32
		py     = 24
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}

	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		dx, dy, sad := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, 8, 0, 0, fullPelReach)
		sum += dx + dy + sad
	}
	fullPelDiamondBenchSink = sum
}

func BenchmarkFullPelDiamondSearch16(b *testing.B) {
	const (
		width  = 112
		height = 96
		stride = 128
		px     = 40
		py     = 32
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}

	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		dx, dy, sad := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, 16, 0, 0, fullPelReach)
		sum += dx + dy + sad
	}
	fullPelDiamondBenchSink = sum
}

func BenchmarkFullPelDiamondSearch32(b *testing.B) {
	const (
		width  = 128
		height = 112
		stride = 160
		px     = 48
		py     = 40
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}

	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		dx, dy, sad := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, 32, 0, 0, fullPelReach)
		sum += dx + dy + sad
	}
	fullPelDiamondBenchSink = sum
}

func BenchmarkFullPelDiamondSearch64(b *testing.B) {
	const (
		width  = 176
		height = 160
		stride = 192
		px     = 64
		py     = 48
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}

	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		dx, dy, sad := fullPelDiamondSearchSeeded(src, ref, stride, width, height, px, py, 64, 0, 0, fullPelReach)
		sum += dx + dy + sad
	}
	fullPelDiamondBenchSink = sum
}

func BenchmarkSubpelRefine8x8(b *testing.B) {
	const (
		width  = 96
		height = 80
		stride = 112
		px     = 32
		py     = 24
	)
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	for i := range src {
		src[i] = uint8(i*7 + i/stride*11)
		ref[i] = uint8(i*13 + i/stride*3 + 17)
	}
	mv := motion.Vector{Row: 8, Col: -8}
	base := py*stride + px
	refBase := (py+int(mv.Row)/8)*stride + px + int(mv.Col)/8
	bestSAD := sadBlock(src, ref, base, refBase, stride, 8, 1<<30)

	var st lossyEncodeState
	b.ReportAllocs()
	sum := 0
	for b.Loop() {
		gotMV, sad := st.subpelRefine8x8(src, ref, stride, width, height, px, py, mv, bestSAD)
		sum += int(gotMV.Row) + int(gotMV.Col) + sad
	}
	fullPelDiamondBenchSink = sum
}
