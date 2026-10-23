package encoder

import (
	"math"
	"math/rand"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/motion"
)

// svtBilinearReference is variance.c SUBPIX_VAR's two passes, transliterated.
func svtBilinearReference(ref []byte, refStride, w, h, xoff, yoff int) []byte {
	fdata := make([]uint16, (h+1)*w)
	for i := 0; i < h+1; i++ {
		for j := 0; j < w; j++ {
			a := ref[i*refStride+j:]
			f := svtBilinearFilters2T[xoff]
			fdata[i*w+j] = uint16((uint32(a[0])*f[0] + uint32(a[1])*f[1] + 64) >> 7)
		}
	}
	out := make([]byte, h*w)
	for i := 0; i < h; i++ {
		for j := 0; j < w; j++ {
			f := svtBilinearFilters2T[yoff]
			out[i*w+j] = uint8((uint32(fdata[i*w+j])*f[0] + uint32(fdata[(i+1)*w+j])*f[1] + 64) >> 7)
		}
	}
	return out
}

func TestSVTBilinearPredictMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const stride = 160
	ref := make([]byte, stride*110)
	for i := range ref {
		ref[i] = byte(rng.Intn(256))
	}
	var scratch svtSubpelScratch
	sizes := [][2]int{{8, 8}, {16, 8}, {8, 16}, {16, 16}, {32, 16}, {16, 32}, {32, 32}, {64, 32}, {32, 64}, {64, 64}, {64, 16}, {16, 64}}
	for _, sz := range sizes {
		w, h := sz[0], sz[1]
		for xoff := 0; xoff < 8; xoff++ {
			for yoff := 0; yoff < 8; yoff++ {
				off := 3*stride + 5
				want := svtBilinearReference(ref[off:], stride, w, h, xoff, yoff)
				svtBilinearPredictArch(&scratch, ref[off:], stride, w, h, xoff, yoff)
				for i := range want {
					if scratch.pred[i] != want[i] {
						t.Fatalf("%dx%d off (%d,%d): sample %d = %d, want %d", w, h, xoff, yoff, i, scratch.pred[i], want[i])
					}
				}
				src := ref[40*stride+70:]
				var sse, sum int64
				for r := 0; r < h; r++ {
					for c := 0; c < w; c++ {
						d := int64(want[r*w+c]) - int64(src[r*stride+c])
						sse += d * d
						sum += d
					}
				}
				wantVar := uint32(sse - sum*sum/int64(w*h))
				if got := svtSubpelVarianceArch(&scratch, ref[off:], stride, src, stride, w, h, xoff, yoff); got != wantVar {
					t.Fatalf("svf %dx%d off (%d,%d) = %d, want %d", w, h, xoff, yoff, got, wantVar)
				}
				svtBilinearPredict(&scratch, ref[off:], stride, w, h, xoff, yoff)
				for i := range want {
					if scratch.pred[i] != want[i] {
						t.Fatalf("scalar %dx%d off (%d,%d): sample %d = %d, want %d", w, h, xoff, yoff, i, scratch.pred[i], want[i])
					}
				}
			}
		}
	}
}

// svtPrunedTreeModel transliterates mcomp.c svt_av1_find_best_sub_pixel_tree_pruned
// (with first_level_check_fast / second_level_check_fast / svt_check_better_fast)
// over scalar reference kernels, for the sub-pel search's differential test.
type svtPrunedTreeModel struct {
	src, ref         []byte
	stride, w, h     int
	width, height    int
	px, py, bw, bh   int
	refMV            motion.Vector
	epb              int64
	qp               uint64
	earlyExitTh      uint64
	itersPerStep     int
	ctrls            svtSubpelCtrls
	besterr, distort uint64
	best             motion.Vector
}

func (m *svtPrunedTreeModel) refAt(x, y int) byte {
	x = min(max(x, 0), m.width-1)
	y = min(max(y, 0), m.height-1)
	return m.ref[y*m.stride+x]
}

func (m *svtPrunedTreeModel) variance(pred []byte) uint64 {
	var sse, sum int64
	for r := 0; r < m.bh; r++ {
		for c := 0; c < m.bw; c++ {
			d := int64(pred[r*m.bw+c]) - int64(m.src[(m.py+r)*m.stride+m.px+c])
			sse += d * d
			sum += d
		}
	}
	return uint64(sse - sum*sum/int64(m.bw*m.bh))
}

func (m *svtPrunedTreeModel) svf(mv motion.Vector) uint64 {
	x, y := m.px+int(mv.Col>>3), m.py+int(mv.Row>>3)
	win := make([]byte, (m.bw+1)*(m.bh+1))
	for r := 0; r <= m.bh; r++ {
		for c := 0; c <= m.bw; c++ {
			win[r*(m.bw+1)+c] = m.refAt(x+c, y+r)
		}
	}
	return m.variance(svtBilinearReference(win, m.bw+1, m.bw, m.bh, int(mv.Col&7), int(mv.Row&7)))
}

func (m *svtPrunedTreeModel) mvCost(mv motion.Vector) uint64 {
	d := absI(int(mv.Row)-int(m.refMV.Row)) + absI(int(mv.Col)-int(m.refMV.Col))
	return uint64((int64(d<<8)*m.epb + 8192) >> 14)
}

func absI(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func (m *svtPrunedTreeModel) checkBetterFast(this motion.Vector) uint64 {
	if !svtSubpelLimits(m.px, m.py, m.bw, m.bh, m.width, m.height, m.refMV).contains(this) {
		return 1<<31 - 1
	}
	cost := m.mvCost(this)
	if bestcost := m.distort + cost; bestcost > m.besterr*m.earlyExitTh/1000 {
		return bestcost
	}
	thismse := m.svf(this)
	cost += thismse
	weight := uint64(100)
	if m.ctrls.biasFP != 0 && m.best.Col%8 == 0 && m.best.Row%8 == 0 {
		weight = m.ctrls.biasFP
	}
	if cost*weight/100 < m.besterr {
		m.besterr, m.best, m.distort = cost, this, thismse
	}
	return cost
}

func (m *svtPrunedTreeModel) firstLevelCheckFast(this motion.Vector, hstep int16, orgerr uint64) motion.Vector {
	left := m.checkBetterFast(motion.Vector{Row: this.Row, Col: this.Col - hstep})
	right := m.checkBetterFast(motion.Vector{Row: this.Row, Col: this.Col + hstep})
	up := m.checkBetterFast(motion.Vector{Row: this.Row - hstep, Col: this.Col})
	down := m.checkBetterFast(motion.Vector{Row: this.Row + hstep, Col: this.Col})
	diag := motion.Vector{Row: hstep, Col: hstep}
	if left <= right {
		diag.Col = -hstep
	}
	if up <= down {
		diag.Row = -hstep
	}
	if m.besterr >= orgerr {
		return diag
	}
	m.checkBetterFast(motion.Vector{Row: this.Row + diag.Row, Col: this.Col + diag.Col})
	return diag
}

func (m *svtPrunedTreeModel) secondLevelCheckFast(this, diag motion.Vector, hstep int16) {
	tr, tc, br, bc := this.Row, this.Col, m.best.Row, m.best.Col
	switch {
	case tr != br && tc != bc:
		m.checkBetterFast(motion.Vector{Row: br, Col: bc + diag.Col})
		m.checkBetterFast(motion.Vector{Row: br + diag.Row, Col: bc})
	case tr == br && tc != bc:
		m.checkBetterFast(motion.Vector{Row: br + hstep, Col: bc + diag.Col})
		m.checkBetterFast(motion.Vector{Row: br - hstep, Col: bc + diag.Col})
		m.checkBetterFast(motion.Vector{Row: br - diag.Row, Col: bc})
	case tr != br && tc == bc:
		m.checkBetterFast(motion.Vector{Row: br + diag.Row, Col: bc + hstep})
		m.checkBetterFast(motion.Vector{Row: br + diag.Row, Col: bc - hstep})
		m.checkBetterFast(motion.Vector{Row: br, Col: bc - diag.Col})
	}
}

func (m *svtPrunedTreeModel) run(start motion.Vector) motion.Vector {
	m.best = start
	x, y := m.px+int(start.Col>>3), m.py+int(start.Row>>3)
	pred := make([]byte, m.bw*m.bh)
	for r := 0; r < m.bh; r++ {
		for c := 0; c < m.bw; c++ {
			pred[r*m.bw+c] = m.refAt(x+c, y+r)
		}
	}
	m.distort = m.variance(pred)
	m.besterr = m.distort + m.mvCost(start)
	if m.qp*m.besterr < uint64((m.bw*m.bh)<<5)*m.ctrls.absThMult {
		return m.best
	}
	if m.ctrls.predVarianceTh != 0 {
		var sse, sum int64
		for _, v := range pred {
			sse += int64(v) * int64(v)
			sum += int64(v)
		}
		n := int64(m.bw * m.bh)
		v := uint64(sse - sum*sum/n)
		if (v+uint64(n/2))/uint64(n) < uint64(m.ctrls.predVarianceTh) {
			return m.best
		}
	}
	var orgerr uint64
	if m.ctrls.skipDiag >= 4 {
		orgerr = 0
	} else {
		demo := uint64(1)
		if m.ctrls.skipDiag >= 2 && (m.bw >= 64 || m.bh >= 64) {
			demo = 2
		}
		orgerr = 1<<31 - 1
		if m.ctrls.skipDiag != 0 {
			orgerr = m.besterr / demo
		}
	}
	hstep := int16(4)
	for iter := 0; iter < m.ctrls.rounds; iter++ {
		prev := m.besterr
		diag := m.firstLevelCheckFast(start, hstep, orgerr)
		if m.besterr < orgerr && m.itersPerStep > 1 {
			m.secondLevelCheckFast(start, diag, hstep)
		}
		hstep >>= 1
		start = m.best
		if m.ctrls.skipDiag != 0 && iter < 1 {
			orgerr = min(orgerr, m.besterr)
		}
		dev := (int64(max(m.besterr, 1)) - int64(max(prev, 1))) * 100 / int64(max(prev, 1))
		if dev >= m.ctrls.roundDevTh {
			return m.best
		}
	}
	return m.best
}

func TestSVTSubpelSearchMatchesPrunedTreeModel(t *testing.T) {
	const width, height, stride = 176, 144, 192
	rng := rand.New(rand.NewSource(7))
	src := make([]byte, stride*height)
	ref := make([]byte, stride*height)
	level6 := svtSubpelCtrls{rounds: 2, roundDevTh: 1<<31 - 1, skipDiag: 3, biasFP: 110}
	sizes := [][2]int{{8, 8}, {16, 16}, {32, 32}, {64, 64}, {16, 8}, {8, 16}, {32, 16}, {16, 32}, {64, 16}, {16, 64}}
	for trial := 0; trial < 300; trial++ {
		// Smooth content with a sub-pel shift so the tree actually moves.
		fx, fy := rng.Float64()*0.3, rng.Float64()*0.3
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				v := 128 + 60*math.Sin(float64(x)*fx+float64(y)*fy) + float64(rng.Intn(9)-4)
				src[y*stride+x] = byte(min(max(v, 0), 255))
				w := 128 + 60*math.Sin((float64(x)+0.4)*fx+(float64(y)-0.3)*fy) + float64(rng.Intn(9)-4)
				ref[y*stride+x] = byte(min(max(w, 0), 255))
			}
		}
		sz := sizes[trial%len(sizes)]
		bw, bh := sz[0], sz[1]
		px := (rng.Intn((width-bw)/8 + 1)) * 8
		py := (rng.Intn((height-bh)/8 + 1)) * 8
		start := motion.Vector{Row: int16(rng.Intn(5)-2) * 8, Col: int16(rng.Intn(5)-2) * 8}
		refMV := motion.Vector{Row: int16(rng.Intn(33) - 16), Col: int16(rng.Intn(33) - 16)}
		ctrls := svtSubpelLevel8
		if trial&1 == 1 {
			ctrls = level6
		}
		var st lossyEncodeState
		st.qIndex = uint8(60 + rng.Intn(180))
		st.rdMult = int64(2000 + rng.Intn(60000))
		got, _ := st.svtSubpelSearch(src, ref, stride, width, height, px, py, bw, bh, start, refMV, &ctrls, st.rdMult)
		sq := max(bw, bh)
		m := svtPrunedTreeModel{
			src: src, ref: ref, stride: stride, width: width, height: height,
			px: px, py: py, bw: bw, bh: bh, refMV: refMV,
			epb: max(st.rdMult>>6, 1), qp: uint64(st.qIndex) >> 2,
			earlyExitTh: uint64(1020 - (sq >> 2)), itersPerStep: 1, ctrls: ctrls,
		}
		if want := m.run(start); got != want {
			t.Fatalf("trial %d %dx%d at (%d,%d) start %+v: got %+v, want %+v", trial, bw, bh, px, py, start, got, want)
		}
	}
}
