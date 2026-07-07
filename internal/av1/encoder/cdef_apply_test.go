package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/threading"
	"github.com/thesyncim/goav1/internal/av1/tile"
)

func cdefDiffFillPlane(pix []byte, stride, w, h, salt int) {
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := (x*37 + y*53 + salt*91 + (x>>3)*29 + (y>>3)*17) & 255
			pix[y*stride+x] = byte(v)
		}
	}
}

func cdefDiffFrame(w, h int) SourceFrame420 {
	cw, ch := w/2, h/2
	f := SourceFrame420{
		Y: make([]byte, w*h), U: make([]byte, cw*ch), V: make([]byte, cw*ch),
		YStride: w, ChromaStride: cw, Width: w, Height: h,
	}
	cdefDiffFillPlane(f.Y, f.YStride, w, h, 0)
	cdefDiffFillPlane(f.U, f.ChromaStride, cw, ch, 1)
	cdefDiffFillPlane(f.V, f.ChromaStride, cw, ch, 2)
	return f
}

func cdefDiffCopyInto(dst *SourceFrame420, src SourceFrame420) {
	copy(dst.Y, src.Y)
	copy(dst.U, src.U)
	copy(dst.V, src.V)
}

func cdefDiffMarkSkippedBlocks(m threading.FrameWorkLoopFilterMap) {
	stride := int(m.Stride)
	rows := int(m.Rows)
	for miRow := 2; miRow+1 < rows; miRow += 8 {
		for miCol := 2; miCol+1 < stride; miCol += 10 {
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					m.Records[(miRow+dy)*stride+miCol+dx].SkipTransform = true
				}
			}
		}
	}
}

func cdefDiffApplySerialSnapshot(t *testing.T, a *cdefApplier, recon *SourceFrame420, params parser.CDEFParams, lfMap *threading.FrameWorkLoopFilterMap) {
	t.Helper()
	active, err := a.bindApplyContext(recon, params, lfMap, true)
	if err != nil {
		t.Fatalf("bind snapshot: %v", err)
	}
	if !active {
		t.Fatal("snapshot route inactive")
	}
	req := a.jobReq
	req.InputScratch = a.bandIn[0]
	req.UnitDstScratch = a.bandUnit[0]
	if _, err := a.jobCtx.ApplyCDEFPostFilterUnitRows(req, 0, a.unitRows); err != nil {
		t.Fatalf("snapshot route: %v", err)
	}
}

func cdefDiffAssertEqual(t *testing.T, snapshot, inPlace SourceFrame420) {
	t.Helper()
	for _, p := range []struct {
		name         string
		a, b         []byte
		stride, w, h int
	}{
		{"Y", snapshot.Y, inPlace.Y, snapshot.YStride, snapshot.Width, snapshot.Height},
		{"U", snapshot.U, inPlace.U, snapshot.ChromaStride, snapshot.Width / 2, snapshot.Height / 2},
		{"V", snapshot.V, inPlace.V, snapshot.ChromaStride, snapshot.Width / 2, snapshot.Height / 2},
	} {
		for y := 0; y < p.h; y++ {
			for x := 0; x < p.w; x++ {
				i := y*p.stride + x
				if p.a[i] != p.b[i] {
					t.Fatalf("plane %s differs at x=%d y=%d: snapshot=%d inplace=%d", p.name, x, y, p.a[i], p.b[i])
				}
			}
		}
	}
}

func TestCDEFApplySerialU8InPlaceMatchesSnapshot(t *testing.T) {
	const width = 160
	const height = 96
	params := parser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{47},
	}

	var lf loopFilterApplier
	if err := lf.init(width, height); err != nil {
		t.Fatalf("loop filter init: %v", err)
	}
	defer lf.close()
	lfDiffFillGrid(lf.filtMap, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true})
	cdefDiffMarkSkippedBlocks(lf.filtMap)

	src := cdefDiffFrame(width, height)
	snapshot := lfDiffCopy(src)
	inPlace := lfDiffCopy(src)

	var snapshotApply cdefApplier
	if err := snapshotApply.init(width, height, params); err != nil {
		t.Fatalf("snapshot cdef init: %v", err)
	}
	defer snapshotApply.close()
	cdefDiffApplySerialSnapshot(t, &snapshotApply, &snapshot, params, &lf.filtMap)

	var inPlaceApply cdefApplier
	if err := inPlaceApply.init(width, height, params); err != nil {
		t.Fatalf("in-place cdef init: %v", err)
	}
	defer inPlaceApply.close()
	if err := inPlaceApply.applySerial(&inPlace, params, &lf.filtMap); err != nil {
		t.Fatalf("in-place route: %v", err)
	}

	changed := false
	for i := range src.Y {
		if inPlace.Y[i] != src.Y[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("degenerate: CDEF changed no luma pixels")
	}
	cdefDiffAssertEqual(t, snapshot, inPlace)
}

func TestCDEFApplyBandedU8InPlaceMatchesSnapshot(t *testing.T) {
	const width = 192
	const height = 640
	params := parser.CDEFParams{
		Damping:       5,
		StrengthCount: 2,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63, 31},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{47, 19},
	}

	var lf loopFilterApplier
	if err := lf.init(width, height); err != nil {
		t.Fatalf("loop filter init: %v", err)
	}
	defer lf.close()
	lfDiffFillGrid(lf.filtMap, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true})
	cdefDiffMarkSkippedBlocks(lf.filtMap)

	src := cdefDiffFrame(width, height)
	snapshot := lfDiffCopy(src)
	inPlace := lfDiffCopy(src)

	var snapshotApply cdefApplier
	if err := snapshotApply.init(width, height, params); err != nil {
		t.Fatalf("snapshot cdef init: %v", err)
	}
	defer snapshotApply.close()
	cdefDiffApplySerialSnapshot(t, &snapshotApply, &snapshot, params, &lf.filtMap)

	var inPlaceApply cdefApplier
	if err := inPlaceApply.init(width, height, params); err != nil {
		t.Fatalf("in-place cdef init: %v", err)
	}
	defer inPlaceApply.close()
	if err := inPlaceApply.apply(&inPlace, params, &lf.filtMap); err != nil {
		t.Fatalf("banded in-place route: %v", err)
	}

	cdefDiffAssertEqual(t, snapshot, inPlace)
}

func TestCDEFApplySerialIsZeroAlloc(t *testing.T) {
	const width = 128
	const height = 128
	params := parser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{47},
	}

	var lf loopFilterApplier
	if err := lf.init(width, height); err != nil {
		t.Fatalf("loop filter init: %v", err)
	}
	defer lf.close()
	lfDiffFillGrid(lf.filtMap, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true})

	var a cdefApplier
	if err := a.init(width, height, params); err != nil {
		t.Fatalf("cdef init: %v", err)
	}
	defer a.close()

	src := cdefDiffFrame(width, height)
	out := lfDiffCopy(src)
	allocs := testing.AllocsPerRun(16, func() {
		cdefDiffCopyInto(&out, src)
		if err := a.applySerial(&out, params, &lf.filtMap); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("serial CDEF apply allocated: %v allocs/op", allocs)
	}
}

func TestCDEFApplyBandedIsZeroAlloc(t *testing.T) {
	const width = 192
	const height = 512
	params := parser.CDEFParams{
		Damping:       5,
		StrengthCount: 1,
		YStrength:     [parser.MaxCDEFStrengths]uint8{63},
		UVStrength:    [parser.MaxCDEFStrengths]uint8{47},
	}

	var lf loopFilterApplier
	if err := lf.init(width, height); err != nil {
		t.Fatalf("loop filter init: %v", err)
	}
	defer lf.close()
	lfDiffFillGrid(lf.filtMap, 4, 4, tile.BlockSize16x16,
		tile.TransformTreeResult{Y: tile.TransformSize16x16, UV: tile.TransformSize8x8, HasUV: true})

	var a cdefApplier
	if err := a.init(width, height, params); err != nil {
		t.Fatalf("cdef init: %v", err)
	}
	defer a.close()

	src := cdefDiffFrame(width, height)
	out := lfDiffCopy(src)
	if err := a.apply(&out, params, &lf.filtMap); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(16, func() {
		cdefDiffCopyInto(&out, src)
		if err := a.apply(&out, params, &lf.filtMap); err != nil {
			t.Fatal(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("banded CDEF apply allocated: %v allocs/op", allocs)
	}
}
