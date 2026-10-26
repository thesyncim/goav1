package encoder

// keyframe_split.go is the keyframe tile coder's decision/write split, the
// shape the P-frame coder already uses (pframe_split.go) and SVT-AV1 runs for
// every picture: ENC-DEC over EncDecSegments (mode decision, prediction,
// transform/quantize and reconstruction; enc_dec_process.c) ahead of the
// serial per-tile ENTROPY CODING stage (entropy_coding_process.c). The
// decision pass records each leaf's mode, palette and quantized coefficients;
// the write pass replays them through the adaptive-CDF writers in coding
// order. The decision pass runs as the SB-row wavefront when the coder has
// lanes (pframe_wavefront.go), so a single-tile keyframe uses every core.
//
// Decisions must not depend on coding order for the wavefront to reproduce
// the serial walk, so the trial coefficient pricing reads tables frozen at
// frame start (SVT prices mode decision from rate tables fixed per picture,
// md_rate_estimation.c): every partition node and leaf restores the trial
// contexts its predecessor's trials adapted (restoreKeyTrial). Mode, palette
// and partition rates read the frame-start CDFs as well, since the decision
// pass runs before any symbol is written.

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/loopfilter"
	"github.com/thesyncim/goav1/internal/av1/parser"
	"github.com/thesyncim/goav1/internal/av1/prediction"
	"github.com/thesyncim/goav1/internal/av1/tile"
	"github.com/thesyncim/goav1/internal/av1/transform"
)

// keyTrialAll marks the whole trial context set dirty (the generic trial
// writer may touch any of it).
const keyTrialAll uint16 = 1 << 15

// noteKeyTrial records which trial contexts a trial count of an n x n plane
// TXB adapts, so restoreKeyTrial can put back only those.
func (st *lossyEncodeState) noteKeyTrial(plane tile.CoeffPlaneType, n int) {
	if st.keyTrialPristine == nil {
		return
	}
	t := -1
	switch n {
	case 4:
		t = 0
	case 8:
		t = 1
	case 16:
		t = 2
	case 32:
		t = 3
	}
	if t < 0 || plane > tile.CoeffPlaneUV {
		st.keyTrialDirty |= keyTrialAll
		return
	}
	st.keyTrialDirty |= 1 << (t*2 + int(plane))
}

// restoreKeyTrial returns the trial contexts the last decision unit adapted
// to their frame-start state.
func (st *lossyEncodeState) restoreKeyTrial() {
	d := st.keyTrialDirty
	p := st.keyTrialPristine
	if d == 0 || p == nil {
		return
	}
	st.keyTrialDirty = 0
	c := &st.trialCDFs
	if d&keyTrialAll != 0 {
		*c = *p
		return
	}
	for t := 0; t < 4; t++ {
		for pl := 0; pl < 2; pl++ {
			if d&(1<<(t*2+pl)) == 0 {
				continue
			}
			c.TXBSkip[t] = p.TXBSkip[t]
			c.EOBExtra[t][pl] = p.EOBExtra[t][pl]
			c.DCSign[pl] = p.DCSign[pl]
			c.CoeffBR[t][pl] = p.CoeffBR[t][pl]
			c.CoeffBase[t][pl] = p.CoeffBase[t][pl]
			c.CoeffBaseEOB[t][pl] = p.CoeffBaseEOB[t][pl]
			switch t {
			case 0:
				c.EOBFlag16[pl] = p.EOBFlag16[pl]
			case 1:
				c.EOBFlag64[pl] = p.EOBFlag64[pl]
			case 2:
				c.EOBFlag256[pl] = p.EOBFlag256[pl]
			default:
				c.EOBFlag1024[pl] = p.EOBFlag1024[pl]
			}
		}
	}
}

// keyIntraRDCost prices one keyframe merge candidate exactly: best intra
// prediction, the true transform-quantize pass for distortion, and the trial
// coefficient coder for rate. Whole-block candidates predict from the coded
// reconstruction; child estimates predict from the source (their neighbours
// are not reconstructed yet, and lossy reconstructions track the source
// closely enough for a split decision).
func (st *lossyEncodeState) keyIntraRDCost(src SourceFrame420, recon *SourceFrame420, walkReq tile.BlockWalkRequest, plane []byte, px, py, n int) int64 {
	haveTop := py > int(walkReq.MIRowStart)*4
	haveLeft := px > int(walkReq.MIColStart)*4
	st.keySelectIntraMode(src.Y, plane, src.YStride, px, py, n, haveTop, haveLeft, st.predY[:n*n])
	st.rdDcode, st.rdDskip, st.rdRcode = 0, 0, 0
	st.prepareInterTXB(src.Y, st.predY[:n*n], n, src.YStride, px, py, n, n, st.yQuant, st.lumaQ2[:n*n])
	lumaD := st.rdDcode
	cost := st.trialTXBBits(tile.CoeffPlaneY, st.lumaQ2[:n*n], n) + lumaD<<7
	// Merge candidates differ in chroma transform structure too: one large
	// block against four small ones.
	return cost + st.trialChromaCost(src, recon, &plane[0] == &recon.Y[0], px, py, n, haveTop, haveLeft)
}

// keyDecidePartition is the keyframe partition decision for one tree node.
func (st *lossyEncodeState) keyDecidePartition(src SourceFrame420, recon *SourceFrame420, walkReq tile.BlockWalkRequest, level tile.BlockLevel, miCol, miRow uint32, haveRight, haveBottom bool) (tile.Partition, error) {
	if level == tile.BlockLevel8x8 {
		return tile.PartitionNone, nil
	}
	if !videoColorIs420(st.color) {
		return tile.PartitionSplit, nil
	}
	if st.keyPD0On {
		px, py := int(miCol)*4, int(miRow)*4
		sb := st.keyPD0ForSB(src, *recon, px&^63, py&^63, int(walkReq.MIColStart)*4)
		if sb != nil {
			x, y := px&63, py&63
			var part tile.Partition
			switch level {
			case tile.BlockLevel64x64:
				part = sb.part64
			case tile.BlockLevel32x32:
				part = sb.part32[(y/32)*2+x/32]
			default:
				part = sb.part16[(y/16)*4+x/16]
			}
			// This coder codes square blocks only: the edge shapes split.
			if part != tile.PartitionNone {
				part = tile.PartitionSplit
			}
			return part, nil
		}
	}
	st.restoreKeyTrial()
	px, py := int(miCol)*4, int(miRow)*4
	if level == tile.BlockLevel32x32 && haveRight && haveBottom {
		// The 32-tier keeps the near-perfect gate: trial pricing misjudges the
		// largest transform even with chroma included.
		if px+32 > src.Width || py+32 > src.Height {
			return tile.PartitionSplit, nil
		}
		st.keySelectIntraMode(src.Y, recon.Y, src.YStride, px, py, 32, py > int(walkReq.MIRowStart)*4, px > int(walkReq.MIColStart)*4, st.predY[:32*32])
		sad := 0
		for r := range 32 {
			row := (py+r)*src.YStride + px
			for c := range 32 {
				d := int(src.Y[row+c]) - int(st.predY[r*32+c])
				if d < 0 {
					d = -d
				}
				sad += d
			}
		}
		if sad <= 32*32*2 {
			return tile.PartitionNone, nil
		}
		return tile.PartitionSplit, nil
	}
	if level == tile.BlockLevel16x16 && haveRight && haveBottom {
		if px+16 > src.Width || py+16 > src.Height {
			return tile.PartitionSplit, nil
		}
		// The priced overhead of one extra block's prefix symbols
		// (partition, mode, angle, chroma).
		headerCost := (int64(20) << 9) * st.rdMult >> 9
		child := int64(0)
		for _, o8 := range [4][2]int{{0, 0}, {8, 0}, {0, 8}, {8, 8}} {
			child += st.keyIntraRDCost(src, recon, walkReq, src.Y, px+o8[0], py+o8[1], 8) + headerCost
		}
		if st.keyIntraRDCost(src, recon, walkReq, recon.Y, px, py, 16)+headerCost <= child {
			return tile.PartitionNone, nil
		}
	}
	return tile.PartitionSplit, nil
}

func keyBlockN(size tile.BlockSize) (int, tile.TransformSize, error) {
	switch size {
	case tile.BlockSize8x8:
		return 8, tile.TransformSize8x8, nil
	case tile.BlockSize16x16:
		return 16, tile.TransformSize16x16, nil
	case tile.BlockSize32x32:
		return 32, tile.TransformSize32x32, nil
	case tile.BlockSize64x64:
		return 64, tile.TransformSize64x64, nil
	}
	return 0, 0, fmt.Errorf("encoder: unexpected keyframe block size %d", size)
}

// keyChromaGeom is a keyframe block's chroma TXB geometry.
type keyChromaGeom struct {
	block      tile.BlockSize
	tx         tile.TransformSize
	scan       []int16
	cw, ch     int
	x, y       int
	x4, y4     uint8
	coeffCount int
}

func (st *lossyEncodeState) keyChroma(block tile.BlockVisit, lumaPX, lumaPY int) (keyChromaGeom, error) {
	var g keyChromaGeom
	var err error
	if g.block, err = tile.PlaneBlockSize(block.Size, st.color, 1); err != nil {
		return g, fmt.Errorf("chroma plane block: %w", err)
	}
	if g.cw, g.ch, err = planeBlockPixels(g.block); err != nil {
		return g, fmt.Errorf("chroma block dimensions: %w", err)
	}
	if g.tx, err = tile.MaxTransformSize(block.Size, st.color, 1); err != nil {
		return g, fmt.Errorf("chroma transform size: %w", err)
	}
	var ok bool
	if g.scan, ok = st.scanForTransformSize(g.tx); !ok {
		g.scan, ok = st.scanForThinTransformSize(g.tx)
	}
	if !ok {
		return g, fmt.Errorf("encoder: unsupported chroma transform %d", g.tx)
	}
	geo, err := txBlockGeometryForTransformSize(g.tx)
	if err != nil {
		return g, err
	}
	g.coeffCount = geo.coeffCount
	g.x, g.y = chromaXForColor(lumaPX, st.color), chromaYForColor(lumaPY, st.color)
	g.x4, g.y4 = chromaX4ForColor(block.X4, st.color), chromaY4ForColor(block.Y4, st.color)
	return g, nil
}

// decideKeyBlock is the decision half of a keyframe leaf: luma mode, the
// directional refinement, the screen-content palette, transform/quantize and
// reconstruction of all three planes. It marks the mode contexts the later
// decisions read (intra modes, palettes) and records the block in rec.
func (st *lossyEncodeState) decideKeyBlock(src SourceFrame420, recon *SourceFrame420, block tile.BlockVisit, scratch *tile.BlockLoopScratch, rec *pframeSplitRecord) error {
	st.restoreKeyTrial()
	n, lumaTX, err := keyBlockN(block.Size)
	if err != nil {
		return err
	}
	modeCtx := &scratch.Mode
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}
	lumaPX := int(block.MICol) * 4
	lumaPY := int(block.MIRow) * 4
	pred := st.predY[:n*n]
	mode := st.keySelectIntraMode(src.Y, recon.Y, src.YStride, lumaPX, lumaPY, n, block.HaveTop, block.HaveLeft, pred)
	angleDelta := 0
	if st.allowScreenContentTools {
		mode, angleDelta = st.improveIntraModeDirectional(src, recon, block, modeCtx, mode, pred, lumaPX, lumaPY, n)
	}
	pal := st.keyframePalette(src, block, modeCtx, lumaPX, lumaPY, n, mode, pred)
	if pal != nil {
		mode, angleDelta = tile.IntraModeDC, 0
	}

	r := rec.appendBlock()
	r.kind = pblockIntra
	r.intraMode = mode
	r.angleDelta = int8(angleDelta)
	r.uvMode, r.cfl = tile.ChromaIntraModeDC, tile.CFLAlphaResult{}
	r.palMap = -1
	if err := modeCtx.MarkIntra(block.Size, int(block.X4), int(block.Y4), true, mode); err != nil {
		return fmt.Errorf("mark intra: %w", err)
	}
	if err := modeCtx.MarkChromaIntra(block.Size, int(block.X4), int(block.Y4), true, tile.ChromaIntraModeDC); err != nil {
		return fmt.Errorf("mark chroma intra: %w", err)
	}
	if pal != nil || st.allowScreenContentTools {
		res := tile.PaletteModeResult{}
		if pal != nil {
			res = pal.res
			res.YMap = nil
			r.pal = res
			r.palMap = int32(rec.storePalMap(pal.m[:n*n]))
		}
		if err := modeCtx.MarkPaletteY(block.Size, int(block.X4), int(block.Y4), res); err != nil {
			return fmt.Errorf("mark palette y: %w", err)
		}
		if err := modeCtx.MarkPaletteUV(block.Size, int(block.X4), int(block.Y4), tile.PaletteModeResult{}); err != nil {
			return fmt.Errorf("mark palette uv: %w", err)
		}
	}

	// SVT's MDS3 transform shortcuts (keyTXShortcut, keyChromaN4).
	shortcut := false
	if st.keyTXSOn {
		// SVT's I-slice RDOQ (keyRDOQQuantize) on every TXB of the leaf.
		st.keyRDOQ = true
		defer func() { st.keyRDOQ = false }()
		yTX := lumaTX
		if mode == tile.IntraModeDC && pal == nil {
			shortcut = keyTXShortcut(src.Y, src.YStride, pred, lumaPX, lumaPY, n, st.qIndex)
			st.pfN4 = shortcut
			yTX, err = st.keyLumaTXS(src, recon, block, scratch, rec, pred, n, lumaTX)
			st.pfN4 = false
			if err != nil {
				return fmt.Errorf("luma txs: %w", err)
			}
			r.splitTX = yTX != lumaTX
		} else if err := st.keyLumaDepth0(src, recon, block, scratch, rec, pred, n, lumaTX); err != nil {
			return fmt.Errorf("luma txb: %w", err)
		}
		blockDims, _ := block.Size.Dimensions()
		txDims, _ := yTX.Dimensions()
		if err := modeCtx.MarkTransformArea(int(block.X4), int(block.Y4), int(blockDims.W4), int(blockDims.H4), txDims.Log2W, txDims.Log2H, true); err != nil {
			return fmt.Errorf("mark transform: %w", err)
		}
	} else {
		geo, _, err := st.prepareIntraTXB(src.Y, src.YStride, lumaPX, lumaPY, n, n, st.yQuant, lumaTX, pred, st.lumaQ[:])
		if err != nil {
			return fmt.Errorf("luma txb: %w", err)
		}
		if err := st.reconKeyTXB(rec, recon.Y, pred, src.YStride, lumaPX, lumaPY, n, n, 0, lumaTX, geo); err != nil {
			return fmt.Errorf("luma txb: %w", err)
		}
	}
	cg, err := st.keyChroma(block, lumaPX, lumaPY)
	if err != nil {
		return err
	}
	dcU := dcPredictRect(recon.U, src.ChromaStride, cg.x, cg.y, cg.cw, cg.ch, block.HaveTop, block.HaveLeft)
	dcV := dcPredictRect(recon.V, src.ChromaStride, cg.x, cg.y, cg.cw, cg.ch, block.HaveTop, block.HaveLeft)
	if st.keyTXSOn && mode == tile.IntraModeDC && pal == nil {
		lumaPred := pred
		if r.splitTX {
			lumaPred = nil
		}
		st.pfN4 = st.keyChromaN4(src, &cg, lumaPred, n, lumaPX, lumaPY, dcU, dcV, shortcut)
		defer func() { st.pfN4 = false }()
	}
	if st.keyTXSOn {
		blockDims, _ := cg.block.Dimensions()
		txDims, _ := cg.tx.Dimensions()
		for plane := 1; plane <= 2; plane++ {
			st.keyChromaCtx[plane] = scratch.CoeffCtx.TXBContextTrusted(tile.CoeffContextRequest{
				Plane: uint8(plane), PlaneBlock: cg.block, Size: cg.tx, X4: cg.x4, Y4: cg.y4,
			}, txDims, blockDims)
		}
		st.keyRDOQPlane = 1
	}
	// SVT's I-slice CfL search (keyframe_cfl.go, cfl_level 2).
	var cflAlphaQ3 [3]int
	if cflAllowed, err := tile.ChromaIntraCFLAllowed(block.Size, st.color, false); err == nil && cflAllowed && videoColorIs420(st.color) {
		if use, alpha := st.keyCFLDecide(src, recon, &cg, mode, lumaPX, lumaPY, n, dcU, dcV); use {
			r.uvMode, r.cfl = tile.ChromaIntraModeCFL, alpha
			for plane := 1; plane <= 2; plane++ {
				a, err := prediction.CFLAlphaQ3(alpha.Index, alpha.JointSign, prediction.CFLPredType(plane-1))
				if err != nil {
					return fmt.Errorf("cfl alpha: %w", err)
				}
				cflAlphaQ3[plane] = a
			}
			if err := modeCtx.MarkChromaIntra(block.Size, int(block.X4), int(block.Y4), true, tile.ChromaIntraModeCFL); err != nil {
				return fmt.Errorf("mark chroma intra: %w", err)
			}
		}
	}
	for plane := 1; plane <= 2; plane++ {
		data, rdata := src.U, recon.U
		if plane == 2 {
			data, rdata = src.V, recon.V
		}
		q := st.uQuant
		if plane == 2 {
			q = st.vQuant
		}
		cpred := st.predU[:cg.cw*cg.ch]
		dc := dcU
		if plane == 2 {
			dc = dcV
		}
		if r.uvMode == tile.ChromaIntraModeCFL {
			if !st.keyCFLPredict(cpred, &cg, dc, cflAlphaQ3[plane]) {
				return fmt.Errorf("encoder: cfl prediction failed")
			}
		} else {
			memsetBytes(cpred, dc)
		}
		st.keyRDOQCtx = st.keyChromaCtx[plane]
		cgeo, _, err := st.prepareIntraTXB(data, src.ChromaStride, cg.x, cg.y, cg.cw, cg.ch, q, cg.tx, cpred, st.lumaQ[:])
		if err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
		if err := st.reconKeyTXB(rec, rdata, cpred, src.ChromaStride, cg.x, cg.y, cg.cw, cg.ch, plane, cg.tx, cgeo); err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
		if st.keyTXSOn {
			// The decision pass's chroma contexts (the next leaves' RDOQ).
			res := interTXBResult(st.lumaQ[:cgeo.coeffCount], cg.scan, cgeo.coeffCount)
			if !res.AllZero {
				res.CulLevel = tile.CoeffCulLevel(st.lumaQ[:cgeo.coeffCount], cg.scan, int(res.EOB))
			}
			if err := scratch.CoeffCtx.MarkTXB(tile.CoeffContextRequest{
				Plane: uint8(plane), PlaneBlock: cg.block, Size: cg.tx, X4: cg.x4, Y4: cg.y4,
			}, res); err != nil {
				return fmt.Errorf("chroma %d context: %w", plane, err)
			}
		}
	}
	return nil
}

// reconKeyTXB stores the prepared coefficients (st.lumaQ) in rec and
// reconstructs the TXB through the decoder-exact path, driven by the
// coefficients' own end of block as the fused writer's result would be.
func (st *lossyEncodeState) reconKeyTXB(rec *pframeSplitRecord, reconPlane, pred []byte, stride, px, py, w, h, plane int, size tile.TransformSize, geo txBlockGeometry) error {
	q := st.yQuant
	switch plane {
	case 1:
		q = st.uQuant
	case 2:
		q = st.vQuant
	}
	scan, ok := st.scanForTransformSize(size)
	if !ok {
		scan, ok = st.scanForThinTransformSize(size)
	}
	if !ok {
		return fmt.Errorf("encoder: unsupported transform %d", size)
	}
	cn := geo.coeffCount
	qcoeff := st.lumaQ[:cn]
	copy(rec.grabCoeffs(cn), qcoeff)
	return st.reconInterTXBResult(reconPlane, pred, w, stride, px, py, w, h, q, qcoeff, geo, scan, st.intraUVTxType, interTXBResult(qcoeff, scan, cn))
}

// writeKeyBlock is the entropy half of a keyframe leaf: it replays the
// recorded decision through the writers and marks in the fused coder's
// symbol order (skip, luma mode, angle, chroma mode, palette_mode_info,
// palette_tokens, then the luma and chroma TXBs).
func (st *lossyEncodeState) writeKeyBlock(block tile.BlockVisit, scratch *tile.BlockLoopScratch, rec *pframeSplitRecord) error {
	r, err := rec.nextBlock()
	if err != nil {
		return err
	}
	n, lumaTX, err := keyBlockN(block.Size)
	if err != nil {
		return err
	}
	modeCtx := &scratch.Mode
	coeffCtx := &scratch.CoeffCtx
	// A split luma codes four quarter-size transforms (keyframe_txs.go).
	yTX := lumaTX
	if r.splitTX {
		yTX = txSizeSquare(n / 2)
	}
	if st.lfMap != nil {
		uvTX, err := tile.MaxTransformSize(block.Size, st.color, 1)
		if err != nil {
			return err
		}
		if err := markLoopFilterBlock(st.lfMap, block, tile.TransformTreeResult{Y: yTX, UV: uvTX, HasUV: true},
			false, true, 0, loopfilter.ModeDeltaClassZero); err != nil {
			return fmt.Errorf("mark loop filter: %w", err)
		}
	}
	prefixReq := tile.BlockModeRequest{Size: block.Size, X4: block.X4, Y4: block.Y4}
	if err := tile.WriteSkipTransform(st.w, &st.modeCDFs, modeCtx, prefixReq, false, false); err != nil {
		return fmt.Errorf("skip: %w", err)
	}
	if err := modeCtx.Mark(block.Size, int(block.X4), int(block.Y4), tile.BlockModeResult{}); err != nil {
		return fmt.Errorf("mark prefix: %w", err)
	}
	mode := r.intraMode
	if err := tile.WriteLumaIntraMode(st.w, &st.intraCDFs, modeCtx, tile.LumaIntraModeRequest{
		Size: block.Size, X4: block.X4, Y4: block.Y4,
	}, mode); err != nil {
		return fmt.Errorf("luma mode: %w", err)
	}
	if err := modeCtx.MarkIntra(block.Size, int(block.X4), int(block.Y4), true, mode); err != nil {
		return fmt.Errorf("mark intra: %w", err)
	}
	if err := tile.WriteIntraAngleDelta(st.w, &st.intraCDFs, tile.IntraAngleDeltaRequest{
		Size: block.Size, Mode: mode,
	}, r.angleDelta); err != nil {
		return fmt.Errorf("angle delta: %w", err)
	}
	cflAllowed, err := tile.ChromaIntraCFLAllowed(block.Size, st.color, false)
	if err != nil {
		return fmt.Errorf("cfl allowed: %w", err)
	}
	if err := tile.WriteChromaIntraMode(st.w, &st.intraCDFs, tile.ChromaIntraModeRequest{
		Size: block.Size, LumaMode: mode, CFLAllowed: cflAllowed,
	}, r.uvMode, r.cfl); err != nil {
		return fmt.Errorf("chroma mode: %w", err)
	}
	if err := modeCtx.MarkChromaIntra(block.Size, int(block.X4), int(block.Y4), true, r.uvMode); err != nil {
		return fmt.Errorf("mark chroma intra: %w", err)
	}
	var pal *paletteCand
	if r.palMap >= 0 {
		if st.pal == nil {
			st.pal = new(paletteSearch)
		}
		pal = &st.pal.cands[0]
		pal.res = r.pal
		copy(pal.m[:n*n], rec.palMap(int(r.palMap), n*n))
		pal.res.YMap = &pal.m
		if err := st.writeKeyframePalette(modeCtx, block, mode, r.uvMode, pal); err != nil {
			return fmt.Errorf("palette: %w", err)
		}
	} else if err := st.writeNoPaletteMode(modeCtx, block, mode, r.uvMode, true); err != nil {
		return err
	}
	if pal != nil {
		req := st.paletteRequest(block, mode)
		req.ChromaMode = r.uvMode
		if err := tile.WritePaletteTokens(st.w, &st.intraCDFs, req, &pal.res); err != nil {
			return fmt.Errorf("palette tokens: %w", err)
		}
	}

	if st.keyTXSOn {
		if _, err := tile.WriteTransformTree(st.w, &st.treeCDFs, modeCtx, tile.TransformTreeRequest{
			Size: block.Size, X4: block.X4, Y4: block.Y4,
			VisibleW4: block.VisibleW4, VisibleH4: block.VisibleH4,
			HaveTop: block.HaveTop, HaveLeft: block.HaveLeft,
			Color: st.color, TransformMode: parser.TransformModeSwitchable,
		}, tile.TransformTreeResult{Y: yTX}); err != nil {
			return fmt.Errorf("transform size: %w", err)
		}
	}

	st.intraTxTypeReq = tile.IntraTransformTypeRequest{Size: yTX, Mode: mode, QIndexKnown: true, QIndex: st.qIndex}
	if st.afterSkipIntra == nil {
		st.afterSkipIntra = func() error {
			return tile.WriteIntraTransformType(st.w, &st.txCDFs, st.intraTxTypeReq, transform.TypeDCTDCT)
		}
	}
	lumaScan, ok := st.scanForTransformSize(yTX)
	if !ok {
		return fmt.Errorf("encoder: unsupported luma transform %d", yTX)
	}
	lumaGeo, err := txBlockGeometryForTransformSize(yTX)
	if err != nil {
		return err
	}
	tn, step := n, 1
	if r.splitTX {
		tn, step = n/2, 2
	}
	for i := range step * step {
		qY, err := rec.nextCoeffs(lumaGeo.coeffCount)
		if err != nil {
			return err
		}
		dx4, dy4 := uint8((i%step)*tn/4), uint8((i/step)*tn/4)
		// The size-specialized trusted writers code the same syntax (tx_type
		// from st.intraTxTypeReq after txb_skip) as the generic one.
		st.intraLumaDCT = true
		_, err = st.writeIntraTXB(tn, tn, qY, tile.CoeffContextRequest{
			Plane: 0, PlaneBlock: block.Size, Size: yTX, X4: block.X4 + dx4, Y4: block.Y4 + dy4,
		}, coeffCtx, lumaScan, st.afterSkipIntra)
		st.intraLumaDCT = false
		if err != nil {
			return fmt.Errorf("luma txb: %w", err)
		}
	}
	cg, err := st.keyChroma(block, int(block.MICol)*4, int(block.MIRow)*4)
	if err != nil {
		return err
	}
	for plane := 1; plane <= 2; plane++ {
		qc, err := rec.nextCoeffs(cg.coeffCount)
		if err != nil {
			return err
		}
		if _, err := st.writeIntraTXB(cg.cw, cg.ch, qc, tile.CoeffContextRequest{
			Plane: uint8(plane), PlaneBlock: cg.block, Size: cg.tx, X4: cg.x4, Y4: cg.y4,
		}, coeffCtx, cg.scan, nil); err != nil {
			return fmt.Errorf("chroma %d txb: %w", plane, err)
		}
	}
	if st.decisionStats != nil {
		st.decisionStats.noteIntraBlock(block.Size)
	}
	return nil
}

// keySelectIntraMode picks a keyframe block's luma mode and fills pred with
// its prediction. SVT-AV1's rtc presets above M11
// (svt_aom_get_intra_mode_levels_rtc) run intra_level MAX_INTRA_LEVEL - 1 on
// camera content, whose intra_mode_end is DC_PRED (set_intra_ctrls): intra
// slices code DC_PRED only. Screen content (sc_class1, which is what enables
// the screen content tools on a keyframe) runs intra_level 4 and keeps the
// mode search.
func (st *lossyEncodeState) keySelectIntraMode(srcPlane, reconPlane []byte, stride, px, py, n int, haveTop, haveLeft bool, pred []byte) tile.IntraMode {
	if st.allowScreenContentTools {
		return selectIntraModeN(srcPlane, reconPlane, stride, px, py, n, haveTop, haveLeft, pred)
	}
	memsetBytes(pred[:n*n], dcPredictN(reconPlane, stride, px, py, n, haveTop, haveLeft))
	return tile.IntraModeDC
}
