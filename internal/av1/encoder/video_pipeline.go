package encoder

import (
	"fmt"

	"github.com/thesyncim/goav1/internal/av1/parser"
)

// video_pipeline.go is the opt-in throughput-pipelining surface. The default
// low-delay realtime path (Encode/EncodeWithTemporalID) is untouched and stays
// byte- and latency-identical to the historical serial encoder.
//
// The lever (PERF_PLAN §E1): under the L1T2 coding order base0, leaf1, base2,
// leaf3, … a droppable LEAF frame (temporalID>0, refresh_frame_flags=0, never
// updates the base symbol context) and the NEXT BASE frame both read ONLY the
// prior base frame's reconstructed reference and adapted CDFs — they are
// mutually independent, so encode(leaf N) can run concurrently with
// encode(base N+1) with BYTE-IDENTICAL output. The only cost is +1 frame of
// latency: base N+1's source is accepted while leaf N is still coding, so
// results emerge one EncodeThroughput() call later.
//
// Concurrency model. The buffered leaf runs on a dedicated coder/HME/filter set
// (leafPC/leafHME/leafLF/leafCdefApp, isolated scratch) on a persistent parked
// worker, while the following base runs its normal serial encodePReusing on the
// caller goroutine. The base path is UNCHANGED — the overlap only reorders work
// in wall-clock time, never the logical encode order or the shared-state chain.
//
// Byte-safe scope. The base's golden-refresh decision reads avgFrameLowMotion,
// which the leaf updates; overlapping the two without serializing that read
// would diverge. So the concurrent overlap is enabled ONLY when golden
// references are disabled (goldenEvery==0), where the base never reads
// leaf-updated state. Every other configuration falls back to the strict serial
// FIFO below: byte-identical, +1 latency, no concurrency. The remaining
// couplings the overlap must respect (HME reference-pyramid handoff, a separate
// CDF snapshot, the disjoint recon double-buffer, and per-layer rate control)
// are handled in encodeOverlap. The byte-identity oracle
// (TestVideoEncoderPipelineByteIdentical) guards both the overlap and the FIFO.

// SetThroughputPipelining toggles opt-in temporal-layer frame pipelining. The
// default is OFF: the encoder keeps the low-delay realtime contract (no added
// latency, bytes unchanged). When ON, encode through EncodeThroughput/Drain:
// results emerge one call later (+1 frame of latency) in exchange for wall
// throughput. It must not be toggled with a frame already in flight; call Drain
// first.
func (e *VideoEncoder) SetThroughputPipelining(on bool) error {
	if e == nil {
		return fmt.Errorf("encoder: nil video encoder")
	}
	if e.pipeHeld || e.pipeHeldDone {
		return fmt.Errorf("encoder: cannot change pipelining with a frame in flight; call Drain first")
	}
	e.pipeline = on
	return nil
}

// ThroughputPipelining reports whether opt-in throughput pipelining is enabled.
func (e *VideoEncoder) ThroughputPipelining() bool {
	return e != nil && e.pipeline
}

// EncodeThroughput submits one source frame in throughput-pipelined mode and
// returns the temporal unit produced by an earlier submission. produced=false
// means the pipeline is still filling and there is no output for this call. The
// returned slice aliases an encoder-owned buffer reused by the next call;
// callers that retain it must copy. Requires SetThroughputPipelining(true).
// Call Drain to flush the final buffered frame.
func (e *VideoEncoder) EncodeThroughput(src SourceFrame420, forceKey bool) (tu []byte, key bool, produced bool, err error) {
	if e == nil {
		return nil, false, false, fmt.Errorf("encoder: nil video encoder")
	}
	if !e.pipeline {
		return nil, false, false, fmt.Errorf("encoder: throughput pipelining not enabled")
	}
	if src.Width != e.renderWidth || src.Height != e.renderHeight {
		return nil, false, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.renderWidth, e.renderHeight)
	}
	// A precomputed base result (from a prior overlap) emits immediately; its
	// encoder state was already committed, so we just hand back the TU.
	if e.pipeHeldDone {
		tu = e.pipeHeldTU
		e.lastRecon = e.pipeHeldRecon
		e.pipeHeldDone = false
		e.holdSource(src, forceKey)
		return tu, false, true, nil
	}
	if !e.pipeHeld {
		e.holdSource(src, forceKey)
		return nil, false, false, nil
	}
	// A leaf is buffered and the incoming frame is the following base: overlap.
	if e.overlapEligible(forceKey) && e.pipeHeldTID > 0 && !e.pipeHeldKey {
		return e.encodeOverlap(src)
	}
	// Otherwise fall back to the strict serial FIFO: code the buffered frame and
	// buffer the new one.
	tu, key, err = e.encodeHeld()
	if err != nil {
		return nil, false, false, err
	}
	e.holdSource(src, forceKey)
	return tu, key, true, nil
}

// Drain flushes the final buffered frame from the throughput pipeline. It
// returns produced=false when the pipeline is empty. After a successful Drain
// the pipeline is empty and can accept a fresh EncodeThroughput sequence.
func (e *VideoEncoder) Drain() (tu []byte, key bool, produced bool, err error) {
	if e == nil {
		return nil, false, false, nil
	}
	if e.pipeHeldDone {
		tu = e.pipeHeldTU
		e.lastRecon = e.pipeHeldRecon
		e.pipeHeldDone = false
		return tu, false, true, nil
	}
	if !e.pipeHeld {
		return nil, false, false, nil
	}
	tu, key, err = e.encodeHeld()
	if err != nil {
		return nil, false, false, err
	}
	return tu, key, true, nil
}

// overlapEligible reports whether the buffered leaf and the incoming base can
// run concurrently with byte-identical output. See the file header for why the
// scope is restricted to golden-disabled L1T2 single-tile 8-bit 4:2:0 streams
// with scene-cut promotion off and no render padding.
func (e *VideoEncoder) overlapEligible(forceKey bool) bool {
	if forceKey || !e.haveKey || !e.pipeline {
		return false
	}
	if e.temporalLayers != 2 || e.goldenEvery != 0 || e.sceneCutKeyframes {
		return false
	}
	if e.tileColsLog2 != 0 {
		return false
	}
	if e.renderWidth != e.width || e.renderHeight != e.height {
		return false
	}
	if e.color.BitDepth != 8 || !videoColorIs420(e.parserColorConfig()) {
		return false
	}
	return true
}

// encodeOverlap codes the buffered leaf concurrently with the incoming base:
// the leaf on the dedicated worker, the base on the caller goroutine. It emits
// the leaf now and buffers the base's finished result for the next call.
// Byte-identity relies on the base path being unchanged and the leaf reading
// only snapshots taken before the base mutates shared state.
func (e *VideoEncoder) encodeOverlap(baseSrc SourceFrame420) (tu []byte, key bool, produced bool, err error) {
	leafSrc := e.pipeHeldSrc

	// Join the previous base's backgrounded in-loop filter before the leaf reads
	// the reference reconstruction: the leaf's motion search reads e.recon, and
	// the prior frame's filter may still be writing those planes. The serial
	// path makes this join at the top of every encodePReusing; the overlap must
	// make it before launching the concurrent leaf.
	if err := e.joinFilter(); err != nil {
		return nil, false, false, err
	}

	// HME reference-pyramid handoff (reproduces the serial swap chain exactly):
	// the leaf's coarse search references the previous base's source
	// quarter-plane (currently in e.hme.refQ); the base's references the leaf's
	// source quarter-plane. Both searches are cheap and run serially before the
	// heavy encodes overlap.
	e.runLeafHME(leafSrc)
	e.runHME(baseSrc)

	// Snapshot the leaf's read-only inputs before the base's encodePReusing
	// mutates them: the base reference reconstruction (a stable double-buffer
	// slot the base will not overwrite) and a copy of the base symbol context.
	refRecon := e.recon
	var prevCtx *frameCDFs
	if e.haveCtx {
		e.pipeLeafCtx = e.frameCtx
		prevCtx = &e.pipeLeafCtx
	}

	e.startLeafWorker()
	e.leafJob.src = leafSrc
	e.leafJob.refRecon = refRecon
	e.leafJob.prevCtx = prevCtx
	e.leafWork <- struct{}{}

	baseTU, berr := e.encodePReusing(baseSrc, 0)

	lerr := <-e.leafDone
	if berr != nil {
		return nil, false, false, berr
	}
	if lerr != nil {
		return nil, false, false, lerr
	}
	leafTU := e.leafJob.tu

	// Serial commit in coding order (leaf, then base). The base already
	// committed its own e.* state inside encodePReusing; here we apply the
	// leaf's deferred bookkeeping and advance the stream. Rate control is
	// per-layer (leaf touches layer 1, base layer 0) so the order is immaterial
	// to bytes; avgFrameLowMotion is unused with golden disabled.
	if e.decisionStatsEnabled {
		e.decisionStats.Frames++
		e.decisionStats.InterFrames++
		e.decisionStats.add(e.leafPC.decisionStats)
	}
	e.frameIndex += 2
	e.lastTemporalID = 0
	e.rcUpdate(len(leafTU)*8, 1)
	e.rcUpdate(len(baseTU)*8, 0)
	e.pipeOverlaps++

	// Emit the leaf now; buffer the base's finished result for the next call.
	e.lastRecon = e.pipeLeafRecon
	e.pipeHeldRecon = e.recon
	e.pipeHeldTU = baseTU
	e.pipeHeldDone = true
	e.pipeHeld = false
	return leafTU, false, true, nil
}

// runLeafHME runs the buffered leaf's hierarchical motion search on the isolated
// leafHME, seeded from the previous base's source quarter-plane (e.hme.refQ),
// then hands the leaf's own source quarter-plane back into e.hme.refQ so the
// base search that follows references it — reproducing the serial swap chain.
func (e *VideoEncoder) runLeafHME(leafSrc SourceFrame420) {
	if !e.hme.armed || e.hme.qw <= 0 || e.hme.qh <= 0 {
		// No primed reference (should not happen post-keyframe); the leaf coder
		// falls back to a seedless search exactly as a serial leaf would.
		e.leafHME.armed = false
		return
	}
	n := e.hme.qw * e.hme.qh
	if len(e.leafHME.srcQ) < n {
		e.leafHME.srcQ = make([]byte, n)
		e.leafHME.refQ = make([]byte, n)
	}
	e.leafHME.qw, e.leafHME.qh = e.hme.qw, e.hme.qh
	copy(e.leafHME.refQ[:n], e.hme.refQ[:n])
	e.leafHME.armed = true
	e.leafHME.runSerial(leafSrc)
	// e.leafHME.refQ now holds the leaf's source quarter-plane; feed it to the
	// base search's reference slot.
	copy(e.hme.refQ[:n], e.leafHME.refQ[:n])
	e.hme.armed = true
}

// startLeafWorker launches the persistent leaf-encode worker once. It parks on
// leafWork between frames and reads its inputs from leafJob, mirroring the tile
// and filter worker pattern so the steady state spawns no goroutines.
func (e *VideoEncoder) startLeafWorker() {
	if e.leafStarted {
		return
	}
	e.leafWork = make(chan struct{}, 1)
	e.leafDone = make(chan error, 1)
	go func() {
		for range e.leafWork {
			j := &e.leafJob
			tu, cnt, err := e.computeLeafTU(j.src, j.refRecon, j.prevCtx)
			j.tu = tu
			j.cntZeroMV = cnt
			e.leafDone <- err
		}
	}()
	e.leafStarted = true
}

// computeLeafTU codes one droppable L1T2 leaf entirely on the isolated leaf
// state and returns its temporal unit and near-zero-motion count. It mirrors
// the droppable single-tile branch of encodePReusing but never touches shared
// encoder state, so it is safe to run concurrently with the base frame's
// encodePReusing. Restricted to the overlap-eligible case (golden off,
// single-tile, 8-bit 4:2:0, no padding).
func (e *VideoEncoder) computeLeafTU(src SourceFrame420, refRecon SourceFrame420, prevCtx *frameCDFs) ([]byte, int, error) {
	color := e.parserColorConfig()
	effQ := e.layerQIndex(1)
	seq := e.sequenceHeader(src.Width, src.Height)
	header, refState := repeatPFrameHeader(src.Width, src.Height, effQ, 0)
	e.leafRefState = refState
	header.References = &e.leafRefState
	e.applyContentHintToFrameHeader(&header.Prefix)

	lfLevel := uint8(0)
	filterSupported := videoColorSupportsInLoopFilters(color)
	if filterSupported && !e.fastestRealtimeEffort() && src.Width*src.Height <= loopFilterMaxArea {
		lfLevel = filterLevelFromQIndex(effQ, false)
	}
	if !filterSupported {
		header.CDEF = CDEFParams{}
	}
	if lfLevel > 0 {
		if !e.leafLF.bound {
			if err := e.leafLF.init(src.Width, src.Height); err != nil {
				return nil, 0, fmt.Errorf("leaf loop filter init: %w", err)
			}
		}
		if err := e.leafLF.reset(); err != nil {
			return nil, 0, fmt.Errorf("leaf loop filter reset: %w", err)
		}
		header.LoopFilter.LevelY = [2]uint8{lfLevel, lfLevel}
		header.LoopFilter.LevelU = lfLevel
		header.LoopFilter.LevelV = lfLevel
		header.CDEF = cdefHeaderParams(effQ, false)
		if !e.leafCdefApp.bound {
			if err := e.leafCdefApp.init(src.Width, src.Height, cdefParserParams(header.CDEF)); err != nil {
				return nil, 0, fmt.Errorf("leaf cdef init: %w", err)
			}
		}
	}
	if prevCtx != nil {
		header.Prefix.ErrorResilientMode = false
		header.Prefix.PrimaryRefFrame = 0
		e.leafPrevLFDeltas = defaultLoopFilterDeltas()
		header.PreviousLFDeltas = &e.leafPrevLFDeltas
	}

	e.leafPC.effortLevel = e.effortLevel
	e.leafPC.decisionStatsEnabled = e.decisionStatsEnabled
	e.leafPC.st.hme = &e.leafHME
	e.leafPC.st.mds0Level = 2
	// Pipelined leaves mirror the serial droppable-leaf depth-removal config
	// (video.go encodePReusing) so pipelining stays byte-identical.
	e.leafPC.st.depthRemovalLevel = depthRemovalLevelForFrame(src.Width, src.Height, false)
	e.leafPC.st.depthRemovalIsRef = false
	// Droppable-leaf LPD1 TX shortcut controls, matching the serial leaf.
	e.leafPC.st.setLPD1TxCtrls(true, false)
	e.leafPC.st.interpSearch = false
	e.leafPC.st.interpShadow = false
	e.leafPC.st.interpDual = seq.EnableDualFilter
	e.leafPC.forceSplit = e.forceSplit
	// Match the serial leaf's decision path so the bytes agree: when inter
	// frames run the SB-row wavefront (SetMaxThreads>1), the serial leaf uses it
	// too. The leaf gets its OWN wavefront workers (leafWavefront) so it never
	// collides with the base's e.wavefront running concurrently.
	leafLanes := e.wavefrontLanes
	if e.singleThread {
		leafLanes = 0
	}
	e.leafPC.wavefront = &e.leafWavefront
	e.leafPC.wavefrontLanes = leafLanes
	if lfLevel > 0 {
		e.leafPC.st.lfMap = &e.leafLF.filtMap
	} else {
		e.leafPC.st.lfMap = nil
	}

	out := &e.pipeLeafRecon
	if out.Y == nil || len(out.Y) != len(src.Y) || len(out.U) != len(src.U) || len(out.V) != len(src.V) {
		*out = SourceFrame420{
			Y:            make([]byte, len(src.Y)),
			U:            make([]byte, len(src.U)),
			V:            make([]byte, len(src.V)),
			YStride:      src.YStride,
			ChromaStride: src.ChromaStride,
			Width:        src.Width,
			Height:       src.Height,
		}
	}
	data, err := e.leafPC.encodeTileWithOptionsColor(src, refRecon, nil, out, effQ, prevCtx, parser.ReferenceModeSingle, header.Prefix.ForceIntegerMV, header.Prefix.AllowScreenContentTools, color, 0, uint16(src.Width/4))
	if err != nil {
		return nil, 0, fmt.Errorf("encode leaf tile: %w", err)
	}
	e.leafPayloads[0] = TilePayload{Data: data}
	if lfLevel > 0 {
		if err := e.applyLeafFilters(out, lfLevel, cdefParserParams(header.CDEF)); err != nil {
			return nil, 0, err
		}
	}
	tu, err := assembleInterTU(seq, header, e.leafPayloads[:1], 1, &e.leafTuGroup, &e.leafTuScratch)
	if err != nil {
		return nil, 0, err
	}
	return tu, e.leafPC.st.cntZeroMV, nil
}

// applyLeafFilters runs the leaf's in-loop deblocking and CDEF on the isolated
// leaf appliers (serial; the leaf reconstruction is never a reference in L1T2,
// so this affects only Recon() output, not the emitted bytes).
func (e *VideoEncoder) applyLeafFilters(out *SourceFrame420, lfLevel uint8, cdef parser.CDEFParams) error {
	lf := parser.LoopFilterParams{
		LevelY: [2]uint8{lfLevel, lfLevel},
		LevelU: lfLevel,
		LevelV: lfLevel,
	}
	if err := e.leafLF.applySerialMasks(out, lf); err != nil {
		return fmt.Errorf("leaf loop filter apply: %w", err)
	}
	if err := e.leafCdefApp.applySerial(out, cdef, &e.leafLF.filtMap); err != nil {
		return fmt.Errorf("leaf cdef apply: %w", err)
	}
	return nil
}

// encodeHeld codes the buffered source through the exact serial path, using the
// temporal ID it was assigned when buffered. Because non-overlapped pipelined
// frames are coded strictly in stream order with no intervening state change,
// the encoder state chain — and therefore the emitted bytes — matches Encode().
func (e *VideoEncoder) encodeHeld() ([]byte, bool, error) {
	e.pipeHeld = false
	return e.EncodeWithTemporalID(e.pipeHeldSrc, e.pipeHeldKey, e.pipeHeldTID)
}

// holdSource buffers src into an encoder-owned frame (double-buffered so the
// overlap can keep the leaf source live alongside the following base source)
// and records the temporal ID it will code at.
func (e *VideoEncoder) holdSource(src SourceFrame420, forceKey bool) {
	buf := &e.pipeSrcBuf[e.pipeSrcIdx]
	e.pipeSrcIdx ^= 1
	copyFrameInto(buf, src)
	e.pipeHeldSrc = *buf
	e.pipeHeldKey = forceKey
	e.pipeHeldTID = e.TemporalID()
	e.pipeHeld = true
}

// prewarmPipeline sizes the double-buffered source-hold planes so the steady
// EncodeThroughput path allocates nothing. It is a no-op unless pipelining is
// enabled.
func (e *VideoEncoder) prewarmPipeline(src SourceFrame420) {
	if !e.pipeline {
		return
	}
	copyFrameInto(&e.pipeSrcBuf[0], src)
	copyFrameInto(&e.pipeSrcBuf[1], src)
	e.pipeHeld = false
	e.pipeHeldDone = false
	e.pipeSrcIdx = 0
}
