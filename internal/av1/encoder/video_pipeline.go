package encoder

import "fmt"

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
// latency: to start base N+1 before leaf N finishes, the encoder must accept
// base N+1's source while leaf N is still coding, so results emerge one
// EncodeThroughput() call later.
//
// This file establishes the opt-in API and the +1-latency ordering contract as
// a strict FIFO around the existing serial encode: every frame is still coded
// in stream order through the exact serial path, so the emitted bytes are
// identical to Encode() for the same source sequence. The concurrent overlap of
// the buffered leaf with the following base is layered on top of this FIFO in a
// follow-up increment; because it only reorders work in wall-clock time (never
// the logical encode order or the shared-state chain) the byte-identity oracle
// (TestVideoEncoderPipelineByteIdentical) guards both increments.

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
	if e.pipeHeld {
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
	if e.pipeHeld {
		tu, key, err = e.encodeHeld()
		if err != nil {
			return nil, false, false, err
		}
		produced = true
	}
	e.holdSource(src, forceKey)
	return tu, key, produced, nil
}

// Drain flushes the final buffered frame from the throughput pipeline. It
// returns produced=false when the pipeline is empty. After a successful Drain
// the pipeline is empty and can accept a fresh EncodeThroughput sequence.
func (e *VideoEncoder) Drain() (tu []byte, key bool, produced bool, err error) {
	if e == nil || !e.pipeHeld {
		return nil, false, false, nil
	}
	tu, key, err = e.encodeHeld()
	if err != nil {
		return nil, false, false, err
	}
	return tu, key, true, nil
}

// encodeHeld codes the buffered source through the exact serial path, using the
// temporal ID it was assigned when buffered. Because pipelined frames are
// buffered and coded strictly in stream order with no intervening state change,
// the encoder state chain — and therefore the emitted bytes — matches Encode().
func (e *VideoEncoder) encodeHeld() ([]byte, bool, error) {
	e.pipeHeld = false
	return e.EncodeWithTemporalID(e.pipeHeldSrc, e.pipeHeldKey, e.pipeHeldTID)
}

// holdSource buffers src into an encoder-owned frame (double-buffered so a
// later overlap increment can keep the leaf source live alongside the following
// base source) and records the temporal ID it will code at.
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
	e.pipeSrcIdx = 0
}
