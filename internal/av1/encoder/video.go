package encoder

import "fmt"

// video.go is the streaming encoder surface: a VideoEncoder owns the reference
// reconstruction state and turns a sequence of source frames into a decodable
// temporal-unit stream — a keyframe first (or on demand), then
// motion-compensated P-frames chained off the previous frame's reconstruction.

// VideoEncoder encodes a stream of same-sized frames, either at a fixed base
// qindex or under closed-loop CBR rate control.
type VideoEncoder struct {
	width, height int
	qIndex        uint8
	recon         SourceFrame420
	haveKey       bool

	rcEnabled      bool
	rcPerFrameBits int
	rcBuffer       int
	rcMinQ, rcMaxQ uint8
}

// RateControlConfig describes closed-loop CBR rate control: a target bitrate
// at a fixed frame rate, with the working qindex clamped to [MinQIndex,
// MaxQIndex].
type RateControlConfig struct {
	TargetBitsPerSecond int
	FramesPerSecond     int
	MinQIndex           uint8
	MaxQIndex           uint8
}

// NewVideoEncoder creates a streaming encoder. Dimensions must be positive
// multiples of 64 (the current P-frame constraint) and qIndex non-zero.
func NewVideoEncoder(width, height int, qIndex uint8) (*VideoEncoder, error) {
	if width <= 0 || height <= 0 || width%64 != 0 || height%64 != 0 {
		return nil, fmt.Errorf("encoder: dimensions must be positive multiples of 64, got %dx%d", width, height)
	}
	if qIndex == 0 {
		return nil, fmt.Errorf("encoder: qindex must be non-zero")
	}
	return &VideoEncoder{width: width, height: height, qIndex: qIndex}, nil
}

// NewVideoEncoderCBR creates a streaming encoder under CBR rate control. The
// encoder starts in the middle of the configured qindex range and adjusts the
// working qindex after every frame from the bit-budget buffer.
func NewVideoEncoderCBR(width, height int, rc RateControlConfig) (*VideoEncoder, error) {
	if rc.TargetBitsPerSecond <= 0 || rc.FramesPerSecond <= 0 {
		return nil, fmt.Errorf("encoder: invalid rate control target %d bps @ %d fps", rc.TargetBitsPerSecond, rc.FramesPerSecond)
	}
	if rc.MinQIndex == 0 || rc.MaxQIndex <= rc.MinQIndex {
		return nil, fmt.Errorf("encoder: invalid qindex range [%d, %d]", rc.MinQIndex, rc.MaxQIndex)
	}
	e, err := NewVideoEncoder(width, height, rc.MinQIndex/2+rc.MaxQIndex/2)
	if err != nil {
		return nil, err
	}
	e.rcEnabled = true
	e.rcPerFrameBits = rc.TargetBitsPerSecond / rc.FramesPerSecond
	e.rcMinQ = rc.MinQIndex
	e.rcMaxQ = rc.MaxQIndex
	return e, nil
}

// rcUpdate feeds one frame's actual size into the leaky-bucket controller and
// steps the working qindex toward the per-frame budget. The step grows with
// the buffer excursion so keyframe overshoot recovers within a few frames.
func (e *VideoEncoder) rcUpdate(frameBits int) {
	if !e.rcEnabled {
		return
	}
	e.rcBuffer += e.rcPerFrameBits - frameBits
	// Clamp the buffer so one huge keyframe cannot dominate forever.
	if limit := 8 * e.rcPerFrameBits; e.rcBuffer < -limit {
		e.rcBuffer = -limit
	} else if e.rcBuffer > limit {
		e.rcBuffer = limit
	}
	q := int(e.qIndex)
	switch {
	case e.rcBuffer < -4*e.rcPerFrameBits:
		q += 12
	case e.rcBuffer < -e.rcPerFrameBits:
		q += 4
	case e.rcBuffer > 4*e.rcPerFrameBits:
		q -= 12
	case e.rcBuffer > e.rcPerFrameBits:
		q -= 4
	}
	if q < int(e.rcMinQ) {
		q = int(e.rcMinQ)
	}
	if q > int(e.rcMaxQ) {
		q = int(e.rcMaxQ)
	}
	e.qIndex = uint8(q)
}

// QIndex reports the working qindex the next frame will use.
func (e *VideoEncoder) QIndex() uint8 {
	return e.qIndex
}

// Encode encodes one frame and returns its temporal unit plus whether it was
// coded as a keyframe. The first frame (and any frame with forceKey set) is a
// keyframe; every other frame predicts from the previous reconstruction.
func (e *VideoEncoder) Encode(src SourceFrame420, forceKey bool) ([]byte, bool, error) {
	if src.Width != e.width || src.Height != e.height {
		return nil, false, fmt.Errorf("encoder: frame %dx%d does not match stream %dx%d", src.Width, src.Height, e.width, e.height)
	}
	if !e.haveKey || forceKey {
		tu, recon, err := EncodeKeyframe(src, e.qIndex)
		if err != nil {
			return nil, false, err
		}
		e.recon = recon
		e.haveKey = true
		e.rcUpdate(len(tu) * 8)
		return tu, true, nil
	}
	tu, recon, err := EncodePFrame(src, e.recon, e.qIndex)
	if err != nil {
		return nil, false, err
	}
	e.recon = recon
	e.rcUpdate(len(tu) * 8)
	return tu, false, nil
}

// Recon returns the most recent frame's reconstruction (what a conformant
// decoder outputs for it).
func (e *VideoEncoder) Recon() SourceFrame420 {
	return e.recon
}
