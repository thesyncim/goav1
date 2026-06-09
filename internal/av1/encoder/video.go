package encoder

import "fmt"

// video.go is the streaming encoder surface: a VideoEncoder owns the reference
// reconstruction state and turns a sequence of source frames into a decodable
// temporal-unit stream — a keyframe first (or on demand), then
// motion-compensated P-frames chained off the previous frame's reconstruction.

// VideoEncoder encodes a stream of same-sized frames at a fixed base qindex.
type VideoEncoder struct {
	width, height int
	qIndex        uint8
	recon         SourceFrame420
	haveKey       bool
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
		return tu, true, nil
	}
	tu, recon, err := EncodePFrame(src, e.recon, e.qIndex)
	if err != nil {
		return nil, false, err
	}
	e.recon = recon
	return tu, false, nil
}

// Recon returns the most recent frame's reconstruction (what a conformant
// decoder outputs for it).
func (e *VideoEncoder) Recon() SourceFrame420 {
	return e.recon
}
