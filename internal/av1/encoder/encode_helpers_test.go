package encoder_test

import (
	"bytes"
	"testing"

	"github.com/thesyncim/goav1/internal/av1/encoder"
)

// encodeKeyframe codes src as a forced keyframe on a fresh KeyframeEncoder.
// The reconstruction stays valid for the rest of the test.
func encodeKeyframe(t testing.TB, src encoder.SourceFrame420, qIndex uint8) ([]byte, encoder.SourceFrame420, error) {
	t.Helper()
	e, err := encoder.NewKeyframeEncoder(src.Width, src.Height, qIndex)
	if err != nil {
		return nil, encoder.SourceFrame420{}, err
	}
	t.Cleanup(func() { _ = e.Close() })
	return e.Encode(src)
}

// encodeSequence codes srcs on one fixed-qindex VideoEncoder, the first as a
// forced keyframe, and returns each temporal unit, a copy of each
// reconstruction and whether each frame was coded as a keyframe.
func encodeSequence(t testing.TB, qIndex uint8, srcs ...encoder.SourceFrame420) ([][]byte, []encoder.SourceFrame420, []bool) {
	t.Helper()
	e, err := encoder.NewVideoEncoder(srcs[0].Width, srcs[0].Height, qIndex)
	if err != nil {
		t.Fatalf("NewVideoEncoder: %v", err)
	}
	defer e.Close()
	tus := make([][]byte, len(srcs))
	recons := make([]encoder.SourceFrame420, len(srcs))
	keys := make([]bool, len(srcs))
	for i, src := range srcs {
		tu, key, err := e.Encode(src, i == 0)
		if err != nil {
			t.Fatalf("encode frame %d: %v", i, err)
		}
		tus[i], recons[i], keys[i] = bytes.Clone(tu), cloneFrame(e.Recon()), key
	}
	return tus, recons, keys
}
