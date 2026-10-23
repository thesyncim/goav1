package encoder

import (
	"testing"

	"github.com/thesyncim/goav1/internal/av1/lfmask"
	"github.com/thesyncim/goav1/internal/av1/threading"
)

// Compare the actual encoder's coding-order producer to the unchanged generic
// map reconstruction, including frame reuse, a partial SB row, and a keyframe
// between two inter chains. Comparing raw masks catches widths even on edges
// where the pixel filter happens to leave the reconstruction unchanged.
func TestVideoEncoderOrderedMasksMatchMap(t *testing.T) {
	const width, height = 192, 136
	enc, err := NewVideoEncoder(width, height, 160)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	enc.SetMaxThreads(1)
	var scratch threading.FrameWorkLoopFilterMaskBuildScratch
	var expected []lfmask.FilterMask
	interFrames := 0
	for n := 0; n < 7; n++ {
		src := lfDiffFrame(width, height)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				src.Y[y*width+x] = byte(40 + ((x+n*3)/5+y/3)%150)
			}
		}
		_, key, err := enc.Encode(src, n == 3)
		if err != nil {
			t.Fatalf("frame %d: %v", n, err)
		}
		if key {
			if enc.lf.orderedMasks {
				t.Fatalf("frame %d: keyframe retained inter mask readiness", n)
			}
			continue
		}
		interFrames++
		if !enc.lf.orderedMasks || !enc.lf.orderedBuilder.Complete(&enc.lf.masks) {
			t.Fatalf("frame %d: ordered mask producer was not reached", n)
		}
		if len(expected) != len(enc.lf.masks.Masks) {
			expected = make([]lfmask.FilterMask, len(enc.lf.masks.Masks))
		}
		ref := enc.lf.masks
		ref.Masks = expected
		if err := ref.BuildFromMap(&scratch, enc.lf.filtMap, encoderLoopFilterSBLog2); err != nil {
			t.Fatalf("frame %d: generic map build: %v", n, err)
		}
		for i, want := range expected {
			if enc.lf.masks.Masks[i] != want {
				t.Fatalf("frame %d: mask region %d differs from generic map", n, i)
			}
		}
	}
	if interFrames != 5 {
		t.Fatalf("inter frames=%d, want 5", interFrames)
	}
}

func TestVideoEncoderOrderedMasksAllocs(t *testing.T) {
	const width, height = 192, 136
	enc, err := NewVideoEncoder(width, height, 160)
	if err != nil {
		t.Fatal(err)
	}
	defer enc.Close()
	enc.SetMaxThreads(1)
	if err := enc.Prewarm(); err != nil {
		t.Fatal(err)
	}
	src := lfDiffFrame(width, height)
	if _, _, err := enc.Encode(src, true); err != nil {
		t.Fatal(err)
	}
	if allocs := testing.AllocsPerRun(3, func() {
		if _, _, err := enc.Encode(src, false); err != nil {
			t.Fatal(err)
		}
	}); allocs != 0 {
		t.Fatalf("ordered-mask encoder allocated %g times", allocs)
	}
}
