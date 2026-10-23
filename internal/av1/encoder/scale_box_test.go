package encoder

import (
	"bytes"
	"testing"
)

func boxTestPlane(w, h, stride int) []byte {
	p := make([]byte, stride*h)
	for y := 0; y < h; y++ {
		for x := 0; x < stride; x++ {
			p[y*stride+x] = byte((x*17 + y*31 + x*y*3 + 11) & 0xff)
		}
	}
	return p
}

// The layer steps of every scalability mode: exact 1/2 (luma and even
// chroma), odd-sized 1/2 (luma floor -> ScalePlaneBox, chroma ceil ->
// bilinear) and 2/3 (bilinear).
var boxTestSteps = []struct{ sw, sh, dw, dh int }{
	{64, 32, 32, 16},
	{38, 26, 19, 13},
	{135, 135, 67, 67},
	{135, 135, 68, 68},
	{270, 135, 135, 68},
	{1008, 576, 672, 384},
	{853, 480, 568, 320},
	{1, 9, 1, 4},
}

func TestScalePlaneBoxExactHalfIsPyramidDownsample(t *testing.T) {
	const sw, sh, pad = 70, 46, 9
	src := boxTestPlane(sw, sh, sw+pad)
	got := make([]byte, (sw/2)*(sh/2))
	want := make([]byte, len(got))
	scalePlaneBox8(got, sw/2, sw/2, sh/2, src, sw+pad, sw, sh, &boxScaleScratch{})
	svtDownsample2D(want, sw/2, src, sw+pad, sw/2, sh/2)
	if !bytes.Equal(got, want) {
		t.Fatal("exact 1/2 box step differs from svt_aom_downsample_2d")
	}
}

func TestScalePlaneBoxKeepsFlatPlanesWithinTruncation(t *testing.T) {
	var scratch boxScaleScratch
	for _, st := range boxTestSteps {
		src := make([]byte, st.sw*st.sh)
		src16 := make([]uint16, st.sw*st.sh)
		for i := range src {
			src[i] = 77
			src16[i] = 777
		}
		dst := make([]byte, st.dw*st.dh)
		dst16 := make([]uint16, st.dw*st.dh)
		scalePlaneBox8(dst, st.dw, st.dw, st.dh, src, st.sw, st.sw, st.sh, &scratch)
		scalePlaneBox16(dst16, st.dw, st.dw, st.dh, src16, st.sw, st.sw, st.sh, &scratch)
		for i := range dst {
			// ScaleAddCols truncates: a box of 3 rows loses up to one step.
			if dst[i] < 76 || dst[i] > 77 || dst16[i] < 776 || dst16[i] > 777 {
				t.Fatalf("%+v: sample %d = %d / %d", st, i, dst[i], dst16[i])
			}
		}
	}
}

// A box/bilinear downscale of a horizontal ramp stays monotone and inside
// the source range (catches mis-stepped or mis-weighted taps).
func TestScalePlaneBoxRampMonotone(t *testing.T) {
	var scratch boxScaleScratch
	for _, st := range boxTestSteps {
		src := make([]byte, st.sw*st.sh)
		for y := 0; y < st.sh; y++ {
			for x := 0; x < st.sw; x++ {
				src[y*st.sw+x] = byte(x * 255 / max(st.sw-1, 1))
			}
		}
		dst := make([]byte, st.dw*st.dh)
		scalePlaneBox8(dst, st.dw, st.dw, st.dh, src, st.sw, st.sw, st.sh, &scratch)
		for y := 0; y < st.dh; y++ {
			for x := 1; x < st.dw; x++ {
				if dst[y*st.dw+x] < dst[y*st.dw+x-1] {
					t.Fatalf("%+v: row %d not monotone at %d", st, y, x)
				}
			}
		}
	}
}

// TestWebRTCPyramidHintMatchesFilteredPyramid checks that the S3 stream's
// lower-layer luma is the top layer's HME pyramid, that buildPyramid takes
// it, and that a hint for another picture is ignored.
func TestWebRTCPyramidHintMatchesFilteredPyramid(t *testing.T) {
	cfg := Config{
		Resolution:        Resolution{Width: 1008, Height: 576},
		MaxFramerate:      Rational{Num: 30, Den: 1},
		MinBitrateKbps:    100,
		MaxBitrateKbps:    900,
		TargetBitrateKbps: 500,
		Scalability:       ScalabilityModeS3T3,
	}
	s, err := NewWebRTCStreamConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	const w, h = 1008, 576
	src := SourceFrame420{Y: boxTestPlane(w, h, w), U: boxTestPlane(w/2, h/2, w/2), V: boxTestPlane(w/2, h/2, w/2), YStride: w, ChromaStride: w / 2, Width: w, Height: h}
	s.pictureGen++
	if _, err := s.sourceForLayer(src, 0, s.config.SpatialLayers[0].Resolution); err != nil {
		t.Fatal(err)
	}
	hint := s.webRTCPyramidHint(src, 2)
	if hint.half.Width != w/2 || hint.six.Width != w/4 {
		t.Fatalf("hint half %d six %d", hint.half.Width, hint.six.Width)
	}
	var want, got svtPyramid
	want.build(src)
	got.setHint(hint)
	got.build(src)
	if !bytes.Equal(got.curHalf.pix, want.curHalf.pix) || !bytes.Equal(got.curSix.pix, want.curSix.pix) {
		t.Fatal("hinted pyramid differs from the filtered one")
	}
	// The hint is taken: a planted sample shows up in the half plane.
	hint.half.Y[0] ^= 0xff
	got.setHint(hint)
	got.build(src)
	if got.curHalf.pix[got.curHalf.org] == want.curHalf.pix[want.curHalf.org] {
		t.Fatal("buildPyramid ignored a matching hint")
	}
	// A hint made from another picture is not.
	other := src
	other.Y = append([]byte(nil), src.Y...)
	got.setHint(hint)
	got.build(other)
	if !bytes.Equal(got.curHalf.pix, want.curHalf.pix) {
		t.Fatal("buildPyramid took another picture's hint")
	}
	hint.half.Y[0] ^= 0xff
	// The next picture invalidates the cached layers.
	s.pictureGen++
	if hint := s.webRTCPyramidHint(src, 2); hint.parent != nil {
		t.Fatal("stale layers produced a hint")
	}
}
