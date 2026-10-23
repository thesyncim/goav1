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

func seedTestFrame(w, h, n int) SourceFrame420 {
	f := SourceFrame420{Y: make([]byte, w*h), U: make([]byte, w*h/4), V: make([]byte, w*h/4), YStride: w, ChromaStride: w / 2, Width: w, Height: h}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			f.Y[y*w+x] = byte(((x+3*n)*7 + (y+n)*13 + ((x+3*n)>>3)*((y+n)>>3)*5) & 0xff)
		}
	}
	for i := range f.U {
		f.U[i], f.V[i] = 128, 128
	}
	return f
}

// TestWebRTCMESeedFiresOnSimulcastInterFrames checks the cross-layer ME
// seed: it centers the upper layer's HME on simulcast inter pictures, never
// on the keyframe (the lower layer ran no ME), never for a lower layer not
// coded in the picture, and not in an SVC mode.
func TestWebRTCMESeedFiresOnSimulcastInterFrames(t *testing.T) {
	for _, tc := range []struct {
		mode ScalabilityMode
		want bool
	}{
		{ScalabilityModeS2T3, true},
		{ScalabilityModeL2T3_KEY, false},
	} {
		cfg := Config{
			Resolution:        Resolution{Width: 640, Height: 360},
			MaxFramerate:      Rational{Num: 30, Den: 1},
			MinBitrateKbps:    100,
			MaxBitrateKbps:    900,
			TargetBitrateKbps: 500,
			Scalability:       tc.mode,
		}
		s, err := NewWebRTCStreamConfig(cfg)
		if err != nil {
			t.Fatal(err)
		}
		for n := 0; n < 4; n++ {
			if _, err := s.EncodePicture(seedTestFrame(640, 360, n), false); err != nil {
				t.Fatalf("%s picture %d: %v", tc.mode, n, err)
			}
			got := s.encoders[1].hme.seedOn
			if want := tc.want && n > 0; got != want {
				t.Fatalf("%s picture %d: seedOn = %v, want %v", tc.mode, n, got, want)
			}
			if s.encoders[1].hme.seed != nil {
				t.Fatalf("%s picture %d: seed left set after the encode", tc.mode, n)
			}
		}
		s.pictureGen++
		if s.webRTCMESeed(1) != nil {
			t.Fatalf("%s: a lower layer not coded in the picture seeded", tc.mode)
		}
		s.Close()
	}
}

func TestBoxInterpolateRow8MatchesScalar(t *testing.T) {
	const n = 71
	s0, s1 := boxTestPlane(n, 1, n), boxTestPlane(n, 2, n)[n:]
	for _, f := range []int{1, 64, 127, 128, 129, 192, 255} {
		got, want := make([]byte, n), make([]byte, n)
		boxInterpolateRow8(got, s0, s1, f)
		boxInterpolateRow(want, s0, s1, f, 128)
		if !bytes.Equal(got, want) {
			t.Fatalf("f=%d: SIMD row differs", f)
		}
	}
}

// The 2/3 column fast path computes ScaleFilterCols' BLENDER values.
func TestBoxFilterColsTwoThirds(t *testing.T) {
	for _, dw := range []int{8, 9, 672} {
		sw := dw * 3 / 2
		s := boxTestPlane(sw, 1, sw)
		got := make([]byte, dw)
		boxFilterCols(got, s, 0x4000, 0x18000)
		for i, x := 0, 0x4000; i < dw; i, x = i+1, x+0x18000 {
			a, b := int(s[x>>16]), int(s[min(x>>16+1, sw-1)])
			if want := byte(a + (((x&0xffff)*(b-a) + 0x8000) >> 16)); got[i] != want {
				t.Fatalf("dw=%d i=%d: %d want %d", dw, i, got[i], want)
			}
		}
	}
}
