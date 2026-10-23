package encoder

import (
	"os"
	"path/filepath"
	"testing"
)

// TestSVTScreenContentDetectionMatchesSVT compares svt_aom_is_screen_content
// counters and classes with SVT-AV1 4.1.0's own (an instrumented
// SvtAv1EncApp --scm 2 on frame 0 of each 1280x720 real-corpus encoder input).
// Set GOAV1_REALCORPUS_YUV to the directory of <clip>_720p.yuv files.
func TestSVTScreenContentDetectionMatchesSVT(t *testing.T) {
	dir := os.Getenv("GOAV1_REALCORPUS_YUV")
	if dir == "" {
		t.Skip("GOAV1_REALCORPUS_YUV not set")
	}
	const w, h = 1280, 720
	cases := []struct {
		clip string
		cnt  svtSCCounts
		cls  svtSCClass
	}{
		{"wikipedia", svtSCCounts{402, 304, 4424, 1274}, svtSCClass{true, true, true, true, true}},
		{"debugging", svtSCCounts{320, 220, 1938, 1706}, svtSCClass{class4: true}},
		{"fourpeople", svtSCCounts{2, 1, 393, 38}, svtSCClass{}},
		{"johnny", svtSCCounts{17, 17, 607, 77}, svtSCClass{}},
		{"mtbike", svtSCCounts{20, 12, 540, 60}, svtSCClass{}},
		{"parkjoy", svtSCCounts{0, 0, 0, 0}, svtSCClass{}},
	}
	var d svtSCDetector
	buf := make([]byte, w*h)
	for _, c := range cases {
		f, err := os.Open(filepath.Join(dir, c.clip+"_720p.yuv"))
		if err != nil {
			t.Skipf("%s: %v", c.clip, err)
		}
		_, err = f.Read(buf)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		cls, cnt := d.detect(buf, w, w, h)
		if cnt != c.cnt || cls != c.cls {
			t.Errorf("%s: counts %+v classes %+v, SVT %+v %+v", c.clip, cnt, cls, c.cnt, c.cls)
		}
	}
}

// TestSVTScreenContentDetectionSynthetic pins the classes on two synthetic
// pictures: flat two-color text-like blocks are screen content (sc_class1),
// smooth noise is not.
func TestSVTScreenContentDetectionSynthetic(t *testing.T) {
	const w, h = 320, 192
	text := make([]byte, w*h)
	noise := make([]byte, w*h)
	seed := uint32(7)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			text[y*w+x] = 255
			if (x/3+y/5)%4 == 0 {
				text[y*w+x] = 20
			}
			seed = seed*1664525 + 1013904223
			noise[y*w+x] = uint8(100 + x/4 + int(seed>>28))
		}
	}
	var d svtSCDetector
	if cls := d.classes(ContentAuto, text, w, w, h); !cls.class1 {
		t.Fatalf("text picture classes %+v, want sc_class1", cls)
	}
	if cls := d.classes(ContentAuto, noise, w, w, h); cls != (svtSCClass{}) {
		t.Fatalf("noise picture classes %+v, want none", cls)
	}
	if cls := d.classes(ContentCamera, text, w, w, h); cls != (svtSCClass{}) {
		t.Fatalf("camera classes %+v, want none", cls)
	}
	if cls := d.classes(ContentScreen, noise, w, w, h); !cls.class1 || !cls.class4 {
		t.Fatalf("screen classes %+v, want all", cls)
	}
	if n := testing.AllocsPerRun(5, func() { d.detect(text, w, w, h) }); n != 0 {
		t.Fatalf("detect allocates %.0f", n)
	}
}
