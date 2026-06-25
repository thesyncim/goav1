package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"runtime"
	"sync"
	"testing"
	"time"

	cdpruntime "github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	"github.com/pion/webrtc/v4"
)

const (
	browserE2EEnv        = "GOAV1_BROWSER_E2E"
	requireBrowserE2EEnv = "GOAV1_REQUIRE_WEBRTC_BROWSER"
	browserExecutableEnv = "GOAV1_BROWSER_EXECUTABLE"
)

func TestBrowserLiveAV1PlaybackStats(t *testing.T) {
	required := os.Getenv(requireBrowserE2EEnv) == "1"
	if !required && os.Getenv(browserE2EEnv) != "1" {
		t.Skipf("set %s=1 to run the live browser/libwebrtc AV1 playback gate", browserE2EEnv)
	}
	browserPath, err := browserExecutable()
	if err != nil {
		if required {
			t.Fatalf("required browser executable unavailable: %v", err)
		}
		t.Skip(err)
	}

	var mu sync.Mutex
	var peers []*webrtc.PeerConnection
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.HandleFunc("/offer", func(w http.ResponseWriter, r *http.Request) {
		err := handleOfferWithPeerConnectionHook(w, r, 4_000_000, func(pc *webrtc.PeerConnection) {
			mu.Lock()
			peers = append(peers, pc)
			mu.Unlock()
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	t.Cleanup(func() {
		mu.Lock()
		defer mu.Unlock()
		for _, pc := range peers {
			_ = pc.Close()
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Second)
	defer cancel()
	allocCtx, cancelAlloc := chromedp.NewExecAllocator(ctx,
		append(chromedp.DefaultExecAllocatorOptions[:],
			chromedp.ExecPath(browserPath),
			chromedp.Flag("headless", "new"),
			chromedp.Flag("autoplay-policy", "no-user-gesture-required"),
			chromedp.Flag("disable-background-timer-throttling", true),
			chromedp.Flag("disable-renderer-backgrounding", true),
			chromedp.Flag("mute-audio", true),
			chromedp.NoSandbox,
			chromedp.UserDataDir(t.TempDir()),
		)...,
	)
	defer cancelAlloc()
	browserCtx, cancelBrowser := chromedp.NewContext(allocCtx)
	defer cancelBrowser()

	var got browserPlaybackEvidence
	if err := chromedp.Run(browserCtx,
		chromedp.Navigate(server.URL),
		chromedp.Evaluate(browserPlaybackProbeJS(30), &got, evalAwaitPromise),
	); err != nil {
		t.Fatalf("browser AV1 playback probe: %v", err)
	}
	if !got.OK {
		t.Fatalf("browser AV1 playback probe failed: %s; last=%+v", got.Error, got)
	}
	if got.VideoWidth != width || got.VideoHeight != height {
		t.Fatalf("browser decoded size=%dx%d want %dx%d", got.VideoWidth, got.VideoHeight, width, height)
	}
	if got.FramesDecoded < 30 || got.PacketsReceived == 0 || got.BytesReceived == 0 {
		t.Fatalf("browser stats frames=%d packets=%d bytes=%d", got.FramesDecoded, got.PacketsReceived, got.BytesReceived)
	}
	if got.CodecMimeType != "" && got.CodecMimeType != "video/AV1" {
		t.Fatalf("browser codec mime=%q want video/AV1", got.CodecMimeType)
	}
	t.Logf("browser AV1 playback: frames=%d keyframes=%d packets=%d bytes=%d decoder=%q pli=%d fir=%d",
		got.FramesDecoded, got.KeyFramesDecoded, got.PacketsReceived, got.BytesReceived,
		got.DecoderImplementation, got.PLICount, got.FIRCount)
}

type browserPlaybackEvidence struct {
	OK                    bool   `json:"ok"`
	Error                 string `json:"error"`
	ConnectionState       string `json:"connectionState"`
	ICEConnectionState    string `json:"iceConnectionState"`
	PageError             string `json:"pageError"`
	VideoReadyState       int    `json:"videoReadyState"`
	VideoCurrentTimeMS    int    `json:"videoCurrentTimeMS"`
	VideoWidth            int    `json:"videoWidth"`
	VideoHeight           int    `json:"videoHeight"`
	FramesDecoded         int    `json:"framesDecoded"`
	KeyFramesDecoded      int    `json:"keyFramesDecoded"`
	FramesReceived        int    `json:"framesReceived"`
	PacketsReceived       int    `json:"packetsReceived"`
	BytesReceived         int    `json:"bytesReceived"`
	PLICount              int    `json:"pliCount"`
	FIRCount              int    `json:"firCount"`
	FreezeCount           int    `json:"freezeCount"`
	JitterMS              int    `json:"jitterMS"`
	CodecMimeType         string `json:"codecMimeType"`
	DecoderImplementation string `json:"decoderImplementation"`
}

func browserPlaybackProbeJS(minFrames int) string {
	return fmt.Sprintf(`(async () => {
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let last = {};
  const snapshot = async () => {
    const pc = window.goav1PC;
    const video = window.goav1Video || document.getElementById('video');
    const out = {
      ok: false,
      error: '',
      connectionState: pc ? pc.connectionState : 'missing',
      iceConnectionState: pc ? pc.iceConnectionState : 'missing',
      pageError: window.goav1Error || '',
      videoReadyState: video ? video.readyState : 0,
      videoCurrentTimeMS: video ? Math.round(video.currentTime * 1000) : 0,
      videoWidth: video ? video.videoWidth : 0,
      videoHeight: video ? video.videoHeight : 0,
      framesDecoded: video && video.webkitDecodedFrameCount ? Number(video.webkitDecodedFrameCount) : 0,
      keyFramesDecoded: 0,
      framesReceived: 0,
      packetsReceived: 0,
      bytesReceived: 0,
      pliCount: 0,
      firCount: 0,
      freezeCount: 0,
      jitterMS: 0,
      codecMimeType: '',
      decoderImplementation: '',
    };
    if (!pc) return out;
    const stats = await pc.getStats();
    const codecs = new Map();
    stats.forEach((report) => {
      if (report.type === 'codec') codecs.set(report.id, report);
    });
    stats.forEach((report) => {
      if (report.type !== 'inbound-rtp' || (report.kind !== 'video' && report.mediaType !== 'video')) return;
      out.framesDecoded = Math.max(out.framesDecoded, Number(report.framesDecoded || 0));
      out.keyFramesDecoded = Number(report.keyFramesDecoded || 0);
      out.framesReceived = Number(report.framesReceived || 0);
      out.packetsReceived = Number(report.packetsReceived || 0);
      out.bytesReceived = Number(report.bytesReceived || 0);
      out.pliCount = Number(report.pliCount || 0);
      out.firCount = Number(report.firCount || 0);
      out.freezeCount = Number(report.freezeCount || 0);
      out.jitterMS = Math.round(Number(report.jitter || 0) * 1000);
      out.decoderImplementation = String(report.decoderImplementation || '');
      const codec = codecs.get(report.codecId);
      if (codec) out.codecMimeType = String(codec.mimeType || '');
    });
    return out;
  };
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    last = await snapshot();
    if (last.pageError) return Object.assign(last, { error: last.pageError });
    if (
      last.connectionState === 'connected' &&
      last.videoReadyState >= 2 &&
      last.videoCurrentTimeMS >= 250 &&
      last.videoWidth > 0 &&
      last.videoHeight > 0 &&
      last.framesDecoded >= %d &&
      last.packetsReceived > 0 &&
      last.bytesReceived > 0
    ) {
      return Object.assign(last, { ok: true });
    }
    await sleep(250);
  }
  return Object.assign(last, { error: 'timed out waiting for live AV1 frames decoded by browser' });
})()`, minFrames)
}

func evalAwaitPromise(p *cdpruntime.EvaluateParams) *cdpruntime.EvaluateParams {
	return p.WithAwaitPromise(true)
}

func browserExecutable() (string, error) {
	if p := os.Getenv(browserExecutableEnv); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("%s=%q: %w", browserExecutableEnv, p, err)
		}
		return p, nil
	}
	candidates := []string{"google-chrome", "chromium", "chromium-browser", "chrome"}
	switch runtime.GOOS {
	case "darwin":
		candidates = append([]string{
			"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
			"/Applications/Chromium.app/Contents/MacOS/Chromium",
			"/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
		}, candidates...)
	case "linux":
		candidates = append([]string{
			"/usr/bin/google-chrome",
			"/usr/bin/google-chrome-stable",
			"/usr/bin/chromium",
			"/usr/bin/chromium-browser",
		}, candidates...)
	case "windows":
		candidates = append([]string{
			`C:\Program Files\Google\Chrome\Application\chrome.exe`,
			`C:\Program Files (x86)\Google\Chrome\Application\chrome.exe`,
			`C:\Program Files\Microsoft\Edge\Application\msedge.exe`,
		}, candidates...)
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
		if path, err := exec.LookPath(candidate); err == nil {
			return path, nil
		}
	}
	return "", errors.New("Chrome/Chromium executable not found; set GOAV1_BROWSER_EXECUTABLE")
}
